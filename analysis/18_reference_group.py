"""Post hoc R3: event-free reference group (REF), concept.md §6.9. Builds REF in MIMIC and HEEDB, scores it with
both models, and refits the primary model with REF as a fourth group.

    python3 -I analysis/18_reference_group.py

REF: no LVSD, AF or LIMBFX code and no HF or cardiomyopathy code at any time; pseudo-anchor at a random ECG of
the patient (seed 20261030); then the window and ECG rules of the event groups. 2,000 entering patients in
MIMIC and 2,000 per HEEDB site, taken in a seeded random order.
Outputs: work/r3_ref_{mimic,heedb}.parquet (patient-level, local only), results/robustness_r3.json (aggregate).
"""
import hashlib
import importlib.util
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
HEEDB = Path.home() / "datasets/heedb"
SEED, N_REF, N_CAND = 20261030, 2000, 3000
GROUPS4 = ["LVSD", "AF", "CTRL", "REF"]


def stable_rank(key):
    """Seeded, process-independent order (Python's hash() is randomized for strings)."""
    return int(hashlib.md5(f"{key}|{SEED}".encode()).hexdigest()[:15], 16)


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


C1 = _mod("cohort", PROJ / "analysis/01_cohort.py")
SCORE = _mod("score", PROJ / "analysis/02_score.py")
AN = _mod("analysis3", PROJ / "analysis/03_analysis.py")
COH = _mod("heedb_cohort", PROJ / "analysis/11_heedb_cohort.py")
HS = _mod("heedb_score", PROJ / "analysis/12_heedb_score.py")
H = _mod("heedb_analysis", PROJ / "analysis/13_heedb_analysis.py")
D15 = _mod("deepecg_score", PROJ / "analysis/15_deepecg_score.py")
D16 = _mod("deepecg_analysis", PROJ / "analysis/16_deepecg_analysis.py")


def entry(win, key):
    """win: window ECG rows with 'eligible' and 'ecg_time'; returns keys of patients with >=2 eligible spanning >=180 d."""
    el = win[win.eligible]
    per = el.groupby(key).ecg_time.agg(["size", "min", "max"])
    return per[(per["size"] >= 2) & ((per["max"] - per["min"]).dt.days >= 180)].index


def qc_cached(paths, cache_f, fn, key_cols):
    cache = pd.read_parquet(cache_f) if cache_f.exists() else pd.DataFrame(columns=key_cols + ["waveform_ok"])
    todo = paths.merge(cache[key_cols], on=key_cols, how="left", indicator=True).query("_merge == 'left_only'")[key_cols]
    if len(todo):
        args = todo.iloc[:, 0].tolist() if len(key_cols) == 1 else list(zip(*[todo[c] for c in key_cols]))
        with Pool(24) as pool:
            ok = pool.map(fn, args, chunksize=128)
        new = todo.copy()
        new["waveform_ok"] = [o for _, o in ok]
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_parquet(cache_f, index=False)
    return paths.merge(cache, on=key_cols, how="left")


# --- MIMIC --------------------------------------------------------------------------------
def ref_mimic():
    con = duckdb.connect()
    C1.load(con, "LIMBFX")
    con.execute(f"""
    CREATE TABLE ev AS SELECT DISTINCT subject_id FROM adm_sets WHERE lvsd OR af OR ctrl OR hfcm;
    CREATE TABLE anc AS
    SELECT subject_id, t AS anchor FROM (
      SELECT e.*, row_number() OVER (PARTITION BY e.subject_id ORDER BY hash(CAST(e.study_id AS VARCHAR) || '{SEED}')) rn
      FROM ecg e WHERE e.subject_id NOT IN (SELECT subject_id FROM ev)) WHERE rn = 1;
    CREATE TABLE win AS
    SELECT a.subject_id, a.anchor, e.study_id, e.path, e.t AS ecg_time,
           date_diff('second', a.anchor, e.t) / (365.25 * 86400) AS t_years,
           coalesce(s.sinus, false) AND NOT coalesce(s.bad_quality, false) AS stmt_ok,
           p.anchor_age + year(a.anchor) - p.anchor_year AS age, p.gender = 'M' AS male, p.anchor_age, p.anchor_year
    FROM anc a JOIN ecg e USING (subject_id) JOIN pat p USING (subject_id) LEFT JOIN stm s USING (study_id)
    WHERE e.t >= a.anchor - INTERVAL 5 YEAR AND e.t <= a.anchor - INTERVAL 30 DAY AND p.anchor_age + year(a.anchor) - p.anchor_year >= 18
    """)
    w = con.execute("SELECT * FROM win").df()
    w["eligible"] = w.stmt_ok
    cand = pd.Series(entry(w, "subject_id"), name="subject_id").to_frame()
    cand["h"] = cand.subject_id.map(stable_rank)
    cand = cand.sort_values(["h", "subject_id"]).head(N_CAND)
    w = w[w.subject_id.isin(cand.subject_id) & w.stmt_ok]
    w = qc_cached(w[["path"]].drop_duplicates(), WORK / "wp1_waveform_qc.parquet", C1.waveform_ok, ["path"]).merge(w, on="path")
    w["eligible"] = w.waveform_ok.fillna(False).astype(bool)
    ok = entry(w, "subject_id")
    keep = cand[cand.subject_id.isin(ok)].head(N_REF).subject_id
    w = w[w.subject_id.isin(keep) & w.eligible].copy()
    return w, {"candidates_after_statement_rules": int(len(cand)), "entering_after_waveform": int(len(ok)), "kept": int(len(keep))}


# --- HEEDB --------------------------------------------------------------------------------
def ref_heedb_site(site, sexcol):
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{HEEDB}/duckdb_tmp'; SET memory_limit = '40GB'; SET threads = 24")
    COH.load_site(con, site, sexcol)
    con.execute(f"""
    CREATE TABLE ev AS SELECT DISTINCT pid FROM day_sets WHERE lvsd OR af OR ctrl OR hfcm;
    CREATE TABLE anc AS
    SELECT pid, t AS anchor FROM (
      SELECT m.*, row_number() OVER (PARTITION BY m.pid ORDER BY hash(m.rel || '{SEED}')) rn
      FROM meta m WHERE m.pid NOT IN (SELECT pid FROM ev)) WHERE rn = 1;
    CREATE TABLE win AS
    SELECT '{site}' site, a.pid, a.anchor, m.rel, m.t AS ecg_time,
           date_diff('second', a.anchor, m.t) / (365.25 * 86400) AS t_years,
           date_diff('day', m.dob, m.t) / 365.25 AS age_at_ecg_real,
           coalesce(s.sinus, false) AND NOT coalesce(s.bad_quality, false) AS stmt_ok,
           date_diff('day', m.dob, a.anchor) / 365.25 AS age, m.male
    FROM anc a JOIN meta m USING (pid) LEFT JOIN stm s USING (rel) JOIN first_code fc USING (pid)
    WHERE m.t >= a.anchor - INTERVAL 5 YEAR AND m.t <= a.anchor - INTERVAL 30 DAY
      AND m.sex_known AND m.dob IS NOT NULL AND date_diff('day', m.dob, a.anchor) / 365.25 >= 18
      AND fc.first_d <= CAST(a.anchor AS DATE) - INTERVAL 365 DAY
    """)
    w = con.execute("SELECT * FROM win").df()
    w["eligible"] = w.stmt_ok
    cand = pd.Series(entry(w, "pid"), name="pid").to_frame()
    cand["h"] = cand.pid.map(stable_rank)
    cand = cand.sort_values(["h", "pid"]).head(N_CAND)
    w = w[w.pid.isin(cand.pid) & w.stmt_ok]
    w = qc_cached(w[["site", "rel"]].drop_duplicates(), WORK / "wp_h_waveform_qc.parquet", COH.waveform_ok, ["site", "rel"]).merge(w, on=["site", "rel"])
    w["eligible"] = w.waveform_ok.fillna(False).astype(bool)
    ok = entry(w, "pid")
    keep = cand[cand.pid.isin(ok)].head(N_REF).pid
    w = w[w.pid.isin(keep) & w.eligible].copy()
    return w, {"candidates_after_statement_rules": int(len(cand)), "entering_after_waveform": int(len(ok)), "kept": int(len(keep))}


def preps_from_json():
    info = json.loads((RESULTS / "wp_d_g2.json").read_text())["preprocessing_frozen"]
    preps = {}
    for src, v in info.items():
        p = D15.Prep()
        p.scale, p.peaks = v["scale"], [tuple(x) for x in v["peak_ranges_hz"]]
        p.ranges = [[tuple(r) for r in lead] for lead in v["flatten_ranges_hz_by_lead"]] if v["flatten_ranges_hz_by_lead"] else None
        preps[src] = p
    return preps


# --- analysis -----------------------------------------------------------------------------
def terms4(df):
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    t = {f"t_{g}": df.t * (df.g == g) for g in GROUPS4}
    t["t_x_age_c"] = df.t * age_c
    t["t_x_male"] = df.t * df.male_anchor.astype(float)
    return t


def block4(df, delta):
    res, used = AN.fit(df, terms4(df))
    out = {"patients": {g: int(used[used.g == g].entity.nunique()) for g in GROUPS4}, "beta": {}, "contrasts": {}}
    for g in GROUPS4:
        e, lo, hi = AN.contrast(res, {f"t_{g}": 1}, 0.05)
        out["beta"][g] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for name, (a, b) in {"L-REF": ("LVSD", "REF"), "A-REF": ("AF", "REF"), "H-REF": ("CTRL", "REF")}.items():
        c95, c90 = AN.contrast(res, {f"t_{a}": 1, f"t_{b}": -1}, 0.05), AN.contrast(res, {f"t_{a}": 1, f"t_{b}": -1}, 0.10)
        out["contrasts"][name] = {"est": round(c95[0], 4), "ci95": [round(c95[1], 4), round(c95[2], 4)],
                                  "ci90": [round(c90[1], 4), round(c90[2], 4)], "class": AN.classify(c95, c90, delta)}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_model = SCORE.load_model(device)
    d_model, _ = _mod("deepecg_check", PROJ / "analysis/14_deepecg_check.py").load_model(device)
    preps = preps_from_json()
    info = {}

    # MIMIC REF: build, then score with both models
    wm, info["mimic"] = ref_mimic()
    wm = wm.sort_values("ecg_time")
    wm["age_at_ecg"] = (wm.anchor_age + wm.ecg_time.dt.year - wm.anchor_year).groupby(wm.subject_id).transform("first")
    wm = wm.merge(SCORE.machine_tabular(duckdb.connect()), on="study_id", how="left")
    wm["male"] = wm.male.astype(int)
    wm = SCORE.score_frame(wm, e_model, device, 24)
    wm = D15.score(wm.assign(src="mimic"), "src", "path", preps, d_model, device)
    wm.to_parquet(WORK / "r3_ref_mimic.parquet", index=False)

    # HEEDB REF per site
    parts = []
    for site, (_, sexcol) in COH.SITES.items():
        w, info[site] = ref_heedb_site(site, sexcol)
        parts.append(w)
    wh = pd.concat(parts, ignore_index=True).sort_values("ecg_time")
    wh["age_at_ecg"] = wh.groupby(["site", "pid"]).age_at_ecg_real.transform("first")
    wh["male"] = wh.male.astype(int)
    wh = HS.score_frame(wh, e_model, device)
    wh = D15.score(wh.assign(src=wh.site), "src", "rel", preps, d_model, device)
    wh.to_parquet(WORK / "r3_ref_heedb.parquet", index=False)

    def ref_rows(w, keycols):
        r = w.assign(g="REF", grp="REF", t=w.t_years, male_anchor=w.male.astype(bool))
        r["entity"] = "REF:" + r[keycols].astype(str).agg(":".join, axis=1)
        return r

    delta_e = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    delta_d = json.loads((RESULTS / "wp_d_g2.json").read_text())["delta_D_points_per_year"]
    cols = ["entity", "g", "t", "age", "male_anchor", "logit_lvef"]
    dm, _ = AN.load("5y_limbfx")
    dm = AN.analysis_set(dm, "primary")
    dmd = D16.with_score(dm, pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    dh = H.load()
    dhd = D16.with_score(dh, pd.read_parquet(WORK / "wp_d_scores_heedb.parquet"), ["site", "grp", "pid", "rel"])
    rm, rh = ref_rows(wm, ["subject_id"]), ref_rows(wh, ["site", "pid"])
    out = {"run_date": str(date.today()), "post_hoc": True, "ref_build": info,
           "ref_age_median": {"mimic": float(rm.drop_duplicates("entity").age.median()), "heedb": float(rh.drop_duplicates("entity").age.median())},
           "echonext": {"mimic": block4(pd.concat([dm[cols], rm[cols]]), delta_e),
                        "heedb": block4(pd.concat([dh[cols], rh[cols]]), delta_e)},
           "deepecg": {"mimic": block4(pd.concat([dmd[cols], rm.assign(logit_lvef=rm.score_d)[cols]]), delta_d),
                       "heedb": block4(pd.concat([dhd[cols], rh.assign(logit_lvef=rh.score_d)[cols]]), delta_d)}}
    (RESULTS / "robustness_r3.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
