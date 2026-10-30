"""Exploratory R6 (concept.md §6.9): event-free reference with a random calendar-date pseudo-anchor (REF-D).

    python3 -I analysis/21_r6_random_date.py

REF-D: R3 inclusion rules (no LVSD, AF, LIMBFX, HF or cardiomyopathy code at any time), pseudo-anchor uniform
between 180 days after the patient's first ECG and the patient's last ECG (md5-seeded), then the window and
ECG rules of the event groups; 2,000 entering patients in MIMIC and 2,000 per HEEDB site; both models.
Model with five groups (LVSD, AF, LIMBFX, REF, REF-D). Outputs: work/r6_refd_{mimic,heedb}.parquet
(patient-level, local only), results/robustness_r6.json (aggregate).
"""
import hashlib
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R3 = _mod("reference_group", PROJ / "analysis/18_reference_group.py")
SCORE, C1, COH, HS, AN, H, D15, D16 = R3.SCORE, R3.C1, R3.COH, R3.HS, R3.AN, R3.H, R3.D15, R3.D16
SEED, N_REF, N_CAND = R3.SEED, R3.N_REF, R3.N_CAND
GROUPS5 = ["LVSD", "AF", "CTRL", "REF", "REFD"]


def frac(key):
    return int(hashlib.md5(f"{key}|date|{SEED}".encode()).hexdigest()[:13], 16) / 16**13


def random_anchor(spans, key):
    """spans: key, first, last -> anchor uniform in [first + 180 d, last]."""
    s = spans[(spans["last"] - spans["first"]).dt.days > 180].copy()
    lo = s["first"] + pd.Timedelta(days=180)
    s["anchor"] = lo + (s["last"] - lo) * s[key].map(frac)
    return s[[key, "anchor"]]


def refd_mimic():
    con = duckdb.connect()
    C1.load(con, "LIMBFX")
    con.execute("CREATE TABLE ev AS SELECT DISTINCT subject_id FROM adm_sets WHERE lvsd OR af OR ctrl OR hfcm")
    spans = con.execute("SELECT subject_id, min(t) AS first, max(t) AS last FROM ecg WHERE subject_id NOT IN (SELECT subject_id FROM ev) GROUP BY 1").df()
    con.register("anc_df", random_anchor(spans, "subject_id"))
    w = con.execute("""
    SELECT a.subject_id, a.anchor, e.study_id, e.path, e.t AS ecg_time,
           date_diff('second', a.anchor, e.t) / (365.25 * 86400) AS t_years,
           coalesce(s.sinus, false) AND NOT coalesce(s.bad_quality, false) AS stmt_ok,
           p.anchor_age + year(a.anchor) - p.anchor_year AS age, p.gender = 'M' AS male, p.anchor_age, p.anchor_year
    FROM anc_df a JOIN ecg e USING (subject_id) JOIN pat p USING (subject_id) LEFT JOIN stm s USING (study_id)
    WHERE e.t >= a.anchor - INTERVAL 5 YEAR AND e.t <= a.anchor - INTERVAL 30 DAY AND p.anchor_age + year(a.anchor) - p.anchor_year >= 18
    """).df()
    return finish(w, "subject_id", ["path"], WORK / "wp1_waveform_qc.parquet", C1.waveform_ok)


def refd_heedb_site(site, sexcol):
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{R3.HEEDB}/duckdb_tmp'; SET memory_limit = '40GB'; SET threads = 24")
    COH.load_site(con, site, sexcol)
    con.execute("CREATE TABLE ev AS SELECT DISTINCT pid FROM day_sets WHERE lvsd OR af OR ctrl OR hfcm")
    spans = con.execute("SELECT pid, min(t) AS first, max(t) AS last FROM meta WHERE pid NOT IN (SELECT pid FROM ev) GROUP BY 1").df()
    con.register("anc_df", random_anchor(spans, "pid"))
    w = con.execute(f"""
    SELECT '{site}' site, a.pid, a.anchor, m.rel, m.t AS ecg_time,
           date_diff('second', a.anchor, m.t) / (365.25 * 86400) AS t_years,
           date_diff('day', m.dob, m.t) / 365.25 AS age_at_ecg_real,
           coalesce(s.sinus, false) AND NOT coalesce(s.bad_quality, false) AS stmt_ok,
           date_diff('day', m.dob, a.anchor) / 365.25 AS age, m.male
    FROM anc_df a JOIN meta m USING (pid) LEFT JOIN stm s USING (rel) JOIN first_code fc USING (pid)
    WHERE m.t >= a.anchor - INTERVAL 5 YEAR AND m.t <= a.anchor - INTERVAL 30 DAY
      AND m.sex_known AND m.dob IS NOT NULL AND date_diff('day', m.dob, a.anchor) / 365.25 >= 18
      AND fc.first_d <= CAST(a.anchor AS DATE) - INTERVAL 365 DAY
    """).df()
    return finish(w, "pid", ["site", "rel"], WORK / "wp_h_waveform_qc.parquet", COH.waveform_ok)


def finish(w, key, qc_keys, cache_f, fn):
    w["eligible"] = w.stmt_ok
    cand = pd.Series(R3.entry(w, key), name=key).to_frame()
    cand["h"] = cand[key].map(R3.stable_rank)
    cand = cand.sort_values(["h", key]).head(N_CAND)
    w = w[w[key].isin(cand[key]) & w.stmt_ok]
    w = R3.qc_cached(w[qc_keys].drop_duplicates(), cache_f, fn, qc_keys).merge(w, on=qc_keys)
    w["eligible"] = w.waveform_ok.fillna(False).astype(bool)
    ok = R3.entry(w, key)
    keep = cand[cand[key].isin(ok)].head(N_REF)[key]
    return w[w[key].isin(keep) & w.eligible].copy(), {"candidates": int(len(cand)), "entering": int(len(ok)), "kept": int(len(keep))}


def block5(df, delta):
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    terms = {f"t_{g}": df.t * (df.g == g) for g in GROUPS5}
    terms["t_x_age_c"], terms["t_x_male"] = df.t * age_c, df.t * df.male_anchor.astype(float)
    res, used = AN.fit(df, terms)
    out = {"patients": {g: int(used[used.g == g].entity.nunique()) for g in GROUPS5}, "beta": {}, "contrasts": {}}
    for g in GROUPS5:
        e, lo, hi = AN.contrast(res, {f"t_{g}": 1}, 0.05)
        e90 = AN.contrast(res, {f"t_{g}": 1}, 0.10)
        out["beta"][g] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)], "ci90": [round(e90[1], 4), round(e90[2], 4)],
                          "class_vs_zero": AN.classify((e, lo, hi), e90, delta)}
    for name, (a, b) in {"REFD-REF": ("REFD", "REF"), "L-REFD": ("LVSD", "REFD"), "A-REFD": ("AF", "REFD"), "H-REFD": ("CTRL", "REFD")}.items():
        c95, c90 = AN.contrast(res, {f"t_{a}": 1, f"t_{b}": -1}, 0.05), AN.contrast(res, {f"t_{a}": 1, f"t_{b}": -1}, 0.10)
        out["contrasts"][name] = {"est": round(c95[0], 4), "ci95": [round(c95[1], 4), round(c95[2], 4)],
                                  "ci90": [round(c90[1], 4), round(c90[2], 4)], "class": AN.classify(c95, c90, delta)}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_model = SCORE.load_model(device)
    d_model, _ = _mod("deepecg_check", PROJ / "analysis/14_deepecg_check.py").load_model(device)
    preps = R3.preps_from_json()
    info = {}

    wm, info["mimic"] = refd_mimic()
    wm = wm.sort_values("ecg_time")
    wm["age_at_ecg"] = (wm.anchor_age + wm.ecg_time.dt.year - wm.anchor_year).groupby(wm.subject_id).transform("first")
    wm = wm.merge(SCORE.machine_tabular(duckdb.connect()), on="study_id", how="left")
    wm["male"] = wm.male.astype(int)
    wm = SCORE.score_frame(wm, e_model, device, 24)
    wm = D15.score(wm.assign(src="mimic"), "src", "path", preps, d_model, device)
    wm.to_parquet(WORK / "r6_refd_mimic.parquet", index=False)

    parts = []
    for site, (_, sexcol) in COH.SITES.items():
        w, info[site] = refd_heedb_site(site, sexcol)
        parts.append(w)
    wh = pd.concat(parts, ignore_index=True).sort_values("ecg_time")
    wh["age_at_ecg"] = wh.groupby(["site", "pid"]).age_at_ecg_real.transform("first")
    wh["male"] = wh.male.astype(int)
    wh = HS.score_frame(wh, e_model, device)
    wh = D15.score(wh.assign(src=wh.site), "src", "rel", preps, d_model, device)
    wh.to_parquet(WORK / "r6_refd_heedb.parquet", index=False)

    def rows(w, keycols, g):
        r = w.assign(g=g, grp=g, t=w.t_years, male_anchor=w.male.astype(bool))
        r["entity"] = g + ":" + r[keycols].astype(str).agg(":".join, axis=1)
        return r

    delta_e = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    delta_d = json.loads((RESULTS / "wp_d_g2.json").read_text())["delta_D_points_per_year"]
    cols = ["entity", "g", "t", "age", "male_anchor", "logit_lvef"]
    dm, _ = AN.load("5y_limbfx")
    dm = AN.analysis_set(dm, "primary")
    dmd = D16.with_score(dm, pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    dh = H.load()
    dhd = D16.with_score(dh, pd.read_parquet(WORK / "wp_d_scores_heedb.parquet"), ["site", "grp", "pid", "rel"])
    refm, refh = pd.read_parquet(WORK / "r3_ref_mimic.parquet"), pd.read_parquet(WORK / "r3_ref_heedb.parquet")
    rm, rh = rows(refm, ["subject_id"], "REF"), rows(refh, ["site", "pid"], "REF")
    dm_, dh_ = rows(wm, ["subject_id"], "REFD"), rows(wh, ["site", "pid"], "REFD")

    def both(ev, ref, refd, score):
        r = ref if score == "logit_lvef" else ref.assign(logit_lvef=ref.score_d)
        rd = refd if score == "logit_lvef" else refd.assign(logit_lvef=refd.score_d)
        return pd.concat([ev[cols], r[cols], rd[cols]], ignore_index=True)

    def gap(w, keys):
        """Median days from a patient's last window ECG to the pseudo-anchor."""
        g = w.groupby(keys).agg(anchor=("anchor", "first"), last=("ecg_time", "max"))
        return float((g.anchor - g["last"]).dt.days.median())
    out = {"run_date": str(date.today()), "exploratory": True, "refd_build": info,
           "median_days_last_window_ecg_to_anchor": {"REFD_mimic": gap(wm, "subject_id"), "REFD_heedb": gap(wh, ["site", "pid"]),
                                                     "REF_mimic": gap(refm, "subject_id"), "REF_heedb": gap(refh, ["site", "pid"])},
           "echonext": {"mimic": block5(both(dm, rm, dm_, "logit_lvef"), delta_e), "heedb": block5(both(dh, rh, dh_, "logit_lvef"), delta_e)},
           "deepecg": {"mimic": block5(both(dmd, rm, dm_, "score_d"), delta_d), "heedb": block5(both(dhd, rh, dh_, "score_d"), delta_d)}}
    (RESULTS / "robustness_r6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
