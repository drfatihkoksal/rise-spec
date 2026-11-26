"""Post hoc R10 and R11 (concept.md §6.9), both descriptive.

    python3 -I analysis/31_r10_r11_parts_and_selection.py

R10  EchoNext-Mini: the tabular inputs enter the final linear layer, so
         logit = waveform part + sum_k w_k * z_k   (k: ventricular rate, PR, QRS, QTc; scaled inputs z_k).
     The primary model (§6.4) and the R8 model (with DHF) are refitted with each part as the outcome; the slopes of
     the parts add up to the slope of the score. In HEEDB only ventricular rate varies. A numerical check on 500
     random MIMIC cohort ECGs compares rescoring with the four inputs at the training medians against w_k * dz_k.
R11  Per group and data source: patients who entered the analysis against patients removed by the ECG rules, among
     those left after the exclusions that do not depend on ECGs (age at the anchor, sex). No score is read.

Output: results/robustness_r10_r11.json (aggregate only).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
SEED = 20261126


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


M30 = _mod("r8_r9", PROJ / "analysis/30_r8_r9_comparators.py")  # loads the scoring, cohort and analysis modules
SCORE, AN, H, CNT, R3 = M30.SCORE, M30.AN, M30.H, M30.CNT, M30.R3
FEATS = SCORE.TAB_PARAMS["features"]                       # age, ventricular rate, atrial rate, PR, QRS, QTc
COL = {f: i + 1 for i, f in enumerate(FEATS)}              # position in the 7-vector (male first)
VARYING = ["ventricular_rate", "pr_interval", "qrs_duration", "qt_corrected"]
G3, G4 = ["LVSD", "AF", "CTRL"], M30.G4
SHORT = M30.SHORT


# --- R10 --------------------------------------------------------------------------------------
def with_parts(d, w_tab, heedb):
    """Adds part_<input> = w_k * z_k and part_waveform = logit - sum of the parts."""
    d = d.copy()
    names = ["ventricular_rate"] if heedb else VARYING
    tab_in = d.assign(atrial_rate=np.nan, pr_interval=np.nan, qrs_duration=np.nan, qt_corrected=np.nan) if heedb else d
    z = SCORE.tabular(tab_in)
    for k in names:
        d[f"part_{k}"] = w_tab[COL[k]] * z[:, COL[k]]
    d["part_waveform"] = d.logit_lvef - d[[f"part_{k}" for k in names]].sum(axis=1)
    return d, names + ["waveform"]


def part_slopes(df, parts, groups, pairs, terms):
    est = {}
    for name in ["total"] + parts:
        y = df.logit_lvef if name == "total" else df[f"part_{name}"]
        res, _ = AN.fit(df.assign(logit_lvef=y), terms(df))
        e = {}
        for g in groups:
            v = AN.contrast(res, {f"t_{g}": 1}, 0.05)
            e[g] = {"est": round(v[0], 4), "ci95": [round(v[1], 4), round(v[2], 4)], "raw": v[0]}
        for a, b in pairs:
            v = AN.contrast(res, {f"t_{a}": 1, f"t_{b}": -1}, 0.05)
            e[f"{SHORT[a]}-{SHORT[b]}"] = {"est": round(v[0], 4), "ci95": [round(v[1], 4), round(v[2], 4)], "raw": v[0]}
        est[name] = e
    keys = list(est["total"])
    gap = max(abs(sum(est[p][k]["raw"] for p in parts) - est["total"][k]["raw"]) for k in keys)
    for name in parts:
        for k in keys:
            est[name][k]["share_of_total"] = round(est[name][k]["raw"] / est["total"][k]["raw"], 3)
    for name in est:
        for k in keys:
            del est[name][k]["raw"]
    return {"parts": est, "largest_gap_between_sum_of_parts_and_total": float(f"{gap:.2e}")}


def additivity_check(model, device, w_tab):
    d = pd.read_parquet(WORK / "wp2_scores_5y_limbfx.parquet").sample(500, random_state=SEED)
    waves = SCORE.preprocess_many(d.path.tolist(), 24)
    ok = np.array([x is not None for x in waves])
    d, wave = d[ok], np.stack([x for x in waves if x is not None])[:, None, :, :]
    tab = SCORE.tabular(d)
    med = dict(zip(FEATS, SCORE.TAB_PARAMS["impute_statistics_scaled"]))
    tab2 = tab.copy()
    for k in VARYING:
        tab2[:, COL[k]] = med[k]
    l1 = SCORE.logits(model, wave, tab, device)[:, SCORE.LVEF]
    l2 = SCORE.logits(model, wave, tab2, device)[:, SCORE.LVEF]
    pred = sum(w_tab[COL[k]] * (tab[:, COL[k]] - med[k]) for k in VARYING)
    return {"ecgs": int(len(d)), "largest_abs_deviation_logit": float(f"{np.abs(l1 - l2 - pred).max():.2e}"),
            "largest_abs_change_logit": round(float(np.abs(l1 - l2).max()), 3),
            "largest_abs_difference_from_stored_logit": float(f"{np.abs(l1 - d.logit_lvef.to_numpy()).max():.2e}")}


def r10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = torch.load(SCORE.WEIGHTS, map_location="cpu", weights_only=True)["model"]["output.weight"].numpy()
    w_tab = w[SCORE.LVEF, -7:]
    out = {"final_layer_weights_lvef_output": dict(zip(["male"] + FEATS, np.round(w_tab, 4).tolist())),
           "additivity_check_mimic": additivity_check(SCORE.load_model(device), device, w_tab)}
    dm, _ = AN.load("5y_limbfx")
    dm = AN.analysis_set(dm, "primary")
    dh = H.load()
    dh = dh.assign(male=dh.male_anchor.astype(int))
    new = {"mimic": pd.read_parquet(WORK / "r8_r9_mimic.parquet"), "heedb": pd.read_parquet(WORK / "r8_r9_heedb.parquet")}
    for src, base in (("mimic", dm), ("heedb", dh)):
        heedb = src == "heedb"
        b, parts = with_parts(base, w_tab, heedb)
        n = new[src][new[src].grp == "DHF"]
        n, _ = with_parts(n.assign(g="DHF", t=n.t_years, male_anchor=n.male.astype(bool)), w_tab, heedb)
        cols = ["entity", "g", "t", "age", "male_anchor", "logit_lvef"] + [f"part_{p}" for p in parts]
        d4 = pd.concat([b[cols], n[cols]], ignore_index=True)
        out[src] = {
            "primary_model": part_slopes(b, parts, G3, [("LVSD", "AF"), ("LVSD", "CTRL")], AN.slope_terms),
            "r8_model": part_slopes(d4, parts, G4, [("LVSD", "AF"), ("LVSD", "CTRL"), ("LVSD", "DHF")], lambda d: M30.slope_terms(d, G4)),
            "pr_missing_share": None if heedb else {g: round(float(v), 4) for g, v in pd.concat(
                [b[["g", "pr_interval"]], n[["g", "pr_interval"]]]).groupby("g").pr_interval.apply(lambda x: x.isna().mean()).items()}}
    return out


# --- R11 --------------------------------------------------------------------------------------
def q(v):
    return [round(float(x), 1) for x in np.percentile(v, [50, 25, 75])]


def compare(base, entered):
    """base: patients left after the exclusions that do not depend on ECGs (age, male); entered: boolean mask."""
    male = base.male.astype(float)
    out = {}
    for name, m in (("entered", entered), ("removed_by_ecg_rules", ~entered)):
        out[name] = {"patients": int(m.sum()), "age_median_iqr": q(base.age[m]), "men_pct": round(100 * float(male[m].mean()), 1)}
    a1, a0, p1, p0 = base.age[entered], base.age[~entered], male[entered].mean(), male[~entered].mean()
    out["standardized_difference"] = {
        "age": round(float((a1.mean() - a0.mean()) / np.sqrt((a1.var() + a0.var()) / 2)), 3),
        "men": round(float((p1 - p0) / np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)), 3)}
    out["entered_pct"] = round(100 * float(entered.mean()), 1)
    return out


def r11():
    out = {"mimic": {}, "heedb": {}}
    # MIMIC, pre-specified groups
    c1 = R3.C1
    con = duckdb.connect()
    c1.load(con, "LIMBFX")
    steps = c1.primary_anchors(con, "LIMBFX")
    a = con.execute("SELECT * FROM anc_f").df()
    base = a[~a[steps[:-1]].any(axis=1)]                      # all exclusions except "no ECG at the anchor"
    fin = pd.read_parquet(WORK / "wp1_anchors_5y_limbfx.parquet").query("cohort == 'primary'")
    for g, b in base.groupby("grp"):
        out["mimic"]["CTRL" if g == "LIMBFX" else g] = compare(b, b.subject_id.isin(fin[fin.grp == g].subject_id))
    # MIMIC, new groups
    _, fr = CNT.mimic()
    old = set(pd.read_parquet(WORK / "wp1_anchors_5y_limbfx.parquet").subject_id)
    ent = fr["entered"]
    dhf = set(ent[(ent.grp == "DHF") & ~ent.in_other_group].subject_id)
    for g, name in (("DHF", "DHF"), ("AS10", "AS")):
        b = fr["anchors"][fr["anchors"].grp == g]
        b = b[~b[["ex_prior_hf_cm", "ex_lvsd_within_1y", "ex_age"]].any(axis=1) & ~b.subject_id.isin(old)]
        b = b[~b.subject_id.isin(dhf)] if g != "DHF" else b
        out["mimic"][name] = compare(b, b.subject_id.isin(ent[(ent.grp == g) & ~ent.in_other_group].subject_id))

    # HEEDB, pre-specified groups
    coh = R3.COH
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{coh.HEEDB}/duckdb_tmp'; SET memory_limit = '40GB'; SET threads = 24")
    parts = []
    for site, (_, sexcol) in coh.SITES.items():
        coh.load_site(con, site, sexcol)
        steps = coh.anchors(con, site)
        a = con.execute("SELECT * FROM anc_f").df()
        parts.append(a[~a.ex_multi_set & ~a[steps[:-1]].any(axis=1)])
    base = pd.concat(parts, ignore_index=True)
    fin = pd.read_parquet(WORK / "wp_h_anchors.parquet")
    key = lambda d: d.site + ":" + d.pid.astype(str)
    for g, b in base.groupby("grp"):
        out["heedb"]["CTRL" if g == "LIMBFX" else g] = compare(b, key(b).isin(key(fin[fin.grp == g])))
    # HEEDB, new groups
    _, fr = CNT.heedb()
    old, ent = set(key(fin)), fr["entered"]
    dhf = set(key(ent[(ent.grp == "DHF") & ~ent.in_other_group]))
    flags = ["ex_prior_hf_cm", "ex_lvsd_within_1y", "ex_lookback_lt_1y", "ex_age_or_sex_missing"]
    for g, name in (("DHF", "DHF"), ("AS10", "AS")):
        b = fr["anchors"][fr["anchors"].grp == g]
        b = b[~b[flags].any(axis=1) & ~key(b).isin(old)]
        b = b[~key(b).isin(dhf)] if g != "DHF" else b
        out["heedb"][name] = compare(b, key(b).isin(key(ent[(ent.grp == g) & ~ent.in_other_group])))
    return out


def main():
    out = {"run_date": str(date.today()), "post_hoc": True, "R10": r10(), "R11": r11()}
    (RESULTS / "robustness_r10_r11.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=int))
    print(json.dumps(out, indent=2, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
