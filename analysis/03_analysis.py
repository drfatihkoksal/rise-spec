"""WP3 — analysis (concept.md §4, §6.4–§6.6).

    python3 -I analysis/03_analysis.py            # primary, event study, level, S1–S4, readout
    python3 -I analysis/03_analysis.py --verify   # also checks the within estimator against LSDV

Input: work/wp2_scores_<tag>.parquet, work/wp1_anchors_<tag>.parquet, results/wp2_g2.json (delta).
Output: results/wp3_results.json (aggregate only).

Model (§6.4), t = years relative to anchor:
    logit(p_ij) = a_i + sum_g b_g * t_ij * 1[g_i = g] + c * t_ij * age_c,i + d * t_ij * male_i + e_ij
Patient fixed effects by within transformation; SEs cluster-robust by patient (t with G-1 df).
"""
import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
GROUPS = ["LVSD", "AF", "CTRL"]  # CTRL = non-cardiac control (LIMBFX after G1; "H" in §4)
BINS = [(-np.inf, -36, "m60_36"), (-36, -24, "m36_24"), (-24, -12, "m24_12"), (-12, -6, "m12_6"), (-6, np.inf, "m6_1")]
REF_BIN = "m60_36"  # §6.5: furthest from the event


def load(tag):
    s = pd.read_parquet(WORK / f"wp2_scores_{tag}.parquet")
    a = pd.read_parquet(WORK / f"wp1_anchors_{tag}.parquet")[["cohort", "grp", "subject_id", "age", "male"]]
    d = s.merge(a.rename(columns={"male": "male_anchor"}), on=["cohort", "grp", "subject_id"], how="left")
    control = [g for g in d.grp.unique() if g not in ("LVSD", "AF", "LVSD_ECHO")][0]
    d["g"] = d.grp.replace({control: "CTRL", "LVSD_ECHO": "LVSD"})
    d["entity"] = d.cohort + ":" + d.grp + ":" + d.subject_id.astype(str)
    d["t"] = d.t_years
    m = d.t * 12
    d["bin"] = pd.cut(m, [b[0] for b in BINS] + [np.inf], labels=[b[2] for b in BINS], right=False).astype(str)
    return d, control


def analysis_set(d, which):
    """Primary: primary cohort. S4: echo-anchored LVSD + primary AF/CTRL, minus anyone in both."""
    if which == "S4":
        lv = d[(d.cohort == "S4")]
        other = d[(d.cohort == "primary") & (d.g != "LVSD") & ~d.subject_id.isin(lv.subject_id)]
        return pd.concat([lv, other], ignore_index=True)
    return d[d.cohort == "primary"].copy()


def fit(df, terms, weights=None):
    """Within (FE) OLS/WLS with patient-clustered SEs. terms: dict name -> column values."""
    df = df.copy()
    X = pd.DataFrame(terms, index=df.index)
    X = X.loc[:, X.abs().sum() > 0]
    y = df.logit_lvef
    # drop patients with one ECG (no within variation)
    keep = df.groupby("entity").entity.transform("size") >= 2
    df, X, y = df[keep], X[keep], y[keep]
    Xd = X - X.groupby(df.entity).transform("mean")
    yd = y - y.groupby(df.entity).transform("mean")
    groups = pd.factorize(df.entity)[0]
    model = sm.WLS(yd, Xd, weights=weights[keep]) if weights is not None else sm.OLS(yd, Xd)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
    return res, df


def slope_terms(df):
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    t = {f"t_{g}": df.t * (df.g == g) for g in GROUPS}
    t["t_x_age_c"] = df.t * age_c
    t["t_x_male"] = df.t * df.male_anchor.astype(float)
    return t


def contrast(res, weights_by_name, alpha):
    names = list(res.params.index)
    r = np.zeros(len(names))
    for n, w in weights_by_name.items():
        r[names.index(n)] = w
    tt = res.t_test(r)
    lo, hi = np.asarray(tt.conf_int(alpha=alpha)).ravel()
    return float(np.asarray(tt.effect).ravel()[0]), float(lo), float(hi)


def classify(est_ci95, ci90, delta):
    _, lo95, hi95 = est_ci95
    _, lo90, hi90 = ci90
    if lo90 > -delta and hi90 < delta:
        return "≡"
    if lo95 > 0:
        return ">"
    if hi95 < 0:
        return "<"
    return "?"


CONTRASTS = {"L-A": {"t_LVSD": 1, "t_AF": -1}, "L-H": {"t_LVSD": 1, "t_CTRL": -1}, "A-H": {"t_AF": 1, "t_CTRL": -1}}


def slope_block(res, df, delta):
    out = {"patients": {g: int(df[df.g == g].entity.nunique()) for g in GROUPS},
           "ecgs": {g: int((df.g == g).sum()) for g in GROUPS},
           "beta": {}, "contrasts": {}}
    for g in GROUPS:
        e, lo, hi = contrast(res, {f"t_{g}": 1}, 0.05)
        out["beta"][g] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for name, w in CONTRASTS.items():
        c95, c90 = contrast(res, w, 0.05), contrast(res, w, 0.10)
        out["contrasts"][name] = {"est": round(c95[0], 4), "ci95": [round(c95[1], 4), round(c95[2], 4)],
                                  "ci90": [round(c90[1], 4), round(c90[2], 4)], "class": classify(c95, c90, delta)}
    return out


def readout(block):
    c = {k: v["class"] for k, v in block["contrasts"].items()}
    lvsd_pos = block["beta"]["LVSD"]["ci95"][0] > 0
    if c["L-A"] == ">" and c["L-H"] == ">" and lvsd_pos:
        return 1, "Target-specific pre-clinical signal (H1)"
    if c["L-A"] == "≡" and c["L-H"] == ">" and c["A-H"] == ">":
        return 2, "Cardiac-generic, not LV-specific"
    if c["L-A"] == "≡" and c["L-H"] == "≡" and c["A-H"] == "≡":
        return 3, "Generic pre-event drift / acquisition context (H0)"
    if c["L-A"] == "<":
        return 4, "Score rises more before AF than before LVSD; against H1"
    return 5, "Inconclusive or unanticipated: no specificity or non-specificity claim"


def event_study(df):
    """Same model with time bins x group (and bins x age_c, bins x male); reference bin per §6.5."""
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    terms = {}
    for b in [b[2] for b in BINS if b[2] != REF_BIN]:
        ind = (df.bin == b).astype(float)
        for g in GROUPS:
            terms[f"{b}_{g}"] = ind * (df.g == g)
        terms[f"{b}_x_age_c"] = ind * age_c
        terms[f"{b}_x_male"] = ind * df.male_anchor.astype(float)
    res, used = fit(df, terms)
    out = {"ecgs_by_bin": {g: used[used.g == g].bin.value_counts().reindex([b[2] for b in BINS], fill_value=0).astype(int).to_dict()
                           for g in GROUPS}, "theta": {}}
    for b in [b[2] for b in BINS if b[2] != REF_BIN]:
        out["theta"][b] = {}
        for g in GROUPS:
            if f"{b}_{g}" in res.params:
                e, lo, hi = contrast(res, {f"{b}_{g}": 1}, 0.05)
                out["theta"][b][g] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    e, lo, hi = contrast(res, {"m6_1_LVSD": 1, "m6_1_AF": -1}, 0.05)
    out["key_secondary_LVSD_minus_AF_m6_1"] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def level(df):
    """Mean logit(p) in the earliest bin per group, from patient means (between-person, Croon-type)."""
    e = df[df.bin == REF_BIN].groupby(["g", "entity"]).logit_lvef.mean().reset_index()
    out = {}
    for g in GROUPS:
        v = e[e.g == g].logit_lvef
        se = v.std(ddof=1) / np.sqrt(len(v))
        out[g] = {"patients": int(len(v)), "mean_logit": round(float(v.mean()), 4),
                  "ci95": [round(float(v.mean() - 1.96 * se), 4), round(float(v.mean() + 1.96 * se), 4)]}
    return out


def lsdv_check(df):
    """Point estimates from explicit patient dummies; must equal the within estimator."""
    df = df[df.groupby("entity").entity.transform("size") >= 2]
    X = pd.DataFrame(slope_terms(df), index=df.index)
    D = pd.get_dummies(df.entity, dtype=float)
    beta = np.linalg.lstsq(np.column_stack([X.to_numpy(), D.to_numpy()]), df.logit_lvef.to_numpy(), rcond=None)[0]
    return dict(zip(X.columns, beta[:X.shape[1]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="5y_limbfx")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    delta = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    d, control = load(args.tag)

    prim = analysis_set(d, "primary")
    res, used = fit(prim, slope_terms(prim))
    primary = slope_block(res, used, delta)
    primary["covariates"] = {k: round(float(res.params[k]), 4) for k in ["t_x_age_c", "t_x_male"]}
    rule, reading = readout(primary)
    out = {"run_date": str(date.today()), "tag": args.tag, "control": control, "delta": delta,
           "primary": primary, "primary_estimand_L_minus_A": primary["contrasts"]["L-A"],
           "readout": {"rule": rule, "reading": reading},
           "event_study": event_study(prim), "level": level(prim), "sensitivity": {}}

    subsets = {
        "S1_outside_stays": prim[~prim.in_stay],
        "S2_qrs_lt120": prim[prim.qrs_ms < 120],
    }
    for name, df in subsets.items():
        r, u = fit(df, slope_terms(df))
        blk = slope_block(r, u, delta)
        blk["readout_rule"] = readout(blk)[0]
        out["sensitivity"][name] = blk
    w = 1.0 / prim.groupby("entity").entity.transform("size")
    r, u = fit(prim, slope_terms(prim), weights=w)
    blk = slope_block(r, u, delta)
    blk["readout_rule"] = readout(blk)[0]
    out["sensitivity"]["S3_equal_patient_weight"] = blk
    s4 = analysis_set(d, "S4")
    r, u = fit(s4, slope_terms(s4))
    blk = slope_block(r, u, delta)
    blk["readout_rule"] = readout(blk)[0]
    blk["dropped_overlap_patients"] = int(d[(d.cohort == "primary") & (d.g != "LVSD") & d.subject_id.isin(d[d.cohort == "S4"].subject_id)].entity.nunique())
    out["sensitivity"]["S4_echo_anchor"] = blk

    if args.verify:
        lsdv = lsdv_check(prim)
        out["verify_lsdv_max_abs_diff"] = float(max(abs(lsdv[k] - res.params[k]) for k in lsdv))

    (RESULTS / "wp3_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
