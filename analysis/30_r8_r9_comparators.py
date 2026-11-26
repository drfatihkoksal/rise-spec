"""Post hoc R8 and R9 (concept.md §6.9): coded diastolic heart failure (DHF) and aortic stenosis (AS) as comparators.

    python3 -I analysis/30_r8_r9_comparators.py

Groups and rules: 29_comparator_counts.py (DHF: I50.3x / 428.3x; AS: I35.0, I35.2, ICD-10 only). The ECGs of
the new groups are scored with both frozen models exactly as the event groups were (02_score.py, 12_heedb_score.py,
15_deepecg_score.py: age input frozen at the first eligible ECG; in HEEDB PR, QRS and QTc held constant).

R8   DHF added to the primary model (§6.4) as a fourth group; contrasts LVSD-DHF (key), DHF-AF, DHF-LIMBFX;
     event study with the same four groups; score level 60 to 36 months before the anchor (descriptive).
R8-E MIMIC only: TTE nearest to the anchor within +-90 days; (a) DHF with LVEF >=50% and LVSD with LVEF <50%;
     (b) the same DHF group and LVSD with LVEF <=45%. AF and LIMBFX unchanged.
R9   AS added to the R8 model as a fifth group; contrasts LVSD-AS, AS-AF, AS-LIMBFX. Where the gate failed
     (MIMIC, 106 patients) the contrasts are reported without a class.
Classes and the margin are those of §4 (same Delta per model).

Outputs: work/r8_r9_{mimic,heedb}.parquet (patient-level, local only), results/robustness_r8_r9.json (aggregate).
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
G4, G5 = ["LVSD", "AF", "CTRL", "DHF"], ["LVSD", "AF", "CTRL", "DHF", "AS"]
SHORT = {"LVSD": "L", "AF": "A", "CTRL": "H", "DHF": "D", "AS": "S"}
PAIRS_R8 = [("LVSD", "DHF"), ("DHF", "AF"), ("DHF", "CTRL"), ("LVSD", "AF"), ("LVSD", "CTRL"), ("AF", "CTRL")]
PAIRS_R9 = [("LVSD", "AS"), ("AS", "AF"), ("AS", "CTRL"), ("AS", "DHF")]
READING = {">": "steeper before systolic heart failure: a component specific to systolic dysfunction",
           "≡": "equivalent: the rise is shared by coded heart failure whatever the systolic function",
           "<": "steeper before diastolic heart failure: against specificity to systolic function",
           "?": "neither shown nor excluded: no claim"}


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R3 = _mod("reference_group", PROJ / "analysis/18_reference_group.py")  # loads and registers the scoring and analysis modules
CNT = _mod("comparator_counts", PROJ / "analysis/29_comparator_counts.py")
SCORE, HS, D15, D16, AN, H = R3.SCORE, R3.HS, R3.D15, R3.D16, R3.AN, R3.H
BIN_NAMES = [b[2] for b in AN.BINS]


def add_bin(d):
    d["bin"] = pd.cut(d.t * 12, [b[0] for b in AN.BINS] + [np.inf], labels=BIN_NAMES, right=False).astype(str)
    return d


# --- cohorts and scores of the new groups ---------------------------------------------------
def protocol_groups(df):
    """Keep DHF and the ICD-10 aortic stenosis group (AS10 in the count script), named AS as in the protocol."""
    return df[df.grp.isin(["DHF", "AS10"])].assign(grp=lambda d: d.grp.replace({"AS10": "AS"}))


def new_mimic(e_model, d_model, preps, device):
    res, fr = CNT.mimic()
    pat = pd.read_csv(SCORE.HOSP / "patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    p = protocol_groups(fr["patients"]).merge(pat, on="subject_id")
    p["age"] = p.anchor_age + p.anchor.dt.year - p.anchor_year
    p["male"] = (p.gender == "M").astype(int)
    w = protocol_groups(fr["ecgs"]).merge(p, on=["grp", "subject_id"]).sort_values("ecg_time")
    w["t_years"] = (w.ecg_time - w.anchor).dt.total_seconds() / (365.25 * 86400)
    w["age_at_ecg_real"] = w.anchor_age + w.ecg_time.dt.year - w.anchor_year
    w["age_at_ecg"] = w.groupby(["grp", "subject_id"]).age_at_ecg_real.transform("first")  # frozen (§6.1)
    w = w.merge(SCORE.machine_tabular(duckdb.connect()), on="study_id", how="left")
    allmiss = w[["ventricular_rate", "pr_interval", "qrs_duration", "qt_corrected"]].isna().all(axis=1)  # as 02_score.py
    w = SCORE.score_frame(w[~allmiss], e_model, device, 24)
    w = D15.score(w.assign(src="mimic"), "src", "path", preps, d_model, device)
    w["entity"] = "new:" + w.grp + ":" + w.subject_id.astype(str)
    return w, fr["lvef"], {"counts": res["gate"], "ecgs_without_measurements_dropped": int(allmiss.sum())}


def new_heedb(e_model, d_model, preps, device):
    res, fr = CNT.heedb()
    w = protocol_groups(fr["ecgs"]).merge(protocol_groups(fr["patients"]), on=["site", "grp", "pid"]).sort_values("ecg_time")
    w["t_years"] = (w.ecg_time - pd.to_datetime(w.anchor_d)).dt.total_seconds() / (365.25 * 86400)
    w["age_at_ecg_real"] = (w.ecg_time.dt.normalize() - w.dob.dt.normalize()).dt.days / 365.25
    w["age_at_ecg"] = w.groupby(["site", "grp", "pid"]).age_at_ecg_real.transform("first")  # frozen (§6.1)
    w["male"] = w.male.astype(int)
    w = HS.score_frame(w, e_model, device)
    w = D15.score(w.assign(src=w.site), "src", "rel", preps, d_model, device)
    w["entity"] = "new:" + w.site + ":" + w.grp + ":" + w.pid.astype(str)
    return w, {"counts": res["gate"]}


def rows(w, grp, score):
    """Analysis rows of one new group."""
    r = w[w.grp == grp].assign(g=grp, t=lambda d: d.t_years, male_anchor=lambda d: d.male.astype(bool))
    r["logit_lvef"] = r[score]
    return add_bin(r)[COLS]


COLS = ["entity", "g", "t", "bin", "age", "male_anchor", "logit_lvef"]


# --- models -----------------------------------------------------------------------------------
def slope_terms(df, groups):
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    t = {f"t_{g}": df.t * (df.g == g) for g in groups}
    t["t_x_age_c"] = df.t * age_c
    t["t_x_male"] = df.t * df.male_anchor.astype(float)
    return t


def slopes(df, groups, pairs, delta, no_class=()):
    res, used = AN.fit(df, slope_terms(df, groups))
    out = {"patients": {g: int(used[used.g == g].entity.nunique()) for g in groups},
           "ecgs": {g: int((used.g == g).sum()) for g in groups}, "beta": {}, "contrasts": {}}
    for g in groups:
        e, lo, hi = AN.contrast(res, {f"t_{g}": 1}, 0.05)
        out["beta"][g] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    for a, b in pairs:
        w = {f"t_{a}": 1, f"t_{b}": -1}
        c95, c90 = AN.contrast(res, w, 0.05), AN.contrast(res, w, 0.10)
        gated = a in no_class or b in no_class
        out["contrasts"][f"{SHORT[a]}-{SHORT[b]}"] = {
            "est": round(c95[0], 4), "ci95": [round(c95[1], 4), round(c95[2], 4)], "ci90": [round(c90[1], 4), round(c90[2], 4)],
            "class": None if gated else AN.classify(c95, c90, delta)}
    return out


def event_study(df, groups):
    """§6.5 with the listed groups: time bins x group, bins x age_c, bins x male; reference 60 to 36 months."""
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    terms, bins = {}, [b for b in BIN_NAMES if b != AN.REF_BIN]
    for b in bins:
        ind = (df.bin == b).astype(float)
        for g in groups:
            terms[f"{b}_{g}"] = ind * (df.g == g)
        terms[f"{b}_x_age_c"] = ind * age_c
        terms[f"{b}_x_male"] = ind * df.male_anchor.astype(float)
    res, used = AN.fit(df, terms)
    out = {"ecgs_by_bin": {g: used[used.g == g].bin.value_counts().reindex(BIN_NAMES, fill_value=0).astype(int).to_dict() for g in groups},
           "theta": {b: {} for b in bins}, "LVSD_minus_DHF": {}}
    for b in bins:
        for g in groups:
            e, lo, hi = AN.contrast(res, {f"{b}_{g}": 1}, 0.05)
            out["theta"][b][g] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        if "DHF" in groups:
            e, lo, hi = AN.contrast(res, {f"{b}_LVSD": 1, f"{b}_DHF": -1}, 0.05)
            out["LVSD_minus_DHF"][b] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def level(df, groups):
    """Mean score 60 to 36 months before the anchor, from patient means (between patients; as AN.level)."""
    e = df[df.bin == AN.REF_BIN].groupby(["g", "entity"]).logit_lvef.mean().reset_index()
    out = {}
    for g in groups:
        v = e[e.g == g].logit_lvef
        se = v.std(ddof=1) / np.sqrt(len(v))
        out[g] = {"patients": int(len(v)), "mean": round(float(v.mean()), 4),
                  "ci95": [round(float(v.mean() - 1.96 * se), 4), round(float(v.mean() + 1.96 * se), 4)]}
    return out


def describe(w, keys):
    out = {}
    for grp, d in w.groupby("grp"):
        per = d.groupby(keys).agg(n=("t_years", "size"), span=("t_years", lambda t: t.max() - t.min()), age=("age", "first"), male=("male", "first"))
        q = lambda v: [round(float(x), 1) for x in np.percentile(v, [50, 25, 75])]
        out[grp] = {
            "patients": int(len(per)), "ecgs": int(len(d)), "age_median_iqr": q(per.age), "men": int(per.male.sum()),
            "ecgs_per_patient_median_iqr": q(per.n), "years_covered_median_iqr": q(per.span),
            "years_from_ecg_to_anchor_median_iqr": q(-d.t_years)}
    return out


def r8_r9(base, new, score, delta, as_gate_passed):
    d4 = pd.concat([base[COLS], rows(new, "DHF", score)], ignore_index=True)
    d5 = pd.concat([d4, rows(new, "AS", score)], ignore_index=True)
    r8 = slopes(d4, G4, PAIRS_R8, delta)
    r8["reading_LVSD_minus_DHF"] = READING[r8["contrasts"]["L-D"]["class"]]
    r8["event_study"] = event_study(d4, G4)
    r8["level"] = level(d4, G4)
    r9 = slopes(d5, G5, PAIRS_R9, delta, no_class=() if as_gate_passed else ("AS",))
    r9["gate_passed"] = as_gate_passed
    r9["level_AS"] = level(d5, ["AS"])["AS"]
    return {"R8": r8, "R9": r9}


def r8e(base, new, lvef, score, delta):
    """LVEF-restricted refits (MIMIC). base: primary frame with subject_id; lvef: nearest TTE per group and patient."""
    ef_d = lvef[lvef.grp == "DHF"].set_index("subject_id").ef
    ef_l = lvef[lvef.grp == "LVSD"].set_index("subject_id").ef
    dhf = new[(new.grp == "DHF") & new.subject_id.map(ef_d).ge(50)]
    out = {}
    for name, keep in (("a_LVSD_lt50", ef_l[ef_l < 50].index), ("b_LVSD_le45", ef_l[ef_l <= 45].index)):
        b = base[(base.g != "LVSD") | base.subject_id.isin(keep)]
        out[name] = slopes(pd.concat([b[COLS], rows(dhf, "DHF", score)], ignore_index=True), G4, PAIRS_R8, delta)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e_model = SCORE.load_model(device)
    d_model, _ = _mod("deepecg_check", PROJ / "analysis/14_deepecg_check.py").load_model(device)
    preps = R3.preps_from_json()

    wm, lvef, info_m = new_mimic(e_model, d_model, preps, device)
    wm.to_parquet(WORK / "r8_r9_mimic.parquet", index=False)
    wh, info_h = new_heedb(e_model, d_model, preps, device)
    wh.to_parquet(WORK / "r8_r9_heedb.parquet", index=False)

    delta_e = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    delta_d = json.loads((RESULTS / "wp_d_g2.json").read_text())["delta_D_points_per_year"]
    dm, _ = AN.load("5y_limbfx")
    dm = AN.analysis_set(dm, "primary")
    dmd = D16.with_score(dm, pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    dh = H.load()
    dhd = D16.with_score(dh, pd.read_parquet(WORK / "wp_d_scores_heedb.parquet"), ["site", "grp", "pid", "rel"])
    gate_m, gate_h = info_m["counts"]["pass"]["AS10"], info_h["counts"]["pass"]["AS10"]

    out = {"run_date": str(date.today()), "post_hoc": True, "delta": {"echonext": delta_e, "deepecg": delta_d},
           "groups": {"mimic": {**info_m, "describe": describe(wm, ["grp", "subject_id"])},
                      "heedb": {**info_h, "describe": describe(wh, ["site", "grp", "pid"])}},
           "echonext": {"mimic": r8_r9(dm, wm, "logit_lvef", delta_e, gate_m), "heedb": r8_r9(dh, wh, "logit_lvef", delta_e, gate_h)},
           "deepecg": {"mimic": r8_r9(dmd, wm, "score_d", delta_d, gate_m), "heedb": r8_r9(dhd, wh, "score_d", delta_d, gate_h)}}
    out["echonext"]["mimic"]["R8E"] = r8e(dm, wm, lvef, "logit_lvef", delta_e)
    out["deepecg"]["mimic"]["R8E"] = r8e(dmd, wm, lvef, "score_d", delta_d)
    (RESULTS / "robustness_r8_r9.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=int))
    print(json.dumps(out, indent=2, ensure_ascii=False, default=int))


if __name__ == "__main__":
    main()
