"""Post hoc R7 (sex), concept.md §6.9. Both models, both data sources, primary analysis set.

    python3 -I analysis/28_r7_sex.py

Stratified: the primary model refitted in women and in men (t x male dropped, t x age_c kept), with the
readout rules of §4. Pooled: one slope per group and sex plus t x age_c; reports the between-sex differences of
the LVSD - AF and LVSD - LIMBFX contrasts (women minus men) with their Δ classes.
Uses the existing scores; no new inference. Output: results/robustness_r7_sex.json (aggregate only).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


AN = _mod("analysis3", PROJ / "analysis/03_analysis.py")
H = _mod("heedb_analysis", PROJ / "analysis/13_heedb_analysis.py")
D = _mod("deepecg_analysis", PROJ / "analysis/16_deepecg_analysis.py")
SEXES = {"women": 0.0, "men": 1.0}


def stratum_terms(df):
    # within one sex the t x male term is constant (all 0 or equal to t) and is dropped
    t = AN.slope_terms(df)
    t.pop("t_x_male")
    return t


def stratified(df, delta):
    out = {}
    for sex, v in SEXES.items():
        s = df[df.male_anchor.astype(float) == v]
        r, u = AN.fit(s, stratum_terms(s))
        b = AN.slope_block(r, u, delta)
        b["readout_rule"], b["reading"] = AN.readout(b)
        out[sex] = b
    return out


def pooled(df, delta):
    male = df.male_anchor.astype(float)
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    terms = {f"t_{g}_{sex}": df.t * (df.g == g) * (male == v) for g in AN.GROUPS for sex, v in SEXES.items()}
    terms["t_x_age_c"] = df.t * age_c
    res, used = AN.fit(df, terms)
    out = {}
    for name, (a, b) in {"L-A": ("LVSD", "AF"), "L-H": ("LVSD", "CTRL")}.items():
        w = {f"t_{a}_women": 1, f"t_{b}_women": -1, f"t_{a}_men": -1, f"t_{b}_men": 1}
        c95, c90 = AN.contrast(res, w, 0.05), AN.contrast(res, w, 0.10)
        out[f"{name}_women_minus_men"] = {"est": round(c95[0], 4), "ci95": [round(c95[1], 4), round(c95[2], 4)],
                                          "ci90": [round(c90[1], 4), round(c90[2], 4)], "class": AN.classify(c95, c90, delta)}
    return out


def run(df, delta):
    s = stratified(df, delta)
    keep = lambda b: {k: b[k] for k in ("readout_rule", "reading", "patients", "ecgs", "beta", "contrasts")}
    return {"stratified": {k: keep(v) for k, v in s.items()}, "sex_difference": pooled(df, delta)}


def main():
    delta_e = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    delta_d = json.loads((RESULTS / "wp_d_g2.json").read_text())["delta_D_points_per_year"]

    dm, _ = AN.load("5y_limbfx")
    dm = AN.analysis_set(dm, "primary")
    dmd = D.with_score(dm, pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    dh = H.load()
    dhd = D.with_score(dh, pd.read_parquet(WORK / "wp_d_scores_heedb.parquet"), ["site", "grp", "pid", "rel"])

    out = {"run_date": str(date.today()), "post_hoc": True, "delta": {"echonext": delta_e, "deepecg": delta_d},
           "echonext": {"mimic": run(dm, delta_e), "heedb": run(dh, delta_e)},
           "deepecg": {"mimic": run(dmd, delta_d), "heedb": run(dhd, delta_d)}}
    (RESULTS / "robustness_r7_sex.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
