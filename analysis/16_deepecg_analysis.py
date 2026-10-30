"""WP-D — analyses with the second model (concept.md §6.8): the EchoNext analyses on the DeepECG score.

    python3 -I analysis/16_deepecg_analysis.py

Inputs: work/wp_d_scores_{mimic,heedb}.parquet, results/wp_d_g2.json (delta_D, G2-D), and the EchoNext
readout rules from results/wp3_results.json and results/wp_h_results.json.
Output: results/wp_d_results.json (aggregate only).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

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


def with_score(d, scores, keys):
    """Replace the EchoNext logit by the DeepECG score (the analysis functions read logit_lvef)."""
    d = d.drop(columns="logit_lvef").merge(scores[keys + ["score_d"]], on=keys, how="inner")
    return d.rename(columns={"score_d": "logit_lvef"})


def main():
    g2 = json.loads((RESULTS / "wp_d_g2.json").read_text())
    delta = g2["delta_D_points_per_year"]
    rules = {"mimic": json.loads((RESULTS / "wp3_results.json").read_text())["readout"]["rule"],
             "heedb": json.loads((RESULTS / "wp_h_results.json").read_text())["primary"]["readout_rule"]}
    out = {"run_date": str(date.today()), "score": "-(predicted LVEF), points", "delta_D": delta}

    d, _ = AN.load("5y_limbfx")
    d = with_score(d, AN.pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    prim = AN.analysis_set(d, "primary")
    m = {"G2_D_pass": g2["G2_D"]["mimic"]["pass"], "primary": H.block(prim, delta),
         "event_study": AN.event_study(prim), "level": AN.level(prim), "sensitivity": {
             "S1_outside_stays": H.block(prim[~prim.in_stay], delta),
             "S2_qrs_lt120": H.block(prim[prim.qrs_ms < 120], delta),
             "S3_equal_patient_weight": H.block(prim, delta, weights=1.0 / prim.groupby("entity").entity.transform("size")),
             "S4_echo_anchor": H.block(AN.analysis_set(d, "S4"), delta)}}
    m["same_rule_as_echonext"] = m["primary"]["readout_rule"] == rules["mimic"]
    out["mimic"] = m

    h = with_score(H.load(), AN.pd.read_parquet(WORK / "wp_d_scores_heedb.parquet"), ["site", "grp", "pid", "rel"])
    hh = {"G2_D_pass": g2["G2_D"]["heedb"]["pass"], "primary": H.block(h, delta),
          "event_study": AN.event_study(h), "level": AN.level(h), "sensitivity": {
              "S2H_no_conduction_statement": H.block(h[~h.conduction], delta),
              "S3_equal_patient_weight": H.block(h, delta, weights=1.0 / h.groupby("entity").entity.transform("size"))},
          "by_site": {s: H.block(h[h.site == s], delta) for s in sorted(h.site.unique())}}
    hh["same_rule_as_echonext"] = hh["primary"]["readout_rule"] == rules["heedb"]
    out["heedb"] = hh
    for k in ("mimic", "heedb"):
        if not out[k]["G2_D_pass"]:
            out[k]["note"] = "G2-D failed: descriptive only (concept §6.8)"
    (RESULTS / "wp_d_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
