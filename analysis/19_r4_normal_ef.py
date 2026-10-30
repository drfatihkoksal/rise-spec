"""Post hoc R4 (concept.md §6.9): does the score rise while LVEF is still documented normal? MIMIC only.

    python3 -I analysis/19_r4_normal_ef.py

Echo-anchored LVSD group (S4) restricted to ECGs on or before the last TTE with LVEF >45% before the anchor;
patients with >=2 such ECGs spanning >=180 days. Comparators: AF and LIMBFX as in S4, plus REF (R3).
Both models. Output: results/robustness_r4.json (aggregate only).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
ECHO = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/echo/echo/structured-measurement.csv.gz")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R3 = _mod("reference_group", PROJ / "analysis/18_reference_group.py")
AN, D16 = sys.modules["analysis3"], sys.modules["deepecg_analysis"]


def last_normal_echo(anchors):
    """Last TTE with LVEF >45% before the S4 anchor, with the S4 LVEF rules (concept §6.6)."""
    con = duckdb.connect()
    con.register("a", anchors)
    return con.execute(f"""
    WITH m AS (SELECT CAST(subject_id AS BIGINT) subject_id, measurement_id, CAST(measurement_datetime AS TIMESTAMP) t,
                      measurement, TRY_CAST(result AS DOUBLE) v
               FROM read_csv_auto('{ECHO}', all_varchar=true)
               WHERE test_type = 'tte' AND measurement IN ('lvef', 'lvef_upper', 'biplane_lvef', 'lvef_3d')),
    p AS (SELECT subject_id, measurement_id, min(t) t,
                 max(v) FILTER (WHERE measurement = 'lvef') lo, max(v) FILTER (WHERE measurement = 'lvef_upper') up,
                 max(v) FILTER (WHERE measurement = 'biplane_lvef') bp, max(v) FILTER (WHERE measurement = 'lvef_3d') d3
          FROM m GROUP BY 1, 2),
    e AS (SELECT subject_id, t, CASE WHEN lo IS NOT NULL AND up IS NOT NULL THEN CASE WHEN up >= lo AND up - lo <= 10 THEN (lo + up) / 2 END
                                     WHEN lo IS NOT NULL THEN lo WHEN bp IS NOT NULL THEN bp ELSE d3 END ef FROM p)
    SELECT a.subject_id, max(e.t) AS last_normal FROM a JOIN e USING (subject_id)
    WHERE e.ef > 45 AND e.ef <= 85 AND e.t < a.anchor GROUP BY 1
    """).df()


def restrict_lvsd(d):
    """Keep S4 LVSD ECGs on or before the last normal echo; patients with >=2 such ECGs spanning >=180 d."""
    a = pd.read_parquet(WORK / "wp1_anchors_5y_limbfx.parquet")
    a = a[a.cohort == "S4"][["subject_id", "anchor"]]
    lv = d[(d.cohort == "S4")].merge(last_normal_echo(a), on="subject_id")
    lv = lv[lv.ecg_time <= lv.last_normal]
    per = lv.groupby("entity").ecg_time.agg(["size", "min", "max"])
    keep = per[(per["size"] >= 2) & ((per["max"] - per["min"]).dt.days >= 180)].index
    return lv[lv.entity.isin(keep)]


def main():
    delta_e = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    delta_d = json.loads((RESULTS / "wp_d_g2.json").read_text())["delta_D_points_per_year"]
    cols = ["entity", "g", "t", "age", "male_anchor", "logit_lvef"]
    d, _ = AN.load("5y_limbfx")
    dd = D16.with_score(d, pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    ref = pd.read_parquet(WORK / "r3_ref_mimic.parquet")
    ref = ref.assign(g="REF", t=ref.t_years, male_anchor=ref.male.astype(bool), entity="REF:" + ref.subject_id.astype(str))

    out = {"run_date": str(date.today()), "post_hoc": True}
    for name, data, ref_score, delta in (("echonext", d, "logit_lvef", delta_e), ("deepecg", dd, "score_d", delta_d)):
        s4 = AN.analysis_set(data, "S4")
        lv = restrict_lvsd(s4)
        comp = s4[s4.g != "LVSD"]
        df = pd.concat([lv[cols], comp[cols], ref.assign(logit_lvef=ref[ref_score])[cols]], ignore_index=True)
        b = R3.block4(df, delta)
        b["contrasts"] = {k.replace("H-", "LIMBFX-"): v for k, v in b["contrasts"].items()}
        res, used = AN.fit(df, R3.terms4(df))
        extra = {}
        for nm, (a_, b_) in {"L-A": ("LVSD", "AF"), "L-LIMBFX": ("LVSD", "CTRL")}.items():
            c95, c90 = AN.contrast(res, {f"t_{a_}": 1, f"t_{b_}": -1}, 0.05), AN.contrast(res, {f"t_{a_}": 1, f"t_{b_}": -1}, 0.10)
            extra[nm] = {"est": round(c95[0], 4), "ci95": [round(c95[1], 4), round(c95[2], 4)],
                         "ci90": [round(c90[1], 4), round(c90[2], 4)], "class": AN.classify(c95, c90, delta)}
        b["lvsd_contrasts"] = extra | {"L-REF": b["contrasts"]["L-REF"]}
        b["lvsd_ecgs"] = int(len(lv))
        b["lvsd_median_years_ecg_before_anchor"] = round(float(-lv.groupby("entity").t.median().median()), 2)
        out[name] = b
    (RESULTS / "robustness_r4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
