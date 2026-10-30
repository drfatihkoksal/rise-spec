"""Post hoc robustness R1 (lag) and R2 (symmetric exclusions), concept.md §6.9. Both models, both data sources.

    python3 -I analysis/17_robustness.py

Uses the existing scores; no new inference. Output: results/robustness_r1_r2.json (aggregate only).
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
MIMIC_HOSP = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/mimiciv/3.1/hosp")
HEEDB = Path.home() / "datasets/heedb"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


AN = _mod("analysis3", PROJ / "analysis/03_analysis.py")
H = _mod("heedb_analysis", PROJ / "analysis/13_heedb_analysis.py")
D = _mod("deepecg_analysis", PROJ / "analysis/16_deepecg_analysis.py")
COH = _mod("heedb_cohort", PROJ / "analysis/11_heedb_cohort.py")
HFCM10, HFCM9 = "^I(50|42)", "^42[58]"


def prior_hfcm_mimic():
    a = pd.read_parquet(WORK / "wp1_anchors_5y_limbfx.parquet")
    a = a[a.cohort == "primary"][["grp", "subject_id", "anchor"]]
    con = duckdb.connect()
    con.register("a", a)
    return con.execute(f"""
    WITH dx AS (SELECT CAST(subject_id AS BIGINT) sid, CAST(hadm_id AS BIGINT) hid FROM read_csv_auto('{MIMIC_HOSP}/diagnoses_icd.csv.gz', all_varchar=true)
                WHERE (icd_version = '10' AND regexp_matches(icd_code, '{HFCM10}')) OR (icd_version = '9' AND regexp_matches(icd_code, '{HFCM9}'))),
    adm AS (SELECT subject_id sid, hadm_id hid, admittime FROM read_csv_auto('{MIMIC_HOSP}/admissions.csv.gz'))
    SELECT a.grp, a.subject_id, EXISTS (SELECT 1 FROM dx JOIN adm USING (sid, hid) WHERE dx.sid = a.subject_id AND adm.admittime < a.anchor) AS prior_hfcm
    FROM a""").df()


def prior_hfcm_heedb():
    a = pd.read_parquet(WORK / "wp_h_anchors.parquet")[["site", "grp", "pid", "anchor_d"]]
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{HEEDB}/duckdb_tmp'; SET memory_limit = '40GB'; SET threads = 24")
    parts = []
    for site in COH.SITES:
        E = HEEDB / "ECG" / site
        for v, col in (("9", "DIAGNOSIS_ICD9_CD"), ("10", "DIAGNOSIS_ICD10_CD")):
            rx = HFCM9 if v == "9" else HFCM10
            parts.append(f"""SELECT '{site}' site, {COH.PID} pid, CAST(TRY_CAST(RECORDED_DT AS TIMESTAMP) AS DATE) d
                             FROM read_csv_auto('{E}/ICD_codes/icd{v}_codes.csv', all_varchar=true)
                             WHERE regexp_matches(replace({col}, '.', ''), '{rx}')""")
    con.execute(f"CREATE TABLE hf AS {' UNION ALL '.join(parts)}")
    con.register("a", a)
    return con.execute("""
    SELECT a.site, a.grp, a.pid, EXISTS (SELECT 1 FROM hf WHERE hf.site = a.site AND hf.pid = a.pid AND hf.d < CAST(a.anchor_d AS DATE)) AS prior_hfcm
    FROM a""").df()


def run(df, delta, prior, keys):
    r1 = H.block(df[df.t <= -1], delta)
    excl = prior[prior.prior_hfcm & prior.grp.isin(["AF", "LIMBFX"])][keys]
    df2 = df.merge(excl.assign(_x=1), on=keys, how="left")
    r2 = H.block(df2[df2._x.isna()].drop(columns="_x"), delta)
    keep = lambda b: {"readout_rule": b["readout_rule"], "patients": b["patients"], "ecgs": b["ecgs"],
                      "beta": b["beta"], "contrasts": b["contrasts"]}
    return {"R1_lag_12m": keep(r1), "R2_symmetric_exclusion": keep(r2) | {"excluded_patients": int(len(excl))}}


def main():
    delta_e = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    delta_d = json.loads((RESULTS / "wp_d_g2.json").read_text())["delta_D_points_per_year"]
    pm, ph = prior_hfcm_mimic(), prior_hfcm_heedb()

    dm, _ = AN.load("5y_limbfx")
    dm = AN.analysis_set(dm, "primary")
    dmd = D.with_score(dm, pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    dh = H.load()
    dhd = D.with_score(dh, pd.read_parquet(WORK / "wp_d_scores_heedb.parquet"), ["site", "grp", "pid", "rel"])

    out = {"run_date": str(date.today()), "post_hoc": True,
           "share_prior_hfcm": {"mimic": pm.groupby("grp").prior_hfcm.mean().round(3).to_dict(),
                                "heedb": ph.groupby("grp").prior_hfcm.mean().round(3).to_dict()},
           "echonext": {"mimic": run(dm, delta_e, pm, ["grp", "subject_id"]), "heedb": run(dh, delta_e, ph, ["site", "grp", "pid"])},
           "deepecg": {"mimic": run(dmd, delta_d, pm, ["grp", "subject_id"]), "heedb": run(dhd, delta_d, ph, ["site", "grp", "pid"])}}
    (RESULTS / "robustness_r1_r2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
