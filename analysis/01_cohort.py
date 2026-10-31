"""WP1 — cohort and anchors (concept.md §6.2, §6.3, §6.6 S4, §7 G1).

Outcome-blind: reads codes, dates, machine statements and raw waveforms only; no model score.
Patient-level outputs stay on this machine (work/); results/wp1_flow_<W>y_<control>.json holds counts only.

    python3 -I analysis/01_cohort.py                                     # -3 y window, hip fracture (default)
    python3 -I analysis/01_cohort.py --window-years 5                    # G1 fallback 1: wider window, once
    python3 -I analysis/01_cohort.py --window-years 5 --control LIMBFX   # G1 fallback 2: limb fractures

Paths and versions: README, Data access.
"""
import argparse
import json
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import wfdb

V = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
HOSP = V / "mimiciv/3.1/hosp"
ECG = V / "mimic-iv-ecg/1.0"
ED = V / "mimic-iv-ed-2.2/ed"
ECHO = V / "echo/echo"
PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"

# Machine statements, lower-cased. Sinus = one statement matches SINUS and none matches NON_SINUS.
SINUS = r"\bsinus (rhythm|bradycardia|tachycardia|arrhythmia)\b"
NON_SINUS = (
    r"fibrillation|flutter|\bpac(ed|ing|emaker)\b|pacemaker"
    r"|junctional (rhythm|bradycardia|tachycardia|escape)|ectopic atrial|idioventricular"
    r"|supraventricular (rhythm|tachycardia)|ventricular tachycardia|undetermined rhythm"
    r"|wandering|\basystole\b|a-?v dissociation|escape rhythm|no further analysis"
)
# Machine-flagged unusable recording or lead, or lead reversal (the model would get the wrong leads).
BAD_QUALITY = r"unsuitable for analysis|lead reversal"
# Waveform: any NaN sample = incomplete lead; peak-to-peak below FLAT_MV in any lead = flat lead.
# 0.05 mV sits in an empty gap of the distribution (3,000 random ECGs: 0.33% near 0 mV, next values >=0.18 mV).
FLAT_MV = 0.05
MISSING = 29999  # machine-measurement codes >= this mean "not measured"
# Non-cardiac control (concept §6.2, §10): (ICD-10 regex, ICD-9 regex). ICD-10 anchors need an
# initial-encounter 7th character (A/B/C); any code of the set counts as history.
CONTROLS = {
    "HIPFX": (r"^S72[012]", r"^820"),
    "LIMBFX": (r"^S(42|52|62|72|82|92)", r"^8[12][0-9]"),
}


def waveform_ok(path):
    try:
        sig, _ = wfdb.rdsamp(str(ECG / path))
    except Exception:
        return path, False
    if sig.shape != (5000, 12) or np.isnan(sig).any():
        return path, False
    return path, bool((np.ptp(sig, axis=0) >= FLAT_MV).all())


def load(con, control):
    con.execute(f"""
    CREATE TABLE adm AS SELECT subject_id, hadm_id, admittime, dischtime FROM read_csv_auto('{HOSP}/admissions.csv.gz');
    CREATE TABLE pat AS SELECT subject_id, gender, anchor_age, anchor_year FROM read_csv_auto('{HOSP}/patients.csv.gz');
    CREATE TABLE edst AS SELECT subject_id, intime, outtime FROM read_csv_auto('{ED}/edstays.csv.gz');
    CREATE TABLE dx AS
    SELECT CAST(subject_id AS BIGINT) subject_id, CAST(hadm_id AS BIGINT) hadm_id, icd_code AS c, CAST(icd_version AS INT) v
    FROM read_csv_auto('{HOSP}/diagnoses_icd.csv.gz', all_varchar=true);
    CREATE TABLE ecg AS
    SELECT subject_id, study_id, path, CAST(ecg_time AS TIMESTAMP) t FROM read_csv_auto('{ECG}/record_list.csv');
    """)
    reports = ", ".join(f"report_{i}" for i in range(18))
    con.execute(f"CREATE TABLE mm AS SELECT * FROM read_csv_auto('{ECG}/machine_measurements.csv', all_varchar=true)")
    con.execute(f"""
    CREATE TABLE stm AS
    WITH l AS (SELECT study_id, lower(trim(unnest([{reports}]))) s FROM mm)
    SELECT CAST(study_id AS BIGINT) study_id,
           bool_or(regexp_matches(s, '{SINUS}')) AND NOT bool_or(regexp_matches(s, '{NON_SINUS}')) AS sinus,
           bool_or(regexp_matches(s, '{BAD_QUALITY}')) AS bad_quality
    FROM l WHERE s IS NOT NULL AND s <> '' GROUP BY 1;
    CREATE TABLE qrs AS
    WITH m AS (SELECT CAST(study_id AS BIGINT) study_id, TRY_CAST(qrs_onset AS INT) a, TRY_CAST(qrs_end AS INT) b FROM mm)
    SELECT study_id, CASE WHEN a < {MISSING} AND b < {MISSING} AND b - a BETWEEN 20 AND 300 THEN b - a END AS qrs_ms FROM m;
    """)
    # Code sets per admission (MIMIC stores codes without dots).
    c10, c9 = CONTROLS[control]
    con.execute(f"""
    CREATE TABLE adm_sets AS
    SELECT subject_id, hadm_id, a.admittime, a.dischtime,
      bool_or((v = 10 AND regexp_matches(c, '^I50[24]')) OR (v = 9 AND regexp_matches(c, '^428[24]'))) AS lvsd,
      bool_or((v = 10 AND c LIKE 'I48%') OR (v = 9 AND c IN ('42731', '42732'))) AS af,
      bool_or((v = 10 AND regexp_matches(c, '{c10}')) OR (v = 9 AND regexp_matches(c, '{c9}'))) AS ctrl,
      bool_or((v = 10 AND regexp_matches(c, '{c10}') AND length(c) = 7 AND right(c, 1) IN ('A', 'B', 'C'))
              OR (v = 9 AND regexp_matches(c, '{c9}'))) AS ctrl_initial,
      bool_or((v = 10 AND regexp_matches(c, '^I(50|42)')) OR (v = 9 AND regexp_matches(c, '^42[58]'))) AS hfcm
    FROM dx JOIN adm a USING (subject_id, hadm_id)
    GROUP BY ALL
    """)


def primary_anchors(con, control):
    """First admission with a code from any of the three sets, with one flag per §6.2 rule."""
    con.execute(f"""
    CREATE TABLE anc AS
    SELECT subject_id, hadm_id, admittime AS anchor, dischtime, ctrl_initial,
           lvsd::INT + af::INT + ctrl::INT AS n_sets,
           CASE WHEN lvsd THEN 'LVSD' WHEN af THEN 'AF' ELSE '{control}' END AS grp
    FROM (SELECT *, row_number() OVER (PARTITION BY subject_id ORDER BY admittime, hadm_id) rn
          FROM adm_sets WHERE lvsd OR af OR ctrl)
    WHERE rn = 1;
    CREATE TABLE anc_f AS
    SELECT 'primary' AS cohort, x.subject_id, x.hadm_id, x.anchor, x.grp,
      x.n_sets > 1 AS ex_multi_set,
      x.grp = '{control}' AND NOT x.ctrl_initial AS ex_control_not_initial,
      x.grp = 'LVSD' AND EXISTS (SELECT 1 FROM adm_sets s WHERE s.subject_id = x.subject_id
                                 AND s.admittime < x.anchor AND s.hfcm) AS ex_prior_hf_cm,
      x.grp IN ('AF', '{control}') AND EXISTS (SELECT 1 FROM adm_sets s WHERE s.subject_id = x.subject_id AND s.lvsd
                                           AND s.admittime > x.anchor AND s.admittime <= x.anchor + INTERVAL 365 DAY) AS ex_lvsd_within_1y,
      p.anchor_age + year(x.anchor) - p.anchor_year < 18 AS ex_age,
      NOT EXISTS (SELECT 1 FROM ecg e WHERE e.subject_id = x.subject_id
                  AND e.t BETWEEN x.anchor - INTERVAL 30 DAY AND x.dischtime) AS ex_no_ecg_at_anchor,
      p.anchor_age + year(x.anchor) - p.anchor_year AS age, p.gender = 'M' AS male
    FROM anc x JOIN pat p USING (subject_id)
    """)
    return ["ex_multi_set", "ex_control_not_initial", "ex_prior_hf_cm", "ex_lvsd_within_1y", "ex_age", "ex_no_ecg_at_anchor"]


def echo_anchors(con):
    """S4: first TTE with LVEF <=45% that follows a TTE with LVEF >45% (concept §6.6)."""
    con.execute(f"""
    CREATE TABLE tte AS
    WITH m AS (SELECT CAST(subject_id AS BIGINT) subject_id, measurement_id, CAST(measurement_datetime AS TIMESTAMP) t,
                      measurement, TRY_CAST(result AS DOUBLE) v
               FROM read_csv_auto('{ECHO}/structured-measurement.csv.gz', all_varchar=true)
               WHERE test_type = 'tte' AND measurement IN ('lvef', 'lvef_upper', 'biplane_lvef', 'lvef_3d')),
    p AS (SELECT subject_id, measurement_id, min(t) t,
                 max(v) FILTER (WHERE measurement = 'lvef') lo, max(v) FILTER (WHERE measurement = 'lvef_upper') up,
                 max(v) FILTER (WHERE measurement = 'biplane_lvef') bp, max(v) FILTER (WHERE measurement = 'lvef_3d') d3
          FROM m GROUP BY 1, 2),
    e AS (SELECT subject_id, measurement_id, t,
                 CASE WHEN lo IS NOT NULL AND up IS NOT NULL
                        THEN CASE WHEN up >= lo AND up - lo <= 10 THEN (lo + up) / 2 END
                      WHEN lo IS NOT NULL THEN lo WHEN bp IS NOT NULL THEN bp ELSE d3 END AS ef
          FROM p)
    SELECT * FROM e WHERE ef BETWEEN 5 AND 85;
    CREATE TABLE s4 AS
    WITH f AS (SELECT subject_id, min(t) AS anchor FROM tte WHERE ef <= 45 GROUP BY 1)
    SELECT f.* FROM f WHERE EXISTS (SELECT 1 FROM tte x WHERE x.subject_id = f.subject_id AND x.ef > 45 AND x.t < f.anchor);
    CREATE TABLE s4_f AS
    SELECT 'S4' AS cohort, s.subject_id, NULL::BIGINT AS hadm_id, s.anchor, 'LVSD_ECHO' AS grp,
      EXISTS (SELECT 1 FROM adm_sets a WHERE a.subject_id = s.subject_id AND a.dischtime < s.anchor
              AND (a.hfcm OR a.lvsd OR a.af OR a.ctrl)) AS ex_prior_codes,
      EXISTS (SELECT 1 FROM adm_sets a WHERE a.subject_id = s.subject_id
              AND s.anchor BETWEEN a.admittime AND a.dischtime AND (a.af OR a.ctrl)) AS ex_concurrent_af_control,
      p.anchor_age + year(s.anchor) - p.anchor_year < 18 AS ex_age,
      NOT EXISTS (SELECT 1 FROM ecg e WHERE e.subject_id = s.subject_id
                  AND e.t BETWEEN s.anchor - INTERVAL 30 DAY AND s.anchor + INTERVAL 30 DAY) AS ex_no_ecg_at_anchor,
      p.anchor_age + year(s.anchor) - p.anchor_year AS age, p.gender = 'M' AS male
    FROM s4 s JOIN pat p USING (subject_id)
    """)
    tte_counts = con.execute("SELECT count(*), count(DISTINCT subject_id) FROM tte").fetchone()
    return ["ex_prior_codes", "ex_concurrent_af_control", "ex_age", "ex_no_ecg_at_anchor"], tte_counts


def flow(df, steps, by="grp"):
    """Sequential exclusion counts per group: remaining after each step."""
    out, keep = {}, pd.Series(True, index=df.index)
    out["start"] = df.groupby(by).size().to_dict()
    for s in steps:
        out[s] = df[keep & df[s]].groupby(by).size().to_dict()
        keep &= ~df[s]
        out[f"after_{s}"] = df[keep].groupby(by).size().to_dict()
    return out, keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-years", type=int, default=3)
    ap.add_argument("--control", choices=sorted(CONTROLS), default="HIPFX")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    W, workers, control = args.window_years, args.workers, args.control
    tag = f"{W}y_{control.lower()}"
    WORK.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    con = duckdb.connect()
    load(con, control)
    steps_p = primary_anchors(con, control)
    steps_s4, tte_counts = echo_anchors(con)

    prim = con.execute("SELECT * FROM anc_f").df()
    # Multi-set patients have no single group: count them overall, then drop before the per-group flow.
    n_first_event, n_multi = len(prim), int(prim.ex_multi_set.sum())
    flow_p, keep_p = flow(prim[~prim.ex_multi_set], steps_p[1:])
    s4 = con.execute("SELECT * FROM s4_f").df()
    flow_s, keep_s = flow(s4, steps_s4)

    cand = pd.concat([prim[~prim.ex_multi_set][keep_p], s4[keep_s]])[["cohort", "grp", "subject_id", "hadm_id", "anchor", "age", "male"]]
    con.register("cand_df", cand)
    con.execute("CREATE TABLE cand AS SELECT * FROM cand_df")

    # ECGs in the window, with the statement-based rules.
    con.execute(f"""
    CREATE TABLE win AS
    SELECT a.cohort, a.grp, a.subject_id, e.study_id, e.path, e.t AS ecg_time,
           date_diff('second', a.anchor, e.t) / (365.25 * 86400) AS t_years,
           coalesce(s.sinus, false) AS sinus, coalesce(s.bad_quality, false) AS bad_quality, q.qrs_ms
    FROM cand a JOIN ecg e USING (subject_id)
    LEFT JOIN stm s USING (study_id) LEFT JOIN qrs q USING (study_id)
    WHERE e.t >= a.anchor - INTERVAL {W} YEAR AND e.t <= a.anchor - INTERVAL 30 DAY
    """)

    # Waveform check only for ECGs that pass the statement rules; results cached across runs.
    cache_f = WORK / "wp1_waveform_qc.parquet"
    cache = pd.read_parquet(cache_f) if cache_f.exists() else pd.DataFrame(columns=["path", "waveform_ok"])
    todo = sorted(set(con.execute("SELECT path FROM win WHERE sinus AND NOT bad_quality").df().path) - set(cache.path))
    if todo:
        with Pool(workers) as pool:
            new = pd.DataFrame(pool.map(waveform_ok, todo, chunksize=200), columns=["path", "waveform_ok"])
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_parquet(cache_f, index=False)
    con.register("qc_df", cache)

    con.execute("""
    CREATE TABLE win_q AS
    SELECT w.*, coalesce(q.waveform_ok, false) AS waveform_ok, w.sinus AND NOT w.bad_quality AND coalesce(q.waveform_ok, false) AS eligible
    FROM win w LEFT JOIN qc_df q USING (path);
    CREATE TABLE entry AS
    SELECT cohort, grp, subject_id,
           count(*) AS n_window,
           count(*) FILTER (WHERE sinus AND NOT bad_quality) AS n_statement_ok,
           count(*) FILTER (WHERE eligible) AS n_eligible,
           date_diff('day', min(ecg_time) FILTER (WHERE eligible), max(ecg_time) FILTER (WHERE eligible)) AS span_days
    FROM win_q GROUP BY ALL;
    """)
    entry = con.execute("""
    SELECT c.cohort, c.grp, c.subject_id, coalesce(e.n_window, 0) n_window, coalesce(e.n_eligible, 0) n_eligible,
           coalesce(e.span_days, 0) span_days
    FROM cand c LEFT JOIN entry e USING (cohort, grp, subject_id)
    """).df()
    entry["ex_no_window_ecg"] = entry.n_window == 0
    entry["ex_lt2_eligible_or_span_lt180d"] = ~entry.ex_no_window_ecg & ~((entry.n_eligible >= 2) & (entry.span_days >= 180))
    final_mask = ~entry.ex_no_window_ecg & ~entry.ex_lt2_eligible_or_span_lt180d
    final = entry[final_mask][["cohort", "grp", "subject_id"]]

    # Patient-level outputs (local only).
    anchors = cand.merge(final, on=["cohort", "grp", "subject_id"])
    con.register("final_df", final)
    ecgs = con.execute("""
    SELECT w.cohort, w.grp, w.subject_id, w.study_id, w.path, w.ecg_time, w.t_years, w.qrs_ms,
           EXISTS (SELECT 1 FROM adm a WHERE a.subject_id = w.subject_id AND w.ecg_time BETWEEN a.admittime AND a.dischtime)
        OR EXISTS (SELECT 1 FROM edst d WHERE d.subject_id = w.subject_id AND w.ecg_time BETWEEN d.intime AND d.outtime) AS in_stay
    FROM win_q w JOIN final_df f ON w.cohort = f.cohort AND w.grp = f.grp AND w.subject_id = f.subject_id
    WHERE w.eligible
    """).df()
    anchors.to_parquet(WORK / f"wp1_anchors_{tag}.parquet", index=False)
    ecgs.to_parquet(WORK / f"wp1_ecgs_{tag}.parquet", index=False)

    # Aggregate outputs.
    def ecg_flow(cohort):
        e = entry[entry.cohort == cohort]
        return {
            "candidates": e.groupby("grp").size().to_dict(),
            "ex_no_window_ecg": e[e.ex_no_window_ecg].groupby("grp").size().to_dict(),
            "ex_lt2_eligible_or_span_lt180d": e[e.ex_lt2_eligible_or_span_lt180d].groupby("grp").size().to_dict(),
            "final_patients": final[final.cohort == cohort].groupby("grp").size().to_dict(),
        }

    def describe(cohort):
        e = ecgs[ecgs.cohort == cohort]
        per = e.groupby(["grp", "subject_id"]).agg(n=("study_id", "size"), span=("t_years", lambda t: t.max() - t.min()))
        d = per.groupby("grp").agg(ecgs=("n", "sum"), median_ecgs_per_patient=("n", "median"), median_span_years=("span", "median"))
        d["share_in_stay"] = e.groupby("grp").in_stay.mean()
        d["share_qrs_lt120"] = e.groupby("grp").qrs_ms.apply(lambda q: (q < 120).mean())
        return d.round(3).to_dict(orient="index")

    win_counts = con.execute("""
    SELECT count(*) n_window_ecgs, count(*) FILTER (WHERE sinus) n_sinus, count(*) FILTER (WHERE sinus AND NOT bad_quality) n_statement_ok,
           count(*) FILTER (WHERE eligible) n_eligible FROM win_q""").df().iloc[0].astype(int).to_dict()
    g1_counts = {g: int(final[(final.cohort == "primary") & (final.grp == g)].shape[0]) for g in ["LVSD", "AF", control]}
    res = {
        "run_date": str(date.today()), "window_years": W, "control": control, "control_codes": CONTROLS[control], "window_end_days_before_anchor": 30,
        "rules": {"sinus": SINUS, "non_sinus": NON_SINUS, "bad_quality": BAD_QUALITY, "flat_mV": FLAT_MV,
                  "waveform": "5000x12 samples, no NaN, peak-to-peak >= flat_mV in every lead"},
        "primary": {"first_event_patients": n_first_event, "ex_multi_set": n_multi,
                    "anchor_flow": flow_p, "ecg_flow": ecg_flow("primary"), "describe": describe("primary")},
        "S4": {"tte_studies_with_valid_lvef": int(tte_counts[0]), "tte_patients_with_valid_lvef": int(tte_counts[1]),
               "anchor_flow": flow_s, "ecg_flow": ecg_flow("S4"), "describe": describe("S4")},
        "window_ecg_counts_all_candidates": win_counts,
        "G1": {"threshold": 500, "counts": g1_counts, "pass": all(n >= 500 for n in g1_counts.values())},
    }
    out = RESULTS / f"wp1_flow_{tag}.json"
    out.write_text(json.dumps(res, indent=2, default=int))
    print(json.dumps(res, indent=2, default=int))


if __name__ == "__main__":
    main()
