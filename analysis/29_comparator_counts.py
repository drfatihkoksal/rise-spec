"""Patient counts for candidate comparator groups (outcome-blind: codes, dates, statements, raw waveforms; no score).

Candidate groups, defined like the existing ones (concept.md §6.2, §6.7):
  DHF   first coded diastolic heart failure: ICD-10 I50.3x; ICD-9 428.3x
  AS    aortic stenosis: ICD-10 I35.0, I35.2; ICD-9 424.1 (424.1 also covers aortic regurgitation)
  AS10  aortic stenosis by ICD-10 only: I35.0, I35.2

Draft rules used for the counts (to be fixed in the protocol before any scoring):
  - anchor = first admission (MIMIC-IV) or first coded date (HEEDB) with a code from any of the sets
    LVSD, AF, limb fracture and DHF (for DHF), plus the aortic stenosis set (for AS and AS10); a patient enters
    a new group only when that first event carries the group's set alone;
  - DHF: no earlier heart failure or cardiomyopathy code (as for LVSD);
  - all new groups: no LVSD code within one year after the anchor (as for AF and fracture);
  - age, coded history (HEEDB), ECG at the anchor, window and ECG eligibility rules exactly as in
    01_cohort.py and 11_heedb_cohort.py (window -5 y to -30 d).
The existing groups are not changed. A patient already in an existing analysis cohort stays there, and an AS
patient who is also a DHF patient stays in DHF; the gate uses the counts after these patients are removed.

    python3 -I analysis/29_comparator_counts.py mimic
    python3 -I analysis/29_comparator_counts.py heedb

Output (aggregate only): results/comparator_counts_<db>.json. Waveform checks are cached in work/ (local only).
mimic() and heedb() also return the patient and ECG lists of the new groups for the analysis script (not written here).
"""
import importlib.util
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import duckdb
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
W, THRESHOLD = 5, 500
# group: (its flag, the sets whose first occurrence is the first event)
GROUPS = {
    "DHF": ("dhf", ["lvsd", "af", "ctrl", "dhf"]),
    "AS": ("as_", ["lvsd", "af", "ctrl", "dhf", "as_"]),
    "AS10": ("as10", ["lvsd", "af", "ctrl", "dhf", "as10"]),
}
# Flags per admission or coded day; `c` is the code without dots, `v` the ICD version.
EXTRA = """
      bool_or((v = 10 AND c LIKE 'I503%') OR (v = 9 AND c LIKE '4283%')) AS dhf,
      bool_or((v = 10 AND c IN ('I350', 'I352')) OR (v = 9 AND c = '4241')) AS as_,
      bool_or(v = 10 AND c IN ('I350', 'I352')) AS as10,
      bool_or((v = 10 AND c LIKE 'I50%') OR (v = 9 AND c LIKE '428%')) AS hf
"""
FIRST_HF = """
    SELECT CASE WHEN lvsd AND dhf THEN 'systolic_and_diastolic_codes' WHEN lvsd THEN 'systolic_or_combined'
                WHEN dhf THEN 'diastolic' ELSE 'unspecified_or_other' END AS k, count(*) n
    FROM (SELECT *, row_number() OVER (PARTITION BY {pid} ORDER BY {t}) rn FROM {tab} WHERE hf) WHERE rn = 1 GROUP BY 1
"""


def module(name):
    spec = importlib.util.spec_from_file_location(name, PROJ / "analysis" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def flow(df, steps):
    out, keep = {"start": df.groupby("grp").size().to_dict()}, pd.Series(True, index=df.index)
    for s in steps:
        out[s] = df[keep & df[s]].groupby("grp").size().to_dict()
        keep &= ~df[s]
        out[f"after_{s}"] = df[keep].groupby("grp").size().to_dict()
    return out, keep


def first_event(tab, pid, order, g):
    """SQL for the first event over the group's sets, kept when it carries the group's flag."""
    flag, sets = GROUPS[g]
    return f"""(SELECT *, '{g}' AS grp, {' + '.join(s + '::INT' for s in sets)} AS n_sets
                FROM (SELECT *, row_number() OVER (PARTITION BY {pid} ORDER BY {order}) rn FROM {tab} WHERE {' OR '.join(sets)})
                WHERE rn = 1 AND {flag})"""


def keep_one_group(final, keys, existing):
    """Mark patients who already belong to an existing cohort, or (AS, AS10) to the DHF group."""
    k = final[keys].apply(tuple, axis=1)
    dhf = set(k[(final.grp == "DHF") & ~k.isin(existing)])
    final["in_other_group"] = k.isin(existing) | ((final.grp != "DHF") & k.isin(dhf))
    return final


def gate(counts):
    return {"threshold": THRESHOLD, "counts": counts, "pass": {g: n >= THRESHOLD for g, n in counts.items()}}


def by_grp(df):
    return {g: int(n) for g, n in df.groupby("grp").size().items()}


def mimic():
    c1 = module("01_cohort")
    con = duckdb.connect()
    c1.load(con, "LIMBFX")
    con.execute(f"""
    CREATE TABLE extra AS SELECT subject_id, hadm_id, {EXTRA} FROM dx GROUP BY ALL;
    CREATE TABLE a5 AS SELECT s.*, x.dhf, x.as_, x.as10, x.hf FROM adm_sets s JOIN extra x USING (subject_id, hadm_id);
    """)
    first_hf = dict(sorted(con.execute(FIRST_HF.format(pid="subject_id", t="admittime, hadm_id", tab="a5")).fetchall()))
    a = pd.concat([con.execute(f"""
        SELECT x.grp, x.subject_id, x.admittime AS anchor,
          x.n_sets > 1 AS ex_multi_set,
          x.grp = 'DHF' AND EXISTS (SELECT 1 FROM adm_sets s WHERE s.subject_id = x.subject_id
                                    AND s.admittime < x.admittime AND s.hfcm) AS ex_prior_hf_cm,
          EXISTS (SELECT 1 FROM adm_sets s WHERE s.subject_id = x.subject_id AND s.lvsd
                  AND s.admittime > x.admittime AND s.admittime <= x.admittime + INTERVAL 365 DAY) AS ex_lvsd_within_1y,
          p.anchor_age + year(x.admittime) - p.anchor_year < 18 AS ex_age,
          NOT EXISTS (SELECT 1 FROM ecg e WHERE e.subject_id = x.subject_id
                      AND e.t BETWEEN x.admittime - INTERVAL 30 DAY AND x.dischtime) AS ex_no_ecg_at_anchor
        FROM {first_event('a5', 'subject_id', 'admittime, hadm_id', g)} x JOIN pat p USING (subject_id)
        """).df() for g in GROUPS], ignore_index=True)
    with_set, multi = by_grp(a), by_grp(a[a.ex_multi_set])
    a = a[~a.ex_multi_set].reset_index(drop=True)
    fl, keep = flow(a, ["ex_prior_hf_cm", "ex_lvsd_within_1y", "ex_age", "ex_no_ecg_at_anchor"])
    cand = a[keep][["grp", "subject_id", "anchor"]]
    con.register("cand_df", cand)
    win = con.execute(f"""
    SELECT a.grp, a.subject_id, e.study_id, e.path, e.t AS ecg_time,
           coalesce(s.sinus, false) AND NOT coalesce(s.bad_quality, false) AS statement_ok
    FROM cand_df a JOIN ecg e USING (subject_id) LEFT JOIN stm s USING (study_id)
    WHERE e.t >= a.anchor - INTERVAL {W} YEAR AND e.t <= a.anchor - INTERVAL 30 DAY
    """).df()

    cache_f = WORK / "comparator_waveform_qc_mimic.parquet"
    mine = pd.read_parquet(cache_f) if cache_f.exists() else pd.DataFrame(columns=["path", "waveform_ok"])
    cache = pd.concat([pd.read_parquet(WORK / "wp1_waveform_qc.parquet"), mine], ignore_index=True)
    todo = sorted(set(win[win.statement_ok].path) - set(cache.path))
    if todo:
        with Pool(16) as pool:
            new = pd.DataFrame(pool.map(c1.waveform_ok, todo, chunksize=200), columns=["path", "waveform_ok"])
        pd.concat([mine, new], ignore_index=True).to_parquet(cache_f, index=False)
        cache = pd.concat([cache, new], ignore_index=True)
    win = win.merge(cache.drop_duplicates("path"), on="path", how="left")
    win["eligible"] = win.statement_ok & win.waveform_ok.eq(True)

    def entered(col):
        e = win[win[col]].groupby(["grp", "subject_id"]).agg(n=("path", "size"), lo=("ecg_time", "min"), hi=("ecg_time", "max")).reset_index()
        # Calendar-day span, as date_diff('day', ...) in 01_cohort.py.
        return e[(e.n >= 2) & ((e.hi.dt.normalize() - e.lo.dt.normalize()).dt.days >= 180)][["grp", "subject_id"]]

    old = pd.read_parquet(WORK / "wp1_anchors_5y_limbfx.parquet")
    final = keep_one_group(entered("eligible").merge(cand, on=["grp", "subject_id"]), ["subject_id"],
                           set((s,) for s in old.subject_id))
    own = final[~final.in_other_group]
    ecgs = win[win.eligible].merge(own[["grp", "subject_id"]], on=["grp", "subject_id"])

    # Echocardiographic LVEF nearest to the anchor (counts for LVEF-restricted sensitivity analyses).
    c1.echo_anchors(con)
    lv = old.query("cohort == 'primary' and grp == 'LVSD'")[["grp", "subject_id", "anchor"]]
    con.register("ef_df", pd.concat([own[["grp", "subject_id", "anchor"]], lv], ignore_index=True))
    ef_pat = con.execute("""
    WITH n AS (SELECT p.grp, p.subject_id, t.ef, row_number() OVER (PARTITION BY p.grp, p.subject_id
                      ORDER BY abs(date_diff('second', p.anchor, t.t))) rn
               FROM ef_df p JOIN tte t USING (subject_id) WHERE t.t BETWEEN p.anchor - INTERVAL 90 DAY AND p.anchor + INTERVAL 90 DAY)
    SELECT grp, subject_id, ef FROM n WHERE rn = 1""").df()
    ef = {g: {"with_tte": int(len(v)), "ef_ge50": int((v >= 50).sum()), "ef_lt50": int((v < 50).sum()),
              "ef_le45": int((v <= 45).sum()), "median_ef": round(float(v.median()), 1)} for g, v in ef_pat.groupby("grp").ef}

    return {
        "run_date": str(date.today()), "database": "MIMIC-IV", "window_years": W, "scores_read": False,
        "first_heart_failure_code_admission_by_type": first_hf,
        "first_event_with_set": with_set, "ex_multi_set": multi, "anchor_flow": fl,
        "ecg_flow": {"candidates": by_grp(cand),
                     "with_window_ecg": {g: int(win[win.grp == g].subject_id.nunique()) for g in GROUPS},
                     "final_before_waveform_check": by_grp(entered("statement_ok")),
                     "final_patients": by_grp(final)},
        "final_patients_not_in_other_group": by_grp(own), "final_ecgs_not_in_other_group": by_grp(ecgs),
        "lvef_nearest_tte_within_90d_of_anchor": ef, "existing_LVSD_cohort_patients": len(lv),
        "gate": gate(by_grp(own)),
    }, {"patients": own[["grp", "subject_id", "anchor"]], "ecgs": ecgs[["grp", "subject_id", "study_id", "path", "ecg_time"]],
        "lvef": ef_pat}


def heedb():
    c11 = module("11_heedb_cohort")
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{c11.HEEDB}/duckdb_tmp'; SET memory_limit = '40GB'; SET threads = 24")
    flows, cands, wins, first_hf = {}, [], [], {}
    for site, (name, sexcol) in c11.SITES.items():
        c11.load_site(con, site, sexcol)
        con.execute(f"""
        CREATE OR REPLACE TABLE d5 AS
        SELECT pid, d,
          bool_or((v = 10 AND regexp_matches(c, '^I50[24]')) OR (v = 9 AND regexp_matches(c, '^428[24]'))) AS lvsd,
          bool_or((v = 10 AND c LIKE 'I48%') OR (v = 9 AND c IN ('42731', '42732'))) AS af,
          bool_or((v = 10 AND regexp_matches(c, '^S(42|52|62|72|82|92)')) OR (v = 9 AND regexp_matches(c, '^8[12][0-9]'))) AS ctrl,
          bool_or((v = 10 AND regexp_matches(c, '^I(50|42)')) OR (v = 9 AND regexp_matches(c, '^42[58]'))) AS hfcm,
          {EXTRA}
        FROM dx_all
        WHERE d IS NOT NULL AND (regexp_matches(c, '^(I50|I42|I48|I350|I352|S42|S52|S62|S72|S82|S92)')
                                 OR regexp_matches(c, '^(428|425|4273|4241|81|82)'))
        GROUP BY ALL;
        CREATE OR REPLACE TABLE pat AS SELECT pid, any_value(dob) dob, any_value(male) male, bool_or(sex_known) sex_known FROM meta GROUP BY 1;
        """)
        first_hf[name] = dict(sorted(con.execute(FIRST_HF.format(pid="pid", t="d", tab="d5")).fetchall()))
        a = pd.concat([con.execute(f"""
            SELECT '{site}' site, x.grp, x.pid, x.d AS anchor_d, p.dob, p.male, date_diff('day', p.dob, x.d) / 365.25 AS age,
              x.n_sets > 1 AS ex_multi_set,
              x.grp = 'DHF' AND EXISTS (SELECT 1 FROM d5 s WHERE s.pid = x.pid AND s.d < x.d AND s.hfcm) AS ex_prior_hf_cm,
              EXISTS (SELECT 1 FROM d5 s WHERE s.pid = x.pid AND s.lvsd AND s.d > x.d AND s.d <= x.d + INTERVAL 365 DAY) AS ex_lvsd_within_1y,
              coalesce(fc.first_d > x.d - INTERVAL 365 DAY, true) AS ex_lookback_lt_1y,
              p.dob IS NULL OR NOT p.sex_known OR date_diff('day', p.dob, x.d) / 365.25 < 18 AS ex_age_or_sex_missing,
              NOT EXISTS (SELECT 1 FROM meta e WHERE e.pid = x.pid
                          AND CAST(e.t AS DATE) BETWEEN x.d - INTERVAL 30 DAY AND x.d + INTERVAL 30 DAY) AS ex_no_ecg_at_anchor
            FROM {first_event('d5', 'pid', 'd', g)} x LEFT JOIN pat p USING (pid) LEFT JOIN first_code fc USING (pid)
            """).df() for g in GROUPS], ignore_index=True)
        with_set, multi = by_grp(a), by_grp(a[a.ex_multi_set])
        a = a[~a.ex_multi_set].reset_index(drop=True)
        fl, keep = flow(a, ["ex_prior_hf_cm", "ex_lvsd_within_1y", "ex_lookback_lt_1y", "ex_age_or_sex_missing", "ex_no_ecg_at_anchor"])
        cand = a[keep][["site", "grp", "pid", "anchor_d", "dob", "male", "age"]]
        con.register("cand_df", cand)
        win = con.execute(f"""
        SELECT c.site, c.grp, c.pid, m.rel, m.t AS ecg_time,
               coalesce(s.sinus, false) AND NOT coalesce(s.bad_quality, false) AS statement_ok
        FROM cand_df c JOIN meta m USING (pid) LEFT JOIN stm s USING (rel)
        WHERE m.t >= CAST(c.anchor_d AS TIMESTAMP) - INTERVAL {W} YEAR AND m.t <= CAST(c.anchor_d AS TIMESTAMP) - INTERVAL 30 DAY
        """).df()
        flows[site] = {"name": name, "first_event_with_set": with_set, "ex_multi_set": multi, "anchor_flow": fl,
                       "candidates": by_grp(cand)}
        cands.append(cand)
        wins.append(win)
        print(site, "loaded", len(cand), "candidates", len(win), "window ECGs", flush=True)

    cand, win = pd.concat(cands, ignore_index=True), pd.concat(wins, ignore_index=True)
    cache_f = WORK / "comparator_waveform_qc_heedb.parquet"
    mine = pd.read_parquet(cache_f) if cache_f.exists() else pd.DataFrame(columns=["site", "rel", "waveform_ok"])
    cache = pd.concat([pd.read_parquet(WORK / "wp_h_waveform_qc.parquet"), mine], ignore_index=True)
    todo = win[win.statement_ok][["site", "rel"]].drop_duplicates()
    todo = todo.merge(cache[["site", "rel"]].drop_duplicates(), on=["site", "rel"], how="left", indicator=True).query("_merge == 'left_only'")
    print("waveform checks to run:", len(todo), flush=True)
    if len(todo):
        with Pool(24) as pool:
            ok = pool.map(c11.waveform_ok, list(zip(todo.site, todo.rel)), chunksize=256)
        new = pd.DataFrame({"site": todo.site.values, "rel": [r for r, _ in ok], "waveform_ok": [o for _, o in ok]})
        pd.concat([mine, new], ignore_index=True).to_parquet(cache_f, index=False)
        cache = pd.concat([cache, new], ignore_index=True)
    win = win.merge(cache.drop_duplicates(["site", "rel"]), on=["site", "rel"], how="left")
    win["eligible"] = win.statement_ok & win.waveform_ok.eq(True)

    def entered(col):
        e = win[win[col]].groupby(["site", "grp", "pid"]).agg(n=("rel", "size"), lo=("ecg_time", "min"), hi=("ecg_time", "max")).reset_index()
        return e[(e.n >= 2) & ((e.hi - e.lo).dt.days >= 180)][["site", "grp", "pid"]]  # as in 11_heedb_cohort.py

    old = pd.read_parquet(WORK / "wp_h_anchors.parquet")
    final = keep_one_group(entered("eligible").merge(cand, on=["site", "grp", "pid"]), ["site", "pid"],
                           set(zip(old.site, old.pid)))
    own = final[~final.in_other_group]
    ecgs = win[win.eligible].merge(own[["site", "grp", "pid"]], on=["site", "grp", "pid"])
    return {
        "run_date": str(date.today()), "database": "HEEDB", "window_years": W, "scores_read": False,
        "first_heart_failure_code_date_by_type": first_hf, "sites": flows,
        "final_before_waveform_check": by_grp(entered("statement_ok")),
        "final_patients": by_grp(final),
        "final_patients_not_in_other_group": by_grp(own),
        "final_patients_not_in_other_group_by_site": {s: by_grp(g) for s, g in own.groupby("site")},
        "final_ecgs_not_in_other_group": by_grp(ecgs),
        "gate": gate(by_grp(own)),
    }, {"patients": own[["site", "grp", "pid", "anchor_d", "dob", "male", "age"]],
        "ecgs": ecgs[["site", "grp", "pid", "rel", "ecg_time"]]}


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else ""
    if db not in ("mimic", "heedb"):
        sys.exit(__doc__)
    res, _ = mimic() if db == "mimic" else heedb()
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"comparator_counts_{db}.json").write_text(json.dumps(res, indent=2, default=int))
    print(json.dumps(res, indent=2, default=int))
