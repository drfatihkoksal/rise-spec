"""WP1-H — HEEDB cohort and anchors (concept.md §6.7; outcome-blind: codes, dates, 12SL statements, raw waveforms).

    python3 -I analysis/11_heedb_cohort.py

Outputs (patient-level, local only): work/wp_h_anchors.parquet, work/wp_h_ecgs.parquet, work/wp_h_g2_sample.parquet
Output (aggregate): results/wp_h_flow.json (flow per site and pooled, G1-H)

Paths: README, Data access (HEEDB 5.0).
"""
import json
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import wfdb

HEEDB = Path.home() / "datasets/heedb"
PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
SITES = {"I0001": ("MGH", "SexDSC"), "I0006": ("Emory", "Sex")}
W = 5  # years, the window applied in MIMIC after G1

# 12SL v24 statement codes (concept §6.7 item 4)
SINUS = [19, 21, 22, 23, 24]
NON_SINUS = ([29, 30, 31, 34, 41, 42, 43, 61, 62, 63, 64, 108, 161, 162, 174, 175, 183, 184, 185, 186, 235, 237, 238,
              242, 248, 249, 266, 267, 268, 270, 271, 272, 273, 275, 279] + list(range(288, 300)) + [326, 1669])
BAD_QUALITY = [1302, 1500, 1595, 1672, 1673]
CONDUCTION = [440, 442, 460, 482, 487]  # S2-H: RBBB, RBBB-or-RVH, LBBB, IVCB, IVCD
FLAT_MV = 0.05
PID = "regexp_replace(CAST(BDSPPatientID AS VARCHAR), '\\.0$', '')"
KEY = "regexp_replace(regexp_replace(regexp_replace(FileName, '\\s', '', 'g'), '^(\\./|/)?(WFDB/)?', ''), '\\.hea$', '')"


def waveform_ok(arg):
    site, rel = arg
    try:
        sig, f = wfdb.rdsamp(str(HEEDB / "ECG" / site / "WFDB" / rel))
    except Exception:
        return rel, False
    if (f["fs"], sig.shape) not in {(500, (5000, 12)), (250, (2500, 12))} or np.isnan(sig).any():
        return rel, False
    return rel, bool((np.ptp(sig, axis=0) >= FLAT_MV).all())


def load_site(con, site, sexcol):
    E = HEEDB / "ECG" / site
    con.execute(f"""
    CREATE OR REPLACE TABLE meta AS
    SELECT '{site}' site, {PID} pid, {KEY} rel, TRY_CAST(ECGAcquisitionTime AS TIMESTAMP) t,
           TRY_CAST(DateOfBirth AS TIMESTAMP) dob, lower({sexcol}) = 'male' AS male, {sexcol} IS NOT NULL AS sex_known
    FROM read_csv_auto('{E}/metadata/metadata.csv', all_varchar=true)
    WHERE TRY_CAST(ECGAcquisitionTime AS TIMESTAMP) IS NOT NULL;
    CREATE OR REPLACE TABLE stm AS
    WITH c AS (SELECT {KEY} rel, list_transform(string_split(codes, ','), x -> TRY_CAST(trim(x) AS INT)) cl
               FROM read_csv_auto('{E}/12SL_diagnoses/diagnoses_v24.csv', all_varchar=true))
    SELECT rel,
           list_has_any(cl, {SINUS}) AND NOT list_has_any(cl, {NON_SINUS}) AS sinus,
           list_has_any(cl, {BAD_QUALITY}) AS bad_quality,
           list_has_any(cl, {CONDUCTION}) AS conduction
    FROM c;
    """)
    parts = []
    for v, col in (("9", "DIAGNOSIS_ICD9_CD"), ("10", "DIAGNOSIS_ICD10_CD")):
        parts.append(f"""SELECT {PID} pid, CAST(TRY_CAST(RECORDED_DT AS TIMESTAMP) AS DATE) d, replace({col}, '.', '') c, {v} v
                         FROM read_csv_auto('{E}/ICD_codes/icd{v}_codes.csv', all_varchar=true)""")
    con.execute(f"CREATE OR REPLACE TABLE dx_all AS {' UNION ALL '.join(parts)}")
    con.execute("""
    CREATE OR REPLACE TABLE first_code AS SELECT pid, min(d) first_d FROM dx_all WHERE d IS NOT NULL GROUP BY 1;
    CREATE OR REPLACE TABLE day_sets AS
    SELECT pid, d,
      bool_or((v = 10 AND regexp_matches(c, '^I50[24]')) OR (v = 9 AND regexp_matches(c, '^428[24]'))) AS lvsd,
      bool_or((v = 10 AND c LIKE 'I48%') OR (v = 9 AND c IN ('42731', '42732'))) AS af,
      bool_or((v = 10 AND regexp_matches(c, '^S(42|52|62|72|82|92)')) OR (v = 9 AND regexp_matches(c, '^8[12][0-9]'))) AS ctrl,
      bool_or((v = 10 AND regexp_matches(c, '^S(42|52|62|72|82|92)') AND length(c) = 7 AND right(c, 1) IN ('A', 'B', 'C'))
              OR (v = 9 AND regexp_matches(c, '^8[12][0-9]'))) AS ctrl_initial,
      bool_or((v = 10 AND regexp_matches(c, '^I(50|42)')) OR (v = 9 AND regexp_matches(c, '^42[58]'))) AS hfcm
    FROM dx_all
    WHERE d IS NOT NULL AND (regexp_matches(c, '^(I50|I42|I48|S42|S52|S62|S72|S82|S92)') OR regexp_matches(c, '^(428|425|4273|81|82)'))
    GROUP BY ALL;
    """)


def anchors(con, site):
    con.execute(f"""
    CREATE OR REPLACE TABLE anc AS
    SELECT pid, d AS anchor_d, lvsd::INT + af::INT + ctrl::INT AS n_sets, ctrl_initial,
           CASE WHEN lvsd THEN 'LVSD' WHEN af THEN 'AF' ELSE 'LIMBFX' END AS grp
    FROM (SELECT *, row_number() OVER (PARTITION BY pid ORDER BY d) rn FROM day_sets WHERE lvsd OR af OR ctrl) WHERE rn = 1;
    CREATE OR REPLACE TABLE pat AS SELECT pid, any_value(dob) dob, any_value(male) male, bool_or(sex_known) sex_known FROM meta GROUP BY 1;
    CREATE OR REPLACE TABLE anc_f AS
    SELECT '{site}' site, x.pid, x.anchor_d, x.grp,
      x.n_sets > 1 AS ex_multi_set,
      x.grp = 'LIMBFX' AND NOT x.ctrl_initial AS ex_control_not_initial,
      x.grp = 'LVSD' AND EXISTS (SELECT 1 FROM day_sets s WHERE s.pid = x.pid AND s.d < x.anchor_d AND s.hfcm) AS ex_prior_hf_cm,
      x.grp IN ('AF', 'LIMBFX') AND EXISTS (SELECT 1 FROM day_sets s WHERE s.pid = x.pid AND s.lvsd
                                            AND s.d > x.anchor_d AND s.d <= x.anchor_d + INTERVAL 365 DAY) AS ex_lvsd_within_1y,
      coalesce(fc.first_d > x.anchor_d - INTERVAL 365 DAY, true) AS ex_lookback_lt_1y,
      p.dob IS NULL OR NOT p.sex_known OR date_diff('day', p.dob, x.anchor_d) / 365.25 < 18 AS ex_age_or_sex_missing,
      NOT EXISTS (SELECT 1 FROM meta e WHERE e.pid = x.pid
                  AND CAST(e.t AS DATE) BETWEEN x.anchor_d - INTERVAL 30 DAY AND x.anchor_d + INTERVAL 30 DAY) AS ex_no_ecg_at_anchor,
      date_diff('day', p.dob, x.anchor_d) / 365.25 AS age, p.male, p.dob
    FROM anc x LEFT JOIN pat p USING (pid) LEFT JOIN first_code fc USING (pid)
    """)
    return ["ex_control_not_initial", "ex_prior_hf_cm", "ex_lvsd_within_1y", "ex_lookback_lt_1y",
            "ex_age_or_sex_missing", "ex_no_ecg_at_anchor"]


def flow(df, steps):
    out, keep = {"start": df.groupby("grp").size().to_dict()}, pd.Series(True, index=df.index)
    for s in steps:
        out[s] = df[keep & df[s]].groupby("grp").size().to_dict()
        keep &= ~df[s]
        out[f"after_{s}"] = df[keep].groupby("grp").size().to_dict()
    return out, keep


def g2_sample(con, site, n=100_000, seed=20261030):
    """G2-H (§6.7 item 10): random patients; one random sinus, quality-ok ECG dated >=365 d after the first code, adult."""
    return con.execute(f"""
    WITH e AS (
      SELECT m.pid, m.rel, m.t, date_diff('day', m.dob, m.t) / 365.25 AS age_at_ecg, m.male
      FROM meta m JOIN stm s USING (rel) JOIN first_code fc USING (pid)
      WHERE s.sinus AND NOT s.bad_quality AND m.sex_known AND m.dob IS NOT NULL
        AND date_diff('day', m.dob, m.t) / 365.25 >= 18 AND CAST(m.t AS DATE) >= fc.first_d + INTERVAL 365 DAY),
    p AS (SELECT pid FROM (SELECT DISTINCT pid FROM e) ORDER BY hash(pid || '{seed}') LIMIT {n})
    SELECT '{site}' site, * FROM (
      SELECT e.*, row_number() OVER (PARTITION BY e.pid ORDER BY hash(e.rel || '{seed}')) rn FROM e JOIN p USING (pid)) WHERE rn = 1
    """).df().drop(columns="rn")


def main():
    WORK.mkdir(exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{HEEDB}/duckdb_tmp'; SET memory_limit = '40GB'; SET threads = 24")
    flows, anc_keep, win_all, g2_all = {}, [], [], []
    for site, (name, sexcol) in SITES.items():
        load_site(con, site, sexcol)
        steps = anchors(con, site)
        a = con.execute("SELECT * FROM anc_f").df()
        n_first, n_multi = len(a), int(a.ex_multi_set.sum())
        f, keep = flow(a[~a.ex_multi_set], steps)
        cand = a[~a.ex_multi_set][keep][["site", "pid", "anchor_d", "grp", "age", "male", "dob"]]
        con.register("cand_df", cand)
        win = con.execute(f"""
        SELECT c.site, c.grp, c.pid, m.rel, m.t AS ecg_time, date_diff('second', CAST(c.anchor_d AS TIMESTAMP), m.t) / (365.25 * 86400) AS t_years,
               date_diff('day', c.dob, m.t) / 365.25 AS age_at_ecg_real,
               s.rel IS NOT NULL AS has_statements,
               coalesce(s.sinus, false) sinus, coalesce(s.bad_quality, false) bad_quality, coalesce(s.conduction, false) conduction
        FROM cand_df c JOIN meta m USING (pid) LEFT JOIN stm s USING (rel)
        WHERE m.t >= CAST(c.anchor_d AS TIMESTAMP) - INTERVAL {W} YEAR AND m.t <= CAST(c.anchor_d AS TIMESTAMP) - INTERVAL 30 DAY
        """).df()
        flows[site] = {"name": name, "first_event_patients": n_first, "ex_multi_set": n_multi, "anchor_flow": f,
                       "share_window_ecgs_with_12sl_v24": round(float(win.has_statements.mean()), 4)}
        anc_keep.append(cand)
        win_all.append(win)
        g2_all.append(g2_sample(con, site))
        print(site, "loaded", len(cand), "candidates", len(win), "window ECGs", flush=True)

    cand, win, g2 = pd.concat(anc_keep), pd.concat(win_all, ignore_index=True), pd.concat(g2_all, ignore_index=True)
    todo = win[win.sinus & ~win.bad_quality][["site", "rel"]].drop_duplicates()
    todo = pd.concat([todo, g2[["site", "rel"]]]).drop_duplicates()
    cache_f = WORK / "wp_h_waveform_qc.parquet"
    cache = pd.read_parquet(cache_f) if cache_f.exists() else pd.DataFrame(columns=["site", "rel", "waveform_ok"])
    todo = todo.merge(cache[["site", "rel"]], on=["site", "rel"], how="left", indicator=True).query("_merge == 'left_only'")
    if len(todo):
        with Pool(24) as pool:
            ok = pool.map(waveform_ok, list(zip(todo.site, todo.rel)), chunksize=256)
        new = pd.DataFrame({"site": todo.site.values, "rel": [r for r, _ in ok], "waveform_ok": [o for _, o in ok]})
        cache = pd.concat([cache, new], ignore_index=True)
        cache.to_parquet(cache_f, index=False)
    win = win.merge(cache, on=["site", "rel"], how="left")
    win["waveform_ok"] = win.waveform_ok.fillna(False).astype(bool)
    win["eligible"] = win.sinus & ~win.bad_quality & win.waveform_ok
    g2 = g2.merge(cache, on=["site", "rel"], how="left")
    n_g2_qc_fail = int((~g2.waveform_ok.fillna(False).astype(bool)).sum())
    g2 = g2[g2.waveform_ok.fillna(False).astype(bool)].drop(columns="waveform_ok")

    el = win[win.eligible]
    per = el.groupby(["site", "grp", "pid"]).agg(n=("rel", "size"), lo=("ecg_time", "min"), hi=("ecg_time", "max")).reset_index()
    per["enter"] = (per.n >= 2) & ((per.hi - per.lo).dt.days >= 180)
    anyw = win.groupby(["site", "grp", "pid"]).size().rename("n_window").reset_index()
    c = cand.merge(anyw, on=["site", "grp", "pid"], how="left").merge(per[["site", "grp", "pid", "enter"]], on=["site", "grp", "pid"], how="left")
    c["ex_no_window_ecg"] = c.n_window.isna()
    c["ex_lt2_eligible_or_span_lt180d"] = ~c.ex_no_window_ecg & ~c.enter.fillna(False).astype(bool)
    final = c[~c.ex_no_window_ecg & ~c.ex_lt2_eligible_or_span_lt180d]
    for site in SITES:
        cs = c[c.site == site]
        flows[site]["ecg_flow"] = {
            "candidates": cs.groupby("grp").size().to_dict(),
            "ex_no_window_ecg": cs[cs.ex_no_window_ecg].groupby("grp").size().to_dict(),
            "ex_lt2_eligible_or_span_lt180d": cs[cs.ex_lt2_eligible_or_span_lt180d].groupby("grp").size().to_dict(),
            "final_patients": final[final.site == site].groupby("grp").size().to_dict()}
    ecgs = el.merge(final[["site", "grp", "pid"]], on=["site", "grp", "pid"])
    final[["site", "grp", "pid", "anchor_d", "age", "male"]].to_parquet(WORK / "wp_h_anchors.parquet", index=False)
    ecgs.drop(columns=["has_statements", "sinus", "bad_quality", "waveform_ok", "eligible"]).to_parquet(WORK / "wp_h_ecgs.parquet", index=False)
    g2.to_parquet(WORK / "wp_h_g2_sample.parquet", index=False)

    pooled = final.groupby("grp").size().to_dict()
    res = {"run_date": str(date.today()), "window_years": W, "sites": flows,
           "pooled_final_patients": pooled,
           "pooled_ecgs": ecgs.groupby("grp").size().to_dict(),
           "share_conduction_statement": ecgs.groupby("grp").conduction.mean().round(3).to_dict(),
           "g2_sample": {"n": len(g2), "dropped_waveform_check": n_g2_qc_fail, "by_site": g2.groupby("site").size().to_dict()},
           "G1_H": {"threshold": 500, "counts": {g: int(pooled.get(g, 0)) for g in ["LVSD", "AF", "LIMBFX"]},
                    "pass": all(pooled.get(g, 0) >= 500 for g in ["LVSD", "AF", "LIMBFX"])}}
    (RESULTS / "wp_h_flow.json").write_text(json.dumps(res, indent=2, default=int))
    print(json.dumps(res, indent=2, default=int))


if __name__ == "__main__":
    main()
