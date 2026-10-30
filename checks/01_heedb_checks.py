"""HEEDB data checks before the replication protocol is fixed (outcome-blind; aggregates only).

Writes checks/01_heedb_checks.json, per site (I0001 = MGH, I0006 = Emory):
- size: ECGs, patients, waveform headers on disk;
- coverage: ECGs and ICD-9/ICD-10 code dates by year (shifted dates);
- date shift: share of sampled ECGs with an ICD code for the same patient on the same day,
  against the same ECGs shifted by +5 years;
- age: AgeAtAcquisition against (ECG time - date of birth), to fix its unit;
- waveform format on 2,000 random headers (sampling rate, samples, units, lead names).

Paths: manuscript-submission skill, references/datasets.md (HEEDB 5.0).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import duckdb
import wfdb

HEEDB = Path.home() / "datasets/heedb"
OUT = Path(__file__).with_suffix(".json")
SITES = {"I0001": "MGH", "I0006": "Emory"}
PID = "regexp_replace(CAST(BDSPPatientID AS VARCHAR), '\\.0$', '')"  # Emory metadata IDs carry a trailing '.0'


def wfdb_path(site_dir, file_name):
    """FileName is relative to WFDB/ (MGH: './S0001/...hea' or '/S0001/...'; Emory: 'WFDB/2010/MUSE_...')."""
    f = file_name.strip().removeprefix("./").lstrip("/").removeprefix("WFDB/").removesuffix(".hea")
    return site_dir / "WFDB" / f

ap = argparse.ArgumentParser()
ap.add_argument("--sites", nargs="+", default=list(SITES))
args = ap.parse_args()

con = duckdb.connect()
con.execute(f"SET temp_directory = '{HEEDB}/duckdb_tmp'; SET memory_limit = '30GB'; SET threads = 16")
res = json.loads(OUT.read_text()) if OUT.exists() else {}
for site in args.sites:
    name = SITES[site]
    E = HEEDB / "ECG" / site
    r = {"name": name}
    con.execute(f"""
    CREATE OR REPLACE TABLE meta AS
    SELECT {PID} pid, FileName, TRY_CAST(ECGAcquisitionTime AS TIMESTAMP) t,
           TRY_CAST(DateOfBirth AS TIMESTAMP) dob, TRY_CAST(AgeAtAcquisition AS DOUBLE) age_raw
    FROM read_csv_auto('{E}/metadata/metadata.csv', all_varchar=true)
    """)
    r["ecgs"], r["patients"] = con.execute("SELECT count(*), count(DISTINCT pid) FROM meta").fetchone()
    r["hea_on_disk"] = sum(1 for _ in (E / "WFDB").rglob("*.hea"))
    r["ecgs_by_year"] = dict(con.execute("SELECT year(t) y, count(*) FROM meta GROUP BY 1 ORDER BY 1").fetchall())
    for v in ("9", "10"):
        con.execute(f"""
        CREATE OR REPLACE TABLE icd{v} AS
        SELECT {PID} pid, CAST(TRY_CAST(RECORDED_DT AS TIMESTAMP) AS DATE) d
        FROM read_csv_auto('{E}/ICD_codes/icd{v}_codes.csv', all_varchar=true)
        """)
        r[f"icd{v}_rows"], r[f"icd{v}_patients"] = con.execute(f"SELECT count(*), count(DISTINCT pid) FROM icd{v}").fetchone()
        r[f"icd{v}_by_year"] = dict(con.execute(f"SELECT year(d) y, count(*) FROM icd{v} GROUP BY 1 ORDER BY 1").fetchall())
    con.execute("CREATE OR REPLACE TABLE days AS SELECT DISTINCT pid, d FROM (SELECT * FROM icd9 UNION ALL SELECT * FROM icd10)")
    con.execute("CREATE OR REPLACE TABLE samp AS SELECT pid, CAST(t AS DATE) d FROM meta WHERE t IS NOT NULL USING SAMPLE 200000 ROWS (reservoir, 7)")
    same, ctrl = con.execute("""
    SELECT avg((EXISTS (SELECT 1 FROM days x WHERE x.pid = s.pid AND x.d = s.d))::INT),
           avg((EXISTS (SELECT 1 FROM days x WHERE x.pid = s.pid AND x.d = s.d + INTERVAL 5 YEAR))::INT)
    FROM samp s""").fetchone()
    r["date_shift"] = {"sampled_ecgs": 200000, "share_with_code_same_day": round(same, 4),
                       "control_share_plus_5y": round(ctrl, 4)}
    r["age"] = dict(zip(["median_age_raw", "median_days_since_dob", "share_raw_equals_days_pm1"], con.execute("""
    SELECT median(age_raw), median(date_diff('day', dob, t)), avg((abs(age_raw - date_diff('day', dob, t)) <= 1)::INT)
    FROM meta WHERE age_raw IS NOT NULL AND dob IS NOT NULL AND t IS NOT NULL""").fetchone()))
    files = [f for (f,) in con.execute("SELECT FileName FROM meta USING SAMPLE 2000 ROWS (reservoir, 3)").fetchall()]
    fmt = Counter()
    for f in files:
        try:
            h = wfdb.rdheader(str(wfdb_path(E, f)))
            fmt[(h.fs, h.sig_len, tuple(h.units), tuple(h.sig_name))] += 1
        except Exception as e:
            fmt[("error", type(e).__name__)] += 1
    r["waveform_format"] = [{"fs_len_units_leads": [str(x) for x in k], "n": n} for k, n in fmt.most_common(5)]
    res[site] = r
    print(site, "done", flush=True)

OUT.write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({s: {k: v for k, v in r.items() if not k.endswith("_by_year")} for s, r in res.items()}, indent=2, default=str))
