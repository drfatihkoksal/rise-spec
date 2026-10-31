"""Pre-WP1 data checks for RISE-SPEC (concept §10). Aggregates only.

Writes checks/00_data_checks.json:
- echo: whether MIMIC-IV-ECHO structured measurements give TTE LVEF, and how many
  patients have an echo-incident LVSD (TTE LVEF >45% followed by <=45%), before any
  ECG eligibility rule;
- date shift: whether MIMIC-IV-ECG and MIMIC-IV hosp/ED share the per-patient date
  shift (share of ECGs inside a same-patient admission or ED stay, against a +5 y
  shifted control);
- coverage: ECGs and admissions by approximate real year (anchor_year_group midpoint,
  so each year is only accurate to about +-1 y).

Paths and versions: README, Data access.
"""
import json
from pathlib import Path

import duckdb

V = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
HOSP = V / "mimiciv/3.1/hosp"
ECG = V / "mimic-iv-ecg/1.0"
ED = V / "mimic-iv-ed-2.2/ed"
ECHO = V / "echo/echo"
ECHO_COPY = Path.home() / "physionet.org/files/mimic-iv-echo/1.0"
OUT = Path(__file__).with_suffix(".json")

con = duckdb.connect()
res = {}

# --- echo ---------------------------------------------------------------
res["echo_copies_identical"] = (ECHO / "SHA256SUMS.txt").read_bytes() == (ECHO_COPY / "SHA256SUMS.txt").read_bytes()
con.execute(f"""
CREATE TABLE ef AS
SELECT subject_id, measurement_datetime AS t, TRY_CAST(result AS DOUBLE) AS lvef
FROM read_csv_auto('{ECHO}/structured-measurement.csv.gz', all_varchar=true)
WHERE test_type = 'tte' AND measurement = 'lvef'
""")
rows, numeric, patients, lo, hi, share45 = con.execute("""
SELECT count(*), count(lvef), count(DISTINCT subject_id) FILTER (WHERE lvef IS NOT NULL),
       min(lvef), max(lvef), avg((lvef <= 45)::INT)
FROM ef
""").fetchone()
incident = con.execute("""
WITH f AS (SELECT subject_id, min(t) AS t_low FROM ef WHERE lvef <= 45 GROUP BY 1)
SELECT count(DISTINCT e.subject_id) FROM ef e JOIN f USING (subject_id)
WHERE e.lvef > 45 AND e.t < f.t_low
""").fetchone()[0]
res["tte_lvef"] = {
    "rows": rows, "numeric": numeric, "patients_numeric": patients,
    "min": lo, "max": hi, "share_le45": round(share45, 4),
    "patients_echo_incident_lvsd": incident,
}

# --- date shift -----------------------------------------------------------
con.execute(f"CREATE VIEW ecg AS SELECT subject_id, CAST(ecg_time AS TIMESTAMP) AS t FROM read_csv_auto('{ECG}/record_list.csv')")
con.execute(f"CREATE VIEW adm AS SELECT subject_id, admittime, dischtime FROM read_csv_auto('{HOSP}/admissions.csv.gz')")
con.execute(f"CREATE VIEW ed AS SELECT subject_id, intime, outtime FROM read_csv_auto('{ED}/edstays.csv.gz')")
con.execute(f"CREATE VIEW pat AS SELECT subject_id, anchor_year, CAST(left(anchor_year_group, 4) AS INT) + 1 AS mid FROM read_csv_auto('{HOSP}/patients.csv.gz')")

n_ecg, n_pat = con.execute("SELECT count(*), count(DISTINCT subject_id) FROM ecg").fetchone()
in_hosp = con.execute("SELECT avg((p.subject_id IS NOT NULL)::INT) FROM (SELECT DISTINCT subject_id FROM ecg) e LEFT JOIN pat p USING (subject_id)").fetchone()[0]


def share_in_stay(shift):
    return con.execute(f"""
    SELECT avg((EXISTS (SELECT 1 FROM adm a WHERE a.subject_id = e.subject_id AND e.t + {shift} BETWEEN a.admittime AND a.dischtime)
             OR EXISTS (SELECT 1 FROM ed d WHERE d.subject_id = e.subject_id AND e.t + {shift} BETWEEN d.intime AND d.outtime))::INT)
    FROM ecg e
    """).fetchone()[0]


res["date_shift"] = {
    "ecgs": n_ecg, "ecg_patients": n_pat, "share_ecg_patients_in_hosp": round(in_hosp, 4),
    "share_ecgs_in_admission_or_ed_stay": round(share_in_stay("INTERVAL 0 DAY"), 4),
    "control_share_after_plus_5y_shift": round(share_in_stay("INTERVAL 5 YEAR"), 4),
}

# --- coverage --------------------------------------------------------------
res["approx_real_year"] = {
    "ecgs": dict(con.execute("SELECT year(e.t) - p.anchor_year + p.mid AS y, count(*) FROM ecg e JOIN pat p USING (subject_id) GROUP BY 1 ORDER BY 1").fetchall()),
    "admissions": dict(con.execute("SELECT year(a.admittime) - p.anchor_year + p.mid AS y, count(*) FROM adm a JOIN pat p USING (subject_id) GROUP BY 1 ORDER BY 1").fetchall()),
}

OUT.write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=2, default=str))
