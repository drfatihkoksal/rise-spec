"""Agreement of waveform-derived intervals with MIMIC-IV-ECG machine measurements (outcome-blind).

    python3 -I checks/02_interval_agreement.py [--n 2000]

On random sinus, quality-checked MIMIC ECGs, compares analysis/tabular_from_waveform.py with the
machine measurements used in WP2: ventricular rate from R peaks, and PR/QRS/QTc from DWT delineation in
three variants (lead II, median or min/max across II, V2, V5). Reports bias (median derived - machine),
median absolute difference, Pearson r and share derivable. Writes checks/02_interval_agreement.json.
"""
import argparse
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import wfdb

PROJ = Path(__file__).resolve().parents[1]
ECG = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/mimic-iv-ecg/1.0")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


TAB = _mod("tabular_from_waveform", PROJ / "analysis/tabular_from_waveform.py")
SCORE = _mod("score", PROJ / "analysis/02_score.py")


VARIANTS = ["II", "median", "minmax"]


def derive(path):
    rec = wfdb.rdrecord(str(ECG / path))
    out = {"ventricular_rate": TAB.ventricular_rate(rec.p_signal, rec.fs, rec.sig_name)}
    for agg in VARIANTS:
        for k, v in TAB.delineated_intervals(rec.p_signal, rec.fs, rec.sig_name, agg).items():
            out[f"{k}__{agg}"] = v
    return path, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    g2 = pd.read_parquet(PROJ / "work/wp2_g2_scores.parquet")[["study_id"]]  # sinus, quality-checked, adult
    rl = pd.read_csv(ECG / "record_list.csv", usecols=["study_id", "path"]).merge(g2, on="study_id").sample(args.n, random_state=11)
    t0 = time.time()
    with Pool(args.workers) as pool:
        der = dict(pool.map(derive, rl.path.tolist(), chunksize=8))
    secs = time.time() - t0
    d = pd.DataFrame([{"path": p, **v} for p, v in der.items()]).merge(rl, on="path")
    m = SCORE.machine_tabular(duckdb.connect()).set_index("study_id")
    d = d.drop(columns=["ventricular_rate"]).join(m, on="study_id").join(
        pd.DataFrame([{"path": p, "ventricular_rate_derived": v["ventricular_rate"]} for p, v in der.items()]).set_index("path"), on="path")
    d = d.rename(columns={"ventricular_rate_derived": "ventricular_rate_d"})
    out = {"n": len(d), "seconds_per_ecg_per_worker": round(secs * args.workers / len(d), 3)}
    pairs = [("ventricular_rate_d", "ventricular_rate")] + [(f"{v}__{agg}", v) for v in ["pr_interval", "qrs_duration", "qt_corrected"] for agg in VARIANTS]
    for v, ref in pairs:
        a, b = d[v], d[ref]
        ok = a.notna() & b.notna()
        diff = (a - b)[ok]
        out[v] = {"derivable": round(float(a.notna().mean()), 3), "bias_median": round(float(diff.median()), 1),
                  "mad": round(float(diff.abs().median()), 1), "r": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3)}
    (Path(__file__).with_suffix(".json")).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
