"""WP2-H — EchoNext-Mini scoring in HEEDB (concept.md §6.7 items 5, 6, 10). No slope is estimated here.

    python3 -I analysis/12_heedb_score.py

Inputs: work/wp_h_ecgs.parquet, work/wp_h_g2_sample.parquet (from analysis/11_heedb_cohort.py).
Outputs: work/wp_h_scores.parquet, work/wp_h_g2_scores.parquet (patient-level, local only),
         results/wp_h_g2.json (aggregate).

Waveforms: as in analysis/02_score.py (mV -> MUSE units, limb leads from I and II, baseline removal,
released clipping and normalization); 500 Hz records are decimated by taking every second sample,
250 Hz records are used as recorded. Tabular inputs: ventricular rate from lead II R peaks; PR, QRS and
QTc held at the training medians (scaled); atrial rate 0; age frozen at the first eligible ECG (cohort)
or at the ECG (G2); sex from the metadata.
"""
import importlib.util
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.metrics import roc_auc_score

PROJ = Path(__file__).resolve().parents[1]
HEEDB = Path.home() / "datasets/heedb"
WORK, RESULTS = PROJ / "work", PROJ / "results"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SCORE = _mod("score", PROJ / "analysis/02_score.py")
TABW = _mod("tabular_from_waveform", PROJ / "analysis/tabular_from_waveform.py")
COH = _mod("heedb_cohort", PROJ / "analysis/11_heedb_cohort.py")
HELD = [4, 5, 6]  # PR, QRS, QTc positions in the 7-vector (male first)
CONST = np.array(SCORE.TAB_PARAMS["impute_statistics_scaled"])[[3, 4, 5]]


def heedb_waveform(arg):
    """(site, rel) -> (model-space (2500, 12) float32 or None, ventricular rate)."""
    site, rel = arg
    try:
        rec = wfdb.rdrecord(str(HEEDB / "ECG" / site / "WFDB" / rel))
    except Exception:
        return None, np.nan
    vr = TABW.ventricular_rate(rec.p_signal, rec.fs, rec.sig_name)
    step = {500: 2, 250: 1}.get(rec.fs)
    if step is None:
        return None, vr
    sig = dict(zip(rec.sig_name, (rec.p_signal * 1000.0 / SCORE.MUSE_UV_PER_BIT).T))
    lead = {k: sig[k][::step] for k in ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]}
    lead["III"] = lead["II"] - lead["I"]
    lead["aVR"] = -(lead["I"] + lead["II"]) / 2
    lead["aVF"] = lead["II"] - lead["I"] / 2
    lead["aVL"] = lead["I"] - lead["II"] / 2
    x = SCORE.baseline_wander_removal(np.stack([lead[k] for k in SCORE.LEAD_ORDER]))
    p = SCORE.WAVE_PARAMS
    x = (np.clip(x, np.array(p["lowerbound"])[:, None], np.array(p["upperbound"])[:, None])
         - np.array(p["mean"])[:, None]) / np.array(p["std"])[:, None]
    return x.T.astype(np.float32), vr


def score_frame(df, model, device, chunk=8000, workers=24):
    out = []
    for i in range(0, len(df), chunk):
        part = df.iloc[i:i + chunk].copy()
        with Pool(workers) as pool:
            res = pool.map(heedb_waveform, list(zip(part.site, part.rel)), chunksize=64)
        ok = np.array([w is not None for w, _ in res])
        part["ventricular_rate"] = [vr for _, vr in res]
        part = part[ok]
        wave = np.stack([w for w, _ in res if w is not None])[:, None, :, :]
        tab_in = part.assign(atrial_rate=np.nan, pr_interval=np.nan, qrs_duration=np.nan, qt_corrected=np.nan)
        tab = SCORE.tabular(tab_in)
        tab[:, HELD] = CONST
        part["logit_lvef"] = SCORE.logits(model, wave, tab, device)[:, SCORE.LVEF]
        out.append(part)
        print(f"  scored {min(i + chunk, len(df))}/{len(df)}", flush=True)
    return pd.concat(out, ignore_index=True)


def lvsd_dates(con):
    parts = []
    for site in COH.SITES:
        E = HEEDB / "ECG" / site
        for v, col in (("9", "DIAGNOSIS_ICD9_CD"), ("10", "DIAGNOSIS_ICD10_CD")):
            rx = "^428[24]" if v == "9" else "^I50[24]"
            parts.append(f"""SELECT DISTINCT '{site}' site, {COH.PID} pid, CAST(TRY_CAST(RECORDED_DT AS TIMESTAMP) AS DATE) d
                             FROM read_csv_auto('{E}/ICD_codes/icd{v}_codes.csv', all_varchar=true)
                             WHERE regexp_matches(replace({col}, '.', ''), '{rx}')""")
    con.execute(f"CREATE TABLE lv AS {' UNION ALL '.join(parts)}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SCORE.load_model(device)

    ecgs = pd.read_parquet(WORK / "wp_h_ecgs.parquet")
    anc = pd.read_parquet(WORK / "wp_h_anchors.parquet")[["site", "grp", "pid", "male"]]
    ecgs = ecgs.merge(anc, on=["site", "grp", "pid"]).sort_values("ecg_time")
    ecgs["age_at_ecg"] = ecgs.groupby(["site", "grp", "pid"]).age_at_ecg_real.transform("first")  # frozen (§6.1)
    ecgs["male"] = ecgs.male.astype(int)
    sc = score_frame(ecgs, model, device)
    sc.to_parquet(WORK / "wp_h_scores.parquet", index=False)

    g2 = pd.read_parquet(WORK / "wp_h_g2_sample.parquet")
    g2["male"] = g2.male.astype(int)
    g2 = score_frame(g2, model, device)
    con = duckdb.connect()
    con.execute(f"SET temp_directory = '{HEEDB}/duckdb_tmp'; SET memory_limit = '40GB'; SET threads = 24")
    lvsd_dates(con)
    con.register("g2_df", g2[["site", "pid", "rel", "t"]])
    lab = con.execute("""
    SELECT g.site, g.rel, EXISTS (SELECT 1 FROM lv WHERE lv.site = g.site AND lv.pid = g.pid AND lv.d <= CAST(g.t AS DATE)) AS prevalent_lvsd
    FROM g2_df g""").df()
    g2 = g2.merge(lab, on=["site", "rel"])
    g2[["site", "pid", "rel", "t", "age_at_ecg", "logit_lvef", "prevalent_lvsd"]].to_parquet(WORK / "wp_h_g2_scores.parquet", index=False)

    def auc_block(d):
        y, s = d.prevalent_lvsd.to_numpy(), d.logit_lvef.to_numpy()
        rng = np.random.default_rng(1)
        boot = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
        return {"n": int(len(y)), "prevalent_lvsd": int(y.sum()), "auroc": round(roc_auc_score(y, s), 4),
                "auroc_95ci_bootstrap": [round(float(np.quantile(boot, q)), 4) for q in (0.025, 0.975)],
                "D_G2_logit": round(float(s[y].mean() - s[~y].mean()), 4)}

    pooled = auc_block(g2)
    res = {"run_date": str(date.today()),
           "cohort_scoring": {"ecgs_in": int(len(ecgs)), "scored": int(len(sc)), "unreadable": int(len(ecgs) - len(sc)),
                              "ventricular_rate_missing_share": round(float(sc.ventricular_rate.isna().mean()), 4)},
           "G2_H": pooled | {"threshold": 0.70, "pass": bool(pooled["auroc"] >= 0.70),
                             "by_site": {s: auc_block(g2[g2.site == s]) for s in COH.SITES}},
           "delta_used": json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]}
    (RESULTS / "wp_h_g2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
