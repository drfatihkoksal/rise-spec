"""WP2 — EchoNext-Mini scoring (concept.md §6.1, §7 G2).

    python3 -I analysis/02_score.py check   # implementation check, before any MIMIC score
    python3 -I analysis/02_score.py score   # cohort ECGs + G2 sample; G2, D_G2 and delta

check  -> results/wp2_check.json: AUROC on the released EchoNext test split (published [10]: LVEF 0.852,
          SHD 0.820), our tabular transform against the released preprocessed features, and MIMIC
          waveform scale against the training normalization parameters.
score  -> work/wp2_scores_<tag>.parquet, work/wp2_g2_scores.parquet (patient-level, local only),
          results/wp2_g2.json (aggregate). No slope is estimated here.

Model code and weights: external/IntroECG/7-EchoNext Minimodel (commit 15233e9). Preprocessing follows
its parse_xml.py and preprocess.py; the tabular pipeline parameters were exported from
tabular_transformer.joblib with an allow-listed unpickler to analysis/echonext_tabular_params.json.
"""
import argparse
import importlib.util
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import scipy.signal
import torch
import wfdb
from sklearn.metrics import roc_auc_score

V = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c")
HOSP = V / "mimiciv/3.1/hosp"
ECG = V / "mimic-iv-ecg/1.0"
ECHONEXT = V / "echonext"
PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
MODEL_DIR = PROJ / "external/IntroECG/7-EchoNext Minimodel"
WEIGHTS = MODEL_DIR / "models/echonext_multilabel_minimodel/weights.pt"
WAVE_PARAMS = json.loads((MODEL_DIR / "models/echonext_multilabel_minimodel/waveform_normalization_params.json").read_text())
TAB_PARAMS = json.loads((PROJ / "analysis/echonext_tabular_params.json").read_text())

LABELS = ["lvef_lte_45", "lvwt_gte_13", "aortic_stenosis", "aortic_regurgitation", "mitral_regurgitation",
          "tricuspid_regurgitation", "pulmonary_regurgitation", "rv_systolic_dysfunction",
          "pericardial_effusion", "pasp_gte_45", "tr_max_gte_32", "shd"]
LABEL_COLS = ["lvef_lte_45_flag", "lvwt_gte_13_flag", "aortic_stenosis_moderate_or_greater_flag",
              "aortic_regurgitation_moderate_or_greater_flag", "mitral_regurgitation_moderate_or_greater_flag",
              "tricuspid_regurgitation_moderate_or_greater_flag", "pulmonary_regurgitation_moderate_or_greater_flag",
              "rv_systolic_dysfunction_moderate_or_greater_flag", "pericardial_effusion_moderate_large_flag",
              "pasp_gte_45_flag", "tr_max_gte_32_flag", "shd_moderate_or_greater_flag"]
LVEF = 0  # output column of lvef_lte_45
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
MUSE_UV_PER_BIT = 4.88  # GE MUSE amplitude unit (xml_input/MUSE_example.xml)
MISSING = 29999


# --- model -----------------------------------------------------------------------------
def load_model(device):
    spec = importlib.util.spec_from_file_location("echonext_resnet", MODEL_DIR / "cradlenet/models/resnet1d_tabular.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ResNet1dWithTabular(len_tabular_feature_vector=7, filter_size=16, num_classes=12)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True)["model"])
    return model.to(device).eval()


@torch.no_grad()
def logits(model, wave, tab, device, batch=512):
    """wave (N, 1, 2500, 12), tab (N, 7) -> logits (N, 12)."""
    out = []
    for i in range(0, len(wave), batch):
        x = torch.as_tensor(np.asarray(wave[i:i + batch]), dtype=torch.float32, device=device)
        t = torch.as_tensor(np.asarray(tab[i:i + batch]), dtype=torch.float32, device=device)
        out.append(model((x, t)).cpu().numpy())
    return np.concatenate(out)


# --- preprocessing (parse_xml.py, preprocess.py) ------------------------------------------
def tabular(df):
    """df columns: male (0/1), age_at_ecg, ventricular_rate, atrial_rate, pr_interval, qrs_duration, qt_corrected."""
    x = df[TAB_PARAMS["features"]].astype(float).copy()
    x["atrial_rate"] = x["atrial_rate"].fillna(0)  # preprocess.py: missing atrial rate and PR are 0
    x["pr_interval"] = x["pr_interval"].fillna(0)
    z = (x.to_numpy() - np.array(TAB_PARAMS["scale_mean"])) / np.array(TAB_PARAMS["scale_scale"])
    z = np.where(np.isnan(z), np.array(TAB_PARAMS["impute_statistics_scaled"]), z)
    return np.column_stack([df["male"].to_numpy(dtype=float), z]).astype(np.float32)


def baseline_wander_removal(x, fs=250):
    """x (12, 2500); two median filters, 0.2 s then 0.6 s."""
    base = np.stack([scipy.signal.medfilt(lead, int(round(0.2 * fs)) + 1) for lead in x])
    base = np.stack([scipy.signal.medfilt(lead, int(round(0.6 * fs)) + 1) for lead in base])
    return x - base


def mimic_waveform(path, normalize=True):
    """MIMIC-IV-ECG record -> (2500, 12) float32 in model space, or None if unreadable."""
    try:
        rec = wfdb.rdrecord(str(ECG / path))
    except Exception:
        return None
    sig = dict(zip(rec.sig_name, (rec.p_signal * 1000.0 / MUSE_UV_PER_BIT).T))  # mV -> MUSE units
    lead = {k: sig[k][::2] for k in ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]}  # 500 -> 250 Hz, as decode_ekg_muse_to_array
    lead["III"] = lead["II"] - lead["I"]  # limb leads derived from I and II, as extract_waveform_from_xml
    lead["aVR"] = -(lead["I"] + lead["II"]) / 2
    lead["aVF"] = lead["II"] - lead["I"] / 2
    lead["aVL"] = lead["I"] - lead["II"] / 2
    x = baseline_wander_removal(np.stack([lead[k] for k in LEAD_ORDER]))
    if not normalize:
        return x.T.astype(np.float32)
    lo, hi = np.array(WAVE_PARAMS["lowerbound"])[:, None], np.array(WAVE_PARAMS["upperbound"])[:, None]
    x = (np.clip(x, lo, hi) - np.array(WAVE_PARAMS["mean"])[:, None]) / np.array(WAVE_PARAMS["std"])[:, None]
    return x.T.astype(np.float32)


def machine_tabular(con):
    """Per study: ventricular rate, PR, QRS, QTc (Bazett) from MIMIC machine measurements; atrial rate not reported."""
    return con.execute(f"""
    WITH m AS (SELECT CAST(study_id AS BIGINT) study_id, TRY_CAST(rr_interval AS INT) rr, TRY_CAST(p_onset AS INT) p,
                      TRY_CAST(qrs_onset AS INT) qa, TRY_CAST(qrs_end AS INT) qb, TRY_CAST(t_end AS INT) te
               FROM read_csv_auto('{ECG}/machine_measurements.csv', all_varchar=true)),
    v AS (SELECT study_id,
            CASE WHEN rr BETWEEN 200 AND 3000 THEN rr END AS rr,
            CASE WHEN p < {MISSING} AND qa < {MISSING} AND qa - p BETWEEN 40 AND 600 THEN qa - p END AS pr_interval,
            CASE WHEN qa < {MISSING} AND qb < {MISSING} AND qb - qa BETWEEN 20 AND 300 THEN qb - qa END AS qrs_duration,
            CASE WHEN qa < {MISSING} AND te < {MISSING} AND te - qa BETWEEN 150 AND 800 THEN te - qa END AS qt
          FROM m)
    SELECT study_id, 60000.0 / rr AS ventricular_rate, NULL::DOUBLE AS atrial_rate, pr_interval, qrs_duration,
           qt / sqrt(rr / 1000.0) AS qt_corrected
    FROM v
    """).df()


def preprocess_many(paths, workers):
    with Pool(workers) as pool:
        return pool.map(mimic_waveform, paths, chunksize=64)


def score_frame(df, model, device, workers, chunk=8000):
    """df needs path + tabular columns; returns df with logit_lvef, dropping unreadable waveforms."""
    out = []
    for i in range(0, len(df), chunk):
        part = df.iloc[i:i + chunk].copy()
        waves = preprocess_many(part.path.tolist(), workers)
        ok = np.array([w is not None for w in waves])
        part = part[ok]
        wave = np.stack([w for w in waves if w is not None])[:, None, :, :]
        part["logit_lvef"] = logits(model, wave, tabular(part), device)[:, LVEF]
        out.append(part)
        print(f"  scored {min(i + chunk, len(df))}/{len(df)}", flush=True)
    return pd.concat(out, ignore_index=True)


# --- check -------------------------------------------------------------------------------
def check(args, device):
    model = load_model(device)
    meta = pd.read_csv(ECHONEXT / "echonext_metadata_100k.csv")
    test = meta[meta.split == "test"].reset_index(drop=True)
    wave = np.load(ECHONEXT / "EchoNext_test_waveforms.npy", mmap_mode="r")
    tab = np.load(ECHONEXT / "EchoNext_test_tabular_features.npy")
    lg = logits(model, wave, tab, device)
    auroc = {lab: round(roc_auc_score(test[col], lg[:, k]), 4) for k, (lab, col) in enumerate(zip(LABELS, LABEL_COLS))}

    # Our tabular transform on the raw test columns vs the released preprocessed features.
    raw = test.assign(male=(test.sex.str.lower() == "male").astype(int))
    ours = tabular(raw)
    tab_diff = np.abs(ours - tab).max(axis=0).round(6).tolist()

    # MIMIC scale: per-lead SD after baseline removal and clipping, against the training SD.
    rl = pd.read_csv(ECG / "record_list.csv", usecols=["path"]).sample(2000, random_state=2)
    with Pool(args.workers) as pool:
        raw_w = [w for w in pool.starmap(mimic_waveform, [(p, False) for p in rl.path]) if w is not None and not np.isnan(w).any()]
    x = np.stack(raw_w)  # (n, 2500, 12)
    x = np.clip(x, np.array(WAVE_PARAMS["lowerbound"]), np.array(WAVE_PARAMS["upperbound"]))
    sd_ratio = (x.std(axis=(0, 1)) / np.array(WAVE_PARAMS["std"])).round(3).tolist()

    res = {"run_date": str(date.today()), "device": str(device),
           "echonext_test": {"n": len(test), "auroc": auroc, "published_auroc_ref10": {"lvef_lte_45": 0.852, "shd": 0.820}},
           "tabular_max_abs_diff_vs_released": dict(zip(["male"] + TAB_PARAMS["features"], tab_diff)),
           "mimic_scale": {"n_ecgs": len(raw_w), "lead_order": LEAD_ORDER, "sd_ratio_to_training": sd_ratio}}
    (RESULTS / "wp2_check.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


# --- score -------------------------------------------------------------------------------
def score(args, device):
    model = load_model(device)
    con = duckdb.connect()
    mt = machine_tabular(con)
    pat = con.execute(f"SELECT subject_id, gender = 'M' AS male, anchor_age, anchor_year FROM read_csv_auto('{HOSP}/patients.csv.gz')").df()

    def add_tabular(df, age):
        df = df.merge(mt, on="study_id", how="left").merge(pat[["subject_id", "male"]], on="subject_id", how="left")
        df["male"] = df.male.astype(int)
        df["age_at_ecg"] = age(df)
        # EchoNext drops ECGs whose measurements are all 0/NaN (atrial rate is never reported in MIMIC).
        allmiss = df[["ventricular_rate", "pr_interval", "qrs_duration", "qt_corrected"]].isna().all(axis=1)
        return df[~allmiss], int(allmiss.sum())

    # Cohort ECGs. Age input frozen at the patient's first eligible ECG (concept §6.1).
    ecgs = pd.read_parquet(WORK / f"wp1_ecgs_{args.tag}.parquet")
    ecgs = ecgs.merge(pat, on="subject_id")
    ecgs["age_at_ecg_real"] = ecgs.anchor_age + ecgs.ecg_time.dt.year - ecgs.anchor_year
    ecgs = ecgs.sort_values("ecg_time")
    ecgs["age_frozen"] = ecgs.groupby(["cohort", "grp", "subject_id"]).age_at_ecg_real.transform("first")
    ecgs = ecgs.drop(columns=["male", "anchor_age", "anchor_year"])
    ecgs, n_allmiss_cohort = add_tabular(ecgs, lambda d: d.age_frozen)
    print(f"cohort ECGs: {len(ecgs)} (dropped {n_allmiss_cohort} with no measurements)", flush=True)
    sc = score_frame(ecgs, model, device, args.workers)
    sc.to_parquet(WORK / f"wp2_scores_{args.tag}.parquet", index=False)

    # G2 sample: one random sinus, non-paced, quality-checked ECG per adult patient (concept §7).
    stm = con.execute(f"""
    WITH l AS (SELECT study_id, lower(trim(unnest([{", ".join(f"report_{i}" for i in range(18))}]))) s
               FROM read_csv_auto('{ECG}/machine_measurements.csv', all_varchar=true))
    SELECT CAST(study_id AS BIGINT) study_id
    FROM l WHERE s IS NOT NULL AND s <> '' GROUP BY 1
    HAVING bool_or(regexp_matches(s, '{COHORT.SINUS}')) AND NOT bool_or(regexp_matches(s, '{COHORT.NON_SINUS}'))
       AND NOT bool_or(regexp_matches(s, '{COHORT.BAD_QUALITY}'))
    """).df()
    rl = pd.read_csv(ECG / "record_list.csv", parse_dates=["ecg_time"]).merge(stm, on="study_id").merge(pat, on="subject_id")
    rl["age_at_ecg_real"] = rl.anchor_age + rl.ecg_time.dt.year - rl.anchor_year
    rl = rl[rl.age_at_ecg_real >= 18]
    g2 = rl.groupby("subject_id", group_keys=False).sample(1, random_state=20261030).drop(columns=["male", "anchor_age", "anchor_year"])
    with Pool(args.workers) as pool:
        qc = dict(pool.map(COHORT.waveform_ok, g2.path.tolist(), chunksize=200))
    n_qc_fail = int((~g2.path.map(qc)).sum())
    g2 = g2[g2.path.map(qc)]
    g2, n_allmiss_g2 = add_tabular(g2, lambda d: d.age_at_ecg_real)
    print(f"G2 ECGs: {len(g2)} (waveform check failed {n_qc_fail}, no measurements {n_allmiss_g2})", flush=True)
    g2 = score_frame(g2, model, device, args.workers)

    # Label: prevalent LVSD code at an admission on or before the ECG date.
    con.register("g2_df", g2[["subject_id", "study_id", "ecg_time"]])
    con.execute(f"""
    CREATE TABLE lv AS
    SELECT DISTINCT CAST(d.subject_id AS BIGINT) subject_id, CAST(a.admittime AS DATE) admit_date
    FROM read_csv_auto('{HOSP}/diagnoses_icd.csv.gz', all_varchar=true) d
    JOIN read_csv_auto('{HOSP}/admissions.csv.gz') a ON CAST(d.hadm_id AS BIGINT) = a.hadm_id
    WHERE (d.icd_version = '10' AND regexp_matches(d.icd_code, '^I50[24]'))
       OR (d.icd_version = '9' AND regexp_matches(d.icd_code, '^428[24]'))
    """)
    lab = con.execute("""
    SELECT g.study_id, EXISTS (SELECT 1 FROM lv WHERE lv.subject_id = g.subject_id
                               AND lv.admit_date <= CAST(g.ecg_time AS DATE)) AS prevalent_lvsd
    FROM g2_df g
    """).df()
    g2 = g2.merge(lab, on="study_id")
    g2[["subject_id", "study_id", "ecg_time", "age_at_ecg_real", "logit_lvef", "prevalent_lvsd"]].to_parquet(
        WORK / "wp2_g2_scores.parquet", index=False)

    y, s = g2.prevalent_lvsd.to_numpy(), g2.logit_lvef.to_numpy()
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(1)
    boot = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    d_g2 = float(s[y].mean() - s[~y].mean())
    res = {
        "run_date": str(date.today()), "tag": args.tag,
        "cohort_scoring": {"ecgs_in": int(len(ecgs) + n_allmiss_cohort), "dropped_no_measurements": n_allmiss_cohort,
                           "scored": int(len(sc)), "unreadable": int(len(ecgs) - len(sc))},
        "G2": {"n": int(len(y)), "prevalent_lvsd": int(y.sum()), "dropped_waveform_check": n_qc_fail,
               "dropped_no_measurements": n_allmiss_g2, "auroc": round(auc, 4),
               "auroc_95ci_bootstrap": [round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4)],
               "threshold": 0.70, "pass": bool(auc >= 0.70)},
        "D_G2_logit": round(d_g2, 4),
        "delta_logit_per_year": round(0.10 * d_g2 / 3, 4),
    }
    (RESULTS / "wp2_g2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def _load_cohort_module():
    spec = importlib.util.spec_from_file_location("cohort", PROJ / "analysis/01_cohort.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cohort"] = mod  # so worker processes can unpickle cohort.waveform_ok
    spec.loader.exec_module(mod)
    return mod


COHORT = _load_cohort_module()  # shares the rhythm/quality rules and the waveform check with WP1

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["check", "score"])
    ap.add_argument("--tag", default="5y_limbfx", help="WP1 output tag (window and control)")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    {"check": check, "score": score}[args.step](args, device)
