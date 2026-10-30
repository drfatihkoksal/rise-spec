"""WP-D — scoring with the second model, DeepECG-SL LVEF regression (concept.md §6.8). No slope is estimated here.

    python3 -I analysis/15_deepecg_score.py

Preprocessing is a port of DeepECG_Docker c81d677 utils/ecg_signal_processor.py (scale_ecg_signals,
detect_peaks_with_sliding_window, find_crossings_for_peaks, widen_ranges, flatten_fft_peak) and the
EfficientNetWrapper input factor. The cohort-level parameters (spectral scale and line-noise frequency ranges)
are fitted once per data source on 10,000 random ECGs of its G2 sample, then frozen.

Outputs: work/wp_d_scores_mimic.parquet, work/wp_d_scores_heedb.parquet, work/wp_d_g2_{mimic,heedb}.parquet
(patient-level, local only); results/wp_d_g2.json (aggregate: frozen parameters, G2-D, plausibility, delta).
"""
import importlib.util
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from sklearn.metrics import roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"
MIMIC_ECG = Path("/media/fatih/48bd54fc-58ee-46af-8646-96be97c8d33c/mimic-iv-ecg/1.0")
HEEDB = Path.home() / "datasets/heedb"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS = 250
PTBXL_POWER_RATIO = 3.003154  # DeepECG_Docker utils/constants.py
MHI_FACTOR = 1 / 0.0048       # DeepECG_Docker models/efficientnet_wrapper.py
FIT_N, SEED = 10_000, 20261030


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# --- ported preprocessing ------------------------------------------------------------------
def mean_spectrum(signals):
    signals = signals.astype(np.float32)
    return np.fft.fftfreq(signals.shape[1], 1 / FS), np.mean(np.abs(np.fft.fft(signals, axis=1)), axis=0)


def detect_peaks(signals, window_size=5, std_threshold=5, min_freq=20, merge_threshold=1):
    fft_freq, mag = mean_spectrum(signals)
    valid = fft_freq >= min_freq
    vf, vm = fft_freq[valid], mag[valid]
    wpts = int(window_size / (fft_freq[1] - fft_freq[0]))
    peaks, harmonics = [], []
    for i in range(len(vm) - wpts + 1):
        win = vm[i:i + wpts]
        mu, sd = np.mean(win), np.std(win)
        for j in range(wpts):
            f = vf[i + j]
            thr = std_threshold * (0.65 if any(np.isclose(f, h * r, rtol=0.05) for h in harmonics for r in [2, 3, 4]) else 1)
            if win[j] > mu + thr * sd:
                peaks.append(i + j)
                harmonics.append(f)
    if not peaks:
        return []
    pf = vf[peaks]
    regions, cur = [], [peaks[0]]
    for i in range(1, len(peaks)):
        if pf[i] - pf[i - 1] <= merge_threshold:
            cur.append(peaks[i])
        else:
            regions.append(cur)
            cur = [peaks[i]]
    regions.append(cur)
    return [(float(vf[r[0]]), float(vf[r[-1]])) for r in regions]


def find_crossings(signals, peak_ranges, frac=0.1, kernel_size=5):
    fft_freq, mag = mean_spectrum(signals)
    valid = fft_freq >= 30
    vf, vm = fft_freq[valid], mag[valid]
    sm = lowess(medfilt(vm, kernel_size=kernel_size), vf, frac=frac, return_sorted=False)
    out = []
    for a, b in peak_ranges:
        mid = np.argmin(np.abs(vf - (a + b) / 2))
        up = next((vf[i] for i in range(mid, 0, -1) if (vm[i - 1] > sm[i - 1] and vm[i] < sm[i]) or (vm[i - 1] < sm[i - 1] and vm[i] > sm[i])), None)
        dn = next((vf[i] for i in range(mid, len(vf) - 1) if (vm[i] > sm[i] and vm[i + 1] < sm[i + 1]) or (vm[i] < sm[i] and vm[i + 1] > sm[i + 1])), None)
        out.append((float(up if up is not None else a), float(dn if dn is not None else b)))
    return out


def flatten_fft_peak(signal, ranges):
    if np.isnan(signal).any():
        signal = np.nan_to_num(signal)
    fr = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / FS)
    n = len(signal) // 2
    phase, magn = np.angle(fr), np.abs(fr)
    for lo, hi in set(ranges):
        if lo >= hi:
            continue
        s = max(np.searchsorted(freqs[:n], lo, side="left"), 0)
        e = min(np.searchsorted(freqs[:n], hi, side="right") - 1, n - 1)
        if s != e:
            idx = np.arange(s, e + 1)
            interp = interp1d([freqs[s], freqs[e]], [magn[s], magn[e]], kind="linear", fill_value="extrapolate")(freqs[idx])
            fr[idx] = interp * np.exp(1j * phase[idx])
            if s > 0:
                fr[-idx] = interp * np.exp(1j * phase[-idx])
    return np.fft.ifft(fr).real


class Prep:
    """Frozen cohort-level parameters for one data source."""

    def fit(self, x):  # x (N, 2500, 12)
        self.scale = float(PTBXL_POWER_RATIO / np.mean(mean_spectrum(x[:, :, 0])[1]))
        xs = x * self.scale
        self.peaks = detect_peaks(xs[:, :, 0])
        self.ranges = [[(a - 1, b + 1) for a, b in find_crossings(xs[:, :, k], self.peaks)] for k in range(12)] if self.peaks else None
        return self

    def params(self):
        return {"scale": self.scale, "peak_ranges_hz": self.peaks, "flatten_ranges_hz_by_lead": self.ranges}

    def transform(self, x):  # x (2500, 12) -> (12, 2500) model input
        x = x * self.scale
        if self.ranges:
            x = np.stack([flatten_fft_peak(x[:, k].astype(np.float32), self.ranges[k]) for k in range(12)], axis=1)
        return (x * MHI_FACTOR).T.astype(np.float32)


# --- reading -------------------------------------------------------------------------------
def read(arg):
    """('mimic', path) or (site, rel) -> (2500, 12) float32 at 250 Hz in lead order, or None."""
    src, rel = arg
    try:
        rec = wfdb.rdrecord(str(MIMIC_ECG / rel) if src == "mimic" else str(HEEDB / "ECG" / src / "WFDB" / rel))
    except Exception:
        return None
    step = {500: 2, 250: 1}.get(rec.fs)
    if step is None:
        return None
    idx = {n: i for i, n in enumerate(rec.sig_name)}
    if any(k not in idx for k in LEADS):
        return None
    x = rec.p_signal[::step][:, [idx[k] for k in LEADS]]
    return x.astype(np.float32) if x.shape == (2500, 12) and not np.isnan(x).any() else None


def read_many(args, workers=24):
    with Pool(workers) as pool:
        return pool.map(read, args, chunksize=64)


_PREP = None


def _transform(x):
    return None if x is None else _PREP.transform(x)


def score(df, src_col, rel_col, preps, model, device, chunk=8000, workers=24):
    """Adds pred_lvef and score_d (= -pred_lvef); drops unreadable ECGs."""
    global _PREP
    out = []
    for src, part in df.groupby(src_col, sort=False):
        _PREP = preps[src]
        for i in range(0, len(part), chunk):
            p = part.iloc[i:i + chunk].copy()
            xs = read_many(list(zip(p[src_col], p[rel_col])), workers)
            with Pool(workers) as pool:
                xt = pool.map(_transform, xs, chunksize=64)
            ok = np.array([x is not None for x in xt])
            p = p[ok]
            batch = np.stack([x for x in xt if x is not None])
            preds = []
            with torch.no_grad():
                for j in range(0, len(batch), 512):
                    preds.append(model(torch.as_tensor(batch[j:j + 512], device=device)).squeeze(1).cpu().numpy())
            p["pred_lvef"] = np.concatenate(preds)
            p["score_d"] = -p.pred_lvef
            out.append(p)
            print(f"  {src}: scored {min(i + chunk, len(part))}/{len(part)}", flush=True)
    return pd.concat(out, ignore_index=True)


def g2_block(d):
    y, s = d.prevalent_lvsd.to_numpy(bool), d.score_d.to_numpy()
    rng = np.random.default_rng(1)
    boot = [roc_auc_score(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    q = np.quantile(d.pred_lvef, [0.25, 0.5, 0.75])
    return {"n": int(len(y)), "prevalent_lvsd": int(y.sum()), "auroc": round(roc_auc_score(y, s), 4),
            "auroc_95ci_bootstrap": [round(float(np.quantile(boot, a)), 4) for a in (0.025, 0.975)],
            "D_G2": round(float(s[y].mean() - s[~y].mean()), 4),
            "pred_lvef_quartiles": [round(float(v), 2) for v in q],
            "pred_lvef_median_lvsd_vs_other": [round(float(np.median(d.pred_lvef[y])), 2), round(float(np.median(d.pred_lvef[~y])), 2)],
            "plausible_median_45_70": bool(45 <= q[1] <= 70)}


def main():
    chk = _mod("deepecg_check", PROJ / "analysis/14_deepecg_check.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = chk.load_model(device)

    # G2 samples (existing, from WP2 and WP2-H) and the paths needed to read them.
    rl = pd.read_csv(MIMIC_ECG / "record_list.csv", usecols=["study_id", "path"])
    g2m = pd.read_parquet(WORK / "wp2_g2_scores.parquet").merge(rl, on="study_id").assign(src="mimic")
    g2h = pd.read_parquet(WORK / "wp_h_g2_scores.parquet").rename(columns={"site": "src"})

    preps, prep_info = {}, {}
    for src, d, col in [("mimic", g2m, "path"), ("I0001", g2h[g2h.src == "I0001"], "rel"), ("I0006", g2h[g2h.src == "I0006"], "rel")]:
        fit = d.sample(FIT_N, random_state=SEED)
        xs = [x for x in read_many([(src, r) for r in fit[col]]) if x is not None]
        preps[src] = Prep().fit(np.stack(xs))
        prep_info[src] = {"fit_ecgs": len(xs)} | preps[src].params()
        print(src, "prep:", prep_info[src]["scale"], prep_info[src]["peak_ranges_hz"], flush=True)

    g2m = score(g2m, "src", "path", preps, model, device)
    g2h = score(g2h, "src", "rel", preps, model, device)
    g2m[["subject_id", "study_id", "pred_lvef", "score_d", "prevalent_lvsd"]].to_parquet(WORK / "wp_d_g2_mimic.parquet", index=False)
    g2h[["src", "pid", "rel", "pred_lvef", "score_d", "prevalent_lvsd"]].to_parquet(WORK / "wp_d_g2_heedb.parquet", index=False)
    gm, gh = g2_block(g2m), g2_block(g2h)
    delta = round(0.10 * gm["D_G2"] / 3, 4)

    cm = pd.read_parquet(WORK / "wp2_scores_5y_limbfx.parquet")[["cohort", "grp", "subject_id", "study_id", "path"]].assign(src="mimic")
    n_cm = len(cm)
    cm = score(cm, "src", "path", preps, model, device)
    cm.drop(columns="src").to_parquet(WORK / "wp_d_scores_mimic.parquet", index=False)
    ch = pd.read_parquet(WORK / "wp_h_scores.parquet")[["site", "grp", "pid", "rel"]].assign(src=lambda d: d.site)
    n_ch = len(ch)
    ch = score(ch, "src", "rel", preps, model, device)
    ch.drop(columns="src").to_parquet(WORK / "wp_d_scores_heedb.parquet", index=False)

    res = {"run_date": str(date.today()), "preprocessing_frozen": prep_info,
           "G2_D": {"mimic": gm | {"pass": gm["auroc"] >= 0.70}, "heedb": gh | {"pass": gh["auroc"] >= 0.70,
                    "by_site": {s: g2_block(g2h[g2h.src == s]) for s in ["I0001", "I0006"]}}},
           "delta_D_points_per_year": delta,
           "cohort_scoring": {"mimic": {"in": n_cm, "scored": int(len(cm))}, "heedb": {"in": n_ch, "scored": int(len(ch))}}}
    (RESULTS / "wp_d_g2.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
