"""EchoNext-Mini tabular inputs from the waveform, for datasets without machine measurements (HEEDB).

ventricular_rate(sig, fs, names): 60000 / median RR, R peaks from lead II (neurokit2 ecg_clean + ecg_peaks);
    RR outside 200-3000 ms is missing, as for the MIMIC machine measurements. Used in HEEDB (concept §6.7).
delineated_intervals(sig, fs, names, agg): PR, QRS and QTc from neurokit2 DWT delineation of leads II, V2
    and V5 ("II" = lead II only, "median" or "minmax" across leads per beat). Not used in any analysis:
    checks/02_interval_agreement.py shows it does not agree with the machine measurements.
"""
import warnings

import neurokit2 as nk
import numpy as np

LEADS = ["II", "V2", "V5"]
RANGES = {"rr": (200, 3000), "pr_interval": (40, 600), "qrs_duration": (20, 300), "qt": (150, 800)}


def _in(v, key):
    lo, hi = RANGES[key]
    return v if v is not None and np.isfinite(v) and lo <= v <= hi else np.nan


def _rpeaks(x, fs):
    _, info = nk.ecg_peaks(x, sampling_rate=fs)
    return np.asarray(info["ECG_R_Peaks"])


def ventricular_rate(sig, fs, names):
    """sig (n_samples, n_leads) in mV. Returns bpm or NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            r = _rpeaks(nk.ecg_clean(sig[:, list(names).index("II")], sampling_rate=fs), fs)
        except Exception:
            return np.nan
    if len(r) < 3:
        return np.nan
    rr = _in(float(np.median(np.diff(r))) * 1000 / fs, "rr")
    return 60000 / rr if np.isfinite(rr) else np.nan


def delineated_intervals(sig, fs, names, agg="II"):
    out = dict(pr_interval=np.nan, qrs_duration=np.nan, qt_corrected=np.nan)
    idx = {n: i for i, n in enumerate(names)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clean = {k: nk.ecg_clean(sig[:, idx[k]], sampling_rate=fs) for k in LEADS}
            r = _rpeaks(clean["II"], fs)
            rr = float(np.median(np.diff(r))) * 1000 / fs
            ks = ["II"] if agg == "II" else LEADS
            w = {k: nk.ecg_delineate(clean[k], r, sampling_rate=fs, method="dwt")[1] for k in ks}
            stack = lambda key: np.vstack([np.asarray(w[k][key], float) for k in ks])
            lo = {"II": lambda a: a[0], "median": lambda a: np.nanmedian(a, axis=0), "minmax": lambda a: np.nanmin(a, axis=0)}[agg]
            hi = {"II": lambda a: a[0], "median": lambda a: np.nanmedian(a, axis=0), "minmax": lambda a: np.nanmax(a, axis=0)}[agg]
            p_on, q_on = lo(stack("ECG_P_Onsets")), lo(stack("ECG_R_Onsets"))
            q_off, t_off = hi(stack("ECG_R_Offsets")), hi(stack("ECG_T_Offsets"))
            ms = 1000 / fs
            out["pr_interval"] = _in(float(np.nanmedian((q_on - p_on) * ms)), "pr_interval")
            out["qrs_duration"] = _in(float(np.nanmedian((q_off - q_on) * ms)), "qrs_duration")
            qt = _in(float(np.nanmedian((t_off - q_on) * ms)), "qt")
            out["qt_corrected"] = qt / np.sqrt(rr / 1000) if np.isfinite(qt) else np.nan
        except Exception:
            pass
    return out
