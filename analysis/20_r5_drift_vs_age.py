"""Exploratory R5 (concept.md §6.9): how much of the within-patient drift in event-free patients is ageing?

    python3 -I analysis/20_r5_drift_vs_age.py

In the REF group (work/r3_ref_{mimic,heedb}.parquet), for both models:
- within-patient drift: fixed-effects slope of the score on time (years), patient-clustered SE;
- cross-sectional age gradient: one ECG per patient (the first eligible), OLS of the score on age at that ECG
  plus sex (per year of age);
- ratio within / cross-sectional, patient bootstrap 95% CI (500 resamples).
EchoNext-Mini: the contribution of the model's age input (final-layer weight x scaled frozen age) is removed
first, because it is constant within a patient but differs between patients.
Output: results/robustness_r5.json (aggregate only).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SCORE = _mod("score", PROJ / "analysis/02_score.py")


def within_slope(d):
    d = d[d.groupby("pid").pid.transform("size") >= 2]
    yd = d.y - d.groupby("pid").y.transform("mean")
    td = d.t - d.groupby("pid").t.transform("mean")
    r = sm.OLS(yd, td.to_frame("t")).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.pid)[0]})
    return float(r.params["t"]), r.conf_int().loc["t"].tolist()


def cross_slope(d):
    f = d.sort_values("t").groupby("pid").head(1)
    X = sm.add_constant(f[["age_ecg", "male"]].astype(float))
    r = sm.OLS(f.y, X).fit(cov_type="HC1")
    return float(r.params["age_ecg"]), r.conf_int().loc["age_ecg"].tolist()


def boot_ratio(d, n=500, seed=1):
    """Patient bootstrap with per-patient sufficient statistics (resampling patients = multinomial patient weights)."""
    d = d[d.groupby("pid").pid.transform("size") >= 2]
    td = d.t - d.groupby("pid").t.transform("mean")
    yd = d.y - d.groupby("pid").y.transform("mean")
    ab = pd.DataFrame({"pid": d.pid, "a": td * yd, "b": td * td}).groupby("pid")[["a", "b"]].sum()
    f = d.sort_values("t").groupby("pid").head(1).set_index("pid").loc[ab.index]
    X = np.column_stack([np.ones(len(f)), f.age_ecg.to_numpy(float), f.male.to_numpy(float)])
    y = f.y.to_numpy(float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = rng.multinomial(len(ab), np.full(len(ab), 1 / len(ab))).astype(float)
        within = (c * ab.a.to_numpy()).sum() / (c * ab.b.to_numpy()).sum()
        sw = np.sqrt(c)
        cross = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0][1]
        out.append(within / cross)
    return [round(float(np.quantile(out, q)), 3) for q in (0.025, 0.975)]


def main():
    w = torch.load(SCORE.WEIGHTS, map_location="cpu", weights_only=True)["model"]["output.weight"].numpy()
    w_age = float(w[SCORE.LVEF, -7:][1])  # tabular order: male, age, ...
    mu, sc = SCORE.TAB_PARAMS["scale_mean"][0], SCORE.TAB_PARAMS["scale_scale"][0]

    m = pd.read_parquet(WORK / "r3_ref_mimic.parquet")
    m = m.assign(pid=m.subject_id.astype(str), t=m.t_years, age_ecg=m.anchor_age + m.ecg_time.dt.year - m.anchor_year, male=m.male.astype(int))
    h = pd.read_parquet(WORK / "r3_ref_heedb.parquet")
    h = h.assign(pid=h.site + ":" + h.pid.astype(str), t=h.t_years, age_ecg=h.age_at_ecg_real, male=h.male.astype(int))

    out = {"run_date": str(date.today()), "exploratory": True, "echonext_age_input_weight": round(w_age, 4)}
    for src, d in (("mimic", m), ("heedb", h)):
        for model, y in (("echonext", d.logit_lvef - w_age * (d.age_at_ecg - mu) / sc), ("deepecg", d.score_d)):
            x = d.assign(y=y)
            ws, wci = within_slope(x)
            cs, cci = cross_slope(x)
            out[f"{model}_{src}"] = {"patients": int(x.pid.nunique()), "within_drift_per_year": round(ws, 4), "within_ci95": [round(v, 4) for v in wci],
                                    "cross_sectional_per_year_of_age": round(cs, 4), "cross_ci95": [round(v, 4) for v in cci],
                                    "ratio_within_to_cross": round(ws / cs, 3), "ratio_ci95_bootstrap": boot_ratio(x)}
            print(model, src, out[f"{model}_{src}"], flush=True)
    (RESULTS / "robustness_r5.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
