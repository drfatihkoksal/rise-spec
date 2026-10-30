"""Bridge check for the HEEDB replication (concept §6.7, item 7).

    python3 -I analysis/05_bridge.py

Recomputes the MIMIC cohort logits with PR, QRS and QTc held at the training medians, as they will be in
HEEDB. The tabular inputs enter the final linear layer, so the change is exact:
    logit_bridge = logit - sum_k w_k * (z_k - c_k),  k in {PR, QRS, QTc}, c_k = training median (scaled)
Then refits the primary model, S2 and S3 of analysis/03_analysis.py on the bridged logits.
-> results/wp_h_bridge.json
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[1]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


SCORE = _mod("score", PROJ / "analysis/02_score.py")
AN = _mod("analysis3", PROJ / "analysis/03_analysis.py")
HELD = {"pr_interval": 4, "qrs_duration": 5, "qt_corrected": 6}  # positions in the 7-vector (male first)


def main():
    w = torch.load(SCORE.WEIGHTS, map_location="cpu", weights_only=True)["model"]["output.weight"].numpy()
    w_tab = w[SCORE.LVEF, -7:]
    const = dict(zip(SCORE.TAB_PARAMS["features"], SCORE.TAB_PARAMS["impute_statistics_scaled"]))
    delta = json.loads((AN.RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    d, control = AN.load("5y_limbfx")
    z = SCORE.tabular(d)
    shift = sum(w_tab[i] * (z[:, i] - const[name]) for name, i in HELD.items())
    d["logit_lvef"] = d.logit_lvef - shift

    out = {"run_date": str(date.today()), "held_constant": list(HELD), "w_lvef_tabular": dict(
        zip(["male"] + SCORE.TAB_PARAMS["features"], np.round(w_tab, 4).tolist())),
        "logit_shift_sd": round(float(np.std(shift)), 4), "delta": delta}
    prim = AN.analysis_set(d, "primary")
    res, used = AN.fit(prim, AN.slope_terms(prim))
    blk = AN.slope_block(res, used, delta)
    blk["readout_rule"] = AN.readout(blk)[0]
    out["primary"] = blk
    s2 = prim[prim.qrs_ms < 120]
    r, u = AN.fit(s2, AN.slope_terms(s2))
    out["S2_qrs_lt120"] = AN.slope_block(r, u, delta) | {"readout_rule": AN.readout(AN.slope_block(r, u, delta))[0]}
    wts = 1.0 / prim.groupby("entity").entity.transform("size")
    r, u = AN.fit(prim, AN.slope_terms(prim), weights=wts)
    out["S3_equal_patient_weight"] = AN.slope_block(r, u, delta) | {"readout_rule": AN.readout(AN.slope_block(r, u, delta))[0]}
    mimic = json.loads((AN.RESULTS / "wp3_results.json").read_text())
    out["mimic_primary_rule"] = mimic["readout"]["rule"]
    out["bridge_agrees"] = out["primary"]["readout_rule"] == mimic["readout"]["rule"]
    (AN.RESULTS / "wp_h_bridge.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
