"""WP3-H — HEEDB replication analysis (concept.md §6.7 items 8, 11, 12), with the functions of analysis/03_analysis.py.

    python3 -I analysis/13_heedb_analysis.py

Input: work/wp_h_scores.parquet, work/wp_h_anchors.parquet; delta from results/wp2_g2.json (MIMIC).
Output: results/wp_h_results.json (aggregate only).
"""
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS = PROJ / "work", PROJ / "results"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


AN = _mod("analysis3", PROJ / "analysis/03_analysis.py")


def load():
    s = pd.read_parquet(WORK / "wp_h_scores.parquet")
    a = pd.read_parquet(WORK / "wp_h_anchors.parquet")[["site", "grp", "pid", "age", "male"]]
    d = s.drop(columns=["male"]).merge(a.rename(columns={"male": "male_anchor"}), on=["site", "grp", "pid"])
    d["g"] = d.grp.replace({"LIMBFX": "CTRL"})
    d["entity"] = d.site + ":" + d.grp + ":" + d.pid.astype(str)
    d["t"] = d.t_years
    m = d.t * 12
    d["bin"] = pd.cut(m, [b[0] for b in AN.BINS] + [np.inf], labels=[b[2] for b in AN.BINS], right=False).astype(str)
    return d


def block(df, delta, weights=None):
    r, u = AN.fit(df, AN.slope_terms(df), weights=weights)
    b = AN.slope_block(r, u, delta)
    b["readout_rule"], b["reading"] = AN.readout(b)
    return b


def main():
    delta = json.loads((RESULTS / "wp2_g2.json").read_text())["delta_logit_per_year"]
    mimic_rule = json.loads((RESULTS / "wp3_results.json").read_text())["readout"]["rule"]
    d = load()
    primary = block(d, delta)
    out = {"run_date": str(date.today()), "delta_from_mimic": delta, "primary": primary,
           "replication": {"mimic_rule": mimic_rule, "heedb_rule": primary["readout_rule"],
                           "agrees": primary["readout_rule"] == mimic_rule},
           "event_study": AN.event_study(d), "level": AN.level(d),
           "sensitivity": {"S2H_no_conduction_statement": block(d[~d.conduction], delta),
                           "S3_equal_patient_weight": block(d, delta, weights=1.0 / d.groupby("entity").entity.transform("size"))},
           "by_site": {s: block(d[d.site == s], delta) for s in sorted(d.site.unique())}}
    (RESULTS / "wp_h_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
