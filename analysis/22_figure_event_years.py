"""Main figure by years before the event: the pre-specified event studies for both models in both data sources (event
groups), with the event-free reference group anchored at a random date (REF-D, R6) from its own model. Bins as pre-specified (§6.5).

    python3 -I analysis/22_figure_event_years.py

-> results/fig_event_years.json (estimates), figures/fig_event_years.{pdf,png}, figures/fig_event_years_data.csv
Each panel: within-patient change against the same patients' ECGs 5 to 3 years before the anchor, from a
patient fixed-effects model with bin x group terms (plus bin x age and bin x sex). DeepECG-SL is drawn as the
change in predicted LVEF (points; down = worse); EchoNext-Mini as the change in its LVEF <=45% logit (up = worse).
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
WORK, RESULTS, OUT = PROJ / "work", PROJ / "results", PROJ / "figures"


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


R6 = _mod("r6", PROJ / "analysis/21_r6_random_date.py")
AN, H, D16 = R6.AN, R6.H, R6.D16
G4 = ["LVSD", "AF", "CTRL", "REFD"]
BIN_MID_Y = {"m60_36": -4.0, "m36_24": -2.5, "m24_12": -1.5, "m12_6": -0.75, "m6_1": -0.29}

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"LVSD": ("LVSD", "#2a78d6", "o"), "AF": ("AF", "#eb6834", "s"),
          "CTRL": ("Limb fracture", "#1baf7a", "^"), "REFD": ("No event (random date)", "#6b6a66", "D")}  # darker than MUTED: CVD-separable from aqua (dataviz validator)
DODGE = {"LVSD": -0.06, "AF": -0.02, "CTRL": 0.02, "REFD": 0.06}
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "xtick.major.size": 0,
                     "ytick.major.size": 0, "savefig.facecolor": "white"})


def event_study_ref(df):
    """Event study for the no-event group alone (bins, bin x age, bin x sex), on its own patient fixed effects."""
    df = df[df.g == "REFD"]
    age_c = df.age - df.drop_duplicates("entity").age.mean()
    bins = [b for b in BIN_MID_Y if b != AN.REF_BIN]
    terms = {}
    for b in bins:
        ind = (df.bin == b).astype(float)
        terms[f"{b}_REFD"], terms[f"{b}_x_age_c"], terms[f"{b}_x_male"] = ind, ind * age_c, ind * df.male_anchor.astype(float)
    res, used = AN.fit(df, terms)
    out = {AN.REF_BIN: {"est": 0.0, "ci95": [0.0, 0.0]}}
    for b in bins:
        e, lo, hi = AN.contrast(res, {f"{b}_REFD": 1}, 0.05)
        out[b] = {"est": round(e, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    return out, int(used.entity.nunique())


def prespecified(key):
    """Event-group estimates from the pre-specified event studies (the values the text reports)."""
    src = {"echonext_mimic": ("wp3_results.json", lambda r: r["event_study"]), "echonext_heedb": ("wp_h_results.json", lambda r: r["event_study"]),
           "deepecg_mimic": ("wp_d_results.json", lambda r: r["mimic"]["event_study"]), "deepecg_heedb": ("wp_d_results.json", lambda r: r["heedb"]["event_study"])}[key]
    es = src[1](json.loads((RESULTS / src[0]).read_text()))
    out = {}
    for g in ["LVSD", "AF", "CTRL"]:
        out[g] = {AN.REF_BIN: {"est": 0.0, "ci95": [0.0, 0.0]}}
        out[g].update({b: es["theta"][b][g] for b in BIN_MID_Y if b != AN.REF_BIN})
    return out


def event_study4(df, key):
    out = prespecified(key)
    out["REFD"], n_ref = event_study_ref(df)
    out["patients"] = {"REFD": n_ref}
    return out


def data():
    def rows(w, keycols):
        r = w.assign(g="REFD", grp="REFD", t=w.t_years, male_anchor=w.male.astype(bool))
        r["entity"] = "REFD:" + r[keycols].astype(str).agg(":".join, axis=1)
        r["bin"] = pd.cut(r.t * 12, [b[0] for b in AN.BINS] + [np.inf], labels=[b[2] for b in AN.BINS], right=False).astype(str)
        return r
    cols = ["entity", "g", "t", "bin", "age", "male_anchor", "logit_lvef"]
    dm, _ = AN.load("5y_limbfx")
    dm = AN.analysis_set(dm, "primary")
    dmd = D16.with_score(dm, pd.read_parquet(WORK / "wp_d_scores_mimic.parquet"), ["cohort", "grp", "subject_id", "study_id"])
    dh = H.load()
    dhd = D16.with_score(dh, pd.read_parquet(WORK / "wp_d_scores_heedb.parquet"), ["site", "grp", "pid", "rel"])
    rm = rows(pd.read_parquet(WORK / "r6_refd_mimic.parquet"), ["subject_id"])
    rh = rows(pd.read_parquet(WORK / "r6_refd_heedb.parquet"), ["site", "pid"])
    return {("echonext", "mimic"): pd.concat([dm[cols], rm[cols]]), ("echonext", "heedb"): pd.concat([dh[cols], rh[cols]]),
            ("deepecg", "mimic"): pd.concat([dmd[cols], rm.assign(logit_lvef=rm.score_d)[cols]]),
            ("deepecg", "heedb"): pd.concat([dhd[cols], rh.assign(logit_lvef=rh.score_d)[cols]])}


def panel(ax, est, model, title, rows):
    sign = -1 if model == "deepecg" else 1  # DeepECG score = -LVEF: plot predicted LVEF change
    for g, (label, color, marker) in SERIES.items():
        xs, ys, lo, hi = [], [], [], []
        for b, x in BIN_MID_Y.items():
            v = est[g][b]
            e, (l, h) = sign * v["est"], sorted(sign * np.array(v["ci95"]))
            xs.append(x + DODGE[g]); ys.append(e); lo.append(e - l); hi.append(h - e)
            rows.append({"model": model, "panel": title, "group": g, "bin": b, "years_mid": x, "est": e, "ci95_lo": l, "ci95_hi": h})
        z = 3 if g == "LVSD" else 2
        ax.plot(xs, ys, color=color, lw=1.5 if g != "REFD" else 1.2, zorder=z)
        ax.errorbar(xs[1:], ys[1:], yerr=[lo[1:], hi[1:]], fmt="none", ecolor=color, elinewidth=0.9, capsize=0, zorder=z)
        ax.plot(xs, ys, ls="none", marker=marker, ms=5 if g != "REFD" else 4.5, mfc=color, mec="white", mew=1.0, zorder=z + 1, label=label)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-5, 0)
    ax.set_xticks([-5, -4, -3, -2, -1, 0])
    ax.set_title(title, fontsize=8, color=INK, loc="left")


def main():
    OUT.mkdir(exist_ok=True)
    ests = {f"{m}_{s}": event_study4(df, f"{m}_{s}") for (m, s), df in data().items()}
    (RESULTS / "fig_event_years.json").write_text(json.dumps(ests, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True, gridspec_kw={"hspace": 0.35, "wspace": 0.28})
    rows = []
    titles = {"mimic": "MIMIC-IV (BIDMC)", "heedb": "HEEDB (MGH + Emory)"}
    for i, model in enumerate(["echonext", "deepecg"]):
        for j, src in enumerate(["mimic", "heedb"]):
            ax = axes[i, j]
            name = {"echonext": "EchoNext-Mini", "deepecg": "DeepECG-SL"}[model]
            panel(ax, ests[f"{model}_{src}"], model, f"{name} · {titles[src]}", rows)
        axes[i, 0].set_ylabel("Score change (logit)" if model == "echonext" else "Predicted LVEF change (points)")
    for ax in axes[1]:
        ax.set_xlabel("Years before the event")
    for i, model in enumerate(["echonext", "deepecg"]):  # same y-range across datasets for each model
        lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo, hi)
    for k, lbl in enumerate("ABCD"):
        axes.flat[k].text(-0.14, 1.06, lbl, transform=axes.flat[k].transAxes, fontsize=10, fontweight="bold", color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.savefig(OUT / "fig_event_years.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_event_years.png", dpi=300, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT / "fig_event_years_data.csv", index=False)
    print("written; patients:", {k: v["patients"] for k, v in ests.items()})


if __name__ == "__main__":
    main()
