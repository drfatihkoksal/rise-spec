"""Figures from results/wp3_results.json (no numbers typed by hand).

    python3 -I analysis/04_figures.py

-> figures/fig_event_study.pdf / .png: (A) event study, change in the LVEF <=45% logit against the
   reference bin; (B) pre-event slopes and their contrasts with the equivalence margin.
-> figures/fig_event_study_data.csv: every plotted value (table view of the figure).

Palette: categorical slots 1-3 of the dataviz reference palette (validated all-pairs on white;
aqua is below 3:1 contrast, so identity also rides on marker shape and the legend).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJ = Path(__file__).resolve().parents[1]
RES = json.loads((PROJ / "results/wp3_results.json").read_text())
OUT = PROJ / "figures"

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
SERIES = {
    "LVSD": ("LVSD (target)", "#2a78d6", "o"),
    "AF": ("AF (cardiac comparator)", "#eb6834", "s"),
    "CTRL": ("Limb fracture (non-cardiac control)", "#1baf7a", "^"),
}
BINS = [("m60_36", "−60 to −36\n(reference)"), ("m36_24", "−36 to −24"), ("m24_12", "−24 to −12"),
        ("m12_6", "−12 to −6"), ("m6_1", "−6 to −1")]
DODGE = {"LVSD": -0.13, "AF": 0.0, "CTRL": 0.13}

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "xtick.major.size": 0, "ytick.major.size": 0, "savefig.facecolor": "white",
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def panel_label(ax, s):
    ax.text(-0.08, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, va="bottom")


def event_study(ax, rows):
    es = RES["event_study"]["theta"]
    for g, (label, color, marker) in SERIES.items():
        xs, ys, lo, hi = [], [], [], []
        for k, (b, _) in enumerate(BINS):
            if b == "m60_36":
                e, l, h = 0.0, 0.0, 0.0
            else:
                e, (l, h) = es[b][g]["est"], es[b][g]["ci95"]
            xs.append(k + DODGE[g]); ys.append(e); lo.append(e - l); hi.append(h - e)
            rows.append({"panel": "A", "group": g, "item": b, "est": e, "ci95_lo": l, "ci95_hi": h,
                         "ecgs": RES["event_study"]["ecgs_by_bin"][g][b]})
        ax.plot(xs, ys, color=color, lw=1.5, solid_capstyle="round", solid_joinstyle="round", zorder=2)
        ax.errorbar(xs[1:], ys[1:], yerr=[lo[1:], hi[1:]], fmt="none", ecolor=color, elinewidth=1.0, capsize=0, zorder=2)
        ax.plot(xs, ys, linestyle="none", marker=marker, ms=6, mfc=color, mec="white", mew=1.2, zorder=3, label=label)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_xticks(range(len(BINS)), [b[1] for b in BINS])
    ax.set_xlim(-0.4, len(BINS) - 0.6)
    ax.set_xlabel("Months before the event")
    ax.set_ylabel("Within-patient change in LVEF ≤45% logit")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.2, labelcolor=INK)


def slopes(ax, rows):
    p, delta = RES["primary"], RES["delta"]
    items = [("LVSD", "LVSD", p["beta"]["LVSD"]), ("AF", "AF", p["beta"]["AF"]), ("CTRL", "Limb fracture", p["beta"]["CTRL"]),
             None,
             ("L-A", "LVSD − AF (primary)", p["contrasts"]["L-A"]), ("L-H", "LVSD − limb fracture", p["contrasts"]["L-H"]),
             ("A-H", "AF − limb fracture", p["contrasts"]["A-H"])]
    reading = {">": "steeper", "<": "flatter", "≡": "equivalent", "?": "inconclusive"}
    y = len(items) - 1
    ticks, labels = [], []
    contrast_ys = []
    for it in items:
        if it is None:
            y -= 1
            continue
        key, label, v = it
        e, (l, h) = v["est"], v["ci95"]
        if key in SERIES:
            color, marker = SERIES[key][1], SERIES[key][2]
        else:
            color, marker = INK2, "D"
            contrast_ys.append(y)
            ax.text(1.02, y, reading[v["class"]], transform=ax.get_yaxis_transform(), va="center", fontsize=7, color=INK)
        ax.plot([l, h], [y, y], color=color, lw=1.0, zorder=2)
        ax.plot([e], [y], marker=marker, ms=5.5 if marker == "D" else 6, mfc=color, mec="white", mew=1.2, zorder=3)
        rows.append({"panel": "B", "group": key, "item": label, "est": e, "ci95_lo": l, "ci95_hi": h,
                     "class": v.get("class", "")})
        ticks.append(y); labels.append(label)
        y -= 1
    top, bot = max(contrast_ys) + 0.5, min(contrast_ys) - 0.5
    ax.fill_between([-delta, delta], [bot, bot], [top, top], color=BAND, lw=0, zorder=0)
    ax.text(0, top + 0.12, f"±Δ = {delta:.3f}", ha="center", va="bottom", fontsize=6.5, color=MUTED, zorder=4,
            bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none"})
    ax.axvline(0, color=AXIS, lw=0.8, zorder=1)
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("Slope, logit per year (95% CI)")
    rows.append({"panel": "B", "group": "delta", "item": "equivalence margin", "est": delta, "ci95_lo": -delta, "ci95_hi": delta, "class": ""})


def main():
    OUT.mkdir(exist_ok=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55})
    rows = []
    for ax in (a, b):
        style(ax)
    event_study(a, rows)
    slopes(b, rows)
    panel_label(a, "A")
    panel_label(b, "B")
    fig.savefig(OUT / "fig_event_study.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_event_study.png", dpi=300, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(OUT / "fig_event_study_data.csv", index=False)
    print("written:", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
