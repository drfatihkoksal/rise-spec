"""Patient flow diagram for both data sources, read from results/wp1_flow_5y_limbfx.json (MIMIC) and
results/wp_h_flow.json (HEEDB, MGH + Emory pooled). No number is typed by hand.

    python3 -I analysis/23_flow_diagram.py

-> figures/fig_flow.{pdf,png}, figures/fig_flow_data.json (the numbers drawn)
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parents[1]
RES, OUT = PROJ / "results", PROJ / "figures"
INK, INK2, EDGE, FILL = "#0b0b0b", "#52514e", "#c3c2b7", "#f9f9f7"
G = ["LVSD", "AF", "LIMBFX"]
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 6.5, "pdf.fonttype": 42, "savefig.facecolor": "white"})


def f(n):
    return f"{int(n):,}"


def trio(d):
    return " / ".join(f(d.get(g, 0)) for g in G)


def summed(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v if not isinstance(v, dict) else dict(Counter(out.get(k, {})) + Counter(v))
    return out


def mimic():
    r = json.loads((RES / "wp1_flow_5y_limbfx.json").read_text())["primary"]
    a, e = r["anchor_flow"], r["ecg_flow"]
    return {"source": "MIMIC-IV (BIDMC)", "first": r["first_event_patients"], "multi": r["ex_multi_set"], "start": a["start"],
            "excl": [("Fracture code not an initial encounter", a["ex_control_not_initial"]),
                     ("LVSD: earlier HF or cardiomyopathy code", a["ex_prior_hf_cm"]),
                     ("AF, fracture: LVSD code within 1 year", a["ex_lvsd_within_1y"]),
                     ("Age <18 years", a["ex_age"]),
                     ("No ECG at the anchor admission", a["ex_no_ecg_at_anchor"])],
            "eligible": a["after_ex_no_ecg_at_anchor"],
            "ecg_excl": [("No ECG in the 5-year window", e["ex_no_window_ecg"]),
                         ("<2 eligible ECGs or span <180 days", e["ex_lt2_eligible_or_span_lt180d"])],
            "final": e["final_patients"], "ecgs": {g: v["ecgs"] for g, v in r["describe"].items()}}


def heedb():
    r = json.loads((RES / "wp_h_flow.json").read_text())
    s = list(r["sites"].values())
    a = summed([x["anchor_flow"] for x in s])
    e = summed([x["ecg_flow"] for x in s])
    return {"source": "HEEDB (MGH + Emory)", "first": sum(x["first_event_patients"] for x in s), "multi": sum(x["ex_multi_set"] for x in s),
            "start": a["start"],
            "excl": [("Fracture code not an initial encounter", a["ex_control_not_initial"]),
                     ("LVSD: earlier HF or cardiomyopathy code", a["ex_prior_hf_cm"]),
                     ("AF, fracture: LVSD code within 1 year", a["ex_lvsd_within_1y"]),
                     ("Coded history <1 year before the anchor", a["ex_lookback_lt_1y"]),
                     ("Age <18 years or sex missing", a["ex_age_or_sex_missing"]),
                     ("No ECG within 30 days of the anchor", a["ex_no_ecg_at_anchor"])],
            "eligible": a["after_ex_no_ecg_at_anchor"],
            "ecg_excl": [("No ECG in the 5-year window", e["ex_no_window_ecg"]),
                         ("<2 eligible ECGs or span <180 days", e["ex_lt2_eligible_or_span_lt180d"])],
            "final": e["final_patients"], "ecgs": r["pooled_ecgs"]}


def box(ax, x, y, text, w, bold=False, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=6.5, color=INK, fontweight="bold" if bold else "normal", linespacing=1.4,
            bbox={"boxstyle": "square,pad=0.5", "fc": FILL if not bold else "white", "ec": EDGE, "lw": 0.8}, wrap=True)


def arrow(ax, x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.8, "mutation_scale": 7})


def side(ax, x0, y, xs, text):
    ax.annotate("", xy=(xs, y), xytext=(x0, y), arrowprops={"arrowstyle": "-|>", "color": INK2, "lw": 0.8, "mutation_scale": 7})
    ax.text(xs + 0.01, y, text, ha="left", va="center", fontsize=6, color=INK, linespacing=1.35,
            bbox={"boxstyle": "square,pad=0.45", "fc": "white", "ec": EDGE, "lw": 0.8})


def column(ax, d):
    ax.set_xlim(0, 1); ax.set_ylim(0.15, 1); ax.axis("off")
    ax.text(0.0, 0.985, d["source"], fontsize=8, fontweight="bold", color=INK, va="top")
    xm, xs = 0.24, 0.47
    lab = "LVSD / AF / fracture"
    ys = [0.90, 0.72, 0.42, 0.22]
    box(ax, xm, ys[0], f"First LVSD, AF or limb-fracture code\nn = {f(d['first'])}", 0.4)
    side(ax, xm, (ys[0] + ys[1]) / 2, xs, f"Codes from more than one group\nat the first event: {f(d['multi'])}")
    box(ax, xm, ys[1], f"Single-group first event\n{lab}\n{trio(d['start'])}", 0.4)
    excl = "\n".join(f"{t}: {trio(v)}" for t, v in d["excl"])
    side(ax, xm, (ys[1] + ys[2]) / 2, xs, f"Excluded ({lab})\n{excl}")
    box(ax, xm, ys[2], f"Eligible anchors\n{lab}\n{trio(d['eligible'])}", 0.4)
    ecg = "\n".join(f"{t}: {trio(v)}" for t, v in d["ecg_excl"])
    side(ax, xm, (ys[2] + ys[3]) / 2, xs, f"ECG rules ({lab})\n{ecg}")
    box(ax, xm, ys[3], f"Analysed\nLVSD {f(d['final'].get('LVSD', 0))} · AF {f(d['final'].get('AF', 0))} · fracture {f(d['final'].get('LIMBFX', 0))}\n"
                       f"ECGs {f(sum(d['ecgs'].values()))}", 0.4, bold=True)
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow(ax, xm, y0 - 0.045, y1 + 0.05)


def main():
    OUT.mkdir(exist_ok=True)
    data = [mimic(), heedb()]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.8), gridspec_kw={"hspace": 0.06})
    for ax, d in zip(axes, data):
        column(ax, d)
    fig.savefig(OUT / "fig_flow.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_flow.png", dpi=300, bbox_inches="tight")
    (OUT / "fig_flow_data.json").write_text(json.dumps(data, indent=2, default=int))
    print("written")


if __name__ == "__main__":
    main()
