"""Plots from curves.csv (calK_jointcomb).
  sweep_swe_critics_slow.png  primary   (critics syn 1 / llm 5)
  sweep_swe_critics_fast.png  secondary (critics all 1)
  sweep_nonswe_headroom.png   non-SWE screen (critics all 1)
usage: plot_sweep.py OUT_DIR"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = sys.argv[1]
C = {"swe_lite": "#2a78d6", "swe_verified": "#eb6834", "swe_pooled": "#1baf7a", "nonswe_pooled": "#2a78d6"}
LAB = {"swe_lite": "SWE-Lite (6 cells)", "swe_verified": "SWE-Verified (6 cells)", "swe_pooled": "SWE pooled (12 cells)",
       "nonswe_pooled": "non-SWE pooled (42 cells)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
rows = [r for r in csv.DictReader(open(os.path.join(OUT, "curves.csv"))) if r["variant"] == "calK_jointcomb"]


def series(crit, scope, col):
    rs = sorted([r for r in rows if r["critics"] == crit and r["scope"] == scope], key=lambda r: float(r["c_ver"]))
    x = [float(r["c_ver"]) for r in rs]
    y = [float(r[col]) for r in rs]
    lo = [float(r[col + "_lo"]) for r in rs] if col + "_lo" in rs[0] else None
    hi = [float(r[col + "_hi"]) for r in rs] if col + "_hi" in rs[0] else None
    return x, y, lo, hi


def style(ax, title, ylab):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("verifier cost C_ver (R = 100, C_gen = 10)", fontsize=9, color=MUTED)
    ax.set_ylabel(ylab, fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks([5, 10, 15, 20, 30, 45, 60, 75, 90])


def swe_fig(crit, fname, crit_label):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [("BDP_minus_ii", "BDP − rule (ii) (base rule, train-picked)", "Δ utility per task"),
              ("BDP_minus_iii", "BDP − rule (iii) (incl. signal-thresholded, train-picked)", "Δ utility per task")]
    for ax, (col, t, yl) in zip(axs[:2], panels):
        for sc in ("swe_lite", "swe_verified", "swe_pooled"):
            x, y, lo, hi = series(crit, sc, col)
            lw = 2.4 if sc == "swe_pooled" else 1.6
            ax.plot(x, y, color=C[sc], lw=lw, marker="o", ms=4, label=LAB[sc])
            ax.fill_between(x, lo, hi, color=C[sc], alpha=0.14 if sc == "swe_pooled" else 0.08, lw=0)
        ax.axvspan(10, 20, color="#9b9a95", alpha=0.10, lw=0)
        style(ax, t, yl)
    ax = axs[2]
    for sc in ("swe_lite", "swe_verified", "swe_pooled"):
        x, y, lo, hi = series(crit, sc, "headroom_vs_BDP")
        ax.plot(x, y, color=C[sc], lw=2.4 if sc == "swe_pooled" else 1.6, marker="o", ms=4)
        x, y, _, _ = series(crit, sc, "headroom_vs_ii")
        ax.plot(x, y, color=C[sc], lw=1.4, ls="--")
    ax.plot([], [], color=MUTED, lw=1.8, marker="o", ms=4, label="perfect − BDP")
    ax.plot([], [], color=MUTED, lw=1.4, ls="--", label="perfect − rule (ii)")
    ax.axvspan(10, 20, color="#9b9a95", alpha=0.10, lw=0)
    style(ax, "Perfect-info headroom (oracle knows Y per patch)", "utility per task")
    axs[0].legend(fontsize=8, frameon=False, loc="upper left")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"SWE verifier-cost sweep, calK_jointcomb, critics {crit_label}; bands = 95% bootstrap CI "
                 "(tasks resampled within cell); shaded = C_ver 10–20", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, fname), dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def nonswe_fig():
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axs[0]
    for col, lab, colr in (("BDP_minus_ii", "BDP − rule (ii)", "#2a78d6"), ("BDP_minus_iii", "BDP − rule (iii)", "#eb6834")):
        x, y, lo, hi = series("f", "nonswe_pooled", col)
        ax.plot(x, y, color=colr, lw=2, marker="o", ms=4, label=lab)
        ax.fill_between(x, lo, hi, color=colr, alpha=0.12, lw=0)
    style(ax, "non-SWE pooled (42 cells): BDP − fair rules", "Δ utility per task")
    ax.legend(fontsize=8, frameon=False)
    ax = axs[1]
    for col, lab, colr in (("headroom_vs_BDP", "perfect − BDP", "#2a78d6"), ("headroom_vs_ii", "perfect − rule (ii)", "#eb6834"),
                           ("headroom_vs_oracle_rule", "perfect − best rule on eval", "#1baf7a")):
        x, y, lo, hi = series("f", "nonswe_pooled", col)
        ax.plot(x, y, color=colr, lw=2, marker="o", ms=4, label=lab)
        ax.fill_between(x, lo, hi, color=colr, alpha=0.10, lw=0)
    style(ax, "non-SWE pooled: perfect-information headroom", "utility per task")
    ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Non-SWE verifier-cost screen, calK_jointcomb, critics all 1", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "sweep_nonswe_headroom.png"), dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


swe_fig("s", "sweep_swe_critics_slow.png", "syn 1 / llm 5 (primary)")
swe_fig("f", "sweep_swe_critics_fast.png", "all 1 (secondary)")
nonswe_fig()
print("ok")
