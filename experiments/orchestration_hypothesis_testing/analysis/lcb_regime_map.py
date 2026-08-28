"""Regime-map figure: scatter of (prior, L2_gap) for every cell, colored by
bayesian_greedy Δ vs always_verify. Lets reviewers see the regime structure
at a glance.

  X-axis: prior_Y1
  Y-axis: L2_gap (informativeness)
  Color:  bayesian_greedy Δ utility (red→green diverging)
  Shape:  benchmark (one shape per benchmark, 7 total)
  Background: shaded regime zones A/B/C

Usage:
  python3 lcb_regime_map.py \\
    --paper-table data/PAPER_TABLE.json \\
    --out-dir data/paper_figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


CELL_MARKER = {
    "lcb_hard":          "o",   # circle
    "lcb_medium":        "h",   # hexagon
    "lcb_easy":          "D",   # diamond
    "swebench_lite":     "s",   # square
    "swebench_verified": "P",   # filled plus
    "mbpp":              "^",   # triangle up
    "humaneval":         "*",   # star
}
CELL_LABEL = {
    "lcb_hard":          "LCB-hard",
    "lcb_medium":        "LCB-medium",
    "lcb_easy":          "LCB-easy",
    "swebench_lite":     "SWE-Lite",
    "swebench_verified": "SWE-Verified",
    "mbpp":              "MBPP+",
    "humaneval":         "HumanEval+",
}
GEN_SHORT = {
    "gpt5_mini":   "gpt5",
    "qwen3_coder": "qwen3",
    "haiku45":     "haiku",
    "sonnet45":    "sonnet",
    "qwen25_32b":  "qwen32",
}


def _smart_offset(pt: dict, neighbours: list[dict]) -> tuple[int, int]:
    """Pick a label offset that reduces overlap with nearby points."""
    x, y = pt["prior"], pt["L2_gap"]
    dx_sum, dy_sum = 0.0, 0.0
    for n in neighbours:
        if n is pt:
            continue
        dx = x - n["prior"]
        dy = y - n["L2_gap"]
        dist2 = dx * dx + dy * dy + 1e-6
        if dist2 < 0.02:
            dx_sum += dx / dist2
            dy_sum += dy / dist2
    if abs(dx_sum) < 1e-9 and abs(dy_sum) < 1e-9:
        return (8, -3)
    norm = np.hypot(dx_sum, dy_sum)
    return (int(round(8 * dx_sum / norm)), int(round(8 * dy_sum / norm)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-table", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = json.loads(args.paper_table.read_text())

    points: list[dict] = []
    for cell, by_gen in table.items():
        for gen, by_rev in by_gen.items():
            rev = by_rev.get("haiku45_default")
            if rev is None:
                continue
            prior = rev.get("prior_Y1")
            l2_gap = rev.get("L2_gap")
            p = rev["policies"].get("bayesian_greedy")
            if p is None:
                continue
            diff = p.get("diff_vs_always_verify")
            if prior is None or l2_gap is None or diff is None:
                continue
            points.append({
                "cell": cell, "generator": gen,
                "prior": prior, "L2_gap": l2_gap, "diff": diff,
            })

    if not points:
        print("no points to plot")
        return

    n_gens = len({p["generator"] for p in points})
    n_cells = len({p["cell"] for p in points})

    fig, ax = plt.subplots(figsize=(11, 7.2))

    # Tighter color range (5th/95th percentile, symmetric)
    diffs = np.array([p["diff"] for p in points])
    cap = max(np.percentile(np.abs(diffs), 95), 5.0)
    norm = TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap)
    cmap = plt.get_cmap("RdYlGn")

    # Background regime shading (very light)
    # Regime A: prior < 0.55, L2 gap > 0.4  (low prior, informative critic)
    # Regime B: 0.4 <= prior <= 0.7, L2 gap > 0.7  (mid prior, near-oracle critic)
    # Regime C: prior > 0.7  (saturated)
    ax.axvspan(0.0, 0.55, alpha=0.05, color="#16a34a", zorder=0)
    ax.axvspan(0.55, 0.75, alpha=0.05, color="#dc2626", zorder=0)
    ax.axvspan(0.75, 1.0,  alpha=0.05, color="#525252", zorder=0)

    # Draw each cell type
    for cell_key, marker in CELL_MARKER.items():
        pts = [p for p in points if p["cell"] == cell_key]
        if not pts:
            continue
        xs = [p["prior"] for p in pts]
        ys = [p["L2_gap"] for p in pts]
        cs = [p["diff"] for p in pts]
        ax.scatter(xs, ys, c=cs, cmap=cmap, norm=norm,
                   marker=marker, s=180,
                   edgecolors="black", linewidths=0.8,
                   zorder=3)

    # Generator labels with smart offsets
    for p in points:
        ox, oy = _smart_offset(p, points)
        ax.annotate(GEN_SHORT.get(p["generator"], p["generator"]),
                    (p["prior"], p["L2_gap"]),
                    textcoords="offset points", xytext=(ox, oy),
                    fontsize=6.5, alpha=0.75, zorder=4)

    # Axes
    ax.set_xlabel("Generator strength on benchmark  (prior $P(Y=1)$)", fontsize=11)
    ax.set_ylabel("L2 critic informativeness  ($P(\\mathrm{pass}|Y{=}1) - P(\\mathrm{pass}|Y{=}0)$)",
                  fontsize=11)
    ax.set_title(f"Regime map: bayesian_greedy $\\Delta$ utility vs always_verify "
                 f"(across {n_gens} generators × {n_cells} benchmarks = {len(points)} cells)",
                 fontsize=12, pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25, linestyle=":", zorder=1)
    ax.axhline(0, color="black", linewidth=0.4)

    # Regime labels in TOP STRIP — out of the way of points
    label_y = 1.015
    ax.text(0.275, label_y, "Regime A — Bayesian wins",
            fontsize=10, fontweight="bold", color="#16a34a", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#16a34a", linewidth=1.0, alpha=0.95))
    ax.text(0.65, label_y, "Regime B — threshold(L2) wins",
            fontsize=10, fontweight="bold", color="#dc2626", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#dc2626", linewidth=1.0, alpha=0.95))
    ax.text(0.875, label_y, "Regime C — Bayesian = always_verify",
            fontsize=10, fontweight="bold", color="#525252", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#525252", linewidth=1.0, alpha=0.95))

    # Color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.015, shrink=0.85)
    cbar.set_label("bayesian_greedy $\\Delta$ utility (vs always_verify)", fontsize=10)

    # Custom legend with shape per benchmark
    legend_elems = [
        Line2D([0], [0], marker=CELL_MARKER[k], color="white",
               markerfacecolor="#999", markeredgecolor="black", markersize=10,
               label=CELL_LABEL[k])
        for k in CELL_MARKER
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=9,
              framealpha=0.95, title="Benchmark", ncol=1)

    fig.tight_layout()
    fig.savefig(args.out_dir / "fig4_regime_map.pdf", bbox_inches="tight")
    fig.savefig(args.out_dir / "fig4_regime_map.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {args.out_dir / 'fig4_regime_map.pdf'}")
    print(f"  wrote {args.out_dir / 'fig4_regime_map.png'}")
    print(f"  {len(points)} points plotted ({n_gens} gens × {n_cells} benches)")


if __name__ == "__main__":
    main()
