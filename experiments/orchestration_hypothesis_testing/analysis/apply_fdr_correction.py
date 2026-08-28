"""Multiple-comparisons correction (Benjamini-Hochberg FDR) across the 28-cell cube.

For each policy reported in PAPER_TABLE.json, computes per-cell two-sided
p-values from the paired-bootstrap CI under the normality approximation
SE = (ci95_hi - ci95_lo) / (2 * 1.96), and applies BH-FDR.

Output:
  data/fdr_correction/per_cell.csv
  data/fdr_correction/summary.json

Usage:
  python3 scripts/apply_fdr_correction.py \\
      --paper-table data/PAPER_TABLE.json \\
      --output-root data/fdr_correction
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

# scipy.stats.norm.cdf
def norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


POLICIES_TO_REPORT = [
    "bayesian_greedy", "bayesian_DP", "threshold_L0", "threshold_L2",
    "threshold_L3", "fixed_pipeline", "best_of_3",
]


def collect_p_values(table: dict) -> list[dict]:
    """For each (cell, generator, policy), extract Δ + CI and compute p-value."""
    rows = []
    for cell, gen_dict in table.items():
        if not isinstance(gen_dict, dict):
            continue
        for gen, variants in gen_dict.items():
            if not isinstance(variants, dict):
                continue
            # Use the default kernel (haiku45_default) as the headline
            v = variants.get("haiku45_default")
            if v is None:
                continue
            policies = v.get("policies", {})
            for pol in POLICIES_TO_REPORT:
                p = policies.get(pol)
                if not p:
                    continue
                delta = p.get("diff_vs_always_verify")
                lo = p.get("ci95_lo")
                hi = p.get("ci95_hi")
                if delta is None or lo is None or hi is None:
                    continue
                half_width = (hi - lo) / 2
                if half_width <= 0:
                    # Degenerate (e.g., always_verify or zero-CI)
                    p_value = 1.0 if abs(delta) < 1e-9 else 0.0
                else:
                    se = half_width / 1.96
                    z = abs(delta) / se if se > 0 else float("inf")
                    p_value = 2 * (1 - norm_cdf(z))
                rows.append({
                    "cell": cell, "gen": gen, "policy": pol,
                    "delta": delta, "ci95_lo": lo, "ci95_hi": hi,
                    "p_value": p_value,
                })
    return rows


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Return (q_values, reject_at_alpha). q_value is BH-FDR-adjusted p."""
    n = len(p_values)
    if n == 0:
        return [], []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    q_values = [0.0] * n
    # BH-step: q_(i) = p_(i) * n / i, then enforce monotonicity from largest
    bh_q = [(idx, p * n / (rank + 1)) for rank, (idx, p) in enumerate(indexed)]
    # Enforce monotonicity: q_(i) = min(q_(i), q_(i+1), ...)
    sorted_qs = [q for _, q in bh_q]
    for i in range(n - 2, -1, -1):
        sorted_qs[i] = min(sorted_qs[i], sorted_qs[i + 1])
    # Cap at 1
    sorted_qs = [min(q, 1.0) for q in sorted_qs]
    # Map back
    for rank, (idx, _) in enumerate(bh_q):
        q_values[idx] = sorted_qs[rank]
    rejects = [q < alpha for q in q_values]
    return q_values, rejects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-table", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    table = json.load(open(args.paper_table))
    rows = collect_p_values(table)
    if not rows:
        print("No rows found.")
        return

    # Apply BH-FDR per policy (separate test family per policy)
    by_policy: dict[str, list[dict]] = {}
    for r in rows:
        by_policy.setdefault(r["policy"], []).append(r)
    for pol, prows in by_policy.items():
        ps = [r["p_value"] for r in prows]
        qs, rejects = benjamini_hochberg(ps, args.alpha)
        for r, q, rej in zip(prows, qs, rejects):
            r["q_value_bh"] = q
            r["reject_at_fdr_05"] = rej

    # Also: global BH-FDR across all (cell × gen × policy) tests
    global_ps = [r["p_value"] for r in rows]
    global_qs, global_rejects = benjamini_hochberg(global_ps, args.alpha)
    for r, q, rej in zip(rows, global_qs, global_rejects):
        r["q_value_global_bh"] = q
        r["reject_global_fdr_05"] = rej

    # Write CSV
    csv_path = args.output_root / "per_cell.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cell", "generator", "policy",
            "delta", "ci95_lo", "ci95_hi", "p_value",
            "q_value_bh_per_policy", "reject_per_policy",
            "q_value_global_bh", "reject_global",
        ])
        for r in sorted(rows, key=lambda x: (x["cell"], x["gen"], x["policy"])):
            w.writerow([
                r["cell"], r["gen"], r["policy"],
                f"{r['delta']:+.3f}", f"{r['ci95_lo']:+.3f}", f"{r['ci95_hi']:+.3f}",
                f"{r['p_value']:.4f}",
                f"{r['q_value_bh']:.4f}", "yes" if r["reject_at_fdr_05"] else "no",
                f"{r['q_value_global_bh']:.4f}", "yes" if r["reject_global_fdr_05"] else "no",
            ])

    # Summary JSON: per-policy and overall
    summary = {"alpha": args.alpha, "n_total_tests": len(rows),
               "per_policy": {}}
    for pol, prows in sorted(by_policy.items()):
        n = len(prows)
        n_signif_uncorrected = sum(1 for r in prows if r["p_value"] < args.alpha)
        n_signif_per_policy = sum(1 for r in prows if r["reject_at_fdr_05"])
        summary["per_policy"][pol] = {
            "n_cells": n,
            "n_significant_uncorrected_p05": n_signif_uncorrected,
            "n_significant_after_BH_FDR_per_policy": n_signif_per_policy,
        }
    n_global_signif = sum(1 for r in rows if r["reject_global_fdr_05"])
    summary["n_significant_global_BH_FDR"] = n_global_signif

    json_path = args.output_root / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print()
    print(f"Total tests: {len(rows)}")
    print(f"Global BH-FDR (alpha={args.alpha}): {n_global_signif} of {len(rows)} cells significant")
    print()
    print("Per-policy BH-FDR:")
    for pol, s in summary["per_policy"].items():
        print(f"  {pol:18}  n={s['n_cells']:>3}  uncorrected_p<.05={s['n_significant_uncorrected_p05']:>3}  BH_FDR<.05={s['n_significant_after_BH_FDR_per_policy']:>3}")


if __name__ == "__main__":
    main()
