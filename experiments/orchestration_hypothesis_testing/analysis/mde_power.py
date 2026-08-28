"""Minimum-detectable-effect (MDE) power analysis per cell.

For each (benchmark, generator) cell we already have a 95% paired-bootstrap CI
on `diff_vs_always_verify` for the headline `bayesian_greedy` policy. From that
CI we derive:

  SE = (ci95_hi - ci95_lo) / (2 * 1.96)
  MDE_80%  = (1.96 + 0.84) * SE   ~= 2.802 * SE        (two-sided alpha=0.05)
  MDE_90%  = (1.96 + 1.282) * SE  ~= 3.242 * SE

`MDE_80%` is the smallest |Δ| we could have detected at 80% power given the
observed sample size and bootstrap variance. The observed effect should
exceed MDE in absolute value if it is robust.

Reads policy_comparison.json from every (benchmark, generator) cell. Handles
both schema variants:
  (a) top-level policy_name -> stats dict          (legacy)
  (b) {kernel_source, kernel_used, policies: ...}   (new schema)

Outputs:
  data/mde_power/per_cell.csv
  data/mde_power/per_cell.json

Usage:
  python3 scripts/mde_power_analysis.py \\
      --data-root data \\
      --output-root data/mde_power
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Z-score thresholds
Z_ALPHA_2 = 1.96       # two-sided alpha=0.05
Z_BETA_80 = 0.8416     # power=0.80
Z_BETA_90 = 1.2816     # power=0.90

BENCHMARKS = [
    ("LCB-hard", "lcb_calibration_v2"),
    ("LCB-medium", "lcb_calibration_medium"),
    ("LCB-easy", "lcb_calibration_easy"),
    ("MBPP+", "mbpp_calibration"),
    ("HumanEval+", "humaneval_calibration"),
    ("SWE-Lite", "swebench_lite"),
    ("SWE-Verified", "swebench_verified"),
]

GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45"]

POLICIES_TO_REPORT = ["bayesian_greedy", "bayesian_DP", "threshold_L2", "always_verify"]


def load_policies(path: Path) -> dict | None:
    """Return the {policy_name: stats} dict from a policy_comparison.json,
    handling both legacy and wrapped schemas. Returns None if missing."""
    if not path.exists():
        return None
    d = json.load(open(path))
    if "policies" in d and isinstance(d["policies"], dict):
        return d["policies"]
    # Legacy: top-level keys are policy names
    if any(p in d for p in POLICIES_TO_REPORT):
        return d
    return None


def derive_n_from_critic_results(critic_path: Path) -> int:
    """Return number of unique instances. We use this as paired-bootstrap n."""
    if not critic_path.exists():
        return 0
    insts = set()
    for line in open(critic_path):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            insts.add(r.get("instance_id"))
        except Exception:
            continue
    return len(insts - {None})


def mde_for_policy(policy_stats: dict, policy_name: str, n_instances: int) -> dict:
    """Compute MDE numbers from the bootstrap CI on diff_vs_always_verify."""
    delta = policy_stats.get("diff_vs_always_verify")
    lo = policy_stats.get("ci95_lo")
    hi = policy_stats.get("ci95_hi")
    if delta is None or lo is None or hi is None:
        return {"policy": policy_name, "skipped": "missing fields"}
    half_width = (hi - lo) / 2
    se = half_width / Z_ALPHA_2 if Z_ALPHA_2 > 0 else float("inf")
    mde_80 = (Z_ALPHA_2 + Z_BETA_80) * se
    mde_90 = (Z_ALPHA_2 + Z_BETA_90) * se
    detected_80 = abs(delta) > mde_80
    detected_90 = abs(delta) > mde_90
    significant = (lo > 0) or (hi < 0)  # strictly excludes zero
    return {
        "policy": policy_name,
        "n_instances": n_instances,
        "delta": delta,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "se": se,
        "mde_80": mde_80,
        "mde_90": mde_90,
        "detected_at_80": detected_80,
        "detected_at_90": detected_90,
        "significant_at_05": significant,
    }


def analyze_cell(bench_dir: Path, gen: str) -> dict:
    pol_path = bench_dir / gen / "policy_comparison.json"
    crit_path = bench_dir / gen / "critic_results.jsonl"
    n_instances = derive_n_from_critic_results(crit_path)
    pols = load_policies(pol_path)
    if pols is None:
        return {"skipped": f"no policy_comparison at {pol_path}", "n_instances": n_instances}
    out: dict = {"n_instances": n_instances, "policies": {}}
    for p in POLICIES_TO_REPORT:
        if p not in pols:
            continue
        out["policies"][p] = mde_for_policy(pols[p], p, n_instances)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    full: dict[str, dict] = {}
    for bench_label, bench_dir in BENCHMARKS:
        for gen in GENERATORS:
            cell_id = f"{bench_label}__{gen}"
            full[cell_id] = analyze_cell(args.data_root / bench_dir, gen)

    # JSON dump
    json_path = args.output_root / "per_cell.json"
    json_path.write_text(json.dumps(full, indent=2))

    # CSV summary - one row per (cell, policy)
    csv_path = args.output_root / "per_cell.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "benchmark", "generator", "policy",
            "n_instances", "delta", "ci95_lo", "ci95_hi",
            "SE", "MDE_80", "MDE_90", "detected_80", "detected_90", "significant",
        ])
        for bench_label, _ in BENCHMARKS:
            for gen in GENERATORS:
                cell = full.get(f"{bench_label}__{gen}", {})
                if "skipped" in cell:
                    w.writerow([bench_label, gen, "", cell.get("n_instances", ""),
                                "skipped"] + [""] * 8)
                    continue
                for p, stats in cell.get("policies", {}).items():
                    if "skipped" in stats:
                        w.writerow([bench_label, gen, p, cell["n_instances"],
                                    "skipped"] + [""] * 8)
                        continue
                    w.writerow([
                        bench_label, gen, p,
                        cell["n_instances"],
                        f"{stats['delta']:.3f}",
                        f"{stats['ci95_lo']:.3f}",
                        f"{stats['ci95_hi']:.3f}",
                        f"{stats['se']:.3f}",
                        f"{stats['mde_80']:.3f}",
                        f"{stats['mde_90']:.3f}",
                        "yes" if stats["detected_at_80"] else "no",
                        "yes" if stats["detected_at_90"] else "no",
                        "yes" if stats["significant_at_05"] else "no",
                    ])

    # Headline summary printed to stdout
    print(f"Per-cell JSON: {json_path}")
    print(f"Per-cell CSV:  {csv_path}")
    print()

    print("=== Headline: MDE_80 vs |observed delta| for bayesian_greedy ===")
    print(f"{'cell':40} {'n':>4}  {'|delta|':>8}  {'MDE_80':>8}  {'detected':>8}  {'signif':>6}")
    n_detected = 0
    n_total = 0
    n_signif = 0
    for bench_label, _ in BENCHMARKS:
        for gen in GENERATORS:
            cell = full.get(f"{bench_label}__{gen}", {})
            if "skipped" in cell:
                continue
            stats = cell.get("policies", {}).get("bayesian_greedy")
            if not stats or "skipped" in stats:
                continue
            label = f"{bench_label}/{gen}"[:40]
            d = abs(stats["delta"])
            mde = stats["mde_80"]
            det = "yes" if stats["detected_at_80"] else "no"
            sig = "yes" if stats["significant_at_05"] else "no"
            print(f"{label:40} {cell['n_instances']:>4}  {d:>8.3f}  {mde:>8.3f}  {det:>8}  {sig:>6}")
            n_total += 1
            if stats["detected_at_80"]:
                n_detected += 1
            if stats["significant_at_05"]:
                n_signif += 1
    print()
    print(f"bayesian_greedy detected at 80% power: {n_detected}/{n_total} cells")
    print(f"bayesian_greedy significant (95% CI excludes 0): {n_signif}/{n_total} cells")


if __name__ == "__main__":
    main()
