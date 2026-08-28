"""Wilson 95% CI on prior_Y1 (binomial proportion) for each cell.

The existing PAPER_TABLE.json reports prior_Y1 as a point estimate (Beta(1,1)
posterior mean). For a methodology section we want the standard Wilson
score interval on the underlying binomial proportion. Output adds a
ci95_lo / ci95_hi pair next to each cell's prior_Y1.

Output:
  data/prior_ci/per_cell.csv
  data/prior_ci/per_cell.json

Usage:
  python3 scripts/wilson_ci_priors.py \\
      --data-root data \\
      --output-root data/prior_ci
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

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


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def count_critic_results(path: Path) -> tuple[int, int]:
    """Return (n_total, n_y1) from critic_results.jsonl."""
    if not path.exists():
        return (0, 0)
    n_total = 0
    n_y1 = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        y = r.get("Y")
        if y not in (0, 1):
            continue
        n_total += 1
        if y == 1:
            n_y1 += 1
    return (n_total, n_y1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for bench_label, bench_dir in BENCHMARKS:
        for gen in GENERATORS:
            path = args.data_root / bench_dir / gen / "critic_results.jsonl"
            n_total, n_y1 = count_critic_results(path)
            if n_total == 0:
                continue
            p_hat = n_y1 / n_total
            beta_smoothed = (n_y1 + 1) / (n_total + 2)
            lo, hi = wilson_ci(n_y1, n_total)
            rows.append({
                "benchmark": bench_label, "generator": gen,
                "n_total": n_total, "n_y1": n_y1,
                "prior_y1_raw": p_hat,
                "prior_y1_beta11": beta_smoothed,
                "wilson_ci_lo": lo, "wilson_ci_hi": hi,
                "ci_width": hi - lo,
            })

    csv_path = args.output_root / "per_cell.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "generator", "n_total", "n_y1",
                    "prior_y1_raw", "prior_y1_beta11",
                    "wilson_ci_lo", "wilson_ci_hi", "ci_width"])
        for r in rows:
            w.writerow([r["benchmark"], r["generator"], r["n_total"], r["n_y1"],
                        f"{r['prior_y1_raw']:.4f}", f"{r['prior_y1_beta11']:.4f}",
                        f"{r['wilson_ci_lo']:.4f}", f"{r['wilson_ci_hi']:.4f}",
                        f"{r['ci_width']:.4f}"])

    json_path = args.output_root / "per_cell.json"
    json_path.write_text(json.dumps(rows, indent=2))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print()
    print(f"{'cell':14} {'gen':14} {'n':>5} {'Y=1':>5} {'p_hat':>7} {'Wilson 95% CI':>22} {'width':>7}")
    for r in rows:
        ci = f"[{r['wilson_ci_lo']:.3f}, {r['wilson_ci_hi']:.3f}]"
        print(f"{r['benchmark']:14} {r['generator']:14} {r['n_total']:>5} {r['n_y1']:>5} {r['prior_y1_raw']:>7.3f} {ci:>22} {r['ci_width']:>7.3f}")


if __name__ == "__main__":
    main()
