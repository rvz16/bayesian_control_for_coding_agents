"""Compute per-generator transition kernel from iter_records.jsonl.

For each generator, counts (Y_t, Y_{t+1}) transitions across all (instance, step)
pairs and computes Beta(1,1)-smoothed:
  - P_fix_given_broken = (Y_t=0 → Y_{t+1}=1) / total Y_t=0
  - P_break_given_correct = (Y_t=1 → Y_{t+1}=0) / total Y_t=1

Wraps _common.kernel.compute_transition_kernel_from_pairs and writes the
output to <gen>/transition_kernel.json in the schema lcb_compare_swap_reviewer
expects (kernel_all wrapper with P_stay_* fields).

Usage:
  python3 -m iter.kernel \\
    --iter-dir data/swebench_verified_iter \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.kernel import (  # noqa: E402
    compute_transition_kernel_from_pairs,
    pairs_from_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    args = parser.parse_args()

    iter_dir = args.iter_dir.resolve()
    print(f"{'gen':<14} {'n_pairs':>8} {'0->0':>5} {'0->1':>5} {'1->0':>5} {'1->1':>5} {'P_fix':>7} {'P_break':>9}")

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        path = iter_dir / gen / "iter_records.jsonl"
        if not path.exists():
            print(f"[{gen}] missing iter_records.jsonl"); continue

        # Group by instance, sorted by step
        by_inst: dict[str, list[dict]] = {}
        for line in open(path):
            if not line.strip(): continue
            r = json.loads(line)
            by_inst.setdefault(r["instance_id"], []).append(r)
        for inst in by_inst:
            by_inst[inst].sort(key=lambda r: r["step"])

        pairs = pairs_from_trajectories(by_inst.values())
        k = compute_transition_kernel_from_pairs(pairs)
        counts = k["raw_counts"]

        print(f"{gen:<14} {k['n_pairs']:>8} {counts['0->0']:>5} {counts['0->1']:>5} "
              f"{counts['1->0']:>5} {counts['1->1']:>5} "
              f"{k['P_fix_given_broken']:>7.3f} {k['P_break_given_correct']:>9.3f}")

        # Save kernel json (matches lcb_compare_swap_reviewer's expected schema)
        kernel = {
            "generator": gen,
            "kernel_all": {
                "P_fix_given_broken": k["P_fix_given_broken"],
                "P_stay_broken": k["P_stay_broken"],
                "P_break_given_correct": k["P_break_given_correct"],
                "P_stay_correct": k["P_stay_correct"],
                "raw_counts": counts,
                "n_pairs": k["n_pairs"],
                "smoothing": "Beta(1,1)",
            },
        }
        out_path = iter_dir / gen / "transition_kernel.json"
        out_path.write_text(json.dumps(kernel, indent=2))


if __name__ == "__main__":
    main()
