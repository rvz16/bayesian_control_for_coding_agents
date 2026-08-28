"""Compute IID baseline kernel from LCB independent-samples data.

For each (instance, patch_id) → (instance, patch_id+1) pair across the
existing critic_results.jsonl 3-patch data, count Y transitions. This
gives us a "no-feedback IID baseline" kernel we can compare to the
measured iterative-refinement kernel.

If P_fix from this baseline ≈ prior_Y1, the IID assumption is confirmed
across independent samples (no feedback). If iterative refinement gives
a DIFFERENT P_fix, that's evidence that feedback matters.

Output: per-generator <gen>/transition_kernel_iid_baseline.json

Usage:
  python3 lcb_baseline_kernel.py \\
    --output-dir data/lcb_calibration_v2 \\
    --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    print(f"{'gen':<14} {'n_pairs':>8} {'0->0':>5} {'0->1':>5} {'1->0':>5} {'1->1':>5} "
          f"{'P_fix':>7} {'P_break':>9} {'prior':>7} {'gap':>9}")

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        rec_path = out_dir / gen / "critic_results.jsonl"
        if not rec_path.exists():
            print(f"[{gen}] no critic_results.jsonl"); continue

        # Group by instance, sort by patch_id
        by_inst: dict[str, list[dict]] = {}
        for line in open(rec_path):
            if not line.strip(): continue
            r = json.loads(line)
            by_inst.setdefault(r["instance_id"], []).append(r)
        for inst in by_inst:
            by_inst[inst].sort(key=lambda r: r["patch_id"])

        # Count transitions across consecutive patch_ids
        counts = {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}
        for inst, traj in by_inst.items():
            for i in range(len(traj) - 1):
                yt = traj[i].get("Y")
                yt1 = traj[i + 1].get("Y")
                if yt is None or yt1 is None: continue
                counts[f"{yt}->{yt1}"] += 1

        n_broken = counts["0->0"] + counts["0->1"]
        n_correct = counts["1->0"] + counts["1->1"]
        n_pairs = n_broken + n_correct
        if n_pairs == 0:
            print(f"[{gen}] no transition data"); continue
        # Beta(1,1)
        P_fix = (counts["0->1"] + 1) / (n_broken + 2)
        P_break = (counts["1->0"] + 1) / (n_correct + 2)

        # Compare to prior
        likes = json.loads((out_dir / gen / "likelihood_tables.json").read_text())
        prior = likes["prior_Y1"]
        gap = P_fix - prior

        print(f"{gen:<14} {n_pairs:>8} {counts['0->0']:>5} {counts['0->1']:>5} "
              f"{counts['1->0']:>5} {counts['1->1']:>5} "
              f"{P_fix:>7.3f} {P_break:>9.3f} {prior:>7.3f} {gap:>+9.3f}")

        kernel = {
            "generator": gen, "source": "iid_independent_samples",
            "kernel_all": {
                "P_fix_given_broken": P_fix,
                "P_break_given_correct": P_break,
                "raw_counts": counts,
                "n_pairs": n_pairs,
                "smoothing": "Beta(1,1)",
            },
            "comparison_with_prior": {
                "prior_Y1": prior,
                "P_fix_minus_prior": gap,
            },
        }
        (out_dir / gen / "transition_kernel_iid_baseline.json").write_text(json.dumps(kernel, indent=2))


if __name__ == "__main__":
    main()
