#!/usr/bin/env python3
"""Compute the generator transition kernel from iterative refinement data.

Reads iterative_results.jsonl (produced by iterative_calibration.py) and
estimates:
  P(Y_{t+1} = 1 | Y_t = 0, a_gen)  — probability of fixing a broken patch
  P(Y_{t+1} = 0 | Y_t = 1, a_gen)  — probability of breaking a correct patch

Unlike the kernel computed from independent samples in compute_likelihoods.py,
this measures real refinement transitions.

Usage:
    python compute_transition_kernel.py
    python compute_transition_kernel.py --update-likelihood-tables
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_INPUT = (
    Path(__file__).resolve().parent / "data" / "iterative_results.jsonl"
)
DEFAULT_LIKELIHOODS = (
    Path(__file__).resolve().parent / "data" / "likelihood_tables.json"
)


def load_trajectories(path: Path) -> dict[str, list[dict]]:
    """Load iterative records and group by instance, sorted by step."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    by_instance: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_instance[r["instance_id"]].append(r)

    for traj in by_instance.values():
        traj.sort(key=lambda r: r["step"])

    return dict(by_instance)


def compute_kernel(
    trajectories: dict[str, list[dict]],
    smoothing: float = 1.0,
) -> dict:
    """Estimate transition probabilities from consecutive refinement steps."""
    fix = 0          # 0 -> 1
    break_ = 0       # 1 -> 0
    stay_broken = 0  # 0 -> 0
    stay_correct = 0  # 1 -> 1

    # Per-step transition counts (to check whether refinement gets better over time)
    by_step: dict[int, dict[str, int]] = defaultdict(lambda: {"fix": 0, "break": 0, "stay_broken": 0, "stay_correct": 0})

    for traj in trajectories.values():
        for i in range(len(traj) - 1):
            y_prev = traj[i]["ground_truth"]
            y_next = traj[i + 1]["ground_truth"]
            step = traj[i]["step"]

            if y_prev == 0 and y_next == 1:
                fix += 1
                by_step[step]["fix"] += 1
            elif y_prev == 1 and y_next == 0:
                break_ += 1
                by_step[step]["break"] += 1
            elif y_prev == 0 and y_next == 0:
                stay_broken += 1
                by_step[step]["stay_broken"] += 1
            else:
                stay_correct += 1
                by_step[step]["stay_correct"] += 1

    n_from_broken = fix + stay_broken
    n_from_correct = break_ + stay_correct

    p_fix = (fix + smoothing) / (n_from_broken + 2 * smoothing) if n_from_broken > 0 else 0.2
    p_break = (break_ + smoothing) / (n_from_correct + 2 * smoothing) if n_from_correct > 0 else 0.08

    return {
        "p_fix_given_broken": round(p_fix, 4),
        "p_break_given_correct": round(p_break, 4),
        "counts": {
            "fix_0_to_1": fix,
            "break_1_to_0": break_,
            "stay_broken_0_to_0": stay_broken,
            "stay_correct_1_to_1": stay_correct,
            "n_from_broken": n_from_broken,
            "n_from_correct": n_from_correct,
            "total_transitions": fix + break_ + stay_broken + stay_correct,
        },
        "per_step_counts": {str(k): v for k, v in sorted(by_step.items())},
    }


def summarize_trajectories(trajectories: dict[str, list[dict]]) -> None:
    """Print per-instance summary."""
    print("\nTrajectory summary:")
    print(f"{'Instance':<40} {'Steps':>6} {'Y sequence':<20}")
    print("-" * 80)
    n_converged = 0
    n_diverged = 0
    n_stuck = 0
    for iid, traj in list(trajectories.items())[:30]:
        ys = [r["ground_truth"] for r in traj]
        y_str = "".join(str(y) for y in ys)
        print(f"{iid:<40} {len(traj):>6} {y_str:<20}")
        if ys and ys[-1] == 1:
            n_converged += 1
            if 0 in ys:
                pass  # started broken, converged
        if ys and ys[0] == 1 and ys[-1] == 0:
            n_diverged += 1
        if all(y == 0 for y in ys):
            n_stuck += 1
    if len(trajectories) > 30:
        print(f"... ({len(trajectories) - 30} more)")

    print(f"\nSummary of all {len(trajectories)} trajectories:")
    all_traj = list(trajectories.values())
    converged = sum(1 for t in all_traj if t and t[-1]["ground_truth"] == 1)
    started_broken_converged = sum(
        1 for t in all_traj
        if t and t[0]["ground_truth"] == 0 and t[-1]["ground_truth"] == 1
    )
    started_correct = sum(1 for t in all_traj if t and t[0]["ground_truth"] == 1)
    all_broken = sum(1 for t in all_traj if all(r["ground_truth"] == 0 for r in t))
    print(f"  Total: {len(all_traj)}")
    print(f"  Started correct (patch_0 Y=1): {started_correct}")
    print(f"  Final Y=1 (resolved): {converged}")
    print(f"  Fixed by refinement (0 -> 1): {started_broken_converged}")
    print(f"  Never fixed (all 0): {all_broken}")


def update_likelihood_tables(
    kernel: dict,
    tables_path: Path,
) -> None:
    """Replace the transition kernel in likelihood_tables.json with the new one."""
    if not tables_path.exists():
        log.error("Likelihood tables not found: %s", tables_path)
        return

    with open(tables_path) as f:
        data = json.load(f)

    data["generator_transition"] = {
        "p_fix_given_broken": kernel["p_fix_given_broken"],
        "p_break_given_correct": kernel["p_break_given_correct"],
    }
    data["transition_counts"] = kernel["counts"]
    data.setdefault("calibration_metadata", {})["transition_source"] = "iterative_refinement"

    with open(tables_path, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Updated %s with new transition kernel", tables_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute transition kernel from iterative calibration data."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--likelihood-tables", default=str(DEFAULT_LIKELIHOODS))
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--update-likelihood-tables", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run iterative_calibration.py first.")
        return

    trajectories = load_trajectories(input_path)
    log.info("Loaded %d trajectories", len(trajectories))

    total_records = sum(len(t) for t in trajectories.values())
    log.info("Total records: %d", total_records)

    kernel = compute_kernel(trajectories, smoothing=args.smoothing)

    print("\n" + "=" * 60)
    print("TRANSITION KERNEL (from iterative refinement)")
    print("=" * 60)
    print(f"P(fix | broken):    {kernel['p_fix_given_broken']:.4f}")
    print(f"P(break | correct): {kernel['p_break_given_correct']:.4f}")
    c = kernel["counts"]
    print(f"\nCounts:")
    print(f"  0 -> 1 (fix):           {c['fix_0_to_1']}")
    print(f"  1 -> 0 (break):         {c['break_1_to_0']}")
    print(f"  0 -> 0 (stay broken):   {c['stay_broken_0_to_0']}")
    print(f"  1 -> 1 (stay correct):  {c['stay_correct_1_to_1']}")
    print(f"  Total from broken:      {c['n_from_broken']}")
    print(f"  Total from correct:     {c['n_from_correct']}")
    print(f"  Total transitions:      {c['total_transitions']}")

    print("\nPer-step transition counts (step t -> step t+1):")
    for step, counts in kernel["per_step_counts"].items():
        print(f"  Step {step}: fix={counts['fix']} break={counts['break']} "
              f"stay_broken={counts['stay_broken']} stay_correct={counts['stay_correct']}")

    summarize_trajectories(trajectories)

    if args.update_likelihood_tables:
        update_likelihood_tables(kernel, Path(args.likelihood_tables))


if __name__ == "__main__":
    main()
