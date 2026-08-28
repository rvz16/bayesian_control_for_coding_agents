#!/usr/bin/env python3
"""Compute likelihood tables from calibration data.

Reads raw_results.jsonl produced by generate_calibration_data.py and estimates:
1. P(z_k = pass | Y=1) and P(z_k = pass | Y=0) for each critic level k
2. Generator transition kernel: P(Y'=1 | Y=0, refine) and P(Y'=0 | Y=1, refine)

Output: likelihood_tables.json — the confusion matrix that the Bayesian controller
loads to perform belief updates via Bayes' rule.

Usage:
    python compute_likelihoods.py
    python compute_likelihoods.py --input data/raw_results.jsonl
    python compute_likelihoods.py --smoothing 1.0  # Laplace smoothing count
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_INPUT = Path(__file__).resolve().parent / "data" / "raw_results_v3.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "likelihood_tables.json"

CRITIC_LEVELS = ["L0_syntax", "L1_lint", "L2_fast_test", "L3_llm_review", "L4_mypy"]


@dataclass
class ConfusionCounts:
    """Counts for a 2x2 confusion matrix: critic outcome vs ground truth."""
    tp: int = 0  # pass and Y=1
    fp: int = 0  # pass and Y=0
    fn: int = 0  # fail and Y=1
    tn: int = 0  # fail and Y=0


def load_records(input_path: Path) -> list[dict]:
    """Load calibration records from JSONL."""
    records: list[dict] = []
    with open(input_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping line %d: %s", line_num, e)
    return records


def compute_critic_likelihoods(
    records: list[dict],
    smoothing: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Compute P(pass|Y=1) and P(pass|Y=0) for each critic level.

    Uses Laplace smoothing to avoid zero probabilities.
    """
    counts: dict[str, ConfusionCounts] = {
        level: ConfusionCounts() for level in CRITIC_LEVELS
    }

    for record in records:
        y = record["ground_truth"]
        critics = record.get("critic_results", {})

        for level in CRITIC_LEVELS:
            critic = critics.get(level)
            if critic is None:
                continue

            passed = critic.get("passed", False)

            if passed and y == 1:
                counts[level].tp += 1
            elif passed and y == 0:
                counts[level].fp += 1
            elif not passed and y == 1:
                counts[level].fn += 1
            else:  # not passed and y == 0
                counts[level].tn += 1

    likelihoods: dict[str, dict[str, float]] = {}
    for level in CRITIC_LEVELS:
        c = counts[level]
        n_correct = c.tp + c.fn
        n_incorrect = c.fp + c.tn

        # P(pass | Y=1) with Laplace smoothing
        p_pass_correct = (c.tp + smoothing) / (n_correct + 2 * smoothing)
        # P(pass | Y=0) with Laplace smoothing
        p_pass_incorrect = (c.fp + smoothing) / (n_incorrect + 2 * smoothing)

        likelihoods[level] = {
            "p_pass_given_correct": round(p_pass_correct, 4),
            "p_pass_given_incorrect": round(p_pass_incorrect, 4),
            "counts": {
                "tp": c.tp,
                "fp": c.fp,
                "fn": c.fn,
                "tn": c.tn,
                "n_correct": n_correct,
                "n_incorrect": n_incorrect,
            },
        }

    return likelihoods


def compute_generator_transition(
    records: list[dict],
    smoothing: float = 1.0,
) -> dict[str, float]:
    """Estimate generator transition kernel from multiple patches per instance.

    For instances with multiple patches, we treat consecutive patches as
    refinement steps and count transitions:
        0->1 (fixed), 1->0 (broken), 0->0 (still broken), 1->1 (still correct)

    This is a rough estimate since the patches are independent samples,
    not actual refinements. The real transition kernel will be calibrated
    in Phase 2 with actual iterative refinement data.
    """
    # Group patches by instance (handle both SWE-bench and LCB schemas)
    by_instance: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = record.get("instance_id") or record.get("question_id") or "?"
        by_instance[key].append(record)

    # Sort by patch_id within each instance
    for patches in by_instance.values():
        patches.sort(key=lambda r: r.get("patch_id", r.get("step", 0)))

    # Count transitions between consecutive patches
    fix_count = 0  # 0 -> 1
    break_count = 0  # 1 -> 0
    stay_broken = 0  # 0 -> 0
    stay_correct = 0  # 1 -> 1

    for patches in by_instance.values():
        for i in range(len(patches) - 1):
            y_prev = patches[i]["ground_truth"]
            y_next = patches[i + 1]["ground_truth"]

            if y_prev == 0 and y_next == 1:
                fix_count += 1
            elif y_prev == 1 and y_next == 0:
                break_count += 1
            elif y_prev == 0 and y_next == 0:
                stay_broken += 1
            else:
                stay_correct += 1

    total_from_broken = fix_count + stay_broken
    total_from_correct = break_count + stay_correct

    p_fix = (fix_count + smoothing) / (total_from_broken + 2 * smoothing) if total_from_broken > 0 else 0.2
    p_break = (break_count + smoothing) / (total_from_correct + 2 * smoothing) if total_from_correct > 0 else 0.08

    return {
        "p_fix_given_broken": round(p_fix, 4),
        "p_break_given_correct": round(p_break, 4),
        "counts": {
            "fix_0_to_1": fix_count,
            "break_1_to_0": break_count,
            "stay_broken_0_to_0": stay_broken,
            "stay_correct_1_to_1": stay_correct,
            "total_transitions": fix_count + break_count + stay_broken + stay_correct,
        },
    }


def print_summary(
    likelihoods: dict[str, dict[str, float]],
    transition: dict[str, float],
    n_records: int,
    n_correct: int,
    n_incorrect: int,
) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("CALIBRATION RESULTS")
    print("=" * 70)
    print(f"Total patches: {n_records}")
    print(f"Correct (Y=1): {n_correct} ({100*n_correct/n_records:.1f}%)")
    print(f"Incorrect (Y=0): {n_incorrect} ({100*n_incorrect/n_records:.1f}%)")

    print(f"\n{'Critic Level':<20} {'P(pass|Y=1)':<15} {'P(pass|Y=0)':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}")
    print("-" * 84)
    for level in CRITIC_LEVELS:
        lk = likelihoods[level]
        c = lk["counts"]
        print(
            f"{level:<20} {lk['p_pass_given_correct']:<15.4f} {lk['p_pass_given_incorrect']:<15.4f} "
            f"{c['tp']:<6} {c['fp']:<6} {c['fn']:<6} {c['tn']:<6}"
        )

    print(f"\n{'Generator Transition Kernel'}")
    print("-" * 40)
    print(f"P(fix | broken):     {transition['p_fix_given_broken']:.4f}")
    print(f"P(break | correct):  {transition['p_break_given_correct']:.4f}")
    tc = transition["counts"]
    print(f"Transitions: {tc['total_transitions']} total "
          f"({tc['fix_0_to_1']} fixes, {tc['break_1_to_0']} breaks, "
          f"{tc['stay_broken_0_to_0']} stay-broken, {tc['stay_correct_1_to_1']} stay-correct)")

    # Informativeness check: L2 should be more informative than L1 > L0
    print("\nInformativeness check (gap = P(pass|Y=1) - P(pass|Y=0), higher = more informative):")
    for level in CRITIC_LEVELS:
        lk = likelihoods[level]
        gap = lk["p_pass_given_correct"] - lk["p_pass_given_incorrect"]
        print(f"  {level}: gap = {gap:.4f}")

    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute likelihood tables from calibration data."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to raw_results.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to output likelihood_tables.json.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Laplace smoothing count (default: 1.0).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        log.error("Run generate_calibration_data.py first.")
        sys.exit(1)

    # Load data
    records = load_records(input_path)
    log.info("Loaded %d calibration records", len(records))

    if not records:
        log.error("No records found in %s", input_path)
        sys.exit(1)

    # Compute statistics
    n_correct = sum(1 for r in records if r["ground_truth"] == 1)
    n_incorrect = sum(1 for r in records if r["ground_truth"] == 0)

    likelihoods = compute_critic_likelihoods(records, smoothing=args.smoothing)
    transition = compute_generator_transition(records, smoothing=args.smoothing)

    # Print summary
    print_summary(likelihoods, transition, len(records), n_correct, n_incorrect)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "critic_likelihoods": {
            level: {
                "p_pass_given_correct": lk["p_pass_given_correct"],
                "p_pass_given_incorrect": lk["p_pass_given_incorrect"],
            }
            for level, lk in likelihoods.items()
        },
        "generator_transition": {
            "p_fix_given_broken": transition["p_fix_given_broken"],
            "p_break_given_correct": transition["p_break_given_correct"],
        },
        "sample_counts": {
            "total_patches": len(records),
            "correct": n_correct,
            "incorrect": n_incorrect,
        },
        "detailed_counts": {
            level: lk["counts"] for level, lk in likelihoods.items()
        },
        "transition_counts": transition["counts"],
        "calibration_metadata": {
            "smoothing": args.smoothing,
            "input_file": str(input_path),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info("Likelihood tables saved to: %s", output_path)
    log.info("Ready for Phase 2: Bayesian controller can load this file.")


if __name__ == "__main__":
    main()
