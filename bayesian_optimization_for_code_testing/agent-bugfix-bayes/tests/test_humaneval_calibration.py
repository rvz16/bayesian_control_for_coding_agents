"""Option A: empirical-Bayes calibration quality on HumanEvalFix pairs.

No LLM calls. Uses the dataset's (canonical, buggy) pairs as balanced
(Y=1, Y=0) calibration data, then evaluates held-out prediction quality
for the fitted likelihood table vs the hand-tuned baseline.

Output artifacts (Allure):
    - calibration_table.txt      — fitted vs hand-tuned per critic
    - counts.json                — sample counts per critic
    - heldout_metrics.json       — accuracy + log-loss for fitted/handtuned/prior
    - per_task_predictions.json  — per-task Bayesian posterior predictions
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import allure
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.bayes_agent import bayes_update
from abbo.realworld.agents.calibration import (
    calibrate_likelihoods,
    calibration_report,
    format_comparison,
)
from abbo.realworld.agents.humaneval_fix import (
    HE_CRITIC_LIKELIHOODS,
    HE_CRITIC_NAMES,
    collect_calibration_samples_from_pairs,
    list_task_ids,
)


# ---- Knobs ----
N_TRAIN = 40
N_TEST = 20
SPLIT_SEED = 42
PRIOR_BELIEF = 0.5   # for held-out prediction (ignorant prior per task)


@allure.parent_suite("humaneval_calibration")
@allure.suite("option_a_no_llm")
@allure.title("Option A: fit + eval calibration on (canonical, buggy) pairs")
def test_humaneval_calibration_quality() -> None:
    """Fit Beta-Binomial likelihoods on a train split, evaluate on held-out.

    Measures whether Bayes updates with fitted likelihoods correctly classify
    held-out patches (Y=1 vs Y=0) from critic observations alone.
    """
    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    test_ids = all_ids[N_TRAIN:N_TRAIN + N_TEST]

    allure.dynamic.description(
        f"HumanEvalFix Python split: {len(all_ids)} tasks total.\n"
        f"Train: {N_TRAIN}  Test: {N_TEST}  (random split, seed={SPLIT_SEED})\n"
        f"Critics: {HE_CRITIC_NAMES}\n"
        f"Prior belief for held-out prediction: {PRIOR_BELIEF}\n"
        f"No LLM calls — uses dataset's canonical (Y=1) and buggy (Y=0) sources."
    )

    # --- Step 1: collect train samples + fit -----------------------------
    with allure.step(f"Collect train samples ({len(train_ids)} tasks × 2 × 4 critics)"):
        train_samples = collect_calibration_samples_from_pairs(train_ids, verbose=True)

    report = calibration_report(
        train_samples, prior_alpha=1.0, prior_beta=1.0,
        hand_tuned=HE_CRITIC_LIKELIHOODS,
    )
    fitted_lk = report.calibrated

    with allure.step("Fitted table vs hand-tuned"):
        table_text = format_comparison(report)
        print("\n" + table_text + "\n")
        allure.attach(
            table_text,
            name="calibration_table.txt",
            attachment_type=allure.attachment_type.TEXT,
        )
        allure.attach(
            json.dumps({
                "hand_tuned": report.hand_tuned,
                "fitted": report.calibrated,
                "per_critic_counts": report.per_critic_counts,
                "n_train_tasks": len(train_ids),
                "n_train_samples": report.n_samples,
            }, indent=2),
            name="likelihoods.json",
            attachment_type=allure.attachment_type.JSON,
        )

    # --- Step 2: collect test samples + predict held-out -----------------
    with allure.step(f"Collect test samples ({len(test_ids)} tasks)"):
        test_samples = collect_calibration_samples_from_pairs(test_ids, verbose=True)

    # Group test samples by (task_id, Y) → list of (critic, passed)
    #   For each test "patch" (one per task per Y), apply all critic observations
    #   in sequence via Bayes update, then compare final belief to actual Y.
    by_patch: dict[tuple[str, str], list[tuple[str, bool]]] = defaultdict(list)
    patch_y: dict[tuple[str, str], bool] = {}
    for s in test_samples:
        key = (s.bug_id, s.arm)
        by_patch[key].append((s.critic, s.critic_passed))
        patch_y[key] = s.patch_correct

    def _predict(observations: list[tuple[str, bool]], lk: dict) -> float:
        """Apply sequential Bayes updates from PRIOR_BELIEF."""
        b = PRIOR_BELIEF
        for critic, passed in observations:
            b = bayes_update(b, critic, passed, likelihoods=lk)
        return b

    def _score(lk: dict, label: str) -> dict:
        correct = 0
        log_loss_sum = 0.0
        per_task = []
        for key, obs in by_patch.items():
            y = patch_y[key]
            b = _predict(obs, lk)
            pred = b > 0.5
            correct += int(pred == y)
            # Log loss (clipped to avoid -inf)
            eps = 1e-6
            b_clip = min(max(b, eps), 1 - eps)
            ll = -(y * math.log(b_clip) + (1 - y) * math.log(1 - b_clip))
            log_loss_sum += ll
            per_task.append({
                "task": key[0], "arm": key[1], "y": y,
                "observations": [{"c": c, "p": p} for c, p in obs],
                "posterior": round(b, 4),
                "predicted": pred,
                "correct": pred == y,
            })
        n = len(by_patch)
        return {
            "label": label,
            "n_patches": n,
            "accuracy": correct / n,
            "avg_log_loss": log_loss_sum / n,
            "per_task": per_task,
        }

    fitted_score = _score(fitted_lk, "fitted")
    handtuned_score = _score(HE_CRITIC_LIKELIHOODS, "hand_tuned")

    # Baseline 3: uniform-prior + ignorant likelihoods (coin flip)
    flat_lk = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in HE_CRITIC_NAMES}
    flat_score = _score(flat_lk, "flat_prior_only")

    summary = {
        "fitted":     {k: v for k, v in fitted_score.items()      if k != "per_task"},
        "hand_tuned": {k: v for k, v in handtuned_score.items()    if k != "per_task"},
        "flat_prior": {k: v for k, v in flat_score.items()         if k != "per_task"},
        "delta_accuracy_fitted_vs_handtuned":
            fitted_score["accuracy"] - handtuned_score["accuracy"],
        "delta_log_loss_fitted_vs_handtuned":
            handtuned_score["avg_log_loss"] - fitted_score["avg_log_loss"],
    }

    print("\n=== Held-out prediction quality ===")
    print(json.dumps(summary, indent=2))
    print()

    with allure.step("Held-out metrics"):
        allure.attach(
            json.dumps(summary, indent=2),
            name="heldout_metrics.json",
            attachment_type=allure.attachment_type.JSON,
        )
        allure.attach(
            json.dumps({
                "fitted": fitted_score["per_task"],
                "hand_tuned": handtuned_score["per_task"],
            }, indent=2),
            name="per_task_predictions.json",
            attachment_type=allure.attachment_type.JSON,
        )

    # Sanity assertions — do not fail the test on performance, only on plumbing
    assert len(train_samples) == N_TRAIN * 2 * len(HE_CRITIC_NAMES)
    assert len(test_samples) == N_TEST * 2 * len(HE_CRITIC_NAMES)
    assert set(fitted_lk.keys()) == set(HE_CRITIC_NAMES)
