"""Option A on CodeContests: harder benchmark, real (correct, incorrect) pools.

No LLM calls. For each task we take the dataset's Python3 accepted and
rejected submissions as balanced (Y=1, Y=0) calibration data, then evaluate
held-out prediction quality for fitted vs hand-tuned likelihoods.

Why this stresses the calibration harder than HumanEvalFix:
    - Codeforces-level difficulty (cf_rating 800..3500)
    - Y=0 samples are real human-written wrong submissions, not hand bugs
    - critic_early / critic_mid run stdin/stdout tests, not assert subsets
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import allure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.bayes_agent import bayes_update
from abbo.realworld.agents.calibration import (
    calibration_report,
    format_comparison,
)
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS,
    CC_CRITIC_NAMES,
    collect_calibration_samples_from_pairs,
    get_metadata,
    list_task_ids,
)


# ---- Knobs ----
N_TRAIN = 30            # bumped: use most of 112 usable Py3 tasks
N_TEST = 80             # bumped: 30 train + 80 test = 110 of 112 usable
K_PER_SIDE = 1          # # correct + # incorrect per task (for speed)
SPLIT_SEED = 42
PRIOR_BELIEF = 0.5


@allure.parent_suite("codecontests_calibration")
@allure.suite("option_a_no_llm")
@allure.title("Option A on CodeContests: fit + eval on real (correct, incorrect) pools")
def test_codecontests_calibration_quality() -> None:
    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    test_ids = all_ids[N_TRAIN:N_TRAIN + N_TEST]

    allure.dynamic.description(
        f"CodeContests test split: {len(all_ids)} usable Python3 problems.\n"
        f"Train: {N_TRAIN}  Test: {N_TEST}  k_per_side={K_PER_SIDE}  "
        f"(split seed {SPLIT_SEED})\n"
        f"Critics: {CC_CRITIC_NAMES}\n"
        f"Prior belief for held-out prediction: {PRIOR_BELIEF}\n"
        f"No LLM calls — uses real accepted + real incorrect human submissions."
    )

    # --- Step 1: collect train samples + fit -----------------------------
    with allure.step(
        f"Collect train samples (up to {len(train_ids)} tasks × 2 × {len(CC_CRITIC_NAMES)} critics)"
    ):
        train_samples, train_kept = collect_calibration_samples_from_pairs(
            train_ids, k_per_side=K_PER_SIDE, verbose=True,
        )

    report = calibration_report(
        train_samples, prior_alpha=1.0, prior_beta=1.0,
        hand_tuned=CC_CRITIC_LIKELIHOODS,
    )
    fitted_lk = report.calibrated

    table_text = format_comparison(report)
    print("\n" + table_text + "\n")
    allure.attach(
        table_text, name="calibration_table.txt",
        attachment_type=allure.attachment_type.TEXT,
    )
    allure.attach(
        json.dumps({
            "hand_tuned": report.hand_tuned,
            "fitted": report.calibrated,
            "per_critic_counts": report.per_critic_counts,
            "n_train_tasks_requested": len(train_ids),
            "n_train_tasks_kept": len(train_kept),
            "n_train_samples": report.n_samples,
            "train_task_metadata": [get_metadata(t) for t in train_kept],
        }, indent=2, default=str),
        name="likelihoods.json",
        attachment_type=allure.attachment_type.JSON,
    )

    # --- Step 2: collect test samples + predict held-out -----------------
    with allure.step(f"Collect test samples (up to {len(test_ids)} tasks)"):
        test_samples, test_kept = collect_calibration_samples_from_pairs(
            test_ids, k_per_side=K_PER_SIDE, verbose=True,
        )

    # Group by (task_id, arm) → list of critic observations; each (task, arm)
    # is one "patch" with a single Y ground truth.
    by_patch: dict[tuple[str, str], list[tuple[str, bool]]] = defaultdict(list)
    patch_y: dict[tuple[str, str], bool] = {}
    for s in test_samples:
        key = (s.bug_id, s.arm)
        by_patch[key].append((s.critic, s.critic_passed))
        patch_y[key] = s.patch_correct

    def _predict(obs, lk):
        b = PRIOR_BELIEF
        for critic, passed in obs:
            b = bayes_update(b, critic, passed, likelihoods=lk)
        return b

    def _score(lk, label):
        correct = 0
        log_loss = 0.0
        per_task = []
        for key, obs in by_patch.items():
            y = patch_y[key]
            b = _predict(obs, lk)
            pred = b > 0.5
            correct += int(pred == y)
            eps = 1e-6
            b_clip = min(max(b, eps), 1 - eps)
            log_loss += -(y * math.log(b_clip) + (1 - y) * math.log(1 - b_clip))
            per_task.append({
                "task": key[0], "arm": key[1], "y": y,
                "observations": [{"c": c, "p": p} for c, p in obs],
                "posterior": round(b, 4),
                "predicted": pred, "correct": pred == y,
            })
        n = len(by_patch)
        return {
            "label": label, "n_patches": n,
            "accuracy": correct / n, "avg_log_loss": log_loss / n,
            "per_task": per_task,
        }

    fitted_score = _score(fitted_lk, "fitted")
    handtuned_score = _score(CC_CRITIC_LIKELIHOODS, "hand_tuned")
    flat_lk = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in CC_CRITIC_NAMES}
    flat_score = _score(flat_lk, "flat_prior_only")

    summary = {
        "fitted":     {k: v for k, v in fitted_score.items()    if k != "per_task"},
        "hand_tuned": {k: v for k, v in handtuned_score.items() if k != "per_task"},
        "flat_prior": {k: v for k, v in flat_score.items()      if k != "per_task"},
        "delta_accuracy_fitted_vs_handtuned":
            fitted_score["accuracy"] - handtuned_score["accuracy"],
        "delta_log_loss_fitted_vs_handtuned":
            handtuned_score["avg_log_loss"] - fitted_score["avg_log_loss"],
    }

    print("\n=== CodeContests held-out prediction quality ===")
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

    # Plumbing sanity checks — loose because tasks where our oracle can't
    # produce both a Y=1 and a Y=0 submission are skipped.
    assert len(train_samples) == len(train_kept) * 2 * K_PER_SIDE * len(CC_CRITIC_NAMES)
    assert len(test_samples) == len(test_kept) * 2 * K_PER_SIDE * len(CC_CRITIC_NAMES)
    assert len(train_kept) >= N_TRAIN // 2, \
        f"kept only {len(train_kept)}/{N_TRAIN} train tasks — too few"
    assert len(test_kept) >= N_TEST // 2, \
        f"kept only {len(test_kept)}/{N_TEST} test tasks — too few"
    assert set(fitted_lk.keys()) == set(CC_CRITIC_NAMES)
