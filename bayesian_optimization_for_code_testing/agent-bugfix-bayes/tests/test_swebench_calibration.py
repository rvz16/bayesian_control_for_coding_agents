"""Option A on SWE-bench Lite: real-repo bug fixes, Docker-isolated.

No LLM. For each instance we produce one Y=1 sample (dataset gold patch
applied) and one Y=0 sample (base_commit unchanged); run all critics on
each; fit Beta-Binomial; evaluate held-out prediction quality.

Requires Docker (Colima/Desktop) running and amd64 emulation.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import allure
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.bayes_agent import bayes_update
from abbo.realworld.agents.calibration import calibration_report, format_comparison
from abbo.realworld.agents.swe_bench import (
    SWE_CRITIC_LIKELIHOODS,
    SWE_CRITIC_NAMES,
    collect_calibration_samples_from_pairs,
    get_ftp,
    get_ptp,
    list_instance_ids,
)

# Knobs — deliberately small: Docker emulation on M4 is expensive.
N_TRAIN = 7
N_TEST = 4
SPLIT_SEED = 42
PRIOR_BELIEF = 0.5


def _docker_available() -> bool:
    import subprocess
    r = subprocess.run(
        ["docker", "info"], capture_output=True, text=True, timeout=10,
    )
    return r.returncode == 0


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not running")
@allure.parent_suite("swebench_calibration")
@allure.suite("option_a_no_llm")
@allure.title("Option A on SWE-bench Lite: fit + eval on real gold vs base pairs")
def test_swebench_calibration_quality() -> None:
    rng = random.Random(SPLIT_SEED)
    all_ids = list_instance_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    test_ids = all_ids[N_TRAIN:N_TRAIN + N_TEST]

    allure.dynamic.description(
        f"SWE-bench Lite curated pool: {len(all_ids)} small-dep instances.\n"
        f"Train: {N_TRAIN}  Test: {N_TEST}  seed={SPLIT_SEED}\n"
        f"Critics: {SWE_CRITIC_NAMES}\n"
        f"Y=1 = gold patch applied; Y=0 = base_commit unchanged.\n"
        f"Docker amd64 emulation on Apple Silicon — expect 20–40 min wall clock."
    )

    # --- Step 1: collect train samples + fit -----------------------------
    with allure.step(f"Collect train samples ({len(train_ids)} instances)"):
        train_samples, train_kept, train_timings = collect_calibration_samples_from_pairs(
            train_ids, verbose=True,
        )
        allure.attach(
            json.dumps([asdict(t) for t in train_timings], indent=2),
            name="train_timings.json",
            attachment_type=allure.attachment_type.JSON,
        )

    report = calibration_report(
        train_samples, prior_alpha=1.0, prior_beta=1.0,
        hand_tuned=SWE_CRITIC_LIKELIHOODS,
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
            "n_train_instances_requested": len(train_ids),
            "n_train_instances_kept": len(train_kept),
            "n_train_samples": report.n_samples,
            "train_instance_metadata": [
                {"id": iid, "ftp": len(get_ftp(iid)), "ptp": len(get_ptp(iid))}
                for iid in train_kept
            ],
        }, indent=2),
        name="likelihoods.json",
        attachment_type=allure.attachment_type.JSON,
    )

    # --- Step 2: collect test samples + predict held-out -----------------
    with allure.step(f"Collect test samples ({len(test_ids)} instances)"):
        test_samples, test_kept, test_timings = collect_calibration_samples_from_pairs(
            test_ids, verbose=True,
        )
        allure.attach(
            json.dumps([asdict(t) for t in test_timings], indent=2),
            name="test_timings.json",
            attachment_type=allure.attachment_type.JSON,
        )

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
                "instance": key[0], "arm": key[1], "y": y,
                "observations": [{"c": c, "p": p} for c, p in obs],
                "posterior": round(b, 4),
                "predicted": pred, "correct": pred == y,
            })
        n = max(1, len(by_patch))
        return {
            "label": label, "n_patches": len(by_patch),
            "accuracy": correct / n, "avg_log_loss": log_loss / n,
            "per_task": per_task,
        }

    fitted_score = _score(fitted_lk, "fitted")
    handtuned_score = _score(SWE_CRITIC_LIKELIHOODS, "hand_tuned")
    flat_lk = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in SWE_CRITIC_NAMES}
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

    print("\n=== SWE-bench Lite held-out prediction quality ===")
    print(json.dumps(summary, indent=2))
    print()

    with allure.step("Held-out metrics"):
        allure.attach(
            json.dumps(summary, indent=2), name="heldout_metrics.json",
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

    assert len(train_samples) == len(train_kept) * 2 * len(SWE_CRITIC_NAMES)
    assert len(test_samples) == len(test_kept) * 2 * len(SWE_CRITIC_NAMES)
    assert len(train_kept) >= N_TRAIN // 2, \
        f"only {len(train_kept)}/{N_TRAIN} train kept — Docker issues?"
    assert len(test_kept) >= N_TEST // 2, \
        f"only {len(test_kept)}/{N_TEST} test kept — Docker issues?"
    assert set(fitted_lk.keys()) == set(SWE_CRITIC_NAMES)
