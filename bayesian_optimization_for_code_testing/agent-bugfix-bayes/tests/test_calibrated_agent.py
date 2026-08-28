"""End-to-end comparison: hand-tuned Bayes DP vs empirical-Bayes-calibrated Bayes DP.

Workflow:
    1. Stratified split of the 50 bugs into train / test.
    2. Collect calibration data on train set (1 LLM call × N_arms × train bugs;
       plus full test + all critics per patch).
    3. Fit Beta-Binomial posterior mean likelihoods.
    4. Run DP agent on test set with HAND-TUNED likelihoods (baseline).
    5. Run DP agent on test set with CALIBRATED likelihoods (main claim).
    6. Attach before/after comparison + per-bug utilities to Allure.

Total cost (default config): ~train_n × 4 arms × 1 min  +  2 × test_n × 1 min
At train_n=10, test_n=10 → ~60 min with gpt-oss-120b.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import allure
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.bayes_agent import CRITIC_LIKELIHOODS, run_bayes_agent
from abbo.realworld.agents.calibration import (
    calibration_report,
    collect_calibration_data,
    format_comparison,
)
from abbo.realworld.agents.llm_provider import LLMConfig, is_openai_endpoint_available
from abbo.realworld.agents.real_bugs import BUGGY_SOURCES_REAL, BUG_METADATA
from abbo.realworld.agents.simple_agent import AgentCostConfig


# --- LLM backend (matches test_agent_comparison.py) ---
INNOPOLIS_BASE = "https://openai.global.ai.innopolis.university"
INNOPOLIS_MODEL = "gpt-oss-120b"

LLM = LLMConfig(
    provider="openai",
    model=INNOPOLIS_MODEL,
    base_url=INNOPOLIS_BASE,
    temperature=0.3,
    max_tokens=8192,
    timeout=600,
)
COSTS = AgentCostConfig()

# Knobs — tune for time budget
N_TRAIN = 10    # bugs used for calibration data
N_TEST = 10     # bugs used for held-out comparison
CALIB_ARMS = ["g1_direct_fix", "g2_localized_fix"]   # 2 arms × N_TRAIN LLM calls
SPLIT_SEED = 42

pytestmark = pytest.mark.skipif(
    not is_openai_endpoint_available(INNOPOLIS_BASE),
    reason=f"LLM backend not reachable: {INNOPOLIS_BASE}",
)

# Where we stash calibration output so the test order chain can pick it up
_CALIB_PATH = Path(__file__).parent / "_calibrated_likelihoods.json"


def _stratified_split() -> tuple[list[str], list[str]]:
    """Deterministic stratified split by difficulty."""
    by_diff: dict[str, list[str]] = defaultdict(list)
    for bug_id in sorted(BUGGY_SOURCES_REAL.keys()):
        by_diff[BUG_METADATA[bug_id]["difficulty"]].append(bug_id)

    rng = random.Random(SPLIT_SEED)
    train, test = [], []
    for diff in ("easy", "medium", "hard"):
        bugs = by_diff[diff][:]
        rng.shuffle(bugs)
        n_train_d = max(1, round(N_TRAIN * len(bugs) / len(BUGGY_SOURCES_REAL)))
        n_test_d = max(1, round(N_TEST * len(bugs) / len(BUGGY_SOURCES_REAL)))
        train.extend(bugs[:n_train_d])
        test.extend(bugs[n_train_d:n_train_d + n_test_d])
    return train, test


# ---------------------------------------------------------------------------
# Step A — fit calibration from train set and write to disk
# ---------------------------------------------------------------------------

@allure.parent_suite("calibrated_agent")
@allure.suite("calibration")
@allure.title("A. Fit empirical-Bayes likelihoods from train set")
def test_fit_calibration() -> None:
    """Collect samples on train bugs + fit Beta-Binomial posterior."""
    train_bugs, test_bugs = _stratified_split()
    allure.dynamic.description(
        f"Train bugs ({len(train_bugs)}): {train_bugs}\n"
        f"Test bugs ({len(test_bugs)}): {test_bugs}\n"
        f"Arms used for calibration: {CALIB_ARMS}\n"
        f"LLM: {INNOPOLIS_MODEL}"
    )

    with allure.step(f"Collect {len(train_bugs)} × {len(CALIB_ARMS)} = "
                     f"{len(train_bugs) * len(CALIB_ARMS)} patches"):
        samples = collect_calibration_data(
            train_bugs, LLM, arms=CALIB_ARMS, verbose=True
        )

    report = calibration_report(samples, prior_alpha=1.0, prior_beta=1.0)

    with allure.step("Fitted table vs hand-tuned"):
        allure.attach(
            format_comparison(report),
            name="calibration_comparison.txt",
            attachment_type=allure.attachment_type.TEXT,
        )
        allure.attach(
            json.dumps({
                "hand_tuned": report.hand_tuned,
                "calibrated": report.calibrated,
                "per_critic_counts": report.per_critic_counts,
                "n_bugs": report.n_bugs,
                "n_samples": report.n_samples,
            }, indent=2),
            name="likelihoods.json",
            attachment_type=allure.attachment_type.JSON,
        )

    _CALIB_PATH.write_text(json.dumps({
        "calibrated": report.calibrated,
        "hand_tuned": report.hand_tuned,
        "train_bugs": train_bugs,
        "test_bugs": test_bugs,
    }, indent=2))

    assert len(samples) > 0
    assert set(report.calibrated.keys()) == set(CRITIC_LIKELIHOODS.keys())


# ---------------------------------------------------------------------------
# Step B — run both agents on the test set, side by side
# ---------------------------------------------------------------------------

def _load_calibration() -> dict:
    if not _CALIB_PATH.exists():
        pytest.skip("Run test_fit_calibration first — no calibrated likelihoods found.")
    return json.loads(_CALIB_PATH.read_text())


def _attach_agent_result(r, kind: str) -> dict:
    meta = BUG_METADATA.get(r.bug_id, {})
    d = {
        "kind": kind,
        "bug_id": r.bug_id,
        "difficulty": meta.get("difficulty", "?"),
        "fixed": r.fixed,
        "total_cost": r.total_cost,
        "reward": r.reward,
        "utility": r.utility,
        "llm_calls": r.llm_calls,
        "test_runs": r.test_runs,
        "critic_runs": r.critic_runs,
        "belief_timeline": [round(b, 3) for b in r.belief_timeline],
        "action_log": r.action_log,
    }
    allure.attach(
        json.dumps(d, indent=2, default=str),
        name=f"{kind}_result.json",
        attachment_type=allure.attachment_type.JSON,
    )
    return d


@allure.parent_suite("calibrated_agent")
@allure.suite("comparison")
@allure.title("B. DP agent: hand-tuned vs calibrated on held-out test set")
def test_compare_handtuned_vs_calibrated() -> None:
    """Run both agents on every test bug and attach per-bug + aggregate utility."""
    data = _load_calibration()
    test_bugs = data["test_bugs"]
    cal_lk = data["calibrated"]

    hand_results, cal_results = [], []

    for bug_id in test_bugs:
        meta = BUG_METADATA[bug_id]
        with allure.step(f"{bug_id} ({meta['difficulty']}) — hand-tuned"):
            r_hand = run_bayes_agent(
                bug_id, LLM, COSTS, use_dp=True,
                policy_name="bayes_pomdp_handtuned",
            )
            hand_results.append(_attach_agent_result(r_hand, "hand"))

        with allure.step(f"{bug_id} ({meta['difficulty']}) — calibrated"):
            r_cal = run_bayes_agent(
                bug_id, LLM, COSTS, use_dp=True,
                critic_likelihoods=cal_lk,
                policy_name="bayes_pomdp_calibrated",
            )
            cal_results.append(_attach_agent_result(r_cal, "calibrated"))

    def _agg(results: list[dict]) -> dict:
        n = len(results)
        return {
            "n": n,
            "fix_rate": sum(1 for r in results if r["fixed"]) / n,
            "avg_cost": sum(r["total_cost"] for r in results) / n,
            "avg_utility": sum(r["utility"] for r in results) / n,
        }

    summary = {
        "hand_tuned": _agg(hand_results),
        "calibrated": _agg(cal_results),
        "delta_utility": (
            sum(r["utility"] for r in cal_results) / len(cal_results)
            - sum(r["utility"] for r in hand_results) / len(hand_results)
        ),
    }
    with allure.step("Aggregate comparison"):
        allure.attach(
            json.dumps(summary, indent=2),
            name="aggregate_comparison.json",
            attachment_type=allure.attachment_type.JSON,
        )

    assert summary["hand_tuned"]["n"] == summary["calibrated"]["n"] == len(test_bugs)
