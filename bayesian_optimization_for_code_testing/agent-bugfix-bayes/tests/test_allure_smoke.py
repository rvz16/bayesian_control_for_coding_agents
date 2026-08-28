"""Allure smoke test: runs a tiny synthetic benchmark and attaches artifacts.

Run with::

    pytest tests/test_allure_smoke.py --alluredir allure-results --clean-alluredir
    allure serve allure-results
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import allure
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.synthetic.env import SyntheticEnv
from abbo.synthetic.exact_math import (
    compute_exact_utilities,
    extract_tree,
    uniform_prior,
)
from abbo.synthetic.policies import BayesOptimalPolicy, OneTestMAPPolicy
from abbo.synthetic.simulate import compute_mc_stats, simulate_policy


@allure.parent_suite("synthetic")
@allure.suite("bayes_optimal")
@allure.sub_suite("exact")
@allure.title("Exact Bayes utility equals 391/3")
def test_exact_bayes_utility_allure() -> None:
    """Verify exact Bayes value and attach the decision tree."""
    env = SyntheticEnv()

    with allure.step("Compute exact utilities"):
        utilities = compute_exact_utilities(env)
        bayes_val = utilities["bayes_optimal"]
        one_test_val = utilities["one_test_map"]

    with allure.step("Verify Bayes optimal = 391/3"):
        assert bayes_val == Fraction(391, 3)

    with allure.step("Verify one-test MAP = 268/3"):
        assert one_test_val == Fraction(268, 3)

    with allure.step("Extract optimal decision tree"):
        tree = extract_tree(env)

    # Attach artifacts
    allure.attach(
        json.dumps(
            {k: {"fraction": str(v), "decimal": float(v)} for k, v in utilities.items()},
            indent=2,
        ),
        name="exact_utilities.json",
        attachment_type=allure.attachment_type.JSON,
    )
    allure.attach(
        json.dumps(tree, indent=2),
        name="optimal_tree.json",
        attachment_type=allure.attachment_type.JSON,
    )


@allure.parent_suite("synthetic")
@allure.suite("bayes_optimal")
@allure.sub_suite("monte_carlo")
@allure.title("MC Bayes utility close to exact (1000 episodes)")
def test_mc_bayes_allure() -> None:
    """Run a small MC simulation and attach per-episode results."""
    env = SyntheticEnv()
    policy = BayesOptimalPolicy()

    with allure.step("Simulate 1000 episodes"):
        results = simulate_policy(env, policy, 1000, seed=7)
        stats = compute_mc_stats(results)

    with allure.step("Check mean utility within tolerance"):
        exact = float(Fraction(391, 3))
        assert abs(stats["mean_utility"] - exact) < 5.0

    # Attach stats summary
    allure.attach(
        json.dumps(stats, indent=2),
        name="mc_stats.json",
        attachment_type=allure.attachment_type.JSON,
    )

    # Attach first episode's belief timeline as example
    first = results[0]
    allure.attach(
        json.dumps(
            {
                "policy": first.policy_name,
                "hidden_class": first.hidden_class,
                "patch_chosen": first.patch_chosen,
                "patch_correct": first.patch_correct,
                "utility": first.utility,
                "actions": [
                    {
                        "kind": a.kind.value,
                        "label": a.label,
                        "observation": a.observation,
                        "cost": a.cost,
                    }
                    for a in first.actions
                ],
                "belief_timeline": first.belief_timeline,
            },
            indent=2,
        ),
        name="example_episode.json",
        attachment_type=allure.attachment_type.JSON,
    )

    # Attach cost breakdown
    allure.attach(
        json.dumps(
            {
                "diagnostic_cost": first.costs.diagnostic_cost,
                "patch_cost": first.costs.patch_cost,
                "verifier_cost": first.costs.verifier_cost,
                "total_cost": first.costs.total_cost,
            },
            indent=2,
        ),
        name="cost_breakdown.json",
        attachment_type=allure.attachment_type.JSON,
    )


@allure.parent_suite("synthetic")
@allure.suite("one_test_map")
@allure.sub_suite("monte_carlo")
@allure.title("MC one-test MAP utility close to exact (1000 episodes)")
def test_mc_one_test_map_allure() -> None:
    """Run one-test MAP and attach comparison data."""
    env = SyntheticEnv()

    with allure.step("Simulate Bayes optimal"):
        bayes_results = simulate_policy(env, BayesOptimalPolicy(), 1000, seed=7)
        bayes_stats = compute_mc_stats(bayes_results)

    with allure.step("Simulate one-test MAP"):
        heur_results = simulate_policy(env, OneTestMAPPolicy("T1"), 1000, seed=7)
        heur_stats = compute_mc_stats(heur_results)

    with allure.step("Verify Bayes beats heuristic"):
        assert bayes_stats["mean_utility"] > heur_stats["mean_utility"]

    # Attach comparison table
    allure.attach(
        json.dumps(
            {
                "bayes_optimal": bayes_stats,
                "one_test_map": heur_stats,
                "difference": bayes_stats["mean_utility"] - heur_stats["mean_utility"],
            },
            indent=2,
        ),
        name="policy_comparison.json",
        attachment_type=allure.attachment_type.JSON,
    )


@allure.parent_suite("synthetic")
@allure.suite("bayes_optimal")
@allure.sub_suite("config")
@allure.title("Synthetic benchmark config is valid")
def test_config_attachment_allure() -> None:
    """Attach the synthetic config as an Allure artifact."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "synthetic.yaml"
    if config_path.exists():
        allure.attach.file(
            str(config_path),
            name="synthetic.yaml",
            attachment_type=allure.attachment_type.TEXT,
        )

    env = SyntheticEnv()
    config_data = {
        "classes": env.classes,
        "tests": env.tests,
        "patches": env.patches,
        "costs": {
            "c_test": env.c_test,
            "c_patch": env.c_patch,
            "c_verifier": env.c_verifier,
            "reward": env.reward,
        },
        "max_tests": env.max_tests,
        "fail_probs": env.fail_probs,
    }
    allure.attach(
        json.dumps(config_data, indent=2),
        name="env_config.json",
        attachment_type=allure.attachment_type.JSON,
    )
    assert len(env.classes) == 3
