"""Tests for policy consistency and budget constraints."""

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.synthetic.env import SyntheticEnv
from abbo.synthetic.exact_math import posterior_update, uniform_prior
from abbo.synthetic.policies import (
    BayesOptimalPolicy,
    OneTestIfFailElseRulePolicy,
    OneTestMAPPolicy,
    TunedThresholdHeuristic,
)


@pytest.fixture
def env() -> SyntheticEnv:
    return SyntheticEnv()


def test_bayes_never_exceeds_budget(env: SyntheticEnv) -> None:
    """Bayes policy should never run more than max_tests diagnostics."""
    rng = np.random.default_rng(42)
    policy = BayesOptimalPolicy()
    for _ in range(1000):
        result = policy.run_episode(env, rng)
        n_diag = sum(1 for a in result.actions if a.kind.value == "diagnostic")
        assert n_diag <= env.max_tests, f"Used {n_diag} diagnostics, max is {env.max_tests}"


def test_all_policies_return_valid_results(env: SyntheticEnv) -> None:
    """All policies should return results with required fields."""
    rng = np.random.default_rng(42)
    policies = [
        BayesOptimalPolicy(),
        OneTestMAPPolicy("T1"),
        OneTestIfFailElseRulePolicy(),
        TunedThresholdHeuristic(0.7),
    ]
    for policy in policies:
        result = policy.run_episode(env, rng)
        assert result.policy_name != ""
        assert result.hidden_class in env.classes
        assert len(result.actions) > 0
        assert result.verified is True


def test_costs_always_positive(env: SyntheticEnv) -> None:
    """Episode costs should never be negative."""
    rng = np.random.default_rng(42)
    for policy_cls in [BayesOptimalPolicy, OneTestMAPPolicy, OneTestIfFailElseRulePolicy]:
        policy = policy_cls() if policy_cls != OneTestMAPPolicy else policy_cls("T1")
        for _ in range(100):
            result = policy.run_episode(env, rng)
            assert result.costs.total_cost >= 0
            assert result.costs.diagnostic_cost >= 0
            assert result.costs.patch_cost >= 0
            assert result.costs.verifier_cost >= 0


def test_belief_update_monotonicity(env: SyntheticEnv) -> None:
    """Seeing a test fail for class A should increase belief in A."""
    prior = uniform_prior(env)
    post_fail = posterior_update(prior, "T1", True, env)
    post_pass = posterior_update(prior, "T1", False, env)
    # T1 has high fail rate for A, so fail should increase A belief
    assert post_fail["A"] > prior["A"]
    assert post_pass["A"] < prior["A"]
