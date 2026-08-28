"""Tests for exact symbolic computation of the synthetic benchmark."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.synthetic.env import SyntheticEnv
from abbo.synthetic.exact_math import (
    compute_exact_utilities,
    extract_tree,
    one_test_map_utility,
    posterior_update,
    solve_dp,
    terminal_value,
    uniform_prior,
)


@pytest.fixture
def env() -> SyntheticEnv:
    return SyntheticEnv()


def test_posterior_update_t1_fail(env: SyntheticEnv) -> None:
    """After T1 fails, belief in A should increase."""
    prior = uniform_prior(env)
    post = posterior_update(prior, "T1", True, env)
    assert post["A"] > prior["A"]
    assert post["B"] < prior["B"]
    assert post["C"] < prior["C"]
    assert abs(sum(post.values()) - 1) < Fraction(1, 10**9)


def test_posterior_update_t1_pass(env: SyntheticEnv) -> None:
    """After T1 passes, belief in A should decrease."""
    prior = uniform_prior(env)
    post = posterior_update(prior, "T1", False, env)
    assert post["A"] < prior["A"]
    assert abs(sum(post.values()) - 1) < Fraction(1, 10**9)


def test_exact_bayes_utility(env: SyntheticEnv) -> None:
    """Exact Bayes optimal expected utility must be 391/3."""
    val = solve_dp(env)
    assert val == Fraction(391, 3), f"Expected 391/3, got {val}"


def test_exact_one_test_map_utility(env: SyntheticEnv) -> None:
    """Exact one-test MAP expected utility must be 268/3."""
    val = one_test_map_utility(env, "T1")
    assert val == Fraction(268, 3), f"Expected 268/3, got {val}"


def test_terminal_value(env: SyntheticEnv) -> None:
    """Terminal value with certain belief should be R - Cpatch - Cver."""
    belief = {"A": Fraction(1), "B": Fraction(0), "C": Fraction(0)}
    tv = terminal_value(belief, env)
    expected = Fraction(200) - Fraction(3) - Fraction(20)
    assert tv == expected


def test_dp_values_monotonic(env: SyntheticEnv) -> None:
    """Higher certainty should yield higher terminal value."""
    b_high = {"A": Fraction(9, 10), "B": Fraction(1, 20), "C": Fraction(1, 20)}
    b_low = {"A": Fraction(1, 3), "B": Fraction(1, 3), "C": Fraction(1, 3)}
    assert terminal_value(b_high, env) > terminal_value(b_low, env)


def test_optimal_tree_structure(env: SyntheticEnv) -> None:
    """Optimal tree should start with T1, branch to T1 on fail, T2 on pass."""
    tree = extract_tree(env)
    assert tree["action"] == "test"
    assert tree["test"] == "T1"
    assert tree["fail"]["action"] == "test"
    assert tree["fail"]["test"] == "T1"  # T1 again after fail
    assert tree["pass"]["action"] == "test"
    assert tree["pass"]["test"] == "T2"  # T2 after pass


def test_compute_exact_utilities(env: SyntheticEnv) -> None:
    """compute_exact_utilities returns both expected values."""
    utils = compute_exact_utilities(env)
    assert utils["bayes_optimal"] == Fraction(391, 3)
    assert utils["one_test_map"] == Fraction(268, 3)
