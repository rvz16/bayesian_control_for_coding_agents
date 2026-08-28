"""Tests for Monte Carlo simulation of the synthetic benchmark."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.synthetic.env import SyntheticEnv
from abbo.synthetic.policies import BayesOptimalPolicy, OneTestMAPPolicy
from abbo.synthetic.simulate import compute_mc_stats, simulate_policy


@pytest.fixture
def env() -> SyntheticEnv:
    return SyntheticEnv()


def test_mc_bayes_close_to_exact(env: SyntheticEnv) -> None:
    """MC Bayes utility over 100k episodes should be close to 391/3."""
    results = simulate_policy(env, BayesOptimalPolicy(), 100_000, seed=7)
    stats = compute_mc_stats(results)
    exact = float(Fraction(391, 3))
    assert abs(stats["mean_utility"] - exact) < 1.5, (
        f"MC mean {stats['mean_utility']:.4f} too far from exact {exact:.4f}"
    )


def test_mc_one_test_map_close_to_exact(env: SyntheticEnv) -> None:
    """MC one-test MAP utility over 100k episodes should be close to 268/3."""
    results = simulate_policy(env, OneTestMAPPolicy("T1"), 100_000, seed=7)
    stats = compute_mc_stats(results)
    exact = float(Fraction(268, 3))
    assert abs(stats["mean_utility"] - exact) < 1.5, (
        f"MC mean {stats['mean_utility']:.4f} too far from exact {exact:.4f}"
    )


def test_mc_reproducibility(env: SyntheticEnv) -> None:
    """Same seed should produce identical results."""
    r1 = simulate_policy(env, BayesOptimalPolicy(), 1000, seed=42)
    r2 = simulate_policy(env, BayesOptimalPolicy(), 1000, seed=42)
    u1 = [r.utility for r in r1]
    u2 = [r.utility for r in r2]
    assert u1 == u2


def test_bayes_beats_heuristic(env: SyntheticEnv) -> None:
    """Bayes optimal should have higher mean utility than one-test MAP."""
    bayes = simulate_policy(env, BayesOptimalPolicy(), 10_000, seed=7)
    heuristic = simulate_policy(env, OneTestMAPPolicy("T1"), 10_000, seed=7)
    bayes_stats = compute_mc_stats(bayes)
    heur_stats = compute_mc_stats(heuristic)
    assert bayes_stats["mean_utility"] > heur_stats["mean_utility"]
