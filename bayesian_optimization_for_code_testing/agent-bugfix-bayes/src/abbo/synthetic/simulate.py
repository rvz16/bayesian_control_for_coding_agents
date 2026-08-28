"""Monte Carlo simulation for the synthetic benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from abbo.core.types import EpisodeResult
from abbo.synthetic.env import SyntheticEnv
from abbo.synthetic.policies import SyntheticPolicy
from abbo.utils.seed import make_rng


def simulate_policy(
    env: SyntheticEnv,
    policy: SyntheticPolicy,
    n_episodes: int,
    seed: int,
) -> list[EpisodeResult]:
    """Run *n_episodes* of *policy* on *env* and return all results.

    Args:
        env: The synthetic environment.
        policy: A policy implementing ``run_episode``.
        n_episodes: Number of episodes to simulate.
        seed: RNG seed for reproducibility.

    Returns:
        List of EpisodeResult objects.
    """
    rng = make_rng(seed)
    results = []
    for _ in range(n_episodes):
        result = policy.run_episode(env, rng)
        results.append(result)
    return results


def compute_mc_stats(results: list[EpisodeResult]) -> dict[str, float]:
    """Compute Monte Carlo statistics from a list of episode results.

    Returns dict with mean_utility, std_utility, ci_lower, ci_upper,
    resolved_rate, n_episodes.
    """
    utilities = [r.utility for r in results]
    n = len(utilities)
    arr = np.array(utilities)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    resolved = sum(1 for r in results if r.patch_correct) / n if n > 0 else 0.0

    return {
        "policy": results[0].policy_name if results else "",
        "n_episodes": n,
        "mean_utility": mean,
        "std_utility": std,
        "se_utility": float(se),
        "ci_lower": mean - 1.96 * se,
        "ci_upper": mean + 1.96 * se,
        "resolved_rate": resolved,
    }


def compare_policies(
    env: SyntheticEnv,
    policies: list[SyntheticPolicy],
    n_episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run multiple policies and return a comparison table.

    All policies use the same base seed but get independent RNG streams.
    """
    rows = []
    for i, policy in enumerate(policies):
        # Each policy gets its own seed derived from the base
        policy_seed = seed + i * 1_000_000
        results = simulate_policy(env, policy, n_episodes, policy_seed)
        stats = compute_mc_stats(results)
        rows.append(stats)
    return rows
