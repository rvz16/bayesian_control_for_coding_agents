"""Metric computation from episode results."""

from __future__ import annotations

from typing import Any

import numpy as np

from abbo.core.types import EpisodeResult


def compute_metrics(episodes: list[EpisodeResult]) -> dict[str, Any]:
    """Compute aggregate metrics over a list of episode results.

    Returns a dict with resolved_rate, mean_utility, mean_verifier_calls,
    mean_critic_calls, mean_generator_calls, mean_wall_clock, and counts.
    """
    if not episodes:
        return {}

    n = len(episodes)
    resolved = sum(1 for e in episodes if e.patch_correct and e.verified)
    utilities = [e.utility for e in episodes]
    verifier_calls = [
        sum(1 for a in e.actions if a.kind.value == "verify") for e in episodes
    ]
    critic_calls = [
        sum(1 for a in e.actions if a.kind.value in ("critic", "diagnostic"))
        for e in episodes
    ]
    generator_calls = [
        sum(1 for a in e.actions if a.kind.value in ("generate", "patch"))
        for e in episodes
    ]

    return {
        "n_episodes": n,
        "resolved_rate": resolved / n,
        "mean_utility": float(np.mean(utilities)),
        "std_utility": float(np.std(utilities, ddof=1)) if n > 1 else 0.0,
        "mean_verifier_calls": float(np.mean(verifier_calls)),
        "mean_critic_calls": float(np.mean(critic_calls)),
        "mean_generator_calls": float(np.mean(generator_calls)),
        "total_cost": sum(e.costs.total_cost for e in episodes),
    }


def per_class_metrics(
    episodes: list[EpisodeResult],
) -> dict[str, dict[str, Any]]:
    """Break down metrics by hidden class / project."""
    by_class: dict[str, list[EpisodeResult]] = {}
    for e in episodes:
        by_class.setdefault(e.hidden_class, []).append(e)
    return {cls: compute_metrics(eps) for cls, eps in sorted(by_class.items())}
