"""Plotting utilities for benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_utility_comparison(
    summary: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Bar chart comparing mean utility across policies."""
    policies = sorted(summary.keys())
    utilities = [summary[p].get("mean_utility", 0) for p in policies]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2196F3" if p == "bayes" else "#9E9E9E" for p in policies]
    ax.bar(policies, utilities, color=colors)
    ax.set_ylabel("Mean Utility")
    ax.set_title("Policy Comparison: Mean Utility")
    ax.axhline(y=0, color="black", linewidth=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_success_rate(
    summary: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Bar chart comparing resolved rates across policies."""
    policies = sorted(summary.keys())
    rates = [summary[p].get("resolved_rate", 0) for p in policies]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4CAF50" if p == "bayes" else "#9E9E9E" for p in policies]
    ax.bar(policies, rates, color=colors)
    ax.set_ylabel("Resolved Rate")
    ax.set_title("Policy Comparison: Resolved Rate")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_posterior_timeline(
    timeline: list[dict[str, float]],
    out_path: Path,
    title: str = "Posterior Belief Over Time",
) -> None:
    """Line plot of posterior belief over episode steps."""
    steps = list(range(len(timeline)))
    beliefs = [t.get("b", t.get("A", 0)) for t in timeline]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, beliefs, "o-", color="#2196F3")
    ax.set_xlabel("Step")
    ax.set_ylabel("P(Y=1 | history)")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
