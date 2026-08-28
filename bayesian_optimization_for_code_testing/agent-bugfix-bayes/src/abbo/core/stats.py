"""Paired bootstrap confidence interval computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    """Result of a paired bootstrap test."""

    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    n_bootstrap: int
    samples: np.ndarray | None = None


def paired_bootstrap(
    values_a: list[float],
    values_b: list[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    store_samples: bool = False,
) -> BootstrapCI:
    """Compute a paired bootstrap CI for the mean difference (A - B).

    Args:
        values_a: Per-bug utility values for policy A.
        values_b: Per-bug utility values for policy B.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95).
        seed: RNG seed for reproducibility.
        store_samples: Whether to keep all bootstrap samples in the result.

    Returns:
        BootstrapCI with mean difference and confidence interval.
    """
    assert len(values_a) == len(values_b), "Must have paired observations"
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    diffs = a - b
    n = len(diffs)

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean_diff = float(diffs.mean())
    p_value = float(np.mean(boot_means <= 0)) if mean_diff > 0 else float(
        np.mean(boot_means >= 0)
    )

    return BootstrapCI(
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_value,
        n_bootstrap=n_bootstrap,
        samples=boot_means if store_samples else None,
    )
