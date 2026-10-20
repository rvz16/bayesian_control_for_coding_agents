"""Shared bootstrap helper (same as action_traces/scripts/run_all.py:21-27)."""
import numpy as np


def boot_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))
