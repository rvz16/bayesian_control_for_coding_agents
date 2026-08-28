"""Deterministic seeding utilities."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """Create a reproducible numpy random generator from an integer seed."""
    return np.random.default_rng(seed)


def split_seed(rng: np.random.Generator, n: int = 2) -> list[int]:
    """Derive *n* child seeds from *rng* without advancing its state ambiguously."""
    return [int(x) for x in rng.integers(0, 2**31, size=n)]
