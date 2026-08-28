"""Synthetic bug-fixing environment.

Implements the paper's synthetic benchmark with three hidden bug classes,
three diagnostic tests, and three targeted patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import numpy as np


# Default confusion matrix: P(fail | hidden_class, test)
DEFAULT_FAIL_PROBS: dict[str, dict[str, float]] = {
    "T1": {"A": 0.9, "B": 0.2, "C": 0.2},
    "T2": {"A": 0.2, "B": 0.9, "C": 0.2},
    "T3": {"A": 0.2, "B": 0.2, "C": 0.9},
}

CLASSES = ["A", "B", "C"]
TESTS = ["T1", "T2", "T3"]
PATCHES = ["PA", "PB", "PC"]


@dataclass
class SyntheticEnv:
    """Synthetic bug-fixing environment.

    Attributes:
        fail_probs: P(fail | class, test) as ``{test: {class: prob}}``.
        classes: List of hidden bug classes.
        tests: List of diagnostic test names.
        patches: List of patch action names.
        c_test: Cost per diagnostic test.
        c_patch: Cost per patch application.
        c_verifier: Cost per full verification.
        reward: Reward for successful fix.
        max_tests: Maximum number of diagnostic tests allowed.
    """

    fail_probs: dict[str, dict[str, float]] = field(
        default_factory=lambda: dict(DEFAULT_FAIL_PROBS)
    )
    classes: list[str] = field(default_factory=lambda: list(CLASSES))
    tests: list[str] = field(default_factory=lambda: list(TESTS))
    patches: list[str] = field(default_factory=lambda: list(PATCHES))
    c_test: float = 1.0
    c_patch: float = 3.0
    c_verifier: float = 20.0
    reward: float = 200.0
    max_tests: int = 2

    def sample_hidden(self, rng: np.random.Generator) -> str:
        """Sample a hidden bug class uniformly at random."""
        return str(rng.choice(self.classes))

    def run_diagnostic(
        self, hidden: str, test_id: str, rng: np.random.Generator
    ) -> bool:
        """Run a diagnostic test.  Returns True if the test *fails*."""
        p_fail = self.fail_probs[test_id][hidden]
        return bool(rng.random() < p_fail)

    def apply_patch(self, hidden: str, patch_id: str) -> bool:
        """Apply a patch.  Returns True if the patch fixes the bug."""
        # PA fixes A, PB fixes B, PC fixes C
        idx = self.patches.index(patch_id)
        return hidden == self.classes[idx]

    def get_fail_prob(self, hidden: str, test_id: str) -> float:
        """Return P(fail | hidden, test)."""
        return self.fail_probs[test_id][hidden]

    def get_fail_prob_fraction(self, hidden: str, test_id: str) -> Fraction:
        """Return P(fail | hidden, test) as an exact Fraction."""
        p = self.fail_probs[test_id][hidden]
        return Fraction(p).limit_denominator(100)

    def patch_for_class(self, cls: str) -> str:
        """Return the patch that fixes a given class."""
        idx = self.classes.index(cls)
        return self.patches[idx]

    def class_for_patch(self, patch_id: str) -> str:
        """Return the class that a given patch targets."""
        idx = self.patches.index(patch_id)
        return self.classes[idx]
