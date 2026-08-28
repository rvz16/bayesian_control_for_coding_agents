"""Diagnostic calibration for the toy benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from abbo.toy.env import ToyEnv
from abbo.utils.seed import make_rng


def run_diagnostic_suite(
    env: ToyEnv,
    test_id: str,
    bug_class: str,
    n_trials: int,
    seed: int,
) -> float:
    """Run *n_trials* of a single diagnostic and return the empirical failure rate."""
    rng = make_rng(seed)
    env.inject_bug(bug_class)
    failures = 0
    for _ in range(n_trials):
        if env.run_diagnostic(test_id, rng):
            failures += 1
    env.remove_bug()
    return failures / n_trials


def calibrate_all(
    env: ToyEnv,
    n_trials: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Calibrate the full 3x3 diagnostic confusion matrix.

    Returns ``{test_id: {bug_class: empirical_fail_rate}}``.
    """
    rng = make_rng(seed)
    matrix: dict[str, dict[str, float]] = {}
    for test_id in env.tests:
        matrix[test_id] = {}
        for bug_class in env.classes:
            child_seed = int(rng.integers(0, 2**31))
            rate = run_diagnostic_suite(env, test_id, bug_class, n_trials, child_seed)
            matrix[test_id][bug_class] = rate
    return matrix


def print_calibration_report(matrix: dict[str, dict[str, float]]) -> None:
    """Print a formatted calibration report to the console."""
    console = Console()
    target = {
        "T1": {"A": 0.9, "B": 0.2, "C": 0.2},
        "T2": {"A": 0.2, "B": 0.9, "C": 0.2},
        "T3": {"A": 0.2, "B": 0.2, "C": 0.9},
    }

    table = Table(title="Diagnostic Calibration Matrix")
    table.add_column("Test", style="cyan")
    table.add_column("Bug A (target)", style="green")
    table.add_column("Bug B (target)", style="green")
    table.add_column("Bug C (target)", style="green")

    for test_id in ["T1", "T2", "T3"]:
        row = matrix[test_id]
        table.add_row(
            test_id,
            f"{row['A']:.3f} ({target[test_id]['A']:.1f})",
            f"{row['B']:.3f} ({target[test_id]['B']:.1f})",
            f"{row['C']:.3f} ({target[test_id]['C']:.1f})",
        )

    console.print(table)
