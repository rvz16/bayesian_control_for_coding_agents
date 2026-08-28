"""Tests for the toy benchmark calibration mapping."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.toy.diagnostics import calibrate_all
from abbo.toy.env import ToyEnv


@pytest.fixture
def toy_env() -> ToyEnv:
    return ToyEnv()


def test_calibration_matrix_shape(toy_env: ToyEnv) -> None:
    """Calibration matrix should be 3x3 (3 tests x 3 bug classes)."""
    matrix = calibrate_all(toy_env, n_trials=1000, seed=42)
    assert len(matrix) == 3
    for test_id in ["T1", "T2", "T3"]:
        assert test_id in matrix
        assert len(matrix[test_id]) == 3


def test_calibration_diagonal_dominant(toy_env: ToyEnv) -> None:
    """Diagonal entries (e.g. T1/A, T2/B, T3/C) should be > 0.6."""
    matrix = calibrate_all(toy_env, n_trials=2000, seed=42)
    assert matrix["T1"]["A"] > 0.6, f"T1/A = {matrix['T1']['A']}"
    assert matrix["T2"]["B"] > 0.6, f"T2/B = {matrix['T2']['B']}"
    assert matrix["T3"]["C"] > 0.6, f"T3/C = {matrix['T3']['C']}"


def test_calibration_off_diagonal_low(toy_env: ToyEnv) -> None:
    """Off-diagonal entries should be < 0.5."""
    matrix = calibrate_all(toy_env, n_trials=2000, seed=42)
    assert matrix["T1"]["B"] < 0.5, f"T1/B = {matrix['T1']['B']}"
    assert matrix["T1"]["C"] < 0.5, f"T1/C = {matrix['T1']['C']}"
    assert matrix["T2"]["A"] < 0.5, f"T2/A = {matrix['T2']['A']}"
    assert matrix["T2"]["C"] < 0.5, f"T2/C = {matrix['T2']['C']}"
    assert matrix["T3"]["A"] < 0.5, f"T3/A = {matrix['T3']['A']}"
    assert matrix["T3"]["B"] < 0.5, f"T3/B = {matrix['T3']['B']}"


def test_toy_bug_injection(toy_env: ToyEnv) -> None:
    """Injecting bug A should cause priority tests to fail."""
    toy_env.inject_bug("A")
    all_pass, results = toy_env.run_full_suite()
    assert not all_pass
    # At least some priority tests should fail
    priority_fails = sum(
        1 for t, v in results.items() if "priority" in t and not v
    )
    assert priority_fails > 0
    toy_env.remove_bug()


def test_toy_patch_correct(toy_env: ToyEnv) -> None:
    """Applying the correct patch should fix the bug."""
    toy_env.inject_bug("A")
    assert toy_env.apply_patch("PA") is True
    assert toy_env.apply_patch("PB") is False


def test_toy_no_bug_passes(toy_env: ToyEnv) -> None:
    """Without any bug injected, full suite should pass."""
    toy_env.remove_bug()
    all_pass, _ = toy_env.run_full_suite()
    assert all_pass
