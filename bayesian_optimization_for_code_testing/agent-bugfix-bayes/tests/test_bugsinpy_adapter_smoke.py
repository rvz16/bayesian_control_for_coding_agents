"""Smoke tests for the BugsInPy adapter using mocked subprocess calls."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.adapters.bugsinpy import BugsInPyAdapter


def test_adapter_creation() -> None:
    """BugsInPyAdapter should instantiate without errors."""
    adapter = BugsInPyAdapter(bugsinpy_path="/tmp/fake_bugsinpy")
    assert adapter.bugsinpy_path == Path("/tmp/fake_bugsinpy")


def test_is_available_check() -> None:
    """is_available should return a boolean."""
    result = BugsInPyAdapter.is_available()
    assert isinstance(result, bool)


@patch("subprocess.run")
def test_checkout_mock(mock_run: MagicMock) -> None:
    """Mocked checkout should complete without errors."""
    mock_run.return_value = MagicMock(
        stdout="Checked out",
        stderr="",
        returncode=0,
    )
    adapter = BugsInPyAdapter(bugsinpy_path="/tmp/fake")
    result = adapter.checkout_bug("flask", 1, "buggy", Path("/tmp/workdir"))
    assert result.returncode == 0
    assert "Checked out" in result.stdout


@patch("subprocess.run")
def test_run_tests_mock(mock_run: MagicMock) -> None:
    """Mocked test run should complete without errors."""
    mock_run.return_value = MagicMock(
        stdout="5 passed",
        stderr="",
        returncode=0,
    )
    adapter = BugsInPyAdapter(bugsinpy_path="/tmp/fake")
    result = adapter.run_all_tests(Path("/tmp/workdir"))
    assert result.returncode == 0


@patch("subprocess.run")
def test_compile_mock(mock_run: MagicMock) -> None:
    """Mocked compile should complete without errors."""
    mock_run.return_value = MagicMock(
        stdout="Compiled",
        stderr="",
        returncode=0,
    )
    adapter = BugsInPyAdapter(bugsinpy_path="/tmp/fake")
    result = adapter.compile(Path("/tmp/workdir"))
    assert result.returncode == 0


def test_list_trigger_tests_no_file() -> None:
    """list_trigger_tests should return empty list if no file exists."""
    adapter = BugsInPyAdapter(bugsinpy_path="/tmp/fake")
    result = adapter.list_trigger_tests(Path("/tmp/nonexistent"))
    assert result == []
