"""Abstract base adapter for real-world bug benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CommandResult:
    """Result of a subprocess command execution."""

    command: str
    stdout: str
    stderr: str
    returncode: int
    duration_seconds: float


class BenchmarkAdapter(ABC):
    """Base class for benchmark adapters (BugsInPy, Defects4J, etc.)."""

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check whether the benchmark tool is installed and usable."""
        ...

    @abstractmethod
    def checkout_bug(
        self, project: str, bug_id: int, version: str, workdir: Path
    ) -> CommandResult:
        """Checkout a specific buggy or fixed version."""
        ...

    @abstractmethod
    def compile(self, workdir: Path) -> CommandResult:
        """Compile the project in *workdir*."""
        ...

    @abstractmethod
    def run_trigger_tests(self, workdir: Path) -> CommandResult:
        """Run only the bug-triggering tests."""
        ...

    @abstractmethod
    def run_all_tests(self, workdir: Path) -> CommandResult:
        """Run the full test suite."""
        ...

    @abstractmethod
    def list_trigger_tests(self, workdir: Path) -> list[str]:
        """Return the names of the trigger tests for the checked-out bug."""
        ...

    @abstractmethod
    def restore_clean_copy(self, workdir: Path) -> CommandResult:
        """Restore the workdir to its clean checked-out state."""
        ...
