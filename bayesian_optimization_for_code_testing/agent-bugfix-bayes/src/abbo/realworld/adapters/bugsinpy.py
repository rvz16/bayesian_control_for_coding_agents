"""BugsInPy benchmark adapter.

Wraps BugsInPy CLI commands in Python subprocess calls with structured
logging of stdout, stderr, return code, duration, and command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from abbo.realworld.adapters.base import BenchmarkAdapter, CommandResult


class BugsInPyAdapter(BenchmarkAdapter):
    """Adapter for the BugsInPy benchmark suite."""

    def __init__(self, bugsinpy_path: str | None = None):
        self.bugsinpy_path = Path(
            bugsinpy_path or os.environ.get("BUGSINPY_PATH", "")
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if BugsInPy is installed by looking for the CLI script."""
        bugsinpy_path = os.environ.get("BUGSINPY_PATH", "")
        if bugsinpy_path:
            return (Path(bugsinpy_path) / "framework" / "bin" / "bugsinpy-checkout").exists()
        return shutil.which("bugsinpy-checkout") is not None

    def _run(self, cmd: list[str], cwd: Path | None = None) -> CommandResult:
        """Run a shell command and capture structured output."""
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            duration = time.perf_counter() - start
            return CommandResult(
                command=" ".join(cmd),
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
                duration_seconds=duration,
            )
        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start
            return CommandResult(
                command=" ".join(cmd),
                stdout="",
                stderr="Command timed out after 600s",
                returncode=-1,
                duration_seconds=duration,
            )

    def _bugsinpy_bin(self, tool: str) -> str:
        if self.bugsinpy_path and self.bugsinpy_path.exists():
            return str(self.bugsinpy_path / "framework" / "bin" / tool)
        return tool

    def checkout_bug(
        self, project: str, bug_id: int, version: str = "buggy", workdir: Path | None = None
    ) -> CommandResult:
        """Checkout a buggy or fixed version of a project."""
        cmd = [
            self._bugsinpy_bin("bugsinpy-checkout"),
            "-p", project,
            "-i", str(bug_id),
            "-v", "0" if version == "buggy" else "1",
        ]
        if workdir:
            cmd.extend(["-w", str(workdir)])
        return self._run(cmd)

    def compile(self, workdir: Path) -> CommandResult:
        """Compile/install the project."""
        return self._run(
            [self._bugsinpy_bin("bugsinpy-compile")],
            cwd=workdir,
        )

    def run_trigger_tests(self, workdir: Path) -> CommandResult:
        """Run the bug-triggering tests."""
        return self._run(
            [self._bugsinpy_bin("bugsinpy-test"), "-a"],
            cwd=workdir,
        )

    def run_all_tests(self, workdir: Path) -> CommandResult:
        """Run the full test suite."""
        return self._run(
            [self._bugsinpy_bin("bugsinpy-test"), "-a"],
            cwd=workdir,
        )

    def list_trigger_tests(self, workdir: Path) -> list[str]:
        """List trigger test names from the bug metadata."""
        trigger_file = workdir / ".bugsinpy" / "trigger_tests.txt"
        if trigger_file.exists():
            return [
                line.strip()
                for line in trigger_file.read_text().splitlines()
                if line.strip()
            ]
        return []

    def restore_clean_copy(self, workdir: Path) -> CommandResult:
        """Restore the workdir via git checkout."""
        return self._run(["git", "checkout", "."], cwd=workdir)
