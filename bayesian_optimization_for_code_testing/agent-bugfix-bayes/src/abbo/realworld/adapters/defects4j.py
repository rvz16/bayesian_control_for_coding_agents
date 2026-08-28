"""Defects4J benchmark adapter (optional, after BugsInPy is stable)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from abbo.realworld.adapters.base import BenchmarkAdapter, CommandResult


class Defects4JAdapter(BenchmarkAdapter):
    """Adapter for the Defects4J benchmark suite."""

    def __init__(self, defects4j_path: str | None = None):
        self.defects4j_path = Path(
            defects4j_path or os.environ.get("DEFECTS4J_PATH", "")
        )

    @classmethod
    def is_available(cls) -> bool:
        """Check if Defects4J is installed."""
        d4j_path = os.environ.get("DEFECTS4J_PATH", "")
        if d4j_path:
            return (Path(d4j_path) / "framework" / "bin" / "defects4j").exists()
        return shutil.which("defects4j") is not None

    def _run(self, cmd: list[str], cwd: Path | None = None) -> CommandResult:
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True, timeout=600
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

    def _d4j_bin(self) -> str:
        if self.defects4j_path and self.defects4j_path.exists():
            return str(self.defects4j_path / "framework" / "bin" / "defects4j")
        return "defects4j"

    def checkout_bug(
        self, project: str, bug_id: int, version: str = "buggy", workdir: Path | None = None
    ) -> CommandResult:
        target = workdir or Path(f"/tmp/d4j_{project}_{bug_id}")
        v = "b" if version == "buggy" else "f"
        return self._run([
            self._d4j_bin(), "checkout",
            "-p", project,
            "-v", f"{bug_id}{v}",
            "-w", str(target),
        ])

    def compile(self, workdir: Path) -> CommandResult:
        return self._run([self._d4j_bin(), "compile"], cwd=workdir)

    def run_trigger_tests(self, workdir: Path) -> CommandResult:
        return self._run([self._d4j_bin(), "test", "-r"], cwd=workdir)

    def run_all_tests(self, workdir: Path) -> CommandResult:
        return self._run([self._d4j_bin(), "test"], cwd=workdir)

    def list_trigger_tests(self, workdir: Path) -> list[str]:
        result = self._run(
            [self._d4j_bin(), "export", "-p", "tests.trigger"],
            cwd=workdir,
        )
        if result.returncode == 0:
            return [l.strip() for l in result.stdout.splitlines() if l.strip()]
        return []

    def restore_clean_copy(self, workdir: Path) -> CommandResult:
        return self._run(["git", "checkout", "."], cwd=workdir)
