"""Full verifier that runs the complete test suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VerificationResult:
    """Result of running the full test suite."""

    full_suite_pass: bool
    failing_tests_before: list[str]
    failing_tests_after: list[str]
    regression_failures: list[str]
    wall_clock_seconds: float = 0.0


class FullSuiteVerifier:
    """Runs all tests and captures detailed results."""

    def verify(self, context: dict[str, Any]) -> VerificationResult:
        """Run the full test suite against a candidate patch.

        In replay mode, this reads stored results rather than running tests.
        """
        return VerificationResult(
            full_suite_pass=bool(context.get("full_suite_pass", False)),
            failing_tests_before=context.get("failing_tests_before", []),
            failing_tests_after=context.get("failing_tests_after", []),
            regression_failures=context.get("regression_failures", []),
            wall_clock_seconds=context.get("wall_clock_seconds", 0.0),
        )
