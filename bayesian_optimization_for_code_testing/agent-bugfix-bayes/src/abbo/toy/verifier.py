"""Full test suite verifier for the toy benchmark."""

from __future__ import annotations

from typing import Any

from abbo.toy.env import ToyEnv


def verify(env: ToyEnv) -> bool:
    """Run the complete test suite.  Returns True if all tests pass."""
    all_pass, _ = env.run_full_suite()
    return all_pass


def get_detailed_results(env: ToyEnv) -> dict[str, Any]:
    """Run the complete test suite and return per-test results."""
    all_pass, results = env.run_full_suite()
    return {
        "all_pass": all_pass,
        "n_passed": sum(1 for v in results.values() if v),
        "n_failed": sum(1 for v in results.values() if not v),
        "n_total": len(results),
        "per_test": results,
        "failed_tests": [t for t, v in results.items() if not v],
    }
