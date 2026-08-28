"""Critic implementations for evaluating candidate patches.

Each critic produces a binary or categorical judgment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Critic(ABC):
    """Base class for critics."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, context: dict[str, Any]) -> bool | str:
        """Evaluate a candidate patch.  Returns binary or label."""
        ...


class TriggerTestsCritic(Critic):
    """Runs the bug-triggering tests against the patched code."""

    name = "trigger_tests_pass"

    def evaluate(self, context: dict[str, Any]) -> bool:
        return bool(context.get("trigger_tests_pass", False))


class SmokeTestsCritic(Critic):
    """Runs a small subset of quick smoke tests."""

    name = "smoke_tests_pass"

    def evaluate(self, context: dict[str, Any]) -> bool:
        return bool(context.get("smoke_tests_pass", False))


class StaticChecksCritic(Critic):
    """Runs static analysis / linting checks."""

    name = "static_checks_pass"

    def evaluate(self, context: dict[str, Any]) -> bool:
        return bool(context.get("static_checks_pass", False))


class CheapJudgeCritic(Critic):
    """Uses a fast LLM call to judge patch quality."""

    name = "cheap_judge_label"

    def evaluate(self, context: dict[str, Any]) -> str:
        return str(context.get("cheap_judge_label", "unknown"))


ALL_CRITICS: dict[str, Critic] = {
    "trigger_tests_pass": TriggerTestsCritic(),
    "smoke_tests_pass": SmokeTestsCritic(),
    "static_checks_pass": StaticChecksCritic(),
    "cheap_judge_label": CheapJudgeCritic(),
}
