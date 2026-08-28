"""Core type definitions used throughout the ABBO toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    """The kind of action taken in an episode step."""

    DIAGNOSTIC = "diagnostic"
    PATCH = "patch"
    VERIFY = "verify"
    GENERATE = "generate"
    CRITIC = "critic"


@dataclass
class ActionRecord:
    """Record of a single action taken during an episode."""

    kind: ActionKind
    label: str
    observation: Any = None
    cost: float = 0.0


@dataclass
class CostBreakdown:
    """Itemised costs incurred during an episode."""

    diagnostic_cost: float = 0.0
    patch_cost: float = 0.0
    verifier_cost: float = 0.0
    generator_cost: float = 0.0
    critic_cost: float = 0.0
    total_cost: float = 0.0

    def compute_total(self) -> float:
        """Sum all cost components."""
        self.total_cost = (
            self.diagnostic_cost
            + self.patch_cost
            + self.verifier_cost
            + self.generator_cost
            + self.critic_cost
        )
        return self.total_cost


@dataclass
class EpisodeResult:
    """Complete result of a single episode (one bug-fix attempt)."""

    policy_name: str
    hidden_class: str = ""
    actions: list[ActionRecord] = field(default_factory=list)
    costs: CostBreakdown = field(default_factory=CostBreakdown)
    patch_chosen: str = ""
    patch_correct: bool = False
    verified: bool = False
    reward: float = 0.0
    utility: float = 0.0
    belief_timeline: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated results from running a benchmark."""

    benchmark_name: str
    policy_name: str
    episodes: list[EpisodeResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
