"""Heuristic baseline policies for real-world bug fixing.

All heuristics use the same generator arms, critics, and verifier
as the Bayesian controller for fair comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HeuristicEpisodeResult:
    """Result of running a heuristic policy on one bug."""

    policy_name: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    total_cost: float = 0.0
    utility: float = 0.0


class HeuristicPolicy(ABC):
    """Base class for heuristic policies."""

    name: str = "base"

    @abstractmethod
    def run_episode(self, context: dict[str, Any]) -> HeuristicEpisodeResult:
        """Run the policy on a single bug using replay data."""
        ...


class H1OneShot(HeuristicPolicy):
    """Direct fix once, then full verify."""

    name = "h1_one_shot"

    def run_episode(self, context: dict[str, Any]) -> HeuristicEpisodeResult:
        actions = [
            {"type": "generate", "arm": "g1_direct_fix"},
            {"type": "verify"},
        ]
        success = context.get("g1_direct_fix_pass", False)
        c_gen = context.get("c_generator", 3.0)
        c_ver = context.get("c_verifier", 20.0)
        reward = context.get("reward", 200.0) if success else 0.0
        total_cost = c_gen + c_ver
        return HeuristicEpisodeResult(
            policy_name=self.name,
            actions=actions,
            success=success,
            total_cost=total_cost,
            utility=reward - total_cost,
        )


class H2FixedWorkflow(HeuristicPolicy):
    """trigger_tests -> localized_fix -> full verify."""

    name = "h2_fixed_workflow"

    def run_episode(self, context: dict[str, Any]) -> HeuristicEpisodeResult:
        actions = [
            {"type": "critic", "critic": "trigger_tests_pass"},
            {"type": "generate", "arm": "g2_localized_fix"},
            {"type": "verify"},
        ]
        success = context.get("g2_localized_fix_pass", False)
        c_crit = context.get("c_critic", 1.0)
        c_gen = context.get("c_generator", 3.0)
        c_ver = context.get("c_verifier", 20.0)
        reward = context.get("reward", 200.0) if success else 0.0
        total_cost = c_crit + c_gen + c_ver
        return HeuristicEpisodeResult(
            policy_name=self.name,
            actions=actions,
            success=success,
            total_cost=total_cost,
            utility=reward - total_cost,
        )


class H3TwoStageFixed(HeuristicPolicy):
    """direct_fix -> trigger_tests -> if fail then test_guided_fix -> verify."""

    name = "h3_two_stage_fixed"

    def run_episode(self, context: dict[str, Any]) -> HeuristicEpisodeResult:
        actions = [{"type": "generate", "arm": "g1_direct_fix"}]
        c_gen = context.get("c_generator", 3.0)
        total_cost = c_gen

        # Run trigger tests as critic
        trigger_pass = context.get("g1_trigger_pass", False)
        actions.append({"type": "critic", "critic": "trigger_tests_pass", "result": trigger_pass})
        total_cost += context.get("c_critic", 1.0)

        if not trigger_pass:
            actions.append({"type": "generate", "arm": "g3_test_guided_fix"})
            total_cost += c_gen
            success = context.get("g3_test_guided_fix_pass", False)
        else:
            success = context.get("g1_direct_fix_pass", False)

        actions.append({"type": "verify"})
        total_cost += context.get("c_verifier", 20.0)
        reward = context.get("reward", 200.0) if success else 0.0

        return HeuristicEpisodeResult(
            policy_name=self.name,
            actions=actions,
            success=success,
            total_cost=total_cost,
            utility=reward - total_cost,
        )


class H4Threshold(HeuristicPolicy):
    """Hand-built threshold policy using a posterior estimate."""

    name = "h4_threshold"

    def __init__(self, verify_threshold: float = 0.8, regenerate_threshold: float = 0.3):
        self.verify_threshold = verify_threshold
        self.regenerate_threshold = regenerate_threshold

    def run_episode(self, context: dict[str, Any]) -> HeuristicEpisodeResult:
        actions = []
        total_cost = 0.0
        belief = context.get("initial_belief", 0.1)

        # Generate first candidate
        actions.append({"type": "generate", "arm": "g1_direct_fix"})
        total_cost += context.get("c_generator", 3.0)

        # Check trigger tests
        trigger_pass = context.get("g1_trigger_pass", False)
        actions.append({"type": "critic", "critic": "trigger_tests_pass"})
        total_cost += context.get("c_critic", 1.0)

        if trigger_pass:
            belief = min(0.9, belief * 3)
        else:
            belief = max(0.05, belief * 0.3)

        if belief < self.regenerate_threshold:
            actions.append({"type": "generate", "arm": "g3_test_guided_fix"})
            total_cost += context.get("c_generator", 3.0)

        actions.append({"type": "verify"})
        total_cost += context.get("c_verifier", 20.0)

        success = context.get("final_pass", False)
        reward = context.get("reward", 200.0) if success else 0.0

        return HeuristicEpisodeResult(
            policy_name=self.name,
            actions=actions,
            success=success,
            total_cost=total_cost,
            utility=reward - total_cost,
        )


class H5SingleCriticThenMAP(HeuristicPolicy):
    """One cheap critic, choose best generator arm, then verify."""

    name = "h5_single_critic_then_map"

    def run_episode(self, context: dict[str, Any]) -> HeuristicEpisodeResult:
        actions = []
        total_cost = 0.0

        # Run smoke tests as cheap critic
        actions.append({"type": "critic", "critic": "smoke_tests_pass"})
        total_cost += context.get("c_critic", 1.0)

        # Choose the arm with highest empirical success rate
        best_arm = context.get("best_arm", "g1_direct_fix")
        actions.append({"type": "generate", "arm": best_arm})
        total_cost += context.get("c_generator", 3.0)

        actions.append({"type": "verify"})
        total_cost += context.get("c_verifier", 20.0)

        success = context.get(f"{best_arm}_pass", False)
        reward = context.get("reward", 200.0) if success else 0.0

        return HeuristicEpisodeResult(
            policy_name=self.name,
            actions=actions,
            success=success,
            total_cost=total_cost,
            utility=reward - total_cost,
        )


ALL_HEURISTICS: dict[str, HeuristicPolicy] = {
    "h1_one_shot": H1OneShot(),
    "h2_fixed_workflow": H2FixedWorkflow(),
    "h3_two_stage_fixed": H3TwoStageFixed(),
    "h4_threshold": H4Threshold(),
    "h5_single_critic_then_map": H5SingleCriticThenMAP(),
}
