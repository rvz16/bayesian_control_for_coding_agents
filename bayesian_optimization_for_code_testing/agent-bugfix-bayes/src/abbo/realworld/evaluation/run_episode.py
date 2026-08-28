"""Episode runner for real-world benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from abbo.core.types import ActionKind, ActionRecord, CostBreakdown, EpisodeResult
from abbo.realworld.candidate_bank.replay import ReplayEngine
from abbo.realworld.policies.bayes_controller import (
    ActionChoice,
    BayesController,
    BudgetState,
)


@dataclass
class EpisodeConfig:
    """Configuration for a single evaluation episode."""

    project: str
    bug_id: int
    max_steps: int = 10
    reward: float = 200.0
    c_verifier: float = 20.0
    c_critic: float = 1.0
    c_generator: float = 3.0


def run_bayes_episode(
    controller: BayesController,
    replay: ReplayEngine,
    config: EpisodeConfig,
) -> EpisodeResult:
    """Run a single episode using the Bayesian controller with replay data."""
    belief = controller.model.prior_correct
    budget = BudgetState(remaining_steps=config.max_steps)
    actions: list[ActionRecord] = []
    belief_timeline = [{"b": belief}]
    costs = CostBreakdown()

    current_arm = "g1_direct_fix"
    current_seed = 0

    while budget.remaining_steps > 0:
        choice = controller.choose_action(belief, budget)

        if choice.action_type == "verify":
            candidate = replay.get_candidate(
                config.project, config.bug_id, current_arm, current_seed
            )
            success = False
            if candidate:
                success = candidate.verifier_outputs.full_suite_pass or False
            actions.append(ActionRecord(
                kind=ActionKind.VERIFY,
                label="full_suite",
                observation="pass" if success else "fail",
                cost=config.c_verifier,
            ))
            costs.verifier_cost += config.c_verifier
            break

        elif choice.action_type == "critic":
            candidate = replay.get_candidate(
                config.project, config.bug_id, current_arm, current_seed
            )
            result = False
            if candidate:
                result = bool(getattr(candidate.critic_outputs, choice.label, False))
            belief = controller.bayes_update_belief(belief, choice.label, result)
            actions.append(ActionRecord(
                kind=ActionKind.CRITIC,
                label=choice.label,
                observation="pass" if result else "fail",
                cost=config.c_critic,
            ))
            costs.critic_cost += config.c_critic
            budget.remaining_critics -= 1

        elif choice.action_type == "generate":
            current_arm = choice.label
            current_seed = 0
            belief = controller.generator_transition(belief, current_arm)
            actions.append(ActionRecord(
                kind=ActionKind.GENERATE,
                label=current_arm,
                cost=config.c_generator,
            ))
            costs.generator_cost += config.c_generator
            budget.remaining_generators -= 1

        budget.remaining_steps -= 1
        belief_timeline.append({"b": belief})

    # Final verify if we didn't already
    verified = any(a.kind == ActionKind.VERIFY for a in actions)
    if not verified:
        candidate = replay.get_candidate(
            config.project, config.bug_id, current_arm, current_seed
        )
        success = candidate.verifier_outputs.full_suite_pass if candidate else False
        actions.append(ActionRecord(
            kind=ActionKind.VERIFY, label="full_suite",
            observation="pass" if success else "fail",
            cost=config.c_verifier,
        ))
        costs.verifier_cost += config.c_verifier
        verified = True

    success = actions[-1].observation == "pass"
    costs.compute_total()
    reward = config.reward if success else 0.0

    return EpisodeResult(
        policy_name="bayes",
        hidden_class=f"{config.project}_{config.bug_id}",
        actions=actions,
        costs=costs,
        patch_correct=success,
        verified=verified,
        reward=reward,
        utility=reward - costs.total_cost,
        belief_timeline=belief_timeline,
    )
