"""Bayesian decision-theoretic controller for real-world bug fixing.

Maintains a posterior belief b_t = P(Y_t=1 | history) over whether the
current candidate patch is correct (Y=1) and uses value-of-information
calculations to decide: run a critic, generate a new patch, or verify.

This is sequential decision-making / hypothesis testing over bug-fixing
actions — not Gaussian-process Bayesian optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from abbo.realworld.calibration.fit import FittedBayesModel
from abbo.realworld.calibration.likelihoods import CriticLikelihood
from abbo.realworld.calibration.transitions import TransitionModel


@dataclass
class BudgetState:
    """Tracks remaining budget for the controller."""

    remaining_steps: int = 10
    remaining_generators: int = 4
    remaining_critics: int = 6
    verified: bool = False


@dataclass
class ActionChoice:
    """The controller's chosen action."""

    action_type: str  # "verify", "critic", "generate"
    label: str = ""  # critic name or generator arm
    q_value: float = 0.0


class BayesController:
    """Bayesian value-of-information controller.

    Uses a discretized belief grid and backward induction to compute
    optimal actions at each belief/budget state.
    """

    def __init__(
        self,
        model: FittedBayesModel,
        reward_success: float = 200.0,
        c_verifier: float = 20.0,
        c_critic: float = 1.0,
        c_generator: float = 3.0,
        grid_resolution: float = 0.01,
    ):
        self.model = model
        self.reward_success = reward_success
        self.c_verifier = c_verifier
        self.c_critic = c_critic
        self.c_generator = c_generator
        self.grid = np.arange(0.0, 1.0 + grid_resolution / 2, grid_resolution)
        self._value_cache: dict[tuple[float, int, int, int], float] = {}

    def _snap_to_grid(self, b: float) -> float:
        """Snap a belief value to the nearest grid point."""
        idx = int(round(b / (self.grid[1] - self.grid[0])))
        idx = max(0, min(idx, len(self.grid) - 1))
        return float(self.grid[idx])

    def q_verify(self, belief: float) -> float:
        """Expected value of verifying now: reward * b - C_ver."""
        return self.reward_success * belief - self.c_verifier

    def bayes_update_belief(
        self, belief: float, critic_name: str, observed_pass: bool
    ) -> float:
        """Update belief after observing a critic result."""
        lk = self.model.critic_likelihoods.get(critic_name)
        if lk is None:
            return belief

        if observed_pass:
            p_z_y1 = lk.p_pass_given_correct
            p_z_y0 = lk.p_pass_given_incorrect
        else:
            p_z_y1 = lk.p_fail_given_correct
            p_z_y0 = lk.p_fail_given_incorrect

        # Bayes rule
        numerator = p_z_y1 * belief
        denominator = p_z_y1 * belief + p_z_y0 * (1 - belief)
        if denominator == 0:
            return belief
        return numerator / denominator

    def generator_transition(self, belief: float, arm: str) -> float:
        """Transition belief after generating a new candidate: T_g(b)."""
        tm = self.model.transition_models.get(arm)
        if tm is None:
            return belief
        return belief * (1 - tm.p10) + (1 - belief) * tm.p01

    def value(self, belief: float, budget: BudgetState) -> float:
        """Compute V(belief, budget) via backward induction."""
        b = self._snap_to_grid(belief)
        key = (b, budget.remaining_steps, budget.remaining_generators, budget.remaining_critics)

        if key in self._value_cache:
            return self._value_cache[key]

        if budget.remaining_steps <= 0:
            val = self.q_verify(b)
            self._value_cache[key] = val
            return val

        best = self.q_verify(b)

        # Critic options
        if budget.remaining_critics > 0:
            for critic_name, lk in self.model.critic_likelihoods.items():
                new_budget = BudgetState(
                    remaining_steps=budget.remaining_steps - 1,
                    remaining_generators=budget.remaining_generators,
                    remaining_critics=budget.remaining_critics - 1,
                    verified=budget.verified,
                )
                # P(pass | b)
                p_pass = lk.p_pass_given_correct * b + lk.p_pass_given_incorrect * (1 - b)
                p_fail = 1 - p_pass

                b_pass = self.bayes_update_belief(b, critic_name, True)
                b_fail = self.bayes_update_belief(b, critic_name, False)

                q = -self.c_critic + p_pass * self.value(b_pass, new_budget) + p_fail * self.value(b_fail, new_budget)
                best = max(best, q)

        # Generator options
        if budget.remaining_generators > 0:
            for arm, tm in self.model.transition_models.items():
                new_budget = BudgetState(
                    remaining_steps=budget.remaining_steps - 1,
                    remaining_generators=budget.remaining_generators - 1,
                    remaining_critics=budget.remaining_critics,
                    verified=budget.verified,
                )
                b_new = self.generator_transition(b, arm)
                q = -self.c_generator + self.value(b_new, new_budget)
                best = max(best, q)

        self._value_cache[key] = best
        return best

    def choose_action(
        self, belief: float, budget: BudgetState
    ) -> ActionChoice:
        """Choose the optimal action given current belief and budget."""
        b = self._snap_to_grid(belief)
        best_action = ActionChoice(
            action_type="verify", label="full_suite", q_value=self.q_verify(b)
        )

        if budget.remaining_steps <= 0:
            return best_action

        # Evaluate critic options
        if budget.remaining_critics > 0:
            for critic_name, lk in self.model.critic_likelihoods.items():
                new_budget = BudgetState(
                    remaining_steps=budget.remaining_steps - 1,
                    remaining_generators=budget.remaining_generators,
                    remaining_critics=budget.remaining_critics - 1,
                    verified=budget.verified,
                )
                p_pass = lk.p_pass_given_correct * b + lk.p_pass_given_incorrect * (1 - b)
                p_fail = 1 - p_pass
                b_pass = self.bayes_update_belief(b, critic_name, True)
                b_fail = self.bayes_update_belief(b, critic_name, False)
                q = -self.c_critic + p_pass * self.value(b_pass, new_budget) + p_fail * self.value(b_fail, new_budget)
                if q > best_action.q_value:
                    best_action = ActionChoice(
                        action_type="critic", label=critic_name, q_value=q
                    )

        # Evaluate generator options
        if budget.remaining_generators > 0:
            for arm in self.model.transition_models:
                new_budget = BudgetState(
                    remaining_steps=budget.remaining_steps - 1,
                    remaining_generators=budget.remaining_generators - 1,
                    remaining_critics=budget.remaining_critics,
                    verified=budget.verified,
                )
                b_new = self.generator_transition(b, arm)
                q = -self.c_generator + self.value(b_new, new_budget)
                if q > best_action.q_value:
                    best_action = ActionChoice(
                        action_type="generate", label=arm, q_value=q
                    )

        return best_action

    def solve(self, max_steps: int = 10) -> None:
        """Pre-solve the value function for all grid points.

        This populates the cache so that ``choose_action`` is fast.
        """
        self._value_cache.clear()
        for b in self.grid:
            budget = BudgetState(remaining_steps=max_steps)
            self.value(float(b), budget)
