"""Policy implementations for the synthetic benchmark.

Each policy implements ``run_episode(env, rng) -> EpisodeResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from fractions import Fraction

import numpy as np

from abbo.core.types import ActionKind, ActionRecord, CostBreakdown, EpisodeResult
from abbo.synthetic.env import SyntheticEnv
from abbo.synthetic.exact_math import (
    Belief,
    posterior_update,
    uniform_prior,
    _dp_value,
)


class SyntheticPolicy(ABC):
    """Base class for synthetic benchmark policies."""

    name: str = "base"

    @abstractmethod
    def run_episode(
        self, env: SyntheticEnv, rng: np.random.Generator
    ) -> EpisodeResult:
        """Run a single episode and return the result."""
        ...


def _map_patch(belief: Belief, env: SyntheticEnv) -> str:
    """Choose the patch corresponding to the MAP class."""
    map_cls = max(belief, key=lambda c: belief[c])
    return env.patch_for_class(map_cls)


def _belief_to_float(belief: Belief) -> dict[str, float]:
    return {c: float(v) for c, v in belief.items()}


# ---------------------------------------------------------------------------
# Bayes-optimal policy (uses DP)
# ---------------------------------------------------------------------------

class BayesOptimalPolicy(SyntheticPolicy):
    """Bayesian decision-theoretic optimal policy using the DP solution."""

    name = "bayes_optimal"

    def run_episode(
        self, env: SyntheticEnv, rng: np.random.Generator
    ) -> EpisodeResult:
        """Run an episode following the DP-optimal strategy."""
        hidden = env.sample_hidden(rng)
        belief = uniform_prior(env)
        actions: list[ActionRecord] = []
        belief_timeline = [_belief_to_float(belief)]
        n_tests = 0
        total_test_cost = 0.0

        remaining = env.max_tests
        while remaining > 0:
            _, best_test = _dp_value(belief, remaining, env)
            if best_test is None:
                break
            obs_fail = env.run_diagnostic(hidden, best_test, rng)
            belief = posterior_update(belief, best_test, obs_fail, env)
            actions.append(
                ActionRecord(
                    kind=ActionKind.DIAGNOSTIC,
                    label=best_test,
                    observation="fail" if obs_fail else "pass",
                    cost=env.c_test,
                )
            )
            belief_timeline.append(_belief_to_float(belief))
            n_tests += 1
            total_test_cost += env.c_test
            remaining -= 1

        # MAP patch + verify
        patch = _map_patch(belief, env)
        correct = env.apply_patch(hidden, patch)
        actions.append(
            ActionRecord(kind=ActionKind.PATCH, label=patch, cost=env.c_patch)
        )
        actions.append(
            ActionRecord(
                kind=ActionKind.VERIFY,
                label="verifier",
                observation="pass" if correct else "fail",
                cost=env.c_verifier,
            )
        )

        reward = env.reward if correct else 0.0
        costs = CostBreakdown(
            diagnostic_cost=total_test_cost,
            patch_cost=env.c_patch,
            verifier_cost=env.c_verifier,
        )
        costs.compute_total()
        utility = reward - costs.total_cost

        return EpisodeResult(
            policy_name=self.name,
            hidden_class=hidden,
            actions=actions,
            costs=costs,
            patch_chosen=patch,
            patch_correct=correct,
            verified=True,
            reward=reward,
            utility=utility,
            belief_timeline=belief_timeline,
        )


# ---------------------------------------------------------------------------
# One-test MAP baseline
# ---------------------------------------------------------------------------

class OneTestMAPPolicy(SyntheticPolicy):
    """Run one fixed diagnostic test, Bayes update, choose MAP patch, verify."""

    name = "one_test_map"

    def __init__(self, test_id: str = "T1"):
        self.test_id = test_id

    def run_episode(
        self, env: SyntheticEnv, rng: np.random.Generator
    ) -> EpisodeResult:
        hidden = env.sample_hidden(rng)
        belief = uniform_prior(env)
        belief_timeline = [_belief_to_float(belief)]

        # Run one test
        obs_fail = env.run_diagnostic(hidden, self.test_id, rng)
        belief = posterior_update(belief, self.test_id, obs_fail, env)
        belief_timeline.append(_belief_to_float(belief))

        actions = [
            ActionRecord(
                kind=ActionKind.DIAGNOSTIC,
                label=self.test_id,
                observation="fail" if obs_fail else "pass",
                cost=env.c_test,
            )
        ]

        # MAP patch + verify
        patch = _map_patch(belief, env)
        correct = env.apply_patch(hidden, patch)
        actions.append(
            ActionRecord(kind=ActionKind.PATCH, label=patch, cost=env.c_patch)
        )
        actions.append(
            ActionRecord(
                kind=ActionKind.VERIFY,
                label="verifier",
                observation="pass" if correct else "fail",
                cost=env.c_verifier,
            )
        )

        reward = env.reward if correct else 0.0
        costs = CostBreakdown(
            diagnostic_cost=env.c_test,
            patch_cost=env.c_patch,
            verifier_cost=env.c_verifier,
        )
        costs.compute_total()
        utility = reward - costs.total_cost

        return EpisodeResult(
            policy_name=self.name,
            hidden_class=hidden,
            actions=actions,
            costs=costs,
            patch_chosen=patch,
            patch_correct=correct,
            verified=True,
            reward=reward,
            utility=utility,
            belief_timeline=belief_timeline,
        )


# ---------------------------------------------------------------------------
# One-test if-fail-else-rule baseline
# ---------------------------------------------------------------------------

class OneTestIfFailElseRulePolicy(SyntheticPolicy):
    """Fixed hand-written baseline from the paper style.

    Run T1.  If fail, patch A.  If pass, run T2.
    If T2 fail, patch B.  If T2 pass, patch C.
    Then verify.
    """

    name = "one_test_if_fail_else_rule"

    def run_episode(
        self, env: SyntheticEnv, rng: np.random.Generator
    ) -> EpisodeResult:
        hidden = env.sample_hidden(rng)
        belief = uniform_prior(env)
        actions: list[ActionRecord] = []
        belief_timeline = [_belief_to_float(belief)]
        test_cost = 0.0

        # Run T1
        t1_fail = env.run_diagnostic(hidden, "T1", rng)
        belief = posterior_update(belief, "T1", t1_fail, env)
        belief_timeline.append(_belief_to_float(belief))
        actions.append(
            ActionRecord(
                kind=ActionKind.DIAGNOSTIC,
                label="T1",
                observation="fail" if t1_fail else "pass",
                cost=env.c_test,
            )
        )
        test_cost += env.c_test

        if t1_fail:
            patch = "PA"
        else:
            # Run T2
            t2_fail = env.run_diagnostic(hidden, "T2", rng)
            belief = posterior_update(belief, "T2", t2_fail, env)
            belief_timeline.append(_belief_to_float(belief))
            actions.append(
                ActionRecord(
                    kind=ActionKind.DIAGNOSTIC,
                    label="T2",
                    observation="fail" if t2_fail else "pass",
                    cost=env.c_test,
                )
            )
            test_cost += env.c_test
            patch = "PB" if t2_fail else "PC"

        correct = env.apply_patch(hidden, patch)
        actions.append(
            ActionRecord(kind=ActionKind.PATCH, label=patch, cost=env.c_patch)
        )
        actions.append(
            ActionRecord(
                kind=ActionKind.VERIFY,
                label="verifier",
                observation="pass" if correct else "fail",
                cost=env.c_verifier,
            )
        )

        reward = env.reward if correct else 0.0
        costs = CostBreakdown(
            diagnostic_cost=test_cost,
            patch_cost=env.c_patch,
            verifier_cost=env.c_verifier,
        )
        costs.compute_total()
        utility = reward - costs.total_cost

        return EpisodeResult(
            policy_name=self.name,
            hidden_class=hidden,
            actions=actions,
            costs=costs,
            patch_chosen=patch,
            patch_correct=correct,
            verified=True,
            reward=reward,
            utility=utility,
            belief_timeline=belief_timeline,
        )


# ---------------------------------------------------------------------------
# Tuned threshold heuristic
# ---------------------------------------------------------------------------

class TunedThresholdHeuristic(SyntheticPolicy):
    """Threshold-based heuristic: test until MAP belief exceeds threshold.

    Runs tests in order T1, T2, T3 until the MAP belief exceeds
    *threshold* or the test budget is exhausted.
    """

    name = "tuned_threshold"

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def run_episode(
        self, env: SyntheticEnv, rng: np.random.Generator
    ) -> EpisodeResult:
        hidden = env.sample_hidden(rng)
        belief = uniform_prior(env)
        actions: list[ActionRecord] = []
        belief_timeline = [_belief_to_float(belief)]
        test_cost = 0.0
        test_order = list(env.tests)

        for i in range(env.max_tests):
            max_b = float(max(belief.values()))
            if max_b >= self.threshold:
                break
            test_id = test_order[i % len(test_order)]
            obs_fail = env.run_diagnostic(hidden, test_id, rng)
            belief = posterior_update(belief, test_id, obs_fail, env)
            belief_timeline.append(_belief_to_float(belief))
            actions.append(
                ActionRecord(
                    kind=ActionKind.DIAGNOSTIC,
                    label=test_id,
                    observation="fail" if obs_fail else "pass",
                    cost=env.c_test,
                )
            )
            test_cost += env.c_test

        patch = _map_patch(belief, env)
        correct = env.apply_patch(hidden, patch)
        actions.append(
            ActionRecord(kind=ActionKind.PATCH, label=patch, cost=env.c_patch)
        )
        actions.append(
            ActionRecord(
                kind=ActionKind.VERIFY,
                label="verifier",
                observation="pass" if correct else "fail",
                cost=env.c_verifier,
            )
        )

        reward = env.reward if correct else 0.0
        costs = CostBreakdown(
            diagnostic_cost=test_cost,
            patch_cost=env.c_patch,
            verifier_cost=env.c_verifier,
        )
        costs.compute_total()
        utility = reward - costs.total_cost

        return EpisodeResult(
            policy_name=self.name,
            hidden_class=hidden,
            actions=actions,
            costs=costs,
            patch_chosen=patch,
            patch_correct=correct,
            verified=True,
            reward=reward,
            utility=utility,
            belief_timeline=belief_timeline,
        )
