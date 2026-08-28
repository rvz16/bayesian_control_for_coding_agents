"""Unit tests for the Bayesian POMDP model components.

Tests belief updates, generator transitions, Q-value functions,
and action selection — all without LLM calls (fast, deterministic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.bayes_agent import (
    BELIEF_GRID_SIZE,
    CRITIC_LIKELIHOODS,
    GENERATOR_TRANSITIONS,
    DPPlanner,
    DPState,
    bayes_update,
    discretize,
    generator_transition,
    grid_belief,
    q_verify,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig


COSTS = AgentCostConfig()


# ---------------------------------------------------------------------------
# Bayes update tests
# ---------------------------------------------------------------------------


class TestBayesUpdate:
    """Tests for posterior belief updates via Bayes rule."""

    def test_critic_pass_increases_belief(self):
        """If a critic passes, belief should increase."""
        for critic in CRITIC_LIKELIHOODS:
            b0 = 0.3
            b1 = bayes_update(b0, critic, passed=True)
            assert b1 > b0, f"{critic}: pass should increase belief"

    def test_critic_fail_decreases_belief(self):
        """If a critic fails, belief should decrease."""
        for critic in CRITIC_LIKELIHOODS:
            b0 = 0.5
            b1 = bayes_update(b0, critic, passed=False)
            assert b1 < b0, f"{critic}: fail should decrease belief"

    def test_belief_stays_in_01(self):
        """Belief must remain in [0, 1] after any update."""
        for critic in CRITIC_LIKELIHOODS:
            for b0 in [0.01, 0.1, 0.5, 0.9, 0.99]:
                for passed in [True, False]:
                    b1 = bayes_update(b0, critic, passed)
                    assert 0 <= b1 <= 1, f"belief {b1} out of range"

    def test_high_belief_stays_high_on_pass(self):
        """A confident belief should remain high after passing critic."""
        b1 = bayes_update(0.9, "filtering_check", passed=True)
        assert b1 > 0.9

    def test_low_belief_drops_further_on_fail(self):
        """A low belief should drop further after failing critic."""
        b1 = bayes_update(0.1, "filtering_check", passed=False)
        assert b1 < 0.1

    def test_filtering_check_most_informative(self):
        """filtering_check has lowest p_pass_y0, so fail gives strongest signal."""
        b0 = 0.5
        drops = {}
        for critic in CRITIC_LIKELIHOODS:
            b1 = bayes_update(b0, critic, passed=False)
            drops[critic] = b0 - b1
        # filtering_check should give the biggest drop on fail
        assert drops["filtering_check"] >= max(
            drops[c] for c in drops if c != "filtering_check"
        )

    def test_repeated_passes_increase_belief(self):
        """Multiple critic passes should monotonically increase belief."""
        b = 0.3
        for critic in CRITIC_LIKELIHOODS:
            b_new = bayes_update(b, critic, passed=True)
            assert b_new > b
            b = b_new
        assert b > 0.7  # after 4 passes, should be confident

    def test_repeated_fails_decrease_belief(self):
        """Multiple critic fails should monotonically decrease belief."""
        b = 0.5
        for critic in CRITIC_LIKELIHOODS:
            b_new = bayes_update(b, critic, passed=False)
            assert b_new < b
            b = b_new
        assert b < 0.1  # after 4 fails, should be very low


# ---------------------------------------------------------------------------
# Generator transition tests
# ---------------------------------------------------------------------------

class TestGeneratorTransition:
    """Tests for belief updates after LLM generation."""

    def test_transition_from_low_belief(self):
        """From low belief, generation should increase it (p01 > p10)."""
        for arm in GENERATOR_TRANSITIONS:
            b0 = 0.1
            b1 = generator_transition(b0, arm)
            assert b1 > b0, f"{arm}: should increase from low belief"

    def test_transition_from_high_belief(self):
        """From very high belief, generation might slightly decrease it (p10 > 0)."""
        for arm in GENERATOR_TRANSITIONS:
            b0 = 0.99
            b1 = generator_transition(b0, arm)
            # b' = 0.99*(1-p10) + 0.01*p01
            # With p10=0.05: b' = 0.99*0.95 + 0.01*0.5 = 0.9405 + 0.005 = 0.9455
            assert b1 < b0, f"{arm}: high belief should slightly decrease (risk of regression)"

    def test_transition_stays_in_01(self):
        """Belief after generation must stay in [0, 1]."""
        for arm in GENERATOR_TRANSITIONS:
            for b0 in [0.0, 0.01, 0.5, 0.99, 1.0]:
                b1 = generator_transition(b0, arm)
                assert 0 <= b1 <= 1

    def test_g1_direct_fix_highest_p01(self):
        """g1_direct_fix should have the highest p01 (works best with small LLMs)."""
        p01s = {arm: GENERATOR_TRANSITIONS[arm]["p01"] for arm in GENERATOR_TRANSITIONS}
        assert p01s["g1_direct_fix"] >= max(
            p01s[a] for a in p01s if a != "g1_direct_fix"
        )

    def test_all_arms_have_low_p10(self):
        """All arms should have low p10 (low regression risk)."""
        for arm, t in GENERATOR_TRANSITIONS.items():
            assert t["p10"] <= 0.10, f"{arm}: p10 should be <= 0.10"


# ---------------------------------------------------------------------------
# Q-value function tests
# ---------------------------------------------------------------------------

class TestQValues:
    """Tests for expected utility calculations."""

    def test_q_verify_positive_high_belief(self):
        """Verify should have positive Q at high belief."""
        q = q_verify(0.9, COSTS)
        assert q > 0  # 100*0.9 - 5 = 85

    def test_q_verify_negative_low_belief(self):
        """Verify should have negative Q at very low belief."""
        q = q_verify(0.01, COSTS)
        assert q < 0  # 100*0.01 - 5 = -4

    def test_q_verify_breakeven(self):
        """Q_verify = 0 at belief = c_full_test / reward."""
        breakeven = COSTS.c_full_test / COSTS.reward  # 5/100 = 0.05
        q = q_verify(breakeven, COSTS)
        assert abs(q) < 0.01

    def test_q_verify_monotonic_in_belief(self):
        """Higher belief → higher Q_verify."""
        beliefs = [0.1, 0.3, 0.5, 0.7, 0.9]
        qs = [q_verify(b, COSTS) for b in beliefs]
        for i in range(len(qs) - 1):
            assert qs[i + 1] > qs[i]

    def test_q_verify_scales_with_reward(self):
        """Doubling reward should roughly double Q at fixed belief."""
        costs1 = AgentCostConfig(reward=100.0)
        costs2 = AgentCostConfig(reward=200.0)
        q1 = q_verify(0.5, costs1)
        q2 = q_verify(0.5, costs2)
        assert q2 > q1

    def test_verify_threshold_belief(self):
        """Agent should only verify when belief > cost/reward."""
        threshold = COSTS.c_full_test / COSTS.reward
        assert q_verify(threshold - 0.01, COSTS) < 0
        assert q_verify(threshold + 0.01, COSTS) > 0


# ---------------------------------------------------------------------------
# DP Planner tests
# ---------------------------------------------------------------------------

class TestDPDiscretization:
    """Tests for belief grid discretization."""

    def test_grid_roundtrip(self):
        """discretize -> grid_belief should be close to original."""
        for b in [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]:
            idx = discretize(b)
            recovered = grid_belief(idx)
            assert abs(recovered - b) < 0.01

    def test_discretize_clamps(self):
        """Values outside [0, 1] should clamp to grid bounds."""
        assert discretize(-0.1) == 0
        assert discretize(1.5) == BELIEF_GRID_SIZE - 1

    def test_grid_size(self):
        assert BELIEF_GRID_SIZE == 101


class TestDPPlanner:
    """Tests for the full Bellman DP planner."""

    @pytest.fixture
    def planner(self):
        p = DPPlanner(COSTS, max_generators=3, max_verifications=2)
        p.solve()
        return p

    def test_dp_solves_without_error(self, planner):
        """DPPlanner.solve() should complete without errors."""
        assert planner.cache_size > 0

    def test_dp_state_space_bounded(self, planner):
        """Cache should be reasonable size (not combinatorial explosion)."""
        assert planner.cache_size < 500_000  # 8 critics -> larger state space

    def test_dp_high_belief_recommends_verify(self, planner):
        """At high belief with verifications left, should verify."""
        action, value = planner.choose_action(
            belief=0.95, gen_left=3, crit_used=frozenset(), ver_left=2
        )
        assert action == "verify", f"Expected verify at b=0.95, got {action}"
        assert value > 0

    def test_dp_low_belief_recommends_generate_or_bail(self, planner):
        """At very low belief, should generate (if available) or bail."""
        action, value = planner.choose_action(
            belief=0.01, gen_left=3, crit_used=frozenset(), ver_left=2
        )
        assert action.startswith("generate:") or action == "bail_out"

    def test_dp_zero_resources_bails(self, planner):
        """With no generators or verifications left, should bail at low belief."""
        action, value = planner.choose_action(
            belief=0.02, gen_left=0,
            crit_used=frozenset(CRITIC_LIKELIHOODS.keys()), ver_left=0
        )
        assert action == "bail_out"

    def test_dp_mid_belief_recommends_critic(self, planner):
        """At moderate belief with unused critics, DP may choose critic."""
        action, value = planner.choose_action(
            belief=0.50, gen_left=2, crit_used=frozenset(), ver_left=2
        )
        assert action != "bail_out"

    def test_dp_value_monotonic_in_belief(self, planner):
        """Higher belief should give higher or equal value."""
        values = []
        for b in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            _, v = planner.choose_action(
                belief=b, gen_left=3, crit_used=frozenset(), ver_left=2
            )
            values.append(v)
        for i in range(len(values) - 1):
            assert values[i + 1] >= values[i] - 0.1, (
                f"Value should be monotonic: V({0.2*(i+1):.1f})={values[i+1]:.1f} "
                f"< V({0.2*i:.1f})={values[i]:.1f}"
            )

    def test_dp_value_positive_at_high_belief(self, planner):
        """V(b=0.9) should be clearly positive."""
        _, v = planner.choose_action(
            belief=0.9, gen_left=3, crit_used=frozenset(), ver_left=2
        )
        assert v > 50

    def test_dp_prefers_cheap_critic_over_full_test(self, planner):
        """At moderate belief, DP should not verify when critics available."""
        action, _ = planner.choose_action(
            belief=0.40, gen_left=2, crit_used=frozenset(), ver_left=2
        )
        assert action != "verify", (
            f"At b=0.40 with unused critics, should not verify (cost=5)"
        )

    def test_dp_matches_greedy_at_extremes(self, planner):
        """At b~1.0 both DP and greedy should verify; at b~0 both should bail."""
        action_high, _ = planner.choose_action(
            belief=0.99, gen_left=0, crit_used=frozenset(CRITIC_LIKELIHOODS.keys()),
            ver_left=1,
        )
        assert action_high == "verify"

        action_low, _ = planner.choose_action(
            belief=0.01, gen_left=0, crit_used=frozenset(CRITIC_LIKELIHOODS.keys()),
            ver_left=0,
        )
        assert action_low == "bail_out"

    def test_dp_different_cost_models(self):
        """DP should work with different cost configurations."""
        expensive = AgentCostConfig(c_llm_call=50.0, c_full_test=20.0,
                                     c_critic_test=5.0, reward=200.0)
        p = DPPlanner(expensive, max_generators=2, max_verifications=1)
        p.solve()
        assert p.cache_size > 0
        action, _ = p.choose_action(
            belief=0.95, gen_left=2, crit_used=frozenset(), ver_left=1
        )
        assert action in ("verify", "bail_out") or action.startswith(("generate:", "critic:"))

