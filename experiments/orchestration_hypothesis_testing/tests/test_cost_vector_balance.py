"""Tests for analysis/cost_vector_balance.py — balance metric + mode
aggregation + recommendation."""
from __future__ import annotations

import pathlib
import sys

import pytest

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from analysis.cost_vector_balance import (  # noqa: E402
    FAST_MODE,
    SLOW_MODE,
    aggregate_mode_balance,
    balance_score,
    mode_for_benchmark,
    recommend_c_ver_for_mode,
)


# ---------------------------------------------------------------------------
# balance_score — the core metric
# ---------------------------------------------------------------------------

def test_balance_score_balanced_histogram():
    """Mix of 3 winners + 3 losers (with non-trivial magnitudes) — balance=3."""
    deltas = {
        "always_verify": 0.0,         # excluded from count
        "bayesian_DP": +10.0,
        "bayesian_greedy": +5.0,
        "gate_test": +3.0,
        "best_of_3": -8.0,
        "fixed_pipeline": -15.0,
        "self_refine": -20.0,
        "reflexion": +0.5,            # neutral (< epsilon)
    }
    bs = balance_score(deltas)
    assert bs["n_above"] == 3
    assert bs["n_below"] == 3
    assert bs["n_neutral"] == 1
    assert bs["n_total"] == 7   # excludes AV
    assert bs["balance"] == 3


def test_balance_score_all_below_av():
    """Degenerate case: everything loses to AV (the lcb_hard/gpt5_mini
    pathology). balance = 0 — uninformative comparison."""
    deltas = {
        "always_verify": 0.0,
        "bayesian_DP": -3.0,
        "bayesian_greedy": -8.0,
        "best_of_3": -12.0,
        "self_refine": -20.0,
    }
    bs = balance_score(deltas)
    assert bs["n_above"] == 0
    assert bs["n_below"] == 4
    assert bs["balance"] == 0


def test_balance_score_all_above_av():
    """Inverse degenerate: every policy beats AV → still uninformative."""
    deltas = {
        "always_verify": 0.0,
        "bayesian_DP": +10.0,
        "bayesian_greedy": +8.0,
        "best_of_3": +5.0,
    }
    bs = balance_score(deltas)
    assert bs["n_above"] == 3
    assert bs["n_below"] == 0
    assert bs["balance"] == 0


def test_balance_score_respects_epsilon():
    """Tiny deltas (< epsilon) count as neutral, not winners/losers."""
    deltas = {
        "always_verify": 0.0,
        "p1": +1.5,    # below default epsilon=2.0
        "p2": -1.5,    # also neutral
        "p3": +5.0,    # winner
        "p4": -5.0,    # loser
    }
    bs = balance_score(deltas)
    assert bs["n_above"] == 1   # only p3
    assert bs["n_below"] == 1   # only p4
    assert bs["n_neutral"] == 2 # p1, p2
    assert bs["balance"] == 1


def test_balance_score_custom_epsilon():
    """epsilon=0 counts every non-zero policy."""
    deltas = {
        "always_verify": 0.0,
        "p1": +0.5,
        "p2": -0.5,
    }
    bs = balance_score(deltas, epsilon=0.0)
    assert bs["n_above"] == 1
    assert bs["n_below"] == 1
    assert bs["balance"] == 1


def test_balance_score_excludes_av_by_default():
    """always_verify itself should not count in the totals (it sits at 0 by
    definition)."""
    deltas = {
        "always_verify": 0.0,
        "p1": +5.0,
        "p2": -5.0,
    }
    bs = balance_score(deltas)
    assert bs["n_total"] == 2   # excludes AV


def test_balance_score_can_include_av():
    """Caller can override exclusion if they want."""
    deltas = {
        "always_verify": 0.0,
        "p1": +5.0,
        "p2": -5.0,
    }
    bs = balance_score(deltas, exclude_av=False)
    assert bs["n_total"] == 3   # includes AV
    # AV is at exactly 0 so it lands in neutral (|0| <= epsilon)
    assert bs["n_neutral"] >= 1


def test_balance_score_empty_input():
    """No policies → balance=0, no crash."""
    bs = balance_score({})
    assert bs["balance"] == 0
    assert bs["n_total"] == 0


def test_balance_score_spread_tiebreaker():
    """For two cells with the same balance, the one with higher spread is
    preferred — confirms spread is reported correctly."""
    bs_tight = balance_score({
        "p1": +3.0, "p2": +3.0, "p3": -3.0, "p4": -3.0,
    })
    bs_wide = balance_score({
        "p1": +30.0, "p2": +30.0, "p3": -30.0, "p4": -30.0,
    })
    assert bs_tight["balance"] == bs_wide["balance"]
    assert bs_wide["spread"] > bs_tight["spread"]


# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

def test_mode_for_benchmark_known():
    assert mode_for_benchmark("lcb_hard") is FAST_MODE
    assert mode_for_benchmark("mbpp") is FAST_MODE
    assert mode_for_benchmark("swe_lite") is SLOW_MODE
    assert mode_for_benchmark("swe_verified") is SLOW_MODE


def test_mode_for_benchmark_unknown():
    assert mode_for_benchmark("zzz_unknown") is None


def test_mode_ranges_are_sane():
    # Both modes use c_gen=10 (unified), c_critic asymmetry is what
    # distinguishes them (FAST=1/1/1, SLOW=1/2/5), AND both modes cap
    # their c_ver sweep at 0.9 · reward = 90 to keep the policy comparison
    # methodologically coherent (c_ver < reward = 100 ensures AV's utility
    # can be non-negative on correct patches).
    assert FAST_MODE.c_gen == 10.0
    assert SLOW_MODE.c_gen == 10.0
    # SLOW critics are heavier (Docker)
    assert SLOW_MODE.c_critic_l2 > FAST_MODE.c_critic_l2
    assert SLOW_MODE.c_critic_l3 > FAST_MODE.c_critic_l3
    # Both modes cap at the same upper bound (reward-anchored)
    assert FAST_MODE.c_ver_range[1] == SLOW_MODE.c_ver_range[1] == 90.0
    # Each mode's current §2.5 c_ver sits inside its sweep range
    assert FAST_MODE.c_ver_range[0] <= FAST_MODE.c_ver_current <= FAST_MODE.c_ver_range[1]
    assert SLOW_MODE.c_ver_range[0] <= SLOW_MODE.c_ver_current <= SLOW_MODE.c_ver_range[1]
    # Lower bound: positive and below current
    assert FAST_MODE.c_ver_range[0] > 0
    assert SLOW_MODE.c_ver_range[0] > 0


# ---------------------------------------------------------------------------
# aggregate_mode_balance
# ---------------------------------------------------------------------------

def test_aggregate_mode_balance_filters_by_mode():
    """Cells outside the mode's benchmarks are excluded from aggregation."""
    sweeps = {
        ("lcb_easy", "haiku45"): [
            {"c_ver": 5.0, "balance": 3, "spread": 10.0, "n_above": 3,
             "n_below": 3, "n_neutral": 1, "n_total": 7, "policy_deltas": {}},
        ],
        ("swe_lite", "haiku45"): [
            # This cell should be EXCLUDED from FAST_MODE aggregation
            {"c_ver": 5.0, "balance": 5, "spread": 50.0, "n_above": 5,
             "n_below": 5, "n_neutral": 0, "n_total": 10, "policy_deltas": {}},
        ],
    }
    agg = aggregate_mode_balance(sweeps, FAST_MODE)
    assert len(agg) == 1
    assert agg[0]["n_cells"] == 1   # only lcb_easy, not swe_lite
    assert agg[0]["mean_balance"] == 3.0


def test_aggregate_mode_balance_averages_across_cells():
    """Multiple cells in the same mode contribute to mean_balance."""
    sweeps = {
        ("lcb_easy", "haiku45"): [
            {"c_ver": 5.0, "balance": 2, "spread": 5.0, "n_above": 2,
             "n_below": 2, "n_neutral": 0, "n_total": 4, "policy_deltas": {}},
        ],
        ("lcb_hard", "sonnet45"): [
            {"c_ver": 5.0, "balance": 4, "spread": 10.0, "n_above": 4,
             "n_below": 4, "n_neutral": 0, "n_total": 8, "policy_deltas": {}},
        ],
    }
    agg = aggregate_mode_balance(sweeps, FAST_MODE)
    assert agg[0]["mean_balance"] == 3.0  # (2+4)/2
    assert agg[0]["n_cells"] == 2


def test_aggregate_mode_balance_empty():
    """No cells in mode → empty result, no crash."""
    sweeps = {}
    agg = aggregate_mode_balance(sweeps, FAST_MODE)
    assert agg == []


# ---------------------------------------------------------------------------
# recommend_c_ver_for_mode
# ---------------------------------------------------------------------------

def test_recommend_c_ver_picks_max_balance():
    """The recommended c_ver should have the highest mean_balance."""
    aggregated = [
        {"c_ver": 1.0, "mean_balance": 0.5, "mean_spread": 5.0,
         "n_cells": 5, "per_cell_balances": {}, "median_balance": 0.0},
        {"c_ver": 10.0, "mean_balance": 3.5, "mean_spread": 8.0,
         "n_cells": 5, "per_cell_balances": {}, "median_balance": 4.0},
        {"c_ver": 30.0, "mean_balance": 2.0, "mean_spread": 6.0,
         "n_cells": 5, "per_cell_balances": {}, "median_balance": 2.0},
    ]
    rec = recommend_c_ver_for_mode(aggregated)
    assert rec is not None
    assert rec["c_ver"] == 10.0


def test_recommend_c_ver_ties_broken_by_spread():
    """When mean_balance is tied, prefer higher spread."""
    aggregated = [
        {"c_ver": 5.0, "mean_balance": 3.0, "mean_spread": 8.0,
         "n_cells": 5, "per_cell_balances": {}, "median_balance": 3.0},
        {"c_ver": 15.0, "mean_balance": 3.0, "mean_spread": 12.0,
         "n_cells": 5, "per_cell_balances": {}, "median_balance": 3.0},
    ]
    rec = recommend_c_ver_for_mode(aggregated)
    assert rec["c_ver"] == 15.0


def test_recommend_c_ver_returns_none_below_threshold():
    """If the best c_ver has mean_balance < min_balance, return None
    (don't pretend a degenerate solution is acceptable)."""
    aggregated = [
        {"c_ver": 5.0, "mean_balance": 0.3, "mean_spread": 5.0,
         "n_cells": 5, "per_cell_balances": {}, "median_balance": 0.0},
        {"c_ver": 15.0, "mean_balance": 0.5, "mean_spread": 7.0,
         "n_cells": 5, "per_cell_balances": {}, "median_balance": 0.0},
    ]
    rec = recommend_c_ver_for_mode(aggregated, min_balance=1.0)
    assert rec is None


def test_recommend_c_ver_empty():
    assert recommend_c_ver_for_mode([]) is None
