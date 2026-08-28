"""Unit tests for empirical-Bayes calibration of critic likelihoods.

Fast, deterministic, no LLM calls. Validates the Beta-Binomial estimator
with synthetic samples of known ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.bayes_agent import (
    CRITIC_LIKELIHOODS,
    DPPlanner,
    bayes_update,
)
from abbo.realworld.agents.calibration import (
    CalibrationSample,
    calibrate_likelihoods,
    calibration_report,
    format_comparison,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig


def _synthetic_samples(
    critic: str,
    n_y1: int,
    pass_rate_y1: float,
    n_y0: int,
    pass_rate_y0: float,
) -> list[CalibrationSample]:
    """Build samples for one critic with known pass rates (deterministic split)."""
    samples: list[CalibrationSample] = []
    n_pass_y1 = int(round(n_y1 * pass_rate_y1))
    for i in range(n_y1):
        samples.append(CalibrationSample(
            bug_id=f"synth_y1_{i}", arm="g1", critic=critic,
            critic_passed=(i < n_pass_y1), patch_correct=True,
        ))
    n_pass_y0 = int(round(n_y0 * pass_rate_y0))
    for i in range(n_y0):
        samples.append(CalibrationSample(
            bug_id=f"synth_y0_{i}", arm="g1", critic=critic,
            critic_passed=(i < n_pass_y0), patch_correct=False,
        ))
    return samples


class TestCalibrateLikelihoods:
    """Beta-Binomial posterior mean behaves correctly under synthetic data."""

    def test_recovers_rates_with_many_samples(self):
        samples = _synthetic_samples("c", n_y1=100, pass_rate_y1=0.9,
                                      n_y0=100, pass_rate_y0=0.3)
        cal, _ = calibrate_likelihoods(samples, prior_alpha=1.0, prior_beta=1.0)
        # With 100 samples, Beta(1,1) prior barely shifts: 91/102, 31/102
        assert abs(cal["c"]["p_pass_y1"] - 91/102) < 1e-6
        assert abs(cal["c"]["p_pass_y0"] - 31/102) < 1e-6

    def test_prior_dominates_small_samples(self):
        samples = _synthetic_samples("c", n_y1=2, pass_rate_y1=1.0,
                                      n_y0=2, pass_rate_y0=0.0)
        # With Beta(2,2), posterior means are (2+2)/(2+2+2)=0.667 and (2+0)/(2+2+2)=0.333
        cal, _ = calibrate_likelihoods(samples, prior_alpha=2.0, prior_beta=2.0)
        assert abs(cal["c"]["p_pass_y1"] - 4/6) < 1e-6
        assert abs(cal["c"]["p_pass_y0"] - 2/6) < 1e-6

    def test_enforces_informativeness_gap(self):
        """If observed y1 and y0 rates are identical, min_gap enforces separation."""
        samples = _synthetic_samples("c", n_y1=100, pass_rate_y1=0.5,
                                      n_y0=100, pass_rate_y0=0.5)
        cal, _ = calibrate_likelihoods(samples, min_gap=0.1)
        gap = cal["c"]["p_pass_y1"] - cal["c"]["p_pass_y0"]
        assert gap >= 0.1 - 1e-9

    def test_clipping(self):
        samples = _synthetic_samples("c", n_y1=1000, pass_rate_y1=1.0,
                                      n_y0=1000, pass_rate_y0=0.0)
        cal, _ = calibrate_likelihoods(samples, prior_alpha=0.001, prior_beta=0.001,
                                        clip=(0.05, 0.95))
        assert cal["c"]["p_pass_y1"] <= 0.95
        assert cal["c"]["p_pass_y0"] >= 0.05

    def test_multiple_critics_independent(self):
        a = _synthetic_samples("a", n_y1=50, pass_rate_y1=0.9, n_y0=50, pass_rate_y0=0.1)
        b = _synthetic_samples("b", n_y1=50, pass_rate_y1=0.6, n_y0=50, pass_rate_y0=0.4)
        cal, _ = calibrate_likelihoods(a + b)
        # Highly-informative critic a should have bigger gap than weakly-informative b
        gap_a = cal["a"]["p_pass_y1"] - cal["a"]["p_pass_y0"]
        gap_b = cal["b"]["p_pass_y1"] - cal["b"]["p_pass_y0"]
        assert gap_a > gap_b

    def test_counts_returned(self):
        samples = _synthetic_samples("c", n_y1=10, pass_rate_y1=0.7,
                                      n_y0=5, pass_rate_y0=0.2)
        _, counts = calibrate_likelihoods(samples)
        assert counts["c"]["n_y1"] == 10
        assert counts["c"]["pass_y1"] == 7
        assert counts["c"]["n_y0"] == 5
        assert counts["c"]["pass_y0"] == 1


class TestCalibrationReport:
    """The report bundles hand-tuned + calibrated + counts for Allure attachments."""

    def test_report_has_both_tables(self):
        samples = _synthetic_samples("filtering_check", n_y1=20, pass_rate_y1=0.9,
                                      n_y0=20, pass_rate_y0=0.4)
        report = calibration_report(samples)
        assert "filtering_check" in report.calibrated
        assert "filtering_check" in report.hand_tuned
        assert report.n_bugs == 40   # distinct synth bug_ids
        assert report.n_samples == 40

    def test_format_comparison_is_string(self):
        samples = _synthetic_samples("filtering_check", n_y1=10, pass_rate_y1=0.9,
                                      n_y0=10, pass_rate_y0=0.4)
        txt = format_comparison(calibration_report(samples))
        assert "filtering_check" in txt
        assert "hand_y1" in txt
        assert "cal_y1" in txt


class TestInjectedLikelihoodsEndToEnd:
    """Verify DPPlanner and bayes_update honor injected likelihoods."""

    def test_bayes_update_uses_injected_table(self):
        # Injected table makes filtering_check a perfect signal; hand-tuned doesn't
        perfect = {name: {"p_pass_y1": 0.99, "p_pass_y0": 0.01}
                   for name in CRITIC_LIKELIHOODS}
        b_default = bayes_update(0.5, "filtering_check", passed=True)
        b_inject = bayes_update(0.5, "filtering_check", passed=True,
                                 likelihoods=perfect)
        assert b_inject > b_default

    def test_dp_planner_uses_injected_table(self):
        """A planner with noisier critics should have lower value at mid belief."""
        costs = AgentCostConfig()
        noisy = {name: {"p_pass_y1": 0.55, "p_pass_y0": 0.50}
                 for name in CRITIC_LIKELIHOODS}
        p_default = DPPlanner(costs, max_generators=3, max_verifications=2)
        p_noisy = DPPlanner(costs, max_generators=3, max_verifications=2,
                            critic_likelihoods=noisy)
        p_default.solve()
        p_noisy.solve()
        _, v_default = p_default.choose_action(
            belief=0.4, gen_left=2, crit_used=frozenset(), ver_left=2
        )
        _, v_noisy = p_noisy.choose_action(
            belief=0.4, gen_left=2, crit_used=frozenset(), ver_left=2
        )
        # Noisier critics → lower expected utility (critics less useful)
        assert v_default >= v_noisy - 0.01
