"""Empirical-Bayes calibration of critic likelihoods.

Fits P(critic_pass | Y=1) and P(critic_pass | Y=0) per critic from observed
(critic_outcome, ground_truth_correctness) pairs, then plugs the calibrated
table into DPPlanner for data-driven (rather than hand-tuned) planning.

Estimator: Beta-Binomial posterior mean with Beta(prior_alpha, prior_beta) prior.

    p_pass_y1 = (alpha + #pass_when_correct) / (alpha + beta + #correct)
    p_pass_y0 = (alpha + #pass_when_wrong)   / (alpha + beta + #wrong)

The critic-informativeness constraint `p_pass_y1 > p_pass_y0` is enforced as a
small floor, so degenerate critics (never observed in one class) still yield
usable likelihoods.
"""

from __future__ import annotations

import copy
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from abbo.realworld.agents.bayes_agent import CRITIC_LIKELIHOODS
from abbo.realworld.agents.code_fixer import extract_code_from_response
from abbo.realworld.agents.llm_provider import LLMConfig, call_llm
from abbo.realworld.agents.real_bugs import (
    BUGGY_SOURCES_REAL,
    LOG_PARSER_TEST_CONTENT,
    REAL_ARM_PROMPTS,
    REAL_CRITIC_TESTS,
)
from abbo.realworld.agents.simple_agent import (
    _run_critic_tests,
    _run_full_tests,
)

console = Console()


@dataclass
class CalibrationSample:
    """One (critic, patch) observation with ground-truth correctness."""
    bug_id: str
    arm: str
    critic: str
    critic_passed: bool
    patch_correct: bool


@dataclass
class CalibrationReport:
    """Summary of a calibration run."""
    samples: list[CalibrationSample]
    hand_tuned: dict[str, dict[str, float]]
    calibrated: dict[str, dict[str, float]]
    n_bugs: int
    n_samples: int
    prior_alpha: float
    prior_beta: float
    per_critic_counts: dict[str, dict[str, int]] = field(default_factory=dict)


def collect_calibration_data(
    bug_ids: list[str],
    llm_config: LLMConfig,
    arms: list[str] | None = None,
    verbose: bool = True,
) -> list[CalibrationSample]:
    """Generate patches and record (critic, patch) outcomes with ground truth.

    For each bug and each arm, we:
      1. Generate a candidate patch with the LLM
      2. Run the full test suite → ground-truth `patch_correct`
      3. Run each critic subset → `critic_passed` per critic

    Cost per (bug, arm): 1 LLM call + 1 full test + K critic runs
    Total samples: len(bug_ids) × len(arms) × K   (K = #critics)
    """
    arms = arms if arms is not None else list(REAL_ARM_PROMPTS.keys())
    samples: list[CalibrationSample] = []

    for i, bug_id in enumerate(bug_ids):
        buggy_source = BUGGY_SOURCES_REAL[bug_id]

        for arm in arms:
            with tempfile.TemporaryDirectory() as tmpdir:
                workdir = Path(tmpdir)
                src_file = workdir / "log_parser.py"
                test_file = workdir / "test_log_parser.py"
                test_file.write_text(
                    LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir))
                )
                src_file.write_text(buggy_source)

                _, test_output, _ = _run_full_tests(workdir)

                prompt = REAL_ARM_PROMPTS[arm].format(
                    source_code=buggy_source,
                    test_output=test_output,
                    test_code=LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)),
                )
                if verbose:
                    console.print(
                        f"  [{i+1}/{len(bug_ids)}] {bug_id} arm={arm}: "
                        f"generating ({llm_config.model})..."
                    )
                llm_resp = call_llm(prompt, llm_config)
                fixed_code = extract_code_from_response(llm_resp.text)
                src_file.write_text(fixed_code)

                all_pass, _, _ = _run_full_tests(workdir)

                for critic_name, test_subset in REAL_CRITIC_TESTS.items():
                    critic_pass, _ = _run_critic_tests(workdir, test_subset)
                    samples.append(CalibrationSample(
                        bug_id=bug_id,
                        arm=arm,
                        critic=critic_name,
                        critic_passed=critic_pass,
                        patch_correct=all_pass,
                    ))

                if verbose:
                    n_crit_pass = sum(
                        1 for s in samples
                        if s.bug_id == bug_id and s.arm == arm and s.critic_passed
                    )
                    color = "green" if all_pass else "red"
                    console.print(
                        f"    [{color}]patch_correct={all_pass}[/{color}]  "
                        f"critics_passed={n_crit_pass}/{len(REAL_CRITIC_TESTS)}"
                    )

    return samples


def calibrate_likelihoods(
    samples: list[CalibrationSample],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    min_gap: float = 0.05,
    clip: tuple[float, float] = (0.02, 0.98),
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """Beta-Binomial posterior mean of p_pass_y1, p_pass_y0 per critic.

    Args:
        samples: observed (critic_pass, patch_correct) pairs.
        prior_alpha, prior_beta: pseudocount prior, Beta(alpha, beta).
            (1, 1) = uniform prior; (2, 2) = mildly concentrated at 0.5;
            (8, 2) = informed prior for critics that should pass on correct patches.
        min_gap: minimum (p_pass_y1 − p_pass_y0) to keep critic informative;
            if violated, both are pulled toward their midpoint.
        clip: (low, high) bounds to keep Bayes rule non-degenerate.

    Returns:
        (calibrated_likelihoods, per_critic_counts)
    """
    calibrated: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    critics = sorted({s.critic for s in samples})

    for critic in critics:
        s_c = [s for s in samples if s.critic == critic]
        y1 = [s for s in s_c if s.patch_correct]
        y0 = [s for s in s_c if not s.patch_correct]

        pass_y1 = sum(1 for s in y1 if s.critic_passed)
        pass_y0 = sum(1 for s in y0 if s.critic_passed)

        p_pass_y1 = (prior_alpha + pass_y1) / (prior_alpha + prior_beta + len(y1))
        p_pass_y0 = (prior_alpha + pass_y0) / (prior_alpha + prior_beta + len(y0))

        lo, hi = clip
        p_pass_y1 = max(lo, min(hi, p_pass_y1))
        p_pass_y0 = max(lo, min(hi, p_pass_y0))

        if p_pass_y1 - p_pass_y0 < min_gap:
            mid = (p_pass_y1 + p_pass_y0) / 2
            p_pass_y1 = min(hi, mid + min_gap / 2)
            p_pass_y0 = max(lo, mid - min_gap / 2)

        calibrated[critic] = {
            "p_pass_y1": p_pass_y1,
            "p_pass_y0": p_pass_y0,
        }
        counts[critic] = {
            "n_y1": len(y1),
            "pass_y1": pass_y1,
            "n_y0": len(y0),
            "pass_y0": pass_y0,
        }

    return calibrated, counts


def calibration_report(
    samples: list[CalibrationSample],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    hand_tuned: dict[str, dict[str, float]] | None = None,
) -> CalibrationReport:
    """Build a report with both hand-tuned and calibrated tables side by side.

    Args:
        hand_tuned: the baseline table to display alongside the fitted one.
            Defaults to log_parser's CRITIC_LIKELIHOODS. Pass HumanEvalFix's
            HE_CRITIC_LIKELIHOODS when calibrating on HumanEvalFix samples.
    """
    calibrated, counts = calibrate_likelihoods(
        samples, prior_alpha=prior_alpha, prior_beta=prior_beta
    )
    baseline = hand_tuned if hand_tuned is not None else CRITIC_LIKELIHOODS
    return CalibrationReport(
        samples=samples,
        hand_tuned=copy.deepcopy(baseline),
        calibrated=calibrated,
        n_bugs=len({s.bug_id for s in samples}),
        n_samples=len(samples),
        prior_alpha=prior_alpha,
        prior_beta=prior_beta,
        per_critic_counts=counts,
    )


def format_comparison(report: CalibrationReport) -> str:
    """Pretty-print hand-tuned vs calibrated side by side."""
    lines = [
        f"Calibration: {report.n_bugs} bugs, {report.n_samples} samples, "
        f"Beta({report.prior_alpha}, {report.prior_beta}) prior",
        "",
        f"{'critic':<25} {'n_y1':>5} {'n_y0':>5} "
        f"{'hand_y1':>8} {'cal_y1':>8} {'hand_y0':>8} {'cal_y0':>8} {'gap_cal':>8}",
        "-" * 90,
    ]
    for critic in report.calibrated:
        hand = report.hand_tuned.get(critic, {"p_pass_y1": 0.0, "p_pass_y0": 0.0})
        cal = report.calibrated[critic]
        cnt = report.per_critic_counts.get(critic, {"n_y1": 0, "n_y0": 0})
        gap_cal = cal["p_pass_y1"] - cal["p_pass_y0"]
        lines.append(
            f"{critic:<25} {cnt['n_y1']:>5} {cnt['n_y0']:>5} "
            f"{hand['p_pass_y1']:>8.3f} {cal['p_pass_y1']:>8.3f} "
            f"{hand['p_pass_y0']:>8.3f} {cal['p_pass_y0']:>8.3f} "
            f"{gap_cal:>8.3f}"
        )
    return "\n".join(lines)
