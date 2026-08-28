"""Critic likelihood estimation with Laplace smoothing.

Estimates P(z | Y=0, d) and P(z | Y=1, d) for each critic d
from the calibration split of the candidate bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from abbo.realworld.candidate_bank.schema import CandidateBank, CandidateManifest


@dataclass
class CriticLikelihood:
    """Likelihood table for a binary critic."""

    critic_name: str
    p_pass_given_correct: float  # P(pass | Y=1)
    p_pass_given_incorrect: float  # P(pass | Y=0)
    p_fail_given_correct: float  # P(fail | Y=1)
    p_fail_given_incorrect: float  # P(fail | Y=0)
    n_correct: int = 0
    n_incorrect: int = 0


def estimate_critic_likelihoods(
    candidates: list[CandidateManifest],
    critic_name: str,
    laplace_alpha: float = 1.0,
) -> CriticLikelihood:
    """Estimate critic likelihoods from labeled candidates.

    Uses Laplace smoothing with parameter *laplace_alpha*.
    Candidates with ``verifier_outputs.full_suite_pass`` as ground truth Y.
    """
    n_correct_pass = 0
    n_correct_fail = 0
    n_incorrect_pass = 0
    n_incorrect_fail = 0

    for c in candidates:
        y = c.verifier_outputs.full_suite_pass
        if y is None:
            continue
        critic_val = getattr(c.critic_outputs, critic_name, None)
        if critic_val is None:
            continue

        # For binary critics, True=pass, False=fail
        passed = bool(critic_val) if isinstance(critic_val, bool) else critic_val != ""
        if y:
            if passed:
                n_correct_pass += 1
            else:
                n_correct_fail += 1
        else:
            if passed:
                n_incorrect_pass += 1
            else:
                n_incorrect_fail += 1

    n_correct = n_correct_pass + n_correct_fail
    n_incorrect = n_incorrect_pass + n_incorrect_fail

    # Laplace smoothing
    p_pass_y1 = (n_correct_pass + laplace_alpha) / (n_correct + 2 * laplace_alpha)
    p_pass_y0 = (n_incorrect_pass + laplace_alpha) / (n_incorrect + 2 * laplace_alpha)

    return CriticLikelihood(
        critic_name=critic_name,
        p_pass_given_correct=p_pass_y1,
        p_pass_given_incorrect=p_pass_y0,
        p_fail_given_correct=1 - p_pass_y1,
        p_fail_given_incorrect=1 - p_pass_y0,
        n_correct=n_correct,
        n_incorrect=n_incorrect,
    )
