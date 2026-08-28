"""Model fitting orchestrator.

Fits all likelihoods and transitions from the calibration split and
saves/loads the fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abbo.config import project_root
from abbo.realworld.calibration.likelihoods import (
    CriticLikelihood,
    estimate_critic_likelihoods,
)
from abbo.realworld.calibration.transitions import (
    TransitionModel,
    estimate_transitions,
)
from abbo.realworld.candidate_bank.schema import CandidateBank, CandidateManifest
from abbo.utils.io import load_json, save_json


@dataclass
class FittedBayesModel:
    """A fitted Bayesian model with all likelihoods and transitions."""

    critic_likelihoods: dict[str, CriticLikelihood] = field(default_factory=dict)
    transition_models: dict[str, TransitionModel] = field(default_factory=dict)
    prior_correct: float = 0.1  # P(Y=1) prior for a new candidate

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for JSON storage."""
        return {
            "critic_likelihoods": {
                name: {
                    "critic_name": lk.critic_name,
                    "p_pass_given_correct": lk.p_pass_given_correct,
                    "p_pass_given_incorrect": lk.p_pass_given_incorrect,
                    "p_fail_given_correct": lk.p_fail_given_correct,
                    "p_fail_given_incorrect": lk.p_fail_given_incorrect,
                    "n_correct": lk.n_correct,
                    "n_incorrect": lk.n_incorrect,
                }
                for name, lk in self.critic_likelihoods.items()
            },
            "transition_models": {
                name: {
                    "arm": tm.arm,
                    "p01": tm.p01,
                    "p10": tm.p10,
                    "n_from_incorrect": tm.n_from_incorrect,
                    "n_from_correct": tm.n_from_correct,
                }
                for name, tm in self.transition_models.items()
            },
            "prior_correct": self.prior_correct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FittedBayesModel:
        """Deserialize from a dict."""
        model = cls(prior_correct=d.get("prior_correct", 0.1))
        for name, lk_d in d.get("critic_likelihoods", {}).items():
            model.critic_likelihoods[name] = CriticLikelihood(**lk_d)
        for name, tm_d in d.get("transition_models", {}).items():
            model.transition_models[name] = TransitionModel(**tm_d)
        return model


def fit_bayes_model(
    candidates: list[CandidateManifest],
    parent_child_pairs: list[tuple[CandidateManifest, CandidateManifest]],
    critic_names: list[str] | None = None,
    arm_names: list[str] | None = None,
) -> FittedBayesModel:
    """Fit all components of the Bayesian model from calibration data."""
    if critic_names is None:
        critic_names = ["trigger_tests_pass", "smoke_tests_pass", "static_checks_pass"]
    if arm_names is None:
        arm_names = ["g1_direct_fix", "g2_localized_fix", "g3_test_guided_fix", "g4_minimal_diff_fix"]

    model = FittedBayesModel()
    for critic in critic_names:
        model.critic_likelihoods[critic] = estimate_critic_likelihoods(
            candidates, critic
        )
    for arm in arm_names:
        model.transition_models[arm] = estimate_transitions(
            parent_child_pairs, arm
        )
    return model


def fit_and_save(config: dict[str, Any], benchmark: str) -> FittedBayesModel:
    """Fit the model and save it to disk."""
    bank_dir = project_root() / config.get(
        "candidate_bank_dir", f"data/candidate_bank/{benchmark}"
    )
    bank = CandidateBank(bank_dir)
    candidates = bank.list_candidates()

    # Build parent-child pairs
    by_id = {c.candidate_id: c for c in candidates}
    pairs = []
    for c in candidates:
        if c.parent_candidate_id and c.parent_candidate_id in by_id:
            pairs.append((by_id[c.parent_candidate_id], c))

    model = fit_bayes_model(candidates, pairs)
    out_path = project_root() / "results" / benchmark / "fitted_model.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(model.to_dict(), out_path)
    return model
