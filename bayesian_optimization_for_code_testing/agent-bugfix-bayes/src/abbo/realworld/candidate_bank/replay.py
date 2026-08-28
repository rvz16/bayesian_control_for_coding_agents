"""Offline replay engine for candidate bank evaluation.

Loads pre-built candidate bank and replays critic/verifier results
without making any live API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from abbo.realworld.candidate_bank.schema import CandidateBank, CandidateManifest


class ReplayEngine:
    """Replays pre-computed candidate results for offline evaluation."""

    def __init__(self, bank: CandidateBank):
        self.bank = bank
        self._cache: dict[str, CandidateManifest] = {}

    def _load_candidates_for_bug(
        self, project: str, bug_id: int
    ) -> dict[str, CandidateManifest]:
        """Load and cache all candidates for a specific bug."""
        key = f"{project}_{bug_id}"
        if key not in self._cache:
            candidates = self.bank.list_candidates(project, bug_id)
            for c in candidates:
                self._cache[c.candidate_id] = c
        return {
            cid: m
            for cid, m in self._cache.items()
            if m.project == project and m.bug_id == bug_id
        }

    def get_candidate(
        self, project: str, bug_id: int, arm: str, seed: int
    ) -> CandidateManifest | None:
        """Get a specific candidate from the bank."""
        candidates = self._load_candidates_for_bug(project, bug_id)
        cid = f"{project}_{bug_id}_{arm}_{seed}"
        return candidates.get(cid)

    def get_critic_result(
        self, candidate_id: str, critic_name: str
    ) -> bool | None:
        """Get the stored critic result for a candidate."""
        manifest = self._cache.get(candidate_id)
        if manifest is None:
            return None
        critics = manifest.critic_outputs
        return getattr(critics, critic_name, None)

    def get_verifier_result(self, candidate_id: str) -> bool | None:
        """Get the stored verifier result for a candidate."""
        manifest = self._cache.get(candidate_id)
        if manifest is None:
            return None
        return manifest.verifier_outputs.full_suite_pass

    def list_arms_for_bug(
        self, project: str, bug_id: int
    ) -> list[str]:
        """List available generator arms for a bug."""
        candidates = self._load_candidates_for_bug(project, bug_id)
        return sorted({m.generator_arm for m in candidates.values() if m.generator_arm != "root"})

    def list_bugs(self) -> list[tuple[str, int]]:
        """List all (project, bug_id) pairs in the bank."""
        return self.bank.get_all_bug_ids()
