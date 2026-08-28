"""Pydantic models for the candidate bank.

Each candidate patch has a manifest with all metadata, critic outputs,
verifier outputs, and artifact paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CriticOutputs(BaseModel):
    """Binary critic outputs for a candidate patch."""

    trigger_tests_pass: bool | None = None
    smoke_tests_pass: bool | None = None
    static_checks_pass: bool | None = None
    cheap_judge_label: str | None = None


class VerifierOutputs(BaseModel):
    """Full verifier outputs for a candidate patch."""

    full_suite_pass: bool | None = None
    failing_tests_before: list[str] = Field(default_factory=list)
    failing_tests_after: list[str] = Field(default_factory=list)
    regression_failures: list[str] = Field(default_factory=list)


class ArtifactPaths(BaseModel):
    """Paths to stored artifacts for a candidate."""

    raw_response: str | None = None
    git_diff: str | None = None
    pytest_logs: str | None = None
    stack_traces: str | None = None
    timing_json: str | None = None


class ProviderMeta(BaseModel):
    """LLM provider metadata."""

    provider: str = ""
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 0


class CandidateManifest(BaseModel):
    """Complete manifest for a single candidate patch."""

    benchmark: str
    project: str
    bug_id: int
    parent_candidate_id: str | None = None
    candidate_id: str
    generator_arm: str
    generator_seed: int
    provider_meta: ProviderMeta = Field(default_factory=ProviderMeta)
    prompt_hash: str = ""
    prompt_text_path: str = ""
    diff_path: str = ""
    changed_files: list[str] = Field(default_factory=list)
    patch_apply_success: bool = False
    wall_clock_seconds: float = 0.0
    api_token_counts: dict[str, int] = Field(default_factory=dict)
    api_dollar_cost: float = 0.0
    critic_outputs: CriticOutputs = Field(default_factory=CriticOutputs)
    verifier_outputs: VerifierOutputs = Field(default_factory=VerifierOutputs)
    artifact_paths: ArtifactPaths = Field(default_factory=ArtifactPaths)

    def save(self, path: Path) -> None:
        """Save manifest to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> CandidateManifest:
        """Load manifest from a JSON file."""
        return cls.model_validate_json(path.read_text())


class CandidateBank:
    """Manages a collection of candidate manifests on disk."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def candidate_dir(
        self, project: str, bug_id: int, arm: str, seed: int
    ) -> Path:
        """Return the directory for a specific candidate."""
        return self.root_dir / project / str(bug_id) / arm / str(seed)

    def save_candidate(self, manifest: CandidateManifest) -> Path:
        """Save a candidate manifest and return its path."""
        d = self.candidate_dir(
            manifest.project,
            manifest.bug_id,
            manifest.generator_arm,
            manifest.generator_seed,
        )
        path = d / "manifest.json"
        manifest.save(path)
        return path

    def load_candidate(
        self, project: str, bug_id: int, arm: str, seed: int
    ) -> CandidateManifest:
        """Load a candidate manifest."""
        path = self.candidate_dir(project, bug_id, arm, seed) / "manifest.json"
        return CandidateManifest.load(path)

    def list_candidates(
        self, project: str | None = None, bug_id: int | None = None
    ) -> list[CandidateManifest]:
        """List all candidates, optionally filtered by project and bug_id."""
        manifests = []
        search_dir = self.root_dir
        if project:
            search_dir = search_dir / project
            if bug_id is not None:
                search_dir = search_dir / str(bug_id)

        for mf in search_dir.rglob("manifest.json"):
            try:
                manifests.append(CandidateManifest.load(mf))
            except Exception:
                continue
        return manifests

    def get_all_bug_ids(self) -> list[tuple[str, int]]:
        """Return all (project, bug_id) pairs in the bank."""
        pairs = set()
        for mf in self.root_dir.rglob("manifest.json"):
            try:
                m = CandidateManifest.load(mf)
                pairs.add((m.project, m.bug_id))
            except Exception:
                continue
        return sorted(pairs)
