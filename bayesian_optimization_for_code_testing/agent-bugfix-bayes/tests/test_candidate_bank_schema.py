"""Tests for candidate bank schema validation."""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.candidate_bank.schema import (
    ArtifactPaths,
    CandidateBank,
    CandidateManifest,
    CriticOutputs,
    ProviderMeta,
    VerifierOutputs,
)


def _make_manifest(**overrides) -> CandidateManifest:
    defaults = {
        "benchmark": "bugsinpy",
        "project": "flask",
        "bug_id": 1,
        "candidate_id": "flask_1_g1_direct_fix_0",
        "generator_arm": "g1_direct_fix",
        "generator_seed": 0,
    }
    defaults.update(overrides)
    return CandidateManifest(**defaults)


def test_manifest_validates() -> None:
    """A valid manifest should pass pydantic validation."""
    m = _make_manifest()
    assert m.benchmark == "bugsinpy"
    assert m.project == "flask"
    assert m.bug_id == 1


def test_manifest_rejects_bad() -> None:
    """Missing required fields should raise a validation error."""
    with pytest.raises(Exception):
        CandidateManifest()  # Missing required fields


def test_manifest_roundtrip() -> None:
    """Save and load should preserve all fields."""
    m = _make_manifest(
        prompt_hash="abc123",
        patch_apply_success=True,
        wall_clock_seconds=1.5,
        critic_outputs=CriticOutputs(
            trigger_tests_pass=True,
            smoke_tests_pass=False,
        ),
        verifier_outputs=VerifierOutputs(
            full_suite_pass=True,
            failing_tests_before=["test_a"],
            failing_tests_after=[],
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "manifest.json"
        m.save(path)
        loaded = CandidateManifest.load(path)
        assert loaded.benchmark == m.benchmark
        assert loaded.prompt_hash == m.prompt_hash
        assert loaded.critic_outputs.trigger_tests_pass is True
        assert loaded.verifier_outputs.full_suite_pass is True
        assert loaded.verifier_outputs.failing_tests_before == ["test_a"]


def test_candidate_id_format() -> None:
    """Candidate ID should follow project_bugid_arm_seed pattern."""
    m = _make_manifest()
    assert "flask" in m.candidate_id
    assert "g1" in m.candidate_id


def test_candidate_bank_save_load() -> None:
    """CandidateBank should save and load candidates correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bank = CandidateBank(Path(tmpdir))
        m = _make_manifest()
        bank.save_candidate(m)
        loaded = bank.load_candidate("flask", 1, "g1_direct_fix", 0)
        assert loaded.candidate_id == m.candidate_id


def test_cost_aggregation() -> None:
    """CostBreakdown should correctly sum costs."""
    from abbo.core.types import CostBreakdown

    costs = CostBreakdown(
        diagnostic_cost=2.0,
        patch_cost=3.0,
        verifier_cost=20.0,
    )
    total = costs.compute_total()
    assert total == 25.0
