"""Generator arm implementations.

Each arm takes the current candidate/repo state and produces a patch diff.
A provider abstraction allows plugging in Claude, OpenAI, or local models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abbo.realworld.agents.prompts import TEMPLATES, hash_prompt
from abbo.realworld.candidate_bank.schema import CandidateManifest, ProviderMeta
from abbo.utils.hashing import short_hash
from abbo.utils.timing import timed


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    provider: str = "stub"
    model: str = "stub"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class GenerationResult:
    """Result of a single patch generation attempt."""

    diff: str = ""
    raw_response: str = ""
    token_counts: dict[str, int] = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    prompt_hash: str = ""


class GeneratorArm(ABC):
    """Base class for patch generator arms."""

    arm_name: str = "base"

    @abstractmethod
    def generate(
        self,
        context: dict[str, Any],
        provider: ProviderConfig,
        seed: int,
    ) -> GenerationResult:
        """Generate a patch diff from the given context."""
        ...


class G1DirectFix(GeneratorArm):
    """Single prompt, minimal patch, using issue/failing test context."""

    arm_name = "g1_direct_fix"

    def generate(
        self, context: dict[str, Any], provider: ProviderConfig, seed: int
    ) -> GenerationResult:
        template = TEMPLATES[self.arm_name]
        ctx = {
            "project": context.get("project", ""),
            "bug_id": context.get("bug_id", ""),
            "failing_test_output": context.get("failing_test_output", ""),
            "source_context": context.get("source_context", ""),
        }
        phash = hash_prompt(template, ctx)
        with timed("g1_direct_fix") as t:
            # In production, this would call the LLM
            diff = context.get("mock_diff", "# no diff generated (stub)")
        return GenerationResult(
            diff=diff, prompt_hash=phash, wall_clock_seconds=t.elapsed_seconds
        )


class G2LocalizedFix(GeneratorArm):
    """First localize suspicious files, then patch top-k files."""

    arm_name = "g2_localized_fix"

    def generate(
        self, context: dict[str, Any], provider: ProviderConfig, seed: int
    ) -> GenerationResult:
        template = TEMPLATES[self.arm_name]
        ctx = {
            "project": context.get("project", ""),
            "bug_id": context.get("bug_id", ""),
            "failing_test_output": context.get("failing_test_output", ""),
            "stack_trace": context.get("stack_trace", ""),
            "suspicious_files": context.get("suspicious_files", ""),
        }
        phash = hash_prompt(template, ctx)
        with timed("g2_localized_fix") as t:
            diff = context.get("mock_diff", "# no diff generated (stub)")
        return GenerationResult(
            diff=diff, prompt_hash=phash, wall_clock_seconds=t.elapsed_seconds
        )


class G3TestGuidedFix(GeneratorArm):
    """Emphasize triggering tests and stack trace."""

    arm_name = "g3_test_guided_fix"

    def generate(
        self, context: dict[str, Any], provider: ProviderConfig, seed: int
    ) -> GenerationResult:
        template = TEMPLATES[self.arm_name]
        ctx = {
            "project": context.get("project", ""),
            "bug_id": context.get("bug_id", ""),
            "trigger_tests": context.get("trigger_tests", ""),
            "stack_trace": context.get("stack_trace", ""),
            "source_context": context.get("source_context", ""),
        }
        phash = hash_prompt(template, ctx)
        with timed("g3_test_guided_fix") as t:
            diff = context.get("mock_diff", "# no diff generated (stub)")
        return GenerationResult(
            diff=diff, prompt_hash=phash, wall_clock_seconds=t.elapsed_seconds
        )


class G4MinimalDiffFix(GeneratorArm):
    """Ask for smallest production-code-only change."""

    arm_name = "g4_minimal_diff_fix"

    def generate(
        self, context: dict[str, Any], provider: ProviderConfig, seed: int
    ) -> GenerationResult:
        template = TEMPLATES[self.arm_name]
        ctx = {
            "project": context.get("project", ""),
            "bug_id": context.get("bug_id", ""),
            "failing_test_output": context.get("failing_test_output", ""),
            "source_context": context.get("source_context", ""),
        }
        phash = hash_prompt(template, ctx)
        with timed("g4_minimal_diff_fix") as t:
            diff = context.get("mock_diff", "# no diff generated (stub)")
        return GenerationResult(
            diff=diff, prompt_hash=phash, wall_clock_seconds=t.elapsed_seconds
        )


GENERATOR_ARMS: dict[str, GeneratorArm] = {
    "g1_direct_fix": G1DirectFix(),
    "g2_localized_fix": G2LocalizedFix(),
    "g3_test_guided_fix": G3TestGuidedFix(),
    "g4_minimal_diff_fix": G4MinimalDiffFix(),
}
