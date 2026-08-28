"""Prompt templates for generator arms.

All templates are versioned and hashed for reproducibility.
"""

from __future__ import annotations

from abbo.utils.hashing import sha256_hex

PROMPT_VERSION = "v1.0"

G1_DIRECT_FIX = """You are a software engineer fixing a bug.

## Bug Information
Project: {project}
Bug ID: {bug_id}

## Failing Test Output
{failing_test_output}

## Relevant Source Files
{source_context}

## Instructions
Produce a minimal patch that fixes this bug. Output only a unified diff.
Do not modify test files. Focus on the root cause.
"""

G2_LOCALIZED_FIX = """You are a software engineer fixing a bug.

## Bug Information
Project: {project}
Bug ID: {bug_id}

## Step 1: Localization
Based on the failing tests and stack trace, identify the most suspicious
files and functions.

## Failing Test Output
{failing_test_output}

## Stack Trace
{stack_trace}

## Suspicious Files (top candidates)
{suspicious_files}

## Instructions
Produce a patch that fixes the bug by modifying only the top-k suspicious files.
Output only a unified diff. Do not modify test files.
"""

G3_TEST_GUIDED_FIX = """You are a software engineer fixing a bug.

## Bug Information
Project: {project}
Bug ID: {bug_id}

## Triggering Tests
{trigger_tests}

## Full Stack Trace
{stack_trace}

## Source Context Around Failure Points
{source_context}

## Instructions
Fix the bug so that all triggering tests pass. You must NOT modify any test files.
Focus on the production code that the tests exercise.
Output only a unified diff.
"""

G4_MINIMAL_DIFF_FIX = """You are a software engineer fixing a bug.

## Bug Information
Project: {project}
Bug ID: {bug_id}

## Failing Test Output
{failing_test_output}

## Source Context
{source_context}

## Instructions
Produce the SMALLEST possible change to the production code that fixes this bug.
- Prefer one-line changes over multi-line changes.
- Do not add new functions or classes unless absolutely necessary.
- Do not refactor or clean up surrounding code.
- Do not modify test files.
Output only a unified diff.
"""

TEMPLATES = {
    "g1_direct_fix": G1_DIRECT_FIX,
    "g2_localized_fix": G2_LOCALIZED_FIX,
    "g3_test_guided_fix": G3_TEST_GUIDED_FIX,
    "g4_minimal_diff_fix": G4_MINIMAL_DIFF_FIX,
}


def get_template(arm: str) -> str:
    """Return the prompt template for a generator arm."""
    return TEMPLATES[arm]


def hash_prompt(template: str, context: dict[str, str]) -> str:
    """Produce a deterministic hash of a rendered prompt."""
    rendered = template.format(**{k: str(v) for k, v in context.items()})
    return sha256_hex(rendered)
