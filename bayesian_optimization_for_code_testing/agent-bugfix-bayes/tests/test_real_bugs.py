"""Tests for the real bug definitions (log parser module).

Verifies that:
- Clean source passes all tests
- Each bug fails the expected subset of tests
- Bug metadata is consistent
- Test subsets (critics) cover different aspects
- Prompt templates are well-formed

All tests run without LLM calls (fast, deterministic).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.real_bugs import (
    LOG_PARSER_CLEAN,
    LOG_PARSER_TEST_CONTENT,
    BUGGY_SOURCES_REAL,
    BUG_METADATA,
    REAL_CRITIC_TESTS,
    REAL_ARM_PROMPTS,
)


def _run_tests_on_source(source: str) -> tuple[int, int, list[str]]:
    """Write source, run pytest, return (passed, failed, failing_names)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "log_parser.py").write_text(source)
        (p / "test_log_parser.py").write_text(
            LOG_PARSER_TEST_CONTENT.format(src_dir=str(p))
        )
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", str(p / "test_log_parser.py")],
            cwd=p, capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        passed = r.stdout.count(" PASSED")
        failed = r.stdout.count(" FAILED")
        failing = []
        for line in r.stdout.splitlines():
            if "FAILED" in line:
                name = line.split("::")[-1].split(" ")[0]
                if name not in failing:
                    failing.append(name)
        return passed, failed, failing


# ---------------------------------------------------------------------------
# Clean source tests
# ---------------------------------------------------------------------------

class TestCleanSource:
    """Verify the clean (correct) log parser passes all tests."""

    def test_clean_passes_all(self):
        passed, failed, _ = _run_tests_on_source(LOG_PARSER_CLEAN)
        assert failed == 0
        assert passed == 36

    def test_clean_source_has_all_functions(self):
        for fn in ["parse_line", "parse_log", "filter_by_level", "filter_by_time",
                    "filter_by_source", "count_by_level", "count_by_source",
                    "error_windows", "error_rate", "latest_by_source", "top_sources"]:
            assert f"def {fn}" in LOG_PARSER_CLEAN, f"Missing function: {fn}"

    def test_clean_source_compact(self):
        """Source should be compact enough for 7B models to reproduce."""
        lines = LOG_PARSER_CLEAN.strip().splitlines()
        assert len(lines) <= 200, f"Source too long: {len(lines)} lines"


# ---------------------------------------------------------------------------
# Bug definition tests
# ---------------------------------------------------------------------------

class TestBugDefinitions:
    """Verify each bug fails the expected tests."""

    @pytest.mark.parametrize("bug_id", sorted(BUGGY_SOURCES_REAL.keys()))
    def test_bug_fails_some_tests(self, bug_id):
        """Each bug should cause at least 1 test failure."""
        _, failed, _ = _run_tests_on_source(BUGGY_SOURCES_REAL[bug_id])
        assert failed > 0, f"{bug_id} should cause failures"

    @pytest.mark.parametrize("bug_id", sorted(BUGGY_SOURCES_REAL.keys()))
    def test_bug_has_metadata(self, bug_id):
        """Each bug should have difficulty and description."""
        assert bug_id in BUG_METADATA
        meta = BUG_METADATA[bug_id]
        assert "difficulty" in meta
        assert meta["difficulty"] in ("easy", "medium", "hard")
        assert "description" in meta
        assert len(meta["description"]) > 5

    def test_difficulty_distribution(self):
        """Should have bugs at all three difficulty levels."""
        difficulties = {BUG_METADATA[b]["difficulty"] for b in BUG_METADATA}
        assert difficulties == {"easy", "medium", "hard"}

    def test_fifty_bugs_defined(self):
        assert len(BUGGY_SOURCES_REAL) == 50

    def test_bug_is_single_line_change(self):
        """Each bug should differ from clean source in exactly 1 line."""
        for bug_id, buggy in BUGGY_SOURCES_REAL.items():
            clean_lines = LOG_PARSER_CLEAN.splitlines()
            buggy_lines = buggy.splitlines()
            assert len(clean_lines) == len(buggy_lines), f"{bug_id}: line count changed"
            diffs = sum(1 for a, b in zip(clean_lines, buggy_lines) if a != b)
            assert diffs == 1, f"{bug_id}: expected 1 changed line, got {diffs}"


# ---------------------------------------------------------------------------
# Critic subset tests
# ---------------------------------------------------------------------------

class TestCriticSubsets:
    """Verify critic test subsets are well-defined."""

    def test_eight_critic_groups(self):
        assert len(REAL_CRITIC_TESTS) == 8

    def test_all_critic_tests_exist_in_suite(self):
        """All critic test names should appear in the test content."""
        for group, tests in REAL_CRITIC_TESTS.items():
            for test_name in tests:
                assert test_name in LOG_PARSER_TEST_CONTENT, (
                    f"{group}: {test_name} not found in test suite"
                )

    def test_critic_groups_non_overlapping(self):
        """Critic groups should have distinct test names."""
        all_tests = []
        for tests in REAL_CRITIC_TESTS.values():
            all_tests.extend(tests)
        assert len(all_tests) == len(set(all_tests)), "Critic groups overlap"

    def test_critics_cover_all_categories(self):
        expected = {
            "parsing_check", "filtering_check", "aggregation_check", "context_check",
            "score_check", "dedup_search_check", "time_merge_check", "lineno_format_check",
        }
        assert set(REAL_CRITIC_TESTS.keys()) == expected


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------

class TestPromptTemplates:
    """Verify LLM prompt templates are well-formed."""

    def test_four_prompt_templates(self):
        assert len(REAL_ARM_PROMPTS) == 4

    def test_all_templates_have_source_code_placeholder(self):
        for key, tmpl in REAL_ARM_PROMPTS.items():
            assert "{source_code}" in tmpl, f"{key}: missing {{source_code}}"

    def test_all_templates_have_test_output_placeholder(self):
        for key, tmpl in REAL_ARM_PROMPTS.items():
            assert "{test_output}" in tmpl, f"{key}: missing {{test_output}}"

    def test_all_templates_ask_for_complete_code(self):
        for key, tmpl in REAL_ARM_PROMPTS.items():
            assert "COMPLETE" in tmpl or "ENTIRE" in tmpl, (
                f"{key}: should ask for complete source code"
            )
