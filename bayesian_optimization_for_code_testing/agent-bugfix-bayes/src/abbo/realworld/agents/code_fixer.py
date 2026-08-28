"""Real LLM-based code fixing engine.

Takes a buggy source file, failing test output, and uses an LLM (via Ollama)
to generate a patch that fixes the bug. Applies the patch, runs tests,
and reports the result.
"""

from __future__ import annotations

import difflib
import importlib
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from abbo.realworld.agents.llm_provider import LLMConfig, LLMResponse, call_llm
from abbo.utils.timing import timed


@dataclass
class FixAttempt:
    """Result of a single code-fix attempt."""

    arm: str
    bug_id: str
    prompt: str
    llm_response: str
    extracted_code: str
    diff: str
    tests_passed: bool
    failing_tests_before: list[str]
    failing_tests_after: list[str]
    wall_clock_seconds: float = 0.0
    model: str = ""
    error: str = ""


@dataclass
class FixResult:
    """Aggregated result of fixing a bug across multiple attempts."""

    bug_id: str
    attempts: list[FixAttempt] = field(default_factory=list)
    best_attempt: FixAttempt | None = None
    fixed: bool = False


# ---------------------------------------------------------------------------
# Bug definitions for the toy router app
# ---------------------------------------------------------------------------

ROUTER_CLEAN_SOURCE = '''"""A small request-routing package used as the toy benchmark target."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Route:
    """A single route definition."""
    pattern: str
    handler: str
    priority: int = 0
    category: str = "default"


class Router:
    """Priority-based request router with pattern matching and categories."""

    def __init__(self) -> None:
        self._routes: list[Route] = []

    def add_route(self, route: Route) -> None:
        """Register a new route."""
        self._routes.append(route)

    def match(self, path: str) -> Route | None:
        """Find the best matching route for path.
        Routes match by prefix or regex. Highest priority wins.
        """
        candidates = []
        for route in self._routes:
            if self._matches(route.pattern, path):
                candidates.append(route)
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.priority)

    def _matches(self, pattern: str, path: str) -> bool:
        """Check if pattern matches path via prefix or regex."""
        if path.startswith(pattern):
            return True
        try:
            return bool(re.fullmatch(pattern, path))
        except re.error:
            return False

    def dispatch(self, path: str) -> str | None:
        """Dispatch a request. Returns handler name or None."""
        route = self.match(path)
        return route.handler if route is not None else None

    def get_routes_by_category(self, category: str) -> list[Route]:
        """Return all routes in the given category."""
        return [r for r in self._routes if r.category == category]

    def list_categories(self) -> list[str]:
        """Return sorted unique categories."""
        return sorted({r.category for r in self._routes})
'''

# Three buggy variants
BUGGY_SOURCES = {
    "bug_a": ROUTER_CLEAN_SOURCE.replace(
        "return max(candidates, key=lambda r: r.priority)",
        "return min(candidates, key=lambda r: r.priority)  # BUG: wrong comparison"
    ),
    "bug_b": ROUTER_CLEAN_SOURCE.replace(
        "if path.startswith(pattern):",
        "if path.endswith(pattern):  # BUG: wrong string method"
    ),
    "bug_c": ROUTER_CLEAN_SOURCE.replace(
        "return [r for r in self._routes if r.category == category]",
        "return [r for r in self._routes if r.category != category]  # BUG: wrong comparison"
    ),
}

TEST_FILE_CONTENT = '''"""Tests for the router."""
import sys
sys.path.insert(0, "{src_dir}")
from routing import Route, Router
import pytest

@pytest.fixture
def router():
    r = Router()
    r.add_route(Route("/api/users", "users_handler", priority=10, category="api"))
    r.add_route(Route("/api/users", "users_v2_handler", priority=20, category="api"))
    r.add_route(Route("/api/orders", "orders_handler", priority=10, category="api"))
    r.add_route(Route("/admin", "admin_handler", priority=5, category="admin"))
    r.add_route(Route("/admin/settings", "settings_handler", priority=15, category="admin"))
    r.add_route(Route("/health", "health_handler", priority=1, category="infra"))
    return r

def test_priority_highest_wins(router):
    """Highest priority route should be returned when multiple match."""
    route = router.match("/api/users")
    assert route is not None
    assert route.handler == "users_v2_handler"

def test_priority_order_independence():
    """Priority should work regardless of insertion order."""
    r = Router()
    r.add_route(Route("/x", "low", priority=1))
    r.add_route(Route("/x", "high", priority=100))
    r.add_route(Route("/x", "mid", priority=50))
    assert r.match("/x").handler == "high"

def test_pattern_prefix_match(router):
    """Prefix matching should work for subpaths."""
    route = router.match("/api/users/123")
    assert route is not None
    assert "users" in route.handler

def test_pattern_exact_match(router):
    """Exact path should match."""
    route = router.match("/health")
    assert route is not None
    assert route.handler == "health_handler"

def test_pattern_no_match(router):
    """Unknown path should return None."""
    assert router.match("/unknown/path") is None

def test_pattern_admin_prefix(router):
    """Admin subpaths should match via prefix."""
    route = router.match("/admin/dashboard")
    assert route is not None
    assert route.category == "admin"

def test_category_filter_api(router):
    """Should return only API routes."""
    routes = router.get_routes_by_category("api")
    assert len(routes) == 3
    assert all(r.category == "api" for r in routes)

def test_category_filter_admin(router):
    """Should return only admin routes."""
    routes = router.get_routes_by_category("admin")
    assert len(routes) == 2
    assert all(r.category == "admin" for r in routes)

def test_category_filter_empty(router):
    """Non-existent category should return empty list."""
    assert len(router.get_routes_by_category("nonexistent")) == 0

def test_category_no_cross_contamination():
    """Category filter must not return routes from other categories."""
    r = Router()
    r.add_route(Route("/a", "h1", category="alpha"))
    r.add_route(Route("/b", "h2", category="beta"))
    alpha = r.get_routes_by_category("alpha")
    assert len(alpha) == 1
    assert alpha[0].category == "alpha"
'''


# ---------------------------------------------------------------------------
# Prompt templates for each generator arm
# ---------------------------------------------------------------------------

PROMPT_DIRECT_FIX = """You are a software engineer. Fix the bug in the following Python code.

## Source file: routing.py
```python
{source_code}
```

## Failing test output
```
{test_output}
```

## Instructions
- Find and fix the bug that causes the tests to fail.
- Return ONLY the complete fixed source code of routing.py.
- Do NOT include any explanation, just the code.
- Start your response with ```python and end with ```
"""

PROMPT_LOCALIZED_FIX = """You are a software engineer debugging a Python routing module.

## Source file: routing.py
```python
{source_code}
```

## Failing test output
```
{test_output}
```

## Step 1: Analyze the failures
Look at which tests fail and what they test. Identify the exact line(s) that are wrong.

## Step 2: Fix
Return ONLY the complete fixed source code of routing.py.
- Start your response with ```python and end with ```
- Do NOT modify test logic, only fix routing.py
"""

PROMPT_TEST_GUIDED_FIX = """You are a software engineer. Some tests are failing. Read the test expectations carefully and fix the production code.

## Source file: routing.py
```python
{source_code}
```

## Test file (DO NOT MODIFY):
```python
{test_code}
```

## Failing test output
```
{test_output}
```

## Instructions
Fix routing.py so all tests pass. Return ONLY the complete fixed routing.py.
Start with ```python and end with ```
"""

PROMPT_MINIMAL_DIFF = """You are a software engineer. Fix this bug with the SMALLEST possible change.

## Source file: routing.py
```python
{source_code}
```

## Failing test output
```
{test_output}
```

## Instructions
- Find the ONE line that is wrong and fix it.
- Return ONLY the complete fixed source code.
- Change as few characters as possible.
- Start with ```python and end with ```
"""

ARM_PROMPTS = {
    "g1_direct_fix": PROMPT_DIRECT_FIX,
    "g2_localized_fix": PROMPT_LOCALIZED_FIX,
    "g3_test_guided_fix": PROMPT_TEST_GUIDED_FIX,
    "g4_minimal_diff_fix": PROMPT_MINIMAL_DIFF,
}


# ---------------------------------------------------------------------------
# Core fix logic
# ---------------------------------------------------------------------------

def extract_code_from_response(response: str) -> str:
    """Extract Python code from an LLM response (between ```python and ```)."""
    # Strip <think>...</think> blocks (Qwen3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Strip "Thinking Process:" prefix (Qwen3.5 plain-text thinking)
    think_match = re.search(r"```python", text, re.IGNORECASE)
    if think_match:
        # Only keep from the first code block onwards for matching
        pass  # the regexes below will find it

    # Try ```python ... ``` first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try ``` ... ```
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code blocks, try to find what looks like a Python module
    lines = text.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith(('"""', "from ", "import ", "class ", "def ", "@")):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)

    return text.strip()


def _run_tests_in_dir(workdir: Path) -> tuple[bool, str, list[str]]:
    """Run pytest in workdir. Returns (all_pass, output, failing_test_names)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", str(workdir / "test_routing.py")],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = result.stdout + result.stderr
    all_pass = result.returncode == 0

    # Extract failing test names
    failing = []
    for line in output.splitlines():
        if "FAILED" in line:
            # e.g. "test_routing.py::test_priority_highest_wins FAILED"
            match = re.search(r"(test_\w+)\s+FAILED", line)
            if match:
                failing.append(match.group(1))

    return all_pass, output, failing


def make_diff(original: str, fixed: str, filename: str = "routing.py") -> str:
    """Generate a unified diff between original and fixed source."""
    orig_lines = original.splitlines(keepends=True)
    fixed_lines = fixed.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, fixed_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )
    return "".join(diff)


def fix_bug(
    bug_id: str,
    arm: str = "g1_direct_fix",
    llm_config: LLMConfig | None = None,
) -> FixAttempt:
    """Attempt to fix a toy bug using an LLM.

    Args:
        bug_id: One of "bug_a", "bug_b", "bug_c".
        arm: Generator arm to use.
        llm_config: LLM configuration (defaults to Ollama qwen2.5:7b).

    Returns:
        FixAttempt with the full trace of what happened.
    """
    if llm_config is None:
        llm_config = LLMConfig()

    buggy_source = BUGGY_SOURCES[bug_id]

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "routing.py"
        test_file = workdir / "test_routing.py"

        # Write buggy source
        src_file.write_text(buggy_source)
        test_file.write_text(TEST_FILE_CONTENT.format(src_dir=str(workdir)))

        # Run tests to get failure output
        _, test_output, failing_before = _run_tests_in_dir(workdir)

        # Build prompt
        template = ARM_PROMPTS[arm]
        prompt_kwargs = {
            "source_code": buggy_source,
            "test_output": test_output,
        }
        if arm == "g3_test_guided_fix":
            prompt_kwargs["test_code"] = TEST_FILE_CONTENT.format(src_dir=str(workdir))

        prompt = template.format(**prompt_kwargs)

        # Call LLM
        with timed(f"llm_{arm}") as t:
            llm_resp = call_llm(prompt, llm_config)

        # Extract code from response
        fixed_code = extract_code_from_response(llm_resp.text)

        # Generate diff
        diff = make_diff(buggy_source, fixed_code)

        # Write fixed source and re-run tests
        src_file.write_text(fixed_code)
        tests_passed, retest_output, failing_after = _run_tests_in_dir(workdir)

        return FixAttempt(
            arm=arm,
            bug_id=bug_id,
            prompt=prompt,
            llm_response=llm_resp.text,
            extracted_code=fixed_code,
            diff=diff,
            tests_passed=tests_passed,
            failing_tests_before=failing_before,
            failing_tests_after=failing_after,
            wall_clock_seconds=t.elapsed_seconds,
            model=llm_resp.model,
            error="" if tests_passed else f"{len(failing_after)} tests still failing",
        )


def fix_bug_all_arms(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    arms: list[str] | None = None,
) -> FixResult:
    """Try all generator arms on a bug and return the best result.

    Args:
        bug_id: One of "bug_a", "bug_b", "bug_c".
        llm_config: LLM configuration.
        arms: List of arms to try. Defaults to all 4.

    Returns:
        FixResult with all attempts and the best one.
    """
    if arms is None:
        arms = list(ARM_PROMPTS.keys())

    result = FixResult(bug_id=bug_id)

    for arm in arms:
        attempt = fix_bug(bug_id, arm, llm_config)
        result.attempts.append(attempt)

        if attempt.tests_passed:
            result.fixed = True
            if result.best_attempt is None:
                result.best_attempt = attempt

    # If no arm fully fixed it, pick the one with fewest remaining failures
    if result.best_attempt is None and result.attempts:
        result.best_attempt = min(
            result.attempts, key=lambda a: len(a.failing_tests_after)
        )

    return result
