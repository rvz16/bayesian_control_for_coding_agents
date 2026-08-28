"""Simple Agent — mimics Claude Code-style bug fixing.

This is the baseline: a straightforward retry loop that represents how
current AI coding assistants (Claude Code, Cursor, Copilot) fix bugs:

1. Read the failing test output
2. Generate a fix with the LLM
3. Run ALL tests
4. If tests fail, feed error output back and retry
5. Repeat until fixed or max retries exhausted

Key differences from the Bayesian POMDP agent:
- No belief tracking (doesn't estimate probability of correctness)
- No adaptive strategy (always uses same prompt template)
- No cheap critics (always runs the full test suite)
- No early bail-out (always uses all retries regardless of progress)
- No value-of-information calculations
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import TimeoutExpired

from rich.console import Console

from abbo.realworld.agents.code_fixer import extract_code_from_response, make_diff
from abbo.realworld.agents.llm_provider import LLMConfig, call_llm
from abbo.realworld.agents.real_bugs import (
    BUGGY_SOURCES_REAL,
    LOG_PARSER_CLEAN,
    LOG_PARSER_TEST_CONTENT,
    REAL_ARM_PROMPTS,
    BUG_METADATA,
)

console = Console()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimpleAgentResult:
    """Result of a simple agent fix attempt."""
    policy_name: str = "simple_agent"
    bug_id: str = ""
    fixed: bool = False
    total_cost: float = 0.0
    reward: float = 0.0
    utility: float = 0.0
    attempts: int = 0
    llm_calls: int = 0
    test_runs: int = 0
    wall_clock_seconds: float = 0.0
    final_diff: str = ""
    attempt_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost model (same as Bayesian agent for fair comparison)
# ---------------------------------------------------------------------------

@dataclass
class AgentCostConfig:
    """Costs for both agents (same model for fair comparison)."""
    c_llm_call: float = 10.0       # cost of one LLM generation
    c_full_test: float = 5.0       # cost of running full test suite
    c_critic_test: float = 1.0     # cost of running a subset of tests
    reward: float = 100.0          # reward for successful fix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_run(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run subprocess with timeout protection."""
    try:
        return subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args, returncode=1,
            stdout="TIMEOUT: tests took too long\n", stderr="",
        )


def _run_full_tests(workdir: Path) -> tuple[bool, str, list[str]]:
    """Run full test suite. Returns (all_pass, output, failing_names)."""
    result = _safe_run(
        [sys.executable, "-m", "pytest", "-v", str(workdir / "test_log_parser.py")],
        cwd=workdir,
    )
    output = result.stdout + result.stderr
    failing = []
    for line in output.splitlines():
        if "FAILED" in line:
            m = re.search(r"(test_\w+)", line)
            if m:
                failing.append(m.group(1))
    return result.returncode == 0, output, failing


def _run_critic_tests(workdir: Path, test_names: list[str]) -> tuple[bool, list[str]]:
    """Run a subset of tests (cheap critic). Returns (all_pass, failing_names)."""
    k_expr = " or ".join(test_names)
    result = _safe_run(
        [sys.executable, "-m", "pytest", "-v", "-k", k_expr,
         str(workdir / "test_log_parser.py")],
        cwd=workdir, timeout=15,
    )
    output = result.stdout + result.stderr
    failing = []
    for line in output.splitlines():
        if "FAILED" in line:
            m = re.search(r"(test_\w+)", line)
            if m:
                failing.append(m.group(1))
    return result.returncode == 0, failing


# ---------------------------------------------------------------------------
# Simple Agent: Claude Code-style retry loop
# ---------------------------------------------------------------------------

def run_simple_agent(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: AgentCostConfig | None = None,
    max_retries: int = 3,
    prompt_key: str = "g1_direct_fix",
) -> SimpleAgentResult:
    """Run the simple agent on a bug.

    Mimics Claude Code behavior:
    1. See failing tests
    2. Generate fix
    3. Run ALL tests
    4. If fail, retry with error feedback
    5. Repeat up to max_retries times
    """
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = AgentCostConfig()

    buggy_source = BUGGY_SOURCES_REAL[bug_id]
    result = SimpleAgentResult(bug_id=bug_id)
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "log_parser.py"
        test_file = workdir / "test_log_parser.py"
        test_file.write_text(LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        current_source = buggy_source

        # Get initial test output
        _, test_output, failing = _run_full_tests(workdir)
        result.test_runs += 1
        result.total_cost += cost_config.c_full_test

        console.print(f"  [dim]Initial: {len(failing)} tests failing[/dim]")

        for attempt in range(max_retries):
            result.attempts = attempt + 1

            # Generate fix
            prompt = REAL_ARM_PROMPTS[prompt_key].format(
                source_code=current_source,
                test_output=test_output,
                test_code=LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)),
            )
            console.print(f"  Attempt {attempt + 1}: generating fix ({llm_config.model})...")
            llm_resp = call_llm(prompt, llm_config)
            fixed_code = extract_code_from_response(llm_resp.text)
            result.llm_calls += 1
            result.total_cost += cost_config.c_llm_call

            # Apply the fix
            src_file.write_text(fixed_code)
            current_source = fixed_code

            # Run ALL tests (simple agent always runs full suite)
            all_pass, test_output, failing = _run_full_tests(workdir)
            result.test_runs += 1
            result.total_cost += cost_config.c_full_test

            result.attempt_log.append({
                "attempt": attempt + 1,
                "passed": all_pass,
                "failing": failing,
                "cost_so_far": result.total_cost,
            })

            if all_pass:
                result.fixed = True
                result.reward = cost_config.reward
                result.final_diff = make_diff(buggy_source, fixed_code)
                console.print(f"  [green]Fixed on attempt {attempt + 1}![/green]")
                break
            else:
                if failing:
                    console.print(
                        f"  Attempt {attempt + 1}: [red]{len(failing)} tests failing[/red]"
                    )
                else:
                    console.print(
                        f"  Attempt {attempt + 1}: [red]tests errored (broken code)[/red]"
                    )

    result.wall_clock_seconds = time.perf_counter() - start_time
    result.utility = result.reward - result.total_cost
    return result


# ---------------------------------------------------------------------------
# Simple Agent variants (different fixed strategies, still no adaptation)
# ---------------------------------------------------------------------------

def run_simple_agent_escalating(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: AgentCostConfig | None = None,
    max_retries: int = 3,
) -> SimpleAgentResult:
    """Simple agent with escalating prompts (still no belief/adaptation).

    Uses increasingly detailed prompts on each retry:
    1. Direct fix (minimal prompt)
    2. Localized fix (step-by-step reasoning)
    3. Test-guided fix (includes test code)

    This is a better simple agent — it changes strategy but in a fixed,
    predetermined order, without considering evidence from tests.
    """
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = AgentCostConfig()

    prompt_sequence = ["g1_direct_fix", "g2_localized_fix", "g3_test_guided_fix"]
    buggy_source = BUGGY_SOURCES_REAL[bug_id]
    result = SimpleAgentResult(policy_name="simple_agent_escalating", bug_id=bug_id)
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "log_parser.py"
        test_file = workdir / "test_log_parser.py"
        test_file.write_text(LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        current_source = buggy_source

        _, test_output, failing = _run_full_tests(workdir)
        result.test_runs += 1
        result.total_cost += cost_config.c_full_test

        console.print(f"  [dim]Initial: {len(failing)} tests failing[/dim]")

        for attempt in range(min(max_retries, len(prompt_sequence))):
            result.attempts = attempt + 1
            prompt_key = prompt_sequence[attempt]

            prompt = REAL_ARM_PROMPTS[prompt_key].format(
                source_code=current_source,
                test_output=test_output,
                test_code=LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)),
            )
            console.print(
                f"  Attempt {attempt + 1} ({prompt_key}): generating fix..."
            )
            llm_resp = call_llm(prompt, llm_config)
            fixed_code = extract_code_from_response(llm_resp.text)
            result.llm_calls += 1
            result.total_cost += cost_config.c_llm_call

            src_file.write_text(fixed_code)
            current_source = fixed_code

            all_pass, test_output, failing = _run_full_tests(workdir)
            result.test_runs += 1
            result.total_cost += cost_config.c_full_test

            result.attempt_log.append({
                "attempt": attempt + 1,
                "prompt": prompt_key,
                "passed": all_pass,
                "failing": failing,
                "cost_so_far": result.total_cost,
            })

            if all_pass:
                result.fixed = True
                result.reward = cost_config.reward
                result.final_diff = make_diff(buggy_source, fixed_code)
                console.print(f"  [green]Fixed on attempt {attempt + 1}![/green]")
                break
            else:
                console.print(
                    f"  Attempt {attempt + 1}: [red]{len(failing)} tests still failing[/red]"
                )

    result.wall_clock_seconds = time.perf_counter() - start_time
    result.utility = result.reward - result.total_cost
    return result
