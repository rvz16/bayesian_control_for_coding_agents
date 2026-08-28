"""Live Bayesian orchestration for real code fixing.

Runs the full decision loop:
1. Start with a buggy codebase
2. Bayesian controller decides: run a critic (cheap test), generate a new patch (LLM call), or verify (full test suite)
3. After each action, belief is updated via Bayes rule
4. Controller compares value-of-information of each option and picks the best
5. Stops when it verifies or exhausts budget

Compared against heuristic baselines that follow fixed sequences.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

from rich.console import Console
from rich.table import Table

from abbo.realworld.agents.code_fixer import (
    ARM_PROMPTS,
    BUGGY_SOURCES,
    ROUTER_CLEAN_SOURCE,
    TEST_FILE_CONTENT,
    extract_code_from_response,
    make_diff,
)
from abbo.realworld.agents.hard_bugs import (
    BUGGY_SOURCES_HARD,
    HARD_ARM_PROMPTS,
    HARD_CRITIC_TESTS,
    SCHEDULER_CLEAN_SOURCE,
    SCHEDULER_TEST_CONTENT,
)
from abbo.realworld.agents.llm_provider import LLMConfig, call_llm

console = Console()


def _safe_run_tests(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run subprocess with timeout, returning a failure result on timeout."""
    try:
        return subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except TimeoutExpired:
        return subprocess.CompletedProcess(
            args=args, returncode=1,
            stdout="TIMEOUT: tests took too long (possible infinite loop)\n",
            stderr="",
        )


def _parse_failing(output: str) -> list[str]:
    """Extract failing test names from pytest output."""
    failing = []
    for line in output.splitlines():
        if "FAILED" in line:
            m = re.search(r"(test_\w+)\s+FAILED", line)
            if m:
                failing.append(m.group(1))
    return failing


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CostConfig:
    """Costs for the orchestration problem."""
    c_critic: float = 1.0       # run a subset of tests (cheap)
    c_generator: float = 10.0   # call LLM to generate a patch
    c_verifier: float = 3.0     # run full test suite
    reward: float = 100.0       # reward for full fix


# ---------------------------------------------------------------------------
# Critics: cheap partial test suites
# ---------------------------------------------------------------------------

CRITIC_TESTS = {
    "priority_check": [
        "test_priority_highest_wins",
        "test_priority_order_independence",
    ],
    "pattern_check": [
        "test_pattern_prefix_match",
        "test_pattern_admin_prefix",
    ],
    "category_check": [
        "test_category_filter_api",
        "test_category_no_cross_contamination",
    ],
}


def _run_specific_tests(
    workdir: Path, test_names: list[str]
) -> tuple[bool, list[str]]:
    """Run specific tests in a workdir.  Returns (all_pass, failing_names)."""
    # Build -k expression
    k_expr = " or ".join(test_names)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-k", k_expr,
         str(workdir / "test_routing.py")],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = result.stdout + result.stderr
    failing = []
    for line in output.splitlines():
        if "FAILED" in line:
            m = re.search(r"(test_\w+)\s+FAILED", line)
            if m:
                failing.append(m.group(1))
    return result.returncode == 0, failing


def _run_full_suite(workdir: Path) -> tuple[bool, list[str]]:
    """Run the full test suite.  Returns (all_pass, failing_names)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v",
         str(workdir / "test_routing.py")],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    output = result.stdout + result.stderr
    failing = []
    for line in output.splitlines():
        if "FAILED" in line:
            m = re.search(r"(test_\w+)\s+FAILED", line)
            if m:
                failing.append(m.group(1))
    return result.returncode == 0, failing


def _get_test_output(workdir: Path) -> str:
    """Get pytest output for prompting the LLM."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v",
         str(workdir / "test_routing.py")],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Action log
# ---------------------------------------------------------------------------

@dataclass
class ActionLog:
    """Record of one action taken by the controller."""
    step: int
    action_type: str   # "critic", "generate", "verify"
    label: str
    cost: float
    observation: str   # "pass"/"fail"/diff summary
    belief_before: float
    belief_after: float


@dataclass
class OrchestratedResult:
    """Full result of an orchestrated fix attempt."""
    policy_name: str
    bug_id: str
    actions: list[ActionLog] = field(default_factory=list)
    total_cost: float = 0.0
    reward: float = 0.0
    utility: float = 0.0
    fixed: bool = False
    final_diff: str = ""
    belief_timeline: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bayesian belief updates
# ---------------------------------------------------------------------------

# Calibrated from the toy benchmark:
# P(critic_pass | patch_correct) and P(critic_pass | patch_incorrect)
CRITIC_LIKELIHOODS = {
    "priority_check": {"p_pass_y1": 0.95, "p_pass_y0": 0.70},
    "pattern_check":  {"p_pass_y1": 0.95, "p_pass_y0": 0.70},
    "category_check": {"p_pass_y1": 0.95, "p_pass_y0": 0.65},
}

# P(next_correct | current_incorrect, arm) — calibrated from fix_bug results
GENERATOR_TRANSITIONS = {
    "g1_direct_fix":      {"p01": 0.75, "p10": 0.05},
    "g2_localized_fix":   {"p01": 0.70, "p10": 0.05},
    "g3_test_guided_fix": {"p01": 0.80, "p10": 0.05},
    "g4_minimal_diff_fix":{"p01": 0.70, "p10": 0.05},
}


def bayes_update(belief: float, critic: str, passed: bool) -> float:
    """Update belief after observing a critic result."""
    lk = CRITIC_LIKELIHOODS[critic]
    if passed:
        p_z_y1 = lk["p_pass_y1"]
        p_z_y0 = lk["p_pass_y0"]
    else:
        p_z_y1 = 1 - lk["p_pass_y1"]
        p_z_y0 = 1 - lk["p_pass_y0"]
    denom = p_z_y1 * belief + p_z_y0 * (1 - belief)
    if denom == 0:
        return belief
    return p_z_y1 * belief / denom


def generator_transition(belief: float, arm: str) -> float:
    """Update belief after generating a new patch."""
    t = GENERATOR_TRANSITIONS[arm]
    return belief * (1 - t["p10"]) + (1 - belief) * t["p01"]


# ---------------------------------------------------------------------------
# Value function (simplified DP for the live setting)
# ---------------------------------------------------------------------------

def q_verify(belief: float, costs: CostConfig) -> float:
    """Expected value of verifying now."""
    return costs.reward * belief - costs.c_verifier


def q_critic(belief: float, critic: str, remaining: int, costs: CostConfig) -> float:
    """Expected value of running a critic."""
    if remaining <= 1:
        # After this critic, must verify
        b_pass = bayes_update(belief, critic, True)
        b_fail = bayes_update(belief, critic, False)
        lk = CRITIC_LIKELIHOODS[critic]
        p_pass = lk["p_pass_y1"] * belief + lk["p_pass_y0"] * (1 - belief)
        ev = p_pass * q_verify(b_pass, costs) + (1 - p_pass) * q_verify(b_fail, costs)
        return -costs.c_critic + ev

    # Otherwise, compare with best future value after update
    b_pass = bayes_update(belief, critic, True)
    b_fail = bayes_update(belief, critic, False)
    lk = CRITIC_LIKELIHOODS[critic]
    p_pass = lk["p_pass_y1"] * belief + lk["p_pass_y0"] * (1 - belief)

    v_pass = max(q_verify(b_pass, costs), 0)  # can always verify or do nothing useful
    v_fail = max(q_verify(b_fail, costs), 0)

    return -costs.c_critic + p_pass * v_pass + (1 - p_pass) * v_fail


def q_generate(belief: float, arm: str, remaining: int, costs: CostConfig) -> float:
    """Expected value of generating a new patch."""
    b_new = generator_transition(belief, arm)
    if remaining <= 1:
        return -costs.c_generator + q_verify(b_new, costs)
    return -costs.c_generator + max(q_verify(b_new, costs), 0)


def choose_action(
    belief: float,
    remaining_steps: int,
    remaining_generators: int,
    critics_used: set[str],
    costs: CostConfig,
) -> tuple[str, str, float]:
    """Choose the best action. Returns (action_type, label, q_value)."""
    best = ("verify", "full_suite", q_verify(belief, costs))

    # Critic options (each critic used at most once)
    for critic in CRITIC_LIKELIHOODS:
        if critic in critics_used:
            continue
        q = q_critic(belief, critic, remaining_steps, costs)
        if q > best[2]:
            best = ("critic", critic, q)

    # Generator options
    if remaining_generators > 0:
        for arm in GENERATOR_TRANSITIONS:
            q = q_generate(belief, arm, remaining_steps, costs)
            if q > best[2]:
                best = ("generate", arm, q)

    return best


# ---------------------------------------------------------------------------
# Orchestrated run
# ---------------------------------------------------------------------------

def run_bayes_orchestrated(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
    max_steps: int = 6,
    max_generators: int = 3,
    prior: float = 0.1,
) -> OrchestratedResult:
    """Run the full Bayesian orchestration loop on a toy bug.

    The controller starts with low belief (prior=0.1), generates a first patch,
    then iteratively decides whether to:
    - Run a cheap critic (subset of tests) to update belief
    - Generate a new patch (LLM call) to improve the candidate
    - Verify (run full suite) and collect reward if fixed
    """
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = CostConfig()

    buggy_source = BUGGY_SOURCES[bug_id]
    result = OrchestratedResult(policy_name="bayes", bug_id=bug_id)
    belief = prior
    result.belief_timeline.append(belief)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "routing.py"
        test_file = workdir / "test_routing.py"
        test_file.write_text(TEST_FILE_CONTENT.format(src_dir=str(workdir)))

        # Write initial buggy source
        src_file.write_text(buggy_source)
        current_source = buggy_source
        critics_used: set[str] = set()
        remaining_generators = max_generators
        step = 0

        # First action: always generate (we start with buggy code, belief is low)
        console.print(f"  [dim]Step {step}: Initial state, belief={belief:.3f}[/dim]")

        for step in range(max_steps):
            action_type, label, q_val = choose_action(
                belief, max_steps - step, remaining_generators, critics_used, cost_config
            )

            console.print(
                f"  Step {step}: belief={belief:.3f} -> "
                f"[yellow]{action_type}:{label}[/yellow] (Q={q_val:.1f})"
            )

            belief_before = belief

            if action_type == "verify":
                all_pass, failing = _run_full_suite(workdir)
                obs = "pass" if all_pass else f"fail ({len(failing)} tests)"
                result.actions.append(ActionLog(
                    step=step, action_type="verify", label="full_suite",
                    cost=cost_config.c_verifier, observation=obs,
                    belief_before=belief, belief_after=belief,
                ))
                result.total_cost += cost_config.c_verifier

                if all_pass:
                    result.fixed = True
                    result.reward = cost_config.reward
                    result.final_diff = make_diff(buggy_source, current_source)
                    console.print(f"  [green]VERIFIED: all tests pass![/green]")
                else:
                    console.print(f"  [red]VERIFY FAILED: {failing}[/red]")
                break

            elif action_type == "critic":
                test_names = CRITIC_TESTS[label]
                passed, failing = _run_specific_tests(workdir, test_names)
                belief = bayes_update(belief, label, passed)
                critics_used.add(label)

                obs = "pass" if passed else f"fail: {failing}"
                result.actions.append(ActionLog(
                    step=step, action_type="critic", label=label,
                    cost=cost_config.c_critic, observation=obs,
                    belief_before=belief_before, belief_after=belief,
                ))
                result.total_cost += cost_config.c_critic

                status = "[green]pass[/green]" if passed else "[red]fail[/red]"
                console.print(f"    Result: {status}, belief: {belief_before:.3f} -> {belief:.3f}")

            elif action_type == "generate":
                # Get current test output for the prompt
                test_output = _get_test_output(workdir)
                template = ARM_PROMPTS[label]
                prompt_kwargs = {
                    "source_code": current_source,
                    "test_output": test_output,
                }
                if label == "g3_test_guided_fix":
                    prompt_kwargs["test_code"] = TEST_FILE_CONTENT.format(src_dir=str(workdir))
                prompt = template.format(**prompt_kwargs)

                console.print(f"    Calling LLM ({llm_config.model})...")
                llm_resp = call_llm(prompt, llm_config)
                fixed_code = extract_code_from_response(llm_resp.text)

                # Apply the new code
                src_file.write_text(fixed_code)
                current_source = fixed_code
                remaining_generators -= 1

                # Update belief via transition model
                belief = generator_transition(belief, label)

                diff_summary = make_diff(buggy_source, fixed_code)
                n_diff_lines = len([l for l in diff_summary.splitlines()
                                   if l.startswith("+") or l.startswith("-")])
                result.actions.append(ActionLog(
                    step=step, action_type="generate", label=label,
                    cost=cost_config.c_generator,
                    observation=f"generated ({n_diff_lines} diff lines)",
                    belief_before=belief_before, belief_after=belief,
                ))
                result.total_cost += cost_config.c_generator
                console.print(
                    f"    Generated patch, belief: {belief_before:.3f} -> {belief:.3f}"
                )

            result.belief_timeline.append(belief)

        # If we never verified, do it now
        if not any(a.action_type == "verify" for a in result.actions):
            all_pass, failing = _run_full_suite(workdir)
            result.total_cost += cost_config.c_verifier
            if all_pass:
                result.fixed = True
                result.reward = cost_config.reward
                result.final_diff = make_diff(buggy_source, current_source)

    result.utility = result.reward - result.total_cost
    return result


# ---------------------------------------------------------------------------
# Heuristic baselines
# ---------------------------------------------------------------------------

def run_heuristic_one_shot(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
) -> OrchestratedResult:
    """H1: Generate once, verify immediately."""
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = CostConfig()

    buggy_source = BUGGY_SOURCES[bug_id]
    result = OrchestratedResult(policy_name="h1_one_shot", bug_id=bug_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "routing.py"
        test_file = workdir / "test_routing.py"
        test_file.write_text(TEST_FILE_CONTENT.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)

        # Generate
        test_output = _get_test_output(workdir)
        prompt = ARM_PROMPTS["g1_direct_fix"].format(
            source_code=buggy_source, test_output=test_output
        )
        console.print(f"  [yellow]generate:g1_direct_fix[/yellow] (calling LLM...)")
        llm_resp = call_llm(prompt, llm_config)
        fixed = extract_code_from_response(llm_resp.text)
        src_file.write_text(fixed)
        result.total_cost += cost_config.c_generator
        result.actions.append(ActionLog(
            step=0, action_type="generate", label="g1_direct_fix",
            cost=cost_config.c_generator, observation="generated",
            belief_before=0.1, belief_after=0.1,
        ))

        # Verify
        all_pass, failing = _run_full_suite(workdir)
        result.total_cost += cost_config.c_verifier
        result.actions.append(ActionLog(
            step=1, action_type="verify", label="full_suite",
            cost=cost_config.c_verifier,
            observation="pass" if all_pass else f"fail ({len(failing)})",
            belief_before=0.1, belief_after=float(all_pass),
        ))
        if all_pass:
            result.fixed = True
            result.reward = cost_config.reward
            result.final_diff = make_diff(buggy_source, fixed)

    result.utility = result.reward - result.total_cost
    return result


def run_heuristic_fixed_workflow(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
) -> OrchestratedResult:
    """H2: Run all critics first, then generate, then verify."""
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = CostConfig()

    buggy_source = BUGGY_SOURCES[bug_id]
    result = OrchestratedResult(policy_name="h2_fixed_workflow", bug_id=bug_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "routing.py"
        test_file = workdir / "test_routing.py"
        test_file.write_text(TEST_FILE_CONTENT.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        step = 0

        # Run all critics on buggy code first
        for critic in CRITIC_TESTS:
            passed, failing = _run_specific_tests(workdir, CRITIC_TESTS[critic])
            result.total_cost += cost_config.c_critic
            result.actions.append(ActionLog(
                step=step, action_type="critic", label=critic,
                cost=cost_config.c_critic,
                observation="pass" if passed else f"fail: {failing}",
                belief_before=0.1, belief_after=0.1,
            ))
            console.print(
                f"  [yellow]critic:{critic}[/yellow] -> "
                f"{'[green]pass[/green]' if passed else '[red]fail[/red]'}"
            )
            step += 1

        # Generate
        test_output = _get_test_output(workdir)
        prompt = ARM_PROMPTS["g3_test_guided_fix"].format(
            source_code=buggy_source, test_output=test_output,
            test_code=TEST_FILE_CONTENT.format(src_dir=str(workdir)),
        )
        console.print(f"  [yellow]generate:g3_test_guided_fix[/yellow] (calling LLM...)")
        llm_resp = call_llm(prompt, llm_config)
        fixed = extract_code_from_response(llm_resp.text)
        src_file.write_text(fixed)
        result.total_cost += cost_config.c_generator
        result.actions.append(ActionLog(
            step=step, action_type="generate", label="g3_test_guided_fix",
            cost=cost_config.c_generator, observation="generated",
            belief_before=0.1, belief_after=0.1,
        ))
        step += 1

        # Verify
        all_pass, failing = _run_full_suite(workdir)
        result.total_cost += cost_config.c_verifier
        result.actions.append(ActionLog(
            step=step, action_type="verify", label="full_suite",
            cost=cost_config.c_verifier,
            observation="pass" if all_pass else f"fail ({len(failing)})",
            belief_before=0.1, belief_after=float(all_pass),
        ))
        if all_pass:
            result.fixed = True
            result.reward = cost_config.reward
            result.final_diff = make_diff(buggy_source, fixed)

    result.utility = result.reward - result.total_cost
    return result


def run_heuristic_generate_twice(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
) -> OrchestratedResult:
    """H3: Generate, check, if fail generate again, verify."""
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = CostConfig()

    buggy_source = BUGGY_SOURCES[bug_id]
    result = OrchestratedResult(policy_name="h3_generate_twice", bug_id=bug_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "routing.py"
        test_file = workdir / "test_routing.py"
        test_file.write_text(TEST_FILE_CONTENT.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        current_source = buggy_source

        for attempt, arm in enumerate(["g1_direct_fix", "g3_test_guided_fix"]):
            test_output = _get_test_output(workdir)
            prompt_kwargs = {"source_code": current_source, "test_output": test_output}
            if arm == "g3_test_guided_fix":
                prompt_kwargs["test_code"] = TEST_FILE_CONTENT.format(src_dir=str(workdir))
            prompt = ARM_PROMPTS[arm].format(**prompt_kwargs)

            console.print(f"  [yellow]generate:{arm}[/yellow] (calling LLM...)")
            llm_resp = call_llm(prompt, llm_config)
            fixed = extract_code_from_response(llm_resp.text)
            src_file.write_text(fixed)
            current_source = fixed
            result.total_cost += cost_config.c_generator
            result.actions.append(ActionLog(
                step=attempt * 2, action_type="generate", label=arm,
                cost=cost_config.c_generator, observation="generated",
                belief_before=0.1, belief_after=0.1,
            ))

            # Quick check after first attempt
            if attempt == 0:
                passed, failing = _run_specific_tests(workdir, CRITIC_TESTS["priority_check"])
                result.total_cost += cost_config.c_critic
                console.print(
                    f"  [yellow]critic:priority_check[/yellow] -> "
                    f"{'[green]pass[/green]' if passed else '[red]fail[/red]'}"
                )
                if passed:
                    break  # looks good, skip second generation

        # Verify
        all_pass, failing = _run_full_suite(workdir)
        result.total_cost += cost_config.c_verifier
        result.actions.append(ActionLog(
            step=len(result.actions), action_type="verify", label="full_suite",
            cost=cost_config.c_verifier,
            observation="pass" if all_pass else f"fail ({len(failing)})",
            belief_before=0.1, belief_after=float(all_pass),
        ))
        if all_pass:
            result.fixed = True
            result.reward = cost_config.reward
            result.final_diff = make_diff(buggy_source, current_source)

    result.utility = result.reward - result.total_cost
    return result


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def run_comparison(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
) -> list[OrchestratedResult]:
    """Run Bayes + all heuristics on one bug and return results."""
    results = []

    console.print(f"\n[bold cyan]--- Bayes Controller ---[/bold cyan]")
    results.append(run_bayes_orchestrated(bug_id, llm_config, cost_config))

    console.print(f"\n[bold cyan]--- H1: One Shot ---[/bold cyan]")
    results.append(run_heuristic_one_shot(bug_id, llm_config, cost_config))

    console.print(f"\n[bold cyan]--- H2: Fixed Workflow ---[/bold cyan]")
    results.append(run_heuristic_fixed_workflow(bug_id, llm_config, cost_config))

    console.print(f"\n[bold cyan]--- H3: Generate Twice ---[/bold cyan]")
    results.append(run_heuristic_generate_twice(bug_id, llm_config, cost_config))

    return results


# ---------------------------------------------------------------------------
# Expected-value analysis (no LLM calls)
# ---------------------------------------------------------------------------

def compute_expected_utilities(
    p_fix: float = 0.5,
    cost_config: CostConfig | None = None,
) -> dict[str, dict[str, float]]:
    """Compute expected utility for each policy given a fix probability.

    This shows the *theoretical* advantage of the Bayesian controller
    across a range of fix probabilities. No LLM calls needed.

    Args:
        p_fix: Probability that a single LLM generation fixes the bug.
        cost_config: Cost parameters.

    Returns:
        Dict mapping policy name to {expected_utility, expected_cost, p_success}.
    """
    if cost_config is None:
        cost_config = CostConfig()

    R = cost_config.reward
    cg = cost_config.c_generator
    cv = cost_config.c_verifier
    cc = cost_config.c_critic

    # H1: One shot — generate once, verify
    h1_cost = cg + cv
    h1_p = p_fix
    h1_eu = R * h1_p - h1_cost

    # H2: Fixed workflow — 3 critics + generate + verify
    h2_cost = 3 * cc + cg + cv
    h2_p = p_fix  # critics don't change the fix
    h2_eu = R * h2_p - h2_cost

    # H3: Generate twice — generate, quick check, maybe generate again, verify
    # If first gen succeeds (p_fix), we pay cg + cc + cv
    # If first fails (1-p_fix), quick check fails, we generate again:
    #   second gen succeeds with p_fix, total: 2*cg + cc + cv
    h3_p = p_fix + (1 - p_fix) * p_fix  # = p_fix * (2 - p_fix)
    h3_cost_success_first = cg + cc + cv
    h3_cost_need_second = 2 * cg + cc + cv
    h3_cost = p_fix * h3_cost_success_first + (1 - p_fix) * h3_cost_need_second
    h3_eu = R * h3_p - h3_cost

    # Bayes: generate, update belief, decide adaptively
    # With prior=0.1, after first gen belief -> b1 = 0.1*(1-0.05) + 0.9*p_fix
    # Then the controller decides based on b1:
    #   If b1 high enough -> verify directly (save cost)
    #   If b1 moderate -> run a cheap critic first
    #   If b1 low -> generate again
    p01 = p_fix  # approximate: actual p01 from calibration
    p10 = 0.05
    b0 = 0.1
    b1 = b0 * (1 - p10) + (1 - b0) * p01

    # Decision at b1:
    q_v = R * b1 - cv
    # Best critic value (approximate): run one critic, then verify
    lk_pass = 0.95 * b1 + 0.70 * (1 - b1)
    b_pass = 0.95 * b1 / lk_pass
    b_fail = 0.05 * b1 / (0.05 * b1 + 0.30 * (1 - b1))
    q_c = -cc + lk_pass * max(R * b_pass - cv, 0) + (1 - lk_pass) * max(R * b_fail - cv, 0)
    # Generate again value
    b2 = b1 * (1 - p10) + (1 - b1) * p01
    q_g = -cg + max(R * b2 - cv, 0)

    bayes_q = max(q_v, q_c, q_g)
    if q_v >= q_c and q_v >= q_g:
        bayes_cost = cg + cv  # generate then verify
        bayes_p = b1
        bayes_path = "gen→verify"
    elif q_c >= q_g:
        bayes_cost = cg + cc + cv  # generate, critic, verify
        bayes_p = lk_pass * b_pass + (1 - lk_pass) * b_fail
        bayes_path = "gen→critic→verify"
    else:
        bayes_cost = 2 * cg + cv  # generate twice then verify
        bayes_p = b2
        bayes_path = "gen→gen→verify"

    bayes_eu = R * bayes_p - bayes_cost

    return {
        "bayes": {
            "expected_utility": bayes_eu,
            "expected_cost": bayes_cost,
            "p_success": bayes_p,
            "path": bayes_path,
        },
        "h1_one_shot": {
            "expected_utility": h1_eu,
            "expected_cost": h1_cost,
            "p_success": h1_p,
        },
        "h2_fixed_workflow": {
            "expected_utility": h2_eu,
            "expected_cost": h2_cost,
            "p_success": h2_p,
        },
        "h3_generate_twice": {
            "expected_utility": h3_eu,
            "expected_cost": h3_cost,
            "p_success": h3_p,
        },
    }


# ---------------------------------------------------------------------------
# Hard-bug orchestration (same logic, different source/tests)
# ---------------------------------------------------------------------------

def _get_bug_config(bug_id: str) -> tuple[str, str, dict, dict, str, str]:
    """Return (buggy_source, test_content, arm_prompts, critic_tests, src_name, test_name)."""
    if bug_id in BUGGY_SOURCES:
        return (
            BUGGY_SOURCES[bug_id],
            TEST_FILE_CONTENT,
            ARM_PROMPTS,
            CRITIC_TESTS,
            "routing.py",
            "test_routing.py",
        )
    elif bug_id in BUGGY_SOURCES_HARD:
        return (
            BUGGY_SOURCES_HARD[bug_id],
            SCHEDULER_TEST_CONTENT,
            HARD_ARM_PROMPTS,
            HARD_CRITIC_TESTS,
            "scheduler.py",
            "test_scheduler.py",
        )
    else:
        raise ValueError(f"Unknown bug_id: {bug_id}. Available: "
                         f"{list(BUGGY_SOURCES) + list(BUGGY_SOURCES_HARD)}")


def run_bayes_hard(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
    max_steps: int = 8,
    max_generators: int = 4,
    prior: float = 0.05,
) -> OrchestratedResult:
    """Run Bayesian orchestration on any bug (easy or hard).

    Uses a lower prior (0.05) for hard bugs, and the controller will use
    critics to update belief before deciding whether to generate or verify.
    """
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = CostConfig()

    buggy_source, test_content, arm_prompts, critic_tests, src_name, test_name = (
        _get_bug_config(bug_id)
    )
    result = OrchestratedResult(policy_name="bayes", bug_id=bug_id)
    belief = prior
    result.belief_timeline.append(belief)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / src_name
        test_file = workdir / test_name
        test_file.write_text(test_content.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        current_source = buggy_source
        critics_used: set[str] = set()
        remaining_generators = max_generators

        # Use the hard critic tests for choose_action
        available_critics = set(critic_tests.keys())

        console.print(f"  [dim]Initial state, belief={belief:.3f}[/dim]")

        for step in range(max_steps):
            # choose_action uses module-level CRITIC_LIKELIHOODS — we use the
            # same Bayesian machinery, just different test subsets to run
            action_type, label, q_val = choose_action(
                belief, max_steps - step, remaining_generators, critics_used, cost_config
            )

            # Map critic labels to the correct test set
            if action_type == "critic" and label not in critic_tests:
                # Fall back to first unused hard critic
                unused = available_critics - critics_used
                if unused:
                    label = sorted(unused)[0]
                else:
                    action_type, label, q_val = "verify", "full_suite", q_verify(belief, cost_config)

            # Bail out if all Q-values are negative (spending more than expected gain)
            if q_val < 0 and step > 0:
                console.print(
                    f"  Step {step}: belief={belief:.3f} -> "
                    f"[dim]BAIL OUT (best Q={q_val:.1f} < 0, not worth spending more)[/dim]"
                )
                break

            console.print(
                f"  Step {step}: belief={belief:.3f} -> "
                f"[yellow]{action_type}:{label}[/yellow] (Q={q_val:.1f})"
            )
            belief_before = belief

            if action_type == "verify":
                result_sub = _safe_run_tests(
                    [sys.executable, "-m", "pytest", "-v", str(workdir / test_name)],
                    cwd=workdir,
                )
                output = result_sub.stdout + result_sub.stderr
                all_pass = result_sub.returncode == 0
                failing = _parse_failing(output)

                obs = "pass" if all_pass else f"fail ({len(failing)} tests)"
                result.actions.append(ActionLog(
                    step=step, action_type="verify", label="full_suite",
                    cost=cost_config.c_verifier, observation=obs,
                    belief_before=belief, belief_after=belief,
                ))
                result.total_cost += cost_config.c_verifier

                if all_pass:
                    result.fixed = True
                    result.reward = cost_config.reward
                    result.final_diff = make_diff(buggy_source, current_source)
                    console.print(f"  [green]VERIFIED: all tests pass![/green]")
                else:
                    console.print(f"  [red]VERIFY FAILED: {failing}[/red]")
                break

            elif action_type == "critic":
                test_names = critic_tests.get(label, list(critic_tests.values())[0])
                k_expr = " or ".join(test_names)
                result_sub = _safe_run_tests(
                    [sys.executable, "-m", "pytest", "-v", "-k", k_expr,
                     str(workdir / test_name)],
                    cwd=workdir, timeout=15,
                )
                passed = result_sub.returncode == 0
                # Map to a known critic key for Bayes update
                critic_key = label if label in CRITIC_LIKELIHOODS else list(CRITIC_LIKELIHOODS.keys())[0]
                belief = bayes_update(belief, critic_key, passed)
                critics_used.add(label)

                obs = "pass" if passed else "fail"
                result.actions.append(ActionLog(
                    step=step, action_type="critic", label=label,
                    cost=cost_config.c_critic, observation=obs,
                    belief_before=belief_before, belief_after=belief,
                ))
                result.total_cost += cost_config.c_critic

                status = "[green]pass[/green]" if passed else "[red]fail[/red]"
                console.print(f"    Result: {status}, belief: {belief_before:.3f} -> {belief:.3f}")

            elif action_type == "generate":
                # Get test output for prompt
                result_sub = _safe_run_tests(
                    [sys.executable, "-m", "pytest", "-v", str(workdir / test_name)],
                    cwd=workdir,
                )
                test_output = result_sub.stdout + result_sub.stderr

                # Pick a generator arm from the right prompt set
                gen_key = label if label in arm_prompts else list(arm_prompts.keys())[0]
                template = arm_prompts[gen_key]
                prompt_kwargs = {
                    "source_code": current_source,
                    "test_output": test_output,
                }
                if "test_code" in template:
                    prompt_kwargs["test_code"] = test_content.format(src_dir=str(workdir))
                try:
                    prompt = template.format(**prompt_kwargs)
                except KeyError:
                    prompt = template.format(
                        source_code=current_source, test_output=test_output
                    )

                console.print(f"    Calling LLM ({llm_config.model})...")
                llm_resp = call_llm(prompt, llm_config)
                fixed_code = extract_code_from_response(llm_resp.text)
                src_file.write_text(fixed_code)
                current_source = fixed_code
                remaining_generators -= 1

                # Run a free quick sanity check after generation to ground belief
                # This is the key Bayesian insight: use actual test feedback
                unused_critics = list(available_critics - critics_used)
                if unused_critics:
                    check_critic = unused_critics[0]
                    check_tests = critic_tests[check_critic]
                    k_expr = " or ".join(check_tests)
                    check_sub = _safe_run_tests(
                        [sys.executable, "-m", "pytest", "-v", "-k", k_expr,
                         str(workdir / test_name)],
                        cwd=workdir, timeout=15,
                    )
                    critic_passed = check_sub.returncode == 0
                    critic_key = check_critic if check_critic in CRITIC_LIKELIHOODS else list(CRITIC_LIKELIHOODS.keys())[0]
                    # Start from transition model estimate, then update with critic
                    gen_trans_key = label if label in GENERATOR_TRANSITIONS else list(GENERATOR_TRANSITIONS.keys())[0]
                    belief = generator_transition(belief, gen_trans_key)
                    belief = bayes_update(belief, critic_key, critic_passed)
                    critics_used.add(check_critic)
                    result.total_cost += cost_config.c_critic  # cheap check
                    status = "[green]pass[/green]" if critic_passed else "[red]fail[/red]"
                    console.print(f"    Quick check ({check_critic}): {status}")
                else:
                    gen_trans_key = label if label in GENERATOR_TRANSITIONS else list(GENERATOR_TRANSITIONS.keys())[0]
                    belief = generator_transition(belief, gen_trans_key)

                diff_summary = make_diff(buggy_source, fixed_code)
                n_diff_lines = len([l for l in diff_summary.splitlines()
                                   if l.startswith("+") or l.startswith("-")])
                result.actions.append(ActionLog(
                    step=step, action_type="generate", label=gen_key,
                    cost=cost_config.c_generator,
                    observation=f"generated ({n_diff_lines} diff lines)",
                    belief_before=belief_before, belief_after=belief,
                ))
                result.total_cost += cost_config.c_generator
                console.print(
                    f"    Generated patch, belief: {belief_before:.3f} -> {belief:.3f}"
                )

            result.belief_timeline.append(belief)

        # If we never verified, do it now
        if not any(a.action_type == "verify" for a in result.actions):
            result_sub = _safe_run_tests(
                [sys.executable, "-m", "pytest", "-v", str(workdir / test_name)],
                cwd=workdir,
            )
            all_pass = result_sub.returncode == 0
            result.total_cost += cost_config.c_verifier
            if all_pass:
                result.fixed = True
                result.reward = cost_config.reward
                result.final_diff = make_diff(buggy_source, current_source)

    result.utility = result.reward - result.total_cost
    return result


def run_heuristic_one_shot_hard(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
) -> OrchestratedResult:
    """H1 on any bug: generate once, verify."""
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = CostConfig()

    buggy_source, test_content, arm_prompts, critic_tests, src_name, test_name = (
        _get_bug_config(bug_id)
    )
    result = OrchestratedResult(policy_name="h1_one_shot", bug_id=bug_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / src_name
        test_file = workdir / test_name
        test_file.write_text(test_content.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)

        # Get test output and generate
        result_sub = _safe_run_tests(
            [sys.executable, "-m", "pytest", "-v", str(workdir / test_name)],
            cwd=workdir,
        )
        test_output = result_sub.stdout + result_sub.stderr
        gen_key = list(arm_prompts.keys())[0]
        prompt = arm_prompts[gen_key].format(
            source_code=buggy_source, test_output=test_output
        )
        console.print(f"  [yellow]generate:{gen_key}[/yellow] (calling LLM...)")
        llm_resp = call_llm(prompt, llm_config)
        fixed = extract_code_from_response(llm_resp.text)
        src_file.write_text(fixed)
        result.total_cost += cost_config.c_generator
        result.actions.append(ActionLog(
            step=0, action_type="generate", label=gen_key,
            cost=cost_config.c_generator, observation="generated",
            belief_before=0.1, belief_after=0.1,
        ))

        # Verify
        result_sub = _safe_run_tests(
            [sys.executable, "-m", "pytest", "-v", str(workdir / test_name)],
            cwd=workdir,
        )
        all_pass = result_sub.returncode == 0
        failing = _parse_failing(result_sub.stdout + result_sub.stderr)
        result.total_cost += cost_config.c_verifier
        result.actions.append(ActionLog(
            step=1, action_type="verify", label="full_suite",
            cost=cost_config.c_verifier,
            observation="pass" if all_pass else f"fail ({len(failing)})",
            belief_before=0.1, belief_after=float(all_pass),
        ))
        if all_pass:
            result.fixed = True
            result.reward = cost_config.reward
            result.final_diff = make_diff(buggy_source, fixed)

    result.utility = result.reward - result.total_cost
    return result


def run_heuristic_generate_twice_hard(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
) -> OrchestratedResult:
    """H3 on any bug: generate, check, maybe generate again, verify."""
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = CostConfig()

    buggy_source, test_content, arm_prompts, critic_tests, src_name, test_name = (
        _get_bug_config(bug_id)
    )
    result = OrchestratedResult(policy_name="h3_generate_twice", bug_id=bug_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / src_name
        test_file = workdir / test_name
        test_file.write_text(test_content.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        current_source = buggy_source

        arm_keys = list(arm_prompts.keys())
        first_arm = arm_keys[0]
        second_arm = arm_keys[2] if len(arm_keys) > 2 else arm_keys[-1]
        first_critic = list(critic_tests.keys())[0]

        for attempt, arm in enumerate([first_arm, second_arm]):
            result_sub = _safe_run_tests(
                [sys.executable, "-m", "pytest", "-v", str(workdir / test_name)],
                cwd=workdir,
            )
            test_output = result_sub.stdout + result_sub.stderr
            prompt_kwargs = {"source_code": current_source, "test_output": test_output}
            if "test_code" in arm_prompts[arm]:
                prompt_kwargs["test_code"] = test_content.format(src_dir=str(workdir))
            try:
                prompt = arm_prompts[arm].format(**prompt_kwargs)
            except KeyError:
                prompt = arm_prompts[arm].format(
                    source_code=current_source, test_output=test_output
                )

            console.print(f"  [yellow]generate:{arm}[/yellow] (calling LLM...)")
            llm_resp = call_llm(prompt, llm_config)
            fixed = extract_code_from_response(llm_resp.text)
            src_file.write_text(fixed)
            current_source = fixed
            result.total_cost += cost_config.c_generator
            result.actions.append(ActionLog(
                step=attempt * 2, action_type="generate", label=arm,
                cost=cost_config.c_generator, observation="generated",
                belief_before=0.1, belief_after=0.1,
            ))

            # Quick check after first attempt
            if attempt == 0:
                test_names = critic_tests[first_critic]
                k_expr = " or ".join(test_names)
                result_sub = _safe_run_tests(
                    [sys.executable, "-m", "pytest", "-v", "-k", k_expr,
                     str(workdir / test_name)],
                    cwd=workdir, timeout=15,
                )
                passed = result_sub.returncode == 0
                result.total_cost += cost_config.c_critic
                console.print(
                    f"  [yellow]critic:{first_critic}[/yellow] -> "
                    f"{'[green]pass[/green]' if passed else '[red]fail[/red]'}"
                )
                if passed:
                    break

        # Verify
        result_sub = _safe_run_tests(
            [sys.executable, "-m", "pytest", "-v", str(workdir / test_name)],
            cwd=workdir,
        )
        all_pass = result_sub.returncode == 0
        failing = _parse_failing(result_sub.stdout + result_sub.stderr)
        result.total_cost += cost_config.c_verifier
        result.actions.append(ActionLog(
            step=len(result.actions), action_type="verify", label="full_suite",
            cost=cost_config.c_verifier,
            observation="pass" if all_pass else f"fail ({len(failing)})",
            belief_before=0.1, belief_after=float(all_pass),
        ))
        if all_pass:
            result.fixed = True
            result.reward = cost_config.reward
            result.final_diff = make_diff(buggy_source, current_source)

    result.utility = result.reward - result.total_cost
    return result


def run_comparison_hard(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: CostConfig | None = None,
) -> list[OrchestratedResult]:
    """Run Bayes + heuristics on any bug (easy or hard)."""
    results = []

    console.print(f"\n[bold cyan]--- Bayes Controller ---[/bold cyan]")
    results.append(run_bayes_hard(bug_id, llm_config, cost_config))

    console.print(f"\n[bold cyan]--- H1: One Shot ---[/bold cyan]")
    results.append(run_heuristic_one_shot_hard(bug_id, llm_config, cost_config))

    console.print(f"\n[bold cyan]--- H3: Generate Twice ---[/bold cyan]")
    results.append(run_heuristic_generate_twice_hard(bug_id, llm_config, cost_config))

    return results
