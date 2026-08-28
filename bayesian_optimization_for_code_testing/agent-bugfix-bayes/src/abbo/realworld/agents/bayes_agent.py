"""Bayesian POMDP Agent — adaptive bug fixing with belief tracking.

Implements sequential decision-making under uncertainty for code fixing
using full Bellman DP recursion (not greedy one-step lookahead).

Core equation:  V(b, s) = max_a [ -C_a + E[ V(b', s') | a, b ] ]
Terminal:        V_term = max( R*b - C_verify, 0 )   (verify or bail)

POMDP formulation:
- State: Y in {0, 1} — is the current patch correct?
- Belief: b = P(Y=1 | history)
- Actions: {generate(arm), critic(subset), verify(full), bail_out}
- Observations: test pass/fail results
- Transitions: generator changes the patch (deterministic belief update)
- Reward: +100 for verified fix, costs per action

The DP planner pre-computes a value table over discretized belief (101 pts)
and resource budgets (generators left, critics used, verifications left).
At runtime, the agent looks up the optimal action in ~O(1).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from collections import namedtuple
from dataclasses import dataclass, field
from functools import lru_cache
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
    REAL_CRITIC_TESTS,
    BUG_METADATA,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig, _safe_run, _run_full_tests, _run_critic_tests

console = Console()


@dataclass
class BayesAgentResult:
    """Result of a Bayesian POMDP agent fix attempt."""
    policy_name: str = "bayes_pomdp"
    bug_id: str = ""
    fixed: bool = False
    total_cost: float = 0.0
    reward: float = 0.0
    utility: float = 0.0
    steps: int = 0
    llm_calls: int = 0
    test_runs: int = 0
    critic_runs: int = 0
    wall_clock_seconds: float = 0.0
    final_diff: str = ""
    belief_timeline: list[float] = field(default_factory=list)
    action_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Bayesian belief model
# ---------------------------------------------------------------------------

# P(critic_pass | Y=1) and P(critic_pass | Y=0) for each test subset
CRITIC_LIKELIHOODS = {
    "parsing_check":       {"p_pass_y1": 0.95, "p_pass_y0": 0.85},
    "filtering_check":     {"p_pass_y1": 0.95, "p_pass_y0": 0.50},
    "aggregation_check":   {"p_pass_y1": 0.95, "p_pass_y0": 0.55},
    "context_check":       {"p_pass_y1": 0.95, "p_pass_y0": 0.60},
    "score_check":         {"p_pass_y1": 0.95, "p_pass_y0": 0.55},
    "dedup_search_check":  {"p_pass_y1": 0.95, "p_pass_y0": 0.65},
    "time_merge_check":    {"p_pass_y1": 0.95, "p_pass_y0": 0.60},
    "lineno_format_check": {"p_pass_y1": 0.95, "p_pass_y0": 0.50},
}

# P(Y'=1 | Y=0, arm) for each generator arm — conservative estimates
GENERATOR_TRANSITIONS = {
    "g1_direct_fix":      {"p01": 0.50, "p10": 0.05},
    "g2_localized_fix":   {"p01": 0.45, "p10": 0.05},
    "g3_test_guided_fix": {"p01": 0.40, "p10": 0.08},
    "g4_error_focused_fix": {"p01": 0.35, "p10": 0.08},
}

# Ordered arm sequence: start with simplest (most effective for small LLMs),
# escalate to more complex prompts only if simpler ones fail
ARM_SEQUENCE = ["g1_direct_fix", "g2_localized_fix", "g3_test_guided_fix"]

# Ordered critic sequence: most informative first
CRITIC_SEQUENCE = [
    "filtering_check", "aggregation_check", "context_check", "parsing_check",
    "score_check", "lineno_format_check", "time_merge_check", "dedup_search_check",
]


def bayes_update(
    belief: float,
    critic: str,
    passed: bool,
    likelihoods: dict | None = None,
) -> float:
    """Update belief P(Y=1) after observing a critic result.

    Args:
        likelihoods: optional override table {critic_name: {p_pass_y1, p_pass_y0}}.
            When None, uses the module-level CRITIC_LIKELIHOODS (hand-tuned).
            Pass in a calibrated table to use empirical-Bayes estimates.
    """
    lk_table = likelihoods if likelihoods is not None else CRITIC_LIKELIHOODS
    lk = lk_table[critic]
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


def q_verify(belief: float, costs: AgentCostConfig) -> float:
    """Expected utility of verifying now."""
    return costs.reward * belief - costs.c_full_test


# ---------------------------------------------------------------------------
# Bellman DP Planner
# ---------------------------------------------------------------------------

BELIEF_GRID_SIZE = 101  # 0.00, 0.01, ..., 1.00

# Ordered lists for indexing into the DP state
_GENERATOR_ARMS = list(GENERATOR_TRANSITIONS.keys())  # 4 arms
_CRITIC_NAMES = list(CRITIC_LIKELIHOODS.keys())        # 4 critics


def discretize(b: float) -> int:
    """Snap a belief to the nearest grid index."""
    return max(0, min(BELIEF_GRID_SIZE - 1, round(b * (BELIEF_GRID_SIZE - 1))))


def grid_belief(idx: int) -> float:
    """Convert grid index back to belief value."""
    return idx / (BELIEF_GRID_SIZE - 1)


# DPState: hashable state for memoization
# belief_idx: int (0..100)
# gen_left: int (0..max_generators)
# crit_used: frozenset of critic names already used
# ver_left: int (0..max_verifications)
DPState = namedtuple("DPState", ["belief_idx", "gen_left", "crit_used", "ver_left"])


class DPPlanner:
    """Full Bellman DP planner for the POMDP bug-fixing problem.

    Pre-computes optimal value and action for every reachable state.
    State = (discretized belief, generators_left, critics_used, verifies_left).

    Usage:
        planner = DPPlanner(costs)
        planner.solve()
        action, value = planner.choose_action(belief=0.3, gen_left=3,
                                               crit_used=frozenset(), ver_left=2)
    """

    def __init__(
        self,
        costs: AgentCostConfig,
        max_generators: int = 3,
        max_verifications: int = 2,
        critic_likelihoods: dict | None = None,
        transition_kernel: dict | None = None,
    ):
        """Args:
            transition_kernel: optional measured kernel that overrides the
                arm-specific GENERATOR_TRANSITIONS. Schema:
                    {"p_fix_broken": float, "p_break_correct": float}
                When provided, all `generate:<arm>` actions use the same
                transition (the arm name is kept for action-log readability
                but the per-arm rates are ignored). This is the second of
                the two POMDP probability tables (the first being
                critic_likelihoods).
        """
        self.costs = costs
        self.max_generators = max_generators
        self.max_verifications = max_verifications
        self.critic_likelihoods = (
            critic_likelihoods if critic_likelihoods is not None
            else CRITIC_LIKELIHOODS
        )
        self.transition_kernel = transition_kernel  # None = use per-arm hand-tuned
        self._critic_names = list(self.critic_likelihoods.keys())
        self._cache: dict[DPState, tuple[float, str]] = {}

    def solve(self) -> None:
        """Pre-compute the value table by calling _value on all initial states."""
        # Warm up by solving from the starting state region
        # (lazy memoization means we only compute reachable states)
        for b_idx in range(BELIEF_GRID_SIZE):
            self._value(DPState(
                belief_idx=b_idx,
                gen_left=self.max_generators,
                crit_used=frozenset(),
                ver_left=self.max_verifications,
            ))

    def _value(self, state: DPState) -> float:
        """Bellman recursion: V(state) = max_a [-C_a + E[V(next) | a]]."""
        if state in self._cache:
            return self._cache[state][0]

        b = grid_belief(state.belief_idx)
        best_val = 0.0       # bail_out always yields 0
        best_action = "bail_out"

        # --- ACTION: verify (run full test suite) ---
        if state.ver_left > 0:
            # If patch is correct (prob b): get reward
            # If patch is wrong (prob 1-b): lose c_full_test, belief drops to ~0.05
            b_fail_idx = discretize(0.05)
            next_after_fail = DPState(
                belief_idx=b_fail_idx,
                gen_left=state.gen_left,
                crit_used=state.crit_used,
                ver_left=state.ver_left - 1,
            )
            v_after_fail = self._value(next_after_fail) if state.ver_left > 1 else 0.0
            q_ver = -self.costs.c_full_test + b * self.costs.reward + (1 - b) * v_after_fail
            if q_ver > best_val:
                best_val = q_ver
                best_action = "verify"

        # --- ACTION: generate(arm) ---
        # If a measured transition_kernel is provided, all arms share the same
        # transition; otherwise fall back to per-arm hand-tuned rates.
        if state.gen_left > 0:
            if self.transition_kernel is not None:
                # Single arm with measured kernel.
                # b' = b·(1 - P(break|correct)) + (1-b)·P(fix|broken)
                p_fix = self.transition_kernel["p_fix_broken"]
                p_break = self.transition_kernel["p_break_correct"]
                b_next = b * (1 - p_break) + (1 - b) * p_fix
                b_next_idx = discretize(b_next)
                next_state = DPState(
                    belief_idx=b_next_idx,
                    gen_left=state.gen_left - 1,
                    crit_used=state.crit_used,
                    ver_left=state.ver_left,
                )
                q_gen = -self.costs.c_llm_call + self._value(next_state)
                if q_gen > best_val:
                    best_val = q_gen
                    best_action = "generate:measured_kernel"
            else:
                for arm_idx, arm in enumerate(_GENERATOR_ARMS):
                    b_next = generator_transition(b, arm)
                    b_next_idx = discretize(b_next)
                    next_state = DPState(
                        belief_idx=b_next_idx,
                        gen_left=state.gen_left - 1,
                        crit_used=state.crit_used,
                        ver_left=state.ver_left,
                    )
                    q_gen = -self.costs.c_llm_call + self._value(next_state)
                    if q_gen > best_val:
                        best_val = q_gen
                        best_action = f"generate:{arm}"

        # --- ACTION: critic(subset) ---
        for critic_name in self._critic_names:
            if critic_name in state.crit_used:
                continue  # already used this critic
            lk = self.critic_likelihoods[critic_name]
            # P(pass) = P(pass|Y=1)*b + P(pass|Y=0)*(1-b)
            p_pass = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)

            b_if_pass = bayes_update(b, critic_name, passed=True,
                                     likelihoods=self.critic_likelihoods)
            b_if_fail = bayes_update(b, critic_name, passed=False,
                                     likelihoods=self.critic_likelihoods)

            next_crit_used = state.crit_used | frozenset([critic_name])

            next_pass = DPState(
                belief_idx=discretize(b_if_pass),
                gen_left=state.gen_left,
                crit_used=next_crit_used,
                ver_left=state.ver_left,
            )
            next_fail = DPState(
                belief_idx=discretize(b_if_fail),
                gen_left=state.gen_left,
                crit_used=next_crit_used,
                ver_left=state.ver_left,
            )

            q_crit = (
                -self.costs.c_critic_test
                + p_pass * self._value(next_pass)
                + (1 - p_pass) * self._value(next_fail)
            )
            if q_crit > best_val:
                best_val = q_crit
                best_action = f"critic:{critic_name}"

        self._cache[state] = (best_val, best_action)
        return best_val

    def choose_action(
        self,
        belief: float,
        gen_left: int,
        crit_used: frozenset[str],
        ver_left: int,
    ) -> tuple[str, float]:
        """Look up the optimal action for the current state.

        Returns (action_string, expected_value).
        action_string is one of: "generate:<arm>", "critic:<name>", "verify", "bail_out"
        """
        state = DPState(
            belief_idx=discretize(belief),
            gen_left=gen_left,
            crit_used=crit_used,
            ver_left=ver_left,
        )
        if state not in self._cache:
            self._value(state)
        return self._cache[state][1], self._cache[state][0]

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Bayesian POMDP Agent
# ---------------------------------------------------------------------------

def run_bayes_agent(
    bug_id: str,
    llm_config: LLMConfig | None = None,
    cost_config: AgentCostConfig | None = None,
    max_generators: int = 3,
    max_verifications: int = 2,
    prior: float = 0.1,
    use_dp: bool = True,
    critic_likelihoods: dict | None = None,
    policy_name: str | None = None,
) -> BayesAgentResult:
    """Run the Bayesian POMDP agent.

    When use_dp=True (default): uses full Bellman DP recursion to choose
    the optimal action at each step — matches the article's formulation.

    When use_dp=False: uses the greedy one-step lookahead (legacy).

    Args:
        critic_likelihoods: optional override for CRITIC_LIKELIHOODS, e.g.
            empirical-Bayes calibrated table. When None, uses the hand-tuned defaults.
        policy_name: label attached to the result (default: "bayes_pomdp" for DP).
    """
    if llm_config is None:
        llm_config = LLMConfig()
    if cost_config is None:
        cost_config = AgentCostConfig()

    if use_dp:
        return _run_dp_loop(bug_id, llm_config, cost_config,
                            max_generators, max_verifications, prior,
                            critic_likelihoods=critic_likelihoods,
                            policy_name=policy_name)
    else:
        return _run_greedy_loop(bug_id, llm_config, cost_config,
                                max_generators, prior)


def _run_dp_loop(
    bug_id: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    max_generators: int,
    max_verifications: int,
    prior: float,
    critic_likelihoods: dict | None = None,
    policy_name: str | None = None,
) -> BayesAgentResult:
    """DP-optimal agent: Bellman recursion chooses every action."""
    lk_table = critic_likelihoods if critic_likelihoods is not None else CRITIC_LIKELIHOODS
    buggy_source = BUGGY_SOURCES_REAL[bug_id]
    result = BayesAgentResult(bug_id=bug_id)
    if policy_name:
        result.policy_name = policy_name
    belief = prior
    result.belief_timeline.append(belief)
    start_time = time.perf_counter()

    # Pre-compute optimal policy
    planner = DPPlanner(cost_config, max_generators, max_verifications,
                        critic_likelihoods=lk_table)
    planner.solve()
    console.print(
        f"  [dim]DP solved: {planner.cache_size} states, "
        f"initial belief={belief:.3f}[/dim]"
    )

    gen_left = max_generators
    ver_left = max_verifications
    crit_used: frozenset[str] = frozenset()
    step = 0
    max_steps = 20  # safety limit

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "log_parser.py"
        test_file = workdir / "test_log_parser.py"
        test_file.write_text(LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        current_source = buggy_source

        while step < max_steps:
            action, value = planner.choose_action(
                belief, gen_left, crit_used, ver_left
            )
            belief_before = belief

            console.print(
                f"  Step {step}: b={belief:.3f} -> "
                f"[yellow]{action}[/yellow] (V={value:.1f})"
            )

            if action == "bail_out":
                result.action_log.append({
                    "step": step, "action": "bail_out",
                    "belief_before": belief, "belief_after": belief,
                    "value": value, "cost": 0,
                })
                console.print(f"  [dim]BAIL OUT: V={value:.1f}[/dim]")
                break

            elif action == "verify":
                all_pass, output, failing = _run_full_tests(workdir)
                result.test_runs += 1
                result.total_cost += cost_config.c_full_test
                ver_left -= 1

                result.action_log.append({
                    "step": step, "action": "verify",
                    "belief_before": belief, "belief_after": belief,
                    "result": "pass" if all_pass else f"fail({len(failing)})",
                    "value": value, "cost": cost_config.c_full_test,
                })
                step += 1
                result.belief_timeline.append(belief)

                if all_pass:
                    result.fixed = True
                    result.reward = cost_config.reward
                    result.final_diff = make_diff(buggy_source, current_source)
                    console.print(f"  [green]VERIFIED: all tests pass![/green]")
                    break
                else:
                    belief = 0.05
                    console.print(f"  [red]VERIFY FAILED: {failing}[/red]")
                    result.belief_timeline.append(belief)

            elif action.startswith("generate:"):
                arm = action.split(":", 1)[1]

                # Get current test output for the prompt
                sub = _safe_run(
                    [sys.executable, "-m", "pytest", "-v",
                     str(workdir / "test_log_parser.py")],
                    cwd=workdir,
                )
                test_output = sub.stdout + sub.stderr

                prompt = REAL_ARM_PROMPTS[arm].format(
                    source_code=current_source,
                    test_output=test_output,
                    test_code=LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)),
                )
                console.print(f"    Calling LLM ({llm_config.model})...")
                llm_resp = call_llm(prompt, llm_config)
                fixed_code = extract_code_from_response(llm_resp.text)

                src_file.write_text(fixed_code)
                current_source = fixed_code
                result.llm_calls += 1
                result.total_cost += cost_config.c_llm_call
                gen_left -= 1

                belief = generator_transition(belief, arm)

                result.action_log.append({
                    "step": step, "action": action,
                    "belief_before": belief_before, "belief_after": belief,
                    "value": value, "cost": cost_config.c_llm_call,
                })
                step += 1
                result.belief_timeline.append(belief)

            elif action.startswith("critic:"):
                critic_name = action.split(":", 1)[1]

                passed, failing = _run_critic_tests(
                    workdir, REAL_CRITIC_TESTS[critic_name]
                )
                result.critic_runs += 1
                result.total_cost += cost_config.c_critic_test
                crit_used = crit_used | frozenset([critic_name])

                belief = bayes_update(belief, critic_name, passed,
                                      likelihoods=lk_table)

                status = "[green]pass[/green]" if passed else "[red]fail[/red]"
                console.print(
                    f"    {status} -> b={belief:.3f}"
                )
                result.action_log.append({
                    "step": step, "action": action,
                    "belief_before": belief_before, "belief_after": belief,
                    "result": "pass" if passed else "fail",
                    "value": value, "cost": cost_config.c_critic_test,
                })
                step += 1
                result.belief_timeline.append(belief)

            else:
                console.print(f"  [red]Unknown action: {action}[/red]")
                break

    result.steps = step
    result.wall_clock_seconds = time.perf_counter() - start_time
    result.utility = result.reward - result.total_cost
    return result


def _run_greedy_loop(
    bug_id: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    max_generators: int,
    prior: float,
) -> BayesAgentResult:
    """Legacy greedy agent: one-step Q-value lookahead."""
    buggy_source = BUGGY_SOURCES_REAL[bug_id]
    result = BayesAgentResult(bug_id=bug_id, policy_name="bayes_greedy")
    belief = prior
    result.belief_timeline.append(belief)
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        src_file = workdir / "log_parser.py"
        test_file = workdir / "test_log_parser.py"
        test_file.write_text(LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)))
        src_file.write_text(buggy_source)
        current_source = buggy_source

        critic_idx = 0
        step = 0

        console.print(f"  [dim]Initial belief={belief:.3f}[/dim]")

        for gen_attempt in range(max_generators):
            arm = ARM_SEQUENCE[gen_attempt % len(ARM_SEQUENCE)]
            belief_before = belief

            sub = _safe_run(
                [sys.executable, "-m", "pytest", "-v",
                 str(workdir / "test_log_parser.py")],
                cwd=workdir,
            )
            test_output = sub.stdout + sub.stderr

            prompt = REAL_ARM_PROMPTS[arm].format(
                source_code=current_source,
                test_output=test_output,
                test_code=LOG_PARSER_TEST_CONTENT.format(src_dir=str(workdir)),
            )
            console.print(
                f"  Step {step}: b={belief:.3f} -> "
                f"[yellow]generate:{arm}[/yellow]"
            )
            console.print(f"    Calling LLM ({llm_config.model})...")
            llm_resp = call_llm(prompt, llm_config)
            fixed_code = extract_code_from_response(llm_resp.text)

            src_file.write_text(fixed_code)
            current_source = fixed_code
            result.llm_calls += 1
            result.total_cost += cost_config.c_llm_call
            belief = generator_transition(belief, arm)

            result.action_log.append({
                "step": step, "action": f"generate:{arm}",
                "belief_before": belief_before, "belief_after": belief,
                "cost": cost_config.c_llm_call,
            })
            step += 1

            if critic_idx < len(CRITIC_SEQUENCE):
                critic_name = CRITIC_SEQUENCE[critic_idx]
                critic_idx += 1
                belief_before_critic = belief

                passed, failing = _run_critic_tests(
                    workdir, REAL_CRITIC_TESTS[critic_name]
                )
                belief = bayes_update(belief, critic_name, passed)
                result.critic_runs += 1
                result.total_cost += cost_config.c_critic_test

                result.action_log.append({
                    "step": step, "action": f"critic:{critic_name}",
                    "belief_before": belief_before_critic, "belief_after": belief,
                    "result": "pass" if passed else "fail",
                    "cost": cost_config.c_critic_test,
                })
                step += 1

            result.belief_timeline.append(belief)

            q_v = q_verify(belief, cost_config)
            if q_v > 0 and belief > 0.5:
                all_pass, output, failing = _run_full_tests(workdir)
                result.test_runs += 1
                result.total_cost += cost_config.c_full_test

                result.action_log.append({
                    "step": step, "action": "verify",
                    "belief_before": belief, "belief_after": belief,
                    "result": "pass" if all_pass else f"fail({len(failing)})",
                    "cost": cost_config.c_full_test,
                })
                step += 1
                result.belief_timeline.append(belief)

                if all_pass:
                    result.fixed = True
                    result.reward = cost_config.reward
                    result.final_diff = make_diff(buggy_source, current_source)
                    break
                else:
                    belief = 0.05
                    result.belief_timeline.append(belief)
                    continue

            if gen_attempt < max_generators - 1:
                next_arm = ARM_SEQUENCE[(gen_attempt + 1) % len(ARM_SEQUENCE)]
                b_next = generator_transition(belief, next_arm)
                q_next_gen = -cost_config.c_llm_call + max(q_verify(b_next, cost_config), 0)
                if q_next_gen < 0:
                    break

        if not result.fixed and not any(
            a.get("action") == "verify" for a in result.action_log
        ):
            q_v = q_verify(belief, cost_config)
            if q_v > 0:
                all_pass, _, _ = _run_full_tests(workdir)
                result.test_runs += 1
                result.total_cost += cost_config.c_full_test
                result.action_log.append({
                    "step": step, "action": "verify",
                    "belief_before": belief, "belief_after": belief,
                    "result": "pass" if all_pass else "fail",
                    "cost": cost_config.c_full_test,
                })
                if all_pass:
                    result.fixed = True
                    result.reward = cost_config.reward
                    result.final_diff = make_diff(buggy_source, current_source)

    result.steps = step
    result.wall_clock_seconds = time.perf_counter() - start_time
    result.utility = result.reward - result.total_cost
    return result
