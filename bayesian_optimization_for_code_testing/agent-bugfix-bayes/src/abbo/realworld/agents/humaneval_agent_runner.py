"""End-to-end Bayesian POMDP agent runner on HumanEvalFix.

Wires an LLM `generate` action into the agent decision loop. Unlike the
simulator (which freezes the patch), this runner produces fresh patches via
the LLM each time the planner picks `generate:<arm>`.

Compares 4 agent variants on the same task set:
    - simple        — always-verify, retry up to N times (no critics, no bail)
    - greedy_hand   — Bayesian Greedy with hand-tuned theta
    - greedy_fitted — Bayesian Greedy with calibrated theta
    - dp_hand       — DP + hand-tuned theta
    - dp_fitted     — DP + calibrated theta (our method)

Output per task per variant: fix flag, total $/cost, wall_clock, tokens used,
action log, belief trajectory.
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from abbo.realworld.agents.bayes_agent import (
    DPPlanner,
    bayes_update,
)
from abbo.realworld.agents.humaneval_fix import (
    HE_CRITIC_LIKELIHOODS,
    HE_CRITIC_NAMES,
    get_buggy_source,
    get_canonical_source,
    get_full_test_script,
    run_critic,
    run_full_test,
)
from abbo.realworld.agents.llm_provider import LLMConfig, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig


# ---------------------------------------------------------------------------
# Generator-arm prompts (4 prompts, increasing complexity / context)
# ---------------------------------------------------------------------------

ARM_PROMPTS: dict[str, str] = {
    "g1_direct_fix": """You are a Python code repair assistant. Given a buggy Python function and its test output, return the COMPLETE corrected function — only the code, no explanation.

Buggy code:
```python
{source_code}
```

Test output (failing):
```
{test_output}
```

Return only the corrected function (no markdown, no comments).""",
    "g2_localized_fix": """The function below has a bug. Identify the smallest change needed and return the COMPLETE corrected function. Be conservative — keep style and structure unchanged.

Buggy code:
```python
{source_code}
```

Test output:
```
{test_output}
```

Return only the corrected function code, nothing else.""",
    "g3_test_guided_fix": """Read the test cases carefully, then fix the function.

Function:
```python
{source_code}
```

Test code:
```python
{test_code}
```

Test output:
```
{test_output}
```

Return only the corrected complete function, no markdown.""",
    "g4_error_focused_fix": """The function below fails the tests below. Look at the assertion error message and fix the root cause.

```python
{source_code}
```

Failing test output:
```
{test_output}
```

Return only the corrected complete function code (no markdown, no commentary).""",
}

# Generator transition probabilities (rough priors for DPPlanner)
GENERATOR_TRANSITIONS = {
    "g1_direct_fix":         {"p01": 0.50, "p10": 0.05},
    "g2_localized_fix":      {"p01": 0.45, "p10": 0.05},
    "g3_test_guided_fix":    {"p01": 0.40, "p10": 0.08},
    "g4_error_focused_fix":  {"p01": 0.35, "p10": 0.08},
}

ARM_SEQUENCE = list(ARM_PROMPTS.keys())


# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL)


def extract_code(llm_text: str | None, fallback: str) -> str:
    """Extract Python code from an LLM response. Prefer fenced blocks; else
    return whole text. Falls back to `fallback` if extraction yields empty."""
    if not isinstance(llm_text, str):
        return fallback
    m = _CODE_FENCE_RE.search(llm_text)
    if m:
        body = m.group(1).strip()
        if body:
            return body
    stripped = llm_text.strip()
    return stripped if stripped else fallback


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    task_id: str
    variant: str
    fixed: bool
    total_cost: float
    wall_clock: float
    n_llm_calls: int
    n_critic_runs: int
    n_full_tests: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    final_action: str = ""
    actions: list[dict] = field(default_factory=list)


def _safe_subprocess_run(cmd, cwd, timeout=15):
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd, returncode=124, stdout="", stderr="TIMEOUT",
        )


def _get_test_output(workdir: Path, task_id: str) -> str:
    """Run the full check() and return stdout+stderr (for prompt context)."""
    script_path = workdir / "_test_for_prompt.py"
    body = (
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('solution', 'solution.py')\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "globals().update({k:v for k,v in vars(mod).items() if not k.startswith('_')})\n"
    )
    body += get_full_test_script(task_id)
    script_path.write_text(body)
    r = _safe_subprocess_run([sys.executable, str(script_path)], workdir, timeout=15)
    return ((r.stdout or "") + (r.stderr or ""))[:1500]


# ---------------------------------------------------------------------------
# Variant 1: Simple agent (always-verify, retry-N)
# ---------------------------------------------------------------------------

def run_simple(
    task_id: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    n_retries: int = 3,
) -> AgentRunResult:
    res = AgentRunResult(task_id=task_id, variant="simple", fixed=False,
                         total_cost=0.0, wall_clock=0.0,
                         n_llm_calls=0, n_critic_runs=0, n_full_tests=0)
    start = time.perf_counter()
    buggy = get_buggy_source(task_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        sol = workdir / "solution.py"
        sol.write_text(buggy)
        current = buggy

        for attempt in range(n_retries):
            test_out = _get_test_output(workdir, task_id)
            arm = ARM_SEQUENCE[attempt % len(ARM_SEQUENCE)]
            prompt = ARM_PROMPTS[arm].format(
                source_code=current,
                test_output=test_out,
                test_code=get_full_test_script(task_id),
            )
            resp = call_llm_or_raise(prompt, llm_config)
            res.n_llm_calls += 1
            res.total_cost += cost_config.c_llm_call
            res.prompt_tokens += resp.prompt_tokens
            res.completion_tokens += resp.completion_tokens
            current = extract_code(resp.text, current)
            sol.write_text(current)

            ok, _ = run_full_test(workdir, task_id)
            res.n_full_tests += 1
            res.total_cost += cost_config.c_full_test
            res.actions.append({"step": attempt, "arm": arm, "verify_pass": ok})
            if ok:
                res.fixed = True
                res.final_action = "verify_pass"
                break

        if not res.fixed:
            res.final_action = "exhausted"

    res.wall_clock = time.perf_counter() - start
    return res


# ---------------------------------------------------------------------------
# Variant 2/3: Greedy with theta
# ---------------------------------------------------------------------------

def _q_one_step_critic(b, critic, theta, costs):
    lk = theta[critic]
    p_pass = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)
    b_pass = bayes_update(b, critic, True, likelihoods=theta)
    b_fail = bayes_update(b, critic, False, likelihoods=theta)
    v_pass = max(0.0, -costs.c_full_test + b_pass * costs.reward)
    v_fail = max(0.0, -costs.c_full_test + b_fail * costs.reward)
    return -costs.c_critic_test + p_pass * v_pass + (1 - p_pass) * v_fail


def _greedy_pick_action(belief, gen_left, crit_used, theta, costs):
    Q_bail = 0.0
    Q_verify = -costs.c_full_test + belief * costs.reward

    Q_critics = {}
    for c in theta.keys():
        if c in crit_used:
            continue
        Q_critics[c] = _q_one_step_critic(belief, c, theta, costs)
    best_critic, best_q = (
        max(Q_critics.items(), key=lambda x: x[1])
        if Q_critics else (None, -math.inf)
    )

    Q_gen = -math.inf
    if gen_left > 0:
        # Greedy gen: assume one more verify after this regen
        b_after_gen = belief * 0.95 + (1 - belief) * 0.50  # rough avg arm
        Q_gen = -costs.c_llm_call - costs.c_full_test + b_after_gen * costs.reward

    choices = [("bail", Q_bail), ("verify", Q_verify)]
    if best_critic is not None:
        choices.append((f"critic:{best_critic}", best_q))
    if gen_left > 0:
        choices.append((f"generate:{ARM_SEQUENCE[0]}", Q_gen))

    return max(choices, key=lambda x: x[1])


def run_greedy(
    task_id: str,
    theta: dict,
    theta_label: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    max_generators: int = 3,
    prior: float = 0.5,
) -> AgentRunResult:
    res = AgentRunResult(task_id=task_id, variant=f"greedy_{theta_label}",
                         fixed=False, total_cost=0.0, wall_clock=0.0,
                         n_llm_calls=0, n_critic_runs=0, n_full_tests=0)
    start = time.perf_counter()
    buggy = get_buggy_source(task_id)

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        sol = workdir / "solution.py"
        sol.write_text(buggy)
        current = buggy
        belief = prior
        gen_left = max_generators
        crit_used: set[str] = set()
        step = 0
        max_steps = 12

        while step < max_steps:
            action, _q = _greedy_pick_action(
                belief, gen_left, crit_used, theta, cost_config,
            )
            if action == "bail":
                res.final_action = "bail"
                res.actions.append({"step": step, "action": "bail", "belief": belief})
                break
            if action == "verify":
                ok, _ = run_full_test(workdir, task_id)
                res.n_full_tests += 1
                res.total_cost += cost_config.c_full_test
                res.actions.append({"step": step, "action": "verify",
                                    "belief": belief, "result": ok})
                if ok:
                    res.fixed = True
                    res.final_action = "verify_pass"
                    break
                # Failed verify: belief crashes, optionally try again
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(workdir, cn, task_id)
                res.n_critic_runs += 1
                res.total_cost += cost_config.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used.add(cn)
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "belief": belief})
            elif action.startswith("generate:"):
                arm = action.split(":", 1)[1]
                test_out = _get_test_output(workdir, task_id)
                prompt = ARM_PROMPTS[arm].format(
                    source_code=current, test_output=test_out,
                    test_code=get_full_test_script(task_id),
                )
                resp = call_llm_or_raise(prompt, llm_config)
                res.n_llm_calls += 1
                res.total_cost += cost_config.c_llm_call
                res.prompt_tokens += resp.prompt_tokens
                res.completion_tokens += resp.completion_tokens
                current = extract_code(resp.text, current)
                sol.write_text(current)
                gen_left -= 1
                belief = belief * 0.95 + (1 - belief) * 0.50  # informed by arm
                crit_used = set()  # reset critic memory after new patch
                res.actions.append({"step": step, "action": action,
                                    "belief": belief, "tokens": resp.completion_tokens})
            step += 1

        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"

    res.wall_clock = time.perf_counter() - start
    return res


# ---------------------------------------------------------------------------
# Variant 4/5: DP with theta
# ---------------------------------------------------------------------------

def run_dp(
    task_id: str,
    theta: dict,
    theta_label: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    max_generators: int = 3,
    max_verifications: int = 2,
    prior: float = 0.5,
    planner: DPPlanner | None = None,
) -> AgentRunResult:
    res = AgentRunResult(task_id=task_id, variant=f"dp_{theta_label}",
                         fixed=False, total_cost=0.0, wall_clock=0.0,
                         n_llm_calls=0, n_critic_runs=0, n_full_tests=0)
    start = time.perf_counter()
    buggy = get_buggy_source(task_id)

    if planner is None:
        planner = DPPlanner(
            costs=cost_config,
            max_generators=max_generators,
            max_verifications=max_verifications,
            critic_likelihoods=theta,
        )
        planner.solve()

    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        sol = workdir / "solution.py"
        sol.write_text(buggy)
        current = buggy
        belief = prior
        gen_left = max_generators
        ver_left = max_verifications
        crit_used: frozenset[str] = frozenset()
        step = 0
        max_steps = 16

        while step < max_steps:
            action, _q = planner.choose_action(
                belief=belief, gen_left=gen_left,
                crit_used=crit_used, ver_left=ver_left,
            )
            if action == "bail_out":
                res.final_action = "bail"
                res.actions.append({"step": step, "action": "bail", "belief": belief})
                break
            if action == "verify":
                ok, _ = run_full_test(workdir, task_id)
                res.n_full_tests += 1
                res.total_cost += cost_config.c_full_test
                ver_left -= 1
                res.actions.append({"step": step, "action": "verify",
                                    "belief": belief, "result": ok})
                if ok:
                    res.fixed = True
                    res.final_action = "verify_pass"
                    break
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(workdir, cn, task_id)
                res.n_critic_runs += 1
                res.total_cost += cost_config.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used = crit_used | frozenset([cn])
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "belief": belief})
            elif action.startswith("generate:"):
                arm = action.split(":", 1)[1]
                if arm not in ARM_PROMPTS:
                    arm = ARM_SEQUENCE[0]
                test_out = _get_test_output(workdir, task_id)
                prompt = ARM_PROMPTS[arm].format(
                    source_code=current, test_output=test_out,
                    test_code=get_full_test_script(task_id),
                )
                resp = call_llm_or_raise(prompt, llm_config)
                res.n_llm_calls += 1
                res.total_cost += cost_config.c_llm_call
                res.prompt_tokens += resp.prompt_tokens
                res.completion_tokens += resp.completion_tokens
                current = extract_code(resp.text, current)
                sol.write_text(current)
                gen_left -= 1
                t = GENERATOR_TRANSITIONS.get(arm, {"p01": 0.4, "p10": 0.05})
                belief = belief * (1 - t["p10"]) + (1 - belief) * t["p01"]
                crit_used = frozenset()  # new patch — reset critic memory
                res.actions.append({"step": step, "action": action,
                                    "belief": belief,
                                    "tokens": resp.completion_tokens})
            step += 1

        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"

    res.wall_clock = time.perf_counter() - start
    return res


# ---------------------------------------------------------------------------
# Top-level: run all variants on a task list
# ---------------------------------------------------------------------------

def run_grid(
    task_ids: list[str],
    fitted_theta: dict,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig | None = None,
    variants: tuple[str, ...] = ("simple", "greedy_hand", "greedy_fitted",
                                  "dp_hand", "dp_fitted"),
    prior: float = 0.5,
    verbose: bool = True,
) -> dict[str, list[AgentRunResult]]:
    """Run every variant on every task. Returns {variant: [results...]}."""
    if cost_config is None:
        cost_config = AgentCostConfig()

    # Pre-solve DP planners (one per theta)
    dp_hand = DPPlanner(cost_config, max_generators=3, max_verifications=2,
                        critic_likelihoods=HE_CRITIC_LIKELIHOODS)
    dp_hand.solve()
    dp_fitted = DPPlanner(cost_config, max_generators=3, max_verifications=2,
                          critic_likelihoods=fitted_theta)
    dp_fitted.solve()

    results: dict[str, list[AgentRunResult]] = {v: [] for v in variants}
    for i, tid in enumerate(task_ids):
        if verbose:
            print(f"\n[{i+1}/{len(task_ids)}] task={tid}")
        for v in variants:
            if v == "simple":
                r = run_simple(tid, llm_config, cost_config)
            elif v == "greedy_hand":
                r = run_greedy(tid, HE_CRITIC_LIKELIHOODS, "hand",
                               llm_config, cost_config, prior=prior)
            elif v == "greedy_fitted":
                r = run_greedy(tid, fitted_theta, "fitted",
                               llm_config, cost_config, prior=prior)
            elif v == "dp_hand":
                r = run_dp(tid, HE_CRITIC_LIKELIHOODS, "hand",
                           llm_config, cost_config, prior=prior, planner=dp_hand)
            elif v == "dp_fitted":
                r = run_dp(tid, fitted_theta, "fitted",
                           llm_config, cost_config, prior=prior, planner=dp_fitted)
            else:
                continue
            results[v].append(r)
            if verbose:
                tag = "✅" if r.fixed else "❌"
                print(f"  {tag} {v:>16}  cost={r.total_cost:6.1f}  "
                      f"llm={r.n_llm_calls}  crit={r.n_critic_runs}  "
                      f"toks={r.completion_tokens}  wc={r.wall_clock:.1f}s  "
                      f"final={r.final_action}")

    return results


def aggregate(results: dict[str, list[AgentRunResult]]) -> dict[str, dict]:
    """Per-variant summary across all tasks."""
    out = {}
    for v, rs in results.items():
        n = len(rs)
        if n == 0:
            continue
        n_fixed = sum(1 for r in rs if r.fixed)
        out[v] = {
            "n_tasks": n,
            "fix_rate": n_fixed / n,
            "avg_cost": sum(r.total_cost for r in rs) / n,
            "avg_llm_calls": sum(r.n_llm_calls for r in rs) / n,
            "avg_critic_runs": sum(r.n_critic_runs for r in rs) / n,
            "avg_full_tests": sum(r.n_full_tests for r in rs) / n,
            "avg_completion_tokens": sum(r.completion_tokens for r in rs) / n,
            "avg_wall_clock": sum(r.wall_clock for r in rs) / n,
        }
    return out


def format_summary(agg: dict[str, dict]) -> str:
    cols = ["fix_rate", "avg_cost", "avg_llm_calls", "avg_critic_runs",
            "avg_full_tests", "avg_completion_tokens", "avg_wall_clock"]
    header = f"{'variant':<18} {'n':>4} " + " ".join(f"{c:>14}" for c in cols)
    lines = [header, "-" * len(header)]
    for v, m in agg.items():
        row = f"{v:<18} {m['n_tasks']:>4} "
        for c in cols:
            val = m.get(c, 0.0)
            row += f"{val:>14.3f} "
        lines.append(row)
    return "\n".join(lines)
