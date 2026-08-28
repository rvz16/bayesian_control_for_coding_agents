#!/usr/bin/env python
"""LIVE end-to-end Bayesian agent on synthesis benchmarks (LCB / MBPP+ / HumanEval+).

Mirrors the methodology of abbo's run_codecontests_full.py / run_humaneval_full.py
but for synthesis tasks (no buggy seed; first action is always `generate`).

Difference from `run_synthesis_endtoend.py` (replay-style):
  - This script makes LIVE LLM calls during the agent loop.
  - When the agent decides to regenerate, the LLM receives feedback
    (e.g., "critic L2 failed with output X"), producing a new patch
    informed by previous failures — true closed-loop iterative refinement.
  - GrFt/DPFt results here are directly comparable to abbo bug-fix
    GrFt/DPFt cells in tab:full_results (both are live end-to-end).

Reads:
  <src-dir>/<gen>/split.json (from synthesis_train_test_split.py)
  <src-dir>/<gen>/likelihood_tables.json (train-fitted theta)

Writes:
  <output>: sim_results JSON in the same schema as abbo *_full_endtoend__*.json
  logs/action_telemetry_synthesis_live__<bench>__<gen>.jsonl

Usage:
  ABBO_LLM_MODEL=anthropic/claude-haiku-4.5 \\
  python scripts/run_synthesis_live.py \\
      --src-dir data/mbpp_full \\
      --benchmark mbpp --generator haiku45 \\
      --output sim_results/synthesis_live__mbpp__haiku45.json \\
      --variants fitted
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Shared kernel utilities (Beta-Binomial online estimator + DEFAULT_KERNEL +
# kernel_update belief propagation + resolve_kernel mode dispatch).
from _common.kernel import (  # noqa: E402
    DEFAULT_KERNEL,
    OnlineKernelCalibration,
    kernel_update,
    resolve_kernel,
)

# abbo bayes machinery (DPPlanner, bayes_update)
ABBO_SRC = ROOT.parent.parent / "bayesian_optimization_for_code_testing" \
                              / "agent-bugfix-bayes" / "src"
sys.path.insert(0, str(ABBO_SRC))
from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update  # noqa: E402

# .env auto-load (matches calibrate scripts)
try:
    from dotenv import load_dotenv
    for env_path in [ROOT / ".env", ROOT.parent / ".env",
                     ROOT.parent.parent / ".env", ROOT.parent.parent.parent / ".env",
                     ROOT.parent.parent.parent.parent / ".env",
                     ROOT.parent.parent.parent.parent.parent / ".env"]:
        if env_path.exists() and env_path.stat().st_size > 0:
            load_dotenv(env_path, override=False)
except ImportError:
    pass

log = logging.getLogger("synth_live")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# ============================================================================
# Constants — fast-oracle cost vector (synthesis benchmarks)
# ============================================================================
CRITIC_KEYS = ["L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"]
PRIOR = 0.5
MAX_GENERATORS = 3
MAX_VERIFICATIONS = 2
LLM_MODEL = os.environ.get("ABBO_LLM_MODEL", "anthropic/claude-haiku-4.5")
TEMPERATURE = 0.7
MAX_TOKENS = 4096
SPLIT_SEED = 42


@dataclass
class Costs:
    c_gen: int = 10
    c_critic: int = 1
    c_ver: int = 5
    reward: int = 100


# ============================================================================
# Result dataclass — matches abbo schema
# ============================================================================
@dataclass
class Result:
    task_id: str
    variant: str
    fixed: bool = False
    total_cost: float = 0.0
    n_llm_calls: int = 0
    n_critic_runs: int = 0
    n_full_tests: int = 0
    completion_tokens: int = 0
    api_cost_usd: float = 0.0
    final_action: str = ""
    actions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "variant": self.variant, "fixed": self.fixed,
            "total_cost": self.total_cost, "n_llm_calls": self.n_llm_calls,
            "n_critic_runs": self.n_critic_runs, "n_full_tests": self.n_full_tests,
            "completion_tokens": self.completion_tokens, "api_cost_usd": self.api_cost_usd,
            "final_action": self.final_action, "actions": self.actions,
        }


# ============================================================================
# Benchmark dispatcher — imports primitives from existing calibrate scripts
# ============================================================================
def _benchmark_loader(benchmark: str):
    """Return (load_fn, prompt_fn, run_critics_fn, run_verify_fn) for the
    given benchmark. Each returned function has a unified signature so the
    agent loop is benchmark-agnostic.

    load_fn() -> list[dict]
    prompt_fn(problem, prev_code=None, feedback=None) -> str
    run_critics_fn(code, problem) -> dict[critic_name -> bool]
    run_verify_fn(code, problem) -> tuple[bool, str]
    """
    if benchmark.startswith("lcb"):
        return _lcb_adapter(benchmark)
    elif benchmark == "mbpp":
        return _mbpp_adapter()
    elif benchmark == "humaneval":
        return _humaneval_adapter()
    else:
        raise ValueError(f"unknown benchmark: {benchmark!r} "
                         f"(supported: lcb_easy/medium/hard, mbpp, humaneval)")


def _lcb_adapter(benchmark: str):
    """Adapter for LCB-easy/medium/hard. Reuses lcb_calibrate primitives."""
    from calibration import lcb  # noqa: E402

    difficulty = benchmark.split("_", 1)[1] if "_" in benchmark else "hard"

    def load_fn():
        return lcb.load_lcb(difficulty=difficulty, platform="leetcode",
                            lcb_version="all")

    def prompt_fn(problem, prev_code=None, feedback=None):
        base = lcb.build_prompt(problem)
        if prev_code is None or feedback is None:
            return base
        # Refine prompt: include previous attempt + feedback
        return (
            f"{base}\n\n"
            f"## Previous attempt (failed):\n```python\n{prev_code}\n```\n\n"
            f"## Feedback:\n{feedback}\n\n"
            f"Return a corrected solution. Only the code, no explanations."
        )

    def run_critics_fn(code, problem, llm_client=None):
        out = {}
        out["L0_syntax"] = lcb.critic_L0_syntax(code)
        out["L1_lint"] = lcb.critic_L1_lint(code)
        # L2: public tests = first k from problem
        pub_tests = problem.get("public_test_cases", []) or []
        if pub_tests:
            n_pass, n_total = lcb.check_tests(code, pub_tests[:3],
                                              starter_code=problem.get("starter_code", ""))
            out["L2_public_tests"] = (n_pass == n_total) and n_total > 0
        else:
            out["L2_public_tests"] = True  # no public tests → vacuously pass
        # L3: LLM judge
        if llm_client is not None:
            try:
                passed, _ = lcb.critic_L3_review(
                    problem.get("question_content", problem.get("problem", "")),
                    code, llm_client)
                out["L3_llm_review"] = passed
            except Exception as e:
                log.warning("L3 judge failed: %s", e)
                out["L3_llm_review"] = False
        else:
            out["L3_llm_review"] = False
        return out

    def run_verify_fn(code, problem):
        priv = lcb.decode_private_tests(problem.get("private_test_cases", ""))
        if not priv:
            return False, "no private tests"
        n_pass, n_total = lcb.check_tests(code, priv,
                                          starter_code=problem.get("starter_code", ""))
        return (n_pass == n_total) and n_total > 0, f"{n_pass}/{n_total}"

    return load_fn, prompt_fn, run_critics_fn, run_verify_fn


def _mbpp_adapter():
    """Adapter for MBPP+. Reuses mbpp_calibrate primitives."""
    from calibration import mbpp  # noqa: E402

    def load_fn():
        return mbpp.load_mbpp_plus(n_instances=10**9, seed=42)  # load all

    def prompt_fn(problem, prev_code=None, feedback=None):
        base = mbpp.build_prompt(problem)
        if prev_code is None or feedback is None:
            return base
        return (
            f"{base}\n\n"
            f"## Previous attempt (failed):\n```python\n{prev_code}\n```\n\n"
            f"## Feedback:\n{feedback}\n\n"
            f"Return a corrected solution. Only the code, no explanations."
        )

    def run_critics_fn(code, problem, llm_client=None):
        out = {}
        out["L0_syntax"] = _syntax_check(code)
        out["L1_lint"] = _lint_check(code)
        # L2: a few public assertions
        pub_asserts = (problem.get("test_list") or [])[:3]
        if pub_asserts:
            n_pass, n_total = mbpp.run_assertions(code, pub_asserts, timeout=5)
            out["L2_public_tests"] = (n_pass == n_total) and n_total > 0
        else:
            out["L2_public_tests"] = True
        out["L3_llm_review"] = _llm_judge(code, problem.get("text", ""), llm_client)
        return out

    def run_verify_fn(code, problem):
        full_block = problem.get("test", "") or "\n".join(problem.get("test_list", []))
        if not full_block.strip():
            return False, "no tests"
        ok = mbpp.run_full_test(code, full_block, problem.get("entry_point"), timeout=10)
        return bool(ok), ""

    return load_fn, prompt_fn, run_critics_fn, run_verify_fn


def _humaneval_adapter():
    """Adapter for HumanEval+. Mirrors mbpp adapter (similar dataset shape)."""
    from calibration import humaneval as he  # noqa: E402

    def load_fn():
        return he.load_humaneval_plus(n_instances=10**9, seed=42)

    def prompt_fn(problem, prev_code=None, feedback=None):
        base = he.build_prompt(problem)
        if prev_code is None or feedback is None:
            return base
        return (
            f"{base}\n\n"
            f"## Previous attempt (failed):\n```python\n{prev_code}\n```\n\n"
            f"## Feedback:\n{feedback}\n\n"
            f"Return a corrected solution. Only the code, no explanations."
        )

    def run_critics_fn(code, problem, llm_client=None):
        out = {}
        out["L0_syntax"] = _syntax_check(code)
        out["L1_lint"] = _lint_check(code)
        # L2: try public test inputs from problem (subset)
        try:
            base_inputs = problem.get("base_input", problem.get("inputs", []))[:3]
            entry = problem.get("entry_point", "")
            canon = problem.get("canonical_solution", "")
            n_pass, n_total = he.run_test_inputs(code, entry, canon, base_inputs, atol=1e-6)
            out["L2_public_tests"] = (n_pass == n_total) and n_total > 0
        except Exception:
            out["L2_public_tests"] = False
        out["L3_llm_review"] = _llm_judge(code, problem.get("prompt", ""), llm_client)
        return out

    def run_verify_fn(code, problem):
        try:
            entry = problem.get("entry_point", "")
            canon = problem.get("canonical_solution", "")
            full_in = problem.get("plus_input", problem.get("inputs", []))
            n_pass, n_total = he.run_test_inputs(code, entry, canon, full_in, atol=1e-6)
            return (n_pass == n_total) and n_total > 0, f"{n_pass}/{n_total}"
        except Exception as e:
            return False, str(e)

    return load_fn, prompt_fn, run_critics_fn, run_verify_fn


# ============================================================================
# Generic critic helpers (syntax / lint / LLM judge) for MBPP/HumanEval
# ============================================================================
def _syntax_check(code: str) -> bool:
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def _lint_check(code: str) -> bool:
    # Same heuristic as lcb_calibrate.critic_L1_lint
    import re
    bad = re.findall(r"\b(print|input|breakpoint|exit|quit)\s*\(", code)
    return len(bad) == 0


def _llm_judge(code: str, problem_text: str, client) -> bool:
    if client is None:
        return False
    prompt = (
        f"You are a strict code reviewer. The candidate solution must solve the "
        f"problem correctly. Respond with YES if you are confident the solution "
        f"is correct on ALL inputs; respond with NO otherwise. Only YES or NO.\n\n"
        f"## Problem:\n{problem_text[:2000]}\n\n"
        f"## Candidate:\n```python\n{code[:3000]}\n```\n\n"
        f"Verdict (YES/NO):"
    )
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=8,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        return text.startswith("Y")
    except Exception as e:
        log.warning("LLM judge call failed: %s", e)
        return False


# ============================================================================
# Live LLM call — generate code (with optional feedback)
# ============================================================================
def llm_generate(client, prompt: str) -> tuple[str, int, int, float]:
    """Returns (extracted_code, prompt_tokens, completion_tokens, api_cost_usd)."""
    from calibration import lcb  # for cost_for_call + extract_code
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    text = resp.choices[0].message.content or ""
    code = lcb.extract_code(text) or text
    pt = resp.usage.prompt_tokens
    ct = resp.usage.completion_tokens
    cost = lcb.cost_for_call(LLM_MODEL, pt, ct)
    return code, pt, ct, cost


# ============================================================================
# Variant runners — LIVE execution
# ============================================================================
def run_simple(task_id, problem, prompt_fn, critics_fn, verify_fn, client, costs,
               n_retries=3) -> Result:
    res = Result(task_id=task_id, variant="simple")
    code = None
    feedback = None
    for attempt in range(n_retries):
        prompt = prompt_fn(problem, prev_code=code, feedback=feedback)
        code, pt, ct, dollar = llm_generate(client, prompt)
        res.n_llm_calls += 1; res.completion_tokens += ct; res.api_cost_usd += dollar
        res.total_cost += costs.c_gen
        res.actions.append({"step": 2*attempt, "action": "generate"})
        ok, info = verify_fn(code, problem)
        res.n_full_tests += 1; res.total_cost += costs.c_ver
        res.actions.append({"step": 2*attempt+1, "action": "verify", "ok": ok, "info": info})
        if ok:
            res.fixed = True; res.final_action = "verify_pass"; return res
        feedback = f"Hidden tests failed ({info})."
    res.final_action = "exhausted"
    return res


def run_best_of_n(task_id, problem, prompt_fn, critics_fn, verify_fn, client, costs,
                  n=3) -> Result:
    res = Result(task_id=task_id, variant="best_of_3")
    for k in range(n):
        code, pt, ct, dollar = llm_generate(client, prompt_fn(problem))
        res.n_llm_calls += 1; res.completion_tokens += ct; res.api_cost_usd += dollar
        res.total_cost += costs.c_gen
        res.actions.append({"step": 2*k, "action": "generate"})
        ok, info = verify_fn(code, problem)
        res.n_full_tests += 1; res.total_cost += costs.c_ver
        res.actions.append({"step": 2*k+1, "action": "verify", "ok": ok, "info": info})
        if ok:
            res.fixed = True; res.final_action = "verify_pass"; return res
    res.final_action = "exhausted"
    return res


def run_threshold(task_id, problem, prompt_fn, critics_fn, verify_fn, client, costs,
                  critics_order: list[str], variant_name: str) -> Result:
    res = Result(task_id=task_id, variant=variant_name)
    code, pt, ct, dollar = llm_generate(client, prompt_fn(problem))
    res.n_llm_calls += 1; res.completion_tokens += ct; res.api_cost_usd += dollar
    res.total_cost += costs.c_gen
    res.actions.append({"step": 0, "action": "generate"})
    crit_results = critics_fn(code, problem, llm_client=client)
    for k, cn in enumerate(critics_order):
        res.n_critic_runs += 1; res.total_cost += costs.c_critic
        passed = crit_results.get(cn, False)
        res.actions.append({"step": 1+k, "action": f"critic:{cn}", "passed": passed})
        if not passed:
            res.final_action = "critic_reject"; return res
    # All critics passed → verify
    ok, info = verify_fn(code, problem)
    res.n_full_tests += 1; res.total_cost += costs.c_ver
    res.fixed = ok
    res.final_action = "verify_pass" if ok else "verify_fail"
    res.actions.append({"step": 1+len(critics_order), "action": "verify", "ok": ok})
    return res


def run_fixed_pipeline(task_id, problem, prompt_fn, critics_fn, verify_fn, client,
                      costs) -> Result:
    res = Result(task_id=task_id, variant="fixed_pipeline")
    code, pt, ct, dollar = llm_generate(client, prompt_fn(problem))
    res.n_llm_calls += 1; res.completion_tokens += ct; res.api_cost_usd += dollar
    res.total_cost += costs.c_gen
    res.actions.append({"step": 0, "action": "generate"})
    crit_results = critics_fn(code, problem, llm_client=client)
    for k, cn in enumerate(CRITIC_KEYS):
        res.n_critic_runs += 1; res.total_cost += costs.c_critic
        passed = crit_results.get(cn, False)
        res.actions.append({"step": 1+k, "action": f"critic:{cn}", "passed": passed})
    ok, info = verify_fn(code, problem)
    res.n_full_tests += 1; res.total_cost += costs.c_ver
    res.fixed = ok
    res.final_action = "verify_pass" if ok else "verify_fail"
    res.actions.append({"step": 1+len(CRITIC_KEYS), "action": "verify", "ok": ok})
    return res


def _q_critic_one_step(b: float, critic_name: str, theta: dict, costs: Costs) -> float:
    lk = theta[critic_name]
    p_pass = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)
    b_pass = lk["p_pass_y1"] * b / max(p_pass, 1e-12)
    b_fail_denom = (1 - lk["p_pass_y1"]) * b + (1 - lk["p_pass_y0"]) * (1 - b)
    b_fail = (1 - lk["p_pass_y1"]) * b / max(b_fail_denom, 1e-12)
    return (
        -costs.c_critic
        + p_pass * max(0.0, -costs.c_ver + b_pass * costs.reward)
        + (1 - p_pass) * max(0.0, -costs.c_ver + b_fail * costs.reward)
    )


def run_greedy(task_id, problem, prompt_fn, critics_fn, verify_fn, client, costs,
               theta: dict, label: str, kernel: dict, prior: float = PRIOR,
               max_gen: int = MAX_GENERATORS,
               online_kernel: "OnlineKernelCalibration | None" = None) -> Result:
    """Live one-step Bayesian agent.

    kernel: transition kernel dict {"p_fix_broken", "p_break_correct"}.
    online_kernel: if provided, observed (Y_before, Y_after) transitions
                    are recorded and the kernel updates after each verify.
    """
    res = Result(task_id=task_id, variant=f"greedy_{label}")
    belief = prior
    gen_left = max_gen
    crit_used: set[str] = set()
    code: str | None = None
    cached_critics: dict = {}
    feedback: str | None = None
    step = 0
    prev_Y: int | None = None  # Y just before the current `code` (None at start)

    while step < 12:
        # Refresh kernel from online posterior if active
        active_kernel = online_kernel.get() if online_kernel is not None else kernel

        # Q-values
        Q_bail = 0.0
        Q_verify = -costs.c_ver + belief * costs.reward
        Q_critics = {}
        if code is not None:
            for cn in theta:
                if cn not in crit_used:
                    Q_critics[cn] = _q_critic_one_step(belief, cn, theta, costs)
        best_c, best_q = (max(Q_critics.items(), key=lambda x: x[1])
                          if Q_critics else (None, -math.inf))
        Q_gen = -math.inf
        if gen_left > 0:
            b_after = kernel_update(belief, active_kernel)
            Q_gen = -costs.c_gen - costs.c_ver + b_after * costs.reward

        # First action forced to generate
        if code is None:
            action = "generate"
        else:
            choices = [("bail", Q_bail), ("verify", Q_verify)]
            if best_c: choices.append((f"critic:{best_c}", best_q))
            if gen_left > 0: choices.append(("generate", Q_gen))
            action, _q = max(choices, key=lambda x: x[1])

        if action == "bail":
            res.final_action = "bail"; break

        if action == "verify":
            ok, info = verify_fn(code, problem)
            y_now = 1 if ok else 0
            if online_kernel is not None and prev_Y is not None:
                online_kernel.update(prev_Y, y_now)
            prev_Y = y_now
            res.n_full_tests += 1; res.total_cost += costs.c_ver
            res.actions.append({"step": step, "action": "verify", "ok": ok, "info": info})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            belief = 0.05
            feedback = f"Hidden tests failed ({info})."
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            if not cached_critics:
                cached_critics = critics_fn(code, problem, llm_client=client)
            passed = cached_critics.get(cn, False)
            res.n_critic_runs += 1; res.total_cost += costs.c_critic
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used.add(cn)
            res.actions.append({"step": step, "action": action,
                                "passed": passed, "b": belief})
        else:  # generate
            prompt = prompt_fn(problem, prev_code=code, feedback=feedback)
            code, pt, ct, dollar = llm_generate(client, prompt)
            res.n_llm_calls += 1; res.completion_tokens += ct; res.api_cost_usd += dollar
            res.total_cost += costs.c_gen
            gen_left -= 1
            belief = kernel_update(belief, active_kernel)
            crit_used = set()
            cached_critics = {}
            res.actions.append({"step": step, "action": "generate", "b": belief})
        step += 1

    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    return res


def run_dp(task_id, problem, prompt_fn, critics_fn, verify_fn, client, costs,
           planner_factory, theta: dict, label: str, kernel: dict,
           prior: float = PRIOR, max_gen: int = MAX_GENERATORS,
           max_ver: int = MAX_VERIFICATIONS,
           online_kernel: "OnlineKernelCalibration | None" = None) -> Result:
    """planner_factory: callable() -> DPPlanner. Called each time we need to
    re-solve (e.g. after online_kernel update). If online_kernel is None,
    only called once at the start of the episode.
    """
    res = Result(task_id=task_id, variant=f"dp_{label}")
    belief = prior
    gen_left = max_gen
    ver_left = max_ver
    crit_used: frozenset[str] = frozenset()
    code: str | None = None
    cached_critics: dict = {}
    feedback: str | None = None
    step = 0
    prev_Y: int | None = None

    planner = planner_factory()  # initial solve

    while step < 16:
        active_kernel = online_kernel.get() if online_kernel is not None else kernel

        if code is None:
            action = "generate:initial"
        else:
            action, _q = planner.choose_action(belief, gen_left, crit_used, ver_left)

        if action == "bail_out":
            res.final_action = "bail"; break

        if action == "verify":
            ok, info = verify_fn(code, problem)
            y_now = 1 if ok else 0
            if online_kernel is not None and prev_Y is not None:
                online_kernel.update(prev_Y, y_now)
                planner = planner_factory()  # re-solve with updated kernel
            prev_Y = y_now
            res.n_full_tests += 1; res.total_cost += costs.c_ver
            ver_left -= 1
            res.actions.append({"step": step, "action": "verify", "ok": ok, "info": info})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            belief = 0.05
            feedback = f"Hidden tests failed ({info})."
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            if not cached_critics:
                cached_critics = critics_fn(code, problem, llm_client=client)
            passed = cached_critics.get(cn, False)
            res.n_critic_runs += 1; res.total_cost += costs.c_critic
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used = crit_used | frozenset([cn])
            res.actions.append({"step": step, "action": action,
                                "passed": passed, "b": belief})
        elif action.startswith("generate"):
            prompt = prompt_fn(problem, prev_code=code, feedback=feedback)
            code, pt, ct, dollar = llm_generate(client, prompt)
            res.n_llm_calls += 1; res.completion_tokens += ct; res.api_cost_usd += dollar
            res.total_cost += costs.c_gen
            gen_left -= 1
            belief = kernel_update(belief, active_kernel)
            crit_used = frozenset()
            cached_critics = {}
            res.actions.append({"step": step, "action": "generate", "b": belief})
        step += 1

    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    return res


# ============================================================================
# Load split + theta + (optional) transition kernel
# ============================================================================
def load_split(gen_dir: Path) -> tuple[list[str], list[str]]:
    j = json.loads((gen_dir / "split.json").read_text())
    return [str(x) for x in j["train_ids"]], [str(x) for x in j["test_ids"]]


def load_theta(gen_dir: Path) -> tuple[dict, float]:
    j = json.loads((gen_dir / "likelihood_tables.json").read_text())
    out = {}
    for n, lk in j["critic_likelihoods"].items():
        out[n] = {
            "p_pass_y1": lk.get("P_pass_given_Y1", lk.get("p_pass_y1")),
            "p_pass_y0": lk.get("P_pass_given_Y0", lk.get("p_pass_y0")),
        }
    return out, j.get("prior_Y1", 0.5)


# Note: DEFAULT_KERNEL, kernel_update, OnlineKernelCalibration moved to
# _common/kernel.py. Now shared with iter/refine.py; the abbo bridge
# (run_codecontests_full.py et al.) is deferred until abbo is retired —
# see DEVELOPMENT_PROCESS.md. load_kernel() was folded into
# resolve_kernel(gen_dir, mode) in the same module.


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir", required=True, type=Path)
    p.add_argument("--benchmark", required=True,
                   help="lcb_easy/lcb_medium/lcb_hard/mbpp/humaneval")
    p.add_argument("--generator", required=True,
                   help="Generator slug (subdir of --src-dir); model is read from "
                        "ABBO_LLM_MODEL env var.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--variants", default="all",
                   help="Comma-separated subset; 'fitted' = greedy_fitted+dp_fitted.")
    p.add_argument("--n-test", type=int, default=0,
                   help="Optional cap on number of test instances (0 = all).")
    p.add_argument("--hand-theta-from", default="")
    p.add_argument("--kernel-mode", default="measured",
                   choices=["measured", "online", "hardcoded"],
                   help="Transition kernel source. "
                        "'measured' (default): load <gen>/transition_kernel.json if exists, "
                        "else fall back to hardcoded. "
                        "'online': start from measured (or hardcoded) and update via "
                        "Beta-Binomial after each verify. "
                        "'hardcoded': always use the abbo default (0.50, 0.05).")
    args = p.parse_args()

    gen_dir = (args.src_dir / args.generator).resolve()
    train_ids, test_ids = load_split(gen_dir)
    fitted_theta, prior_y1 = load_theta(gen_dir)
    if args.hand_theta_from:
        hj = json.loads(Path(args.hand_theta_from).read_text())
        hand_theta = {n: {"p_pass_y1": lk.get("P_pass_given_Y1", lk.get("p_pass_y1")),
                          "p_pass_y0": lk.get("P_pass_given_Y0", lk.get("p_pass_y0"))}
                      for n, lk in hj["critic_likelihoods"].items()}
    else:
        hand_theta = fitted_theta

    if args.n_test > 0:
        test_ids = test_ids[:args.n_test]

    log.info("Benchmark=%s  generator=%s  model=%s", args.benchmark, args.generator,
             LLM_MODEL)
    log.info("Train=%d  Test=%d  prior_Y1=%.3f", len(train_ids), len(test_ids), prior_y1)

    # Resolve variants filter
    all_variants = ["simple", "best_of_3", "threshold_L0", "threshold_L2", "threshold_L3",
                    "fixed_pipeline", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted"]
    if args.variants == "all":
        wanted = set(all_variants)
    elif args.variants == "fitted":
        wanted = {"greedy_fitted", "dp_fitted"}
    else:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        unknown = wanted - set(all_variants)
        if unknown:
            raise SystemExit(f"unknown variants: {sorted(unknown)}. Valid: {all_variants}")
    wanted.add("simple")  # baseline for delta_vs_simple
    log.info("Variants: %s", sorted(wanted))

    # LLM client (re-use lcb_calibrate's factory; works for OpenRouter)
    from calibration import lcb
    client = lcb._make_client()

    # Benchmark dispatcher
    load_fn, prompt_fn, critics_fn, verify_fn = _benchmark_loader(args.benchmark)
    problems = load_fn()
    # Index problems by id; we accept either "question_id" or "instance_id"
    by_id = {}
    for prob in problems:
        for k in ("question_id", "instance_id", "task_id"):
            if k in prob:
                by_id[str(prob[k])] = prob; break

    # Resolve transition kernel via _common.kernel. The third return value
    # (the OnlineKernelCalibration instance) is discarded here because each
    # per-variant branch below builds its own per-task estimator — preserving
    # the existing reset-per-task semantics.
    kernel, kernel_src, _ = resolve_kernel(gen_dir, args.kernel_mode)
    log.info("transition kernel: source=%s, p_fix=%.3f, p_break=%.3f",
             kernel_src, kernel["p_fix_broken"], kernel["p_break_correct"])

    # Pre-solve DP planners (DPPlanner expects abbo's AgentCostConfig)
    costs = Costs()
    from abbo.realworld.agents.simple_agent import AgentCostConfig as AbboCosts
    abbo_costs = AbboCosts(c_llm_call=costs.c_gen, c_critic_test=costs.c_critic,
                            c_full_test=costs.c_ver, reward=costs.reward)

    def _make_dp(theta_dict: dict, current_kernel: dict) -> DPPlanner:
        p = DPPlanner(abbo_costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                      critic_likelihoods=theta_dict,
                      transition_kernel=current_kernel)
        p.solve()
        return p

    # For online mode, planner_factory builds a fresh DP with current online kernel
    # For measured/hardcoded mode, planner is pre-solved once
    if args.kernel_mode == "online":
        log.info("online kernel mode: DP will be re-solved after each verify")

    # Output container
    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {"results": {}, "n_train": len(train_ids), "n_test": len(test_ids),
             "benchmark": args.benchmark, "generator": args.generator,
             "llm_model": LLM_MODEL, "prior_Y1": prior_y1,
             "split_seed": SPLIT_SEED, "fitted_theta": fitted_theta,
             "transition_kernel": kernel, "kernel_source": kernel_src,
             "kernel_mode": args.kernel_mode,
             "costs": {"c_gen": costs.c_gen, "c_critic": costs.c_critic,
                       "c_ver": costs.c_ver, "reward": costs.reward}}

    # Resume support: if output exists, load and skip done (task,variant) pairs
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        state["results"].update(existing.get("results", {}))
        log.info("resuming with %d already-done (task,variant) pairs",
                 len(state["results"]))

    # Main loop
    t0 = time.perf_counter()
    for i, tid in enumerate(test_ids):
        prob = by_id.get(tid)
        if prob is None:
            log.warning("test_id %s not found in current dataset — skipping", tid)
            continue
        elapsed = (time.perf_counter() - t0) / 60.0
        log.info("[%d/%d] %s  elapsed=%.1fmin", i+1, len(test_ids), tid, elapsed)
        for v_name in sorted(wanted):
            key = f"{tid}|{v_name}"
            if key in state["results"]:
                continue
            try:
                if v_name == "simple":
                    r = run_simple(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs)
                elif v_name == "best_of_3":
                    r = run_best_of_n(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs, n=3)
                elif v_name == "threshold_L0":
                    r = run_threshold(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs,
                                      ["L0_syntax"], "threshold_L0")
                elif v_name == "threshold_L2":
                    r = run_threshold(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs,
                                      ["L0_syntax", "L1_lint", "L2_public_tests"], "threshold_L2")
                elif v_name == "threshold_L3":
                    r = run_threshold(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs,
                                      ["L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"],
                                      "threshold_L3")
                elif v_name == "fixed_pipeline":
                    r = run_fixed_pipeline(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs)
                elif v_name == "greedy_hand":
                    ok_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                                  if args.kernel_mode == "online" else None)
                    r = run_greedy(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs,
                                   hand_theta, "hand", kernel, online_kernel=ok_kernel)
                elif v_name == "greedy_fitted":
                    ok_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                                  if args.kernel_mode == "online" else None)
                    r = run_greedy(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs,
                                   fitted_theta, "fitted", kernel, online_kernel=ok_kernel)
                elif v_name == "dp_hand":
                    ok_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                                  if args.kernel_mode == "online" else None)
                    def _factory_hand(k=kernel, ok=ok_kernel):
                        return _make_dp(hand_theta, ok.get() if ok else k)
                    r = run_dp(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs,
                               _factory_hand, hand_theta, "hand", kernel,
                               online_kernel=ok_kernel)
                elif v_name == "dp_fitted":
                    ok_kernel = (OnlineKernelCalibration(init_kernel=kernel)
                                  if args.kernel_mode == "online" else None)
                    def _factory_fitted(k=kernel, ok=ok_kernel):
                        return _make_dp(fitted_theta, ok.get() if ok else k)
                    r = run_dp(tid, prob, prompt_fn, critics_fn, verify_fn, client, costs,
                               _factory_fitted, fitted_theta, "fitted", kernel,
                               online_kernel=ok_kernel)
                else:
                    log.warning("unknown variant: %s", v_name); continue
                state["results"][key] = r.to_dict()
                # Save incrementally so kills don't lose progress
                args.output.write_text(json.dumps(state, indent=2))
                log.info("  %-16s fix=%s cost=%.1f llm=%d crit=%d wc=N/A",
                         v_name, r.fixed, r.total_cost, r.n_llm_calls, r.n_critic_runs)
            except Exception as e:
                log.error("variant %s failed on %s: %s", v_name, tid, e)
                continue

    # Aggregate per variant
    from collections import defaultdict
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for k, v in state["results"].items():
        by_variant[v["variant"]].append(v)
    summaries: list[dict] = []
    av_util = None
    for v_name in all_variants:
        rs = by_variant.get(v_name, [])
        if not rs: continue
        n = len(rs)
        fix_rate = sum(1 for r in rs if r["fixed"]) / n
        mean_cost = sum(r["total_cost"] for r in rs) / n
        mean_util = costs.reward * fix_rate - mean_cost
        if v_name == "simple":
            av_util = mean_util
        entry = {"policy": v_name, "n_episodes": n,
                 "mean_utility": round(mean_util, 4), "pass_rate": round(fix_rate, 4),
                 "fix_rate": round(fix_rate, 4), "mean_cost": round(mean_cost, 4)}
        if av_util is not None:
            entry["delta_vs_simple"] = round(mean_util - av_util, 4)
        summaries.append(entry)
        print(f"  {v_name:<20} n={n}  fix={fix_rate:.3f}  cost={mean_cost:7.3f}  "
              f"Ū={mean_util:8.3f}"
              + (f"  Δ={mean_util - av_util:+8.3f}" if av_util is not None else ""))

    state["summaries"] = summaries
    args.output.write_text(json.dumps(state, indent=2))
    log.info("saved → %s", args.output)


if __name__ == "__main__":
    main()
