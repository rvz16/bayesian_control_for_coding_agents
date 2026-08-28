#!/usr/bin/env python
"""End-to-end agent run on CodeContests held-out (82 tasks).

Train split: same 30 tasks as test_codecontests_calibration.py used to fit
fitted_theta. Held-out: remaining ~82 tasks for the agent comparison.

Resilience: saves results after every (task, variant) pair. Re-running this
script skips already-completed pairs, so it's safe to interrupt + resume.

Usage:
    python scripts/run_codecontests_full.py
    python scripts/run_codecontests_full.py --model qwen/qwen3-coder \\
        --results sim_results/codecontests_full__qwen3_coder.json

See run_humaneval_full.py module docstring for paper generator ↔ model IDs
and policy naming caveats (bugfix variants vs orchestration replay).
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES,
    get_solution_pool, get_test_cases, get_metadata,
    list_task_ids, run_critic, run_full_test,
)
from abbo.realworld.agents.llm_provider import build_llm_config_from_env, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig

# ---- Config ----
SPLIT_SEED = 42
N_TRAIN = 30        # tasks used by calibration to fit fitted_theta
PRIOR = 0.5
MAX_GENERATORS = 3
MAX_VERIFICATIONS = 2
DEFAULT_LLM_MODEL = "openai/gpt-oss-20b:free"
# Backward compat for scripts that import LLM_MODEL from this module
LLM_MODEL = DEFAULT_LLM_MODEL

DEFAULT_VARIANTS = ("simple", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted")

# Cached fitted theta from the n=146 CodeContests calibration run
FITTED_THETA = {
    "critic_early": {"p_pass_y1": 0.9687500000000001, "p_pass_y0": 0.375},
    "critic_lint":  {"p_pass_y1": 0.728125,           "p_pass_y0": 0.678125},
    "critic_mid":   {"p_pass_y1": 0.9687500000000001, "p_pass_y0": 0.125},
    "critic_syntax":{"p_pass_y1": 0.9803571428571428, "p_pass_y0": 0.94375},
}

# ---- Prompt + extraction ----
PROMPT_TEMPLATE = """You are a competitive programming assistant. The Python solution below has a bug that makes it fail at least one test case. Return the COMPLETE corrected Python program — only the code, no explanation, no markdown fences.

Buggy code:
```python
{source_code}
```

Sample test cases (input → expected output):
{test_examples}

Recent failed run output:
```
{test_output}
```

Return only the corrected Python program."""

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL)


def extract_code(llm_text, fallback):
    if not isinstance(llm_text, str):
        return fallback
    m = CODE_FENCE_RE.search(llm_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    s = llm_text.strip()
    return s if s else fallback


def format_test_examples(task_id, n=2):
    tests = get_test_cases(task_id, max_tests=n)
    parts = []
    for i, (inp, exp) in enumerate(tests, 1):
        parts.append(f"Test {i}:\n  input:\n{inp[:300]}\n  expected output:\n{exp[:200]}")
    return "\n\n".join(parts)


def get_test_output(workdir, task_id):
    tests = get_test_cases(task_id, max_tests=1)
    if not tests:
        return "(no test cases)"
    src = workdir / "solution.py"
    # NB: timeout was 4s — Codeforces problems can have ≥2s time limits, and
    # cold-start Python + macOS scheduler jitter pushed correct solutions past
    # the 4s wall. Bumping to 10s reduces false-timeout noise in the LLM-facing
    # error trace (the trace feeds back into the next generate prompt).
    try:
        r = subprocess.run(
            [sys.executable, str(src)], input=tests[0][0],
            text=True, capture_output=True, timeout=10,
        )
        return f"stdout: {r.stdout[:300]}\nstderr: {r.stderr[:200]}\nexpected: {tests[0][1][:200]}"
    except Exception as e:
        return f"runtime error: {e}"


# ---- Result type ----
@dataclass
class Result:
    task_id: str
    variant: str
    cf_rating: int | None = None
    fixed: bool = False
    total_cost: float = 0.0
    wall_clock: float = 0.0
    n_llm_calls: int = 0
    n_critic_runs: int = 0
    n_full_tests: int = 0
    completion_tokens: int = 0
    final_action: str = ""
    actions: list = field(default_factory=list)


# ---- Variants ----
def run_simple(task_id, llm_cfg, costs, n_retries=3):
    res = Result(task_id=task_id, variant="simple",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        for attempt in range(n_retries):
            test_out = get_test_output(wd, task_id)
            prompt = PROMPT_TEMPLATE.format(
                source_code=current,
                test_examples=format_test_examples(task_id),
                test_output=test_out,
            )
            r = call_llm_or_raise(prompt, llm_cfg)
            res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
            res.completion_tokens += r.completion_tokens
            current = extract_code(r.text, current)
            sol.write_text(current)
            ok, _ = run_full_test(wd, task_id)
            res.n_full_tests += 1; res.total_cost += costs.c_full_test
            res.actions.append({"step": attempt, "verify_pass": ok})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
        if not res.fixed:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def _q_one_step_critic(b, c, theta, costs):
    lk = theta[c]
    p = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)
    bp = bayes_update(b, c, True, likelihoods=theta)
    bf = bayes_update(b, c, False, likelihoods=theta)
    return -costs.c_critic_test \
        + p     * max(0.0, -costs.c_full_test + bp * costs.reward) \
        + (1-p) * max(0.0, -costs.c_full_test + bf * costs.reward)


def run_greedy(task_id, theta, label, llm_cfg, costs, max_gen=3, prior=0.5):
    res = Result(task_id=task_id, variant=f"greedy_{label}",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        belief = prior; gen_left = max_gen
        crit_used: set[str] = set(); step = 0
        while step < 12:
            Q_bail = 0.0
            Q_verify = -costs.c_full_test + belief * costs.reward
            Q_critics = {c: _q_one_step_critic(belief, c, theta, costs)
                         for c in theta if c not in crit_used}
            best_c, best_q = (max(Q_critics.items(), key=lambda x: x[1])
                              if Q_critics else (None, -math.inf))
            Q_gen = -math.inf
            if gen_left > 0:
                b_after = belief * 0.95 + (1-belief) * 0.50
                Q_gen = -costs.c_llm_call - costs.c_full_test + b_after * costs.reward
            choices = [("bail", Q_bail), ("verify", Q_verify)]
            if best_c: choices.append((f"critic:{best_c}", best_q))
            if gen_left > 0: choices.append(("generate", Q_gen))
            action, _q = max(choices, key=lambda x: x[1])

            if action == "bail":
                res.final_action = "bail"; break
            if action == "verify":
                ok, _ = run_full_test(wd, task_id)
                res.n_full_tests += 1; res.total_cost += costs.c_full_test
                res.actions.append({"step": step, "action": "verify", "ok": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(wd, cn, task_id)
                res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used.add(cn)
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
            else:  # generate
                test_out = get_test_output(wd, task_id)
                prompt = PROMPT_TEMPLATE.format(
                    source_code=current,
                    test_examples=format_test_examples(task_id),
                    test_output=test_out,
                )
                r = call_llm_or_raise(prompt, llm_cfg)
                res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
                res.completion_tokens += r.completion_tokens
                current = extract_code(r.text, current)
                sol.write_text(current)
                gen_left -= 1
                belief = belief * 0.95 + (1-belief) * 0.50
                crit_used = set()
                res.actions.append({"step": step, "action": "generate", "b": belief})
            step += 1
        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_dp(task_id, theta, label, llm_cfg, costs, planner,
           max_gen=3, max_ver=2, prior=0.5):
    res = Result(task_id=task_id, variant=f"dp_{label}",
                 cf_rating=get_metadata(task_id).get("cf_rating"))
    start = time.perf_counter()
    incorrect = get_solution_pool(task_id, "incorrect")
    if not incorrect:
        res.final_action = "no_buggy_seed"
        res.wall_clock = time.perf_counter() - start
        return res
    buggy = incorrect[0]
    with tempfile.TemporaryDirectory() as tmp:
        wd = Path(tmp); sol = wd / "solution.py"
        sol.write_text(buggy); current = buggy
        belief = prior; gen_left = max_gen; ver_left = max_ver
        crit_used: frozenset[str] = frozenset(); step = 0
        while step < 16:
            action, _q = planner.choose_action(
                belief, gen_left, crit_used, ver_left,
            )
            if action == "bail_out":
                res.final_action = "bail"; break
            if action == "verify":
                ok, _ = run_full_test(wd, task_id)
                res.n_full_tests += 1; res.total_cost += costs.c_full_test
                ver_left -= 1
                res.actions.append({"step": step, "action": "verify", "ok": ok})
                if ok:
                    res.fixed = True; res.final_action = "verify_pass"; break
                belief = 0.05
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(wd, cn, task_id)
                res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used = crit_used | frozenset([cn])
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
            elif action.startswith("generate:"):
                test_out = get_test_output(wd, task_id)
                prompt = PROMPT_TEMPLATE.format(
                    source_code=current,
                    test_examples=format_test_examples(task_id),
                    test_output=test_out,
                )
                r = call_llm_or_raise(prompt, llm_cfg)
                res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
                res.completion_tokens += r.completion_tokens
                current = extract_code(r.text, current)
                sol.write_text(current)
                gen_left -= 1
                belief = belief * 0.95 + (1-belief) * 0.50
                crit_used = frozenset()
                res.actions.append({"step": step, "action": "generate", "b": belief})
            step += 1
        if not res.fixed and not res.final_action:
            res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


# ---- Resume-safe save/load ----
def load_existing(path):
    if not path.exists():
        return {"results": {}}
    with open(path) as f:
        return json.load(f)


def save_progress(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def serialize(r):
    return {
        "task_id": r.task_id, "variant": r.variant, "cf_rating": r.cf_rating,
        "fixed": r.fixed, "total_cost": r.total_cost, "wall_clock": r.wall_clock,
        "n_llm_calls": r.n_llm_calls, "n_critic_runs": r.n_critic_runs,
        "n_full_tests": r.n_full_tests,
        "completion_tokens": r.completion_tokens,
        "final_action": r.final_action, "actions": r.actions,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CodeContests held-out agent comparison (resume-safe).")
    p.add_argument(
        "--model",
        default=None,
        help="OpenAI-compatible model id (overrides ABBO_LLM_MODEL for this run).",
    )
    p.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Output JSON path (default: sim_results/codecontests_full_endtoend.json).",
    )
    p.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated subset of: simple,greedy_hand,greedy_fitted,dp_hand,dp_fitted",
    )
    return p.parse_args()


def main():
    args = parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    for v in variants:
        if v not in DEFAULT_VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}; allowed: {DEFAULT_VARIANTS}")
    results_path = args.results or (ROOT / "sim_results" / "codecontests_full_endtoend.json")
    llm_model = (args.model or "").strip() or DEFAULT_LLM_MODEL

    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    test_ids = all_ids[N_TRAIN:]
    print(f"Train: {len(train_ids)}  Held-out: {len(test_ids)}")

    state = load_existing(results_path)
    results = state.setdefault("results", {})

    costs = AgentCostConfig()
    dp_hand = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                        critic_likelihoods=CC_CRITIC_LIKELIHOODS); dp_hand.solve()
    dp_fitted = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                          critic_likelihoods=FITTED_THETA); dp_fitted.solve()

    llm_cfg = build_llm_config_from_env(
        default_provider="openrouter",
        default_model=llm_model,
        default_base_url="https://openrouter.ai/api",
        default_temperature=0.1,
        default_max_tokens=8192,  # was 2048; CC patches with CoT got truncated
        default_timeout=120,
    )
    if args.model:
        llm_cfg.model = args.model.strip()
    print(f"LLM provider={llm_cfg.provider} model={llm_cfg.model} base_url={llm_cfg.base_url}")

    total = len(test_ids) * len(variants)
    done = sum(1 for tid in test_ids for v in variants if results.get(f"{tid}|{v}"))
    print(f"\nResume: {done}/{total} (task, variant) pairs already done.\n")

    started = time.time()
    for i, tid in enumerate(test_ids):
        elapsed = time.time() - started
        rate = (i + 1) / max(0.001, elapsed)
        eta_min = (len(test_ids) - i - 1) / max(0.0001, rate) / 60
        m = get_metadata(tid)
        print(f"\n[{i+1}/{len(test_ids)}] task={tid}  cf={m.get('cf_rating')}  "
              f"diff={m.get('difficulty')}  elapsed={elapsed/60:.1f}min  "
              f"ETA={eta_min:.1f}min")
        for v in variants:
            key = f"{tid}|{v}"
            if results.get(key):
                continue
            try:
                if v == "simple":
                    r = run_simple(tid, llm_cfg, costs)
                elif v == "greedy_hand":
                    r = run_greedy(tid, CC_CRITIC_LIKELIHOODS, "hand",
                                   llm_cfg, costs, MAX_GENERATORS, PRIOR)
                elif v == "greedy_fitted":
                    r = run_greedy(tid, FITTED_THETA, "fitted",
                                   llm_cfg, costs, MAX_GENERATORS, PRIOR)
                elif v == "dp_hand":
                    r = run_dp(tid, CC_CRITIC_LIKELIHOODS, "hand",
                               llm_cfg, costs, dp_hand,
                               MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR)
                elif v == "dp_fitted":
                    r = run_dp(tid, FITTED_THETA, "fitted",
                               llm_cfg, costs, dp_fitted,
                               MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR)
                else:
                    continue
            except Exception as e:
                print(f"  [{v}] EXCEPTION: {e}")
                continue
            results[key] = serialize(r)
            tag = "OK" if r.fixed else "no"
            print(f"  {v:<16} fix={tag}  cost={r.total_cost:5.1f}  "
                  f"llm={r.n_llm_calls}  crit={r.n_critic_runs}  "
                  f"toks={r.completion_tokens}  wc={r.wall_clock:.1f}s  "
                  f"final={r.final_action}")
            save_progress(results_path, state)

    # Final aggregate
    print("\n=== Final aggregate ===")
    R = 100
    from collections import defaultdict
    by_v = defaultdict(list)
    for rec in results.values():
        by_v[rec["variant"]].append(rec)
    print(f"{'variant':<16} {'n':>4} {'fix%':>6} {'cost':>7} {'Ū_π':>8} {'Δ_π':>8}")
    print('-' * 55)
    if "simple" in by_v and by_v["simple"]:
        baseline = sum((R if r["fixed"] else 0) - r["total_cost"]
                       for r in by_v["simple"]) / len(by_v["simple"])
    else:
        baseline = 0.0
    for v in variants:
        rs = by_v.get(v, [])
        if not rs: continue
        n = len(rs)
        fix = sum(1 for r in rs if r["fixed"]) / n * 100
        c = sum(r["total_cost"] for r in rs) / n
        u = sum((R if r["fixed"] else 0) - r["total_cost"] for r in rs) / n
        d = u - baseline
        print(f"{v:<16} {n:>4} {fix:>5.1f}% {c:>7.2f} {u:>8.2f} {d:>+8.2f}")

    state["llm_model"] = llm_cfg.model
    state["fitted_theta"] = FITTED_THETA
    state["n_train"] = len(train_ids)
    state["n_test"] = len(test_ids)
    save_progress(results_path, state)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
