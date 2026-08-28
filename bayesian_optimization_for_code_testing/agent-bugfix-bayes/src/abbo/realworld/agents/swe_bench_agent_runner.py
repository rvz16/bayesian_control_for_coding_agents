"""End-to-end Bayesian POMDP agent runner on SWE-bench Lite.

Wires an LLM `generate` action into the decision loop.  The LLM receives the
GitHub issue text + relevant file contents and produces SEARCH/REPLACE blocks
that are applied inside a long-lived Docker container via base64 file writes.

Why SEARCH/REPLACE over unified diff:
    LLMs reliably produce exact-text matches; unified diffs require correct
    line-number offsets which LLMs frequently get wrong on first attempt.
    Agentless and SWE-agent use the same pattern.

Critics (syntax, lint, early, mid) and `verify` are the Docker-based checks
from swe_bench.py.  Between generate attempts the container is reset to
base_commit + test_patch via prepare_repo(apply_fix=False).

For batch runs on a held-out set, use scripts/run_swebench_full.py directly
(resume-safe, persists JSON after every pair).  This module provides the
same logic as importable functions for test suites.

Strategies:
    simple        — always apply patch, verify, retry N times (baseline)
    greedy_hand   — Bayesian Greedy + hand-tuned SWE_CRITIC_LIKELIHOODS
    greedy_fitted — Bayesian Greedy + calibrated theta_hat
    dp_hand       — DP-optimal + hand-tuned theta
    dp_fitted     — DP-optimal + fitted theta_hat  (our method)

Recommended models via env:
    gpt-oss-20b   ABBO_LLM_MODEL=openai/gpt-oss-20b:free    ABBO_LLM_PROVIDER=openrouter
    qwen3-coder   ABBO_LLM_MODEL=qwen/qwen-2.5-coder-32b-instruct  ABBO_LLM_PROVIDER=openrouter
                  ABBO_LLM_MODEL=qwen2.5-coder:32b           ABBO_LLM_PROVIDER=ollama
"""

from __future__ import annotations

import base64
import math
import re
import time
from dataclasses import dataclass, field

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.llm_provider import LLMConfig, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.swe_bench import (
    SWE_CRITIC_LIKELIHOODS,
    SWE_CRITIC_NAMES,
    _exec,
    changed_files_from_patch,
    get_ftp,
    get_instance,
    prepare_repo,
    pull_image,
    run_critic,
    run_full_test,
    start_container,
    stop_container,
)


# Knobs (must mirror scripts/run_swebench_full.py for fair comparison).
ISSUE_CHAR_CAP = 32000   # was 4000 — long issues lost the bug-relevant section
FILE_CHAR_CAP  = 32000   # was 6000 — large files (e.g. Django models.py) were
                         # shown to the LLM only partially, breaking patches.


# ---------------------------------------------------------------------------
# Prompt + patch format  (SEARCH/REPLACE — same as scripts/run_swebench_full.py)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are a software engineer fixing a bug in a Python repository.

Issue:
{issue}

Files you may need to modify (current contents shown below):
{files_block}

Produce one or more SEARCH/REPLACE blocks that fix the bug. EACH BLOCK MUST
use the ACTUAL FILE PATH (the same path shown in the "### " heading above the
file content), not a placeholder like "path/to/file.py". For example, if the
file shown is "### django/db/backends/base/schema.py", the block must say
<<<<<<< SEARCH django/db/backends/base/schema.py — not <<<<<<< SEARCH
path/to/file.py.

The block format:

<<<<<<< SEARCH <real file path here, e.g. django/db/backends/base/schema.py>
exact lines to find
(must match file contents byte-for-byte including indentation)
=======
exact replacement lines
>>>>>>> REPLACE

Return ONLY the SEARCH/REPLACE blocks (no explanation, no markdown fence
around the whole thing). Keep blocks small and targeted; one block per
change-site."""


SR_BLOCK_RE = re.compile(
    r"<<<<<<<\s*SEARCH\s+([^\n]+)\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)

# Generator transition priors (belief update when `generate:arm` fires).
# p01 = P(correct | was broken),  p10 = P(broken | was correct).
GENERATOR_TRANSITIONS = {"p01": 0.13, "p10": 0.06}   # gpt5-mini SWE-Verified lit prior


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SWEAgentRunResult:
    instance_id: str
    variant: str
    fixed: bool = False
    total_cost: float = 0.0
    wall_clock: float = 0.0
    n_llm_calls: int = 0
    n_critic_runs: int = 0
    n_full_tests: int = 0
    n_patch_apply_fails: int = 0
    completion_tokens: int = 0
    final_action: str = ""
    actions: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def get_files_block(cname: str, instance: dict, max_chars: int = FILE_CHAR_CAP) -> str:
    """Read files touched by the gold patch (scaffolding, not the answer —
    localization step tells the LLM *which* files are relevant)."""
    try:
        paths = changed_files_from_patch(instance.get("patch") or "")[:3]
    except Exception:
        paths = []
    parts = []
    for p in paths:
        if not p:
            continue
        # NB: was `head -200` (~20K chars on 100-char lines). For files where
        # the bug lives past line 200 (Django models.py, pandas frame.py)
        # the LLM never saw the relevant code. Rely on FILE_CHAR_CAP instead.
        r = _exec(cname, f"cat /testbed/{p} 2>&1", timeout=30)
        body = (r.stdout or "")[:max_chars]
        parts.append(f"### {p}\n```python\n{body}\n```")
    return "\n\n".join(parts) if parts else "(no files identified)"


def _extract_sr_blocks(llm_text: str) -> list[tuple[str, str, str]]:
    """Returns list of (file_path, search_text, replace_text)."""
    return [
        (m.group(1).strip(), m.group(2), m.group(3))
        for m in SR_BLOCK_RE.finditer(llm_text or "")
    ]


def apply_llm_patch(cname: str, llm_text: str) -> tuple[bool, int, int]:
    """Apply SEARCH/REPLACE blocks to /testbed via base64 file writes.

    Returns (any_applied, n_blocks_attempted, n_blocks_applied).
    One malformed block does not abort the rest.
    """
    blocks = _extract_sr_blocks(llm_text)
    if not blocks:
        return False, 0, 0
    n_applied = 0
    for path, search, replace in blocks:
        try:
            if not path or not path.endswith(".py"):
                continue
            r = _exec(cname, f"cat /testbed/{path}", timeout=20)
            if r.returncode != 0 or not r.stdout:
                continue
            content = r.stdout
            if not search or search not in content:
                continue
            new_content = content.replace(search, replace or "", 1)
            b64 = base64.b64encode(new_content.encode("utf-8")).decode("ascii")
            wr = _exec(cname, f"echo '{b64}' | base64 -d > /testbed/{path}", timeout=20)
            if wr.returncode == 0:
                n_applied += 1
        except Exception:
            continue
    return n_applied > 0, len(blocks), n_applied


def reset_repo(cname: str, instance: dict) -> None:
    """Reset to base_commit + test_patch (no LLM patch)."""
    prepare_repo(cname, instance, apply_fix=False)


# ---------------------------------------------------------------------------
# Variant 1: Simple (always apply → verify → retry)
# ---------------------------------------------------------------------------

def run_simple(
    instance_id: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    n_retries: int = 2,
) -> SWEAgentRunResult:
    res = SWEAgentRunResult(instance_id=instance_id, variant="simple")
    start = time.perf_counter()
    inst = get_instance(instance_id)
    issue = inst["problem_statement"][:ISSUE_CHAR_CAP]

    pull_image(instance_id, verbose=True)
    cname = start_container(instance_id)
    try:
        reset_repo(cname, inst)
        for attempt in range(n_retries):
            files = get_files_block(cname, inst)
            prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
            resp = call_llm_or_raise(prompt, llm_config)
            res.n_llm_calls += 1
            res.total_cost += cost_config.c_llm_call
            res.completion_tokens += resp.completion_tokens

            applied, n_blocks, n_ok = apply_llm_patch(cname, resp.text)
            if not applied:
                res.n_patch_apply_fails += 1
                res.actions.append({"step": attempt, "n_blocks": n_blocks,
                                    "n_applied": n_ok, "applied": False})
                reset_repo(cname, inst)
                continue

            ok, _ = run_full_test(cname, inst)
            res.n_full_tests += 1
            res.total_cost += cost_config.c_full_test
            res.actions.append({"step": attempt, "n_blocks": n_blocks,
                                 "n_applied": n_ok, "applied": True, "verify_pass": ok})
            if ok:
                res.fixed = True
                res.final_action = "verify_pass"
                break
            reset_repo(cname, inst)

        res.final_action = res.final_action or "exhausted"
    finally:
        stop_container(cname)
    res.wall_clock = time.perf_counter() - start
    return res


# ---------------------------------------------------------------------------
# Variant 2/3: Greedy with theta
# ---------------------------------------------------------------------------

def _q_one_step_critic(b, c, theta, costs):
    lk = theta[c]
    p = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)
    bp = bayes_update(b, c, True, likelihoods=theta)
    bf = bayes_update(b, c, False, likelihoods=theta)
    return (-costs.c_critic_test
            + p     * max(0.0, -costs.c_full_test + bp * costs.reward)
            + (1-p) * max(0.0, -costs.c_full_test + bf * costs.reward))


def run_greedy(
    instance_id: str,
    theta: dict,
    theta_label: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    max_generators: int = 2,
    prior: float = 0.5,
) -> SWEAgentRunResult:
    res = SWEAgentRunResult(instance_id=instance_id, variant=f"greedy_{theta_label}")
    start = time.perf_counter()
    inst = get_instance(instance_id)
    issue = inst["problem_statement"][:ISSUE_CHAR_CAP]

    pull_image(instance_id, verbose=True)
    cname = start_container(instance_id)
    try:
        reset_repo(cname, inst)
        belief = prior
        gen_left = max_generators
        crit_used: set[str] = set()
        step = 0

        while step < 10:
            Q_bail = 0.0
            Q_verify = -cost_config.c_full_test + belief * cost_config.reward
            Q_critics = {c: _q_one_step_critic(belief, c, theta, cost_config)
                         for c in theta if c not in crit_used}
            best_c, best_q = (
                max(Q_critics.items(), key=lambda x: x[1])
                if Q_critics else (None, -math.inf)
            )
            Q_gen = -math.inf
            if gen_left > 0:
                b_after = belief * (1 - GENERATOR_TRANSITIONS["p10"]) + \
                          (1 - belief) * GENERATOR_TRANSITIONS["p01"]
                Q_gen = -cost_config.c_llm_call - cost_config.c_full_test + \
                        b_after * cost_config.reward

            choices = [("bail", Q_bail), ("verify", Q_verify)]
            if best_c:
                choices.append((f"critic:{best_c}", best_q))
            if gen_left > 0:
                choices.append(("generate", Q_gen))
            action, _q = max(choices, key=lambda x: x[1])

            if action == "bail":
                res.final_action = "bail"
                break
            if action == "verify":
                ok, _ = run_full_test(cname, inst)
                res.n_full_tests += 1
                res.total_cost += cost_config.c_full_test
                res.actions.append({"step": step, "action": "verify", "ok": ok, "b": belief})
                if ok:
                    res.fixed = True
                    res.final_action = "verify_pass"
                    break
                belief = 0.05
                reset_repo(cname, inst)
                crit_used = set()
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(cname, cn, inst)
                res.n_critic_runs += 1
                res.total_cost += cost_config.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used.add(cn)
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
            else:  # generate
                files = get_files_block(cname, inst)
                prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
                resp = call_llm_or_raise(prompt, llm_config)
                res.n_llm_calls += 1
                res.total_cost += cost_config.c_llm_call
                res.completion_tokens += resp.completion_tokens
                applied, n_blocks, n_ok = apply_llm_patch(cname, resp.text)
                if not applied:
                    res.n_patch_apply_fails += 1
                    reset_repo(cname, inst)
                gen_left -= 1
                belief = (belief * (1 - GENERATOR_TRANSITIONS["p10"]) +
                          (1 - belief) * GENERATOR_TRANSITIONS["p01"])
                crit_used = set()
                res.actions.append({"step": step, "action": "generate",
                                    "applied": applied, "b": belief})
            step += 1

        res.final_action = res.final_action or "exhausted"
    finally:
        stop_container(cname)
    res.wall_clock = time.perf_counter() - start
    return res


# ---------------------------------------------------------------------------
# Variant 4/5: DP with theta
# ---------------------------------------------------------------------------

def run_dp(
    instance_id: str,
    theta: dict,
    theta_label: str,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig,
    max_generators: int = 2,
    max_verifications: int = 1,
    prior: float = 0.5,
    planner: DPPlanner | None = None,
) -> SWEAgentRunResult:
    res = SWEAgentRunResult(instance_id=instance_id, variant=f"dp_{theta_label}")
    start = time.perf_counter()
    inst = get_instance(instance_id)
    issue = inst["problem_statement"][:ISSUE_CHAR_CAP]

    if planner is None:
        planner = DPPlanner(cost_config, max_generators, max_verifications,
                            critic_likelihoods=theta)
        planner.solve()

    pull_image(instance_id, verbose=True)
    cname = start_container(instance_id)
    try:
        reset_repo(cname, inst)
        belief = prior
        gen_left = max_generators
        ver_left = max_verifications
        crit_used: frozenset[str] = frozenset()
        step = 0

        while step < 12:
            action, _q = planner.choose_action(belief, gen_left, crit_used, ver_left)

            if action == "bail_out":
                res.final_action = "bail"
                break
            if action == "verify":
                ok, _ = run_full_test(cname, inst)
                res.n_full_tests += 1
                res.total_cost += cost_config.c_full_test
                ver_left -= 1
                res.actions.append({"step": step, "action": "verify",
                                    "ok": ok, "b": belief})
                if ok:
                    res.fixed = True
                    res.final_action = "verify_pass"
                    break
                belief = 0.05
                reset_repo(cname, inst)
                crit_used = frozenset()
            elif action.startswith("critic:"):
                cn = action.split(":", 1)[1]
                passed, _ = run_critic(cname, cn, inst)
                res.n_critic_runs += 1
                res.total_cost += cost_config.c_critic_test
                belief = bayes_update(belief, cn, passed, likelihoods=theta)
                crit_used = crit_used | frozenset([cn])
                res.actions.append({"step": step, "action": action,
                                    "passed": passed, "b": belief})
            elif action.startswith("generate:"):
                files = get_files_block(cname, inst)
                prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
                resp = call_llm_or_raise(prompt, llm_config)
                res.n_llm_calls += 1
                res.total_cost += cost_config.c_llm_call
                res.completion_tokens += resp.completion_tokens
                applied, n_blocks, n_ok = apply_llm_patch(cname, resp.text)
                if not applied:
                    res.n_patch_apply_fails += 1
                    reset_repo(cname, inst)
                gen_left -= 1
                belief = (belief * (1 - GENERATOR_TRANSITIONS["p10"]) +
                          (1 - belief) * GENERATOR_TRANSITIONS["p01"])
                crit_used = frozenset()
                res.actions.append({"step": step, "action": "generate",
                                    "applied": applied, "b": belief})
            step += 1

        res.final_action = res.final_action or "exhausted"
    finally:
        stop_container(cname)
    res.wall_clock = time.perf_counter() - start
    return res


# ---------------------------------------------------------------------------
# Top-level grid runner
# ---------------------------------------------------------------------------

def run_grid(
    instance_ids: list[str],
    fitted_theta: dict,
    llm_config: LLMConfig,
    cost_config: AgentCostConfig | None = None,
    variants: tuple[str, ...] = (
        "simple", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted",
    ),
    prior: float = 0.5,
    verbose: bool = True,
) -> dict[str, list[SWEAgentRunResult]]:
    if cost_config is None:
        cost_config = AgentCostConfig()

    dp_hand = DPPlanner(cost_config, 2, 1, critic_likelihoods=SWE_CRITIC_LIKELIHOODS)
    dp_hand.solve()
    dp_fitted = DPPlanner(cost_config, 2, 1, critic_likelihoods=fitted_theta)
    dp_fitted.solve()

    results: dict[str, list[SWEAgentRunResult]] = {v: [] for v in variants}
    for i, iid in enumerate(instance_ids):
        if verbose:
            print(f"\n[{i+1}/{len(instance_ids)}] {iid}")
        for v in variants:
            try:
                if v == "simple":
                    r = run_simple(iid, llm_config, cost_config)
                elif v == "greedy_hand":
                    r = run_greedy(iid, SWE_CRITIC_LIKELIHOODS, "hand",
                                   llm_config, cost_config, prior=prior)
                elif v == "greedy_fitted":
                    r = run_greedy(iid, fitted_theta, "fitted",
                                   llm_config, cost_config, prior=prior)
                elif v == "dp_hand":
                    r = run_dp(iid, SWE_CRITIC_LIKELIHOODS, "hand",
                               llm_config, cost_config, prior=prior, planner=dp_hand)
                elif v == "dp_fitted":
                    r = run_dp(iid, fitted_theta, "fitted",
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
            except Exception as e:
                if verbose:
                    print(f"  [error] {v} on {iid}: {type(e).__name__}: {e}")
    return results


def aggregate(results: dict[str, list[SWEAgentRunResult]]) -> dict[str, dict]:
    out = {}
    for v, rs in results.items():
        n = len(rs)
        if not n:
            continue
        out[v] = {
            "n_instances": n,
            "fix_rate": sum(1 for r in rs if r.fixed) / n,
            "avg_cost": sum(r.total_cost for r in rs) / n,
            "avg_llm_calls": sum(r.n_llm_calls for r in rs) / n,
            "avg_critic_runs": sum(r.n_critic_runs for r in rs) / n,
            "avg_full_tests": sum(r.n_full_tests for r in rs) / n,
            "avg_patch_apply_fails": sum(r.n_patch_apply_fails for r in rs) / n,
            "avg_completion_tokens": sum(r.completion_tokens for r in rs) / n,
            "avg_wall_clock": sum(r.wall_clock for r in rs) / n,
        }
    return out


def format_summary(agg: dict[str, dict]) -> str:
    cols = ["fix_rate", "avg_cost", "avg_llm_calls", "avg_critic_runs",
            "avg_full_tests", "avg_patch_apply_fails", "avg_completion_tokens"]
    header = f"{'variant':<20} {'n':>4} " + " ".join(f"{c:>14}" for c in cols)
    lines = [header, "-" * len(header)]
    for v, m in agg.items():
        row = f"{v:<20} {m['n_instances']:>4} "
        for c in cols:
            row += f"{m.get(c, 0.0):>14.3f} "
        lines.append(row)
    return "\n".join(lines)
