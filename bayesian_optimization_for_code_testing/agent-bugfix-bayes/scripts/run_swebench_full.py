#!/usr/bin/env python
"""End-to-end agent run on SWE-Bench Lite, mirroring run_codecontests_full.py.

Per instance:
  1. Pull image + start container (cached after first run)
  2. Reset to base + apply test_patch (Y=0 starting state)
  3. For each agent variant:
        a. Reset state again (clean slate per variant)
        b. Run agent loop: critic / verify / generate (LLM produces unified diff)
        c. Apply LLM patches via git apply inside container
        d. Verify = run FAIL_TO_PASS + PASS_TO_PASS

Resume-safe: saves after every (instance, variant) pair.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.llm_provider import build_llm_config_from_env, call_llm_or_raise
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.swe_bench import (
    SWE_CRITIC_LIKELIHOODS, SWE_INSTANCE_POOL, SWE_CRITIC_NAMES,
    _exec, _exec_stdin,
    changed_files_from_patch, get_instance,
    list_instance_ids, prepare_repo, pull_image,
    run_critic, run_full_test,
    start_container, stop_container,
)


# ---- Knobs ----
SPLIT_SEED = 42
N_TRAIN = 7              # same as test_swebench_calibration.py
PRIOR = 0.5
MAX_GENERATORS = 2       # SWE patches are big — keep budget small
MAX_VERIFICATIONS = 1
ISSUE_CHAR_CAP = 32000   # raise from 4000 — long issues were truncated
                         # below the bug-relevant section and the planner
                         # received noise; this is char count, not tokens.
MAX_OUTPUT_TOKENS = 8192 # SWE patches can run long; 2048 truncated some.
LLM_MODEL = os.environ.get("ABBO_LLM_MODEL", "openai/gpt-oss-20b:free")
_model_slug = LLM_MODEL.split("/")[-1].replace(":", "_").replace(".", "_")
# Output JSON path is bound to (dataset, model) inside main(); see
# results_path computation there.

VARIANTS = ("simple",
            "best_of_3",
            "threshold_L0", "threshold_L2", "threshold_L3",
            "fixed_pipeline",
            "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted")

# Stateless-baseline critic cascades. SWE-Bench has 4 critics:
#   critic_syntax (L0), critic_lint (L1-ish), critic_early (L2 = first
#   FAIL_TO_PASS test passes), critic_mid (all FAIL_TO_PASS tests pass).
# threshold_L3 in the paper uses an LLM-judge critic which SWE-Bench does
# not have; we use critic_mid (full FAIL_TO_PASS pass) as the strongest
# critic-stack stand-in for that level.
THRESHOLD_L0_CRITICS = ["critic_syntax"]
THRESHOLD_L2_CRITICS = ["critic_syntax", "critic_lint", "critic_early"]
THRESHOLD_L3_CRITICS = ["critic_syntax", "critic_lint", "critic_early", "critic_mid"]

# best_of_3 generates this many candidate patches before giving up.
BEST_OF_N = 3

# Cached fitted theta from the SWE calibration run (allure artifact 9e7fd0d7)
FITTED_THETA = {
    "critic_syntax": {"p_pass_y1": 0.914, "p_pass_y0": 0.864},
    "critic_lint":   {"p_pass_y1": 0.247, "p_pass_y0": 0.197},
    "critic_early":  {"p_pass_y1": 0.667, "p_pass_y0": 0.444},
    "critic_mid":    {"p_pass_y1": 0.667, "p_pass_y0": 0.222},
}

# Measured kernel placeholder (we have NO iter data for SWE yet).
# Fall back to literature-prior numbers from the supervisor deck (gpt5_mini
# SWE-Verified): P(fix|broken)=0.13, P(break|correct)=0.06.
SWE_KERNEL_LITERATURE = {"p_fix_broken": 0.13, "p_break_correct": 0.06}


PROMPT_TEMPLATE = """You are a software engineer fixing a bug in a Python repository.

Issue:
{issue}

Files you may need to modify (current contents shown below):
{files_block}

Produce one or more SEARCH/REPLACE blocks that fix the bug. EACH BLOCK MUST
use the ACTUAL FILE PATH (the same path shown in the "### " heading above
the file content), not a placeholder like "path/to/file.py". For example, if
the file shown is "### django/db/backends/base/schema.py", the block must say
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


# SEARCH/REPLACE block parser — tolerates 5-7 angle-bracket chars and case variations
SR_BLOCK_RE = re.compile(
    r"<{5,7}\s*SEARCH\s+([^\n]+)\n(.*?)\n={5,7}\n(.*?)\n>{5,7}\s*REPLACE",
    re.DOTALL | re.IGNORECASE,
)

# Unified-diff code-fence extractor (```diff … ``` or ```patch … ```)
_DIFF_FENCE_RE = re.compile(r"```(?:diff|patch)[^\n]*\n(.*?)```", re.DOTALL)

# Thinking-block stripper (qwen3-coder wraps CoT in <think>…</think>)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_sr_blocks(llm_text: str) -> list[tuple[str, str, str]]:
    """Returns list of (file_path, search_text, replace_text). Empty if none found."""
    return [(m.group(1).strip(), m.group(2), m.group(3))
            for m in SR_BLOCK_RE.finditer(llm_text)]


def _looks_like_diff(text: str) -> bool:
    return bool(re.search(r"^(?:---\s|\+\+\+\s|@@\s)", text, re.MULTILINE))


def _apply_unified_diff(cname: str, diff_text: str) -> bool:
    r = _exec_stdin(
        cname,
        "cd /testbed && git apply --ignore-whitespace --recount -",
        diff_text,
        timeout=30,
    )
    return r.returncode == 0


def get_files_block(cname: str, instance: dict, max_chars_per_file: int = 32000) -> str:
    """Cat the files touched by the gold patch (we tell the LLM which files
    to look at — that's a fair scaffolding, not the answer).

    Two earlier defaults were bugs:
    - `head -200` lost code past line 200 (Django models.py, pandas frame.py).
    - `max_chars_per_file=6000` truncated files even within the 200-line view.
    Both are now 32k chars (≈ 800-1000 lines of typical Python).
    """
    try:
        paths = changed_files_from_patch(instance.get("patch") or "")[:3]
    except Exception:
        paths = []
    parts = []
    for p in paths:
        if not isinstance(p, str) or not p:
            continue
        try:
            r = _exec(cname, f"cat /testbed/{p} 2>&1", timeout=30)
            body = (r.stdout or "")[:max_chars_per_file]
            parts.append(f"### {p}\n```python\n{body}\n```")
        except Exception:
            continue
    return "\n\n".join(parts) if parts else "(no files identified)"


_PLACEHOLDER_PATH_RE = re.compile(
    r"^(?:path/to/|<|\.\./|/tmp/|\$|YOUR_|<your_|<insert_|<file)",
    re.IGNORECASE,
)


def _resolve_placeholder_path(cname: str, search: str) -> str | None:
    """When the LLM gives a placeholder like 'path/to/file.py', try to figure
    out the real path inside /testbed by searching for a recognizable signature
    of the SEARCH block.

    Heuristic: take the first non-blank line of SEARCH that looks searchable
    (>= 8 chars, no leading '#', no leading shell metas) and `git grep -l` it.
    If exactly one .py file matches, use that. Otherwise give up.
    """
    if not search:
        return None
    # Pick a signature line from SEARCH. Prefer a line containing 'def ' or
    # 'class ' (rare globally, so very specific); fall back to the first
    # substantial non-trivial line.
    needle = None
    for line in search.splitlines():
        s = line.strip()
        if len(s) < 8:
            continue
        if "def " in s or "class " in s:
            needle = s.rstrip(":")
            break
    if needle is None:
        for line in search.splitlines():
            s = line.strip()
            if len(s) >= 12 and not s.startswith("#"):
                needle = s
                break
    if not needle:
        return None
    # git grep is fast; escape quotes for the shell.
    needle_esc = needle.replace("'", "'\\''")
    r = _exec(
        cname,
        f"cd /testbed && git grep -l --max-count=1 -F -- '{needle_esc}' "
        "-- '*.py' 2>/dev/null | head -5",
        timeout=20,
    )
    if r.returncode != 0:
        return None
    candidates = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()]
    candidates = [c for c in candidates if c.endswith(".py")]
    if len(candidates) == 1:
        return candidates[0]
    return None


def apply_llm_patch(cname: str, llm_text: str) -> tuple[bool, int, int]:
    """Apply LLM patch to /testbed.

    Tries SEARCH/REPLACE blocks first, then falls back to unified diff via
    git apply. Strips <think>…</think> blocks before parsing (qwen3-coder).

    Returns (any_applied, n_sr_blocks_found, n_sr_blocks_applied).
    """
    import base64
    llm_text = _strip_thinking(llm_text or "")

    # --- Method 1: SEARCH/REPLACE blocks ---
    try:
        blocks = extract_sr_blocks(llm_text)
    except Exception:
        blocks = []

    n_applied = 0
    for path, search, replace in blocks:
        try:
            # Detect placeholder paths (smaller models copy 'path/to/file.py'
            # verbatim from the prompt example) and try to resolve them by
            # searching for a signature of the SEARCH block in /testbed.
            if path and _PLACEHOLDER_PATH_RE.match(path):
                resolved = _resolve_placeholder_path(cname, search)
                if resolved:
                    path = resolved
            if not path or not path.endswith(".py"):
                continue
            r = _exec(cname, f"cat /testbed/{path}", timeout=20)
            if r.returncode != 0 or r.stdout is None:
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

    if n_applied > 0:
        return True, len(blocks), n_applied

    # --- Method 2: unified diff via git apply ---
    # Prefer explicit ```diff fences; fall back to raw text.
    diff_candidates = [m.group(1) for m in _DIFF_FENCE_RE.finditer(llm_text)]
    diff_candidates.append(llm_text)
    for candidate in diff_candidates:
        if _looks_like_diff(candidate) and _apply_unified_diff(cname, candidate):
            return True, len(blocks), 1

    return False, len(blocks), 0


def reset_repo_for_variant(cname: str, instance: dict) -> None:
    """Reset to base + test_patch (no fix). Run between variants."""
    prepare_repo(cname, instance, apply_fix=False)


# ---- Result ----
@dataclass
class Result:
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
    actions: list = field(default_factory=list)


# ---- Variants ----
def run_simple(instance, cname, llm_cfg, costs, n_retries=2):
    res = Result(instance_id=instance["instance_id"], variant="simple")
    start = time.perf_counter()
    issue = instance["problem_statement"][:ISSUE_CHAR_CAP]
    for attempt in range(n_retries):
        files = get_files_block(cname, instance)
        prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
        r = call_llm_or_raise(prompt, llm_cfg)
        res.n_llm_calls += 1
        res.total_cost += costs.c_llm_call
        res.completion_tokens += r.completion_tokens
        applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
        if not applied:
            res.n_patch_apply_fails += 1
            res.actions.append({"step": attempt, "n_blocks": n_blocks,
                                "n_applied": n_ok, "applied": False})
            # Reset for next attempt (the failed apply may have left state weird)
            reset_repo_for_variant(cname, instance)
            continue
        ok, _ = run_full_test(cname, instance)
        res.n_full_tests += 1
        res.total_cost += costs.c_full_test
        res.actions.append({"step": attempt, "n_blocks": n_blocks,
                            "n_applied": n_ok, "applied": True,
                            "verify_pass": ok})
        if ok:
            res.fixed = True
            res.final_action = "verify_pass"
            break
        # Failed verify → reset for next attempt
        reset_repo_for_variant(cname, instance)
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


# --------------------------------------------------------------------------
# Stateless paper baselines
# --------------------------------------------------------------------------

def _generate_patch_into_container(instance, cname, llm_cfg, costs, res, step: int):
    """Common: build prompt, LLM call, apply patch. Mutates res (counters,
    actions, completion_tokens). Returns (applied, n_blocks, n_ok)."""
    issue = instance["problem_statement"][:ISSUE_CHAR_CAP]
    files = get_files_block(cname, instance)
    prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
    r = call_llm_or_raise(prompt, llm_cfg)
    res.n_llm_calls += 1
    res.total_cost += costs.c_llm_call
    res.completion_tokens += r.completion_tokens
    applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
    if not applied:
        res.n_patch_apply_fails += 1
    res.actions.append({
        "step": step, "action": "generate",
        "n_blocks": n_blocks, "n_applied": n_ok, "applied": applied,
    })
    return applied, n_blocks, n_ok


def run_best_of_3(instance, cname, llm_cfg, costs, n: int = BEST_OF_N):
    """Generate N independent patches, verify each. fixed=True if ANY passes.

    Paper convention (slide 10): cost = N·C_gen + N·C_ver per instance.
    Reset between attempts so each generation is independent (the LLM sees the
    same starting state, like best-of-N sampling)."""
    res = Result(instance_id=instance["instance_id"], variant="best_of_3")
    start = time.perf_counter()
    for i in range(n):
        # Fresh state per sample
        reset_repo_for_variant(cname, instance)
        applied, _, _ = _generate_patch_into_container(
            instance, cname, llm_cfg, costs, res, step=2 * i,
        )
        if not applied:
            continue
        ok, _ = run_full_test(cname, instance)
        res.n_full_tests += 1
        res.total_cost += costs.c_full_test
        res.actions.append({"step": 2 * i + 1, "action": "verify", "ok": ok})
        if ok:
            res.fixed = True
            res.final_action = "verify_pass"
            break
    if not res.fixed:
        res.final_action = res.final_action or "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_threshold(instance, cname, llm_cfg, costs,
                  critic_names: list[str], label: str):
    """Generate → run critics in cascade order, bail on first FAIL → if all
    pass, verify. Paper's "gate(Cr_*)" stateless baseline.

    Per-instance budget: 1 LLM call + up to len(critic_names) critics + 0/1
    verify (verify happens only if all critics passed)."""
    res = Result(instance_id=instance["instance_id"], variant=label)
    start = time.perf_counter()
    reset_repo_for_variant(cname, instance)
    applied, _, _ = _generate_patch_into_container(
        instance, cname, llm_cfg, costs, res, step=0,
    )
    if not applied:
        res.final_action = "no_patch"
        res.wall_clock = time.perf_counter() - start
        return res
    for k, cn in enumerate(critic_names, start=1):
        passed, _ = run_critic(cname, cn, instance)
        res.n_critic_runs += 1
        res.total_cost += costs.c_critic_test
        res.actions.append({"step": k, "action": f"critic:{cn}", "passed": passed})
        if not passed:
            res.final_action = "critic_reject"
            res.wall_clock = time.perf_counter() - start
            return res
    # All critics passed → verify
    ok, _ = run_full_test(cname, instance)
    res.n_full_tests += 1
    res.total_cost += costs.c_full_test
    res.actions.append({"step": len(critic_names) + 1, "action": "verify", "ok": ok})
    res.fixed = ok
    res.final_action = "verify_pass" if ok else "verify_fail"
    res.wall_clock = time.perf_counter() - start
    return res


def run_fixed_pipeline(instance, cname, llm_cfg, costs):
    """Generate → run ALL critics (NOT for gating, just for cost accounting) →
    verify regardless of critic outcomes. Paper's "FP" baseline.

    The point of FP is to show what happens if you naively chain every critic
    you have and ALWAYS pay for verify on top — typically loses to selective
    gating (threshold_L*) and to Bayesian planners."""
    res = Result(instance_id=instance["instance_id"], variant="fixed_pipeline")
    start = time.perf_counter()
    reset_repo_for_variant(cname, instance)
    applied, _, _ = _generate_patch_into_container(
        instance, cname, llm_cfg, costs, res, step=0,
    )
    if not applied:
        res.final_action = "no_patch"
        res.wall_clock = time.perf_counter() - start
        return res
    for k, cn in enumerate(SWE_CRITIC_NAMES, start=1):
        passed, _ = run_critic(cname, cn, instance)
        res.n_critic_runs += 1
        res.total_cost += costs.c_critic_test
        res.actions.append({"step": k, "action": f"critic:{cn}", "passed": passed})
    # Always verify, regardless of critic results
    ok, _ = run_full_test(cname, instance)
    res.n_full_tests += 1
    res.total_cost += costs.c_full_test
    res.actions.append({"step": len(SWE_CRITIC_NAMES) + 1, "action": "verify", "ok": ok})
    res.fixed = ok
    res.final_action = "verify_pass" if ok else "verify_fail"
    res.wall_clock = time.perf_counter() - start
    return res


# --------------------------------------------------------------------------
# Bayesian-greedy / DP planners
# --------------------------------------------------------------------------

def _q_one_step_critic(b, c, theta, costs):
    lk = theta[c]
    p = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)
    bp = bayes_update(b, c, True, likelihoods=theta)
    bf = bayes_update(b, c, False, likelihoods=theta)
    return -costs.c_critic_test \
        + p     * max(0.0, -costs.c_full_test + bp * costs.reward) \
        + (1-p) * max(0.0, -costs.c_full_test + bf * costs.reward)


def run_greedy(instance, cname, theta, label, llm_cfg, costs, max_gen=2, prior=0.5):
    res = Result(instance_id=instance["instance_id"], variant=f"greedy_{label}")
    start = time.perf_counter()
    issue = instance["problem_statement"][:ISSUE_CHAR_CAP]
    belief = prior; gen_left = max_gen
    crit_used: set[str] = set(); step = 0
    has_patch = False
    while step < 10:
        Q_bail = 0.0
        # Only allow verify once a patch is actually in place.
        Q_verify = (-costs.c_full_test + belief * costs.reward) if has_patch else -math.inf
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
            ok, _ = run_full_test(cname, instance)
            res.n_full_tests += 1; res.total_cost += costs.c_full_test
            res.actions.append({"step": step, "action": "verify", "ok": ok, "b": belief})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            belief = 0.05
            has_patch = False
            crit_used = set()  # repo reset → new critic epoch
            reset_repo_for_variant(cname, instance)
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            passed, _ = run_critic(cname, cn, instance)
            res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used.add(cn)
            res.actions.append({"step": step, "action": action, "passed": passed, "b": belief})
        else:  # generate
            files = get_files_block(cname, instance)
            prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
            r = call_llm_or_raise(prompt, llm_cfg)
            res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
            res.completion_tokens += r.completion_tokens
            applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
            gen_left -= 1
            if applied:
                has_patch = True
                belief = belief * 0.95 + (1-belief) * 0.50
                crit_used = set()
            else:
                # Patch didn't apply — repo unchanged, belief unchanged.
                res.n_patch_apply_fails += 1
                reset_repo_for_variant(cname, instance)
                has_patch = False
                crit_used = set()
            res.actions.append({"step": step, "action": "generate", "applied": applied, "b": belief})
        step += 1
    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


def run_dp(instance, cname, theta, label, llm_cfg, costs, planner,
           max_gen=2, max_ver=1, prior=0.5):
    res = Result(instance_id=instance["instance_id"], variant=f"dp_{label}")
    start = time.perf_counter()
    issue = instance["problem_statement"][:ISSUE_CHAR_CAP]
    belief = prior; gen_left = max_gen; ver_left = max_ver
    crit_used: frozenset[str] = frozenset(); step = 0
    has_patch = False
    while step < 12:
        action, _q = planner.choose_action(belief, gen_left, crit_used, ver_left)
        # DPPlanner doesn't know about has_patch — override verify before any patch exists.
        if action == "verify" and not has_patch:
            action = "generate:override" if gen_left > 0 else "bail_out"
        if action == "bail_out":
            res.final_action = "bail"; break
        if action == "verify":
            ok, _ = run_full_test(cname, instance)
            res.n_full_tests += 1; res.total_cost += costs.c_full_test
            ver_left -= 1
            res.actions.append({"step": step, "action": "verify", "ok": ok, "b": belief})
            if ok:
                res.fixed = True; res.final_action = "verify_pass"; break
            belief = 0.05
            has_patch = False
            crit_used = frozenset()  # repo reset → new critic epoch
            reset_repo_for_variant(cname, instance)
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            passed, _ = run_critic(cname, cn, instance)
            res.n_critic_runs += 1; res.total_cost += costs.c_critic_test
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used = crit_used | frozenset([cn])
            res.actions.append({"step": step, "action": action, "passed": passed, "b": belief})
        elif action.startswith("generate:"):
            files = get_files_block(cname, instance)
            prompt = PROMPT_TEMPLATE.format(issue=issue, files_block=files)
            r = call_llm_or_raise(prompt, llm_cfg)
            res.n_llm_calls += 1; res.total_cost += costs.c_llm_call
            res.completion_tokens += r.completion_tokens
            applied, n_blocks, n_ok = apply_llm_patch(cname, r.text)
            gen_left -= 1
            if applied:
                has_patch = True
                belief = belief * 0.95 + (1-belief) * 0.50
                crit_used = frozenset()
            else:
                # Patch didn't apply — repo unchanged, belief unchanged.
                res.n_patch_apply_fails += 1
                reset_repo_for_variant(cname, instance)
                has_patch = False
                crit_used = frozenset()
            res.actions.append({"step": step, "action": "generate", "applied": applied, "b": belief})
        step += 1
    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    res.wall_clock = time.perf_counter() - start
    return res


# ---- Resume-safe helpers ----
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
        "instance_id": r.instance_id, "variant": r.variant,
        "fixed": r.fixed, "total_cost": r.total_cost, "wall_clock": r.wall_clock,
        "n_llm_calls": r.n_llm_calls, "n_critic_runs": r.n_critic_runs,
        "n_full_tests": r.n_full_tests,
        "n_patch_apply_fails": r.n_patch_apply_fails,
        "completion_tokens": r.completion_tokens,
        "final_action": r.final_action, "actions": r.actions,
    }


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="SWE-Bench held-out agent comparison.")
    p.add_argument(
        "--dataset", choices=["lite", "verified"], default=None,
        help="Which SWE-Bench upstream split to evaluate against. "
             "Overrides SWE_BENCH_DATASET env var. Default (env unset): verified.",
    )
    p.add_argument(
        "--model", default=None,
        help="OpenAI-compatible model id (overrides ABBO_LLM_MODEL for this "
             "run). Example: openai/gpt-5-mini, anthropic/claude-haiku-4.5.",
    )
    p.add_argument(
        "--results", type=Path, default=None,
        help="Output JSON path (default depends on --dataset + model).",
    )
    p.add_argument(
        "--variants", default=",".join(VARIANTS),
        help="Comma-separated subset of policies to run. Default = all 10. "
             "Use to skip expensive baselines, e.g. "
             "'--variants simple,greedy_hand,greedy_fitted,dp_hand,dp_fitted' "
             "for simple + BG + BDP only.",
    )
    p.add_argument(
        "--instance-ids-file", type=Path, default=None,
        help="Path to a JSON file containing a list of SWE-Bench instance_ids "
             "to run. Overrides the default SWE_INSTANCE_POOL[N_TRAIN:] split. "
             "Use with the output of extract_swe_failed_instances.py to rerun "
             "only the instances no prior method has solved.",
    )
    return p.parse_args()


def _model_slug_for(model_id: str) -> str:
    return model_id.split("/")[-1].replace(":", "_").replace(".", "_")


def main():
    args = _parse_args()
    if args.dataset:
        os.environ["SWE_BENCH_DATASET"] = args.dataset
    dataset_tag = os.environ.get("SWE_BENCH_DATASET", "verified").lower()

    # Resolve model: CLI > env > default
    llm_model = (args.model or "").strip() or LLM_MODEL
    model_slug = _model_slug_for(llm_model)

    # Bind results path to (dataset, model) so Lite and Verified runs (and
    # different models) don't overwrite each other on disk.
    results_path = args.results or (
        ROOT / "sim_results"
        / f"swebench_full_endtoend__{dataset_tag}__{model_slug}.json"
    )

    if args.instance_ids_file:
        # External instance list (e.g. extract_swe_failed_instances.py output).
        # Filter to those that actually exist in the chosen SWE-Bench split
        # so a Lite list passed against --dataset verified doesn't silently
        # try unknown ids (it will raise inside get_instance() instead).
        try:
            test_ids = json.loads(args.instance_ids_file.read_text())
        except Exception as e:
            raise SystemExit(f"failed to read {args.instance_ids_file}: {e}")
        if not isinstance(test_ids, list) or not all(isinstance(x, str) for x in test_ids):
            raise SystemExit("--instance-ids-file must be a JSON list of strings")
        print(f"Dataset: SWE-bench_{dataset_tag.capitalize()}  "
              f"Instance list: {args.instance_ids_file}  "
              f"({len(test_ids)} instances)  Results: {results_path}")
    else:
        rng = random.Random(SPLIT_SEED)
        all_ids = SWE_INSTANCE_POOL[:]   # 11 small-deps instances
        rng.shuffle(all_ids)
        test_ids = all_ids[N_TRAIN:]
        print(f"Dataset: SWE-bench_{dataset_tag.capitalize()}  "
              f"Held-out: {len(test_ids)} instances  Results: {results_path}")
    for tid in test_ids:
        print(f"  {tid}")

    state = load_existing(results_path)
    results = state.setdefault("results", {})

    costs = AgentCostConfig()
    dp_hand = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                        critic_likelihoods=SWE_CRITIC_LIKELIHOODS); dp_hand.solve()
    dp_fitted = DPPlanner(costs, MAX_GENERATORS, MAX_VERIFICATIONS,
                          critic_likelihoods=FITTED_THETA); dp_fitted.solve()

    llm_cfg = build_llm_config_from_env(
        default_provider="openrouter",
        default_model=llm_model,
        default_base_url="https://openrouter.ai/api",
        default_temperature=0.1,
        default_max_tokens=MAX_OUTPUT_TOKENS,
        default_timeout=180,
    )
    # CLI --model takes precedence over any env-supplied model in llm_cfg
    if args.model:
        llm_cfg.model = args.model.strip()
    print(f"LLM provider={llm_cfg.provider} model={llm_cfg.model} "
          f"base_url={llm_cfg.base_url} max_tokens={llm_cfg.max_tokens}")

    # Resolve --variants subset (default = all 10). Fail loudly on unknown
    # names so a typo doesn't silently skip a policy.
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(
            f"Unknown variants: {unknown}. Allowed: {list(VARIANTS)}"
        )
    print(f"Variants this run ({len(variants)}/{len(VARIANTS)}): {list(variants)}")

    total = len(test_ids) * len(variants)
    done = sum(1 for tid in test_ids for v in variants if results.get(f"{tid}|{v}"))
    print(f"\nResume: {done}/{total} pairs already done.\n")

    started = time.time()
    for i, tid in enumerate(test_ids):
        # Skip if all selected variants done for this instance
        if all(results.get(f"{tid}|{v}") for v in variants):
            print(f"\n[{i+1}/{len(test_ids)}] {tid}: all selected variants done, skipping")
            continue

        elapsed = time.time() - started
        print(f"\n[{i+1}/{len(test_ids)}] {tid}  elapsed={elapsed/60:.1f}min")
        try:
            pull_image(tid, verbose=True)
            cname = start_container(tid)
        except Exception as e:
            print(f"  FAILED to pull/start container: {e}")
            continue

        try:
            instance = get_instance(tid)
            for v in variants:
                key = f"{tid}|{v}"
                if results.get(key):
                    continue
                # Reset state before each variant
                try:
                    reset_repo_for_variant(cname, instance)
                except Exception as e:
                    print(f"  [{v}] reset failed: {e}")
                    continue

                try:
                    if v == "simple":
                        r = run_simple(instance, cname, llm_cfg, costs)
                    elif v == "best_of_3":
                        r = run_best_of_3(instance, cname, llm_cfg, costs)
                    elif v == "threshold_L0":
                        r = run_threshold(instance, cname, llm_cfg, costs,
                                          THRESHOLD_L0_CRITICS, "threshold_L0")
                    elif v == "threshold_L2":
                        r = run_threshold(instance, cname, llm_cfg, costs,
                                          THRESHOLD_L2_CRITICS, "threshold_L2")
                    elif v == "threshold_L3":
                        r = run_threshold(instance, cname, llm_cfg, costs,
                                          THRESHOLD_L3_CRITICS, "threshold_L3")
                    elif v == "fixed_pipeline":
                        r = run_fixed_pipeline(instance, cname, llm_cfg, costs)
                    elif v == "greedy_hand":
                        r = run_greedy(instance, cname, SWE_CRITIC_LIKELIHOODS, "hand",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR)
                    elif v == "greedy_fitted":
                        r = run_greedy(instance, cname, FITTED_THETA, "fitted",
                                       llm_cfg, costs, MAX_GENERATORS, PRIOR)
                    elif v == "dp_hand":
                        r = run_dp(instance, cname, SWE_CRITIC_LIKELIHOODS, "hand",
                                   llm_cfg, costs, dp_hand,
                                   MAX_GENERATORS, MAX_VERIFICATIONS, PRIOR)
                    elif v == "dp_fitted":
                        r = run_dp(instance, cname, FITTED_THETA, "fitted",
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
                      f"toks={r.completion_tokens}  apply_fail={r.n_patch_apply_fails}  "
                      f"wc={r.wall_clock:.1f}s  final={r.final_action}")
                save_progress(results_path, state)
        finally:
            stop_container(cname)

    # Final aggregate
    print("\n=== Final aggregate ===")
    R = 100
    from collections import defaultdict
    by_v = defaultdict(list)
    for rec in results.values():
        by_v[rec["variant"]].append(rec)
    print(f"{'variant':<16} {'n':>3} {'fix%':>6} {'cost':>7} {'Ū_π':>8} {'Δ_π':>8}")
    print('-' * 55)
    if "simple" in by_v and by_v["simple"]:
        baseline = sum((R if r["fixed"] else 0) - r["total_cost"]
                       for r in by_v["simple"]) / len(by_v["simple"])
    else:
        baseline = 0.0
    # Show only the variants that ran this session (others may be from
    # earlier runs and aren't relevant to the live aggregate print).
    for v in variants:
        rs = by_v.get(v, [])
        if not rs: continue
        n = len(rs)
        fix = sum(1 for r in rs if r["fixed"]) / n * 100
        c = sum(r["total_cost"] for r in rs) / n
        u = sum((R if r["fixed"] else 0) - r["total_cost"] for r in rs) / n
        d = u - baseline
        print(f"{v:<16} {n:>3} {fix:>5.1f}% {c:>7.2f} {u:>+8.2f} {d:>+8.2f}")

    state["llm_model"] = llm_cfg.model
    state["fitted_theta"] = FITTED_THETA
    state["n_train"] = N_TRAIN
    state["n_test"] = len(test_ids)
    save_progress(results_path, state)
    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
