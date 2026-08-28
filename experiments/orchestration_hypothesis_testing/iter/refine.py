"""Real implementations of Self-Refine [Madaan 2023] + Reflexion [Shinn 2023].

Replaces the policy-replay approximations with genuine API-driven loops:
  - Self-Refine: model generates self-critique → refines based on critique. No
    external test signal. Stops on substring "CRITIQUE_OK" in critique.
  - Reflexion: external test (L2_public_tests for LCB) returns binary signal.
    On fail, model writes a verbal "reflection". Reflections accumulate in
    memory, used as context for next refinement. Stops on test pass.

Faithful to the canonical implementations (see github.com/madaan/self-refine
and github.com/noahshinn/reflexion).

LOGGING SPEC (designed so re-runs are never needed):
  Output dir: <output-dir>/<gen>/<method>/
    - iter_records.jsonl          per-step trajectory: code, critics, Y, costs
    - raw_calls/<inst>_step<k>_<purpose>.json    full prompts + completions
    - cost_log.jsonl              append-only per-API-call audit
    - cost_summary.json           aggregated cost
    - transition_kernel.json      P_fix / P_break under method
    - stop_distribution.json      step at which trajectories stopped
    - RUN_CONFIG.json             CLI args + git SHA + prompt-template hashes

Variant support:
  --variant lcb : uses inline LCB test runner (cheap, no external infra).
                  Reflexion external test = L2_public_tests.
  --variant swe : Self-Refine only (Reflexion on SWE deferred).
  --variant mbpp / humaneval : uses EvalPlus-style assertion runner.
                  Reflexion external test = original (non-plus) asserts.
  --variant humanevalfix : uses HumanEvalPack (python) -- the original
                  assertions act as BOTH L2 (Reflexion signal) and Y
                  oracle (no public/private split on this benchmark).
  --variant codecontests : uses stdin/stdout subprocess test runner.
                  Reflexion external test = public_tests; Y oracle is
                  private_tests + generated_tests (capped at 30).

Usage:
  # LCB Self-Refine (4 gens):
  python3 scripts/iter_refine_real_baselines.py \\
      --method selfrefine --variant lcb \\
      --src-dir data/lcb_calibration_v2 \\
      --output-dir data/lcb_calibration_v2_realbaselines \\
      --difficulty hard --platform leetcode \\
      --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
      --n-instances 30 --steps 5 \\
      --max-cost-usd-per-model 3.0

  # LCB Reflexion (4 gens):
  python3 scripts/iter_refine_real_baselines.py \\
      --method reflexion --variant lcb \\
      [...same args...]

  # SWE-Verified Self-Refine (4 gens):
  python3 scripts/iter_refine_real_baselines.py \\
      --method selfrefine --variant swe \\
      --dataset princeton-nlp/SWE-bench_Verified \\
      --src-dir data/swebench_verified_n30 \\
      --output-dir data/swebench_verified_realbaselines \\
      --generators gpt5_mini,qwen3_coder,haiku45,sonnet45 \\
      --n-instances 30 --steps 5 \\
      --max-cost-usd-per-model 8.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Shared transition-kernel utilities — moved here from a copy that previously
# lived in compute_kernel() below. _common.kernel also hosts the online
# Beta-Binomial estimator that --kernel-mode online activates.
from _common.kernel import (  # noqa: E402
    OnlineKernelCalibration,
    compute_transition_kernel_from_pairs,
    pairs_from_trajectories,
    resolve_kernel,
)
# Shared per-action telemetry (TelemetryLogger). Each iter run writes one
# action_telemetry.jsonl row per generate / refine / reflect / critic_L3 /
# verify call. See _common/telemetry.py for the row schema.
from _common.telemetry import TelemetryLogger  # noqa: E402

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("real_baselines")


# ============================================================================
# Method-specific prompt templates
# ============================================================================

# Self-Refine: canonical loop is critique → refine. No external test.
SELFREFINE_CRITIQUE_PROMPT = """\
You are reviewing a candidate solution to the programming problem below.

## Problem
{problem}

## Candidate code
```python
{code}
```

Carefully review the candidate code. Consider:
- Does it correctly solve the problem?
- Does it handle edge cases (empty inputs, large inputs, boundary conditions)?
- Are there logic errors or off-by-one mistakes?
- Is the algorithmic complexity reasonable?

If the code is correct and handles edge cases well, respond with exactly:
CRITIQUE_OK

Otherwise, list specific issues (one per line) that need to be fixed. Be precise
and concrete. Focus on what's wrong, not on style. Do not write code in this
response — only the critique.
"""

SELFREFINE_REFINE_PROMPT = """\
You are improving a programming solution based on a self-critique.

## Problem
{problem}

## Previous code
```python
{prev_code}
```

## Critique of previous code
{critique}

Now provide a refined version of the code that addresses the critique. Output
ONLY the refined code in a Python code block. Use the same `class Solution:`
format as before.
"""

# Reflexion: canonical loop is attempt → external_test → reflect → re-attempt.
REFLEXION_REFLECT_PROMPT = """\
Your previous attempt at this programming problem failed.

## Problem
{problem}

## Your previous code
```python
{prev_code}
```

## Test feedback
{test_feedback}

In 2-3 sentences, reflect on what went wrong with your previous attempt and
what strategy you should try next. This reflection will be your guidance for
the next attempt — make it specific and actionable, not vague.
"""

REFLEXION_REFINE_PROMPT = """\
You are solving a programming problem. You have learned from past failed
attempts. Use these reflections as guidance.

## Problem
{problem}

## Reflections from past attempts (most recent last)
{reflections_section}

Now provide a NEW solution that addresses the issues identified in the
reflections. Output ONLY the code in a Python code block, using the same
`class Solution:` format as the original problem expects.
"""


# ============================================================================
# Logging utilities
# ============================================================================

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict) -> None:
    """Atomic append to JSONL — flushes immediately so process death doesn't
    lose records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def get_git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=ROOT.parent, text=True).strip()
    except Exception:
        return "unknown"


# ============================================================================
# CallLogger — every API call gets full audit
# ============================================================================

class CallLogger:
    """Records every API call: full prompt, full response, cost, tokens, etc.

    Each call gets a separate JSON file in raw_calls/ AND a row in cost_log.jsonl.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.raw_dir = out_dir / "raw_calls"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cost_log_path = out_dir / "cost_log.jsonl"
        self.cumulative_cost = 0.0
        self.lock = threading.Lock()

    def log_call(self, *, instance_id: str, step: int, purpose: str,
                  model: str, prompt: str, response: str,
                  prompt_tokens: int, completion_tokens: int, cost_usd: float,
                  latency_ms: float | None = None,
                  extra: dict | None = None) -> None:
        with self.lock:
            self.cumulative_cost += cost_usd
            cumulative = self.cumulative_cost

        # Full prompt + response on disk
        raw_path = self.raw_dir / f"{instance_id}_step{step}_{purpose}.json"
        raw_record = {
            "ts": now_iso(),
            "instance_id": instance_id, "step": step, "purpose": purpose,
            "model": model,
            "prompt": prompt,
            "response": response,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "extra": extra or {},
        }
        write_json(raw_path, raw_record)

        # Compact audit row
        append_jsonl(self.cost_log_path, {
            "ts": now_iso(),
            "instance_id": instance_id, "step": step, "purpose": purpose,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "cumulative_usd": cumulative,
            "latency_ms": latency_ms,
            "prompt_hash": sha256_str(prompt),
            "response_hash": sha256_str(response),
        })


# ============================================================================
# Generic-variant helpers — used by MBPP+, HumanEval+, HumanEvalFix, CodeContests.
# The SR/Rfx LOOP logic is identical to LCB; only these per-variant pieces
# differ: instance schema, problem-text extraction, and L0/L1/L2/Y evaluation.
# ============================================================================
import tempfile

def _get_inst_id(inst: dict, variant: str) -> str:
    """Extract a stable instance ID from a benchmark-specific record."""
    if variant == "mbpp":
        return str(inst.get("task_id"))
    elif variant == "humaneval":
        return str(inst.get("task_id"))
    elif variant == "humanevalfix":
        # bigcode/humanevalpack uses task_id like "Python/0"
        return str(inst.get("task_id"))
    elif variant == "codecontests":
        # deepmind/code_contests uses `name` as a stable string ID
        return str(inst.get("name", inst.get("task_id", "")))
    raise ValueError(f"_get_inst_id: unknown variant {variant!r}")

def _get_problem_text(inst: dict, variant: str) -> str:
    """Extract the human-readable problem statement for the SR/Rfx prompts."""
    if variant == "mbpp":
        # MBPP+ stores the natural-language prompt in "text" (+ sample asserts)
        return (inst.get("prompt") or inst.get("text") or "")[:4000]
    elif variant == "humaneval":
        return (inst.get("prompt") or "")[:4000]
    elif variant == "humanevalfix":
        # HumanEvalPack provides the docstring prompt + the buggy function;
        # frame as a bug-fix instruction.
        prompt   = inst.get("prompt", "") or ""
        buggy    = inst.get("buggy_solution", "") or inst.get("declaration", "")
        return (f"Fix the bugs in this function:\n\n{prompt}\n{buggy}")[:4000]
    elif variant == "codecontests":
        return (inst.get("description") or "")[:4000]
    raise ValueError(f"_get_problem_text: unknown variant {variant!r}")

def _run_stdio_tests(code: str, inputs: list, outputs: list,
                       timeout: float = 5.0) -> tuple[int, int]:
    """CodeContests-style stdin/stdout test runner. For each (input, expected
    output) pair, run `code` as a subprocess with the input piped to stdin and
    compare the stripped stdout against the expected. Returns (n_pass, n_total).

    Caps at min(len(inputs), len(outputs)) to handle malformed pairs.
    """
    total = min(len(inputs), len(outputs))
    if total == 0:
        return 0, 0
    n_pass = 0
    tf = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    try:
        tf.write(code); tf.flush(); tf.close()
        for inp, exp in zip(inputs[:total], outputs[:total]):
            try:
                result = subprocess.run(
                    [sys.executable, tf.name],
                    input=str(inp), capture_output=True, text=True, timeout=timeout)
                actual_lines = (result.stdout or "").strip().splitlines()
                expected_lines = (str(exp) or "").strip().splitlines()
                # CodeContests accepts trailing-whitespace and trailing-newline
                # differences; compare line-by-line after strip().
                if [a.rstrip() for a in actual_lines] == [e.rstrip() for e in expected_lines]:
                    n_pass += 1
            except subprocess.TimeoutExpired:
                pass
            except Exception:
                pass
    finally:
        try: os.unlink(tf.name)
        except Exception: pass
    return n_pass, total

def _eval_patch(code: str, inst: dict, variant: str,
                reviewer_client=None, problem_text: str = "",
                cost_lock=None, cost_counter=None, cap_usd: float = 1e9,
                call_logger=None, inst_id: str = "", step: int = 0
                ) -> tuple[dict, float]:
    """Run the inline critic stack (L0 syntax / L1 lint / L2 test / Y oracle)
    for the given variant. Returns ({L0,L1,L2_ok,L2_pass_n,L2_total,Y}, l3_cost).

    The L3 (LLM judge) is run separately by the caller (matches the LCB
    runner's structure) because it requires the cost accounting + API client.
    """
    from calibration.lcb import critic_L0_syntax, critic_L1_lint
    l0 = bool(critic_L0_syntax(code))
    l1 = bool(critic_L1_lint(code))

    if variant == "mbpp":
        from calibration.mbpp import run_assertions, run_full_test
        # L2 (public): subset of original MBPP asserts -> tuple[int,int].
        # Y (oracle): full EvalPlus test_block -> BOOL (run_full_test returns
        # a single bool, not a tuple). MBPP+ has no entry_point field; the
        # test_block defines the test function inline.
        asserts_public = inst.get("test_list", []) or inst.get("base_input", [])
        if isinstance(asserts_public, str):
            asserts_public = [asserts_public]
        l2_pass, l2_total = run_assertions(code, asserts_public, timeout=10)
        test_block  = inst.get("test", "") or "\n".join(asserts_public)
        entry_point = inst.get("entry_point") or None
        y_ok = run_full_test(code, test_block, entry_point, timeout=10)
        Y = 1 if y_ok else 0

    elif variant == "humaneval":
        # HumanEval+ (EvalPlus schema): base_input + plus_input as
        # input-list vectors; use run_test_inputs which feeds inputs to the
        # candidate function and compares the return value.
        from calibration.humaneval import run_test_inputs
        entry_point = inst.get("entry_point", "")
        canon_full  = inst.get("canonical_full", "") or inst.get("canonical_solution", "") or inst.get("prompt", "")
        l2_inputs = inst.get("base_input", []) or []
        y_inputs  = inst.get("plus_input", l2_inputs) or l2_inputs
        atol = inst.get("atol", 0.0) or 0.0
        if l2_inputs:
            l2_pass, l2_total = run_test_inputs(
                code, entry_point, canon_full, l2_inputs, atol, timeout=10)
        else:
            l2_pass, l2_total = 0, 0
        if y_inputs:
            y_pass, y_total = run_test_inputs(
                code, entry_point, canon_full, y_inputs, atol, timeout=10)
            Y = 1 if (y_total > 0) and (y_pass == y_total) else 0
        else:
            Y = 0

    elif variant == "humanevalfix":
        # HumanEvalPack (python): tests are CODE BLOCKS that define a
        # `check(candidate)` function and then call it on `entry_point`.
        # Schema: `test` (oracle), `example_test` (public), `test_setup`
        # (shared imports), `entry_point`. Use mbpp's run_full_test since
        # it accepts a test block and an entry point (sets `candidate`
        # before running). L2 = example_test; Y = test (full).
        from calibration.mbpp import run_full_test
        entry_point = inst.get("entry_point", "") or None
        test_setup  = inst.get("test_setup", "") or ""
        l2_block    = inst.get("example_test", "") or ""
        y_block     = inst.get("test", "") or l2_block
        # Prepend test_setup imports if any (HEFix sometimes uses external libs)
        l2_full = (test_setup + "\n\n" + l2_block) if test_setup else l2_block
        y_full  = (test_setup + "\n\n" + y_block)  if test_setup else y_block
        # L2 as bool -> (1,1) if pass, (0,1) if fail; (0,0) if no public test.
        if l2_block.strip():
            l2_ok_bool = run_full_test(code, l2_full, entry_point, timeout=10)
            l2_pass, l2_total = (1, 1) if l2_ok_bool else (0, 1)
        else:
            l2_pass, l2_total = 0, 0
        y_ok = run_full_test(code, y_full, entry_point, timeout=10) if y_full.strip() else False
        Y = 1 if y_ok else 0

    elif variant == "codecontests":
        # CodeContests: stdin/stdout tests stored as {"input":[...], "output":[...]}
        public_tests  = inst.get("public_tests",  {}) or {}
        private_tests = inst.get("private_tests", {}) or {}
        gen_tests     = inst.get("generated_tests", {}) or {}
        l2_inputs  = list(public_tests.get("input",  []))
        l2_outputs = list(public_tests.get("output", []))
        y_inputs   = (list(private_tests.get("input",  [])) +
                      list(gen_tests.get("input",  [])) + l2_inputs)
        y_outputs  = (list(private_tests.get("output", [])) +
                      list(gen_tests.get("output", [])) + l2_outputs)
        l2_pass, l2_total = _run_stdio_tests(code, l2_inputs, l2_outputs, timeout=5)
        # Cap private+generated at 30 to keep iter-eval latency bounded.
        y_pass, y_total = _run_stdio_tests(code, y_inputs[:30], y_outputs[:30], timeout=5)
        Y = 1 if (y_total > 0) and (y_pass == y_total) else 0

    else:
        raise ValueError(f"_eval_patch: unknown variant {variant!r}")

    l2_ok = (l2_total > 0) and (l2_pass == l2_total)

    # Optional L3 LLM review (caller supplies the client). Mirrors the LCB
    # runner: try the review, account for cost, swallow exceptions.
    l3 = None
    l3_cost = 0.0
    if reviewer_client is not None and problem_text:
        from calibration.lcb import critic_L3_review
        if cost_lock is None or cost_counter is None or cost_counter.get("v", 0.0) < cap_usd:
            try:
                l3_pass, l3_cost = critic_L3_review(problem_text, code, reviewer_client)
                l3 = bool(l3_pass)
            except Exception:
                l3, l3_cost = None, 0.0

    return ({
        "L0_syntax": l0,
        "L1_lint":   l1,
        "L2_public_tests": l2_ok,
        "L2_pass_n":      int(l2_pass),
        "L2_total":       int(l2_total),
        "Y":              int(Y),
        "L3_llm_review":  l3,
    }, float(l3_cost))


def _load_instances_for_variant(variant: str, n_instances: int, seed: int,
                                  plus_input_cap: int = 200) -> list:
    """Load benchmark instances. Mirrors the existing LCB loader pattern;
    SR/Rfx run on the same N=n_instances sample the calibration ran on
    (paired comparison via the shared seed).
    """
    if variant == "mbpp":
        from calibration.mbpp import load_mbpp_plus
        return load_mbpp_plus(n_instances, seed)
    elif variant == "humaneval":
        from calibration.humaneval import load_humaneval_plus
        return load_humaneval_plus(n_instances, plus_input_cap, seed)
    elif variant == "humanevalfix":
        from datasets import load_dataset
        ds = load_dataset("bigcode/humanevalpack", "python", split="test")
        problems = [dict(r) for r in ds]
        rng = random.Random(seed)
        rng.shuffle(problems)
        return problems[:n_instances] if n_instances > 0 else problems
    elif variant == "codecontests":
        # Direct parquet fetch (63 MB) rather than load_dataset which pulls
        # the full 13 GB repo. Same workaround as calibration/codecontests.py.
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        parquet_path = hf_hub_download(
            repo_id="deepmind/code_contests",
            filename="data/test-00000-of-00001-9c49eeff30aacaa8.parquet",
            repo_type="dataset",
        )
        problems = pq.read_table(parquet_path).to_pylist()
        rng = random.Random(seed)
        rng.shuffle(problems)
        return problems[:n_instances] if n_instances > 0 else problems
    raise ValueError(f"_load_instances_for_variant: unknown variant {variant!r}")


# ============================================================================
# LCB variant — Self-Refine + Reflexion using inline test runner
# ============================================================================

def _run_lcb_one_instance(*, inst: dict, step0_code: str, step0_record: dict,
                           method: str, model_id: str, gen_name: str,
                           steps: int, temperature: float, client,
                           gen_client=None,
                           call_logger: CallLogger,
                           tele: "TelemetryLogger | None" = None,
                           cost_lock: threading.Lock, cost_counter: dict,
                           cap_usd: float, scg_helpers) -> dict:
    if gen_client is None:
        gen_client = client
    """Run one LCB instance trajectory under Self-Refine or Reflexion.

    Returns a dict with the full per-step trajectory, stop info, and per-call
    references. Critic outcomes (L0/L1/L2/L3 + Y) computed inline."""
    from calibration.lcb import (cost_for_call, check_tests, decode_private_tests,
                                 critic_L0_syntax, critic_L1_lint,
                                 critic_L3_review, MAX_PRIVATE_TESTS,
                                 build_prompt, extract_code)
    inst_id = str(inst["question_id"])
    public_tests = inst.get("public_test_cases") or []
    if isinstance(public_tests, str):
        try:
            public_tests = json.loads(public_tests)
        except Exception:
            public_tests = []
    private_tests = decode_private_tests(inst.get("private_test_cases", "") or "")
    starter = inst.get("starter_code", "") or ""
    problem_text = inst.get("question_content", "")[:4000]

    # Step 0 record (sunk)
    step0_l2_pass, step0_l2_total = check_tests(step0_code, public_tests, starter_code=starter)
    step0_l2_ok = step0_l2_total > 0 and step0_l2_pass == step0_l2_total
    traj = [{
        "step": 0, "instance_id": inst_id, "method": method,
        "code_chars": len(step0_code),
        "Y": step0_record.get("Y"),
        "L0_syntax": step0_record.get("L0_syntax"),
        "L1_lint": step0_record.get("L1_lint"),
        "L2_public_tests": step0_l2_ok,
        "L3_llm_review": step0_record.get("L3_llm_review"),
        "step_cost_usd": 0.0,
        "method_specific": {},
    }]

    # Reflexion memory: list of past reflections
    reflections: list[str] = []
    prev_code = step0_code

    stop_step = None
    stop_reason = None

    for t in range(1, steps):
        with cost_lock:
            if cost_counter["v"] >= cap_usd:
                stop_reason = "cost_cap"
                stop_step = t
                break

        method_specific: dict = {}
        step_cost = 0.0

        # ----- Stage 1: Method-specific pre-refine call (critique or reflect) -----
        if method == "selfrefine":
            # Self-critique call
            critique_prompt = SELFREFINE_CRITIQUE_PROMPT.format(
                problem=problem_text, code=prev_code[:4000])
            _t0 = time.perf_counter()
            try:
                resp = gen_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": critique_prompt}],
                    temperature=temperature, max_tokens=1500)
                critique_text = resp.choices[0].message.content or ""
                u = resp.usage
                cost = cost_for_call(model_id, u.prompt_tokens, u.completion_tokens)
            except Exception as e:
                log.warning("[%s/%s] step %d critique failed: %s", gen_name, inst_id, t, e)
                stop_reason = "critique_api_error"
                stop_step = t
                break
            _critique_rt = time.perf_counter() - _t0
            with cost_lock:
                cost_counter["v"] += cost
            step_cost += cost
            call_logger.log_call(
                instance_id=inst_id, step=t, purpose="critique",
                model=model_id, prompt=critique_prompt, response=critique_text,
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                cost_usd=cost)
            if tele is not None:
                tele.record(action_type="reflect", runtime_s=_critique_rt,
                            instance_id=inst_id, step=t, api_cost_usd=cost,
                            extra={"purpose": "selfrefine_critique",
                                   "variant": "lcb"})
            method_specific["critique_text"] = critique_text
            # Stop check: "CRITIQUE_OK" substring
            if "CRITIQUE_OK" in critique_text.upper():
                stop_reason = "selfrefine_ok"
                stop_step = t
                # Record this step's no-op decision (kept for trajectory completeness)
                traj.append({
                    "step": t, "instance_id": inst_id, "method": method,
                    "code_chars": len(prev_code),
                    "Y": traj[t - 1]["Y"],  # no new code → carry over Y
                    "L0_syntax": traj[t - 1]["L0_syntax"],
                    "L1_lint": traj[t - 1]["L1_lint"],
                    "L2_public_tests": traj[t - 1]["L2_public_tests"],
                    "L3_llm_review": traj[t - 1]["L3_llm_review"],
                    "step_cost_usd": step_cost,
                    "method_specific": method_specific,
                    "stop_decision": True,
                })
                break

        elif method == "reflexion":
            # External test signal: L2_public_tests on prev_code
            external_test_pass = traj[t - 1]["L2_public_tests"]
            test_feedback_text = (
                f"Public tests passed ({step0_l2_pass}/{step0_l2_total}). "
                "But hidden tests may have failed."
                if external_test_pass
                else f"Public tests failed ({step0_l2_pass}/{step0_l2_total}). "
                     "Code did not pass visible test cases."
            )
            method_specific["external_test_pass"] = external_test_pass
            method_specific["test_feedback"] = test_feedback_text

            if external_test_pass:
                stop_reason = "reflexion_test_pass"
                stop_step = t
                traj.append({
                    "step": t, "instance_id": inst_id, "method": method,
                    "code_chars": len(prev_code),
                    "Y": traj[t - 1]["Y"],
                    "L0_syntax": traj[t - 1]["L0_syntax"],
                    "L1_lint": traj[t - 1]["L1_lint"],
                    "L2_public_tests": True,
                    "L3_llm_review": traj[t - 1]["L3_llm_review"],
                    "step_cost_usd": 0.0,
                    "method_specific": method_specific,
                    "stop_decision": True,
                })
                break

            # Reflection call
            reflect_prompt = REFLEXION_REFLECT_PROMPT.format(
                problem=problem_text, prev_code=prev_code[:4000],
                test_feedback=test_feedback_text)
            _t0 = time.perf_counter()
            try:
                resp = gen_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": reflect_prompt}],
                    temperature=temperature, max_tokens=500)
                reflection_text = resp.choices[0].message.content or ""
                u = resp.usage
                cost = cost_for_call(model_id, u.prompt_tokens, u.completion_tokens)
            except Exception as e:
                log.warning("[%s/%s] step %d reflect failed: %s", gen_name, inst_id, t, e)
                stop_reason = "reflect_api_error"
                stop_step = t
                break
            _reflect_rt = time.perf_counter() - _t0
            with cost_lock:
                cost_counter["v"] += cost
            step_cost += cost
            call_logger.log_call(
                instance_id=inst_id, step=t, purpose="reflect",
                model=model_id, prompt=reflect_prompt, response=reflection_text,
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                cost_usd=cost)
            if tele is not None:
                tele.record(action_type="reflect", runtime_s=_reflect_rt,
                            instance_id=inst_id, step=t, api_cost_usd=cost,
                            extra={"purpose": "reflexion_reflect",
                                   "variant": "lcb"})
            reflections.append(reflection_text)
            method_specific["reflection_text"] = reflection_text
            method_specific["memory_size"] = len(reflections)

        else:
            raise ValueError(f"unknown method: {method}")

        # ----- Stage 2: Refine call -----
        if method == "selfrefine":
            refine_prompt = SELFREFINE_REFINE_PROMPT.format(
                problem=problem_text, prev_code=prev_code[:4000],
                critique=method_specific.get("critique_text", "")[:1500])
        elif method == "reflexion":
            refl_section = "\n\n".join(
                f"Reflection {i+1}: {r}" for i, r in enumerate(reflections))
            refine_prompt = REFLEXION_REFINE_PROMPT.format(
                problem=problem_text,
                reflections_section=refl_section[:3000])
        else:
            raise ValueError(method)

        _t0 = time.perf_counter()
        try:
            resp = gen_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=temperature, max_tokens=4000)
            text = resp.choices[0].message.content or ""
            u = resp.usage
            cost = cost_for_call(model_id, u.prompt_tokens, u.completion_tokens)
        except Exception as e:
            log.warning("[%s/%s] step %d refine failed: %s", gen_name, inst_id, t, e)
            stop_reason = "refine_api_error"
            stop_step = t
            break
        _refine_rt = time.perf_counter() - _t0
        with cost_lock:
            cost_counter["v"] += cost
        step_cost += cost
        call_logger.log_call(
            instance_id=inst_id, step=t, purpose="refine",
            model=model_id, prompt=refine_prompt, response=text,
            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
            cost_usd=cost)
        if tele is not None:
            tele.record(action_type="refine", runtime_s=_refine_rt,
                        instance_id=inst_id, step=t, api_cost_usd=cost,
                        extra={"variant": "lcb"})

        code = extract_code(text)

        # ----- Inline critic eval -----
        _t_critic = time.perf_counter()
        try:
            l2_pass, l2_total = check_tests(code, public_tests, starter_code=starter)
            l2_ok = (l2_total > 0) and (l2_pass == l2_total)
            y_pass, y_total = check_tests(code, private_tests[:MAX_PRIVATE_TESTS], starter_code=starter)
            Y = 1 if (y_total > 0) and (y_pass == y_total) else 0
            l0 = critic_L0_syntax(code)
            l1 = critic_L1_lint(code)
        except Exception as e:
            log.warning("[%s/%s] step %d critic_inline failed: %s", gen_name, inst_id, t, e)
            stop_reason = "critic_eval_error"
            stop_step = t
            break
        # One record covers the combined L0+L1+L2+verify block. Disaggregating
        # would require restructuring _eval_patch; for the latency-analysis
        # use case, "wall time spent in non-LLM critics per step" is the
        # signal we need anyway.
        if tele is not None:
            tele.record(action_type="verify",
                        runtime_s=time.perf_counter() - _t_critic,
                        instance_id=inst_id, step=t, passed=bool(Y),
                        extra={"variant": "lcb",
                               "L0": bool(l0), "L1": bool(l1),
                               "L2_ok": bool(l2_ok),
                               "L2_pass_n": int(l2_pass),
                               "L2_total": int(l2_total)})

        # L3 review (additional API call)
        l3 = None
        with cost_lock:
            cap_ok = cost_counter["v"] < cap_usd
        if cap_ok:
            _t_l3 = time.perf_counter()
            try:
                l3_pass, l3_cost = critic_L3_review(problem_text, code, client)
                l3 = bool(l3_pass)
                with cost_lock:
                    cost_counter["v"] += l3_cost
                step_cost += l3_cost
                if tele is not None:
                    tele.record(action_type="critic_L3",
                                runtime_s=time.perf_counter() - _t_l3,
                                instance_id=inst_id, step=t,
                                passed=l3, api_cost_usd=l3_cost,
                                extra={"variant": "lcb"})
                # Note: critic_L3_review handles its own logging internally if any;
                # we add a lightweight cost-log entry but skip the raw_calls dump
                # since it's a separate review prompt with its own template.
                append_jsonl(call_logger.cost_log_path, {
                    "ts": now_iso(),
                    "instance_id": inst_id, "step": t, "purpose": "L3_review",
                    "model": "L3_reviewer", "prompt_tokens": -1, "completion_tokens": -1,
                    "cost_usd": l3_cost,
                    "cumulative_usd": cost_counter["v"],
                })
            except Exception as e:
                log.warning("[%s/%s] step %d L3 failed: %s", gen_name, inst_id, t, e)

        # Trajectory row
        traj.append({
            "step": t, "instance_id": inst_id, "method": method,
            "code_chars": len(code),
            "Y": int(Y),
            "L0_syntax": bool(l0), "L1_lint": bool(l1),
            "L2_public_tests": bool(l2_ok),
            "L3_llm_review": l3,
            "step_cost_usd": step_cost,
            "method_specific": method_specific,
            "stop_decision": False,
        })
        log.info("[%s/%s/%s] step %d: Y=%d L2=%s L0=%s",
                 method, gen_name, inst_id, t, Y, l2_ok, l0)
        prev_code = code

    return {
        "instance_id": inst_id,
        "trajectory": traj,
        "stop_step": stop_step,
        "stop_reason": stop_reason,
        "n_reflections": len(reflections),
    }


# ============================================================================
# Generic-variant runner — MBPP+, HumanEval+, HumanEvalFix, CodeContests.
# Structurally mirrors _run_lcb_one_instance: same SR/Rfx loop, same stop
# conditions, same trajectory schema. Only the per-step critic eval and the
# instance schema differ; both are routed through _eval_patch + _get_*().
# ============================================================================

def _run_generic_one_instance(*, inst: dict, step0_code: str, step0_record: dict,
                                method: str, variant: str,
                                model_id: str, gen_name: str,
                                steps: int, temperature: float, client,
                                gen_client=None,
                                call_logger: CallLogger,
                                tele: "TelemetryLogger | None" = None,
                                cost_lock: threading.Lock, cost_counter: dict,
                                cap_usd: float) -> dict:
    """SR/Rfx trajectory for the new variants. See _run_lcb_one_instance for
    the canonical version this is patterned after."""
    if gen_client is None:
        gen_client = client
    from calibration.lcb import cost_for_call, extract_code
    inst_id = _get_inst_id(inst, variant)
    problem_text = _get_problem_text(inst, variant)

    # Step 0 record (sunk) -- re-evaluate the step-0 code through the variant
    # critic stack so we have a consistent L2_ok / L0 / L1 / Y for the
    # trajectory's row 0 (the calibration record may not have L2 in the
    # variant's "public test" sense, especially for HEFix where L2==Y).
    step0_eval, _step0_l3_cost = _eval_patch(
        step0_code, inst, variant,
        reviewer_client=None,  # don't waste L3 call on the sunk step
        problem_text=problem_text)
    traj = [{
        "step": 0, "instance_id": inst_id, "method": method,
        "code_chars": len(step0_code),
        "Y": step0_record.get("Y", step0_eval["Y"]),
        "L0_syntax": step0_record.get("L0_syntax", step0_eval["L0_syntax"]),
        "L1_lint":   step0_record.get("L1_lint",   step0_eval["L1_lint"]),
        "L2_public_tests": step0_eval["L2_public_tests"],
        "L3_llm_review":   step0_record.get("L3_llm_review"),
        "step_cost_usd": 0.0,
        "method_specific": {},
    }]

    reflections: list[str] = []
    prev_code = step0_code
    stop_step = None
    stop_reason = None

    for t in range(1, steps):
        with cost_lock:
            if cost_counter["v"] >= cap_usd:
                stop_reason = "cost_cap"; stop_step = t; break

        method_specific: dict = {}
        step_cost = 0.0

        # ----- Stage 1: critique (Self-Refine) or reflect (Reflexion) -----
        if method == "selfrefine":
            critique_prompt = SELFREFINE_CRITIQUE_PROMPT.format(
                problem=problem_text, code=prev_code[:4000])
            _t0 = time.perf_counter()
            try:
                resp = gen_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": critique_prompt}],
                    temperature=temperature, max_tokens=1500)
                critique_text = resp.choices[0].message.content or ""
                u = resp.usage
                cost = cost_for_call(model_id, u.prompt_tokens, u.completion_tokens)
            except Exception as e:
                log.warning("[%s/%s] step %d critique failed: %s", gen_name, inst_id, t, e)
                stop_reason = "critique_api_error"; stop_step = t; break
            _critique_rt = time.perf_counter() - _t0
            with cost_lock: cost_counter["v"] += cost
            step_cost += cost
            call_logger.log_call(
                instance_id=inst_id, step=t, purpose="critique", model=model_id,
                prompt=critique_prompt, response=critique_text,
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                cost_usd=cost)
            if tele is not None:
                tele.record(action_type="reflect", runtime_s=_critique_rt,
                            instance_id=inst_id, step=t, api_cost_usd=cost,
                            extra={"purpose": "selfrefine_critique",
                                   "variant": variant})
            method_specific["critique_text"] = critique_text
            if "CRITIQUE_OK" in critique_text.upper():
                stop_reason = "selfrefine_ok"; stop_step = t
                traj.append({
                    "step": t, "instance_id": inst_id, "method": method,
                    "code_chars": len(prev_code),
                    "Y":               traj[t - 1]["Y"],
                    "L0_syntax":       traj[t - 1]["L0_syntax"],
                    "L1_lint":         traj[t - 1]["L1_lint"],
                    "L2_public_tests": traj[t - 1]["L2_public_tests"],
                    "L3_llm_review":   traj[t - 1]["L3_llm_review"],
                    "step_cost_usd": step_cost,
                    "method_specific": method_specific,
                    "stop_decision": True,
                })
                break

        elif method == "reflexion":
            external_test_pass = traj[t - 1]["L2_public_tests"]
            test_feedback_text = (
                "Public tests passed. But hidden tests may have failed."
                if external_test_pass
                else "Public tests failed. Code did not pass visible test cases."
            )
            method_specific["external_test_pass"] = external_test_pass
            method_specific["test_feedback"] = test_feedback_text

            if external_test_pass:
                stop_reason = "reflexion_test_pass"; stop_step = t
                traj.append({
                    "step": t, "instance_id": inst_id, "method": method,
                    "code_chars": len(prev_code),
                    "Y":               traj[t - 1]["Y"],
                    "L0_syntax":       traj[t - 1]["L0_syntax"],
                    "L1_lint":         traj[t - 1]["L1_lint"],
                    "L2_public_tests": True,
                    "L3_llm_review":   traj[t - 1]["L3_llm_review"],
                    "step_cost_usd": 0.0,
                    "method_specific": method_specific,
                    "stop_decision": True,
                })
                break

            reflect_prompt = REFLEXION_REFLECT_PROMPT.format(
                problem=problem_text, prev_code=prev_code[:4000],
                test_feedback=test_feedback_text)
            _t0 = time.perf_counter()
            try:
                resp = gen_client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": reflect_prompt}],
                    temperature=temperature, max_tokens=500)
                reflection_text = resp.choices[0].message.content or ""
                u = resp.usage
                cost = cost_for_call(model_id, u.prompt_tokens, u.completion_tokens)
            except Exception as e:
                log.warning("[%s/%s] step %d reflect failed: %s", gen_name, inst_id, t, e)
                stop_reason = "reflect_api_error"; stop_step = t; break
            _reflect_rt = time.perf_counter() - _t0
            with cost_lock: cost_counter["v"] += cost
            step_cost += cost
            call_logger.log_call(
                instance_id=inst_id, step=t, purpose="reflect", model=model_id,
                prompt=reflect_prompt, response=reflection_text,
                prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                cost_usd=cost)
            if tele is not None:
                tele.record(action_type="reflect", runtime_s=_reflect_rt,
                            instance_id=inst_id, step=t, api_cost_usd=cost,
                            extra={"purpose": "reflexion_reflect",
                                   "variant": variant})
            reflections.append(reflection_text)
            method_specific["reflection_text"] = reflection_text
            method_specific["memory_size"] = len(reflections)
        else:
            raise ValueError(f"unknown method: {method}")

        # ----- Stage 2: refine -----
        if method == "selfrefine":
            refine_prompt = SELFREFINE_REFINE_PROMPT.format(
                problem=problem_text, prev_code=prev_code[:4000],
                critique=method_specific.get("critique_text", "")[:1500])
        else:
            refl_section = "\n\n".join(
                f"Reflection {i+1}: {r}" for i, r in enumerate(reflections))
            refine_prompt = REFLEXION_REFINE_PROMPT.format(
                problem=problem_text, reflections_section=refl_section[:3000])

        _t0 = time.perf_counter()
        try:
            resp = gen_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": refine_prompt}],
                temperature=temperature, max_tokens=4000)
            text = resp.choices[0].message.content or ""
            u = resp.usage
            cost = cost_for_call(model_id, u.prompt_tokens, u.completion_tokens)
        except Exception as e:
            log.warning("[%s/%s] step %d refine failed: %s", gen_name, inst_id, t, e)
            stop_reason = "refine_api_error"; stop_step = t; break
        _refine_rt = time.perf_counter() - _t0
        with cost_lock: cost_counter["v"] += cost
        step_cost += cost
        call_logger.log_call(
            instance_id=inst_id, step=t, purpose="refine", model=model_id,
            prompt=refine_prompt, response=text,
            prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
            cost_usd=cost)
        if tele is not None:
            tele.record(action_type="refine", runtime_s=_refine_rt,
                        instance_id=inst_id, step=t, api_cost_usd=cost,
                        extra={"variant": variant})

        code = extract_code(text)

        # ----- Inline critic eval (variant-specific) -----
        _t_eval = time.perf_counter()
        try:
            ev, l3_cost = _eval_patch(
                code, inst, variant,
                reviewer_client=client, problem_text=problem_text,
                cost_lock=cost_lock, cost_counter=cost_counter, cap_usd=cap_usd,
                call_logger=call_logger, inst_id=inst_id, step=t)
        except Exception as e:
            log.warning("[%s/%s] step %d eval failed: %s", gen_name, inst_id, t, e)
            stop_reason = "critic_eval_error"; stop_step = t; break
        # Single record for the whole L0+L1+L2+verify (+optional L3) block —
        # _eval_patch fuses them. Granular per-critic timing inside _eval_patch
        # is a follow-up. The L3 sub-cost is teased out below.
        if tele is not None:
            tele.record(action_type="verify",
                        runtime_s=time.perf_counter() - _t_eval,
                        instance_id=inst_id, step=t, passed=bool(ev["Y"]),
                        api_cost_usd=l3_cost,
                        extra={"variant": variant,
                               "L0": bool(ev["L0_syntax"]),
                               "L1": bool(ev["L1_lint"]),
                               "L2_ok": bool(ev["L2_public_tests"]),
                               "L3": ev["L3_llm_review"]})

        if l3_cost > 0:
            with cost_lock: cost_counter["v"] += l3_cost
            step_cost += l3_cost
            append_jsonl(call_logger.cost_log_path, {
                "ts": now_iso(),
                "instance_id": inst_id, "step": t, "purpose": "L3_review",
                "model": "L3_reviewer", "prompt_tokens": -1, "completion_tokens": -1,
                "cost_usd": l3_cost, "cumulative_usd": cost_counter["v"],
            })

        traj.append({
            "step": t, "instance_id": inst_id, "method": method,
            "code_chars": len(code),
            "Y":               ev["Y"],
            "L0_syntax":       ev["L0_syntax"],
            "L1_lint":         ev["L1_lint"],
            "L2_public_tests": ev["L2_public_tests"],
            "L3_llm_review":   ev["L3_llm_review"],
            "step_cost_usd": step_cost,
            "method_specific": method_specific,
            "stop_decision": False,
        })
        log.info("[%s/%s/%s] step %d: Y=%d L2=%s L0=%s",
                 method, gen_name, inst_id, t, ev["Y"], ev["L2_public_tests"], ev["L0_syntax"])
        prev_code = code

    return {
        "instance_id": inst_id,
        "trajectory": traj,
        "stop_step": stop_step,
        "stop_reason": stop_reason,
        "n_reflections": len(reflections),
    }


# ============================================================================
# Environment loading + combined-policy writer
# (ported from PR #5: iter_refine_real_baselines auto-load .env + merge
# per-method policy_comparison.json into one combined file)
# ============================================================================

OPENROUTER_KEY_NAMES = ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPEN_ROUTER")


def load_env_chain() -> None:
    """Auto-load .env from up to 5 parent directories of this script.

    Uses python-dotenv if available; falls back to a minimal manual parser
    so the script still works on machines without the package. Lets callers
    skip the `set -a; source .env; set +a` dance.

    Discovery order (first match per file: ROOT/.env first, then walk up):
      ROOT/.env -> ROOT/../.env -> ../../.env -> ../../../.env -> ../../../../.env
    """
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        load_dotenv = None

    for env_path in [
        ROOT / ".env",
        ROOT.parent / ".env",
        ROOT.parent.parent / ".env",
        ROOT.parent.parent.parent / ".env",
        ROOT.parent.parent.parent.parent / ".env",
    ]:
        if not env_path.exists() or env_path.stat().st_size == 0:
            continue
        if load_dotenv is not None:
            load_dotenv(env_path, override=False)
            continue
        # Fallback minimal parser if dotenv is missing.
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or key in os.environ:
                continue
            if value and value[0] in {"'", '"'}:
                try:
                    parsed = shlex.split(f"dummy={value}", posix=True)
                except ValueError:
                    parsed = [f"dummy={value.strip(chr(34))}"]
                value = parsed[0].split("=", 1)[1] if parsed else value
            else:
                value = value.split(" #", 1)[0].strip()
            os.environ[key] = value


def load_openrouter_key() -> str:
    """Return the first non-empty value from any of OPENROUTER_API_KEY,
    OPEN_ROUTER_API_KEY, OPEN_ROUTER in env. Returns '' if all are unset."""
    for key_name in OPENROUTER_KEY_NAMES:
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    return ""


def write_combined_iter_policy(gen_root: Path) -> None:
    """Merge the per-method policy_comparison.json files into one combined
    JSON so downstream consumers can see SR + Rfx side by side without
    needing two file reads.

    Output: <gen_root>/policy_comparison_iter_replay_baselines.json with:
      policies.always_verify         -> shared baseline (taken from either)
      policies.selfrefine_last       -> from selfrefine/policy_comparison.json
      policies.reflexion_first_pass  -> from reflexion/policy_comparison.json
      sources, cost_model, n_instances -> provenance metadata

    Only writes if at least 2 policies make it into combined.policies
    (otherwise the file would be a near-empty stub).
    """
    combined = {"policies": {}, "sources": {}}
    for method, out_key in (
        ("selfrefine", "selfrefine_last"),
        ("reflexion", "reflexion_first_pass"),
    ):
        path = gen_root / method / "policy_comparison.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        policies = data.get("policies", {})
        if "always_verify" in policies and "always_verify" not in combined["policies"]:
            combined["policies"]["always_verify"] = policies["always_verify"]
        if method in policies:
            combined["policies"][out_key] = policies[method]
            combined["sources"][out_key] = str(path)
            combined["n_instances"] = policies[method].get("n", combined.get("n_instances"))
        if "cost_model" in data:
            combined["cost_model"] = data["cost_model"]
    if len(combined["policies"]) >= 2:
        write_json(gen_root / "policy_comparison_iter_replay_baselines.json", combined)


# ============================================================================
# Aggregation: kernel + policy comparison
# ============================================================================

def compute_kernel(records_path: Path, generator: str) -> dict:
    """Compute Beta(1,1)-smoothed (Y_t, Y_{t+1}) transition kernel from
    trajectory rows.

    Thin wrapper over _common.kernel.compute_transition_kernel_from_pairs.
    Preserves the iter-output schema: {generator, kernel_all: {...},
    n_instances}, with kernel_all containing P_fix_given_broken,
    P_break_given_correct, raw_counts, n_pairs, smoothing — but NOT the
    P_stay_* keys (which downstream consumers don't read here).

    Returns p_fix / p_break as None (not the Laplace-uniform 0.5) when the
    corresponding regime is empty, matching the pre-shared-module behavior.
    """
    if not records_path.exists():
        return {}
    by_inst: dict[str, list[dict]] = {}
    for line in open(records_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        inst = r["instance_id"]
        by_inst.setdefault(inst, []).append(r)
    for rs in by_inst.values():
        rs.sort(key=lambda r: r["step"])

    pairs = pairs_from_trajectories(by_inst.values())
    k = compute_transition_kernel_from_pairs(pairs)
    # Iter's legacy contract: None when a regime is empty (callers do
    # `if p_fix is not None`). The shared helper always returns the
    # Laplace-uniform 0.5; override here for back-compat.
    n_y0 = k["n_broken_observed"]
    n_y1 = k["n_correct_observed"]
    return {
        "generator": generator,
        "kernel_all": {
            "P_fix_given_broken": k["P_fix_given_broken"] if n_y0 > 0 else None,
            "P_break_given_correct": k["P_break_given_correct"] if n_y1 > 0 else None,
            "raw_counts": k["raw_counts"],
            "n_pairs": k["n_pairs"],
            "smoothing": "Beta(1,1)",
        },
        "n_instances": len(by_inst),
    }


def compute_policy_comparison(records_path: Path, generator: str,
                                method: str) -> dict:
    """Replay a 'method-policy' over the trajectory: take the chosen patch's Y
    according to the method's stop semantics, compute mean utility vs
    always_verify with paired-bootstrap CI."""
    COSTS = {"c_gen": 5, "c_critique_reflect": 5, "c_ver": 30, "reward": 100}
    if not records_path.exists():
        return {}
    by_inst: dict[str, list[dict]] = {}
    for line in open(records_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        by_inst.setdefault(r["instance_id"], []).append(r)
    for rs in by_inst.values():
        rs.sort(key=lambda r: r["step"])

    # Per-instance utility
    util_method = []
    util_av = []
    for inst, traj in by_inst.items():
        # always_verify on step-0 patch
        y0 = int(bool(traj[0].get("Y") or 0))
        util_av.append(COSTS["reward"] * y0 - COSTS["c_ver"])

        # Method: walk trajectory, take first stop_decision OR last step;
        # cost = sum of step_cost_usd × (in-method-units) + c_ver at end.
        # Simplification: count generation actions. Each (refine) step costs
        # c_gen. Each (critique/reflect) step costs c_critique_reflect.
        chosen_idx = len(traj) - 1
        n_refines = 0
        n_critiques = 0
        for i, row in enumerate(traj):
            if row.get("stop_decision"):
                chosen_idx = i
                break
            if i == 0:
                continue
            n_refines += 1
            n_critiques += 1
        chosen_y = int(bool(traj[chosen_idx].get("Y") or 0))
        cost = (n_refines * COSTS["c_gen"] +
                 n_critiques * COSTS["c_critique_reflect"] +
                 COSTS["c_ver"])
        util_method.append(COSTS["reward"] * chosen_y - cost)

    n = len(util_av)
    if n == 0:
        return {}

    # Paired bootstrap CI on Δ
    rng = random.Random(42)
    diffs = [a - b for a, b in zip(util_method, util_av)]
    mean_diff = sum(diffs) / n
    boot = []
    for _ in range(1000):
        idxs = [rng.randrange(n) for _ in range(n)]
        boot.append(sum(diffs[i] for i in idxs) / n)
    boot.sort()
    return {
        "policies": {
            "always_verify": {
                "mean_utility": sum(util_av) / n, "n": n,
                "diff_vs_always_verify": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0,
            },
            method: {
                "mean_utility": sum(util_method) / n, "n": n,
                "diff_vs_always_verify": mean_diff,
                "ci95_lo": boot[25], "ci95_hi": boot[975],
            },
        },
        "cost_model": COSTS,
        "method": method,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["selfrefine", "reflexion"])
    parser.add_argument("--variant", required=True, choices=["lcb", "swe", "mbpp", "humaneval", "humanevalfix", "codecontests"])
    parser.add_argument("--src-dir", required=True, type=Path,
                        help="dir with <gen>/critic_results.jsonl + raw_responses/")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--n-instances", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-cost-usd-per-model", type=str, default="3.0",
                        help="Either a single float (applies to all generators) "
                             "or 'key=val,key=val,...' for per-model overrides.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--difficulty", default="hard", help="LCB only")
    parser.add_argument("--platform", default="leetcode", help="LCB only")
    parser.add_argument("--dataset", default=None, help="SWE only")
    parser.add_argument("--lcb-version", default="v1", choices=["v1", "all"],
                        help="v1 = original test.jsonl; all = union of v1..v6")
    parser.add_argument("--extend-existing", action="store_true",
                        help="If set, skip instance_ids already present in iter_records.jsonl "
                             "(only run iter for new instances added by LCB pool expansion).")
    parser.add_argument("--kernel-mode", default="measured",
                        choices=["measured", "online", "hardcoded"],
                        help="Transition-kernel source. 'measured' (default): "
                             "load <src>/<gen>/transition_kernel.json if present, "
                             "else fall back to literature default. 'online': "
                             "also accumulate (Y_t, Y_{t+1}) transitions during "
                             "this run and write <out>/<gen>/<method>/"
                             "transition_kernel_online_final.json with the "
                             "final Beta-Binomial posterior. 'hardcoded': "
                             "always use the literature default (no file). The "
                             "post-hoc kernel written to transition_kernel.json "
                             "from this run's own trajectory is independent of "
                             "this flag.")
    args = parser.parse_args()

    if args.variant == "swe" and args.method == "reflexion":
        log.error("Reflexion-on-SWE deferred for this submission. "
                   "Use replay or run separately with harness-in-loop.")
        sys.exit(2)

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Two-client setup:
    #   reviewer_client (OpenRouter): used by critic_L3_review.
    #   gen_client (per-generator): vLLM for qwen25_32b, OpenRouter otherwise.
    # Auto-load .env walking up the tree (matches mbpp/humaneval_calibrate
    # pattern; lets callers skip `set -a; source .env; set +a`).
    load_env_chain()
    api_key = load_openrouter_key()
    if not api_key:
        log.error("OPENROUTER_API_KEY not set (needed for L3 reviewer); "
                  "tried %s in env + .env files up to 5 levels above ROOT",
                  ", ".join(OPENROUTER_KEY_NAMES))
        sys.exit(1)
    from openai import OpenAI
    reviewer_client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    client = reviewer_client  # alias for backward compat in non-iter callers

    GENERIC_VARIANTS = {"mbpp", "humaneval", "humanevalfix", "codecontests"}
    if args.variant == "lcb":
        # Load LCB
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        from calibration.lcb import load_lcb, GENERATORS as SCG_GENERATORS
        problems = load_lcb(difficulty=args.difficulty, platform=args.platform,
                            lcb_version=args.lcb_version)
        log.info("loaded %d %s/%s LCB problems", len(problems), args.difficulty, args.platform)
        scg_helpers = None
    elif args.variant in GENERIC_VARIANTS:
        # MBPP+ / HumanEval+ / HumanEvalFix / CodeContests
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        # Use lcb_calibrate's GENERATORS table since it carries the
        # OpenRouter/vLLM endpoints for every model id Artem might pass.
        from calibration.lcb import GENERATORS as SCG_GENERATORS
        problems = _load_instances_for_variant(args.variant,
                                                args.n_instances, args.seed)
        log.info("loaded %d %s instances", len(problems), args.variant)
        scg_helpers = None
    else:
        # SWE: load HF dataset
        import datasets as hf_datasets
        ds = hf_datasets.load_dataset(args.dataset, split="test")
        problems = [dict(r) for r in ds]
        from spot_check_generators import GENERATORS as SCG_GENERATORS
        import spot_check_generators as scg
        scg_helpers = scg
        log.info("loaded %d SWE-bench instances", len(problems))

    # Parse per-model cost caps
    def _parse_caps(s: str, default: float = 3.0) -> dict:
        try:
            return {None: float(s)}  # uniform float
        except ValueError:
            d = {}
            for kv in s.split(","):
                if "=" not in kv: continue
                k, v = kv.split("=", 1)
                d[k.strip()] = float(v.strip())
            return d
    caps_dict = _parse_caps(args.max_cost_usd_per_model)

    # Iterate generators
    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    summary_per_gen: dict[str, dict] = {}
    for gen in gens:
        if gen not in SCG_GENERATORS:
            log.error("unknown generator: %s", gen)
            continue
        if args.variant == "lcb":
            model_id = SCG_GENERATORS[gen][0]  # tuple format (model, base_url, thinking)
        else:
            model_id = SCG_GENERATORS[gen][0]
        # Per-generator generation client: qwen25_32b -> vLLM, others -> OpenRouter.
        from calibration.lcb import _make_client
        gen_client = _make_client(gen)

        gen_out = out_root / gen / args.method
        gen_out.mkdir(parents=True, exist_ok=True)

        # Run config
        run_config = {
            "ts": now_iso(), "git_sha": get_git_sha(),
            "method": args.method, "variant": args.variant,
            "generator": gen, "model_id": model_id,
            "n_instances": args.n_instances, "steps": args.steps,
            "seed": args.seed, "temperature": args.temperature,
            "max_workers": args.max_workers,
            "max_cost_usd_per_model": args.max_cost_usd_per_model,
            "src_dir": str(args.src_dir),
            "output_dir": str(out_root),
            "prompt_template_hashes": {
                "selfrefine_critique": sha256_str(SELFREFINE_CRITIQUE_PROMPT),
                "selfrefine_refine": sha256_str(SELFREFINE_REFINE_PROMPT),
                "reflexion_reflect": sha256_str(REFLEXION_REFLECT_PROMPT),
                "reflexion_refine": sha256_str(REFLEXION_REFINE_PROMPT),
            },
            "argv": sys.argv,
        }
        write_json(gen_out / "RUN_CONFIG.json", run_config)

        # Load step-0 trajectories from src_dir
        crit_path = args.src_dir / gen / "critic_results.jsonl"
        if not crit_path.exists():
            log.error("[%s] no critic_results at %s", gen, crit_path)
            continue
        step0_records = {}
        for line in open(crit_path):
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("patch_id", -1) != 0:
                continue
            step0_records[r["instance_id"]] = r

        # Load step-0 code
        if args.variant == "lcb":
            raw_dir = args.src_dir / gen / "raw_responses"
        else:
            raw_dir = args.src_dir / gen / "raw_responses"
        step0_code: dict[str, str] = {}
        for inst_id, rec in step0_records.items():
            # Calibration scripts sanitize inst_id for filesystem paths
            # (replace "/" with "_") so HumanEval/61 -> HumanEval_61_p0.txt.
            # LCB uses integer IDs; the replace is a no-op there.
            # Match calibration scripts' filename sanitization. HumanEval
            # uses inst_id.replace("/", "_") so HumanEval/61 -> HumanEval_61.
            # CodeContests instance names contain spaces and dots
            # (e.g. "1598_D. Training Session") so calibration also strips
            # spaces. Apply both substitutions here.
            safe_id = str(inst_id).replace("/", "_").replace(" ", "_")
            p = raw_dir / f"{safe_id}_p0.txt"
            if p.exists():
                from calibration.lcb import extract_code as lcb_extract
                if args.variant == "lcb":
                    text = p.read_text()
                    step0_code[inst_id] = lcb_extract(text)
                else:
                    step0_code[inst_id] = p.read_text()
            else:
                # Fall back to embedded code in the record if available
                step0_code[inst_id] = rec.get("code", "") or rec.get("diff", "")

        # Eligible instances: those with step0 code AND in problems
        if args.variant == "lcb":
            inst_to_problem = {str(p["question_id"]): p for p in problems}
        elif args.variant in GENERIC_VARIANTS:
            inst_to_problem = {_get_inst_id(p, args.variant): p for p in problems}
        else:
            inst_to_problem = {p["instance_id"]: p for p in problems}
        eligible = [iid for iid, code in step0_code.items()
                    if code and iid in inst_to_problem][: args.n_instances]
        log.info("[%s/%s] %d eligible instances", gen, args.method, len(eligible))

        # Cost tracking
        cost_lock = threading.Lock()
        cost_counter = {"v": 0.0}
        cap_usd = caps_dict.get(gen, caps_dict.get(None, 3.0))

        call_logger = CallLogger(gen_out)
        # Per-action telemetry, separate from CallLogger's per-LLM-call raw
        # audit. Closes the latency gap between calibration (step-0 only)
        # and iter (refinement steps); feeds tab:action_latency joins.
        tele = TelemetryLogger(gen_out / "action_telemetry.jsonl",
                               dataset=str(args.output_dir.name),
                               model_name=gen)
        records_path = gen_out / "iter_records.jsonl"

        # All per-generator work below is wrapped in try/finally so the
        # action_telemetry.jsonl handle is durably closed on every exit
        # path — including the SWE-not-implemented `continue` and any
        # exception in compute_kernel / compute_policy_comparison /
        # write_combined_iter_policy. Matches the same try/finally
        # pattern used in iter/refine_swe.py.
        try:
            # Resume / extend logic: if --extend-existing is set and a records
            # file already exists, drop instance_ids that are already
            # represented (by any step). The records file is append-only so
            # existing rows are kept; the kernel + policy comparison are
            # recomputed from the full file at the end. This lets us re-run
            # ONLY for new LCB pool instances.
            if args.extend_existing and records_path.exists():
                done_iids = set()
                for line in open(records_path):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    iid = rec.get("instance_id")
                    if iid is not None:
                        done_iids.add(str(iid))
                before = len(eligible)
                eligible = [iid for iid in eligible if iid not in done_iids]
                log.info("[%s/%s] --extend-existing: %d already done -> %d new instances to run",
                         gen, args.method, before - len(eligible), len(eligible))

            if args.variant == "lcb" or args.variant in GENERIC_VARIANTS:
                with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                    futures = {}
                    for inst_id in eligible:
                        inst = inst_to_problem[inst_id]
                        if args.variant == "lcb":
                            fut = ex.submit(
                                _run_lcb_one_instance,
                                inst=inst, step0_code=step0_code[inst_id],
                                step0_record=step0_records[inst_id],
                                method=args.method, model_id=model_id, gen_name=gen,
                                steps=args.steps, temperature=args.temperature,
                                client=reviewer_client, gen_client=gen_client,
                                call_logger=call_logger, tele=tele,
                                cost_lock=cost_lock, cost_counter=cost_counter,
                                cap_usd=cap_usd, scg_helpers=scg_helpers,
                            )
                        else:
                            fut = ex.submit(
                                _run_generic_one_instance,
                                inst=inst, step0_code=step0_code[inst_id],
                                step0_record=step0_records[inst_id],
                                method=args.method, variant=args.variant,
                                model_id=model_id, gen_name=gen,
                                steps=args.steps, temperature=args.temperature,
                                client=reviewer_client, gen_client=gen_client,
                                call_logger=call_logger, tele=tele,
                                cost_lock=cost_lock, cost_counter=cost_counter,
                                cap_usd=cap_usd,
                            )
                        futures[fut] = inst_id
                    stop_distribution = []
                    for fut in as_completed(futures):
                        inst_id = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as e:
                            log.error("[%s/%s/%s] failed: %s", args.method, gen, inst_id, e)
                            continue
                        # Append all trajectory rows for this instance
                        for row in result["trajectory"]:
                            append_jsonl(records_path, row)
                        stop_distribution.append({
                            "instance_id": inst_id,
                            "stop_step": result["stop_step"],
                            "stop_reason": result["stop_reason"],
                            "n_reflections": result["n_reflections"],
                        })
            else:
                log.error("SWE variant for real baselines not yet implemented in this build")
                continue

            write_json(gen_out / "stop_distribution.json", {
                "n_instances": len(stop_distribution),
                "instances": stop_distribution,
            })

            # Cost summary
            total_cost = cost_counter["v"]
            write_json(gen_out / "cost_summary.json", {
                "generator": gen, "method": args.method,
                "n_instances_completed": len(stop_distribution),
                "total_cost_usd": total_cost,
                "cap_usd": cap_usd, "cap_hit": total_cost >= cap_usd,
            })

            # Compute kernel + policy comparison
            kernel = compute_kernel(records_path, gen)
            write_json(gen_out / "transition_kernel.json", kernel)

            # --kernel-mode online: also write a Beta-Binomial posterior
            # obtained by streaming this run's (Y_t, Y_{t+1}) transitions
            # through an OnlineKernelCalibration seeded from the calibration
            # kernel (or DEFAULT_KERNEL if none exists). This file is
            # INFORMATIONAL — compute_policy_comparison below intentionally
            # still uses the post-hoc kernel; consuming the online posterior
            # is a separate methodology change. Hardcoded mode forces
            # DEFAULT_KERNEL as the seed; measured/online mode reads the
            # src-dir calibration kernel.
            if args.kernel_mode in ("online", "hardcoded"):
                calib_src_dir = args.src_dir / gen
                seed_kernel, seed_src, online_kernel = resolve_kernel(
                    calib_src_dir, mode=args.kernel_mode,
                )
                if online_kernel is None:
                    # hardcoded mode — instantiate explicitly so we can
                    # still stream transitions through it for the summary.
                    online_kernel = OnlineKernelCalibration(init_kernel=seed_kernel)
                # Replay this run's trajectory through the estimator in step order
                if records_path.exists():
                    by_inst: dict[str, list[dict]] = {}
                    for line in open(records_path):
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        by_inst.setdefault(r["instance_id"], []).append(r)
                    for rs in by_inst.values():
                        rs.sort(key=lambda r: r["step"])
                        for i in range(len(rs) - 1):
                            yt = rs[i].get("Y")
                            yt1 = rs[i + 1].get("Y")
                            if yt not in (0, 1) or yt1 not in (0, 1):
                                continue
                            online_kernel.update(int(yt), int(yt1))
                write_json(gen_out / "transition_kernel_online_final.json", {
                    "generator": gen,
                    "method": args.method,
                    "kernel_mode": args.kernel_mode,
                    "seed_source": seed_src,
                    "seed_kernel": seed_kernel,
                    "online_posterior": online_kernel.summary(),
                })

            policy = compute_policy_comparison(records_path, gen, args.method)
            write_json(gen_out / "policy_comparison.json", policy)

            # If the OTHER method has already been run for this generator,
            # write a combined policy comparison so downstream consumers see
            # SR + Rfx side-by-side without two file reads. No-op if only one
            # method has been run so far (the other run will pick this up).
            write_combined_iter_policy(gen_out.parent)
        finally:
            # Always close the per-generator telemetry handle — covers the
            # happy path, the SWE-not-implemented `continue` above, and any
            # exception in the post-iter writes (compute_kernel et al.).
            tele.close()

        log.info("[%s/%s] done: %d instances, $%.3f spent (cap $%.2f)",
                 gen, args.method, len(stop_distribution), total_cost, cap_usd)
        summary_per_gen[gen] = {
            "n_instances": len(stop_distribution),
            "total_cost_usd": total_cost,
            "kernel": kernel.get("kernel_all", {}),
            "delta_vs_av": policy.get("policies", {}).get(args.method, {}).get("diff_vs_always_verify"),
        }

    # Top-level summary
    write_json(out_root / f"SUMMARY_{args.method}.json", summary_per_gen)
    log.info("Method %s done. Per-gen summary at %s/SUMMARY_%s.json",
             args.method, out_root, args.method)


if __name__ == "__main__":
    main()
