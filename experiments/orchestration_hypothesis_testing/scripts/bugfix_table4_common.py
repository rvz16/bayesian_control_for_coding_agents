"""Shared helpers for bug-fixing Table 4 calibration / replay scripts.

The paper's orchestration code operates on a 3-critic action space
{L0, L2, L3, verify, regen, stop}. The bug-fixing benchmarks in
`agent-bugfix-bayes` expose four critics:

  - critic_syntax
  - critic_lint
  - critic_early
  - critic_mid

For Table 4 compatibility we map them into the existing slots as:

  - L0_syntax      <- critic_syntax
  - L2_public_tests <- critic_early
  - L3_llm_review   <- critic_mid

`L1_lint` is still recorded for diagnostics / likelihood audits, but the
policy layer intentionally ignores it, matching the rest of the paper code.
"""
from __future__ import annotations

import json
import importlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent.parent
BUGFIX_ROOT = PROJECT_ROOT / "bayesian_optimization_for_code_testing" / "agent-bugfix-bayes"
BUGFIX_SRC = BUGFIX_ROOT / "src"
if str(BUGFIX_SRC) not in sys.path:
    sys.path.insert(0, str(BUGFIX_SRC))

CRITIC_SLOT_MAP = {
    "L0_syntax": "critic_syntax",
    "L1_lint": "critic_lint",
    "L2_public_tests": "critic_early",
    "L3_llm_review": "critic_mid",
}

CODE_FENCE_RE = re.compile(r"```(?:python|py)?\n(.*?)\n```", re.DOTALL)
SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_HEF = None
_CC = None

HUMANEVALFIX_DIRECT_PROMPT = """You are a Python code repair assistant. Given a buggy Python function and its test output, return the COMPLETE corrected function and nothing else.

Buggy code:
```python
{source_code}
```

Test output (failing):
```
{test_output}
```
"""

CODECONTESTS_DIRECT_PROMPT = """You are a competitive programming assistant. The Python solution below has a bug that makes it fail at least one test case. Return the COMPLETE corrected Python program and nothing else.

Buggy code:
```python
{source_code}
```

Sample test cases (input -> expected output):
{test_examples}

Recent failed run output:
```
{test_output}
```
"""


def _hef_module():
    global _HEF
    if _HEF is None:
        _HEF = importlib.import_module("abbo.realworld.agents.humaneval_fix")
    return _HEF


def _cc_module():
    global _CC
    if _CC is None:
        _CC = importlib.import_module("abbo.realworld.agents.code_contests")
    return _CC


def safe_stem(instance_id: str) -> str:
    return SAFE_STEM_RE.sub("_", str(instance_id)).strip("_") or "instance"


def extract_code(response_text: str | None, fallback: str) -> str:
    if not isinstance(response_text, str):
        return fallback
    m = CODE_FENCE_RE.search(response_text)
    if m and m.group(1).strip():
        return m.group(1).strip()
    stripped = response_text.strip()
    return stripped if stripped else fallback


def list_task_ids(benchmark: str) -> list[str]:
    if benchmark == "humanevalfix":
        return _hef_module().list_task_ids()
    if benchmark == "codecontests":
        return _cc_module().list_task_ids()
    raise ValueError(f"unknown benchmark: {benchmark}")


def get_initial_source(benchmark: str, task_id: str) -> str:
    if benchmark == "humanevalfix":
        return _hef_module().get_buggy_source(task_id)
    if benchmark == "codecontests":
        pool = _cc_module().get_solution_pool(task_id, "incorrect")
        return pool[0] if pool else ""
    raise ValueError(f"unknown benchmark: {benchmark}")


def _humanevalfix_test_output(source_code: str, task_id: str) -> str:
    hef = _hef_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        (workdir / "solution.py").write_text(source_code)
        ok, detail = hef.run_full_test(workdir, task_id)
    prefix = "PASS" if ok else "FAIL"
    return f"{prefix}\n{detail[:1500]}".strip()


def _format_codecontests_examples(task_id: str, n: int = 2) -> str:
    cc = _cc_module()
    parts = []
    for i, (inp, exp) in enumerate(cc.get_test_cases(task_id, max_tests=n), 1):
        parts.append(
            f"Test {i}:\n"
            f"  input:\n{inp[:300]}\n"
            f"  expected output:\n{exp[:200]}"
        )
    return "\n\n".join(parts) or "(no test cases)"


def _codecontests_test_output(source_code: str, task_id: str) -> str:
    cc = _cc_module()
    tests = cc.get_test_cases(task_id, max_tests=1)
    if not tests:
        return "(no test cases)"
    with tempfile.TemporaryDirectory() as tmpdir:
        workdir = Path(tmpdir)
        source_path = workdir / "solution.py"
        source_path.write_text(source_code)
        try:
            r = subprocess.run(
                [sys.executable, str(source_path)],
                input=tests[0][0],
                text=True,
                capture_output=True,
                timeout=4,
            )
            return (
                f"stdout: {r.stdout[:300]}\n"
                f"stderr: {r.stderr[:200]}\n"
                f"expected: {tests[0][1][:200]}"
            )
        except Exception as exc:
            return f"runtime error: {exc}"


def get_failure_output(benchmark: str, task_id: str, source_code: str) -> str:
    if benchmark == "humanevalfix":
        return _humanevalfix_test_output(source_code, task_id)
    if benchmark == "codecontests":
        return _codecontests_test_output(source_code, task_id)
    raise ValueError(f"unknown benchmark: {benchmark}")


def build_initial_prompt(benchmark: str, task_id: str, source_code: str, test_output: str) -> str:
    if benchmark == "humanevalfix":
        return HUMANEVALFIX_DIRECT_PROMPT.format(
            source_code=source_code,
            test_output=test_output,
        )
    if benchmark == "codecontests":
        return CODECONTESTS_DIRECT_PROMPT.format(
            source_code=source_code,
            test_examples=_format_codecontests_examples(task_id),
            test_output=test_output,
        )
    raise ValueError(f"unknown benchmark: {benchmark}")


def evaluate_candidate(benchmark: str, task_id: str, code: str) -> dict:
    if benchmark == "humanevalfix":
        hef = _hef_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            (workdir / "solution.py").write_text(code)
            l0, l0_detail = hef.run_critic_syntax(workdir)
            l1, l1_detail = hef.run_critic_lint(workdir)
            l2, l2_detail = hef.run_critic_asserts(workdir, task_id, n_asserts=3)
            l3, l3_detail = hef.run_critic_asserts(workdir, task_id, n_asserts=6)
            y, y_detail = hef.run_full_test(workdir, task_id)
        return {
            "Y": int(bool(y)),
            "L0_syntax": bool(l0),
            "L1_lint": bool(l1),
            "L2_public_tests": bool(l2),
            "L3_llm_review": bool(l3),
            "oracle_detail": y_detail[:400],
            "l0_detail": l0_detail[:200],
            "l1_detail": l1_detail[:200],
            "l2_detail": l2_detail[:200],
            "l3_detail": l3_detail[:200],
        }
    if benchmark == "codecontests":
        cc = _cc_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            (workdir / "solution.py").write_text(code)
            l0, l0_detail = cc.run_critic_syntax(workdir)
            l1, l1_detail = cc.run_critic_lint(workdir)
            l2, l2_detail = cc.run_critic_early(workdir, task_id)
            l3, l3_detail = cc.run_critic_mid(workdir, task_id)
            y, y_detail = cc.run_full_test(workdir, task_id)
        return {
            "Y": int(bool(y)),
            "L0_syntax": bool(l0),
            "L1_lint": bool(l1),
            "L2_public_tests": bool(l2),
            "L3_llm_review": bool(l3),
            "oracle_detail": y_detail[:400],
            "l0_detail": l0_detail[:200],
            "l1_detail": l1_detail[:200],
            "l2_detail": l2_detail[:200],
            "l3_detail": l3_detail[:200],
        }
    raise ValueError(f"unknown benchmark: {benchmark}")


def build_likelihood_tables(records: list[dict], gen_key: str, benchmark: str) -> dict:
    rows = [r for r in records if r.get("Y") in (0, 1)]
    n = len(rows)
    n_y1 = sum(1 for r in rows if r["Y"] == 1)
    n_y0 = n - n_y1

    def likes_for(field: str) -> dict:
        tp = sum(1 for r in rows if r["Y"] == 1 and r.get(field))
        fp = sum(1 for r in rows if r["Y"] == 0 and r.get(field))
        fn = n_y1 - tp
        tn = n_y0 - fp
        p1 = (tp + 1) / (n_y1 + 2) if n_y1 >= 0 else 0.5
        p0 = (fp + 1) / (n_y0 + 2) if n_y0 >= 0 else 0.5
        return {
            "P_pass_given_Y1": p1,
            "P_pass_given_Y0": p0,
            "gap": p1 - p0,
            "TP": tp,
            "FN": fn,
            "FP": fp,
            "TN": tn,
        }

    return {
        "generator": gen_key,
        "benchmark": benchmark,
        "n_records": n,
        "n_evaluated": n,
        "n_resolved": n_y1,
        "prior_Y1": (n_y1 + 1) / (n + 2) if n else 0.5,
        "critic_likelihoods": {
            "L0_syntax": likes_for("L0_syntax"),
            "L1_lint": likes_for("L1_lint"),
            "L2_public_tests": likes_for("L2_public_tests"),
            "L3_llm_review": likes_for("L3_llm_review"),
        },
        "critic_slot_map": CRITIC_SLOT_MAP,
        "smoothing": "Beta(1,1)",
    }
