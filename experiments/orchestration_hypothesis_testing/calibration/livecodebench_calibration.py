#!/usr/bin/env python3
"""Calibration pipeline for LiveCodeBench.

LiveCodeBench has two tiers of tests per problem:
  - public_test_cases (1-6 tests): visible to the solver. Use as L2 critic.
  - private_test_cases (12+ tests, hidden): use as Verifier.

This is the natural partial-information regime the paper's theory targets:
passing public tests is evidence of correctness but NOT a guarantee. Running
public tests is cheap; running private tests is the expensive oracle.

For each problem:
  1. Generate N candidate solutions with an LLM.
  2. For each solution:
     - L0: ast.parse (syntax)
     - L1: ruff check (lint)
     - L2: run public test cases
     - L4: run mypy (optional, if time allows)
     - Verifier: run private test cases → ground truth Y
  3. Record everything as JSONL.

Usage:
    python livecodebench_calibration.py --limit 20 --patches-per-instance 3 \\
        --platform leetcode --difficulty easy
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import logging
import os
import pickle
import re
import signal
import subprocess
import sys
import tempfile
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from sage_agent.llm.openrouter import OpenRouterClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_PATCHES_PER_INSTANCE = 3
DEFAULT_TEMPERATURE = 0.8
CODE_TIMEOUT = 10  # seconds per test case
LINT_TIMEOUT = 10
MYPY_TIMEOUT = 15
TEST_PYTHON = sys.executable


@dataclass(frozen=True)
class CriticResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class LCBRecord:
    question_id: str
    patch_id: int
    code: str
    critic_results: dict[str, dict[str, object]]
    ground_truth: int
    metadata: dict[str, str]


# ============================================================================
# Dataset loading
# ============================================================================

LCB_PATH = (
    os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")) + "/hub/"
    "datasets--livecodebench--code_generation_lite/snapshots/"
    "0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl"
)


def load_livecodebench() -> list[dict]:
    if not Path(LCB_PATH).exists():
        log.info("Downloading LiveCodeBench from HuggingFace...")
        path = hf_hub_download(
            "livecodebench/code_generation_lite",
            "test.jsonl",
            repo_type="dataset",
        )
    else:
        path = LCB_PATH
    with open(path) as f:
        return [json.loads(line) for line in f]


def decode_private_tests(encoded: str) -> list[dict]:
    if not encoded:
        return []
    try:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(encoded))))
    except Exception as e:
        log.warning("Failed to decode private tests: %s", e)
        return []


# ============================================================================
# Code generation
# ============================================================================

LCB_PROMPT_TEMPLATE = """\
You are a competitive programmer. Solve this problem in Python.

## Problem
{problem}

{starter_block}

## Requirements
- Output ONLY the Python code, wrapped in ```python ... ``` fences.
- Handle edge cases.
- Use efficient algorithms.
{io_hint}
"""


def build_prompt(problem: dict) -> str:
    starter = problem.get("starter_code", "").strip()
    testtype = _detect_testtype(problem)

    if starter:
        starter_block = (
            "## Starter code (complete this)\n"
            f"```python\n{starter}\n```\n"
        )
    else:
        starter_block = ""

    if testtype == "stdin":
        io_hint = "- Read from stdin, write to stdout. Use `input()` and `print()`."
    else:
        io_hint = "- Complete the Solution class method as specified by the starter code."

    return LCB_PROMPT_TEMPLATE.format(
        problem=problem["question_content"],
        starter_block=starter_block,
        io_hint=io_hint,
    )


def _detect_testtype(problem: dict) -> str:
    pub = json.loads(problem["public_test_cases"]) if problem["public_test_cases"] else []
    if pub and pub[0].get("testtype") == "functional":
        return "functional"
    return "stdin"


def extract_code(response: str) -> str:
    """Extract Python code block from LLM response."""
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def generate_solutions(
    llm: OpenRouterClient,
    problem: dict,
    n: int,
) -> list[str]:
    prompt = build_prompt(problem)
    solutions = []
    for i in range(n):
        try:
            response = llm.complete(prompt)
            code = extract_code(response)
            solutions.append(code)
        except Exception as e:
            log.warning("Gen failed: %s", e)
            solutions.append("")
    return solutions


# ============================================================================
# Test runners
# ============================================================================

_STDIN_RUNNER = """\
import sys
import io
from contextlib import redirect_stdout

# User solution injected above

def _run_test(stdin_input):
    buf = io.StringIO()
    old_stdin = sys.stdin
    sys.stdin = io.StringIO(stdin_input)
    try:
        with redirect_stdout(buf):
            exec(open(__file__).read().split('__TEST_HARNESS_SPLIT__')[0], {{'__name__': '__main__'}})
    finally:
        sys.stdin = old_stdin
    return buf.getvalue()
"""


def _run_stdin_solution(code: str, stdin: str, timeout: int) -> tuple[bool, str]:
    """Run code as a standalone script with given stdin."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        script = f.name

    try:
        result = subprocess.run(
            [TEST_PYTHON, script],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, (result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return False, ("", "timeout")
    except Exception as e:
        return False, ("", str(e))
    finally:
        os.unlink(script)


def _run_functional_test(
    code: str,
    inputs: str,
    expected: str,
    timeout: int,
    entry_point: Optional[str] = None,
) -> tuple[bool, str]:
    """Run a LeetCode-style functional test.

    The code defines a Solution class with a method. We parse `inputs` as
    the function arguments, call the method, and compare the result to
    `expected`.

    LCB input format for LeetCode functional: each argument on its own line
    as a Python literal (or a single line with the full expression). We try
    both forms.
    """
    # Embed the raw input/expected strings into the harness as Python literals
    # using repr() so any quotes/escapes are handled.
    harness = f"""
import sys
import ast
from typing import List, Optional, Dict, Any, Tuple, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
import re

{code}

if __name__ == '__main__':
    _args_str = {inputs!r}
    _expected_str = {expected!r}

    # Parse arguments. LCB format: each line is one Python literal arg.
    try:
        lines = [l for l in _args_str.strip().split('\\n') if l.strip()]
        if len(lines) == 1:
            # Single arg — try to literal_eval directly
            try:
                _args = (ast.literal_eval(lines[0]),)
            except Exception:
                _args = (lines[0],)
        else:
            _args = tuple(ast.literal_eval(l) for l in lines)
    except Exception as e:
        print(f'ARG_PARSE_ERROR: {{type(e).__name__}}: {{e}}', file=sys.stderr)
        sys.exit(1)

    try:
        _sol = Solution()
        _methods = [m for m in dir(_sol) if not m.startswith('_') and callable(getattr(_sol, m))]
        if not _methods:
            print('NO_METHOD', file=sys.stderr)
            sys.exit(1)
        _method = getattr(_sol, _methods[0])
        _result = _method(*_args)
    except Exception as e:
        import traceback
        print(f'RUN_ERROR: {{type(e).__name__}}: {{e}}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    try:
        _expected_val = ast.literal_eval(_expected_str)
    except Exception:
        _expected_val = _expected_str.strip()

    # Coerce to comparable types
    def _norm(x):
        if isinstance(x, list):
            return [_norm(i) for i in x]
        if isinstance(x, tuple):
            return [_norm(i) for i in x]
        return x

    if _norm(_result) == _norm(_expected_val):
        print('PASS')
        sys.exit(0)
    else:
        print(f'FAIL: expected {{_expected_val!r}}, got {{_result!r}}', file=sys.stderr)
        sys.exit(1)
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(harness)
        script = f.name

    try:
        result = subprocess.run(
            [TEST_PYTHON, script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0 and "PASS" in result.stdout
        detail = result.stdout[:200] if passed else result.stderr[:200]
        return passed, detail
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(script)


def run_test_cases(
    code: str,
    tests: list[dict],
    timeout_per_test: int = CODE_TIMEOUT,
) -> tuple[int, int, str]:
    """Run a list of test cases, return (passed, total, first_failure_detail)."""
    if not code.strip():
        return 0, len(tests), "empty code"
    if not tests:
        return 0, 0, "no tests"

    passed = 0
    first_fail = ""
    for t in tests:
        testtype = t.get("testtype", "stdin")
        inp = t.get("input", "")
        exp = t.get("output", "").strip()

        if testtype == "functional":
            ok, detail = _run_functional_test(code, inp, exp, timeout_per_test)
        else:
            ok, output = _run_stdin_solution(code, inp, timeout_per_test)
            stdout_val = output[0] if isinstance(output, tuple) else ""
            ok = ok and stdout_val.strip() == exp
            detail = f"stdout={stdout_val[:100]!r} expected={exp[:100]!r}" if not ok else ""

        if ok:
            passed += 1
        elif not first_fail:
            first_fail = str(detail)[:200]

    return passed, len(tests), first_fail


# ============================================================================
# Critics
# ============================================================================

def critic_l0_syntax(code: str) -> CriticResult:
    if not code.strip():
        return CriticResult(passed=False, detail="empty code")
    try:
        ast.parse(code)
        return CriticResult(passed=True, detail="")
    except SyntaxError as e:
        return CriticResult(passed=False, detail=f"{e.lineno}: {e.msg}")


def critic_l1_lint(code: str) -> CriticResult:
    if not code.strip():
        return CriticResult(passed=False, detail="empty code")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(code)
        script = f.name
    try:
        result = subprocess.run(
            ["ruff", "check", "--select=E,F", "--no-fix", script],
            capture_output=True,
            text=True,
            timeout=LINT_TIMEOUT,
        )
        if result.returncode == 0:
            return CriticResult(passed=True, detail="")
        errs = result.stdout.strip().split("\n")[:3]
        return CriticResult(passed=False, detail="; ".join(errs)[:200])
    except FileNotFoundError:
        return CriticResult(passed=True, detail="ruff not installed")
    except subprocess.TimeoutExpired:
        return CriticResult(passed=False, detail="ruff timeout")
    finally:
        os.unlink(script)


def critic_l2_public_tests(code: str, public_tests: list[dict]) -> CriticResult:
    """Run public test cases. Passes iff ALL public tests pass."""
    passed, total, detail = run_test_cases(code, public_tests)
    if total == 0:
        return CriticResult(passed=False, detail="no public tests")
    if passed == total:
        return CriticResult(passed=True, detail=f"{passed}/{total}")
    return CriticResult(passed=False, detail=f"{passed}/{total}: {detail}")


def verifier_private_tests(code: str, private_tests: list[dict]) -> int:
    """Run private tests. Y=1 iff ALL private tests pass."""
    passed, total, _ = run_test_cases(code, private_tests)
    if total == 0:
        return 0
    return 1 if passed == total else 0


# ============================================================================
# Main pipeline
# ============================================================================

def load_completed(output_file: Path) -> set[str]:
    completed = set()
    if not output_file.exists():
        return completed
    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add(f"{r['question_id']}_{r['patch_id']}")
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--patches-per-instance", type=int, default=DEFAULT_PATCHES_PER_INSTANCE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--platform",
        choices=["leetcode", "atcoder", "codeforces", "all"],
        default="leetcode",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "all"],
        default="easy",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "lcb_results.jsonl"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-private-tests", type=int, default=20)
    args = parser.parse_args()

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_file) if args.resume else set()
    if completed:
        log.info("Resume: %d records already done", len(completed))

    problems = load_livecodebench()
    log.info("Loaded %d LCB problems", len(problems))

    if args.platform != "all":
        problems = [p for p in problems if p["platform"] == args.platform]
    if args.difficulty != "all":
        problems = [p for p in problems if p["difficulty"] == args.difficulty]
    log.info("After filtering: %d problems", len(problems))

    if args.limit > 0:
        problems = problems[: args.limit]

    llm = OpenRouterClient(model=args.model)
    log.info("LLM: %s", args.model)

    n_records = 0
    n_y1 = 0
    for idx, problem in enumerate(problems):
        qid = problem["question_id"]
        log.info("[%d/%d] %s (%s, %s)", idx + 1, len(problems), qid, problem["platform"], problem["difficulty"])

        all_done = all(
            f"{qid}_{i}" in completed
            for i in range(args.patches_per_instance)
        )
        if all_done:
            continue

        public_tests = json.loads(problem["public_test_cases"]) if problem["public_test_cases"] else []
        private_tests = decode_private_tests(problem["private_test_cases"])
        # Cap private tests for speed
        private_tests = private_tests[: args.max_private_tests]
        log.info("  public=%d private=%d", len(public_tests), len(private_tests))

        solutions = generate_solutions(llm, problem, args.patches_per_instance)

        for patch_id, code in enumerate(solutions):
            key = f"{qid}_{patch_id}"
            if key in completed:
                continue

            log.info("  Patch %d: evaluating...", patch_id)
            l0 = critic_l0_syntax(code)
            l1 = critic_l1_lint(code)
            l2 = critic_l2_public_tests(code, public_tests)
            y = verifier_private_tests(code, private_tests)

            log.info(
                "    L0=%s L1=%s L2=%s Y=%d",
                "P" if l0.passed else "F",
                "P" if l1.passed else "F",
                "P" if l2.passed else "F",
                y,
            )

            record = {
                "question_id": qid,
                "patch_id": patch_id,
                "code": code,
                "critic_results": {
                    "L0_syntax": asdict(l0),
                    "L1_lint": asdict(l1),
                    "L2_fast_test": asdict(l2),
                },
                "ground_truth": y,
                "metadata": {
                    "model": args.model,
                    "platform": problem["platform"],
                    "difficulty": problem["difficulty"],
                    "n_public": len(public_tests),
                    "n_private": len(private_tests),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            n_records += 1
            n_y1 += y

    log.info("=" * 60)
    log.info("Done: %d records, Y=1: %d (%.0f%%)",
             n_records, n_y1, 100 * n_y1 / max(n_records, 1))
    log.info("Output: %s", output_file)


if __name__ == "__main__":
    main()
