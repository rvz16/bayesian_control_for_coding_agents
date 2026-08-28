#!/usr/bin/env python3
"""Re-score LiveCodeBench calibration data with an LLM-review critic (L3).

LCB records use question_id/code schema instead of instance_id/patch.
This script reads the LCB problem content and candidate code, asks Haiku
to judge correctness, and adds L3_llm_review to critic_results.

Usage:
    python rescore_lcb_with_llm_critic.py
    python rescore_lcb_with_llm_critic.py --limit 20
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import pickle
import re
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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

DEFAULT_INPUT = (
    Path(__file__).resolve().parent / "data" / "lcb_results.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "lcb_results_v2.jsonl"
)
LCB_PATH = (
    os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")) + "/hub/"
    "datasets--livecodebench--code_generation_lite/snapshots/"
    "0fe84c3912ea0c4d4a78037083943e8f0c4dd505/test.jsonl"
)
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"


L3_PROMPT_TEMPLATE = """\
You are a senior engineer reviewing a proposed solution to a competitive programming problem.

## Problem
{problem}

## Proposed solution
```python
{code}
```

## Task
Judge whether this solution correctly solves the problem for ALL valid inputs
(not just the visible examples). Consider:
- Does the algorithm handle edge cases (empty inputs, extremes, corner cases)?
- Is the complexity acceptable for the problem's constraints?
- Are there off-by-one errors, wrong base cases, or logic bugs?
- Does it handle the full range of inputs described in the problem?

Answer with EXACTLY one of:
VERDICT: PASS
VERDICT: FAIL

Then one sentence of justification."""


@dataclass(frozen=True)
class L3Result:
    passed: bool
    detail: str


def _parse_verdict(response: str) -> L3Result:
    match = re.search(r"VERDICT:\s*(PASS|FAIL)", response, re.IGNORECASE)
    if match:
        passed = match.group(1).upper() == "PASS"
        justification = response[match.end():].strip().split("\n")[0][:200]
        return L3Result(passed=passed, detail=justification)
    first_line = response.strip().split("\n")[0].upper()
    if "PASS" in first_line and "FAIL" not in first_line:
        return L3Result(passed=True, detail=first_line[:200])
    if "FAIL" in first_line and "PASS" not in first_line:
        return L3Result(passed=False, detail=first_line[:200])
    return L3Result(passed=False, detail=f"unparseable: {response[:100]}")


def review_code(
    llm: OpenRouterClient,
    problem: str,
    code: str,
) -> L3Result:
    if not code.strip():
        return L3Result(passed=False, detail="empty code")
    p = problem[:2500]
    c = code[:3500]
    prompt = L3_PROMPT_TEMPLATE.format(problem=p, code=c)
    try:
        response = llm.complete(prompt)
        return _parse_verdict(response)
    except Exception as e:
        log.warning("L3 review failed: %s", e)
        return L3Result(passed=False, detail=f"error: {str(e)[:100]}")


def load_lcb_problems() -> dict[str, dict]:
    with open(LCB_PATH) as f:
        return {r["question_id"]: r for r in (json.loads(l) for l in f)}


def load_completed(output_path: Path) -> set[str]:
    completed = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = f"{r['question_id']}_{r['patch_id']}"
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d LCB records", len(records))

    problems = load_lcb_problems()
    log.info("Loaded %d LCB problems", len(problems))

    completed = load_completed(output_path) if args.resume else set()
    if completed:
        log.info("Resume: %d records already done", len(completed))

    if args.limit > 0:
        records = records[: args.limit]

    llm = OpenRouterClient(model=args.model)
    log.info("L3 LLM: %s", args.model)

    n_done = 0
    n_y1_pass = 0
    n_y1 = 0
    n_y0_pass = 0
    n_y0 = 0

    for i, record in enumerate(records):
        key = f"{record['question_id']}_{record['patch_id']}"
        if key in completed:
            continue

        qid = record["question_id"]
        problem_info = problems.get(qid)
        if problem_info is None:
            log.warning("Problem %s not found, skipping", qid)
            continue

        code = record.get("code", "")
        problem_content = problem_info["question_content"]

        l3 = review_code(llm, problem_content, code)
        time.sleep(0.05)

        record["critic_results"]["L3_llm_review"] = {
            "passed": l3.passed,
            "detail": l3.detail,
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        n_done += 1
        y = record["ground_truth"]
        if y == 1:
            n_y1 += 1
            if l3.passed:
                n_y1_pass += 1
        else:
            n_y0 += 1
            if l3.passed:
                n_y0_pass += 1

        if (i + 1) % 20 == 0:
            log.info(
                "[%d/%d] Y1 L3-pass=%d/%d Y0 L3-pass=%d/%d",
                i + 1, len(records), n_y1_pass, n_y1, n_y0_pass, n_y0,
            )

    log.info("=" * 60)
    log.info("Rescoring complete: %d records", n_done)
    if n_y1 > 0:
        log.info("P(L3 pass | Y=1) = %d/%d = %.3f", n_y1_pass, n_y1, n_y1_pass / n_y1)
    if n_y0 > 0:
        log.info("P(L3 pass | Y=0) = %d/%d = %.3f", n_y0_pass, n_y0, n_y0_pass / n_y0)
    if n_y1 > 0 and n_y0 > 0:
        gap = (n_y1_pass / n_y1) - (n_y0_pass / n_y0)
        log.info("L3 gap = %.3f", gap)
    log.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
