#!/usr/bin/env python3
"""Re-score existing calibration data with an LLM-review critic (L3).

Reads raw_results.jsonl, looks up each patch's problem statement from the
SWE-bench dataset, calls a cheap LLM (Haiku) to review whether the patch
looks correct, and writes a new raw_results_v2.jsonl with L3_llm_review
added to critic_results.

This is the "noisy critic" the paper's theory is designed for: cheap and
moderately informative, unlike the near-perfect L2 fast test.

Usage:
    python rescore_with_llm_critic.py
    python rescore_with_llm_critic.py --model anthropic/claude-haiku-4-5
    python rescore_with_llm_critic.py --limit 20  # quick test
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from datasets import load_dataset

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
    Path(__file__).resolve().parent / "data" / "raw_results.jsonl"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "raw_results_v2.jsonl"
)
DEFAULT_MODEL = "anthropic/claude-haiku-4.5"

L3_PROMPT_TEMPLATE = """\
You are a senior engineer reviewing a proposed patch for a bug fix.

## Issue
{problem_statement}

## Proposed patch (unified diff)
```diff
{patch}
```

## Task
Judge whether this patch correctly fixes the issue described above.

Consider:
- Does the change target the right code location?
- Does the logic of the change match what the issue asks for?
- Are there obvious errors (wrong variable, typo, missing edge case)?

Answer with EXACTLY one of:
VERDICT: PASS
VERDICT: FAIL

Then one sentence of justification."""


@dataclass(frozen=True)
class L3Result:
    passed: bool
    detail: str


def _parse_verdict(response: str) -> L3Result:
    """Extract PASS/FAIL from LLM response."""
    match = re.search(r"VERDICT:\s*(PASS|FAIL)", response, re.IGNORECASE)
    if match:
        passed = match.group(1).upper() == "PASS"
        justification = response[match.end():].strip().split("\n")[0][:200]
        return L3Result(passed=passed, detail=justification)

    # Fallback: look for the words PASS or FAIL in the first line
    first_line = response.strip().split("\n")[0].upper()
    if "PASS" in first_line and "FAIL" not in first_line:
        return L3Result(passed=True, detail=first_line[:200])
    if "FAIL" in first_line and "PASS" not in first_line:
        return L3Result(passed=False, detail=first_line[:200])

    return L3Result(passed=False, detail=f"unparseable: {response[:100]}")


def review_patch(
    llm: OpenRouterClient,
    problem_statement: str,
    patch: str,
) -> L3Result:
    """Ask the LLM whether the patch looks correct."""
    # Truncate problem statement if too long (keep first 2000 chars)
    ps = problem_statement[:2000]
    # Truncate patch if very long (keep first 3000 chars)
    p = patch[:3000] if patch else "(empty patch)"

    prompt = L3_PROMPT_TEMPLATE.format(problem_statement=ps, patch=p)

    try:
        response = llm.complete(prompt)
        return _parse_verdict(response)
    except Exception as e:
        log.warning("L3 review failed: %s", e)
        return L3Result(passed=False, detail=f"error: {str(e)[:100]}")


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_problem_statements() -> dict[str, str]:
    """Load problem statements for all SWE-bench Lite instances."""
    log.info("Loading SWE-bench Lite dataset for problem statements...")
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    return {d["instance_id"]: d["problem_statement"] for d in ds}


def load_completed_ids(output_path: Path) -> set[str]:
    """Load keys of records already rescored."""
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
                key = f"{r['instance_id']}_{r['patch_id']}"
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def rescore(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_records(input_path)
    log.info("Loaded %d records from %s", len(records), input_path)

    problem_statements = load_problem_statements()
    log.info("Loaded %d problem statements", len(problem_statements))

    completed = load_completed_ids(output_path) if args.resume else set()
    if completed:
        log.info("Resuming: %d records already rescored", len(completed))

    if args.limit > 0:
        records = records[: args.limit]

    llm = OpenRouterClient(model=args.model, verbose=args.verbose)
    log.info("L3 critic LLM: %s", args.model)

    total_l3_pass = 0
    total_evaluated = 0
    n_y1_l3_pass = 0
    n_y0_l3_pass = 0
    n_y1 = 0
    n_y0 = 0

    for i, record in enumerate(records):
        key = f"{record['instance_id']}_{record['patch_id']}"
        if key in completed:
            continue

        instance_id = record["instance_id"]
        problem = problem_statements.get(instance_id, "")
        patch = record.get("patch", "")

        if not patch:
            l3 = L3Result(passed=False, detail="empty patch")
        else:
            l3 = review_patch(llm, problem, patch)
            time.sleep(0.05)  # small delay to avoid rate limits

        record["critic_results"]["L3_llm_review"] = {
            "passed": l3.passed,
            "detail": l3.detail,
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        total_evaluated += 1
        total_l3_pass += int(l3.passed)
        y = record["ground_truth"]
        if y == 1:
            n_y1 += 1
            if l3.passed:
                n_y1_l3_pass += 1
        else:
            n_y0 += 1
            if l3.passed:
                n_y0_l3_pass += 1

        if (i + 1) % 20 == 0:
            log.info(
                "[%d/%d] L3 pass=%d/%d (%.0f%%); Y1 L3-pass=%d/%d; Y0 L3-pass=%d/%d",
                i + 1,
                len(records),
                total_l3_pass,
                total_evaluated,
                100 * total_l3_pass / max(total_evaluated, 1),
                n_y1_l3_pass,
                n_y1,
                n_y0_l3_pass,
                n_y0,
            )

    log.info("=" * 60)
    log.info("Rescoring complete")
    log.info("Total evaluated: %d", total_evaluated)
    if n_y1 > 0:
        log.info("P(L3 pass | Y=1) = %.4f (%d/%d)", n_y1_l3_pass / n_y1, n_y1_l3_pass, n_y1)
    if n_y0 > 0:
        log.info("P(L3 pass | Y=0) = %.4f (%d/%d)", n_y0_l3_pass / n_y0, n_y0_l3_pass, n_y0)
    if n_y1 > 0 and n_y0 > 0:
        gap = (n_y1_l3_pass / n_y1) - (n_y0_l3_pass / n_y0)
        log.info("L3 gap = %.4f", gap)
    log.info("Output: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score calibration data with an LLM-review critic."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model for the L3 reviewer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of records (0 = all).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    rescore(args)


if __name__ == "__main__":
    main()
