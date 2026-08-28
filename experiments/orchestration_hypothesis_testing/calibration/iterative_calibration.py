#!/usr/bin/env python3
"""Iterative refinement calibration for the generator transition kernel.

Unlike generate_calibration_data.py (which samples N independent patches per
instance), this script runs real refinement: patch_{t+1} = refine(patch_t,
critic_feedback_t). Transitions between consecutive patches measure the
actual transition kernel P(Y_{t+1} | Y_t, a_gen).

For each instance:
  1. Generate patch_0 from the original problem (no feedback).
  2. Evaluate patch_0 with all critics and the verifier (to get Y_0).
  3. For steps 1..N-1:
     - Build a refinement prompt with (problem, previous patch, critic feedback).
     - Generate patch_t.
     - Evaluate patch_t.
  4. Record the full trajectory including all Y values for transition estimation.

Output: iterative_results.jsonl, one record per (instance, refinement_step).

Usage:
    # Quick test
    python iterative_calibration.py --limit 3 --steps 5

    # Full run on all sympy instances
    python iterative_calibration.py --limit 77 --steps 5 --repos sympy/sympy
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from sage_agent.llm.openrouter import OpenRouterClient  # noqa: E402

# Reuse helpers from the main calibration script
sys.path.insert(0, str(Path(__file__).parent))
from generate_calibration_data import (  # noqa: E402
    CriticResult,
    GeneratedPatch,
    MAX_FILE_LINES,
    PATCH_PROMPT_TEMPLATE,
    TEST_PYTHON,
    _apply_changes_to_content,
    _format_file_contents,
    _make_diff,
    _parse_change_blocks,
    _parse_file_blocks,
    _read_oracle_files,
    evaluate_patch,
    setup_repo,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_STEPS = 5


REFINE_PROMPT_TEMPLATE = """\
You are an expert software engineer refining a bug fix in the {repo} repository.

## Issue
{problem_statement}

{hints_section}

## Files that likely need changes

{file_contents}

## Your previous attempt (step {prev_step})
{prev_patch_repr}

## Diagnostic feedback on your previous attempt
{feedback}

## Task
Produce an improved patch. Use the diagnostic feedback to guide your fix:
- If the syntax/lint failed, fix the syntax errors.
- If the test failed, look at the error message and adjust the logic.
- If all cheap diagnostics passed but the patch is still wrong, reconsider the
  semantic meaning of the issue and try a different approach.

Output your changes as SEARCH/REPLACE blocks. The SEARCH text must match the
ORIGINAL file contents exactly (not your previous patch's output).

<<<CHANGE path/to/file.py
SEARCH
(exact lines from the ORIGINAL file that you want to replace)
REPLACE
(the new lines that should replace the search block)
CHANGE>>>

Keep changes minimal and focused on fixing the issue.
"""


def _format_patch_for_refinement(patch: GeneratedPatch) -> str:
    """Render a GeneratedPatch as a readable summary for the next prompt."""
    if not patch.modified_files:
        return "(previous attempt was empty or malformed)"

    parts = []
    for fpath, content in patch.modified_files.items():
        parts.append(f"### Modified {fpath}")
        # Show the diff instead of the full file to save tokens
        parts.append("```diff")
        parts.append(patch.diff[:2000] if patch.diff else "(no diff computed)")
        parts.append("```")
    return "\n".join(parts)


def _format_feedback(
    l0: CriticResult,
    l1: CriticResult,
    l2: CriticResult,
) -> str:
    """Format critic outputs as refinement feedback."""
    lines = []
    if l0.passed:
        lines.append("- Syntax check (L0): PASS")
    else:
        lines.append(f"- Syntax check (L0): FAIL — {l0.detail[:200]}")
    if l1.passed:
        lines.append("- Lint (L1): PASS")
    else:
        lines.append(f"- Lint (L1): FAIL — {l1.detail[:300]}")
    if l2.passed:
        lines.append("- Fast test (L2): PASS")
    else:
        lines.append(f"- Fast test (L2): FAIL — {l2.detail[:400]}")
    return "\n".join(lines)


def _build_generated_patch(
    response: str,
    oracle_files: dict[str, str],
) -> GeneratedPatch:
    """Parse an LLM response into a GeneratedPatch.

    Tries SEARCH/REPLACE blocks first, then FILE blocks as fallback.
    """
    change_blocks = _parse_change_blocks(response)
    if change_blocks:
        file_changes: dict[str, list[tuple[str, str]]] = {}
        for fpath, search, replace in change_blocks:
            resolved = fpath
            if fpath not in oracle_files:
                for opath in oracle_files:
                    if opath.endswith(fpath) or fpath.endswith(opath):
                        resolved = opath
                        break
            file_changes.setdefault(resolved, []).append((search, replace))

        resolved_files: dict[str, str] = {}
        diffs: list[str] = []
        for fpath, changes in file_changes.items():
            original = oracle_files.get(fpath, "")
            if not original:
                continue
            modified = _apply_changes_to_content(original, changes)
            if modified != original:
                resolved_files[fpath] = modified
                diff = _make_diff(original, modified, fpath)
                if diff:
                    diffs.append(diff)

        if resolved_files:
            return GeneratedPatch(
                diff="\n".join(diffs),
                modified_files=resolved_files,
            )

    # Fallback: FILE blocks
    blocks = _parse_file_blocks(response)
    if blocks:
        resolved_files = {}
        diffs = []
        for fpath, modified_content in blocks.items():
            original = oracle_files.get(fpath, "")
            if not original:
                for opath, ocontent in oracle_files.items():
                    if opath.endswith(fpath) or fpath.endswith(opath):
                        original = ocontent
                        fpath = opath
                        break
            resolved_files[fpath] = modified_content
            diff = _make_diff(original, modified_content, fpath)
            if diff:
                diffs.append(diff)
        return GeneratedPatch(diff="\n".join(diffs), modified_files=resolved_files)

    return GeneratedPatch(diff="", modified_files={})


def generate_initial_patch(
    llm: OpenRouterClient,
    problem: str,
    repo: str,
    hints: str,
    oracle_files: dict[str, str],
) -> GeneratedPatch:
    """Step 0: generate a patch from scratch without feedback."""
    hints_section = f"## Hints\n{hints}" if hints else ""
    file_contents = _format_file_contents(oracle_files)
    prompt = PATCH_PROMPT_TEMPLATE.format(
        repo=repo,
        problem_statement=problem,
        hints_section=hints_section,
        file_contents=file_contents,
    )
    try:
        response = llm.complete(prompt)
        return _build_generated_patch(response, oracle_files)
    except Exception as e:
        log.warning("Initial patch generation failed: %s", e)
        return GeneratedPatch(diff="", modified_files={})


def generate_refined_patch(
    llm: OpenRouterClient,
    problem: str,
    repo: str,
    hints: str,
    oracle_files: dict[str, str],
    prev_patch: GeneratedPatch,
    prev_step: int,
    l0: CriticResult,
    l1: CriticResult,
    l2: CriticResult,
) -> GeneratedPatch:
    """Steps 1..N: refine given the previous patch and critic feedback."""
    hints_section = f"## Hints\n{hints}" if hints else ""
    file_contents = _format_file_contents(oracle_files)
    prev_repr = _format_patch_for_refinement(prev_patch)
    feedback = _format_feedback(l0, l1, l2)

    prompt = REFINE_PROMPT_TEMPLATE.format(
        repo=repo,
        problem_statement=problem,
        hints_section=hints_section,
        file_contents=file_contents,
        prev_step=prev_step,
        prev_patch_repr=prev_repr,
        feedback=feedback,
    )

    try:
        response = llm.complete(prompt)
        return _build_generated_patch(response, oracle_files)
    except Exception as e:
        log.warning("Refinement step failed: %s", e)
        return GeneratedPatch(diff="", modified_files={})


def run_one_instance(
    llm: OpenRouterClient,
    instance: dict,
    repo_path: Path,
    n_steps: int,
    model_name: str,
    output_path: Optional[Path] = None,
    early_stop: bool = False,
) -> list[dict]:
    """Run the full refinement trajectory for one instance.

    If output_path is provided, writes each record immediately after the step
    completes (for crash resilience and live monitoring).

    If early_stop is True, stops as soon as a patch reaches Y=1. For
    calibration we want early_stop=False so we can measure P(break|correct).
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    problem = instance["problem_statement"]
    hints = instance.get("hints_text", "")
    test_patch = instance.get("test_patch", "")
    gold_patch = instance.get("patch", "")

    oracle_files = _read_oracle_files(repo_path, gold_patch)
    if not oracle_files:
        log.warning("  No oracle files available, skipping")
        return []

    records: list[dict] = []
    current_patch: Optional[GeneratedPatch] = None
    current_l0: Optional[CriticResult] = None
    current_l1: Optional[CriticResult] = None
    current_l2: Optional[CriticResult] = None

    for step in range(n_steps):
        log.info("  Step %d/%d: generating...", step, n_steps - 1)

        if step == 0:
            patch = generate_initial_patch(
                llm, problem, repo, hints, oracle_files
            )
        else:
            patch = generate_refined_patch(
                llm, problem, repo, hints, oracle_files,
                current_patch, step - 1,
                current_l0, current_l1, current_l2,
            )

        if not patch.modified_files and not patch.diff:
            log.warning("  Step %d: empty patch, marking as all-fail", step)
            record = {
                "instance_id": instance_id,
                "step": step,
                "patch": "",
                "critic_results": {
                    "L0_syntax": {"passed": False, "detail": "empty patch"},
                    "L1_lint": {"passed": False, "detail": "empty patch"},
                    "L2_fast_test": {"passed": False, "detail": "empty patch"},
                },
                "ground_truth": 0,
                "metadata": {
                    "model": model_name,
                    "repo": repo,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            records.append(record)
            if output_path:
                with open(output_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            # For next refinement step, treat empty as all-fail feedback
            current_patch = patch
            current_l0 = CriticResult(passed=False, detail="empty patch")
            current_l1 = CriticResult(passed=False, detail="empty patch")
            current_l2 = CriticResult(passed=False, detail="empty patch")
            continue

        log.info("  Step %d: evaluating...", step)
        l0, l1, l2, y = evaluate_patch(
            patch.diff, repo_path, test_patch,
            modified_files=patch.modified_files or None,
            instance_id=instance_id,
            model_name=model_name,
        )
        log.info(
            "    L0=%s L1=%s L2=%s Y=%d",
            "P" if l0.passed else "F",
            "P" if l1.passed else "F",
            "P" if l2.passed else "F",
            y,
        )

        record = {
            "instance_id": instance_id,
            "step": step,
            "patch": patch.diff,
            "critic_results": {
                "L0_syntax": asdict(l0),
                "L1_lint": asdict(l1),
                "L2_fast_test": asdict(l2),
            },
            "ground_truth": y,
            "metadata": {
                "model": model_name,
                "repo": repo,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        records.append(record)
        if output_path:
            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")

        current_patch = patch
        current_l0 = l0
        current_l1 = l1
        current_l2 = l2

        # Optional early stop: if verifier passes, no need to refine further
        if early_stop and y == 1:
            log.info("  Step %d: Y=1 reached, stopping refinement", step)
            break

    return records


def load_completed_instances(output_path: Path) -> set[str]:
    """Load instance_ids that already have records in the output file."""
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                completed.add(r["instance_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative refinement calibration for transition kernel."
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--repos", nargs="+", default=["sympy/sympy"])
    parser.add_argument("--dataset", default="lite")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workdir", default="/tmp/calibration_repos")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "iterative_results.jsonl"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop refinement when Y=1 is reached (production-like, but loses "
             "P(break|correct) data for calibration).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
    }
    log.info("Loading %s...", dataset_map[args.dataset])
    dataset = load_dataset(dataset_map[args.dataset], split="test")

    if args.repos:
        indices = [i for i, d in enumerate(dataset) if d["repo"] in args.repos]
        dataset = dataset.select(indices)
        log.info("Filtered to repos %s: %d instances", args.repos, len(dataset))

    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    completed = load_completed_instances(output_path) if args.resume else set()
    if completed:
        log.info("Resume: %d instances already completed", len(completed))

    llm = OpenRouterClient(model=args.model)
    log.info("LLM: %s, steps=%d, instances=%d", args.model, args.steps, len(dataset))
    log.info("Output: %s", output_path)

    total_trajectories = 0
    total_y1 = 0

    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        if instance_id in completed:
            log.info("[%d/%d] %s (skip)", idx + 1, len(dataset), instance_id)
            continue

        log.info(
            "[%d/%d] %s (%s)",
            idx + 1, len(dataset), instance_id, instance["repo"],
        )

        try:
            repo_path = setup_repo(instance["repo"], instance["base_commit"], workdir)
        except Exception as e:
            log.error("  Setup failed: %s", e)
            continue

        records = run_one_instance(
            llm, instance, repo_path, args.steps, args.model,
            output_path=output_path,
            early_stop=args.early_stop,
        )

        total_trajectories += 1
        if records and records[-1].get("ground_truth") == 1:
            total_y1 += 1

    log.info("=" * 60)
    log.info("Iterative calibration complete")
    log.info("Total trajectories: %d", total_trajectories)
    log.info("Final-step Y=1: %d", total_y1)
    log.info("Output: %s", output_path)
    log.info("Next: python compute_transition_kernel.py")


if __name__ == "__main__":
    main()
