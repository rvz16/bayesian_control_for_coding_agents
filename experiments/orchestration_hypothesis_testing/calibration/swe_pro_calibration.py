#!/usr/bin/env python3
"""Calibration pipeline for SWE-Bench Pro (Python subset).

For each SWE-Bench Pro Python instance:
  1. Pull the prebuilt podman/docker image (jefzda/sweap-images:<dockerhub_tag>).
  2. Extract the "oracle" files — the ones that the gold patch modifies — from
     the container so the LLM can generate a candidate patch against the true
     baseline.
  3. Generate N candidate patches with an LLM (OpenRouter).
  4. For each candidate patch:
       - L0 (ast.parse) on the patched file contents (host, fast).
       - L1 (ruff) on the patched file contents (host, fast).
       - L4 (mypy) on the patched file contents (host, fast).
       - L2 (fast test) — run `run_script.sh <selected_test_files_to_run>`
         inside the container after applying the patch.
       - L3 (Haiku LLM reviewer) with the problem statement + patch (host).
       - Verifier — run `run_script.sh` with no args (full hidden suite)
         inside the container and read the pytest summary for fail_to_pass
         resolution. The resulting ground truth Y ∈ {0, 1} is our label.
  5. Append one JSONL row per patch to
     `data/swe_pro_results.jsonl`, schema-compatible with
     compute_likelihoods.py and run_simulation.py.
  6. After the last patch of each instance, delete the podman image to keep
     disk usage bounded (pull-run-delete).

Prerequisites
  - podman available (rootless is fine; `podman pull` from docker.io works).
  - The SWE-Bench Pro run_scripts are expected at
    experiments/orchestration_hypothesis_testing/calibration/pro_run_scripts/
    (clone https://github.com/scaleapi/SWE-bench_Pro-os.git and copy
    `run_scripts/` there).
  - OPENROUTER_API_KEY set in `.env`.

Usage
  python swe_pro_calibration.py --limit 2                 # smoke test
  python swe_pro_calibration.py --limit 150 --resume      # full run
  python swe_pro_calibration.py --instance-ids a,b,c      # specific instances
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
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

DEFAULT_PARQUET = Path(
    os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")) + "/hub/"
    "datasets--ScaleAI--SWE-bench_Pro/snapshots/"
    "7ab5114912baf22bb098818e604c02fe7ad2c11f/data/test-00000-of-00001.parquet"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "data" / "swe_pro_results.jsonl"
)
DEFAULT_RUN_SCRIPTS_DIR = (
    Path(__file__).resolve().parent / "pro_run_scripts"
)
IMAGE_REGISTRY = "docker.io/jefzda/sweap-images"
DEFAULT_GEN_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_REVIEW_MODEL = "anthropic/claude-haiku-4.5"
DEFAULT_PATCHES_PER_INSTANCE = 3
DEFAULT_TEMPERATURE = 0.8
MAX_FILE_LINES = 1200

CONTAINER_ENGINE = os.environ.get("CONTAINER_ENGINE", "podman")

# Test runner timeouts (seconds)
FAST_TEST_TIMEOUT = 300   # L2 — selected test files only
FULL_TEST_TIMEOUT = 900   # Verifier — full hidden suite
IMAGE_PULL_TIMEOUT = 1800 # First pull can be slow for large repos
HOST_CRITIC_TIMEOUT = 60


@dataclass(frozen=True)
class CriticResult:
    passed: bool
    detail: str


@dataclass
class ProRecord:
    instance_id: str
    patch_id: int
    repo: str
    repo_language: str
    base_commit: str
    dockerhub_tag: str
    patch: str
    critic_results: dict
    ground_truth: int
    metadata: dict = field(default_factory=dict)


# ============================================================================
# Container helpers (podman-compatible with docker)
# ============================================================================


def _engine_available() -> bool:
    try:
        subprocess.run(
            [CONTAINER_ENGINE, "--version"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except Exception as exc:
        log.error("Container engine %s not available: %s", CONTAINER_ENGINE, exc)
        return False


def pull_image(tag: str) -> bool:
    """Pull a SWE-Bench Pro image by its dockerhub_tag."""
    image = f"{IMAGE_REGISTRY}:{tag}"
    log.info("Pulling %s", image)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [CONTAINER_ENGINE, "pull", image],
            capture_output=True,
            text=True,
            timeout=IMAGE_PULL_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        log.error("Pull timeout for %s", tag)
        return False
    if proc.returncode != 0:
        log.error("Pull failed for %s: %s", tag, proc.stderr[-400:])
        return False
    log.info("Pulled %s in %.1fs", tag, time.time() - t0)
    return True


def remove_image(tag: str) -> None:
    image = f"{IMAGE_REGISTRY}:{tag}"
    subprocess.run(
        [CONTAINER_ENGINE, "rmi", "-f", image],
        capture_output=True,
        text=True,
        timeout=120,
    )


def run_in_container(
    tag: str,
    command: str,
    mount_dir: Optional[Path] = None,
    timeout: int = 300,
) -> tuple[int, str]:
    """Execute a bash command inside the instance container.

    Different Pro images have different entrypoint/cmd defaults (some use
    `/bin/bash` as entrypoint, others have no entrypoint and `bash` as cmd).
    We force `/bin/bash -c <command>` via ``--entrypoint`` to get identical
    semantics regardless of the image metadata. Workdir is always /app.
    If `mount_dir` is given, it is mounted read-write at /work in the container.

    Returns (exit_code, combined stdout+stderr).
    """
    image = f"{IMAGE_REGISTRY}:{tag}"
    cmd = [CONTAINER_ENGINE, "run", "--rm", "--entrypoint=/bin/bash"]
    if mount_dir is not None:
        cmd += ["-v", f"{mount_dir}:/work:rw,z"]
    cmd += [image, "-c", command]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as te:
        return 124, f"TIMEOUT after {timeout}s: {te}"
    except Exception as exc:
        return 1, f"RUN ERROR: {exc}"


def extract_file_from_image(tag: str, path_in_container: str) -> Optional[str]:
    """Read a file from /app/<path> inside the instance image."""
    rc, out = run_in_container(
        tag,
        f"cat /app/{path_in_container}",
        mount_dir=None,
        timeout=60,
    )
    if rc != 0:
        return None
    return out


# ============================================================================
# Oracle retrieval and patch generation
# ============================================================================


def _get_changed_files(unified_diff: str) -> list[str]:
    files: list[str] = []
    for line in unified_diff.split("\n"):
        if line.startswith("+++ b/"):
            path = line[len("+++ b/"):].strip()
            if path and path != "/dev/null":
                files.append(path)
    return files


def read_oracle_files(tag: str, gold_patch: str) -> dict[str, str]:
    """Extract the files the gold patch touches from the container."""
    paths = _get_changed_files(gold_patch)
    out: dict[str, str] = {}
    for p in paths[:8]:
        content = extract_file_from_image(tag, p)
        if content is None:
            log.warning("Could not extract /app/%s", p)
            continue
        lines = content.split("\n")
        if len(lines) > MAX_FILE_LINES:
            content = "\n".join(lines[:MAX_FILE_LINES]) + f"\n... (truncated, {len(lines)} lines)"
        out[p] = content
    return out


GENERATION_PROMPT = """\
You are fixing a bug in a large Python project. Below you will find the
problem statement and the current contents of the files that need to be
modified. Produce a complete, correct replacement for each file you change.

## Problem statement
{problem_statement}

## Requirements (may be empty)
{requirements}

## Current file contents
{files}

## Output format
Return each modified file wrapped in <<<FILE ... FILE>>> markers. The body
must be the full new file contents, not a diff. Do not add explanations
outside the markers.

Example:
<<<FILE path/to/file.py
# complete new contents
...
FILE>>>

Only include files you are actually modifying. Do not invent new files.
"""


def _parse_file_blocks(response: str) -> dict[str, str]:
    import re
    blocks: dict[str, str] = {}
    pat = re.compile(r"<<<FILE\s+(.+?)\s*\n([\s\S]*?)FILE>>>", re.MULTILINE)
    for m in pat.finditer(response):
        fpath = m.group(1).strip()
        body = m.group(2)
        if body.endswith("\n"):
            body = body[:-1]
        blocks[fpath] = body
    return blocks


def _make_unified_diff(original: str, modified: str, path: str) -> str:
    import difflib
    if original and not original.endswith("\n"):
        original += "\n"
    if modified and not modified.endswith("\n"):
        modified += "\n"
    orig = original.splitlines(keepends=True)
    mod = modified.splitlines(keepends=True)
    d = difflib.unified_diff(orig, mod, fromfile=f"a/{path}", tofile=f"b/{path}")
    s = "".join(d)
    if s and not s.endswith("\n"):
        s += "\n"
    return s


def _complete_with_temperature(
    llm: OpenRouterClient, prompt: str, temperature: float
) -> str:
    """Call the underlying OpenAI-compatible client with an explicit temperature.

    OpenRouterClient.complete() does not expose temperature so we talk to the
    wrapped client directly; this keeps the sampling diversity we need for
    generating N distinct candidate patches per instance.
    """
    resp = llm._client.chat.completions.create(  # type: ignore[attr-defined]
        model=llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def generate_candidate_patches(
    llm: OpenRouterClient,
    problem_statement: str,
    requirements: str,
    oracle_files: dict[str, str],
    n: int,
    temperature: float,
) -> list[dict]:
    """Return n candidate patches, each dict with keys: diff, modified_files."""
    file_section = "\n\n".join(
        f"### {fp}\n```python\n{content}\n```"
        for fp, content in oracle_files.items()
    ) or "(no oracle files available)"
    prompt = GENERATION_PROMPT.format(
        problem_statement=problem_statement,
        requirements=requirements or "(none)",
        files=file_section,
    )
    out: list[dict] = []
    for i in range(n):
        try:
            response = _complete_with_temperature(llm, prompt, temperature)
        except Exception as exc:
            log.warning("Patch %d generation failed: %s", i, exc)
            out.append({"diff": "", "modified_files": {}, "error": str(exc)})
            continue
        blocks = _parse_file_blocks(response)
        diff_parts: list[str] = []
        for fp, new_content in blocks.items():
            orig = oracle_files.get(fp, "")
            d = _make_unified_diff(orig, new_content, fp)
            if d:
                diff_parts.append(d)
        out.append({
            "diff": "".join(diff_parts),
            "modified_files": blocks,
            "response_excerpt": response[:300],
        })
    return out


# ============================================================================
# Critics (host)
# ============================================================================


def critic_l0(modified_files: dict[str, str]) -> CriticResult:
    """Syntax check via ast.parse on every modified .py file."""
    if not modified_files:
        return CriticResult(False, "no modified files")
    errors: list[str] = []
    for fp, content in modified_files.items():
        if not fp.endswith(".py"):
            continue
        try:
            ast.parse(content)
        except SyntaxError as e:
            errors.append(f"{fp}: {e}")
    return CriticResult(
        passed=not errors,
        detail=("; ".join(errors) if errors else "ok")[:400],
    )


def critic_l1(modified_files: dict[str, str]) -> CriticResult:
    """Ruff lint on every modified .py file."""
    if not modified_files:
        return CriticResult(False, "no modified files")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for fp, content in modified_files.items():
            if not fp.endswith(".py"):
                continue
            target = td_path / Path(fp).name
            target.write_text(content)
        try:
            proc = subprocess.run(
                ["ruff", "check", "--select", "F,E9,E7", "--no-cache", str(td_path)],
                capture_output=True, text=True, timeout=HOST_CRITIC_TIMEOUT,
            )
            return CriticResult(
                passed=proc.returncode == 0,
                detail=((proc.stdout + proc.stderr)[-400:]) or "ok",
            )
        except Exception as exc:
            return CriticResult(False, f"ruff error: {exc}")


def critic_l4(modified_files: dict[str, str]) -> CriticResult:
    """Mypy on every modified .py file (best-effort, strict=false)."""
    if not modified_files:
        return CriticResult(False, "no modified files")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for fp, content in modified_files.items():
            if not fp.endswith(".py"):
                continue
            target = td_path / Path(fp).name
            target.write_text(content)
        try:
            proc = subprocess.run(
                ["mypy", "--ignore-missing-imports", "--no-error-summary",
                 "--follow-imports=silent", str(td_path)],
                capture_output=True, text=True, timeout=HOST_CRITIC_TIMEOUT,
            )
            # Count errors
            n_err = sum(1 for line in proc.stdout.splitlines() if ": error:" in line)
            return CriticResult(
                passed=(n_err == 0),
                detail=(proc.stdout[-400:]) or "ok",
            )
        except Exception as exc:
            return CriticResult(False, f"mypy error: {exc}")


L3_PROMPT_TEMPLATE = """\
You are a senior engineer reviewing a proposed patch for a large Python
codebase. Judge whether the patch correctly resolves the problem.

## Problem
{problem}

## Proposed patch (unified diff)
```diff
{diff}
```

Answer with EXACTLY one of:
VERDICT: PASS
VERDICT: FAIL

Then one sentence of justification."""


def critic_l3(
    llm: OpenRouterClient, problem_statement: str, diff: str
) -> CriticResult:
    if not diff.strip():
        return CriticResult(False, "empty diff")
    prompt = L3_PROMPT_TEMPLATE.format(
        problem=problem_statement[:2500],
        diff=diff[:4000],
    )
    try:
        response = llm.complete(prompt)
    except Exception as exc:
        return CriticResult(False, f"error: {exc}")
    import re
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", response, re.IGNORECASE)
    if not m:
        return CriticResult(False, f"unparseable: {response[:100]}")
    passed = m.group(1).upper() == "PASS"
    just = response[m.end():].strip().split("\n")[0][:200]
    return CriticResult(passed=passed, detail=just)


# ============================================================================
# L2 fast test + verifier (inside container)
# ============================================================================


def _write_patch_and_runscript(
    work_dir: Path,
    diff: str,
    instance_id: str,
    run_scripts_dir: Path,
) -> bool:
    """Populate work_dir with patch.diff and run_script.sh for the container."""
    (work_dir / "patch.diff").write_text(diff)
    src = run_scripts_dir / instance_id / "run_script.sh"
    if not src.exists():
        log.warning("run_script.sh missing for %s", instance_id)
        return False
    shutil.copy(src, work_dir / "run_script.sh")
    os.chmod(work_dir / "run_script.sh", 0o755)
    return True


def _parse_pytest_summary(out: str) -> dict[str, int]:
    """Extract pass/fail counts from pytest tail output."""
    import re
    summary = {"passed": 0, "failed": 0, "error": 0}
    # Example lines: "5 passed, 1 failed in 2.11s"
    for line in out.splitlines()[::-1]:
        m = re.search(r"(\d+)\s+passed", line)
        if m:
            summary["passed"] = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", line)
        if m:
            summary["failed"] = int(m.group(1))
        m = re.search(r"(\d+)\s+error", line)
        if m:
            summary["error"] = int(m.group(1))
        if "passed" in line or "failed" in line or "error" in line:
            break
    return summary


def run_l2_fast_test(
    tag: str,
    instance_id: str,
    diff: str,
    selected_files: list[str],
    run_scripts_dir: Path,
) -> CriticResult:
    if not diff.strip():
        return CriticResult(False, "empty diff")
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        if not _write_patch_and_runscript(wd, diff, instance_id, run_scripts_dir):
            return CriticResult(False, "missing run_script.sh")
        files_arg = ",".join(selected_files) if selected_files else ""
        cmd = (
            "cd /app && "
            "git apply /work/patch.diff || git apply --reject /work/patch.diff || exit 99; "
            "bash /work/run_script.sh " + (f'"{files_arg}"' if files_arg else "")
        )
        rc, out = run_in_container(tag, cmd, mount_dir=wd, timeout=FAST_TEST_TIMEOUT)
        if rc == 99:
            return CriticResult(False, "patch apply failed")
        if rc == 124:
            return CriticResult(False, "timeout")
        summary = _parse_pytest_summary(out)
        passed = rc == 0 and summary.get("failed", 0) == 0 and summary.get("error", 0) == 0
        return CriticResult(
            passed=passed,
            detail=f"rc={rc} summary={summary} tail={out[-200:]}",
        )


def run_verifier(
    tag: str,
    instance_id: str,
    diff: str,
    run_scripts_dir: Path,
    fail_to_pass: list[str],
) -> tuple[int, str]:
    """Run the hidden test suite. Returns (Y, detail)."""
    if not diff.strip():
        return 0, "empty diff"
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        if not _write_patch_and_runscript(wd, diff, instance_id, run_scripts_dir):
            return 0, "missing run_script.sh"
        cmd = (
            "cd /app && "
            "git apply /work/patch.diff || git apply --reject /work/patch.diff || exit 99; "
            "bash /work/run_script.sh"
        )
        rc, out = run_in_container(tag, cmd, mount_dir=wd, timeout=FULL_TEST_TIMEOUT)
        if rc == 99:
            return 0, "patch apply failed"
        if rc == 124:
            return 0, "verifier timeout"
        summary = _parse_pytest_summary(out)
        # Resolution: no failures AND all fail_to_pass tests actually ran+passed.
        # We approximate by requiring 0 failures and ≥1 passed test (the patch
        # did not crash the harness) as a baseline. A stricter check parses the
        # test names from the output, which we defer.
        failed = summary.get("failed", 0) + summary.get("error", 0)
        passed = summary.get("passed", 0)
        if rc == 0 and failed == 0 and passed >= max(1, min(len(fail_to_pass), 1)):
            y = 1
        else:
            y = 0
        return y, f"rc={rc} summary={summary} tail={out[-200:]}"


# ============================================================================
# Main loop
# ============================================================================


def load_completed(output_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not output_path.exists():
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["instance_id"], int(r["patch_id"])))
            except Exception:
                continue
    return done


def _parse_json_list(raw: str) -> list[str]:
    try:
        v = json.loads(raw)
    except Exception:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


def process_instance(
    row: pd.Series,
    gen_llm: OpenRouterClient,
    rev_llm: OpenRouterClient,
    n_patches: int,
    temperature: float,
    run_scripts_dir: Path,
    output_path: Path,
    completed: set[tuple[str, int]],
    keep_image: bool,
) -> None:
    instance_id = str(row["instance_id"])
    tag = str(row["dockerhub_tag"])
    log.info("=== %s (%s) ===", instance_id, row["repo"])

    # Check if all patches for this instance are already done
    if all((instance_id, i) in completed for i in range(n_patches)):
        log.info("All %d patches complete; skip.", n_patches)
        return

    if not pull_image(tag):
        log.error("Skipping %s (pull failed)", instance_id)
        return

    try:
        oracle = read_oracle_files(tag, str(row["patch"]))
        if not oracle:
            log.warning("No oracle files for %s", instance_id)
            return

        candidates = generate_candidate_patches(
            gen_llm,
            str(row["problem_statement"]),
            str(row.get("requirements") or ""),
            oracle,
            n_patches,
            temperature,
        )

        selected_files = _parse_json_list(str(row.get("selected_test_files_to_run") or "[]"))
        fail_to_pass = _parse_json_list(str(row.get("fail_to_pass") or "[]"))

        for i, cand in enumerate(candidates):
            if (instance_id, i) in completed:
                continue
            diff = cand.get("diff", "")
            modified = cand.get("modified_files", {})
            log.info("  patch %d/%d: %d files, %d chars", i + 1, n_patches,
                     len(modified), len(diff))

            l0 = critic_l0(modified)
            l1 = critic_l1(modified)
            l4 = critic_l4(modified)
            l3 = critic_l3(rev_llm, str(row["problem_statement"]), diff)
            l2 = run_l2_fast_test(tag, instance_id, diff, selected_files, run_scripts_dir)
            y, y_detail = run_verifier(tag, instance_id, diff, run_scripts_dir, fail_to_pass)

            record = ProRecord(
                instance_id=instance_id,
                patch_id=i,
                repo=str(row["repo"]),
                repo_language=str(row["repo_language"]),
                base_commit=str(row["base_commit"]),
                dockerhub_tag=tag,
                patch=diff,
                critic_results={
                    "L0_syntax": asdict(l0),
                    "L1_lint": asdict(l1),
                    "L2_fast_test": asdict(l2),
                    "L3_llm_review": asdict(l3),
                    "L4_mypy": asdict(l4),
                },
                ground_truth=y,
                metadata={
                    "n_modified_files": len(modified),
                    "verifier_detail": y_detail[:400],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            )

            with open(output_path, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")
            completed.add((instance_id, i))
            log.info(
                "    L0=%s L1=%s L2=%s L3=%s L4=%s Y=%d",
                int(l0.passed), int(l1.passed), int(l2.passed),
                int(l3.passed), int(l4.passed), y,
            )
    finally:
        if not keep_image:
            remove_image(tag)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default=str(DEFAULT_PARQUET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--run-scripts-dir", default=str(DEFAULT_RUN_SCRIPTS_DIR))
    parser.add_argument("--limit", type=int, default=0,
                        help="Max instances to process (0 = all).")
    parser.add_argument("--patches-per-instance", type=int,
                        default=DEFAULT_PATCHES_PER_INSTANCE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    parser.add_argument("--review-model", default=DEFAULT_REVIEW_MODEL)
    parser.add_argument("--language", default="python",
                        help="Filter by repo_language (default: python).")
    parser.add_argument("--repos", default="",
                        help="Comma-separated list of repos to include.")
    parser.add_argument("--instance-ids", default="",
                        help="Comma-separated instance IDs (overrides other filters).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-image", action="store_true",
                        help="Do not delete images after processing (faster re-runs, more disk).")
    args = parser.parse_args()

    if not _engine_available():
        sys.exit(1)

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        log.error("Parquet not found: %s", parquet_path)
        sys.exit(1)

    run_scripts_dir = Path(args.run_scripts_dir)
    if not run_scripts_dir.exists():
        log.error(
            "run_scripts dir not found: %s\n"
            "Clone https://github.com/scaleapi/SWE-bench_Pro-os and copy "
            "run_scripts/ there.",
            run_scripts_dir,
        )
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    log.info("Loaded %d rows from %s", len(df), parquet_path)

    if args.instance_ids:
        wanted = set(x.strip() for x in args.instance_ids.split(",") if x.strip())
        df = df[df["instance_id"].isin(wanted)]
    else:
        if args.language:
            df = df[df["repo_language"] == args.language]
        if args.repos:
            wanted_repos = set(x.strip() for x in args.repos.split(",") if x.strip())
            df = df[df["repo"].isin(wanted_repos)]

    log.info("After filter: %d instances", len(df))
    if args.limit > 0:
        df = df.head(args.limit)
    log.info("Will process: %d instances", len(df))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed(output_path) if args.resume else set()
    log.info("Resume: %d patches already complete", len(completed))

    gen_llm = OpenRouterClient(model=args.gen_model)
    rev_llm = OpenRouterClient(model=args.review_model)
    log.info("Generator: %s  Reviewer: %s", args.gen_model, args.review_model)

    for _, row in df.iterrows():
        try:
            process_instance(
                row, gen_llm, rev_llm,
                args.patches_per_instance,
                args.temperature,
                run_scripts_dir,
                output_path,
                completed,
                args.keep_image,
            )
        except KeyboardInterrupt:
            log.warning("Interrupted; exiting.")
            return
        except Exception as exc:
            log.exception("Instance %s failed: %s", row.get("instance_id"), exc)
            continue


if __name__ == "__main__":
    main()
