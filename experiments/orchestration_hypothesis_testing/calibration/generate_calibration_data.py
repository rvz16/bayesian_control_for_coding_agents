#!/usr/bin/env python3
"""Generate calibration data for the orchestration-as-hypothesis-testing experiment.

For each SWE-bench instance, generates N patches using an LLM, runs tiered critics
(L0: syntax, L1: lint, L2: fast test) on each patch, and runs the full test suite
to get ground-truth labels Y in {0, 1}.

The output is a JSONL file where each line contains:
    {instance_id, patch_id, patch, critic_results, ground_truth, metadata}

This calibration data is used by compute_likelihoods.py to estimate the confusion
matrix P(z|Y) for each critic level, which the Bayesian controller needs for
belief updates.

Usage:
    # Quick test on 5 instances
    python generate_calibration_data.py --limit 5 --patches-per-instance 3

    # Full calibration run
    python generate_calibration_data.py --limit 300 --patches-per-instance 3

    # Resume interrupted run
    python generate_calibration_data.py --limit 300 --resume
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from datasets import load_dataset

# Project root
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Load .env from project root
load_dotenv(ROOT / ".env")

from sage_agent.llm.openrouter import OpenRouterClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Defaults
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"
DEFAULT_PATCHES_PER_INSTANCE = 3
DEFAULT_TEMPERATURE = 0.8
PATCH_GEN_TIMEOUT = 60  # seconds for LLM call
TEST_TIMEOUT = 300  # seconds for full test suite
FAST_TEST_TIMEOUT = 60  # seconds for single test file
LINT_TIMEOUT = 30

# Python binary for test execution (may need older Python for SWE-bench repos)
# Use conda env if available, otherwise system python
CONDA_PY39 = Path.home() / "miniconda3/envs/swebench_py39/bin/python"
TEST_PYTHON = str(CONDA_PY39) if CONDA_PY39.exists() else "python"

# Patch application strategies (from SWE-bench harness:
# https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/run_evaluation.py)
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


# ============================================================================
# Data structures
# ============================================================================

@dataclass(frozen=True)
class CriticResult:
    passed: bool
    detail: str


@dataclass(frozen=True)
class PatchCalibrationRecord:
    instance_id: str
    patch_id: int
    patch: str
    critic_results: dict[str, dict[str, object]]
    ground_truth: int  # 0 or 1
    metadata: dict[str, str]


# ============================================================================
# Patch generation (oracle retrieval + complete file output)
# ============================================================================

MAX_FILE_LINES = 3000  # Allow large files — Claude Sonnet 4 handles 200K context

PATCH_PROMPT_TEMPLATE = """\
You are an expert software engineer fixing a bug in the {repo} repository.

## Issue Description
{problem_statement}

{hints_section}

## Files that likely need changes

{file_contents}

## Task
Fix the issue by modifying the file(s) above. Output your changes as a SEARCH/REPLACE
block for each change:

<<<CHANGE path/to/file.py
SEARCH
(exact lines from the original file that you want to replace)
REPLACE
(the new lines that should replace the search block)
CHANGE>>>

You can output multiple CHANGE blocks if needed. The SEARCH text must match the
original file exactly (including indentation). Keep changes minimal and focused.
Do NOT include unchanged files. Do NOT output entire files.
"""


def _read_oracle_files(repo_path: Path, gold_patch: str) -> dict[str, str]:
    """Read the files that the gold patch modifies (oracle retrieval).

    This gives the model actual file content so it can produce valid edits.
    We use the gold patch file paths but NOT the gold patch content.
    """
    file_paths = _get_changed_files_from_patch(gold_patch)
    contents: dict[str, str] = {}
    for fpath in file_paths:
        full_path = repo_path / fpath
        if not full_path.exists():
            continue
        try:
            text = full_path.read_text(errors="replace")
            lines = text.split("\n")
            if len(lines) > MAX_FILE_LINES:
                text = "\n".join(lines[:MAX_FILE_LINES]) + f"\n... (truncated, {len(lines)} lines total)"
            contents[fpath] = text
        except Exception:
            continue
    return contents


def _format_file_contents(files: dict[str, str]) -> str:
    """Format file contents for the prompt."""
    parts: list[str] = []
    for fpath, content in files.items():
        parts.append(f"### {fpath}\n```python\n{content}\n```")
    return "\n\n".join(parts) if parts else "(no files available)"


def _make_diff(original: str, modified: str, file_path: str) -> str:
    """Compute unified diff between original and modified file content.

    Ensures the diff ends with a newline (required by `patch` command)
    and adds 'No newline at end of file' markers when needed.
    """
    import difflib

    # Ensure both strings end with newline for clean diffs
    if original and not original.endswith("\n"):
        original += "\n"
    if modified and not modified.endswith("\n"):
        modified += "\n"

    orig_lines = original.splitlines(keepends=True)
    mod_lines = modified.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, mod_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    result = "".join(diff)
    # Ensure trailing newline (patch command requires it)
    if result and not result.endswith("\n"):
        result += "\n"
    return result


def _parse_file_blocks(response: str) -> dict[str, str]:
    """Parse <<<FILE path ... FILE>>> blocks from LLM response."""
    blocks: dict[str, str] = {}
    pattern = re.compile(
        r"<<<FILE\s+(.+?)\s*\n([\s\S]*?)FILE>>>",
        re.MULTILINE,
    )
    for match in pattern.finditer(response):
        fpath = match.group(1).strip()
        content = match.group(2)
        if content.endswith("\n"):
            content = content[:-1]
        blocks[fpath] = content
    return blocks


def _parse_change_blocks(response: str) -> list[tuple[str, str, str]]:
    """Parse <<<CHANGE path ... CHANGE>>> blocks with SEARCH/REPLACE.

    Returns list of (file_path, search_text, replace_text).
    """
    changes: list[tuple[str, str, str]] = []
    pattern = re.compile(
        r"<<<CHANGE\s+(.+?)\s*\n([\s\S]*?)CHANGE>>>",
        re.MULTILINE,
    )
    for match in pattern.finditer(response):
        fpath = match.group(1).strip()
        body = match.group(2)

        # Split on SEARCH and REPLACE markers
        parts = re.split(r"^SEARCH\s*$", body, maxsplit=1, flags=re.MULTILINE)
        if len(parts) < 2:
            continue
        rest = parts[1]

        parts2 = re.split(r"^REPLACE\s*$", rest, maxsplit=1, flags=re.MULTILINE)
        if len(parts2) < 2:
            continue

        search_text = parts2[0].strip("\n")
        replace_text = parts2[1].strip("\n")
        changes.append((fpath, search_text, replace_text))

    return changes


def _apply_changes_to_content(
    original: str,
    changes: list[tuple[str, str]],
) -> str:
    """Apply search/replace changes to file content."""
    result = original
    for search, replace in changes:
        if search in result:
            result = result.replace(search, replace, 1)
        else:
            # Try with relaxed whitespace matching
            search_stripped = "\n".join(l.rstrip() for l in search.split("\n"))
            result_stripped = "\n".join(l.rstrip() for l in result.split("\n"))
            if search_stripped in result_stripped:
                # Find the position in stripped version, apply to original
                result = result.replace(
                    search.rstrip(),
                    replace.rstrip(),
                    1,
                )
    return result


@dataclass(frozen=True)
class GeneratedPatch:
    """A generated patch with both diff and direct file modifications."""
    diff: str
    modified_files: dict[str, str]  # fpath -> complete modified content


def generate_patches(
    llm: OpenRouterClient,
    problem_statement: str,
    repo: str,
    hints: str,
    n_patches: int,
    temperature: float,
    repo_path: Optional[Path] = None,
    gold_patch: str = "",
) -> list[GeneratedPatch]:
    """Generate N diverse patches for a SWE-bench instance.

    Uses oracle retrieval: reads the files that the gold patch modifies
    and includes their content in the prompt. The model outputs complete
    modified files. We store both the raw modified files (for direct
    application) and the computed diff (for the calibration record).
    """
    hints_section = f"## Hints\n{hints}" if hints else ""

    # Oracle retrieval: read files the gold patch touches
    oracle_files: dict[str, str] = {}
    if repo_path and gold_patch:
        oracle_files = _read_oracle_files(repo_path, gold_patch)

    file_contents = _format_file_contents(oracle_files)
    prompt = PATCH_PROMPT_TEMPLATE.format(
        repo=repo,
        problem_statement=problem_statement,
        hints_section=hints_section,
        file_contents=file_contents,
    )

    patches: list[GeneratedPatch] = []
    for i in range(n_patches):
        try:
            response = llm.complete(prompt)

            # Try CHANGE blocks first (search/replace format)
            change_blocks = _parse_change_blocks(response)
            if change_blocks:
                # Group changes by file
                file_changes: dict[str, list[tuple[str, str]]] = {}
                for fpath, search, replace in change_blocks:
                    # Resolve path
                    resolved = fpath
                    if fpath not in oracle_files:
                        for opath in oracle_files:
                            if opath.endswith(fpath) or fpath.endswith(opath):
                                resolved = opath
                                break
                    if resolved not in file_changes:
                        file_changes[resolved] = []
                    file_changes[resolved].append((search, replace))

                # Apply changes to get modified files
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
                    patches.append(GeneratedPatch(
                        diff="\n".join(diffs),
                        modified_files=resolved_files,
                    ))
                    continue

            # Fallback: try FILE blocks (complete file format)
            blocks = _parse_file_blocks(response)
            if blocks:
                diffs = []
                resolved_files = {}
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

                patches.append(GeneratedPatch(
                    diff="\n".join(diffs),
                    modified_files=resolved_files,
                ))
            else:
                # Last fallback: try raw diff extraction
                diff = _response_to_patch(response, oracle_files)
                patches.append(GeneratedPatch(diff=diff, modified_files={}))
        except Exception as e:
            log.warning("  Patch %d generation failed: %s", i, e)
            patches.append(GeneratedPatch(diff="", modified_files={}))

    return patches


def _response_to_patch(response: str, oracle_files: dict[str, str]) -> str:
    """Convert LLM response (complete file blocks) to a unified diff patch.

    If the response contains <<<FILE blocks, computes diff against originals.
    Falls back to extracting raw diff from response if no blocks found.
    """
    # Try parsing <<<FILE blocks first
    blocks = _parse_file_blocks(response)
    if blocks:
        diffs: list[str] = []
        for fpath, modified_content in blocks.items():
            original = oracle_files.get(fpath, "")
            if not original:
                # Try without leading path components
                for opath, ocontent in oracle_files.items():
                    if opath.endswith(fpath) or fpath.endswith(opath):
                        original = ocontent
                        fpath = opath
                        break
            diff = _make_diff(original, modified_content, fpath)
            if diff:
                diffs.append(diff)
        return "\n".join(diffs) if diffs else ""

    # Fallback: try extracting raw diff from response
    diff_match = re.search(r"```(?:diff)?\n([\s\S]*?)```", response)
    if diff_match:
        return diff_match.group(1).strip()

    diff_pattern = re.search(
        r"(---\s+a/.*?\n\+\+\+\s+b/.*?\n[\s\S]*?)(?:\n\n|$)", response
    )
    if diff_pattern:
        return diff_pattern.group(1).strip()

    return ""


# ============================================================================
# Critics
# ============================================================================

def _get_changed_files_from_patch(patch: str) -> list[str]:
    """Extract file paths modified by a unified diff patch."""
    files: list[str] = []
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            files.append(line[6:])
        elif line.startswith("--- a/"):
            pass
    return files


def _apply_patch(cwd: Path, patch: str) -> subprocess.CompletedProcess[str]:
    """Apply a unified diff using the same fallback chain as the SWE-bench harness.

    Tries multiple strategies in order of strictness:
    1. git apply --verbose (strict)
    2. git apply --verbose --reject (applies what it can)
    3. patch --batch --fuzz=5 -p1 (maximum fuzz tolerance)

    For strategy 3 (patch command), we write to a temp file since `patch -i`
    requires a file path rather than stdin for reliable behavior.
    """
    # Write patch to temp file for the `patch` command fallback
    patch_file = cwd / "_tmp_patch.diff"
    patch_file.write_text(patch)

    last_result = None
    for cmd_template in GIT_APPLY_CMDS:
        if cmd_template.endswith("-i"):
            cmd = f"{cmd_template} {patch_file}"
        else:
            cmd = cmd_template

        try:
            if "git apply" in cmd:
                last_result = subprocess.run(
                    cmd.split(),
                    cwd=cwd,
                    input=patch,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
            else:
                # patch command uses -i flag with file
                last_result = subprocess.run(
                    f"{cmd_template} {patch_file}",
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        except subprocess.TimeoutExpired:
            continue

        if last_result.returncode == 0:
            patch_file.unlink(missing_ok=True)
            return last_result

    patch_file.unlink(missing_ok=True)
    return last_result or subprocess.CompletedProcess(
        args="", returncode=1, stdout="", stderr="all apply strategies failed"
    )


def _get_patched_content(repo_path: Path, patch: str, file_path: str) -> Optional[str]:
    """Apply patch in-memory and return the patched file content."""
    original_file = repo_path / file_path
    if not original_file.exists():
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_repo = Path(tmpdir) / "repo"
        tmp_file = tmp_repo / file_path
        tmp_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_file, tmp_file)

        result = _apply_patch(tmp_repo, patch)
        if result.returncode != 0:
            return None

        patched_file = tmp_repo / file_path
        if patched_file.exists():
            return patched_file.read_text()
    return None


def _reset_repo(repo_path: Path) -> None:
    """Reset repo to clean state (undo applied patches)."""
    subprocess.run(
        ["git", "checkout", "-f", "."],
        cwd=repo_path, capture_output=True,
    )
    subprocess.run(
        ["git", "clean", "-fd"],
        cwd=repo_path, capture_output=True,
    )


def _apply_modified_files(repo_path: Path, modified_files: dict[str, str]) -> bool:
    """Apply modifications by directly writing file contents to the repo.

    This bypasses diff/patch entirely — the model outputs complete file
    content, and we just overwrite the files. Guaranteed to "apply".
    """
    for fpath, content in modified_files.items():
        target = repo_path / fpath
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return True


def evaluate_patch(
    patch: str,
    repo_path: Path,
    test_patch: str,
    modified_files: Optional[dict[str, str]] = None,
    swebench_eval: bool = False,
    instance_id: str = "",
    model_name: str = "",
) -> tuple[CriticResult, CriticResult, CriticResult, int]:
    """Run all critics and verifier on a single patch IN-PLACE on the repo.

    If modified_files is provided (from oracle retrieval), applies changes
    by directly writing files (always succeeds). Falls back to diff-based
    application if modified_files is not available.

    If swebench_eval is True, uses the SWE-bench Docker harness for ground
    truth evaluation instead of running tests directly (handles deps/env).

    Returns: (l0_syntax, l1_lint, l2_fast_test, ground_truth)
    """
    test_files = _get_changed_files_from_patch(test_patch)
    test_files = [f for f in test_files if "test" in f.lower()]

    # Determine changed Python files
    if modified_files:
        py_files = [f for f in modified_files if f.endswith(".py")]
    else:
        py_files = [f for f in _get_changed_files_from_patch(patch) if f.endswith(".py")]

    # --- L0: Syntax check (check content before applying) ---
    l0 = CriticResult(passed=True, detail="")
    if py_files:
        errors: list[str] = []
        for fpath in py_files:
            content = modified_files.get(fpath) if modified_files else _get_patched_content(repo_path, patch, fpath)
            if content is None:
                continue
            try:
                ast.parse(content, filename=fpath)
            except SyntaxError as e:
                errors.append(f"{fpath}:{e.lineno}: {e.msg}")
        if errors:
            l0 = CriticResult(passed=False, detail="; ".join(errors[:3]))

    # --- Apply changes to repo in-place ---
    if modified_files:
        _apply_modified_files(repo_path, modified_files)
        patch_applied = True
    else:
        apply_result = _apply_patch(repo_path, patch)
        patch_applied = apply_result.returncode == 0

    if not patch_applied:
        _reset_repo(repo_path)
        fail = CriticResult(passed=False, detail=f"patch apply failed: {apply_result.stderr[:200]}")
        return l0, fail, fail, 0

    # --- L1: Lint on changed files ---
    l1 = CriticResult(passed=True, detail="")
    if py_files:
        abs_files = [str(repo_path / f) for f in py_files if (repo_path / f).exists()]
        if abs_files:
            try:
                result = subprocess.run(
                    ["ruff", "check", "--select=E,F"] + abs_files,
                    cwd=repo_path,
                    capture_output=True, text=True,
                    timeout=LINT_TIMEOUT,
                )
                if result.returncode != 0:
                    errors = result.stdout.strip().split("\n")[:5]
                    l1 = CriticResult(passed=False, detail="; ".join(errors))
            except FileNotFoundError:
                l1 = CriticResult(passed=True, detail="ruff not installed, skipped")
            except subprocess.TimeoutExpired:
                l1 = CriticResult(passed=False, detail="ruff timeout")

    # --- L2: Fast test (apply test_patch too, run specific test files) ---
    l2 = CriticResult(passed=False, detail="no test files")
    if test_files:
        test_apply = _apply_patch(repo_path, test_patch)
        if test_apply.returncode != 0:
            l2 = CriticResult(passed=False, detail=f"test patch apply failed: {test_apply.stderr[:200]}")
        else:
            test_cmd = [TEST_PYTHON, "-m", "pytest", "-x", "--tb=short"] + test_files
            try:
                result = subprocess.run(
                    test_cmd, cwd=repo_path,
                    capture_output=True, text=True,
                    timeout=FAST_TEST_TIMEOUT,
                )
                if result.returncode == 0:
                    l2 = CriticResult(passed=True, detail="")
                else:
                    l2 = CriticResult(
                        passed=False,
                        detail=f"exit={result.returncode}; {result.stdout[-200:]}",
                    )
            except subprocess.TimeoutExpired:
                l2 = CriticResult(passed=False, detail="fast test timeout")

    # Reset before verifier (need clean state to re-apply)
    _reset_repo(repo_path)

    # --- Verifier: ground truth ---
    ground_truth = 0

    if swebench_eval and instance_id and patch:
        # Use SWE-bench Docker harness for reliable ground truth
        ground_truth = _swebench_docker_eval(instance_id, patch, model_name)
    else:
        # Fallback: direct test execution (use modified_files for reliable apply)
        if modified_files:
            _apply_modified_files(repo_path, modified_files)
        else:
            apply_result = _apply_patch(repo_path, patch)
            if apply_result.returncode != 0:
                _reset_repo(repo_path)
                return l0, l1, l2, 0

        test_apply = _apply_patch(repo_path, test_patch)
        if test_apply.returncode == 0 and test_files:
            test_cmd = [TEST_PYTHON, "-m", "pytest", "-x", "--tb=short"] + test_files
            try:
                result = subprocess.run(
                    test_cmd, cwd=repo_path,
                    capture_output=True, text=True,
                    timeout=TEST_TIMEOUT,
                )
                ground_truth = 1 if result.returncode == 0 else 0
            except subprocess.TimeoutExpired:
                ground_truth = 0

    # Final reset
    _reset_repo(repo_path)
    return l0, l1, l2, ground_truth


def _swebench_docker_eval(instance_id: str, patch: str, model_name: str) -> int:
    """Evaluate a patch using SWE-bench Docker harness for reliable ground truth.

    Creates a temporary predictions file and runs swebench evaluation
    in a Docker container. Returns 1 if patch passes, 0 otherwise.
    """
    import tempfile

    pred = {
        "instance_id": instance_id,
        "model_name_or_path": model_name or "calibration",
        "model_patch": patch,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="swebench_pred_"
    ) as f:
        f.write(json.dumps(pred) + "\n")
        pred_path = f.name

    try:
        result = subprocess.run(
            [
                "python", "-m", "swebench.harness.run_evaluation",
                "--dataset_name", "princeton-nlp/SWE-bench_Lite",
                "--predictions_path", pred_path,
                "--max_workers", "1",
                "--run_id", f"cal_{instance_id.replace('/', '_')}",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max per instance
        )

        # Check the evaluation report for pass/fail
        if result.returncode == 0:
            # SWE-bench writes results to a report file
            # Parse stdout for resolved status
            if "RESOLVED" in result.stdout or '"resolved": true' in result.stdout.lower():
                return 1
            # Also check for the instance in resolved list
            if instance_id in result.stdout and "resolved" in result.stdout.lower():
                return 1
    except subprocess.TimeoutExpired:
        log.warning("SWE-bench eval timed out for %s", instance_id)
    except Exception as e:
        log.warning("SWE-bench eval failed for %s: %s", instance_id, e)
    finally:
        Path(pred_path).unlink(missing_ok=True)

    return 0


# ============================================================================
# Repo management
# ============================================================================

def _install_repo_deps(repo: str, repo_path: Path) -> None:
    """Install minimal dependencies needed to run tests for a given repo.

    Uses the conda swebench_py39 env's pip to install into that env.
    Only installs once per repo clone (creates a .deps_installed marker).
    """
    # Track installed repos globally (deps go into py39 env, not the repo)
    if not hasattr(_install_repo_deps, '_installed'):
        _install_repo_deps._installed = set()
    if repo in _install_repo_deps._installed:
        return

    pip = str(Path.home() / "miniconda3/envs/swebench_py39/bin/pip")
    if not Path(pip).exists():
        pip = "pip"

    # Repo-specific dependency installation
    deps_map: dict[str, list[str]] = {
        "sympy/sympy": ["mpmath"],
        "psf/requests": ["pytest-httpbin", "pytest-mock", "trustme"],
        "pallets/flask": ["pytest", "werkzeug", "jinja2", "markupsafe", "itsdangerous", "click", "blinker"],
        "pytest-dev/pytest": [],  # pytest can test itself
        "scikit-learn/scikit-learn": ["numpy", "scipy", "cython", "joblib", "threadpoolctl"],
        "matplotlib/matplotlib": ["numpy", "pyparsing", "cycler", "kiwisolver", "pillow"],
        "pydata/xarray": ["numpy", "pandas"],
        "mwaskom/seaborn": ["numpy", "pandas", "matplotlib"],
        "sphinx-doc/sphinx": ["docutils", "jinja2", "pygments", "snowballstemmer", "babel",
                               "alabaster", "imagesize", "packaging", "requests"],
        "pylint-dev/pylint": ["astroid", "isort", "mccabe", "tomlkit"],
    }

    deps = deps_map.get(repo, [])
    if deps:
        log.info("  Installing deps for %s: %s", repo, deps)
        subprocess.run(
            [pip, "install", "-q"] + deps,
            capture_output=True,
            timeout=120,
        )

    # Skip pip install -e for repos that work from source (sympy, etc.)
    skip_install = {"sympy/sympy", "django/django"}
    if repo not in skip_install:
        setup_py = repo_path / "setup.py"
        pyproject = repo_path / "pyproject.toml"
        if setup_py.exists() or pyproject.exists():
            log.info("  Installing %s in dev mode...", repo)
            result = subprocess.run(
                [pip, "install", "-e", ".", "--no-deps", "-q"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                log.warning("  Dev install failed (non-critical): %s", result.stderr[:200])

    _install_repo_deps._installed.add(repo)


def setup_repo(repo: str, base_commit: str, workdir: Path) -> Path:
    """Clone repository and checkout base commit. Reuses existing clones."""
    repo_name = repo.replace("/", "__")
    repo_path = workdir / repo_name

    if repo_path.exists():
        # Reset to base commit instead of re-cloning
        subprocess.run(
            ["git", "checkout", "-f", base_commit],
            cwd=repo_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_path,
            capture_output=True,
        )
        _install_repo_deps(repo, repo_path)
        return repo_path

    log.info("Cloning %s...", repo)
    subprocess.run(
        ["git", "clone", f"https://github.com/{repo}.git", str(repo_path)],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=repo_path,
        capture_output=True,
        check=True,
    )

    _install_repo_deps(repo, repo_path)
    return repo_path


# ============================================================================
# Main pipeline
# ============================================================================

def load_completed_ids(output_file: Path) -> set[str]:
    """Load instance_id+patch_id pairs already in the output file."""
    completed: set[str] = set()
    if not output_file.exists():
        return completed

    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = f"{record['instance_id']}_{record['patch_id']}"
                completed.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def run_calibration(args: argparse.Namespace) -> None:
    """Main calibration pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "raw_results.jsonl"

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="calibration_"))
    workdir.mkdir(parents=True, exist_ok=True)

    # Load completed records for resume
    completed = load_completed_ids(output_file) if args.resume else set()
    if completed:
        log.info("Resuming: %d patch records already completed", len(completed))

    # Initialize LLM
    llm = OpenRouterClient(model=args.model, verbose=args.verbose)
    log.info("LLM: %s", args.model)

    # Load dataset
    dataset_map = {
        "lite": "princeton-nlp/SWE-bench_Lite",
        "verified": "princeton-nlp/SWE-bench_Verified",
        "full": "princeton-nlp/SWE-bench",
    }
    dataset_name = dataset_map[args.dataset]
    log.info("Loading %s...", dataset_name)
    dataset = load_dataset(dataset_name, split="test")

    # Filter by repos if specified
    if args.repos:
        indices = [i for i, d in enumerate(dataset) if d["repo"] in args.repos]
        dataset = dataset.select(indices)
        log.info("Filtered to repos %s: %d instances", args.repos, len(dataset))

    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    log.info("Processing %d instances, %d patches each", len(dataset), args.patches_per_instance)
    log.info("Output: %s", output_file)

    total_records = 0
    total_correct = 0

    for idx, instance in enumerate(dataset):
        instance_id = instance["instance_id"]
        repo = instance["repo"]
        base_commit = instance["base_commit"]
        problem_statement = instance["problem_statement"]
        hints = instance.get("hints_text", "")
        test_patch = instance.get("test_patch", "")

        log.info(
            "[%d/%d] %s (%s)",
            idx + 1, len(dataset), instance_id, repo,
        )

        # Check if all patches for this instance are already done
        all_done = all(
            f"{instance_id}_{i}" in completed
            for i in range(args.patches_per_instance)
        )
        if all_done:
            log.info("  Skipping (all patches already completed)")
            continue

        # Setup repo
        try:
            repo_path = setup_repo(repo, base_commit, workdir)
        except Exception as e:
            log.error("  Failed to setup repo: %s", e)
            continue

        # Get gold patch for oracle file retrieval (we use file paths only, not content)
        gold_patch = instance.get("patch", "")

        # Generate patches with oracle retrieval
        patches = generate_patches(
            llm=llm,
            problem_statement=problem_statement,
            repo=repo,
            hints=hints,
            n_patches=args.patches_per_instance,
            temperature=args.temperature,
            repo_path=repo_path,
            gold_patch=gold_patch,
        )

        # Evaluate each patch
        for patch_id, gen_patch in enumerate(patches):
            key = f"{instance_id}_{patch_id}"
            if key in completed:
                log.info("  Patch %d: skipping (already done)", patch_id)
                continue

            patch = gen_patch.diff
            if not patch and not gen_patch.modified_files:
                log.warning("  Patch %d: empty, recording as all-fail", patch_id)
                record = PatchCalibrationRecord(
                    instance_id=instance_id,
                    patch_id=patch_id,
                    patch="",
                    critic_results={
                        "L0_syntax": {"passed": False, "detail": "empty patch"},
                        "L1_lint": {"passed": False, "detail": "empty patch"},
                        "L2_fast_test": {"passed": False, "detail": "empty patch"},
                    },
                    ground_truth=0,
                    metadata={
                        "model": args.model,
                        "repo": repo,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
                with open(output_file, "a") as f:
                    f.write(json.dumps(asdict(record)) + "\n")
                total_records += 1
                continue

            log.info("  Patch %d: running critics + verifier...", patch_id)

            l0, l1, l2, ground_truth = evaluate_patch(
                patch, repo_path, test_patch,
                modified_files=gen_patch.modified_files or None,
                swebench_eval=args.swebench_eval,
                instance_id=instance_id,
                model_name=args.model,
            )
            log.info("    L0 syntax: %s", "PASS" if l0.passed else f"FAIL ({l0.detail[:60]})")
            log.info("    L1 lint:   %s", "PASS" if l1.passed else f"FAIL ({l1.detail[:60]})")
            log.info("    L2 test:   %s", "PASS" if l2.passed else f"FAIL ({l2.detail[:60]})")
            log.info("    Ground truth: Y=%d", ground_truth)

            total_records += 1
            total_correct += ground_truth

            record = PatchCalibrationRecord(
                instance_id=instance_id,
                patch_id=patch_id,
                patch=patch,
                critic_results={
                    "L0_syntax": asdict(l0),
                    "L1_lint": asdict(l1),
                    "L2_fast_test": asdict(l2),
                },
                ground_truth=ground_truth,
                metadata={
                    "model": args.model,
                    "repo": repo,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

            # Append to JSONL (incremental checkpoint)
            with open(output_file, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")

    # Summary
    log.info("=" * 60)
    log.info("Calibration complete")
    log.info("Total patches evaluated: %d", total_records)
    log.info("Correct patches (Y=1): %d (%.1f%%)",
             total_correct,
             100 * total_correct / total_records if total_records else 0)
    log.info("Output: %s", output_file)
    log.info("Next step: python compute_likelihoods.py --input %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate calibration data for orchestration-as-hypothesis-testing."
    )
    parser.add_argument(
        "--dataset",
        choices=["lite", "verified", "full"],
        default="lite",
        help="SWE-bench dataset variant (default: lite, 300 instances).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of instances to process (0 = all). Default: 5 for quick test.",
    )
    parser.add_argument(
        "--patches-per-instance",
        type=int,
        default=DEFAULT_PATCHES_PER_INSTANCE,
        help="Number of patches to generate per instance (default: 3).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model for patch generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for patch diversity.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for calibration data.",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for repo clones (default: temp dir).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skipping completed records.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print LLM prompts and responses.",
    )
    parser.add_argument(
        "--swebench-eval",
        action="store_true",
        help="Use SWE-bench Docker harness for ground truth evaluation.",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        default=None,
        help="Filter to specific repos (e.g., sympy/sympy psf/requests).",
    )

    args = parser.parse_args()
    run_calibration(args)


if __name__ == "__main__":
    main()
