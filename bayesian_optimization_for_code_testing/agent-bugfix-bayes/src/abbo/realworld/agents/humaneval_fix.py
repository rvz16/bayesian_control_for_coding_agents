"""HumanEvalFix benchmark harness.

Loads the `bigcode/humanevalpack` Python split (164 single-function bugs) and
exposes the same shape of interfaces as `real_bugs.py` so the rest of the
agent pipeline can reuse `DPPlanner`, `bayes_update`, and the calibration
module without modification.

Critics (Option 2 — static + assertion subsets):
    critic_syntax  — `ast.parse` accepts the solution     (cheapest)
    critic_lint    — `pyflakes` reports no errors
    critic_early   — first 3 assertions of check() pass   (semantic, partial)
    critic_mid     — first 6 assertions of check() pass   (semantic, more)

`verify` runs the full check() function — the ground-truth oracle.
"""

from __future__ import annotations

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_dataset() -> list[dict]:
    """Lazy-load the HumanEvalFix dataset (cached)."""
    ds = load_dataset("bigcode/humanevalpack", "python", split="test")
    return list(ds)


@lru_cache(maxsize=1)
def _by_id() -> dict[str, dict]:
    return {ex["task_id"]: ex for ex in _load_dataset()}


def list_task_ids(n_max: int | None = None) -> list[str]:
    """All task IDs (e.g. 'Python/0' … 'Python/163'), deterministic order."""
    ids = [ex["task_id"] for ex in _load_dataset()]
    return ids[:n_max] if n_max else ids


def get_buggy_source(task_id: str) -> str:
    """Full runnable buggy source = prompt (signature+docstring) + buggy body."""
    ex = _by_id()[task_id]
    return ex["prompt"] + ex["buggy_solution"]


def get_canonical_source(task_id: str) -> str:
    ex = _by_id()[task_id]
    return ex["prompt"] + ex["canonical_solution"]


def get_entry_point(task_id: str) -> str:
    return _by_id()[task_id]["entry_point"]


def get_metadata(task_id: str) -> dict:
    ex = _by_id()[task_id]
    return {
        "task_id": task_id,
        "entry_point": ex["entry_point"],
        "bug_type": ex["bug_type"],
        "failure_symptoms": ex["failure_symptoms"],
    }


# ---------------------------------------------------------------------------
# Test script construction
# ---------------------------------------------------------------------------

def get_full_test_script(task_id: str) -> str:
    """A runnable script body: solution.py's symbols + check() + invocation."""
    ex = _by_id()[task_id]
    test = ex["test"]
    entry = ex["entry_point"]
    # Dataset's `test` defines check(candidate) and ends with `check(<entry>)`
    # We wrap it so import-via-from works.
    if f"check({entry})" not in test:
        test = test.rstrip() + f"\n\ncheck({entry})\n"
    return test


_ASSERT_RE = re.compile(r"^(\s*assert .+)$", re.MULTILINE)


def _assertion_lines(test: str) -> list[tuple[int, str]]:
    """Return [(line_idx, line_text), ...] of every top-level assert inside check()."""
    matches = []
    for m in _ASSERT_RE.finditer(test):
        matches.append((m.start(), m.group(1)))
    return matches


def get_subset_test_script(task_id: str, n_asserts: int) -> str:
    """check() script but with only the first `n_asserts` assertions."""
    ex = _by_id()[task_id]
    test = ex["test"]
    entry = ex["entry_point"]
    asserts = _assertion_lines(test)
    if len(asserts) <= n_asserts:
        return get_full_test_script(task_id)
    cutoff = asserts[n_asserts][0]  # start of the (n+1)-th assertion
    truncated = test[:cutoff] + "\n"
    truncated = truncated.rstrip() + f"\n\ncheck({entry})\n"
    return truncated


# ---------------------------------------------------------------------------
# Critic runners (all return (passed: bool, detail: str))
# ---------------------------------------------------------------------------

TIMEOUT_SEC = 15


def _safe_run(cmd: list[str], cwd: Path, timeout: int = TIMEOUT_SEC):
    try:
        return subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=cmd, returncode=124, stdout="", stderr="TIMEOUT\n",
        )


def run_critic_syntax(workdir: Path) -> tuple[bool, str]:
    """ast.parse succeeds on solution.py."""
    code = "import ast,sys; ast.parse(open('solution.py').read()); sys.exit(0)"
    r = _safe_run([sys.executable, "-c", code], workdir)
    return r.returncode == 0, (r.stderr or "").strip()[:400]


def run_critic_lint(workdir: Path) -> tuple[bool, str]:
    """pyflakes has no findings."""
    r = _safe_run([sys.executable, "-m", "pyflakes", "solution.py"], workdir)
    # pyflakes: exit 0 if clean, 1 if warnings
    return r.returncode == 0, (r.stdout + r.stderr).strip()[:400]


def _run_script(workdir: Path, script_path: Path) -> tuple[bool, str]:
    r = _safe_run([sys.executable, str(script_path)], workdir)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[:400]


def run_critic_asserts(
    workdir: Path, task_id: str, n_asserts: int,
) -> tuple[bool, str]:
    """Run check() with only the first n_asserts assertions.

    Writes a temporary script that imports solution.py's namespace then runs
    the truncated check(). Returns (all_asserts_passed, detail).
    """
    script = workdir / f"_critic_{n_asserts}.py"
    body = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('solution', 'solution.py')\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "globals().update({k:v for k,v in vars(mod).items() if not k.startswith('_')})\n"
    )
    body += get_subset_test_script(task_id, n_asserts)
    script.write_text(body)
    return _run_script(workdir, script)


def run_full_test(workdir: Path, task_id: str) -> tuple[bool, str]:
    """Run all assertions of check() — the ground truth."""
    script = workdir / "_full_test.py"
    body = (
        "import importlib.util\n"
        "spec = importlib.util.spec_from_file_location('solution', 'solution.py')\n"
        "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
        "globals().update({k:v for k,v in vars(mod).items() if not k.startswith('_')})\n"
    )
    body += get_full_test_script(task_id)
    script.write_text(body)
    return _run_script(workdir, script)


# ---------------------------------------------------------------------------
# Critic registry (same shape as REAL_CRITIC_TESTS in real_bugs.py)
# ---------------------------------------------------------------------------

HE_CRITIC_NAMES: list[str] = [
    "critic_syntax",
    "critic_lint",
    "critic_early",
    "critic_mid",
]


def run_critic(workdir: Path, critic_name: str, task_id: str) -> tuple[bool, str]:
    if critic_name == "critic_syntax":
        return run_critic_syntax(workdir)
    if critic_name == "critic_lint":
        return run_critic_lint(workdir)
    if critic_name == "critic_early":
        return run_critic_asserts(workdir, task_id, n_asserts=3)
    if critic_name == "critic_mid":
        return run_critic_asserts(workdir, task_id, n_asserts=6)
    raise ValueError(f"unknown critic: {critic_name}")


# ---------------------------------------------------------------------------
# Hand-tuned likelihoods (prior to calibration). Replaces CRITIC_LIKELIHOODS
# for HumanEvalFix runs.
# ---------------------------------------------------------------------------

HE_CRITIC_LIKELIHOODS: dict[str, dict[str, float]] = {
    # Syntax check: correct patches always parse; broken ones *usually* parse too.
    # Weak signal — mostly catches LLM cut-offs.
    "critic_syntax": {"p_pass_y1": 0.99, "p_pass_y0": 0.90},
    # Lint: correct patches may have style issues, broken ones may be clean.
    # Weak but different signal from syntax (catches undefined names).
    "critic_lint":   {"p_pass_y1": 0.90, "p_pass_y0": 0.70},
    # First 3 assertions: strong signal — early test cases catch most bugs.
    "critic_early":  {"p_pass_y1": 0.95, "p_pass_y0": 0.30},
    # First 6 assertions: even stronger, closer to full signal.
    "critic_mid":    {"p_pass_y1": 0.95, "p_pass_y0": 0.20},
}


# ---------------------------------------------------------------------------
# Arm prompts — reuse REAL_ARM_PROMPTS (all 4 arms: g1, g2, g3, g4).
# These are generic "fix this Python module" prompts; they work for
# single-function HumanEvalFix tasks too.
# ---------------------------------------------------------------------------

from abbo.realworld.agents.real_bugs import REAL_ARM_PROMPTS as HE_ARM_PROMPTS  # noqa: E402


# ---------------------------------------------------------------------------
# Calibration data collection from (canonical, buggy) pairs — no LLM calls.
#
# Each task ships a ground-truth Y=1 sample (canonical solution) and a
# Y=0 sample (hand-authored buggy solution). Running all critics on both
# yields balanced calibration data at zero LLM cost.
# ---------------------------------------------------------------------------

def collect_calibration_samples_from_pairs(
    task_ids: list[str],
    verbose: bool = True,
):
    """Run every critic on each task's canonical (Y=1) and buggy (Y=0) source.

    Returns a list of CalibrationSample (imported lazily to avoid cycles).
    """
    import tempfile

    from abbo.realworld.agents.calibration import CalibrationSample

    samples: list[CalibrationSample] = []
    for i, tid in enumerate(task_ids, 1):
        canonical = get_canonical_source(tid)
        buggy = get_buggy_source(tid)

        for arm_label, source, y in [("canonical", canonical, True),
                                      ("buggy", buggy, False)]:
            with tempfile.TemporaryDirectory() as d:
                wd = Path(d)
                (wd / "solution.py").write_text(source)
                for critic in HE_CRITIC_NAMES:
                    passed, _ = run_critic(wd, critic, tid)
                    samples.append(CalibrationSample(
                        bug_id=tid,
                        arm=arm_label,
                        critic=critic,
                        critic_passed=passed,
                        patch_correct=y,
                    ))
        if verbose and i % 10 == 0:
            print(f"  collected {i}/{len(task_ids)} tasks ({len(samples)} samples)")
    return samples

