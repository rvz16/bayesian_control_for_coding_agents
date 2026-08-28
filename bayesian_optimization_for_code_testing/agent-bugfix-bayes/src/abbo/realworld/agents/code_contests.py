"""CodeContests benchmark harness.

Loads the `deepmind/code_contests` test split (165 competitive-programming
problems) and exposes the same interface shape as `humaneval_fix.py`, using
the dataset's `solutions` (Y=1) and `incorrect_solutions` (Y=0) pools as
balanced calibration samples — no LLM calls.

Why this is harder than HumanEvalFix:
    - Codeforces problems (cf_rating 800..3500) are well beyond HumanEvalFix;
      gpt-oss-120b does not saturate here.
    - Each task ships many accepted AND many rejected human submissions.
      The Y=0 pool is diverse real-world buggy code, not a single canonical bug.
    - Many stdin/stdout test cases per task → critic_early/mid can slice
      across different test counts with meaningful signal gradation.

Critics:
    critic_syntax  — ast.parse accepts solution.py       (cheap, static)
    critic_lint    — pyflakes reports no errors          (cheap, static)
    critic_early   — first N_EARLY stdin/stdout tests pass  (semantic, partial)
    critic_mid     — first N_MID   stdin/stdout tests pass  (semantic, more)

`run_full_test` runs up to MAX_FULL_TESTS tests — the ground-truth oracle.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from datasets import load_dataset

# Language code in the dataset: 2=CPP, 3=PYTHON3, 4=JAVA (confirmed empirically).
PYTHON3_LANG = 3

# Critic / oracle knobs — chosen so critics stay cheap and oracle fits in ~1 min.
N_EARLY = 3
N_MID = 10
MAX_FULL_TESTS = 40       # oracle depth; deeper = more reliable Y labels
PER_TEST_TIMEOUT = 10     # seconds; was 4, but correct solutions on hard
                          # problems (CF time limit ≈ 2 s, Python ≈ 2-3× slower)
                          # were exceeding it during cold-start + scheduler
                          # jitter — causing false-negative `fixed=False` for
                          # patches that ran fine but missed the wall.
                          # 10 s is comfortably above CF's worst-case Python
                          # judge limit and reduces verify noise.
MAX_SEARCH_POOL = 8       # max submissions to scan when finding a Y=1/Y=0 example


# ---------------------------------------------------------------------------
# Dataset — cached, filtered to problems with enough Python3 samples.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_filtered(
    min_correct: int = 1,
    min_incorrect: int = 1,
) -> list[dict]:
    """Load test split, keep only problems with enough Python3 solutions.

    Downloads ONLY the single test parquet (avoids the 10 GB+ train shards
    that `load_dataset(..., split="test")` pulls by default).
    """
    # hf_hub_download grabs just one file; load via pandas/pyarrow.
    from huggingface_hub import hf_hub_download

    pq_path = hf_hub_download(
        repo_id="deepmind/code_contests",
        filename="data/test-00000-of-00001-9c49eeff30aacaa8.parquet",
        repo_type="dataset",
    )
    raw = load_dataset("parquet", data_files=pq_path, split="train")
    out = []
    for ex in raw:
        py_sols = [
            s for s, l in zip(ex["solutions"]["solution"], ex["solutions"]["language"])
            if l == PYTHON3_LANG
        ]
        py_inc = [
            s for s, l in zip(
                ex["incorrect_solutions"]["solution"],
                ex["incorrect_solutions"]["language"],
            )
            if l == PYTHON3_LANG
        ]
        if len(py_sols) < min_correct or len(py_inc) < min_incorrect:
            continue
        # Collect test cases (public + generated, as available).
        ins = list(ex["public_tests"]["input"]) + list(ex["generated_tests"]["input"])
        outs = list(ex["public_tests"]["output"]) + list(ex["generated_tests"]["output"])
        if len(ins) == 0:
            continue
        out.append({
            "task_id": ex["name"],
            "difficulty": ex.get("difficulty"),
            "cf_rating": ex.get("cf_rating"),
            "py_sols": py_sols,
            "py_inc": py_inc,
            "tests_in": ins,
            "tests_out": outs,
        })
    return out


@lru_cache(maxsize=1)
def _by_id() -> dict[str, dict]:
    return {ex["task_id"]: ex for ex in _load_filtered()}


def list_task_ids(n_max: int | None = None) -> list[str]:
    """Task ids (problem names) in dataset order, filtered to usable Python3 problems."""
    ids = [ex["task_id"] for ex in _load_filtered()]
    return ids[:n_max] if n_max else ids


def get_solution_pool(task_id: str, kind: str) -> list[str]:
    """Return the Python3 source list for `kind` in {"correct", "incorrect"}."""
    ex = _by_id()[task_id]
    if kind == "correct":
        return ex["py_sols"]
    if kind == "incorrect":
        return ex["py_inc"]
    raise ValueError(f"unknown kind: {kind}")


def get_test_cases(
    task_id: str, max_tests: int = MAX_FULL_TESTS,
) -> list[tuple[str, str]]:
    ex = _by_id()[task_id]
    return list(zip(ex["tests_in"][:max_tests], ex["tests_out"][:max_tests]))


def get_metadata(task_id: str) -> dict:
    ex = _by_id()[task_id]
    return {
        "task_id": task_id,
        "difficulty": ex["difficulty"],
        "cf_rating": ex["cf_rating"],
        "n_tests": len(ex["tests_in"]),
        "n_correct_py3": len(ex["py_sols"]),
        "n_incorrect_py3": len(ex["py_inc"]),
    }


# ---------------------------------------------------------------------------
# Test runner (stdin/stdout)
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Token-level normalization tolerant to trailing whitespace and blank lines."""
    return "\n".join(line.rstrip() for line in s.strip().splitlines())


def _tokens(s: str) -> list[str]:
    return s.split()


def _run_one(source_path: Path, stdin: str) -> tuple[bool, str]:
    """Run solution with given stdin; return (non-crash, stdout)."""
    try:
        r = subprocess.run(
            [sys.executable, str(source_path)],
            input=stdin, text=True, capture_output=True,
            timeout=PER_TEST_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return False, ""
    except Exception:
        return False, ""
    return r.returncode == 0, r.stdout


def _passes_tests(
    workdir: Path, tests: list[tuple[str, str]],
) -> tuple[bool, int]:
    """Run all given tests sequentially. Short-circuits on first fail.

    Returns (all_passed, n_checked_until_first_fail_inclusive).
    """
    src = workdir / "solution.py"
    for i, (stdin, expected) in enumerate(tests, 1):
        ok, stdout = _run_one(src, stdin)
        if not ok:
            return False, i
        # Primary: token-wise match (spacing-insensitive).
        if _tokens(stdout) != _tokens(expected):
            # Fallback: line-wise normalized match (some problems are sensitive
            # to newlines vs spaces — keep a conservative second check).
            if _normalize(stdout) != _normalize(expected):
                return False, i
    return True, len(tests)


# ---------------------------------------------------------------------------
# Critics
# ---------------------------------------------------------------------------

def run_critic_syntax(workdir: Path) -> tuple[bool, str]:
    code = "import ast,sys; ast.parse(open('solution.py').read()); sys.exit(0)"
    try:
        r = subprocess.run(
            [sys.executable, "-c", code], cwd=workdir,
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return r.returncode == 0, (r.stderr or "").strip()[:300]


def run_critic_lint(workdir: Path) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pyflakes", "solution.py"], cwd=workdir,
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    return r.returncode == 0, (r.stdout + r.stderr).strip()[:300]


def run_critic_early(workdir: Path, task_id: str) -> tuple[bool, str]:
    tests = get_test_cases(task_id, max_tests=N_EARLY)
    ok, idx = _passes_tests(workdir, tests)
    return ok, f"{idx}/{len(tests)} ok={ok}"


def run_critic_mid(workdir: Path, task_id: str) -> tuple[bool, str]:
    tests = get_test_cases(task_id, max_tests=N_MID)
    ok, idx = _passes_tests(workdir, tests)
    return ok, f"{idx}/{len(tests)} ok={ok}"


def run_full_test(workdir: Path, task_id: str) -> tuple[bool, str]:
    tests = get_test_cases(task_id, max_tests=MAX_FULL_TESTS)
    ok, idx = _passes_tests(workdir, tests)
    return ok, f"{idx}/{len(tests)} ok={ok}"


# ---------------------------------------------------------------------------
# Critic registry + hand-tuned likelihoods (same shape as HumanEvalFix)
# ---------------------------------------------------------------------------

CC_CRITIC_NAMES: list[str] = [
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
        return run_critic_early(workdir, task_id)
    if critic_name == "critic_mid":
        return run_critic_mid(workdir, task_id)
    raise ValueError(f"unknown critic: {critic_name}")


# Hand-tuned prior-to-calibration. Codeforces submissions are varied, so
# expectations differ from HumanEvalFix: many "wrong" solutions still parse
# and pass a few early trivial tests, so critic_early has weaker signal than
# in HumanEvalFix; critic_mid should still be strong.
CC_CRITIC_LIKELIHOODS: dict[str, dict[str, float]] = {
    "critic_syntax": {"p_pass_y1": 0.99, "p_pass_y0": 0.95},
    "critic_lint":   {"p_pass_y1": 0.80, "p_pass_y0": 0.70},
    "critic_early":  {"p_pass_y1": 0.95, "p_pass_y0": 0.60},
    "critic_mid":    {"p_pass_y1": 0.95, "p_pass_y0": 0.25},
}


# ---------------------------------------------------------------------------
# Calibration data collection from real (correct, incorrect) pools.
# ---------------------------------------------------------------------------

def _find_labeled(
    task_id: str, kind: str, target_y: bool,
) -> tuple[str, int] | None:
    """Scan up to MAX_SEARCH_POOL submissions from `kind` pool; return the first
    whose oracle label matches `target_y`. Returns (source, pool_index) or None.
    """
    import tempfile
    pool = get_solution_pool(task_id, kind)[:MAX_SEARCH_POOL]
    for j, source in enumerate(pool):
        with tempfile.TemporaryDirectory() as d:
            wd = Path(d)
            (wd / "solution.py").write_text(source)
            ok, _ = run_full_test(wd, task_id)
        if ok == target_y:
            return source, j
    return None


def collect_calibration_samples_from_pairs(
    task_ids: list[str],
    k_per_side: int = 1,
    verbose: bool = True,
):
    """For each task, find one Py3 submission that OUR oracle labels Y=1 and
    one that it labels Y=0 (searching up to MAX_SEARCH_POOL each way). Tasks
    where either side can't be found are skipped.

    Y is defined as `run_full_test` passing — the same oracle the agent uses
    at runtime — so calibration is trained against the right ground truth.

    Samples/kept_task = 2 * k_per_side * len(CC_CRITIC_NAMES).
    Returns (samples, kept_task_ids).
    """
    import tempfile

    from abbo.realworld.agents.calibration import CalibrationSample

    samples: list[CalibrationSample] = []
    kept: list[str] = []
    for i, tid in enumerate(task_ids, 1):
        # k_per_side is capped at 1 here — finding multiple distinct Y=1/Y=0
        # examples per task via linear scan is cheap; keep the loop simple.
        y1 = _find_labeled(tid, "correct", True)
        y0 = _find_labeled(tid, "incorrect", False)
        if y1 is None or y0 is None:
            if verbose:
                print(f"  [skip] {tid}: y1={y1 is not None} y0={y0 is not None}", flush=True)
            continue
        kept.append(tid)

        for source, y, arm_label in (
            (y1[0], True,  f"correct_{y1[1]}"),
            (y0[0], False, f"incorrect_{y0[1]}"),
        ):
            with tempfile.TemporaryDirectory() as d:
                wd = Path(d)
                (wd / "solution.py").write_text(source)
                for critic in CC_CRITIC_NAMES:
                    passed, _ = run_critic(wd, critic, tid)
                    samples.append(CalibrationSample(
                        bug_id=tid,
                        arm=arm_label,
                        critic=critic,
                        critic_passed=passed,
                        patch_correct=y,
                    ))
        if verbose and i % 5 == 0:
            print(
                f"  {i}/{len(task_ids)} tasks scanned, {len(kept)} usable, "
                f"{len(samples)} samples", flush=True,
            )
    return samples, kept
