#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/users/avazhentsev/agents_with_uncertainty_research")
ACCOUNT = os.environ.get("SBATCH_ACCOUNT", "a0142")
SLEEP_SECONDS = int(os.environ.get("WATCH_SLEEP_SECONDS", "900"))

RUNS = [
    {
        "name": "gpt",
        "job": "uq-swe-gpt20b",
        "wrapper": ROOT / "experiments/orchestration_hypothesis_testing/slurm/run_sage_uncertainty_swe_gpt_oss_20b.singlebench.sbatch",
        "root": Path("/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_swe_gpt_oss_20b/2656597_20260630_220737"),
        "generator": "gpt_oss_20b_local",
    },
    {
        "name": "qwen",
        "job": "uq-swe-qwen32b",
        "wrapper": ROOT / "experiments/orchestration_hypothesis_testing/slurm/run_sage_uncertainty_swe_qwen25_32b.singlebench.sbatch",
        "root": Path("/capstor/store/cscs/swissai/a0142/agents_uq/sage_uncertainty_swe_qwen25_32b/2656598_20260630_220804"),
        "generator": "qwen25_32b",
    },
]

EXPECTED = {
    "swebench_lite": 225,
    "swebench_verified": 375,
}

SBATCH = os.environ.get("SBATCH_BIN", "sbatch")


def is_bad_result(row: dict) -> bool:
    if row.get("split") != "test":
        return False
    final_action = str(row.get("final_action", "")).lower()
    return final_action.startswith("exception") or final_action == "context_overflow_skip"


def rewrite_jsonl_without_ids(path: Path, bad_ids: set[str]) -> int:
    if not path.exists() or not bad_ids:
        return 0
    kept: list[str] = []
    removed = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                kept.append(line)
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if str(row.get("instance_id")) in bad_ids:
                removed += 1
            else:
                kept.append(line)
    if removed:
        backup = path.with_suffix(path.suffix + f".pre_badrow_cleanup.{int(time.time())}.bak")
        path.rename(backup)
        path.write_text("".join(kept))
        print(f"cleaned {removed} rows from {path} (backup={backup})", flush=True)
    return removed


def cleanup_bad_rows(run: dict, bench: str) -> None:
    result_path = run["root"] / f"{bench}__{run['generator']}.jsonl"
    if not result_path.exists():
        return
    bad_ids: set[str] = set()
    with result_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if is_bad_result(row):
                bad_ids.add(str(row.get("instance_id")))
    if not bad_ids:
        return
    print(f"bad rows for {run['name']} {bench}: {sorted(bad_ids)}", flush=True)
    stem = run["root"] / f"{bench}__{run['generator']}"
    for path in [
        result_path,
        Path(str(stem) + ".generation_logprobs.jsonl"),
        Path(str(stem) + ".actions.jsonl"),
        Path(str(stem) + ".verbalized_2s.jsonl"),
    ]:
        rewrite_jsonl_without_ids(path, bad_ids)
    readable = run["root"] / "readable" / bench
    for stale in [
        readable / "metric_scores.csv",
        readable / "final_logprob_bayes_quality.csv",
        readable / "final_logprob_bayes_quality.jsonl",
        readable / "generation_trajectory_scores.csv",
        readable / "tool_success_by_instance.csv",
        readable / "analysis_summary.json",
    ]:
        if stale.exists():
            stale.unlink()
            print(f"removed stale analysis file {stale}", flush=True)


def count_test_rows(path: Path) -> int:
    if not path.exists():
        return 0
    rows_by_id: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("split") == "test":
                rows_by_id[str(row.get("instance_id"))] = row
    return sum(1 for row in rows_by_id.values() if not is_bad_result(row))


def progress(run: dict) -> dict[str, int]:
    return {
        bench: count_test_rows(run["root"] / f"{bench}__{run['generator']}.jsonl")
        for bench in EXPECTED
    }


def bench_metric_done(run: dict, bench: str) -> bool:
    return (run["root"] / "readable" / bench / "metric_scores.csv").exists()


def bench_done(run: dict, bench: str, prog: dict[str, int]) -> bool:
    return prog.get(bench, 0) >= EXPECTED[bench] and bench_metric_done(run, bench)


def analysis_done(run: dict) -> bool:
    return all(
        (run["root"] / "readable" / bench / "metric_scores.csv").exists()
        for bench in EXPECTED
    )


def is_done(run: dict, prog: dict[str, int]) -> bool:
    return all(bench_done(run, bench, prog) for bench in EXPECTED)


def bench_job_name(run: dict, bench: str) -> str:
    suffix = "lite" if bench == "swebench_lite" else "verified"
    return f"{run['job']}-{suffix}"


def job_active(job_name: str) -> bool:
    res = subprocess.run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", job_name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return bool(res.stdout.strip())


def submit(run: dict, bench: str) -> None:
    envs = ",".join(
        [
            "ALL",
            "CONDA_ENV=agents",
            "VLLM_CONDA_ENV=thinkbooster",
            f"BENCHMARKS={bench}",
            "N_INSTANCES=0",
            "TRAIN_FRACTION=0.25",
            "PRIOR_PATCHES=1",
            "RUN_ANALYSIS=1",
            "RESUME=1",
            "RUN_ROOT_EXACT=1",
            f"RUN_ROOT={run['root']}",
        ]
    )
    cmd = [
        SBATCH,
        "-A",
        ACCOUNT,
        f"--job-name={bench_job_name(run, bench)}",
        f"--export={envs}",
        str(run["wrapper"]),
    ]
    print("submit:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    while True:
        all_done = True
        for run in RUNS:
            prog = progress(run)
            done = is_done(run, prog)
            all_done = all_done and done
            active = {
                bench: job_active(bench_job_name(run, bench))
                for bench in EXPECTED
            }
            print(
                f"{run['name']}: "
                + " ".join(f"{k}={prog[k]}/{EXPECTED[k]}" for k in EXPECTED)
                + " "
                + " ".join(f"{k}_active={active[k]}" for k in EXPECTED)
                + f" done={done}",
                flush=True,
            )
            for bench in EXPECTED:
                if not bench_done(run, bench, prog) and not active[bench]:
                    submit(run, bench)
        if all_done:
            print("all SWE uncertainty runs are complete", flush=True)
            return
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
