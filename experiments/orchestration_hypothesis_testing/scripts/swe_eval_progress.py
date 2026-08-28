#!/usr/bin/env python3
"""Print live SWE-bench harness progress from filesystem artifacts.

This is intentionally independent from the training/eval process: it can be
run against already-running Slurm jobs because SWE-bench writes per-instance
artifacts under eval/logs/run_evaluation while the harness is still active.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ARTIFACTS = (
    ("run_logs", "run_instance.log"),
    ("patch_diff", "patch.diff"),
    ("eval_sh", "eval.sh"),
    ("test_output", "test_output.txt"),
    ("reports", "report.json"),
)


@dataclass
class ProgressRow:
    eval_root: Path
    run_id: str
    model_dir: str
    launched_dirs: int
    run_logs: int
    patch_diff: int
    eval_sh: int
    test_output: int
    reports: int
    aggregate: bool
    latest_file: float | None


def _fmt_mtime(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _safe_dirs(path: Path) -> list[Path]:
    try:
        return sorted(child for child in path.iterdir() if child.is_dir())
    except OSError:
        return []


def _latest_direct_file_mtime(paths: Iterable[Path]) -> float | None:
    latest: float | None = None
    for path in paths:
        try:
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if latest is None or mtime > latest:
            latest = mtime
    return latest


def _count_named_files(instance_dirs: list[Path], name: str) -> int:
    return sum(1 for instance_dir in instance_dirs if (instance_dir / name).exists())


def _discover_eval_roots(paths: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for path in paths:
        path = path.expanduser().resolve()
        if (path / "logs" / "run_evaluation").exists():
            roots.append(path)
            continue
        if (path / "eval" / "logs" / "run_evaluation").exists():
            roots.append(path)
            continue
        for child in ("swebench_lite", "swebench_verified"):
            candidate = path / child
            if (candidate / "eval" / "logs" / "run_evaluation").exists():
                roots.append(candidate)
        for run_root in _safe_dirs(path):
            for child in ("swebench_lite", "swebench_verified"):
                candidate = run_root / child
                if (candidate / "eval" / "logs" / "run_evaluation").exists():
                    roots.append(candidate)
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            unique.append(root)
    return unique


def _eval_dir(eval_root: Path) -> Path:
    if (eval_root / "logs" / "run_evaluation").exists():
        return eval_root
    return eval_root / "eval"


def _build_summary(eval_root: Path) -> str:
    eval_dir = _eval_dir(eval_root)
    build_root = eval_dir / "logs" / "build_images"
    env_logs = list((build_root / "env").glob("*/build_image.log"))
    instance_logs = list((build_root / "instances").glob("*/build_image.log"))
    latest = _latest_direct_file_mtime([*env_logs, *instance_logs])
    return (
        f"build_env_logs={len(env_logs)} "
        f"build_instance_logs={len(instance_logs)} "
        f"latest_build={_fmt_mtime(latest)}"
    )


def summarize_eval_root(eval_root: Path) -> list[ProgressRow]:
    eval_dir = _eval_dir(eval_root)
    run_eval_root = eval_dir / "logs" / "run_evaluation"
    rows: list[ProgressRow] = []
    for run_dir in _safe_dirs(run_eval_root):
        run_id = run_dir.name
        aggregate = bool(list(eval_dir.glob(f"*.{run_id}.json")))
        for model_dir in _safe_dirs(run_dir):
            instance_dirs = _safe_dirs(model_dir)
            direct_files: list[Path] = []
            for instance_dir in instance_dirs:
                try:
                    direct_files.extend(path for path in instance_dir.iterdir() if path.is_file())
                except OSError:
                    continue
            counts = {
                label: _count_named_files(instance_dirs, filename)
                for label, filename in ARTIFACTS
            }
            rows.append(
                ProgressRow(
                    eval_root=eval_root,
                    run_id=run_id,
                    model_dir=model_dir.name,
                    launched_dirs=len(instance_dirs),
                    run_logs=counts["run_logs"],
                    patch_diff=counts["patch_diff"],
                    eval_sh=counts["eval_sh"],
                    test_output=counts["test_output"],
                    reports=counts["reports"],
                    aggregate=aggregate,
                    latest_file=_latest_direct_file_mtime(direct_files),
                )
            )
    return rows


def print_snapshot(paths: list[Path]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    eval_roots = _discover_eval_roots(paths)
    print(f"[{now}] roots={len(eval_roots)}")
    if not eval_roots:
        print("  no SWE eval roots found")
        return

    for eval_root in eval_roots:
        rows = summarize_eval_root(eval_root)
        print(f"eval_root={eval_root}")
        print(f"  {_build_summary(eval_root)}")
        if not rows:
            print("  no run_evaluation rows yet")
            continue
        total_launched = sum(row.launched_dirs for row in rows)
        total_reports = sum(row.reports for row in rows)
        total_test_output = sum(row.test_output for row in rows)
        print(
            "  total "
            f"run_rows={len(rows)} launched_dirs={total_launched} "
            f"test_output={total_test_output} reports={total_reports}"
        )
        for row in rows:
            print(
                "  "
                f"run_id={row.run_id} model={row.model_dir} "
                f"launched_dirs={row.launched_dirs} "
                f"run_logs={row.run_logs} "
                f"patch_diff={row.patch_diff} "
                f"eval_sh={row.eval_sh} "
                f"test_output={row.test_output} "
                f"reports={row.reports} "
                f"aggregate={'yes' if row.aggregate else 'no'} "
                f"latest_file={_fmt_mtime(row.latest_file)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="SWE eval roots or run roots containing swebench_lite/swebench_verified",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep printing snapshots until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="Seconds between snapshots in --watch mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        print_snapshot(args.paths)
        if not args.watch:
            break
        print("", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
