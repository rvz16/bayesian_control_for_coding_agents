"""Reconstruct SWE-bench aggregate JSON files from per-instance reports.

This is useful after reports were produced on multiple machines and merged into
one cell.  By default incomplete steps are rejected.  ``--allow-incomplete``
writes a clearly marked partial aggregate whose missing instances remain
unlabelled when ``swe_backfill_y.py`` is run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_predictions(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def reconstruct(cell: Path, step: int, allow_incomplete: bool) -> dict:
    generator = cell.parent.name
    method = cell.name
    run_id = f"{generator}_{method}_iter_step{step}"
    predictions_path = cell / f"predictions_iter_step{step}.jsonl"
    report_root = cell / "eval" / "logs" / "run_evaluation" / run_id

    predictions = load_predictions(predictions_path)
    expected = {
        row["instance_id"]
        for row in predictions
        if (row.get("model_patch") or "").strip()
    }
    empty = {
        row["instance_id"]
        for row in predictions
        if not (row.get("model_patch") or "").strip()
    }

    verdicts: dict[str, bool] = {}
    malformed: list[str] = []
    for report_path in report_root.rglob("report.json"):
        try:
            payload = json.loads(report_path.read_text())
        except Exception:
            malformed.append(str(report_path))
            continue
        for instance_id, report in payload.items():
            if instance_id not in expected:
                continue
            resolved = report.get("resolved")
            if not isinstance(resolved, bool):
                malformed.append(str(report_path))
                continue
            previous = verdicts.get(instance_id)
            if previous is not None and previous != resolved:
                raise RuntimeError(
                    f"conflicting verdicts for {instance_id}: {previous} vs {resolved}"
                )
            verdicts[instance_id] = resolved

    completed = set(verdicts)
    resolved = {instance_id for instance_id, value in verdicts.items() if value}
    unresolved = completed - resolved
    missing = expected - completed
    if (missing or malformed) and not allow_incomplete:
        raise RuntimeError(
            f"step {step} is incomplete: missing={len(missing)} malformed={len(malformed)}"
        )

    aggregate = {
        "total_instances": len(predictions),
        "submitted_instances": len(expected),
        "completed_instances": len(completed),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": len(empty),
        "error_instances": len(malformed),
        "completed_ids": sorted(completed),
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "empty_patch_ids": sorted(empty),
        "error_ids": [],
        "missing_ids": sorted(missing),
        "partial": bool(missing or malformed),
        "malformed_report_paths": malformed,
    }
    output = cell / "eval" / f"{run_id}.{run_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    for step in args.steps:
        aggregate = reconstruct(args.cell_dir.resolve(), step, args.allow_incomplete)
        print(
            f"step={step} submitted={aggregate['submitted_instances']} "
            f"completed={aggregate['completed_instances']} "
            f"resolved={aggregate['resolved_instances']} "
            f"missing={len(aggregate['missing_ids'])} "
            f"partial={aggregate['partial']}"
        )


if __name__ == "__main__":
    main()
