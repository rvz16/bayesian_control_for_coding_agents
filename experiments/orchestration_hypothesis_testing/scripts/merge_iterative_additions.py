"""Atomically append disjoint iterative trajectories to a completed cell."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def bad_stop(row: dict) -> bool:
    reason = str(row.get("stop_reason") or "")
    return reason == "cost_cap" or reason.endswith("api_error")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-cell", required=True, type=Path)
    parser.add_argument("--additions-cell", required=True, type=Path)
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--backup-name", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--method", required=True, choices=["selfrefine", "reflexion"])
    parser.add_argument("--expected-original", required=True, type=int)
    parser.add_argument("--expected-additions", required=True, type=int)
    parser.add_argument("--expected-final", required=True, type=int)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()

    original = args.original_cell.resolve()
    additions = args.additions_cell.resolve()
    backup = original.parent / args.backup_name
    staging = original.parent / f".{original.name}.additions_staging"
    if not original.is_dir() or not additions.is_dir():
        raise SystemExit(f"missing input cell: original={original} additions={additions}")
    if backup.exists() or staging.exists():
        raise SystemExit(f"refusing existing backup/staging: {backup} {staging}")

    old_stops_doc = load_json(original / "stop_distribution.json")
    add_stops_doc = load_json(additions / "stop_distribution.json")
    old_stops = old_stops_doc["instances"]
    add_stops = add_stops_doc["instances"]
    old_ids = {row["instance_id"] for row in old_stops}
    add_ids = {row["instance_id"] for row in add_stops}

    if len(old_ids) != args.expected_original or len(old_stops) != len(old_ids):
        raise SystemExit(
            f"original count mismatch: unique={len(old_ids)} rows={len(old_stops)} "
            f"expected={args.expected_original}"
        )
    if len(add_ids) != args.expected_additions or len(add_stops) != len(add_ids):
        raise SystemExit(
            f"addition count mismatch: unique={len(add_ids)} rows={len(add_stops)} "
            f"expected={args.expected_additions}"
        )
    overlap = old_ids & add_ids
    if overlap:
        raise SystemExit(f"additions overlap original IDs: {sorted(overlap)[:10]}")
    if len(old_ids | add_ids) != args.expected_final:
        raise SystemExit(
            f"final ID count mismatch: {len(old_ids | add_ids)} "
            f"expected={args.expected_final}"
        )
    bad_old = [row for row in old_stops if bad_stop(row)]
    bad_add = [row for row in add_stops if bad_stop(row)]
    if bad_old or bad_add:
        raise SystemExit(f"bad stops: original={len(bad_old)} additions={len(bad_add)}")

    add_cost = load_json(additions / "cost_summary.json")
    if add_cost.get("cap_hit"):
        raise SystemExit(f"additions hit cap: {add_cost}")
    if add_cost.get("n_instances_completed") != args.expected_additions:
        raise SystemExit(f"incomplete additions: {add_cost}")

    old_records = load_jsonl(original / "iter_records.jsonl")
    add_records = load_jsonl(additions / "iter_records.jsonl")
    old_record_ids = {row["instance_id"] for row in old_records}
    add_record_ids = {row["instance_id"] for row in add_records}
    if old_record_ids != old_ids or add_record_ids != add_ids:
        raise SystemExit(
            "stop/record ID mismatch: "
            f"old_missing={sorted(old_ids-old_record_ids)[:5]} "
            f"add_missing={sorted(add_ids-add_record_ids)[:5]}"
        )
    merged_records = old_records + add_records
    record_keys = [(row["instance_id"], row["step"]) for row in merged_records]
    if len(record_keys) != len(set(record_keys)):
        raise SystemExit("duplicate (instance_id, step) rows after merge")

    shutil.copytree(original, staging)
    write_jsonl(staging / "iter_records.jsonl", merged_records)
    write_json(
        staging / "stop_distribution.json",
        {"n_instances": args.expected_final, "instances": old_stops + add_stops},
    )

    for step in range(1, 5):
        rows = [
            {
                "instance_id": row["instance_id"],
                "model_patch": row["diff"],
                "model_name_or_path": f"{args.generator}_{args.method}_iter_step{step}",
            }
            for row in merged_records
            if row.get("step") == step and row.get("diff")
        ]
        write_jsonl(staging / f"predictions_iter_step{step}.jsonl", rows)

    merged_cost_rows = load_jsonl(original / "cost_log.jsonl") + load_jsonl(
        additions / "cost_log.jsonl"
    )
    cumulative = 0.0
    for row in merged_cost_rows:
        cumulative += float(row.get("cost_usd") or 0.0)
        row["cumulative_usd"] = cumulative
    write_jsonl(staging / "cost_log.jsonl", merged_cost_rows)

    new_telemetry = load_jsonl(additions / "action_telemetry.jsonl")
    for row in new_telemetry:
        row["dataset"] = args.dataset_name
    write_jsonl(
        staging / "action_telemetry.jsonl",
        load_jsonl(original / "action_telemetry.jsonl") + new_telemetry,
    )

    raw_dir = staging / "raw_calls"
    raw_dir.mkdir(exist_ok=True)
    for path in (additions / "raw_calls").glob("*.json"):
        target = raw_dir / path.name
        if target.exists():
            raise SystemExit(f"raw-call collision: {target}")
        shutil.copy2(path, target)

    old_cost = load_json(original / "cost_summary.json")
    total_cost = sum(float(row.get("cost_usd") or 0.0) for row in merged_cost_rows)
    combined_cap = float(old_cost.get("cap_usd") or 0.0) + float(
        add_cost.get("cap_usd") or 0.0
    )
    additions_meta = {
        "n_added_instances": len(add_ids),
        "addition_cell": str(additions),
        "instance_ids": sorted(add_ids),
    }
    write_json(
        staging / "cost_summary.json",
        {
            "generator": args.generator,
            "method": args.method,
            "n_instances_completed": args.expected_final,
            "total_cost_usd": total_cost,
            "cap_usd": combined_cap,
            "cap_hit": False,
            "iterative_additions": additions_meta,
        },
    )
    config = load_json(staging / "RUN_CONFIG.json")
    config["n_instances"] = args.expected_final
    config["iterative_additions"] = additions_meta
    write_json(staging / "RUN_CONFIG.json", config)

    original.rename(backup)
    staging.rename(original)

    summary = load_json(args.summary_file) if args.summary_file.exists() else {}
    summary[args.generator] = {
        "n_instances": args.expected_final,
        "total_cost_usd": total_cost,
        "cap_hit": False,
    }
    write_json(args.summary_file, summary)
    print(
        f"merge_additions=PASS added={len(add_ids)} total={args.expected_final} "
        f"total_cost=${total_cost:.6f} backup={backup}"
    )


if __name__ == "__main__":
    main()
