"""Atomically replace API-error Self-Refine trajectories with a clean retry."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


BAD_REASONS = {"cost_cap", "critique_api_error", "refine_api_error"}


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-cell", required=True, type=Path)
    parser.add_argument("--repair-cell", required=True, type=Path)
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--backup-name", required=True)
    parser.add_argument("--generator", default="gpt5_mini")
    parser.add_argument("--method", default="selfrefine")
    parser.add_argument("--expected-total", type=int, default=482)
    parser.add_argument("--expected-repairs", type=int, default=None)
    parser.add_argument("--cap-usd", type=float, default=25.0)
    parser.add_argument(
        "--dataset-name",
        default="swebench_verified_realbaselines_selfrefine_exp",
    )
    args = parser.parse_args()

    original = args.original_cell.resolve()
    repair = args.repair_cell.resolve()
    backup = original.parent / args.backup_name
    staging = original.parent / f".{original.name}.merge_staging"
    if backup.exists() or staging.exists():
        raise SystemExit(f"refusing existing backup/staging: {backup} {staging}")

    old_stops_doc = load_json(original / "stop_distribution.json")
    repair_stops_doc = load_json(repair / "stop_distribution.json")
    old_stops = old_stops_doc["instances"]
    repair_stops = repair_stops_doc["instances"]
    expected_ids = {
        row["instance_id"] for row in old_stops
        if str(row.get("stop_reason") or "").endswith("api_error")
    }
    repair_ids = {row["instance_id"] for row in repair_stops}
    if args.expected_repairs is not None and len(expected_ids) != args.expected_repairs:
        raise SystemExit(
            f"expected {args.expected_repairs} API-error IDs, got {len(expected_ids)}"
        )
    if not expected_ids or repair_ids != expected_ids:
        raise SystemExit(
            f"repair ID mismatch expected={len(expected_ids)} got={len(repair_ids)} "
            f"missing={sorted(expected_ids-repair_ids)[:5]} "
            f"extra={sorted(repair_ids-expected_ids)[:5]}"
        )
    bad_repairs = [row for row in repair_stops if row.get("stop_reason") in BAD_REASONS]
    if bad_repairs:
        raise SystemExit(f"repair still contains {len(bad_repairs)} bad stops")
    repair_cost = load_json(repair / "cost_summary.json")
    if repair_cost.get("cap_hit"):
        raise SystemExit(f"repair hit cap: {repair_cost}")

    old_records = load_jsonl(original / "iter_records.jsonl")
    repair_records = load_jsonl(repair / "iter_records.jsonl")
    merged_records = [r for r in old_records if r["instance_id"] not in repair_ids]
    merged_records.extend(repair_records)
    instance_ids = {r["instance_id"] for r in merged_records}
    if len(instance_ids) != args.expected_total:
        raise SystemExit(
            f"merged instance count is {len(instance_ids)}, "
            f"expected {args.expected_total}"
        )
    keys = [(r["instance_id"], r["step"]) for r in merged_records]
    if len(keys) != len(set(keys)):
        raise SystemExit("duplicate (instance_id, step) rows after merge")

    replacement = {r["instance_id"]: r for r in repair_stops}
    merged_stops = [replacement.get(r["instance_id"], r) for r in old_stops]
    bad_final = [r for r in merged_stops if r.get("stop_reason") in BAD_REASONS]
    if bad_final:
        raise SystemExit(f"merged stop distribution still has {len(bad_final)} bad stops")

    shutil.copytree(original, staging)
    write_jsonl(staging / "iter_records.jsonl", merged_records)
    write_json(staging / "stop_distribution.json", {
        "n_instances": args.expected_total,
        "instances": merged_stops,
    })

    for step in range(1, 5):
        rows = []
        for row in merged_records:
            if row.get("step") == step and row.get("diff"):
                rows.append({
                    "instance_id": row["instance_id"],
                    "model_patch": row["diff"],
                    "model_name_or_path": (
                        f"{args.generator}_{args.method}_iter_step{step}"
                    ),
                })
        write_jsonl(staging / f"predictions_iter_step{step}.jsonl", rows)

    old_cost_rows = [
        r for r in load_jsonl(original / "cost_log.jsonl")
        if r.get("instance_id") not in repair_ids
    ]
    new_cost_rows = load_jsonl(repair / "cost_log.jsonl")
    cumulative = 0.0
    for row in old_cost_rows + new_cost_rows:
        cumulative += float(row.get("cost_usd") or 0.0)
        row["cumulative_usd"] = cumulative
    merged_cost_rows = old_cost_rows + new_cost_rows
    write_jsonl(staging / "cost_log.jsonl", merged_cost_rows)

    old_telemetry = [
        r for r in load_jsonl(original / "action_telemetry.jsonl")
        if r.get("instance_id") not in repair_ids
    ]
    new_telemetry = load_jsonl(repair / "action_telemetry.jsonl")
    for row in new_telemetry:
        row["dataset"] = args.dataset_name
    write_jsonl(staging / "action_telemetry.jsonl", old_telemetry + new_telemetry)

    raw_dir = staging / "raw_calls"
    for iid in repair_ids:
        for path in raw_dir.glob(f"{iid}_step*_*.json"):
            path.unlink()
    for path in (repair / "raw_calls").glob("*.json"):
        shutil.copy2(path, raw_dir / path.name)

    total_cost = sum(float(r.get("cost_usd") or 0.0) for r in merged_cost_rows)
    write_json(staging / "cost_summary.json", {
        "generator": args.generator,
        "method": args.method,
        "n_instances_completed": args.expected_total,
        "total_cost_usd": total_cost,
        "cap_usd": args.cap_usd,
        "cap_hit": False,
        "api_error_repair": {
            "n_repaired_instances": len(repair_ids),
            "repair_cell": str(repair),
        },
    })
    config = load_json(staging / "RUN_CONFIG.json")
    config["api_error_repair"] = {
        "n_repaired_instances": len(repair_ids),
        "repair_cell": str(repair),
    }
    write_json(staging / "RUN_CONFIG.json", config)

    original.rename(backup)
    staging.rename(original)

    summary = load_json(args.summary_file) if args.summary_file.exists() else {}
    summary[args.generator] = {
        "n_instances": args.expected_total,
        "total_cost_usd": total_cost,
        "cap_hit": False,
    }
    write_json(args.summary_file, summary)
    print(
        f"merge=PASS repaired={len(repair_ids)} total_cost=${total_cost:.6f} "
        f"backup={backup}"
    )


if __name__ == "__main__":
    main()
