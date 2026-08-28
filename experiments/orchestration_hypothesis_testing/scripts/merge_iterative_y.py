"""Atomically merge SWE-bench Y verdicts from one iterative cell into another.

The source may contain only a subset of the target trajectories (for example,
the pre-add8 server cell or the eight-ID Yandex delta).  Rows are matched by
``(instance_id, step)`` and their diffs must be byte-identical before Y is
copied.  Existing conflicting verdicts are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(row: dict) -> tuple[str, int]:
    return row["instance_id"], int(row["step"])


def has_patch(row: dict) -> bool:
    return bool((row.get("diff") or "").strip())


def binary_y(row: dict) -> bool:
    return row.get("Y") in (0, 1)


def unique_map(rows: list[dict], label: str) -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = row_key(row)
        if key in result:
            raise SystemExit(f"duplicate {label} key: {key}")
        result[key] = row
    return result


def write_jsonl_atomic(path: Path, rows: list[dict], backup_suffix: str) -> Path:
    backup = path.with_name(f"{path.name}.{backup_suffix}")
    if backup.exists():
        raise SystemExit(f"backup already exists: {backup}")
    shutil.copy2(path, backup)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-cell", required=True, type=Path)
    parser.add_argument("--source-cell", required=True, type=Path)
    parser.add_argument("--backup-suffix", required=True)
    parser.add_argument("--expected-source-ids", type=int)
    parser.add_argument("--require-source-complete", action="store_true")
    parser.add_argument("--require-target-complete", action="store_true")
    parser.add_argument(
        "--copy-eval-as",
        help="Optional audit directory name created under the target cell.",
    )
    args = parser.parse_args()

    target_path = args.target_cell.resolve() / "iter_records.jsonl"
    source_path = args.source_cell.resolve() / "iter_records.jsonl"
    if not target_path.is_file() or not source_path.is_file():
        raise SystemExit(f"missing records: target={target_path} source={source_path}")

    target_rows = load_jsonl(target_path)
    source_rows = load_jsonl(source_path)
    target = unique_map(target_rows, "target")
    source = unique_map(source_rows, "source")
    source_ids = {key[0] for key in source}
    target_ids = {key[0] for key in target}
    if args.expected_source_ids is not None and len(source_ids) != args.expected_source_ids:
        raise SystemExit(
            f"source ID mismatch: actual={len(source_ids)} "
            f"expected={args.expected_source_ids}"
        )
    if not source_ids <= target_ids:
        raise SystemExit(f"source IDs absent from target: {sorted(source_ids-target_ids)[:10]}")

    missing_source_y = [
        key for key, row in source.items()
        if key[1] > 0 and has_patch(row) and not binary_y(row)
    ]
    if args.require_source_complete and missing_source_y:
        raise SystemExit(
            f"source has {len(missing_source_y)} non-empty iterative rows without Y: "
            f"{missing_source_y[:10]}"
        )

    updated = matched = 0
    for key, source_row in source.items():
        # Step 0 is the base-eval seed and is already canonical in the target.
        # Small repair cells may intentionally carry an empty placeholder diff
        # at step 0, so only iterative verdicts are merged here.
        if key[1] == 0:
            continue
        if not binary_y(source_row):
            continue
        target_row = target.get(key)
        if target_row is None:
            raise SystemExit(f"source key absent from target: {key}")
        if (source_row.get("diff") or "") != (target_row.get("diff") or ""):
            raise SystemExit(f"diff mismatch for {key}")
        matched += 1
        old_y = target_row.get("Y")
        new_y = source_row["Y"]
        if old_y in (0, 1) and old_y != new_y:
            raise SystemExit(f"conflicting Y for {key}: target={old_y} source={new_y}")
        if old_y not in (0, 1):
            target_row["Y"] = new_y
            updated += 1

    missing_target_y = [
        key for key, row in target.items()
        if key[1] > 0 and has_patch(row) and not binary_y(row)
    ]
    if args.require_target_complete and missing_target_y:
        raise SystemExit(
            f"target would retain {len(missing_target_y)} non-empty iterative rows "
            f"without Y: {missing_target_y[:10]}"
        )

    backup = write_jsonl_atomic(target_path, target_rows, args.backup_suffix)

    copied_eval = None
    if args.copy_eval_as:
        source_eval = args.source_cell.resolve() / "eval"
        copied_eval = args.target_cell.resolve() / args.copy_eval_as
        if not source_eval.is_dir():
            raise SystemExit(f"source eval directory is missing: {source_eval}")
        if copied_eval.exists():
            raise SystemExit(f"audit eval directory already exists: {copied_eval}")
        shutil.copytree(source_eval, copied_eval)

    print(
        "merge_y=PASS "
        f"source_ids={len(source_ids)} matched_Y={matched} updated_Y={updated} "
        f"source_missing_Y={len(missing_source_y)} "
        f"target_missing_Y={len(missing_target_y)} backup={backup} "
        f"copied_eval={copied_eval}"
    )


if __name__ == "__main__":
    main()
