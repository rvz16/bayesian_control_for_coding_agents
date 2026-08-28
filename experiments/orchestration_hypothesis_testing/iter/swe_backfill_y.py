"""Backfill Y values into iter_records.jsonl from harness eval reports.

For each cell (gen × method × benchmark), reads each step's harness report
and updates iter_records.jsonl in place: Y=1 for resolved, 0 for unresolved,
None for instances that errored or weren't submitted.

Usage:
  python3 backfill_swe_iter_y.py [--dry-run]
  python3 iter/swe_backfill_y.py --cell-dir data/swebench_verified_realbaselines_reflexion_full/haiku45/reflexion
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]  # orchestration_hypothesis_testing/


def load_resolved(eval_path: Path) -> tuple[set[str], set[str], set[str]]:
    """Return (resolved_ids, unresolved_ids, error_ids) sets."""
    if not eval_path.exists():
        return set(), set(), set()
    d = json.loads(eval_path.read_text())
    return set(d.get("resolved_ids", [])), set(d.get("unresolved_ids", [])), set(d.get("error_ids", []))


def backfill_cell(cell_dir: Path, dry_run: bool = False) -> dict:
    """Backfill Y in cell_dir/iter_records.jsonl. Returns stats."""
    iter_path = cell_dir / "iter_records.jsonl"
    eval_dir = cell_dir / "eval"
    if not iter_path.exists():
        return {"err": "no iter_records"}
    if not eval_dir.exists():
        return {"err": "no eval dir"}

    # Detect cell name pattern from any sibling dir or its path
    parts = cell_dir.parts
    bench = "verified" if "verified" in str(cell_dir) else "lite"
    method = parts[-1]
    gen = parts[-2]
    run_id_template = f"{gen}_{method}_iter_step{{step}}"

    # Load resolved sets per step
    sets_by_step: dict[int, tuple[set, set, set]] = {}
    for step in range(1, 6):
        run_id = run_id_template.format(step=step)
        eval_path = eval_dir / f"{run_id}.{run_id}.json"
        sets_by_step[step] = load_resolved(eval_path)

    # Read all rows, backfill Y, write back
    rows = []
    with open(iter_path) as f:
        for line in f:
            rows.append(json.loads(line))

    n_filled = 0
    n_step0_kept = 0
    n_no_data = 0
    for r in rows:
        step = r.get("step")
        inst = r["instance_id"]
        if step == 0:
            # step 0 Y comes from calibration src; preserve
            n_step0_kept += 1
            continue
        if step not in sets_by_step:
            continue
        resolved, unresolved, errored = sets_by_step[step]
        if inst in resolved:
            r["Y"] = 1
            n_filled += 1
        elif inst in unresolved:
            r["Y"] = 0
            n_filled += 1
        else:
            r["Y"] = None  # errored or not submitted; no data
            n_no_data += 1

    if not dry_run:
        bak = iter_path.with_suffix(".jsonl.pre_backfill.bak")
        if not bak.exists():
            bak.write_text(iter_path.read_text())
        with open(iter_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    return {
        "n_rows": len(rows),
        "n_filled_y": n_filled,
        "n_step0_kept": n_step0_kept,
        "n_no_data": n_no_data,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print what would change without writing")
    parser.add_argument(
        "--cell-dir",
        action="append",
        type=Path,
        default=[],
        help="Specific <data>/<bench>/<gen>/<method> cell to backfill. Can be repeated.",
    )
    args = parser.parse_args()

    print(f"{'cell':<55s} {'rows':>5s} {'filled':>7s} {'step0':>6s} {'nodata':>7s}  status")
    if args.cell_dir:
        cell_dirs = args.cell_dir
    else:
        cell_dirs = []
        for bench_dir in [ROOT / "data/swebench_lite_realbaselines", ROOT / "data/swebench_verified_realbaselines"]:
            if not bench_dir.exists():
                continue
            cell_dirs.extend(sorted(p for p in bench_dir.glob("*/*") if p.is_dir()))

    for cell_dir in cell_dirs:
        cell_dir = cell_dir.resolve()
        stats = backfill_cell(cell_dir, args.dry_run)
        cell_name = "/".join(cell_dir.parts[-3:])
        if "err" in stats:
            print(f"{cell_name:<55s} {'-':>5s} {'-':>7s} {'-':>6s} {'-':>7s}  SKIP: {stats['err']}")
            continue
        tag = "DRY" if args.dry_run else "WROTE"
        print(f"{cell_name:<55s} {stats['n_rows']:>5d} {stats['n_filled_y']:>7d} {stats['n_step0_kept']:>6d} {stats['n_no_data']:>7d}  {tag}")


if __name__ == "__main__":
    main()
