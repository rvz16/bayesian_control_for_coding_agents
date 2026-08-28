#!/usr/bin/env python
"""Prepare a held-out train/test split before synthesis calibration.

This script selects benchmark instance ids without generating or evaluating
patches, writes train/test manifests, and creates `<src-dir>/<gen>/split.json`.
Calibration should then be run with `--sample-ids-file <src-dir>/train_sample.json`
so the calibration harness never touches held-out test ids.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_synthesis_live as synth  # noqa: E402


def problem_id(problem: dict[str, Any]) -> str:
    for key in ("question_id", "instance_id", "task_id", "id"):
        if key in problem:
            return str(problem[key])
    raise KeyError(f"problem has no id field; keys={list(problem)[:8]}")


def sample_row(benchmark: str, instance_id: str) -> dict[str, str]:
    row = {"instance_id": instance_id}
    if benchmark.startswith("lcb"):
        row["question_id"] = instance_id
    else:
        row["task_id"] = instance_id
    return row


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src-dir", required=True, type=Path)
    p.add_argument("--benchmark", required=True,
                   choices=["mbpp", "humaneval", "lcb_easy", "lcb_medium", "lcb_hard"])
    p.add_argument("--generators", required=True)
    p.add_argument("--n-instances", type=int, default=0,
                   help="Total ids to select before splitting (0 = all loaded ids).")
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42)
    args = p.parse_args()

    if args.n_train is None and args.train_fraction is None:
        raise SystemExit("provide either --n-train or --train-fraction")
    if args.train_fraction is not None and not (0.0 < args.train_fraction < 1.0):
        raise SystemExit("--train-fraction must be in (0, 1)")

    load_fn, _prompt_fn, _critics_fn, _verify_fn = synth._benchmark_loader(args.benchmark)
    problems = load_fn()
    ids = sorted({problem_id(problem) for problem in problems})
    if not ids:
        raise SystemExit(f"no ids loaded for benchmark {args.benchmark}")

    sample_rng = random.Random(args.sample_seed)
    selected = ids[:]
    sample_rng.shuffle(selected)
    if args.n_instances and args.n_instances < len(selected):
        selected = selected[:args.n_instances]

    if args.train_fraction is not None:
        n_train = max(1, int(len(selected) * args.train_fraction))
    else:
        n_train = int(args.n_train)
    if n_train >= len(selected):
        raise SystemExit(
            f"n_train={n_train} >= selected={len(selected)}; need at least one test id"
        )

    split_rng = random.Random(args.split_seed)
    shuffled = selected[:]
    split_rng.shuffle(shuffled)
    train_ids = sorted(shuffled[:n_train])
    test_ids = sorted(shuffled[n_train:])
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise SystemExit(f"internal split overlap; examples: {overlap[:5]}")

    src_dir = args.src_dir.resolve()
    src_dir.mkdir(parents=True, exist_ok=True)
    generators = parse_csv(args.generators)
    if not generators:
        raise SystemExit("--generators is empty")

    manifest = {
        "benchmark": args.benchmark,
        "sample_seed": args.sample_seed,
        "split_seed": args.split_seed,
        "n_instances_total_loaded": len(ids),
        "n_instances_selected": len(selected),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "train_ids": train_ids,
        "test_ids": test_ids,
    }
    (src_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    (src_dir / "sample.json").write_text(json.dumps(
        [sample_row(args.benchmark, item) for item in sorted(selected)], indent=2
    ))
    (src_dir / "train_sample.json").write_text(json.dumps(
        [sample_row(args.benchmark, item) for item in train_ids], indent=2
    ))
    (src_dir / "test_sample.json").write_text(json.dumps(
        [sample_row(args.benchmark, item) for item in test_ids], indent=2
    ))

    for gen in generators:
        gen_dir = src_dir / gen
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "split.json").write_text(json.dumps({
            "benchmark": args.benchmark,
            "split_seed": args.split_seed,
            "sample_seed": args.sample_seed,
            "n_instances_total_loaded": len(ids),
            "n_instances_selected": len(selected),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "train_ids": train_ids,
            "test_ids": test_ids,
        }, indent=2))

    print(
        f"[{args.benchmark}] selected={len(selected)} train={len(train_ids)} "
        f"test={len(test_ids)} src_dir={src_dir}"
    )


if __name__ == "__main__":
    main()
