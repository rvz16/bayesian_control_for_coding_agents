#!/usr/bin/env python
"""Extract SWE-Bench instance lists for rerun targeting.

For each (dataset, model) cell we have two on-disk sources of per-instance
Y labels from previous runs:

  * calib__swe_{lite,verified}__{model}/critic_results.jsonl  — one record
    per (instance, patch_id) with single-shot Y in {0,1}.
  * iter__swe_{lite,verified}__{model}__{method}/iter_records.jsonl —
    one record per (instance, refinement step) with Y in {0,1,null}
    (null = step where no fresh patch was produced).

For each cell we union all instance_ids where SOME prior run got Y=1
("already solved by some method") and write that to disk. The complement
against the full SWE-Bench split is the rerun target — instances no
existing policy/method has solved, where the now-fixed runner has the
best chance of producing new wins.

Outputs (under sim_results/swebench_rerun_targets/):
  solved__<dataset>__<model>.json   — list of solved instance_ids
  failed__<dataset>__<model>.json   — list of NOT-yet-solved instance_ids
  summary.json                      — counts per cell

The failed-instances JSON is the file you pass to run_swebench_full.py
via --instance-ids-file when rerunning. The reset of the pipeline
(harness eval via eval_steps.py, refine via refine_swe.py) can use the
same file with their own --instance-ids flags if/when those are added.

Usage:
  python extract_swe_failed_instances.py                  # all cells
  python extract_swe_failed_instances.py --model gpt5_mini --dataset lite
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "experiments" / "orchestration" / "wandb" / ".cache" / "raw_evidence"
OUT = (ROOT / "bayesian_optimization_for_code_testing" / "agent-bugfix-bayes"
       / "sim_results" / "swebench_rerun_targets")

DATASETS = ("swe_lite", "swe_verified")
MODEL_SLUGS = ("gpt5_mini", "haiku45", "sonnet45",
               "qwen3_coder", "qwen25_32b", "gpt_oss_20b")

# Map our internal model slugs to the HuggingFace SWE-Bench dataset names
# used by load_dataset (only needed when computing the "full set" via HF).
_HF_DATASET = {"swe_lite": "princeton-nlp/SWE-bench_Lite",
               "swe_verified": "princeton-nlp/SWE-bench_Verified"}


def _load_jsonl(p: Path):
    if not p.exists():
        return
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def _y_is_truthy(y) -> bool:
    """SWE iter_records use Y ∈ {True, False, None}; calib uses 0/1.
    Treat anything that evaluates to 1/True as solved."""
    return bool(y) and y is not None


def _collect_solved_for_cell(dataset: str, model_slug: str) -> set[str]:
    """Walk every calib + iter cache directory for this cell and return
    the set of instance_ids with at least one Y=1 record."""
    solved: set[str] = set()

    # 1. calibration patches
    calib = RAW / f"calib__{dataset}__{model_slug}" / "critic_results.jsonl"
    for rec in _load_jsonl(calib):
        if _y_is_truthy(rec.get("Y")):
            iid = rec.get("instance_id")
            if iid: solved.add(iid)

    # 2. iter trajectories (selfrefine / reflexion / single_method / ...)
    prefix = f"iter__{dataset}__{model_slug}__"
    if RAW.exists():
        for sub in RAW.iterdir():
            if not sub.is_dir() or not sub.name.startswith(prefix):
                continue
            for rec in _load_jsonl(sub / "iter_records.jsonl"):
                if _y_is_truthy(rec.get("Y")):
                    iid = rec.get("instance_id")
                    if iid: solved.add(iid)
    return solved


def _all_instances_for(dataset: str) -> list[str]:
    """Full instance pool for the dataset, in upstream order."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("WARN: datasets library missing — cannot derive failed list "
              "without it. Install with `pip install datasets`.", file=sys.stderr)
        return []
    name = _HF_DATASET[dataset]
    ds = load_dataset(name, split="test")
    return [ex["instance_id"] for ex in ds]


def _seen_instances_for_cell(dataset: str, model_slug: str) -> set[str]:
    """Every instance_id we have ANY record for in this cell, regardless of Y.
    Used as the rerun universe when the HF datasets lib isn't available."""
    seen: set[str] = set()
    calib = RAW / f"calib__{dataset}__{model_slug}" / "critic_results.jsonl"
    for rec in _load_jsonl(calib):
        if iid := rec.get("instance_id"):
            seen.add(iid)
    prefix = f"iter__{dataset}__{model_slug}__"
    if RAW.exists():
        for sub in RAW.iterdir():
            if sub.is_dir() and sub.name.startswith(prefix):
                for rec in _load_jsonl(sub / "iter_records.jsonl"):
                    if iid := rec.get("instance_id"):
                        seen.add(iid)
    return seen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, default=None,
                   help="Restrict to one dataset (default: both).")
    p.add_argument("--model", default=None,
                   help="Restrict to one model slug (default: all).")
    p.add_argument("--universe", choices=["hf", "seen"], default="hf",
                   help="hf: full upstream split via HuggingFace (recommended). "
                        "seen: just instances we have records for "
                        "(falls back if HF datasets lib missing).")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cells = [(d, m) for d in DATASETS for m in MODEL_SLUGS
             if (args.dataset is None or d == args.dataset)
             and (args.model is None or m == args.model)]

    summary = []
    for dataset, model_slug in cells:
        solved = _collect_solved_for_cell(dataset, model_slug)
        if args.universe == "hf":
            universe = _all_instances_for(dataset)
        else:
            universe = sorted(_seen_instances_for_cell(dataset, model_slug))
        if not universe:
            print(f"  {dataset} / {model_slug}: empty universe — skipping")
            continue
        failed = [iid for iid in universe if iid not in solved]

        slv = OUT / f"solved__{dataset}__{model_slug}.json"
        fld = OUT / f"failed__{dataset}__{model_slug}.json"
        slv.write_text(json.dumps(sorted(solved), indent=2))
        fld.write_text(json.dumps(failed, indent=2))
        row = {"dataset": dataset, "model": model_slug,
               "universe_n": len(universe),
               "solved_n": len(solved),
               "failed_n": len(failed),
               "universe": args.universe}
        summary.append(row)
        print(f"  {dataset:<13} / {model_slug:<12} "
              f"universe={len(universe):3d}  solved={len(solved):3d}  "
              f"failed={len(failed):3d}  → {fld.name}")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(summary)} cells to {OUT}")


if __name__ == "__main__":
    main()
