#!/usr/bin/env python3
"""Experiment 1: per-step (trajectory) UQ aggregation vs the last-step baseline.

The existing pipeline scores each instance using the uncertainty of the *final*
generation only (``generation_trace[-1]``). This script instead reduces the
uncertainty of *all* generation steps of an instance into a single per-instance
score with several aggregators (mean/max/min/std/first/last/range), then scores
each aggregator with the same UQ quality metrics used elsewhere
(``spearman`` / ``PRR`` / ``PRR_05``).

Input is a ``generation_trajectory_scores.jsonl`` produced by
``analyze_lcb_llm_tool_agent_logs.py`` (or the SWE collector). Each row is one
generation step and carries the per-step logprob scores plus, on the final
candidate row, ``final_quality``.

Example:
  python scripts/aggregate_trajectory_uq.py \
      --trajectory sim_results/sage_uq/readable/lcb_hard/generation_trajectory_scores.jsonl \
      --out sim_results/sage_uq/readable/lcb_hard/metric_scores_trajectory.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the exact metric implementations (lm-polygraph-compatible PRR, etc.).
from analyze_lcb_llm_tool_agent_logs import (  # noqa: E402
    prediction_rejection_area,
    prr,
    spearman,
)

# Base per-step scores and whether a higher raw value means "more correct".
#   llm_perplexity   = mean token logprob   -> higher is better
#   llm_log_seq_prob = sum  token logprob   -> higher is better
#   perplexity       = exp(-mean logprob)   -> higher is worse
BASE_SCORES: list[tuple[str, bool]] = [
    ("llm_perplexity", True),
    ("perplexity", False),
    ("llm_log_seq_prob", True),
]

AGGREGATORS: dict[str, Callable[[list[float]], float]] = {
    "last": lambda v: v[-1],
    "first": lambda v: v[0],
    "mean": lambda v: statistics.fmean(v),
    "max": lambda v: max(v),
    "min": lambda v: min(v),
    "std": lambda v: statistics.pstdev(v) if len(v) > 1 else 0.0,
    "range": lambda v: max(v) - min(v),
}


def finite_or_none(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def instance_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("instance_id")), str(row.get("policy", "")))


def collect_instances(
    traj_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group trajectory rows per (instance_id, policy), ordered by generation.

    Returns per instance: ordered per-step values for each base score, the
    number of usable generation steps, and the quality label.
    """
    by_inst: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in traj_rows:
        by_inst[instance_key(row)].append(row)

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in by_inst.items():
        rows = sorted(rows, key=lambda r: int(r.get("patch_idx", r.get("action_step", 0))))
        quality = None
        for r in rows:
            if r.get("final_quality") is not None:
                quality = int(bool(r.get("final_quality")))
        series: dict[str, list[float]] = {name: [] for name, _ in BASE_SCORES}
        for r in rows:
            if r.get("logprobs_supported") is False:
                continue
            for name, _ in BASE_SCORES:
                val = finite_or_none(r.get(name))
                if val is not None:
                    series[name].append(val)
        out[key] = {"series": series, "quality": quality, "n_steps": len(rows)}
    return out


def _bootstrap_prr_ci(
    confs: list[float], quality: list[int], *, n_boot: int, seed: int = 0
) -> tuple[float | None, float | None]:
    """Percentile bootstrap 95% CI for PRR by resampling instances.

    ``oracle`` and ``random`` normalisers are estimated once on the full sample
    (the expensive part of PRR is the random-baseline shuffles), then each
    resample only recomputes the cheap prediction-rejection area.
    """
    import numpy as np

    pairs = [(c, q) for c, q in zip(confs, quality) if c is not None]
    if n_boot <= 0 or len(pairs) < 3:
        return (None, None)
    conf_full = [c for c, _ in pairs]
    q_full = [q for _, q in pairs]
    oracle = prediction_rejection_area([-float(q) for q in q_full], q_full, 1.0)
    rng = np.random.RandomState(seed)
    rnd_arr = np.arange(len(q_full))
    rnd_scores = []
    for _ in range(200):
        rng.shuffle(rnd_arr)
        s = prediction_rejection_area(rnd_arr.tolist(), q_full, 1.0)
        if s is not None:
            rnd_scores.append(s)
    random = float(np.mean(rnd_scores)) if rnd_scores else None
    if oracle is None or random is None or abs(oracle - random) < 1e-12:
        return (None, None)
    idx = np.arange(len(pairs))
    scores: list[float] = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=len(idx), replace=True)
        bc = [conf_full[i] for i in take]
        bq = [q_full[i] for i in take]
        if len(set(bq)) < 2:  # PRR undefined without both classes
            continue
        area = prediction_rejection_area([-c for c in bc], bq, 1.0)
        if area is not None:
            scores.append((area - random) / (oracle - random))
    if len(scores) < max(10, n_boot // 4):
        return (None, None)
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def build_rows(
    instances: dict[tuple[str, str], dict[str, Any]],
    *,
    min_generations: int,
    n_boot: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base, higher_is_better in BASE_SCORES:
        for agg_name, agg in AGGREGATORS.items():
            confs: list[float | None] = []
            quality: list[int] = []
            n_multi = 0
            for inst in instances.values():
                vals = inst["series"][base]
                q = inst["quality"]
                if q is None or not vals:
                    continue
                if len(vals) >= 2:
                    n_multi += 1
                if len(vals) < min_generations:
                    continue
                score = agg(vals)
                # Convert to a confidence where higher == more likely correct.
                conf = score if higher_is_better else -score
                confs.append(conf)
                quality.append(q)
            ci_lo, ci_hi = _bootstrap_prr_ci(
                [c for c in confs if c is not None], quality, n_boot=n_boot
            )
            rows.append(
                {
                    "score": f"{base}_{agg_name}",
                    "base": base,
                    "aggregator": agg_name,
                    "higher_is_better": higher_is_better,
                    "n": len(quality),
                    "n_multi_gen": n_multi,
                    "spearman": spearman(confs, quality),
                    "PRR": prr(confs, quality, 1.0),
                    "PRR_ci_lo": ci_lo,
                    "PRR_ci_hi": ci_hi,
                    "PRR_05": prr(confs, quality, 0.5),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory", type=Path, required=True,
                   help="generation_trajectory_scores.jsonl")
    p.add_argument("--out", type=Path, required=True,
                   help="output metric_scores CSV")
    p.add_argument("--min-generations", type=int, default=1,
                   help="only score instances with >= this many usable generation steps "
                        "(use 2 to isolate the multi-generation effect)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="bootstrap resamples for a 95%% PRR CI (e.g. 1000); 0 = off")
    args = p.parse_args()

    traj_rows = read_jsonl(args.trajectory)
    instances = collect_instances(traj_rows)
    rows = build_rows(instances, min_generations=args.min_generations, n_boot=args.bootstrap)
    write_csv(args.out, rows)

    n_inst = sum(1 for i in instances.values() if i["quality"] is not None)
    n_multi = sum(1 for i in instances.values()
                  if any(len(i["series"][b]) >= 2 for b, _ in BASE_SCORES))
    print(f"instances_with_quality={n_inst}  multi_generation_instances={n_multi}")
    print(f"wrote {len(rows)} metric rows -> {args.out}")


if __name__ == "__main__":
    main()
