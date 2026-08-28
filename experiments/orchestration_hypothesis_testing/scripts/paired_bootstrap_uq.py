#!/usr/bin/env python3
"""Paired bootstrap: is an aggregation significantly better than `last`?

Comparing two wide PRR confidence intervals is a weak test — it ignores that
both aggregators are scored on the *same* instances, so their sampling noise is
correlated. This computes the CI of the paired *difference* PRR(agg) - PRR(last)
by resampling instances once per bootstrap and scoring both aggregators on that
same resample. A small but consistent edge shows up here even when the raw CIs
overlap. If the difference CI excludes 0, the aggregation beats `last`.

Signals:
  perplexity, llm_log_seq_prob  — from generation_trajectory_scores.jsonl
  verbalized                    — from a verb_cache.json (iid -> [conf,...])
                                  + final_logprob_bayes_quality.csv for quality

Example:
  python scripts/paired_bootstrap_uq.py \
      --readable-dir .../readable/lcb_medium \
      --verb-cache   .../readable/lcb_medium/verb_cache.json \
      --n-boot 2000
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_lcb_llm_tool_agent_logs import prediction_rejection_area  # noqa: E402

AGGS = {
    "mean": lambda v: statistics.fmean(v),
    "max": lambda v: max(v),
    "min": lambda v: min(v),
    "std": lambda v: statistics.pstdev(v) if len(v) > 1 else 0.0,
    "range": lambda v: (max(v) - min(v)) if v else 0.0,
}
# raw-score direction: True = higher raw value means more likely correct
HIGHER_BETTER = {
    "perplexity": False,          # exp(-mean logprob): higher = worse
    "llm_log_seq_prob": True,     # sum logprob: higher = better
    "verbalized": True,           # confidence in [0,1]
}


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def load_quality(csv_path: Path) -> dict[str, int]:
    out = {}
    for r in csv.DictReader(csv_path.open()):
        try:
            out[str(r["instance_id"])] = int(r["quality"])
        except (KeyError, ValueError):
            pass
    return out


def series_from_trajectory(traj_path: Path, field: str) -> dict[str, list[float]]:
    by: dict[str, list[float]] = defaultdict(list)
    rows = sorted(read_jsonl(traj_path),
                  key=lambda r: (str(r["instance_id"]), r.get("patch_idx", 0)))
    for r in rows:
        if r.get("logprobs_supported") and r.get(field) is not None:
            by[str(r["instance_id"])].append(float(r[field]))
    return by


def prr_fast(conf, quality, oracle, random):
    area = prediction_rejection_area([-c for c in conf], quality, 1.0)
    if area is None:
        return None
    return (area - random) / (oracle - random)


def paired_diff_ci(agg_conf, last_conf, quality, *, n_boot, seed=0):
    """CI of PRR(agg) - PRR(last) via paired instance resampling."""
    import numpy as np
    n = len(quality)
    oracle = prediction_rejection_area([-float(q) for q in quality], quality, 1.0)
    rng = np.random.RandomState(seed)
    arr = np.arange(n)
    rnd = []
    for _ in range(200):
        rng.shuffle(arr)
        s = prediction_rejection_area(arr.tolist(), quality, 1.0)
        if s is not None:
            rnd.append(s)
    random = float(np.mean(rnd)) if rnd else None
    if oracle is None or random is None or abs(oracle - random) < 1e-12:
        return (None, None, None)
    base_agg = prr_fast(agg_conf, quality, oracle, random)
    base_last = prr_fast(last_conf, quality, oracle, random)
    idx = np.arange(n)
    diffs = []
    for _ in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        q = [quality[i] for i in take]
        if len(set(q)) < 2:
            continue
        a = prr_fast([agg_conf[i] for i in take], q, oracle, random)
        l = prr_fast([last_conf[i] for i in take], q, oracle, random)
        if a is not None and l is not None:
            diffs.append(a - l)
    if len(diffs) < max(10, n_boot // 4):
        return (base_agg, base_last, None)
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return (base_agg, base_last, (lo, hi))


def run_signal(name: str, series: dict[str, list[float]], quality: dict[str, int],
               *, n_boot: int):
    hib = HIGHER_BETTER[name]
    ids = [i for i in series if i in quality and series[i]]
    q = [quality[i] for i in ids]
    if len(set(q)) < 2:
        print(f"\n== {name}: degenerate (single class), skip =="); return
    last = [(series[i][-1] if hib else -series[i][-1]) for i in ids]
    print(f"\n== {name}  n={len(ids)}  pass@1={sum(q)/len(q):.2f} ==")
    print(f"{'agg vs last':<16}{'PRR(agg)':>9}{'PRR(last)':>10}{'Δ':>8}   95% CI(Δ)   signif")
    for agg_name, fn in AGGS.items():
        agg = [(fn(series[i]) if hib else -fn(series[i])) for i in ids]
        a, l, ci = paired_diff_ci(agg, last, q, n_boot=n_boot)
        if ci is None:
            print(f"{agg_name:<16}{a:>9.3f}{l:>10.3f}{'':>8}   n/a"); continue
        d = a - l
        sig = "YES" if (ci[0] > 0 or ci[1] < 0) else "no"
        print(f"{agg_name:<16}{a:>9.3f}{l:>10.3f}{d:>+8.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}]   {sig}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readable-dir", type=Path, required=True)
    p.add_argument("--verb-cache", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args()

    quality = load_quality(args.readable_dir / "final_logprob_bayes_quality.csv")
    traj = args.readable_dir / "generation_trajectory_scores.jsonl"

    for field in ("perplexity", "llm_log_seq_prob"):
        run_signal(field, series_from_trajectory(traj, field), quality, n_boot=args.n_boot)

    if args.verb_cache and args.verb_cache.exists():
        verbs = {k: [float(x) for x in v]
                 for k, v in json.loads(args.verb_cache.read_text()).items()}
        run_signal("verbalized", verbs, quality, n_boot=args.n_boot)
    else:
        print("\n(verbalized skipped: no --verb-cache)")


if __name__ == "__main__":
    main()
