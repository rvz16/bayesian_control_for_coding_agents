#!/usr/bin/env python3
"""Token-count baselines for UQ (colleague's idea).

Baseline: more tokens == higher uncertainty (long answer = model unsure).
Extension: deviation of the final answer's length from the mean / first
generation across regenerations = "the model is thrashing" => uncertain.

Scores per instance (completion_tokens per generation):
  ntok_last            final answer length                (more = uncertain)
  ntok_mean            mean length over generations        (more = uncertain)
  ntok_last_dev_mean   |last - mean|  over gens (>=2)       (bigger = uncertain)
  ntok_last_dev_first  |last - first| over gens (>=2)       (bigger = uncertain)
  ntok_std             std of lengths over gens (>=2)       (bigger = uncertain)

All framed so LOWER raw value = more confident (higher_is_better=False),
i.e. confidence = -score. Scored with PRR@0.5 vs quality, + bootstrap CI.

Example:
  python scripts/ntokens_baseline.py \
      --logprob-dir /…/gptoss03.08part1/codecontests \
      --quality-dir /…/sage_uncertainty_export/gpt_oss_20b/codecontests \
      --out /tmp/ntok_cc.csv
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
from analyze_lcb_llm_tool_agent_logs import prr, prediction_rejection_area as area  # noqa: E402

MR = 0.5


def load_quality(csv_path: Path) -> dict[str, int]:
    out = {}
    for r in csv.DictReader(csv_path.open()):
        try:
            out[str(r["instance_id"])] = int(r["quality"])
        except (KeyError, ValueError):
            pass
    return out


def read_lengths(path: Path) -> dict[str, list[int]]:
    """instance_id -> [completion_tokens per generation, ordered]."""
    by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ct = r.get("completion_tokens")
            if ct is None:
                content = (r.get("logprobs") or {}).get("content") or []
                ct = len(content)
            gi = int(r.get("generation_index", r.get("step", 0)))
            by[str(r["instance_id"])].append((gi, int(ct)))
    out = {}
    for iid, lst in by.items():
        lst.sort(key=lambda x: x[0])
        out[iid] = [c for _, c in lst]
    return out


def bootstrap_ci(conf, quality, *, n_boot=800, seed=0):
    import numpy as np
    n = len(quality); orc = area([-float(q) for q in quality], quality, MR)
    rng = np.random.RandomState(seed); arr = np.arange(n); rnd = []
    for _ in range(200):
        rng.shuffle(arr); s = area(arr.tolist(), quality, MR)
        if s is not None: rnd.append(s)
    ran = float(np.mean(rnd)) if rnd else None
    if orc is None or ran is None or abs(orc - ran) < 1e-12:
        return (None, None)
    idx = np.arange(n); sc = []
    for _ in range(n_boot):
        t = rng.choice(idx, size=n, replace=True); q = [quality[i] for i in t]
        if len(set(q)) < 2: continue
        a = area([-conf[i] for i in t], q, MR)
        if a is not None: sc.append((a - ran) / (orc - ran))
    if len(sc) < 200: return (None, None)
    return (float(np.percentile(sc, 2.5)), float(np.percentile(sc, 97.5)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logprob-dir", type=Path, required=True)
    p.add_argument("--quality-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=800)
    args = p.parse_args()

    quality = load_quality(args.quality_dir / "final_logprob_bayes_quality.csv")
    lengths = read_lengths(args.logprob_dir / "generation_logprobs.jsonl")

    # build score dicts; all "higher raw = more uncertain"
    scores = {k: {} for k in
              ["ntok_last", "ntok_mean", "ntok_last_dev_mean",
               "ntok_last_dev_first", "ntok_std"]}
    for iid, L in lengths.items():
        if iid not in quality or not L:
            continue
        scores["ntok_last"][iid] = L[-1]
        scores["ntok_mean"][iid] = statistics.fmean(L)
        if len(L) >= 2:
            scores["ntok_last_dev_mean"][iid] = abs(L[-1] - statistics.fmean(L))
            scores["ntok_last_dev_first"][iid] = abs(L[-1] - L[0])
            scores["ntok_std"][iid] = statistics.pstdev(L)

    out_rows = []
    print(f"total instances with quality+length: "
          f"{len([i for i in scores['ntok_last']])}")
    for name, d in scores.items():
        ids = list(d)
        q = [quality[i] for i in ids]
        if len(set(q)) < 2 or len(ids) < 12:
            print(f"  {name:<22} n={len(ids)} skip (degenerate/small)")
            continue
        conf = [-d[i] for i in ids]                 # more tokens = less confident
        lo, hi = bootstrap_ci(conf, q, n_boot=args.bootstrap)
        pr = prr(conf, q, MR)
        out_rows.append({"signal": name, "n": len(ids),
                         "pass1": round(sum(q)/len(q), 3),
                         "PRR": pr, "PRR_ci_lo": lo, "PRR_ci_hi": hi})
        cis = f" [{lo:+.2f},{hi:+.2f}]" if lo is not None else ""
        print(f"  {name:<22} n={len(ids):<4} pass@1={sum(q)/len(q):.2f}  "
              f"PRR={pr:+.3f}{cis}")

    if out_rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
