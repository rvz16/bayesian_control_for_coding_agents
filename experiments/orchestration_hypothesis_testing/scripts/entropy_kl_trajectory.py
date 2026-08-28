#!/usr/bin/env python3
"""Mean token entropy (and KL vs uniform) from raw top_logprobs.

perplexity/seqprob use only the *chosen* token's logprob. Entropy/KL use the
full per-position distribution (top-k logprobs), so they capture how *spread*
the distribution is, not just how likely the taken path was.

Per generation, over its N tokens with top-k distribution p_i(v):
  mean_token_entropy = mean_i [ -sum_v p_i(v) log p_i(v) ]        (nats)
  mean_token_kl_unif = mean_i [ sum_v p_i(v) log(p_i(v)/(1/k)) ]  (= log k - H_i)

Then aggregate per instance over generations (last/mean/min/max) and score with
PRR@0.5 vs the final quality label, exactly like the logprob signals.

Input: colleague's raw export dir with per-dataset generation_logprobs.jsonl
(fields: instance_id, generation_index, logprobs.content[i].top_logprobs).
Quality label from the matching final_logprob_bayes_quality.csv (may live in a
separate export — pass --quality-dir).

Example:
  python scripts/entropy_kl_trajectory.py \
      --logprob-dir /…/gptoss03.08part1/lcb_medium \
      --quality-dir "/…/sage_uncertainty_export/gpt_oss_20b/lcb_medium" \
      --out /…/entropy_lcb_medium.csv
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_lcb_llm_tool_agent_logs import prr, prediction_rejection_area as area  # noqa: E402

MR = 0.5


def token_entropy_kl(top_logprobs: list) -> tuple[float, float] | None:
    """Return (entropy, kl_vs_uniform) in nats for one token's top-k dist."""
    lps = [t.get("logprob") for t in top_logprobs if t.get("logprob") is not None]
    lps = [x for x in lps if math.isfinite(x)]
    if len(lps) < 2:
        return None
    ps = [math.exp(x) for x in lps]
    Z = sum(ps)
    if Z <= 0:
        return None
    ps = [p / Z for p in ps]                     # renormalise over the top-k
    H = -sum(p * math.log(p) for p in ps if p > 0)
    k = len(ps)
    kl = math.log(k) - H                          # KL(p || uniform_k)
    return H, kl


def token_self_certainty(top_logprobs: list) -> float | None:
    """lm-polygraph self-certainty per token = KL(uniform || p)
       = -mean_v(log p_v) - log V, using the RAW top-k logprobs (not renormalised).
       Higher = more certain. NB: computed on top-k here (API), not full vocab."""
    lps = [t.get("logprob") for t in top_logprobs if t.get("logprob") is not None]
    lps = [x for x in lps if math.isfinite(x) and x > -9998]
    if len(lps) < 2:
        return None
    return -statistics.fmean(lps) - math.log(len(lps))


def gen_stats(content: list) -> tuple[float, float, float] | None:
    """Mean token entropy, mean KL(vs uniform), and mean self-certainty."""
    Hs, KLs, SCs = [], [], []
    for tok in content:
        tl = tok.get("top_logprobs") or []
        r = token_entropy_kl(tl)
        if r is not None:
            Hs.append(r[0]); KLs.append(r[1])
        sc = token_self_certainty(tl)
        if sc is not None:
            SCs.append(sc)
    if not Hs or not SCs:
        return None
    return statistics.fmean(Hs), statistics.fmean(KLs), statistics.fmean(SCs)


def load_quality(csv_path: Path) -> dict[str, int]:
    out = {}
    for r in csv.DictReader(csv_path.open()):
        try:
            out[str(r["instance_id"])] = int(r["quality"])
        except (KeyError, ValueError):
            pass
    return out


def read_generations(path: Path):
    """Stream (instance_id, gen_index, entropy, kl) — avoids loading 10 GB."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            content = (r.get("logprobs") or {}).get("content") or []
            st = gen_stats(content)
            if st is None:
                continue
            yield (str(r["instance_id"]),
                   int(r.get("generation_index", r.get("step", 0))),
                   st[0], st[1], st[2])


AGGS = {"last": lambda v: v[-1], "mean": statistics.fmean, "min": min, "max": max}


def bootstrap_ci(conf, quality, *, n_boot=1000, seed=0):
    import numpy as np
    n = len(quality)
    orc = area([-float(q) for q in quality], quality, MR)
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
    p.add_argument("--logprob-dir", type=Path, required=True,
                   help="dir with generation_logprobs.jsonl (raw top_logprobs)")
    p.add_argument("--quality-dir", type=Path, required=True,
                   help="dir with final_logprob_bayes_quality.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    quality = load_quality(args.quality_dir / "final_logprob_bayes_quality.csv")

    # entropy LOWER = more confident (hib=False); kl_uniform & self_certainty
    # HIGHER = more confident (hib=True).
    ent = defaultdict(list); kl = defaultdict(list); sc = defaultdict(list)
    rows = list(read_generations(args.logprob_dir / "generation_logprobs.jsonl"))
    rows.sort(key=lambda r: (r[0], r[1]))
    for iid, gi, H, K, S in rows:
        if iid in quality:
            ent[iid].append(H); kl[iid].append(K); sc[iid].append(S)

    out_rows = []
    for name, series, higher_better in [("entropy", ent, False),
                                        ("kl_uniform", kl, True),
                                        ("self_certainty", sc, True)]:
        ids = [i for i in series if series[i]]
        q = [quality[i] for i in ids]
        if len(set(q)) < 2:
            continue
        for agg, fn in AGGS.items():
            conf = [(fn(series[i]) if higher_better else -fn(series[i])) for i in ids]
            lo, hi = bootstrap_ci(conf, q, n_boot=args.bootstrap)
            out_rows.append({
                "signal": name, "aggregator": agg, "n": len(ids),
                "higher_is_better": higher_better,
                "PRR": prr(conf, q, MR), "PRR_ci_lo": lo, "PRR_ci_hi": hi,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    n = len({r["n"] for r in out_rows})
    print(f"n_instances={out_rows[0]['n']}  pass@1="
          f"{sum(quality[i] for i in ent if i in quality)/max(1,len([i for i in ent if i in quality])):.2f}")
    for r in out_rows:
        ci = (f" [{r['PRR_ci_lo']:+.2f},{r['PRR_ci_hi']:+.2f}]"
              if r["PRR_ci_lo"] is not None else "")
        print(f"  {r['signal']+'_'+r['aggregator']:<22} PRR={r['PRR']:+.3f}{ci}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
