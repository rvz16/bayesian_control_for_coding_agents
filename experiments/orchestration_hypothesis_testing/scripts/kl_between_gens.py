#!/usr/bin/env python3
"""KL divergence between the first and final generation of an instance.

Colleague's idea: compare the model's token distribution when it solved in one
generation vs when it regenerated several times. Concretely, for an instance
with >=2 generations we compare the FIRST generation's distribution with the
FINAL one:  KL( p_first || p_final ).

Since the two generations are different code of different length, per-position
distributions aren't aligned. We approximate each generation by a *mixture*
distribution: average the top-k token probabilities over all positions
("which tokens the model tends to use in this attempt"), then take the KL
between the first and final mixtures over the union vocabulary (floor-smoothed).

Interpretation: large KL = the model changed its token repertoire a lot between
attempts = it was "unsure / searching". We score BOTH sign conventions with
PRR@0.5 and let the data say which direction predicts correctness.

Instances with a single generation are excluded (no pair to compare).

Example:
  python scripts/kl_between_gens.py \
      --logprob-dir /…/gptoss03.08part1/codecontests \
      --quality-dir /…/sage_uncertainty_export/gpt_oss_20b/codecontests \
      --out /tmp/kl_cc.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_lcb_llm_tool_agent_logs import prr, prediction_rejection_area as area  # noqa: E402

MR = 0.5
FLOOR = 1e-8


def mixture_dist(content: list) -> dict[str, float] | None:
    """Average top-k token probabilities over all positions of a generation."""
    acc: dict[str, float] = defaultdict(float)
    npos = 0
    for tok in content:
        tl = tok.get("top_logprobs") or []
        if not tl:
            continue
        npos += 1
        for t in tl:
            lp = t.get("logprob")
            if lp is not None and math.isfinite(lp):
                acc[t.get("token")] += math.exp(lp)
    if npos == 0 or not acc:
        return None
    Z = sum(acc.values())
    if Z <= 0:
        return None
    return {k: v / Z for k, v in acc.items()}


def kl(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over union vocab, floor-smoothed on q."""
    out = 0.0
    for t, pt in p.items():
        if pt <= 0:
            continue
        qt = q.get(t, FLOOR)
        out += pt * math.log(pt / max(qt, FLOOR))
    return out


def load_quality(csv_path: Path) -> dict[str, int]:
    out = {}
    for r in csv.DictReader(csv_path.open()):
        try:
            out[str(r["instance_id"])] = int(r["quality"])
        except (KeyError, ValueError):
            pass
    return out


def read_instance_gens(path: Path):
    """Stream generations grouped per instance (ordered by gen index).

    Yields (instance_id, [mixture_dist per generation in order]).
    """
    # group line offsets by instance to keep memory bounded-ish; here we build
    # mixtures on the fly (each generation reduced to a small dict immediately).
    by: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            content = (r.get("logprobs") or {}).get("content") or []
            m = mixture_dist(content)
            if m is None:
                continue
            gi = int(r.get("generation_index", r.get("step", 0)))
            by[str(r["instance_id"])].append((gi, m))
    for iid, lst in by.items():
        lst.sort(key=lambda x: x[0])
        yield iid, [m for _, m in lst]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logprob-dir", type=Path, required=True)
    p.add_argument("--quality-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    quality = load_quality(args.quality_dir / "final_logprob_bayes_quality.csv")

    kls: dict[str, float] = {}
    n_single = 0
    for iid, mixes in read_instance_gens(args.logprob_dir / "generation_logprobs.jsonl"):
        if iid not in quality:
            continue
        if len(mixes) < 2:
            n_single += 1
            continue
        kls[iid] = kl(mixes[0], mixes[-1])   # first vs final

    ids = list(kls)
    q = [quality[i] for i in ids]
    print(f"instances with >=2 gen: {len(ids)}  (single-gen excluded: {n_single})")
    if len(set(q)) < 2:
        print("degenerate on multi-gen subset, abort"); return
    print(f"pass@1 (multi-gen subset): {sum(q)/len(q):.2f}")

    import numpy as np
    def ci(conf):
        n = len(q); orc = area([-float(x) for x in q], q, MR)
        rng = np.random.RandomState(0); arr = np.arange(n); rnd = []
        for _ in range(200):
            rng.shuffle(arr); s = area(arr.tolist(), q, MR)
            if s is not None: rnd.append(s)
        ran = float(np.mean(rnd))
        if abs(orc - ran) < 1e-12: return (None, None)
        idx = np.arange(n); sc = []
        for _ in range(args.bootstrap):
            t = rng.choice(idx, size=n, replace=True); qq = [q[i] for i in t]
            if len(set(qq)) < 2: continue
            a = area([-conf[i] for i in t], qq, MR)
            if a is not None: sc.append((a - ran) / (orc - ran))
        return (float(np.percentile(sc, 2.5)), float(np.percentile(sc, 97.5))) if len(sc) >= 200 else (None, None)

    rows = []
    for name, conf in [("kl_high_conf", [kls[i] for i in ids]),        # large KL = confident
                       ("kl_low_conf", [-kls[i] for i in ids])]:       # small KL = confident
        lo, hi = ci(conf)
        rows.append({"signal": name, "n": len(ids), "PRR": prr(conf, q, MR),
                     "PRR_ci_lo": lo, "PRR_ci_hi": hi})
        cis = f" [{lo:+.2f},{hi:+.2f}]" if lo is not None else ""
        print(f"  {name:<14} PRR={prr(conf,q,MR):+.3f}{cis}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
