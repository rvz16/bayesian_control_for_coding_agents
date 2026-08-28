#!/usr/bin/env python3
"""Experiment 2c: fuse MULTIPLE UQ critics into bayes_state.

Colleague's follow-up: try bayes + verbalized, and bayes + double(seqprob) +
verbalized (3 critics). Verbalized needs no logits, so it can add signal on
qwen where seqprob is compressed.

Each critic = a binarised UQ score fused via bayes_update, with threshold+theta
fit per fold (honest k-fold, LR-threshold mode from experiment2). Critics are
applied sequentially to the belief.

Signals:
  seqprob     — llm_log_seq_prob from final_logprob_bayes_quality.csv
  verbalized  — from verb_cache.json (per-generation), aggregated (default mean)

Configurations compared (PRR@0.5, paired bootstrap vs bayes):
  bayes                         (baseline)
  bayes + seqprob(double)       (Exp 2b winner)
  bayes + verb
  bayes + seqprob(double) + verb

Example:
  python scripts/experiment2c_multicritic.py \
      --readable-dir .../sage_uncertainty_export/qwen25_32b/lcb_medium \
      --verb-cache   .../sage_uncertainty_export 3/qwen25_32b/lcb_medium/verb_cache.json
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_lcb_llm_tool_agent_logs import prr, prediction_rejection_area as area  # noqa: E402
from experiment2_uq_bayes_critic import bayes_update, fit_threshold_theta  # noqa: E402

MR = 0.5
VERB_AGG = {"mean": statistics.fmean, "last": lambda v: v[-1],
            "min": min, "max": max}


def load_rows(readable_dir: Path, verb_cache: Path, verb_agg: str):
    """Per-instance: bayes, quality, seqprob (raw), verb (aggregated)."""
    verbs = {}
    if verb_cache and verb_cache.exists():
        verbs = {k: [float(x) for x in v] for k, v in json.loads(verb_cache.read_text()).items()}
    agg = VERB_AGG[verb_agg]
    rows = []
    for r in csv.DictReader((readable_dir / "final_logprob_bayes_quality.csv").open()):
        iid = str(r["instance_id"])
        try:
            row = {"iid": iid, "bayes": float(r["bayes_state"]),
                   "quality": int(r["quality"]),
                   "seqprob": float(r["llm_log_seq_prob"])}
        except (KeyError, ValueError):
            continue
        row["verb"] = agg(verbs[iid]) if iid in verbs and verbs[iid] else None
        rows.append(row)
    return rows


# critic spec: (feature_key, higher_better, mode). 'double' expands to two.
def _critic_list(spec: str):
    out = []
    for token in spec.split("+"):
        token = token.strip()
        if token == "seqprob_double":
            out += [("seqprob", True, "lr_pos"), ("seqprob", True, "lr_neg")]
        elif token == "seqprob":
            out += [("seqprob", True, "lr_neg")]
        elif token == "verb":
            out += [("verb", True, "lr_neg")]
        elif token == "verb_sep":
            out += [("verb", True, "sep")]
    return out


def kfold_fuse_multi(rows, critics, *, k=5, seed=0):
    import numpy as np
    # only instances with all needed features present
    need = {c[0] for c in critics}
    usable = [r for r in rows if all(r.get(f) is not None for f in need)]
    rng = np.random.RandomState(seed)
    order = list(range(len(usable)))
    rng.shuffle(order)
    folds = [order[i::k] for i in range(k)]
    fused = {}
    for f in range(k):
        test_idx = set(folds[f])
        train = [usable[i] for i in order if i not in test_idx]
        tq = [r["quality"] for r in train]
        fitted = []
        for feat, hb, mode in critics:
            tf = [r[feat] for r in train]
            thr, p1, p0 = fit_threshold_theta(tf, tq, hb, mode=mode)
            fitted.append((feat, hb, thr, p1, p0))
        for i in test_idx:
            r = usable[i]
            belief = r["bayes"]
            for feat, hb, thr, p1, p0 in fitted:
                passed = (r[feat] >= thr) if hb else (r[feat] <= thr)
                belief = bayes_update(belief, p1, p0, passed)
            fused[r["iid"]] = belief
    return fused, usable


def paired_diff_ci(a_conf, b_conf, quality, *, n_boot=1500, seed=0):
    import numpy as np
    n = len(quality)
    orc = area([-float(q) for q in quality], quality, MR)
    rng = np.random.RandomState(seed); arr = np.arange(n); rnd = []
    for _ in range(200):
        rng.shuffle(arr); s = area(arr.tolist(), quality, MR)
        if s is not None: rnd.append(s)
    ran = float(np.mean(rnd)) if rnd else None
    if orc is None or ran is None or abs(orc - ran) < 1e-12:
        return None
    idx = np.arange(n); diffs = []
    for _ in range(n_boot):
        t = rng.choice(idx, size=n, replace=True); q = [quality[i] for i in t]
        if len(set(q)) < 2: continue
        pa = area([-a_conf[i] for i in t], q, MR); pb = area([-b_conf[i] for i in t], q, MR)
        if pa is not None and pb is not None:
            diffs.append(((pa-ran)/(orc-ran)) - ((pb-ran)/(orc-ran)))
    if len(diffs) < 300: return None
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readable-dir", type=Path, required=True)
    p.add_argument("--verb-cache", type=Path, required=True)
    p.add_argument("--verb-agg", choices=list(VERB_AGG), default="mean")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1500)
    args = p.parse_args()

    rows = load_rows(args.readable_dir, args.verb_cache, args.verb_agg)
    # score only instances that have verb (so all configs are on the same set)
    rows = [r for r in rows if r.get("verb") is not None]
    q = [r["quality"] for r in rows]
    if len(set(q)) < 2:
        print("degenerate, abort"); return
    ids = [r["iid"] for r in rows]
    conf_bayes = {r["iid"]: r["bayes"] for r in rows}

    configs = [
        ("bayes (baseline)", None),
        ("bayes + seqprob(double)", "seqprob_double"),
        ("bayes + verb", "verb"),
        ("bayes + seqprob(double) + verb", "seqprob_double+verb"),
    ]
    print(f"dir={args.readable_dir.name}  n={len(ids)}  pass@1={sum(q)/len(q):.2f}  "
          f"verb_agg={args.verb_agg}  metric=PRR@0.5")
    base_conf = [conf_bayes[i] for i in ids]
    prr_base = prr(base_conf, q, MR)
    print(f"{'config':<34}{'PRR':>8}{'Δ vs bayes':>12}   95% CI")
    print(f"{'bayes (baseline)':<34}{prr_base:>8.3f}")
    for name, spec in configs[1:]:
        critics = _critic_list(spec)
        fused, usable = kfold_fuse_multi(rows, critics, k=args.k)
        # align to full id set (usable == rows here since all have verb+seqprob)
        conf = [fused.get(i, conf_bayes[i]) for i in ids]
        pr = prr(conf, q, MR)
        ci = paired_diff_ci(conf, base_conf, q, n_boot=args.n_boot)
        d = pr - prr_base
        if ci:
            sig = "YES" if (ci[0] > 0 or ci[1] < 0) else "no"
            print(f"{name:<34}{pr:>8.3f}{d:>+12.3f}   [{ci[0]:+.3f}, {ci[1]:+.3f}] {sig}")
        else:
            print(f"{name:<34}{pr:>8.3f}{d:>+12.3f}   n/a")


if __name__ == "__main__":
    main()
