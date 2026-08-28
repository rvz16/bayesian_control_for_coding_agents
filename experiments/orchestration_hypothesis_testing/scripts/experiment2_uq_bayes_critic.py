#!/usr/bin/env python3
"""Experiment 2: fuse a logprob UQ signal into the Bayesian belief as a critic.

`bayes_state` is a posterior P(correct) built from critic/verifier outcomes.
Here we treat a logprob UQ score (default: llm_log_seq_prob of the final answer,
the strongest single signal from Experiment 1) as ONE MORE binary critic:

    uq_passed   = 1[ score >= threshold ]                    # "model is confident"
    theta_uq    = { P(uq_passed | Y=1), P(uq_passed | Y=0) } # learned from data
    belief_uq   = bayes_update(bayes_state, theta_uq, uq_passed)

Direction/strength are learned via theta (a near-uninformative theta => the
fusion barely moves belief => UQ adds nothing beyond the tools). Threshold and
theta are fit with honest k-fold CV over instances (no instance is scored by a
model that saw it), then we compare PRR of:
  - bayes_state (baseline)
  - the UQ feature alone
  - the fused belief
plus a PAIRED bootstrap CI of (fused - bayes) to test significance.

All inputs come from <readable>/final_logprob_bayes_quality.csv — no re-run.

Example:
  python scripts/experiment2_uq_bayes_critic.py \
      --readable-dir .../readable/lcb_medium --feature llm_log_seq_prob --k 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_lcb_llm_tool_agent_logs import prediction_rejection_area, prr  # noqa: E402

# raw-score direction: higher raw value == more likely correct?
HIGHER_BETTER = {
    "llm_log_seq_prob": True,   # sum logprob
    "llm_perplexity": True,     # mean logprob (higher==closer to 0==confident)
    "perplexity": False,        # exp(-mean logprob): higher==worse
}


def load_final(path: Path, feature: str) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open()):
        try:
            out.append({
                "iid": str(r["instance_id"]),
                "bayes": float(r["bayes_state"]),
                "quality": int(r["quality"]),
                "feat_raw": float(r[feature]),
            })
        except (KeyError, ValueError):
            continue
    return out


def bayes_update(belief: float, p_pass_y1: float, p_pass_y0: float, passed: bool) -> float:
    p = p_pass_y1 if passed else (1.0 - p_pass_y1)
    q = p_pass_y0 if passed else (1.0 - p_pass_y0)
    num = belief * p
    den = num + (1.0 - belief) * q
    return num / den if den > 0 else belief


def _theta_for_threshold(feats, quality, thr, higher_better):
    """θ = (p1, p0) for the critic uq_passed = feat>=thr (or <= if lower-better)."""
    n1 = sum(quality) or 1
    n0 = (len(quality) - sum(quality)) or 1
    passed = [(f >= thr) if higher_better else (f <= thr) for f in feats]
    p1 = (sum(p for p, y in zip(passed, quality) if y == 1) + 1) / (n1 + 2)  # Beta(1,1)
    p0 = (sum(p for p, y in zip(passed, quality) if y == 0) + 1) / (n0 + 2)
    return p1, p0


def fit_threshold_theta(feats, quality, higher_better, *, mode="sep"):
    """Pick a threshold over a quantile grid, by one of several criteria:

      sep     — max |p1 - p0|            (balanced separation; the original)
      lr_pos  — max p1/p0                (decisive-correct: 'pass' ⇒ likely Y=1)
      lr_neg  — min (1-p1)/(1-p0)        (decisive-error:   'fail' ⇒ likely Y=0)

    Returns (thr, p1, p0). Beta(1,1) smoothing keeps p0>0 / p1<1 so updates stay
    finite even at an 'ideal' threshold.
    """
    srt = sorted(feats)
    best = None
    for qtl in [i / 20 for i in range(1, 20)]:
        thr = srt[min(len(srt) - 1, int(qtl * len(srt)))]
        p1, p0 = _theta_for_threshold(feats, quality, thr, higher_better)
        if mode == "sep":
            score = abs(p1 - p0)
        elif mode == "lr_pos":
            score = p1 / p0                      # want ≫ 1
        elif mode == "lr_neg":
            score = -((1 - p1) / (1 - p0))       # want (1-p1)/(1-p0) ≪ 1
        else:
            raise ValueError(mode)
        if best is None or score > best[0]:
            best = (score, thr, p1, p0)
    return best[1], best[2], best[3]


def kfold_fuse(rows, higher_better, *, k, seed, mode="sep"):
    """Fuse UQ critic(s) into bayes via honest k-fold.

    mode in {sep, lr_pos, lr_neg}: one critic with that threshold criterion.
    mode == 'double': fit BOTH lr_pos and lr_neg on train, apply both critics
    sequentially to the belief (two pieces of evidence from one score).
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    folds = [order[i::k] for i in range(k)]
    fused = {}
    modes = ["lr_pos", "lr_neg"] if mode == "double" else [mode]
    for f in range(k):
        test_idx = set(folds[f])
        train = [rows[i] for i in order if i not in test_idx]
        tf = [r["feat_raw"] for r in train]
        tq = [r["quality"] for r in train]
        fitted = [(m, *fit_threshold_theta(tf, tq, higher_better, mode=m)) for m in modes]
        for i in test_idx:
            r = rows[i]
            belief = r["bayes"]
            for _m, thr, p1, p0 in fitted:
                passed = (r["feat_raw"] >= thr) if higher_better else (r["feat_raw"] <= thr)
                belief = bayes_update(belief, p1, p0, passed)
            fused[r["iid"]] = belief
    return fused


def paired_diff_ci(a_conf, b_conf, quality, *, n_boot, max_rej=0.5, seed=0):
    """CI of PRR(a) - PRR(b) at rejection budget max_rej, paired resampling."""
    import numpy as np
    n = len(quality)
    oracle = prediction_rejection_area([-float(q) for q in quality], quality, max_rej)
    rng = np.random.RandomState(seed)
    arr = np.arange(n)
    rnd = []
    for _ in range(200):
        rng.shuffle(arr)
        s = prediction_rejection_area(arr.tolist(), quality, max_rej)
        if s is not None:
            rnd.append(s)
    random = float(np.mean(rnd)) if rnd else None
    if oracle is None or random is None or abs(oracle - random) < 1e-12:
        return None

    diffs = []
    idx = np.arange(n)
    for _ in range(n_boot):
        take = rng.choice(idx, size=n, replace=True)
        q = [quality[i] for i in take]
        if len(set(q)) < 2:
            continue
        pa = prediction_rejection_area([-a_conf[i] for i in take], q, max_rej)
        pb = prediction_rejection_area([-b_conf[i] for i in take], q, max_rej)
        if pa is not None and pb is not None:
            diffs.append(((pa - random) / (oracle - random))
                         - ((pb - random) / (oracle - random)))
    if len(diffs) < max(10, n_boot // 4):
        return None
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--readable-dir", type=Path, required=True)
    p.add_argument("--feature", choices=list(HIGHER_BETTER), default="llm_log_seq_prob")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--max-rej", type=float, default=0.5,
                   help="PRR rejection budget; 0.5 = PRR@0.5 (matches the main tables)")
    p.add_argument("--mode", choices=["sep", "lr_pos", "lr_neg", "double"], default="sep",
                   help="threshold criterion: sep=|p1-p0| (orig); lr_pos=decisive-correct; "
                        "lr_neg=decisive-error; double=fuse both lr critics")
    args = p.parse_args()

    hb = HIGHER_BETTER[args.feature]
    rows = load_final(args.readable_dir / "final_logprob_bayes_quality.csv", args.feature)
    q = [r["quality"] for r in rows]
    if len(set(q)) < 2:
        print("degenerate (single class), abort"); return

    fused = kfold_fuse(rows, hb, k=args.k, seed=args.seed, mode=args.mode)

    ids = [r["iid"] for r in rows]
    quality = [r["quality"] for r in rows]
    conf_bayes = [r["bayes"] for r in rows]
    conf_feat = [(r["feat_raw"] if hb else -r["feat_raw"]) for r in rows]
    conf_fused = [fused[r["iid"]] for r in rows]

    mr = args.max_rej
    prr_bayes = prr(conf_bayes, quality, mr)
    prr_feat = prr(conf_feat, quality, mr)
    prr_fused = prr(conf_fused, quality, mr)
    ci = paired_diff_ci(conf_fused, conf_bayes, quality, n_boot=args.n_boot, max_rej=mr)

    tag = f"PRR@{mr:g}"
    print(f"dir={args.readable_dir}  feature={args.feature}  n={len(ids)}  "
          f"pass@1={sum(quality)/len(quality):.2f}  k={args.k}  metric={tag}")
    print(f"{'signal':<26}{tag:>8}")
    print(f"{'bayes_state (baseline)':<26}{prr_bayes:>8.3f}")
    print(f"{args.feature + ' alone':<26}{prr_feat:>8.3f}")
    print(f"{'bayes + UQ critic (fused)':<26}{prr_fused:>8.3f}")
    if ci is not None:
        d = prr_fused - prr_bayes
        sig = "YES" if (ci[0] > 0 or ci[1] < 0) else "no"
        print(f"\nΔ(fused - bayes) = {d:+.3f}   95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]   "
              f"significant: {sig}")
    else:
        print("\nΔ CI: n/a")


if __name__ == "__main__":
    main()
