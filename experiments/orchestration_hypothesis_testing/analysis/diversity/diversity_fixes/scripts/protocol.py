"""Fitting protocol of action_traces (copied verbatim from action_traces/scripts/replay_traces.py,
functions npg_split / fit_likelihoods / fit_kernel / build_cell, plus the cost regimes).
Kept as a copy so that diversity_fixes/ does not import from (or write bytecode into) action_traces/."""
from __future__ import annotations

from collections import Counter

import numpy as np

REGIMES = {
    # experiments.tex:32 "slow oracle, fast critics"
    "slow": dict(c_gen=10.0, c_L0=1.0, c_L2=2.0, c_L3=5.0, c_ver=90.0, reward=100.0),
    # experiments.tex:32 "fast oracle, balanced critics"
    "fast": dict(c_gen=10.0, c_L0=1.0, c_L2=1.0, c_L3=1.0, c_ver=5.0, reward=100.0),
}


def paper_regime(bench: str) -> str:
    return "slow" if bench.startswith("swe") else "fast"


def npg_split(ids, frac=0.75, rounding="int"):
    s = sorted(set(ids))
    o = np.random.default_rng(42).permutation(len(s))
    nt = int(len(s) * frac) if rounding == "int" else int(round(len(s) * frac))
    return [s[i] for i in o[:nt]], [s[i] for i in o[nt:]]


def fit_likelihoods(train_recs, critic_fields):
    n = len(train_recs)
    k = sum(1 for r in train_recs if int(r["Y"]) == 1)
    prior = (k + 1) / (n + 2)
    likes = {}
    for f in critic_fields:
        TP = FP = TN = FN = 0
        for r in train_recs:
            z = r.get(f)
            if z is None:
                continue
            y = int(r["Y"])
            if y == 1 and z: TP += 1
            elif y == 0 and z: FP += 1
            elif y == 0: TN += 1
            else: FN += 1
        if TP + FP + TN + FN == 0:
            continue
        likes[f] = {"P_pass_given_Y1": (TP + 1) / (TP + FN + 2),
                    "P_pass_given_Y0": (FP + 1) / (FP + TN + 2),
                    "TP": TP, "FP": FP, "TN": TN, "FN": FN}
    return prior, {"prior_Y1": prior, "critic_likelihoods": likes}


def fit_kernel(iter_recs, ids=None):
    by = {}
    for r in iter_recs:
        i = str(r["instance_id"])
        if ids is not None and i not in ids:
            continue
        by.setdefault(i, []).append(r)
    c = Counter()
    for rs in by.values():
        rs.sort(key=lambda r: r["step"])
        for a, b in zip(rs, rs[1:]):
            if a.get("Y") is None or b.get("Y") is None:
                continue
            c[f"{int(a['Y'])}{int(b['Y'])}"] += 1
    pf = (c["01"] + 1) / (c["00"] + c["01"] + 2)
    pb = (c["10"] + 1) / (c["10"] + c["11"] + 2)
    return {"P_fix_given_broken": pf, "P_break_given_correct": pb, "counts": dict(c)}


def build_cell(cal_recs, sr_recs, rx_recs, bench, *, kernel_from="selfrefine",
               cal_rounding="int", iter_rounding="round", restrict_pool=False, extra_fields=()):
    by = {}
    for r in cal_recs:
        by.setdefault(str(r["instance_id"]), []).append(r)
    for k in by:
        by[k].sort(key=lambda r: int(r.get("patch_id", 0)))
    train_ids, eval_ids = npg_split(list(by), rounding=cal_rounding)
    train_recs = [r for i in train_ids for r in by[i]]
    fields = ["L0_syntax", "L3_llm_review"] if bench.startswith("swe") else ["L0_syntax", "L2_public_tests", "L3_llm_review"]
    fields = fields + [f for f in extra_fields if f not in fields]
    prior, likes = fit_likelihoods(train_recs, fields)
    src = sr_recs if kernel_from == "selfrefine" else rx_recs
    if src:
        itr, _ = npg_split([str(r["instance_id"]) for r in src], rounding=iter_rounding)
        kernel = fit_kernel(src, set(itr))
    else:
        kernel = {"P_fix_given_broken": prior, "P_break_given_correct": 1 - prior, "counts": None, "iid": True}
    pool = list(eval_ids)
    if restrict_pool:
        sri = {str(r["instance_id"]) for r in sr_recs} if sr_recs else None
        rxi = {str(r["instance_id"]) for r in rx_recs} if rx_recs else None
        pool = [i for i in pool if (sri is None or i in sri) and (rxi is None or i in rxi)]
    return {"by": by, "train_ids": train_ids, "eval_ids": eval_ids, "pool": pool,
            "prior": prior, "likes": likes, "kernel": kernel}
