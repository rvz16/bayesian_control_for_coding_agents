"""Conditional independence of the free signals from the critics (and from each other) given Y, on TRAIN patches.
For each cell and each pair (signal-bin, X) with X in {L0, L2, L3, other signal's bin}: within each Y stratum a
G-test of independence between the bin and X; G and df are summed over the two strata (CMH-style), p from chi2.
Also reported: per-bin P(X pass | Y, bin) for the critics. Bins = SignalModel of variant calK_comb (K=4).
usage: ci_check.py OUT_CSV"""
import sys
sys.dont_write_bytecode = True
import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MATCHED, all_cells, gname  # noqa: E402
from prior_model import SignalModel  # noqa: E402

OUT = sys.argv[1]
SR = defaultdict(dict)
with open(os.path.join(os.path.dirname(OUT), "signals.csv")) as f:
    for r in csv.DictReader(f):
        SR[(r["bench"], r["gen"])][(r["instance_id"], int(r["patch_pos"]))] = r


def gtest(a, x):
    a = np.asarray(a); x = np.asarray(x)
    ua, ux = np.unique(a), np.unique(x)
    if len(ua) < 2 or len(ux) < 2:
        return 0.0, 0
    T = np.array([[np.sum((a == i) & (x == j)) for j in ux] for i in ua], float)
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    m = T > 0
    return float(2 * np.sum(T[m] * np.log(T[m] / E[m]))), (len(ua) - 1) * (len(ux) - 1)


rows = []
for b, g, gg, cell in all_cells():
    rb = SR[(b, gg)]
    sm = SignalModel(dict(task="combtask", patch=("len",), K=4), cell["prior"], rb, cell["train_ids"])
    tr = [r for (i, _), r in rb.items() if i in set(cell["train_ids"])]
    Y = np.array([int(r["Y"]) for r in tr])
    sig = {"task_bin": np.array([sm._tbin(r) for r in tr]),
           "len_bin": np.array([sm._pbin_one(sm.pfeat[0], r) for r in tr])}
    others = {c: np.array([np.nan if r[c] in ("", None) else float(r[c]) for r in tr]) for c in ("L0_syntax", "L2_public_tests", "L3_llm_review")}
    for s, a in sig.items():
        targets = dict(others)
        if s == "task_bin":
            targets["len_bin"] = sig["len_bin"]
        for xname, x in targets.items():
            ok = ~np.isnan(x.astype(float))
            if not ok.any():
                continue
            G = df = 0.0
            per = {}
            for y in (0, 1):
                m = ok & (Y == y)
                gi, di = gtest(a[m], x[m])
                G += gi; df += di
                if xname != "len_bin":
                    per[f"P_pass_Y{y}_by_bin"] = {int(j): round(float(x[m & (a == j)].mean()), 3) if (m & (a == j)).any() else None
                                                  for j in np.unique(a)}
                    per[f"n_Y{y}_by_bin"] = {int(j): int((m & (a == j)).sum()) for j in np.unique(a)}
            p = float(chi2.sf(G, df)) if df > 0 else float("nan")
            rows.append(dict(bench=b, gen=gg, cell=f"{b}/{gg}", matched_paper_cell=MATCHED[(b, gg)], signal=s, against=xname,
                             G=round(G, 3), df=int(df), p=p, violated_p01=bool(df > 0 and p < 0.01),
                             detail=json.dumps(per)))
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
agg = defaultdict(lambda: [0, 0])
for r in rows:
    if r["df"] > 0:
        agg[(r["signal"], r["against"])][0] += 1
        agg[(r["signal"], r["against"])][1] += int(r["violated_p01"])
for k, (n, v) in sorted(agg.items()):
    print(k, f"{v}/{n} cells p<0.01")
