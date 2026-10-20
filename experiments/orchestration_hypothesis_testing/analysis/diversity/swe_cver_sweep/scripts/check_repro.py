"""Reproduction gate: the sweep at cv90_s must equal per_task_prior's 'slow' and cv5_f its 'fast' regime, exactly.
Compares, per (variant, cell): per-instance eval utilities of every policy, train means, thresholded-rule eval
utilities / train means, BG/BDP traces (actions, beliefs, obs, patch, utility) and the summary rows.
usage: check_repro.py SWEEP_RUNS PTP_RUNS OUT_JSON
"""
import glob
import json
import os
import pickle
import sys

SW, PT, OUT = sys.argv[1:4]
PAIRS = [("cv90_s", "slow"), ("cv5_f", "fast")]
DROP = {"regime", "paper_regime"}


def strip(x):
    return {k: v for k, v in x.items() if k not in DROP}


rep = dict(checked=0, mismatches=[], missing=[], n_policy_vectors=0, n_thr_vectors=0, n_traces=0)
for reg, ref in PAIRS:
    for f in sorted(glob.glob(os.path.join(SW, reg, "*", "*.pkl"))):
        v = os.path.basename(os.path.dirname(f)); cell = os.path.basename(f)
        g = os.path.join(PT, v, cell)
        if not os.path.exists(g):
            rep["missing"].append(f"{v}/{cell}"); continue
        a = pickle.load(open(f, "rb")); b = pickle.load(open(g, "rb"))
        ka = (a["bench"], a["gen"], reg); kb = (a["bench"], a["gen"], ref)
        if kb not in b["util_eval"]:
            rep["missing"].append(f"{v}/{cell}:{ref}"); continue
        bad = []
        if a["util_eval"][ka] != b["util_eval"][kb]: bad.append("util_eval")
        if a["util_train"][ka] != b["util_train"][kb]: bad.append("util_train")
        if a["thr_eval"].get(ka) != b["thr_eval"].get(kb): bad.append("thr_eval")
        if a["thr_train"].get(ka) != b["thr_train"].get(kb): bad.append("thr_train")
        ta = [strip(t) for t in a["traces"]]; tb = [strip(t) for t in b["traces"] if t["regime"] == ref]
        if ta != tb: bad.append("traces")
        ra = [strip(r) for r in a["rows"]]; rb = [strip(r) for r in b["rows"] if r["regime"] == ref]
        if ra != rb: bad.append("rows")
        rep["checked"] += 1
        rep["n_policy_vectors"] += len(a["util_eval"][ka]); rep["n_thr_vectors"] += len(a["thr_eval"].get(ka, {}))
        rep["n_traces"] += len(ta)
        if bad:
            rep["mismatches"].append(dict(variant=v, cell=cell, regime=reg, fields=bad))
rep["PASS"] = rep["checked"] > 0 and not rep["mismatches"] and not rep["missing"]
json.dump(rep, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "mismatches"}), "mismatches:", len(rep["mismatches"]))
