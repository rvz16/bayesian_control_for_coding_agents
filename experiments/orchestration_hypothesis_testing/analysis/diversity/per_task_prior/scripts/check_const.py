"""Equivalence check: a constant (single-bin) signal must reproduce the diversity_fixes base exactly.
  calK_const  vs diversity_fixes/runs/step0_calK.pkl
  g201_const  vs diversity_fixes/runs/step0_g201.pkl
Compares, for BDP and BG, every paper-regime eval episode (actions, beliefs to 4 dp, patch, obs, utility) and,
for every regime that both runs contain, the per-instance utilities of every policy present in both.
usage: check_const.py RUNS_DIR OUT_JSON"""
import sys
sys.dont_write_bytecode = True
import glob
import json
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT  # noqa: E402

RUNS, OUT = sys.argv[1], sys.argv[2]
DFR = os.path.join(ROOT, "..", "diversity_fixes", "runs")
res = {}
for mine, ref in (("calK_const", "step0_calK"), ("g201_const", "step0_g201")):
    R = pickle.load(open(os.path.join(DFR, f"{ref}.pkl"), "rb"))
    rt = {(t["cell"], t["policy"], t["instance_id"]): t for t in R["traces"]}
    n_ep = bad_ep = n_u = bad_u = 0
    ex = []
    files = sorted(glob.glob(os.path.join(RUNS, mine, "*.pkl")))
    for f in files:
        M = pickle.load(open(f, "rb"))
        for t in M["traces"]:
            if not t["regime"] == ("slow" if t["bench"].startswith("swe") else "fast"):
                continue
            r = rt.get((t["cell"], t["policy"], t["instance_id"]))
            n_ep += 1
            fields = ("actions", "beliefs", "patch", "obs", "utility")
            if r is None or any(t[k] != r[k] for k in fields):
                bad_ep += 1
                if len(ex) < 3:
                    ex.append(dict(mine={k: t[k] for k in fields}, ref=None if r is None else {k: r[k] for k in fields}))
        for key, U in M["util_eval"].items():
            RU = R["util_eval"].get(key)
            if RU is None:
                continue
            for p, u in U.items():
                if p in RU:
                    n_u += 1
                    if list(u) != list(RU[p]):
                        bad_u += 1
                        if len(ex) < 6:
                            ex.append(dict(key=list(key), policy=p))
    res[mine] = dict(reference=ref, n_cells=len(files), n_episodes=n_ep, n_episode_mismatch=bad_ep,
                     n_policy_util_vectors=n_u, n_util_mismatch=bad_u, examples=ex,
                     PASS=(len(files) == 54 and bad_ep == 0 and bad_u == 0 and n_ep > 0))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in res.items()}, indent=1))
