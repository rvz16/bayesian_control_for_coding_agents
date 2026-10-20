"""step1_vs_step0.csv: paired (same eval instances) utility difference of each step-1 variant vs its step-0
reference, per replayable cell, paper cost regime, bootstrap 2000 / seed 42.
Reference: step0_g201 for the reset-to-b0 step-1 variants, step0_calK for the calK diagnostics.
usage: repair_effect.py RUNS_DIR OUT_CSV"""
import sys
sys.dont_write_bytecode = True
import csv, os, pickle

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import VARIANTS
from analyze_util import boot_ci

RUNS, OUT = sys.argv[1], sys.argv[2]
load = lambda v: pickle.load(open(os.path.join(RUNS, f"{v}.pkl"), "rb"))
refs = {"step0_g201": load("step0_g201"), "step0_calK": load("step0_calK")}
rows = []
for v in VARIANTS:
    if not VARIANTS[v].get("chains") or not os.path.exists(os.path.join(RUNS, f"{v}.pkl")):
        continue
    ref = "step0_calK" if VARIANTS[v]["flags"].get("calK") else "step0_g201"
    D, R = load(v), refs[ref]
    for k, U in D["util_eval"].items():
        b, g, regime = k
        if regime != ("slow" if b.startswith("swe") else "fast"):
            continue
        m = D["meta"][(b, g)]
        if m["n_replayable_eval"] == 0:
            continue
        S = {r["policy"]: r for r in D["rows"] if (r["bench"], r["gen"], r["regime"]) == k}
        row = dict(variant=v, reference=ref, bench=b, gen=g, n_eval=len(U["bayesian_DP"]), n_replayable_eval=m["n_replayable_eval"])
        for c, p in (("BDP", "bayesian_DP"), ("BG", "bayesian_greedy")):
            d = np.array(U[p]) - np.array(R["util_eval"][k][p])
            lo, hi = boot_ci(d)
            row.update({f"{c}_U": round(float(np.mean(U[p])), 3), f"{c}_U_ref": round(float(np.mean(R["util_eval"][k][p])), 3),
                        f"{c}_diff": round(float(d.mean()), 3), f"{c}_ci95": f"[{lo:+.2f},{hi:+.2f}]",
                        f"{c}_repairs_per_episode": round(S[p]["mean_repair"], 3),
                        f"{c}_frac_episodes_with_repair": round(S[p]["frac_any_repair"], 3)})
        rows.append(row)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", len(rows))
