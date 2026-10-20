"""Aggregate RUNS/<variant>/*.pkl -> summary.csv, traces.jsonl, rules_vs_bdp.csv, variant_table.csv,
calibration.csv, examples.json (per_task_prior/).

Rule comparisons (paired bootstrap over eval instances, 2000 resamples, seed 42 = analyze_util.boot_ci):
  (i)   oracle rule per cell, picked on EVAL among the base rules (upper bound)
  (ii)  per-benchmark base rule picked on TRAIN (mean train utility over the benchmark's cells)
  (iii) per-benchmark signal-aware rule picked on TRAIN among {base rules} U {thresholded (q, A, B)}:
        at each cached patch follow A if the patch's free-signal posterior >= train quantile q, else B.
  base  the same base controller without signals (variant <base>_const), paired by task.
usage: analyze_ptp.py RUNS_DIR OUT_DIR
"""
import sys
sys.dont_write_bytecode = True
import csv
import glob
import json
import math
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MATCHED, boot_ci, fast_auc  # noqa: E402

RUNS, OUT = sys.argv[1], sys.argv[2]
CTRL = ("bayesian_DP", "bayesian_greedy")
mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
BASE_OF = {"calK": "calK_const", "g201": "g201_const", "calKg201": "calKg201_const"}


def load():
    D = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        v = os.path.basename(d)
        fs = sorted(glob.glob(os.path.join(d, "*.pkl")))
        if len(fs) != 54:
            print("incomplete", v, len(fs)); continue
        D[v] = [pickle.load(open(f, "rb")) for f in fs]
    return D


def ece(p, y, nb=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    idx = np.minimum((p * nb).astype(int), nb - 1)
    return float(sum(abs(p[idx == j].mean() - y[idx == j].mean()) * (idx == j).sum() for j in range(nb) if (idx == j).any()) / len(p))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def is_paper(regime, b):
    return regime == ("slow" if b.startswith("swe") else "fast")


D = load()
order = sorted(D, key=lambda v: (v.split("_")[0], v))

# ------------------------------------------------------------------ summary.csv + traces.jsonl
summary = []
with open(os.path.join(OUT, "traces.jsonl"), "w") as f:
    for v in order:
        for R in D[v]:
            for r in R["rows"]:
                row = dict(r); row["matched_paper_cell"] = MATCHED[(r["bench"], r["gen"])]
                for k in ("top5", "first_action_frac"):
                    row[k] = json.dumps(row[k])
                for k, x in list(row.items()):
                    if isinstance(x, float):
                        row[k] = round(x, 4)
                summary.append(row)
            for t in R["traces"]:
                f.write(json.dumps(t, separators=(",", ":")) + "\n")
keys = []
for r in summary:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)

# ------------------------------------------------------------------ rules_vs_bdp.csv
def thr_util(R, key, combo, split):
    """utility of thresholded rule; a non-informative cell degenerates to rule A (s == tau everywhere)."""
    src = R["thr_eval" if split == "eval" else "thr_train"].get(key)
    if src is not None:
        return src[combo]
    return (R["util_eval"][key] if split == "eval" else R["util_train"][key])[combo[1]]


rule_rows = []
for v in order:
    base = v.split("_")[0]
    Rb = {(R["bench"], R["gen"]): R for R in D.get(BASE_OF[base], [])}
    Rs = {(R["bench"], R["gen"]): R for R in D[v]}
    allkeys = sorted({k for R in D[v] for k in R["util_eval"]})
    regimes_sets = {"paper": [k for k in allkeys if is_paper(k[2], k[0])],
                    "swe_fast": [k for k in allkeys if k[0].startswith("swe") and k[2] == "fast"]}
    for rset, keys_ in regimes_sets.items():
        rules_of = {k: [p for p in Rs[k[:2]]["util_eval"][k] if p not in CTRL] for k in keys_}
        bench_rule, bench_thr = {}, {}
        for bb in sorted({k[0] for k in keys_}):
            ks = [k for k in keys_ if k[0] == bb]
            rs = [r for r in rules_of[ks[0]] if all(r in rules_of[k] for k in ks)]
            s2 = {r: mean([Rs[k[:2]]["util_train"][k][r] for k in ks]) for r in rs}
            bench_rule[bb] = max(s2, key=s2.get)
            combos = [(q, A, B) for q in (0.25, 0.5, 0.75) for A in rs for B in rs if A != B]
            s3 = dict(s2)
            for c in combos:
                s3[c] = mean([thr_util(Rs[k[:2]], k, c, "train") for k in ks])
            bench_thr[bb] = max(s3, key=lambda c: (s3[c], isinstance(c, str)))
        for k in keys_:
            b, g, regime = k
            R = Rs[k[:2]]; U = R["util_eval"][k]
            bdp, bg = np.array(U["bayesian_DP"]), np.array(U["bayesian_greedy"])
            rules = rules_of[k]
            oracle = max(rules, key=lambda r: mean(U[r]))
            thr = bench_thr[b]
            u_thr = np.array(U[thr] if isinstance(thr, str) else thr_util(R, k, thr, "eval"))
            row = dict(variant=v, base=base, regime_set=rset, bench=b, gen=g, regime=regime, matched_paper_cell=MATCHED[(b, g)],
                       informative=R["informative"], n_eval=len(bdp), U_BDP=round(mean(bdp), 3), U_BG=round(mean(bg), 3),
                       U_always_verify=round(mean(U["always_verify"]), 3),
                       oracle_rule=oracle, bench_rule_trainpicked=bench_rule[b],
                       thr_rule_trainpicked=thr if isinstance(thr, str) else f"q{thr[0]}:{thr[1]}|{thr[2]}")
            comps = [("oracle_rule", np.array(U[oracle])), ("bench_rule_trainpicked", np.array(U[bench_rule[b]])),
                     ("thr_rule_trainpicked", u_thr), ("always_verify", np.array(U["always_verify"]))]
            if k[:2] in Rb and k in Rb[k[:2]]["util_eval"]:
                comps.append(("base_controller", None))
            for label, arr in comps:
                for cname, mine in (("BDP", bdp), ("BG", bg)):
                    if label == "base_controller":
                        arr = np.array(Rb[k[:2]]["util_eval"][k]["bayesian_DP" if cname == "BDP" else "bayesian_greedy"])
                    d = mine - arr
                    lo, hi = boot_ci(d)
                    row[f"{cname}_minus_{label}"] = round(float(d.mean()), 3)
                    row[f"{cname}_ci95_{label}"] = f"[{lo:+.2f},{hi:+.2f}]"
                    row[f"{cname}_ci_lo_{label}"] = round(lo, 3); row[f"{cname}_ci_hi_{label}"] = round(hi, 3)
            row["U_thr_rule"] = round(mean(u_thr), 3)
            for r in rules:
                row[f"U_{r}"] = round(mean(U[r]), 3)
            rule_rows.append(row)
keys = []
for r in rule_rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(OUT, "rules_vs_bdp.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rule_rows)

# ------------------------------------------------------------------ variant_table.csv
def ent(c):
    n = sum(c.values())
    return -sum((x / n) * math.log2(x / n) for x in c.values() if x) if n else 0.0


SUBSETS = [("all54", lambda b, g: True), ("matched43", lambda b, g: MATCHED[(b, g)])]
SUBSETS += [(f"bench_{bb}", (lambda bb: lambda b, g: b == bb)(bb)) for bb in
            ["lcb_hard", "lcb_medium", "lcb_easy", "mbpp", "humaneval", "humanevalfix", "codecontests", "swe_lite", "swe_verified"]]
vt = []
for v in order:
    for rset in ("paper", "swe_fast"):
        for sname, keep in SUBSETS:
            if rset == "swe_fast" and not (sname.startswith("bench_swe") or sname == "all54"):
                continue
            for ctrl, pol in (("BDP", "bayesian_DP"), ("BG", "bayesian_greedy")):
                rows = [r for R in D[v] for r in R["rows"] if r["policy"] == pol and keep(r["bench"], r["gen"])
                        and (is_paper(r["regime"], r["bench"]) if rset == "paper" else (r["bench"].startswith("swe") and r["regime"] == "fast"))]
                if not rows:
                    continue
                trs = [t for R in D[v] for t in R["traces"] if t["policy"] == pol and keep(t["bench"], t["gen"])
                       and (is_paper(t["regime"], t["bench"]) if rset == "paper" else (t["bench"].startswith("swe") and t["regime"] == "fast"))]
                rr = [r for r in rule_rows if r["variant"] == v and r["regime_set"] == rset and keep(r["bench"], r["gen"])]
                nseq = [r["n_distinct_seq"] for r in rows]
                e = [r["entropy_bits"] for r in rows]
                row = dict(variant=v, regime_set=rset, subset=sname, controller=ctrl, n_cells=len(rows),
                           cells_first_action_varies=sum(1 for r in rows if r["n_first_actions"] > 1),
                           nseq_median=float(np.median(nseq)), nseq_mean=round(mean(nseq), 2), nseq_max=max(nseq),
                           cells_nseq1=sum(1 for x in nseq if x == 1), pooled_distinct_seq=len({tuple(t["actions"]) for t in trs}),
                           entropy_median=round(float(np.median(e)), 2), entropy_mean=round(mean(e), 2), entropy_max=round(max(e), 2),
                           mean_ep_len=round(mean([r["mean_n_actions"] for r in rows]), 2),
                           mean_delta_vs_av=round(mean([r["delta_vs_av"] for r in rows]), 2))
                for label in ("base_controller", "oracle_rule", "bench_rule_trainpicked", "thr_rule_trainpicked"):
                    ks = [r for r in rr if f"{ctrl}_minus_{label}" in r]
                    d = [r[f"{ctrl}_minus_{label}"] for r in ks]
                    row[f"{label}_mean"] = round(mean(d), 2)
                    row[f"{label}_win_ci"] = sum(1 for r in ks if r[f"{ctrl}_ci_lo_{label}"] > 0)
                    row[f"{label}_lose_ci"] = sum(1 for r in ks if r[f"{ctrl}_ci_hi_{label}"] < 0)
                    row[f"{label}_gt1"] = sum(1 for x in d if x > 1)
                    row[f"{label}_tie"] = sum(1 for x in d if abs(x) <= 1)
                    row[f"{label}_ltm1"] = sum(1 for x in d if x < -1)
                both = [f"{r['bench']}/{r['gen']}" for r in rr if r.get(f"{ctrl}_ci_lo_bench_rule_trainpicked", -1) > 0
                        and r.get(f"{ctrl}_ci_lo_thr_rule_trainpicked", -1) > 0]
                row["cells_beat_ii_and_iii_ci"] = ";".join(both)
                both1 = [f"{r['bench']}/{r['gen']}" for r in rr if r.get(f"{ctrl}_ci_lo_bench_rule_trainpicked", -1) > 0
                         and r.get(f"{ctrl}_ci_lo_thr_rule_trainpicked", -1) > 0
                         and r[f"{ctrl}_minus_bench_rule_trainpicked"] > 1 and r[f"{ctrl}_minus_thr_rule_trainpicked"] > 1]
                row["cells_beat_ii_and_iii_ci_gt1"] = ";".join(both1)
                vt.append(row)
with open(os.path.join(OUT, "variant_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(vt[0].keys())); w.writeheader(); w.writerows(vt)

# ------------------------------------------------------------------ calibration.csv
cal = []
for v in order:
    if not v.endswith(("_const", "_len", "_lint0", "_task", "_diff", "_comb", "_jointcomb", "_jointlen")) or not v.startswith("calK_"):
        continue
    pools = defaultdict(lambda: defaultdict(list))
    for R in D[v]:
        b, g = R["bench"], R["gen"]
        C = R["calib"]
        y = [c["Y"] for c in C]; pr = [c["prior"] for c in C]; b0 = [c["b0"] for c in C]
        y0 = [c["Y"] for c in C if c["patch_pos"] == 0]; b00 = [c["b0"] for c in C if c["patch_pos"] == 0]
        ver = [(t["beliefs"][-1], int(t["reward"] > 0)) for t in R["traces"] if t["policy"] == "bayesian_DP"
               and is_paper(t["regime"], b) and t["actions"] and t["actions"][-1] == "verify"]
        row = dict(variant=v, bench=b, gen=g, matched_paper_cell=MATCHED[(b, g)], n_eval_patches=len(y),
                   brier_prior=round(brier(pr, y), 4), brier_b0=round(brier(b0, y), 4),
                   ece_prior=round(ece(pr, y), 4), ece_b0=round(ece(b0, y), 4),
                   auc_b0=round(fast_auc(b0, y), 4), brier_b0_patch0=round(brier(b00, y0), 4),
                   n_verify=len(ver), brier_at_verify=round(brier([a for a, _ in ver], [c for _, c in ver]), 4) if ver else None,
                   ece_at_verify=round(ece([a for a, _ in ver], [c for _, c in ver]), 4) if ver else None,
                   mean_belief_at_verify=round(mean([a for a, _ in ver]), 4) if ver else None,
                   frac_correct_at_verify=round(mean([c for _, c in ver]), 4) if ver else None)
        cal.append(row)
        for sc in ("all54", "matched43" if MATCHED[(b, g)] else None):
            if sc:
                pools[sc]["y"] += y; pools[sc]["pr"] += pr; pools[sc]["b0"] += b0
                pools[sc]["vb"] += [a for a, _ in ver]; pools[sc]["vy"] += [c for _, c in ver]
    for sc, P in pools.items():
        cal.append(dict(variant=v, bench=f"POOLED_{sc}", gen="", matched_paper_cell="", n_eval_patches=len(P["y"]),
                        brier_prior=round(brier(P["pr"], P["y"]), 4), brier_b0=round(brier(P["b0"], P["y"]), 4),
                        ece_prior=round(ece(P["pr"], P["y"]), 4), ece_b0=round(ece(P["b0"], P["y"]), 4), auc_b0="",
                        brier_b0_patch0="", n_verify=len(P["vy"]), brier_at_verify=round(brier(P["vb"], P["vy"]), 4),
                        ece_at_verify=round(ece(P["vb"], P["vy"]), 4), mean_belief_at_verify=round(mean(P["vb"]), 4),
                        frac_correct_at_verify=round(mean(P["vy"]), 4)))
with open(os.path.join(OUT, "calibration.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(cal[0].keys())); w.writeheader(); w.writerows(cal)

# ------------------------------------------------------------------ examples.json: different first actions within one cell
sig = {}
with open(os.path.join(OUT, "signals.csv")) as f:
    for r in csv.DictReader(f):
        if r["patch_pos"] == "0":
            sig[(r["cell"], r["instance_id"])] = r
ex = {}
for v in ("calK_comb", "calK_jointcomb", "calK_len", "calK_task", "g201_comb"):
    if v not in D:
        continue
    cands = []
    for R in D[v]:
        b = R["bench"]
        tr = [t for t in R["traces"] if t["policy"] == "bayesian_DP" and is_paper(t["regime"], b)]
        firsts = Counter(t["actions"][0] for t in tr)
        if len(firsts) > 1:
            cands.append((MATCHED[(b, R["gen"])], len(firsts), ent(Counter(tuple(t["actions"]) for t in tr)), R, tr, firsts))
    cands.sort(key=lambda c: (-int(c[0]), -c[1], -c[2]))
    out = []
    for matched, nf, H, R, tr, firsts in cands[:4]:
        cell = f"{R['bench']}/{R['gen']}"
        picks = []
        for a in firsts:
            ts = [t for t in tr if t["actions"][0] == a]
            t = sorted(ts, key=lambda t: t["instance_id"])[0]
            s = sig.get((cell, t["instance_id"]), {})
            feats = {k: s[k] for k in s if (k.startswith("t_") and s[k] not in ("", None)) or k in ("diff_chars", "L1_lint", "combtask", "tasklr", "difficulty")}
            picks.append(dict(first_action=a, n_tasks_with_this_first_action=len(ts), instance_id=t["instance_id"],
                              task_features=feats, bT=t["bT"], b0=t["b0"], tbin=t["tbin"], pcombos=t["pcombos"],
                              actions=t["actions"], obs=t["obs"], beliefs=t["beliefs"], patch=t["patch"],
                              Y_fresh=t["Y_fresh"], utility=t["utility"],
                              mean_utility_first_action=round(mean([x["utility"] for x in ts]), 2)))
        out.append(dict(cell=cell, matched_paper_cell=matched, prior=round(R["prior"], 4), first_action_counts=dict(firsts),
                        signal_model=R["signal_model"], examples=picks))
    ex[v] = out
json.dump(ex, open(os.path.join(OUT, "examples.json"), "w"), indent=1, default=str)
print("variants", len(order), "summary", len(summary), "rule rows", len(rule_rows), "vt", len(vt), "calib", len(cal))
