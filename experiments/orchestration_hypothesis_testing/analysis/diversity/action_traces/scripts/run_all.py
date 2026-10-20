"""Replay all 54 cells x {paper regime, both regimes}; write traces + summaries + rule analysis.

usage: run_all.py OUT_DIR. Without PAPER_TEX_DIR the paper comparison is skipped and versions.json /
reproduction_check.csv are not written (the committed copies carry the reproduces_paper_BG_BDP flags)."""
import csv, json, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cells import cell_keys, load_cell
try:
    from paper import full_results, priors
    HAVE_PAPER = True
except ImportError as e:  # PAPER_TEX_DIR unset
    print(f"note: {e}; skipping the paper comparison", file=sys.stderr)
    HAVE_PAPER = False
import replay_traces as rt

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
fr, pr = (full_results(), priors()) if HAVE_PAPER else ({}, {})
RULES = ["always_verify", "never_submit", "gate_syn", "gate_test", "gate_test_giveup", "gate_llm", "fixed_pipeline"]
POLS = ["bayesian_greedy", "bayesian_DP"] + RULES
PAPER_COL = {"gate_syn": "t_syn", "gate_test": "t_test", "gate_llm": "t_llm", "fixed_pipeline": "FP"}


def mean(x): return float(np.mean(x)) if len(x) else float("nan")


def boot_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    if len(d) == 0: return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


versions, repro, summary_rows, rule_rows = {}, [], [], []
train_util = {}   # (cell, regime) -> {rule: mean utility on train}
eval_util = {}    # (cell, regime) -> {policy: per-instance utilities on eval}
tf = open(os.path.join(OUT, "traces.jsonl"), "w")
for (b, g) in cell_keys():
    gg = g.replace("_local", "")
    C = load_cell(b, g)
    cell = rt.build_cell(C["cal"], C["sr"], C["rx"], b, kernel_from="selfrefine",
                         cal_rounding="int", iter_rounding="round", restrict_pool=False)
    versions[f"{b}/{gg}"] = dict(C["versions"], n_instances=len(cell["by"]), n_train=len(cell["train_ids"]),
                                 n_eval=len(cell["pool"]), prior_fit=round(cell["prior"], 4), prior_paper=pr.get((b, gg)),
                                 kernel_selfrefine_75pct=dict(P_fix=round(cell["kernel"]["P_fix_given_broken"], 4),
                                                              P_break=round(cell["kernel"]["P_break_given_correct"], 4)),
                                 likelihoods={k: dict(p1=round(v["P_pass_given_Y1"], 4), p0=round(v["P_pass_given_Y0"], 4))
                                              for k, v in cell["likes"]["critic_likelihoods"].items()})
    for regime in ["fast", "slow"]:
        is_paper = regime == rt.paper_regime(b)
        out = rt.run_cell(cell, b, regime)
        tr_out = rt.run_cell(cell, b, regime, ids=cell["train_ids"])
        train_util[(b, gg, regime)] = {p: mean([e["utility"] for e in tr_out[p]]) for p in tr_out}
        eval_util[(b, gg, regime)] = {p: [e["utility"] for e in out[p]] for p in out}
        u = {p: mean([e["utility"] for e in out[p]]) for p in out}
        av = u["always_verify"]
        if is_paper and HAVE_PAPER:
            row = dict(bench=b, gen=gg, n_eval=len(cell["pool"]),
                       paper_BG=fr[(b, gg)]["BG"], replay_BG=round(u["bayesian_greedy"] - av, 2),
                       paper_BDP=fr[(b, gg)]["BDP"], replay_BDP=round(u["bayesian_DP"] - av, 2))
            for p, col in PAPER_COL.items():
                if p in u:
                    row[f"paper_{col}"] = fr[(b, gg)][col]; row[f"replay_{col}"] = round(u[p] - av, 2)
            row["match_BG"] = abs(row["replay_BG"] - row["paper_BG"]) <= 0.0501
            row["match_BDP"] = abs(row["replay_BDP"] - row["paper_BDP"]) <= 0.0501
            repro.append(row)
        for p in POLS:
            if p not in out: continue
            eps = out[p]
            s = rt.summarize(eps)
            summary_rows.append(dict(bench=b, gen=gg, regime=regime, paper_regime=is_paper, policy=p,
                                     delta_vs_av=round(u[p] - av, 3), **{k: (json.dumps(v) if isinstance(v, (dict, list)) else
                                                                           (round(v, 4) if isinstance(v, float) else v)) for k, v in s.items()}))
            for e in eps:
                tf.write(json.dumps(dict(cell=f"{b}/{gg}", bench=b, gen=gg, regime=regime, paper_regime=is_paper, policy=p,
                                         instance_id=e["instance_id"], actions=e["actions"],
                                         beliefs=[None if x is None else round(float(x), 4) for x in e["beliefs"]],
                                         patch_idx=e["patch_idx"], obs=e["obs"],
                                         Y_patches=[int(r["Y"]) for r in cell["by"][e["instance_id"]]],
                                         utility=e["utility"], cost=e["cost"], reward=e["reward"])) + "\n")
tf.close()
from cells import art_dir
for c, v in versions.items():
    v["artifact_meta"] = {}
    for k in ("calibration", "selfrefine", "reflexion"):
        if v.get(k):
            name, ver = v[k].split(":")
            m = json.load(open(os.path.join(art_dir(name, ver), "_artifact_meta.json")))
            v["artifact_meta"][k] = {"created": m["created"], "digest": m["digest"], "logged_by_run": m.get("logged_by")}
    v["reproduces_paper_BG_BDP"] = next(r["match_BG"] and r["match_BDP"] for r in repro if f"{r['bench']}/{r['gen']}" == c) if HAVE_PAPER else None
if HAVE_PAPER:
    json.dump({"wandb_project": "nlpresearch.group/orchestration-hypothesis-testing",
               "selection_rule": "calibration version whose Beta(1,1)-Laplace train-split prior matches prior_pass_rate.tex; "
                                 "iter (Self-Refine/Reflexion) versions whose 75%-cohort kernel matches transition_kernel.tex; all created <= 2026-08-31",
               "cells": versions}, open(os.path.join(OUT, "versions.json"), "w"), indent=1)
    with open(os.path.join(OUT, "reproduction_check.csv"), "w", newline="") as f:
        keys = sorted({k for r in repro for k in r}, key=lambda k: list(repro[0].keys()).index(k) if k in repro[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(repro)
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)

# ---------------- rule analysis ----------------
cells_paper = [(b, g, "slow" if b.startswith("swe") else "fast") for (b, g, r) in eval_util if r == ("slow" if b.startswith("swe") else "fast")]
avail = lambda key: [r for r in RULES if r in eval_util[key]]
# best single fixed rule across all 54 cells (by mean of per-cell eval utility; and by train)
def single_best(source):
    common = [r for r in RULES if all(r in eval_util[k] for k in cells_paper)]
    # gate_test rules are undefined on SWE; fall back to gate_syn there for a "single" rule? -> report only rules defined everywhere
    score = {r: mean([mean(eval_util[k][r]) if source == "eval" else train_util[k][r] for k in cells_paper]) for r in common}
    return max(score, key=score.get), score
single_eval, single_eval_scores = single_best("eval")
single_train, single_train_scores = single_best("train")
# per-benchmark rule chosen on train
bench_rule = {}
for bb in sorted({k[0] for k in cells_paper}):
    ks = [k for k in cells_paper if k[0] == bb]
    rs = [r for r in RULES if all(r in eval_util[k] for k in ks)]
    sc = {r: mean([train_util[k][r] for k in ks]) for r in rs}
    bench_rule[bb] = max(sc, key=sc.get)
for k in cells_paper:
    b, gg, regime = k
    U = eval_util[k]; rules = avail(k)
    bdp = np.array(U["bayesian_DP"]); bg = np.array(U["bayesian_greedy"])
    oracle = max(rules, key=lambda r: mean(U[r]))
    cell_train = max(rules, key=lambda r: train_util[k][r])
    row = dict(bench=b, gen=gg, regime=regime, n_eval=len(bdp), U_BDP=round(mean(bdp), 3), U_BG=round(mean(bg), 3))
    for r in rules: row[f"U_{r}"] = round(mean(U[r]), 3)
    for label, r in [("oracle_rule", oracle), ("single_rule_evalpicked", single_eval), ("single_rule_trainpicked", single_train),
                     ("bench_rule_trainpicked", bench_rule[b]), ("cell_rule_trainpicked", cell_train)]:
        d = bdp - np.array(U[r]); lo, hi = boot_ci(d)
        row[label] = r; row[f"BDP_minus_{label}"] = round(float(d.mean()), 3); row[f"ci95_{label}"] = f"[{lo:+.2f},{hi:+.2f}]"
        d2 = bg - np.array(U[r]); row[f"BG_minus_{label}"] = round(float(d2.mean()), 3)
    rule_rows.append(row)
with open(os.path.join(OUT, "rules_vs_bdp.csv"), "w", newline="") as f:
    keys = []
    for r in rule_rows:
        for k in r:
            if k not in keys: keys.append(k)
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rule_rows)
json.dump(dict(single_rule_evalpicked=single_eval, single_eval_scores=single_eval_scores,
               single_rule_trainpicked=single_train, single_train_scores=single_train_scores,
               bench_rule_trainpicked=bench_rule), open(os.path.join(OUT, "rules_meta.json"), "w"), indent=1)
print("done", len(repro), "cells;", sum(r["match_BG"] and r["match_BDP"] for r in repro), "match both")
