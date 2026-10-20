"""Step 2/3 runner: BDP / BG with per-task / per-patch starting beliefs + fair rules (incl. signal-thresholded).
One pickle per (variant, cell) under RUNS/<variant>/ (skip existing; FORCE=1 to redo).

usage: run_ptp.py RUNS_DIR [variant ...]
"""
import sys
sys.dont_write_bytecode = True
import csv
import os
import math
import pickle
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, gname  # noqa: E402
from cells import cell_keys  # noqa: E402
from engine import Greedy, make_gate, policy_always_verify  # noqa: E402
from protocol import paper_regime  # noqa: E402
from runner import cal_kernel, make_policies, prepare, run  # noqa: E402
from variants import VARIANTS as DFV  # noqa: E402
from prior_model import JointSignalModel, SignalModel  # noqa: E402
from ptp_engine import (JointDP, JointInfo, SigDP, TaskInfo, make_joint_dp_policy, make_joint_greedy_policy,  # noqa: E402
                        make_sig_dp_policy, make_sig_greedy_policy, make_threshold_rule)

RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs")
_argv = sys.argv
sys.argv = [_argv[0], RUNS]          # run_variants.py reads WORK=sys.argv[1] at import; reuse its summarize()
os.makedirs(RUNS, exist_ok=True)
from run_variants import summarize  # noqa: E402
sys.argv = _argv

BASES = {
    "calK": DFV["step0_calK"]["flags"],                         # primary base (H3, 51-point grid)
    "g201": DFV["step0_g201"]["flags"],                         # secondary base (reset on regenerate)
    "calKg201": dict(DFV["step0_calK"]["flags"], n_b=201),      # sensitivity: calK on the 201-point grid
}
SIG = {
    "const": dict(task="const", patch=()),
    "len": dict(task=None, patch=("len",)),
    "lint0": dict(task=None, patch=("lint",)),
    "task": dict(task="tasklr", patch=()),
    "diff": dict(task="difficulty", patch=()),
    "comb": dict(task="combtask", patch=("len", "lint")),
    "comb_nolint": dict(task="combtask", patch=("len",)),
}
VARIANTS = {}
for base in ("calK", "g201"):
    for s in ("const", "len", "lint0", "task", "diff", "comb"):
        VARIANTS[f"{base}_{s}"] = dict(base=base, sig=dict(SIG[s], K=4), L1=False)
VARIANTS["calK_comb_lint1"] = dict(base="calK", sig=dict(SIG["comb_nolint"], K=4), L1=True)   # lint as a cost-1 critic
VARIANTS["calK_comb_K3"] = dict(base="calK", sig=dict(SIG["comb"], K=3), L1=False)
VARIANTS["calK_comb_K5"] = dict(base="calK", sig=dict(SIG["comb"], K=5), L1=False)
VARIANTS["calKg201_comb"] = dict(base="calKg201", sig=dict(SIG["comb"], K=4), L1=False)
VARIANTS["calKg201_const"] = dict(base="calKg201", sig=dict(SIG["const"], K=4), L1=False)
# joint likelihoods (ci_check.py rejects naive Bayes): P(len bin | Y, task bin), P(critic | Y, len bin)
VARIANTS["calK_jointcomb"] = dict(base="calK", sig=dict(SIG["comb_nolint"], K=4), L1=False, joint=True)
VARIANTS["calK_jointlen"] = dict(base="calK", sig=dict(SIG["len"], K=4), L1=False, joint=True)
VARIANTS["g201_jointcomb"] = dict(base="g201", sig=dict(SIG["comb_nolint"], K=4), L1=False, joint=True)

# Phase-B sizing (NOT a real signal): synthetic patch-level score s = delta*Y + N(0,1), delta = sqrt(2)*Phi^-1(a),
# so its population AUROC is a. Generated from the outcome on purpose, to measure the utility a free signal of a
# given AUROC would buy BDP and the signal-thresholded rules; never mixed into the real-signal variants.
for a in (0.6, 0.7, 0.8, 0.9, 0.95):
    VARIANTS[f"calK_synth{int(round(a * 100))}"] = dict(base="calK", sig=dict(task=None, patch=("col:synth",), K=5), L1=False, synth=a)

QS = (0.25, 0.5, 0.75)


def add_synth(rows_by_patch, a, cellname):
    import zlib
    from scipy.stats import norm
    delta = math.sqrt(2) * norm.ppf(a)
    rng = np.random.default_rng(zlib.crc32(cellname.encode()))
    out = {}
    for k in sorted(rows_by_patch):
        r = dict(rows_by_patch[k])
        r["synth"] = delta * int(r["Y"]) + rng.standard_normal()
        out[k] = r
    return out
_SIGROWS = None


def sigrows():
    global _SIGROWS
    if _SIGROWS is None:
        _SIGROWS = defaultdict(dict)
        with open(os.path.join(ROOT, "signals.csv")) as f:
            for r in csv.DictReader(f):
                _SIGROWS[(r["bench"], r["gen"])][(r["instance_id"], int(r["patch_pos"]))] = r
    return _SIGROWS


def regimes_for(b):
    p = paper_regime(b)
    return [p] + (["fast"] if p == "slow" else [])


def run_cell(job):
    vname, b, g = job
    gg = gname(g)
    out_path = os.path.join(RUNS, vname, f"{b}__{gg}.pkl")
    if os.path.exists(out_path) and os.environ.get("FORCE") != "1":
        return vname, b, gg, "skip"
    Vx = VARIANTS[vname]
    V = dict(flags=BASES[Vx["base"]], L1=Vx["L1"])
    C, cell, cd, eps = prepare(b, g, V)
    rows_by_patch = sigrows()[(b, gg)]
    if Vx.get("synth"):
        rows_by_patch = add_synth(rows_by_patch, Vx["synth"], f"{b}/{gg}")
    joint = Vx.get("joint", False)
    infos = {}
    if joint:
        sm = JointSignalModel(Vx["sig"], cell["prior"], rows_by_patch, cell["train_ids"], cell["likes"]["critic_likelihoods"])
        for i, ep in eps.items():
            bT, tbin = sm.task_belief(rows_by_patch[(i, 0)])
            infos[i] = JointInfo(bT, tbin, [sm.lbin(rows_by_patch[(i, k)]) for k in range(len(ep.fresh))], sm)
    else:
        sm = SignalModel(Vx["sig"], cell["prior"], rows_by_patch, cell["train_ids"])
        for i, ep in eps.items():
            bT, tbin = sm.task_belief(rows_by_patch[(i, 0)])
            infos[i] = TaskInfo(bT, tbin, [sm.patch_combo(rows_by_patch[(i, k)]) for k in range(len(ep.fresh))], sm)
    informative = (sm.n_task_bins > 1) or (sm.patch_lik() is not None)
    tau = {q: float(np.quantile([infos[i].b0[k] for i in cell["train_ids"] for k in range(len(eps[i].fresh))], q)) for q in QS}
    res = dict(variant=vname, bench=b, gen=gg, prior=cell["prior"],
               synth_auc_eval=(None if not Vx.get("synth") else __import__("common").fast_auc(
                   [rows_by_patch[(i, k)]["synth"] for i in cell["pool"] for k in range(len(eps[i].fresh))],
                   [int(rows_by_patch[(i, k)]["Y"]) for i in cell["pool"] for k in range(len(eps[i].fresh))])), signal_model=sm.describe(), tau=tau,
               informative=informative, n_eval=len(cell["pool"]), rows=[], util_eval={}, util_train={},
               thr_eval={}, thr_train={}, traces=[], calib=[])
    for i in cell["pool"]:
        for k, r in enumerate(eps[i].fresh):
            res["calib"].append(dict(instance_id=i, patch_pos=k, Y=int(r["Y"]), prior=cell["prior"], bT=infos[i].bT,
                                     tbin=infos[i].tbin, pcombo=infos[i].pcombos[k], b0=infos[i].b0[k]))
    for regime in regimes_for(b):
        rpols, cost, fl = make_policies(cell, cd, b, regime, V, only_rules=True)
        rpols["gate_syn_giveup"] = make_gate("L0", "give_up")
        rpols["gate_llm_giveup"] = make_gate("L3", "give_up")
        rule_names = list(rpols)
        ck = cal_kernel(cell)
        likes = cell["likes"]["critic_likelihoods"]
        critics = None if V.get("L1") else ["L0", "L2", "L3"]
        mode = "calK" if fl.calK else "task"
        dps, grs = {}, {}

        def get_dp(info):
            if joint:
                key = (info.tbin, info.bT)
                if key not in dps:
                    dps[key] = JointDP(cell["prior"] if mode == "calK" else info.bT, likes, cell["kernel"], cost, fl,
                                       critics=critics, cal_kernel=ck, jsm=sm, tbin=info.tbin)
                return dps[key]
            key = None if mode == "calK" else info.bT
            if key not in dps:
                dps[key] = SigDP(cell["prior"] if mode == "calK" else info.bT, likes, cell["kernel"], cost, fl,
                                 critics=critics, cal_kernel=ck, patch_lik=sm.patch_lik())
            return dps[key]

        def get_gr(info):
            key = None if mode == "calK" else info.bT
            if key not in grs:
                grs[key] = Greedy(cell["prior"] if mode == "calK" else info.bT, likes, cost, fl, critics=critics, cal_kernel=ck)
            return grs[key]

        if joint:
            pols = {"always_verify": policy_always_verify, "bayesian_DP": make_joint_dp_policy(get_dp, infos, sm),
                    "bayesian_greedy": make_joint_greedy_policy(get_gr, infos, sm)}
        else:
            pols = {"always_verify": policy_always_verify, "bayesian_DP": make_sig_dp_policy(get_dp, infos, sm),
                    "bayesian_greedy": make_sig_greedy_policy(get_gr, infos, sm)}
        pols.update(rpols)
        out = run(cell, eps, pols, cost, fl, cell["pool"])
        tr = run(cell, eps, rpols, cost, fl, cell["train_ids"])
        key = (b, gg, regime)
        res["util_eval"][key] = {p: [e["utility"] for e in E] for p, E in out.items()}
        res["util_train"][key] = {p: float(np.mean([e["utility"] for e in E])) for p, E in tr.items()}
        av = float(np.mean(res["util_eval"][key]["always_verify"]))
        for p, E in out.items():
            s = summarize(E)
            res["rows"].append(dict(variant=vname, bench=b, gen=gg, regime=regime, paper_regime=(regime == paper_regime(b)),
                                    policy=p, delta_vs_av=s["mean_utility"] - av, **s))
            if p in ("bayesian_DP", "bayesian_greedy"):
                for e in E:
                    i = e["instance_id"]; inf = infos[i]
                    res["traces"].append(dict(variant=vname, cell=f"{b}/{gg}", bench=b, gen=gg, regime=regime, policy=p,
                                              instance_id=i, actions=e["actions"],
                                              beliefs=[None if x is None else round(float(x), 4) for x in e["beliefs"]],
                                              patch=e["patch_idx"], obs=e["obs"], Y_fresh=[int(r["Y"]) for r in eps[i].fresh],
                                              bT=round(inf.bT, 4), tbin=inf.tbin, pcombos=inf.pcombos,
                                              b0=[round(x, 4) for x in inf.b0], utility=e["utility"], cost=e["cost"],
                                              reward=e["reward"]))
        if informative:
            s_of = lambda i, k: infos[i].b0[min(k, len(infos[i].b0) - 1)]
            for q in QS:
                for A in rule_names:
                    for Bn in rule_names:
                        if A == Bn:
                            continue
                        pol = {"thr": make_threshold_rule(rpols[A], rpols[Bn], s_of, tau[q])}
                        ue = run(cell, eps, pol, cost, fl, cell["pool"])["thr"]
                        ut = run(cell, eps, pol, cost, fl, cell["train_ids"])["thr"]
                        res["thr_eval"].setdefault(key, {})[(q, A, Bn)] = [e["utility"] for e in ue]
                        res["thr_train"].setdefault(key, {})[(q, A, Bn)] = float(np.mean([e["utility"] for e in ut]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    os.replace(tmp, out_path)
    return vname, b, gg, "done"


if __name__ == "__main__":
    names = sys.argv[2:] or list(VARIANTS)
    jobs = [(v, b, g) for v in names for (b, g) in cell_keys()]
    with Pool(min(12, os.cpu_count() or 4)) as P:
        for v, b, gg, st in P.imap_unordered(run_cell, jobs):
            print(st, v, b, gg, flush=True)
