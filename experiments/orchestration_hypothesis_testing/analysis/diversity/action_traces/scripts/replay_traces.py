"""Action-trace replay of the Bayesian controllers (BG, BDP) and simple baselines.

Decision logic is imported from the paper repo (rvz16/bayesian_control_for_coding_agents,
experiments/orchestration_hypothesis_testing; unchanged at d8876f5, BG called with
reset_on_generate=False so it keeps the camera-ready behaviour):
  - analysis/controller.py : BayesianController (DP), make_bayesian_policy, CostModel,
                             policy_always_verify, policy_fixed_pipeline
  - analysis/lcb_compare.py: GreedyController, make_greedy_policy
The only patch is `simulate_policy_traced`, a line-for-line copy of
analysis/controller.py:simulate_policy (L177-230) that additionally returns the
per-step action list (already recorded in state["actions"] by the original),
the controller's belief after each step, the patch index each action was taken
on, and the critic observation. It does not alter control flow.

Protocol (reverse-engineered to match full_results.tex, see SUMMARY.md):
  * split: instance ids sorted, numpy default_rng(42).permutation, first
    int(0.75*n) = train, rest = eval
  * prior: Beta(1,1)-Laplace (k+1)/(n+2) over train patches
  * critic likelihoods: Beta(1,1) over train patches (L0_syntax, L2_public_tests, L3_llm_review)
  * kernel (BDP only): Beta(1,1) over consecutive-step (Y_t, Y_t+1) pairs of the
    Self-Refine (or Reflexion) iter corpus, 75% iter-fit cohort (same numpy split)
  * eval pool: eval ids, restricted to instances present in both SR and Rfx corpora
  * DP horizon 3, 51-point grid; 3 cached patches per instance
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

import numpy as np

# experiments/orchestration_hypothesis_testing (this file lives in analysis/diversity/action_traces/scripts)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4))
sys.path.insert(0, REPO)
from analysis.controller import (  # noqa: E402  (unchanged paper code)
    BayesianController, CostModel, make_bayesian_policy,
    policy_always_verify, policy_fixed_pipeline,
)
from analysis.lcb_compare import GreedyController, make_greedy_policy  # noqa: E402

CRITICS = {"L0": "L0_syntax", "L2": "L2_public_tests", "L3": "L3_llm_review"}

REGIMES = {
    # experiments.tex:32 "slow oracle, fast critics"
    "slow": dict(c_gen=10.0, c_L0=1.0, c_L2=2.0, c_L3=5.0, c_ver=90.0, reward=100.0),
    # experiments.tex:32 "fast oracle, balanced critics"
    "fast": dict(c_gen=10.0, c_L0=1.0, c_L2=1.0, c_L3=1.0, c_ver=5.0, reward=100.0),
}


def paper_regime(bench: str) -> str:
    # full_results.tex caption: SWE-Bench rows slow, all other rows fast
    return "slow" if bench.startswith("swe") else "fast"


# --------------------------------------------------------------------------
# Patched copy of analysis/controller.py:simulate_policy (L177-230).
# Control flow identical; adds trace outputs.
# --------------------------------------------------------------------------
def simulate_policy_traced(traj, policy_fn, cost, *, max_actions=64, prior=None):
    state = {
        "step": 0,
        "patch_idx": 0,
        "max_patches": len(traj),
        "verified": False,
        "given_up": False,
        "actions": [],
    }
    beliefs, patch_of_action, obs = [], [], []
    cum_cost = 0.0
    reward = 0.0
    while not (state["verified"] or state["given_up"]) and state["step"] < len(traj):
        if len(state["actions"]) >= max_actions:
            state["given_up"] = True
            state["actions"].append("give_up_cap")
            beliefs.append(state.get("belief", prior)); patch_of_action.append(state["patch_idx"]); obs.append(None)
            break
        rec = traj[state["patch_idx"]]
        pidx = state["patch_idx"]
        action = policy_fn(state, rec)
        state["actions"].append(action)
        if action == "verify":
            cum_cost += cost.c_ver
            reward = cost.reward * rec["Y"]
            state["verified"] = True
        elif action == "give_up":
            state["given_up"] = True
        elif action == "generate":
            cum_cost += cost.c_gen
            state["patch_idx"] += 1
            state["step"] += 1
            if state["patch_idx"] >= len(traj):
                state["given_up"] = True
        elif action == "L0":
            cum_cost += cost.c_L0
        elif action == "L2":
            cum_cost += cost.c_L2
        elif action == "L3":
            cum_cost += cost.c_L3
        # --- trace additions (read-only) ---
        beliefs.append(state.get("belief", prior))
        patch_of_action.append(pidx)
        if action in CRITICS:
            obs.append(bool(rec.get(CRITICS[action])))
        elif action == "verify":
            obs.append(int(rec["Y"]))
        else:
            obs.append(None)
    return {"reward": reward, "cost": cum_cost,
            "utility": reward - cum_cost,
            "verified": state["verified"],
            "n_actions": len(state["actions"]),
            # trace outputs
            "actions": list(state["actions"]),
            "beliefs": beliefs,
            "patch_idx": patch_of_action,
            "obs": obs,
            # the episode ended because generate ran past the last cached patch
            "exhausted": (not state["verified"]) and bool(state["actions"]) and state["actions"][-1] == "generate"}


# --------------------------------------------------------------------------
# Simple rule baselines (stateless), written in the same policy_fn interface.
# gate(Cr) semantics follow appendix_setup.tex:220-231 (PASS->verify, FAIL->regen),
# with the last cached patch falling back to give_up as in controller.py:policy_threshold_*.
# --------------------------------------------------------------------------
def policy_never_submit(state, rec):
    return "give_up"


def make_gate(critic: str, on_fail: str = "regen"):
    field = CRITICS[critic]
    key = f"_gate_{critic}_done"

    def _p(state, rec):
        if not state.get(key):
            state[key] = True
            return critic
        state[key] = False
        if rec.get(field):
            return "verify"
        if on_fail == "give_up":
            return "give_up"
        if state["patch_idx"] + 1 < state.get("max_patches", 3):
            return "generate"
        return "give_up"
    return _p


def make_fixed_pipeline(critics):
    """Run critics in order, verify iff all PASS, else regen (paper FP)."""
    def _p(state, rec):
        for c in critics:
            if not state.get(f"_fp_{c}"):
                state[f"_fp_{c}"] = True
                return c
        for c in critics:
            state[f"_fp_{c}"] = False
        if all(rec.get(CRITICS[c]) for c in critics):
            return "verify"
        if state["patch_idx"] + 1 < state.get("max_patches", 3):
            return "generate"
        return "give_up"
    return _p


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# One cell
# --------------------------------------------------------------------------
def build_cell(cal_recs, sr_recs, rx_recs, bench, *, kernel_from="selfrefine",
               cal_rounding="int", iter_rounding="round", restrict_pool=True):
    by = {}
    for r in cal_recs:
        by.setdefault(str(r["instance_id"]), []).append(r)
    for k in by:
        by[k].sort(key=lambda r: int(r.get("patch_id", 0)))
    train_ids, eval_ids = npg_split(list(by), rounding=cal_rounding)
    train_recs = [r for i in train_ids for r in by[i]]
    fields = ["L0_syntax", "L3_llm_review"] if bench.startswith("swe") else ["L0_syntax", "L2_public_tests", "L3_llm_review"]
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


def make_policies(cell, cost, bench, horizon=3, include_rules=True):
    prior, likes, kernel = cell["prior"], cell["likes"], cell["kernel"]
    dp = BayesianController(prior, likes, {"kernel_all": kernel}, cost, horizon=horizon, n_belief=51)
    gr = GreedyController(prior, likes, cost)
    pols = {
        "always_verify": policy_always_verify,
        # reset_on_generate=False: the camera-ready BG keeps the post-FAIL belief across generate
        # (lcb_compare.make_greedy_policy before the fix); this replay reproduces the paper's BG.
        "bayesian_greedy": make_greedy_policy(gr, reset_on_generate=False),
        "bayesian_DP": make_bayesian_policy(dp),
    }
    if include_rules:
        pols["never_submit"] = policy_never_submit
        pols["gate_syn"] = make_gate("L0", "regen")
        pols["gate_llm"] = make_gate("L3", "regen")
        if not bench.startswith("swe"):
            pols["gate_test"] = make_gate("L2", "regen")
            pols["gate_test_giveup"] = make_gate("L2", "give_up")
        # paper FP = analysis/controller.py:policy_fixed_pipeline (L0 -> L3, AND gate), unchanged
        pols["fixed_pipeline"] = policy_fixed_pipeline
    return pols, dp


def run_cell(cell, bench, regime, include_rules=True, ids=None):
    cost = CostModel(**REGIMES[regime])
    pols, dp = make_policies(cell, cost, bench, include_rules=include_rules)
    out = {}
    ids = cell["pool"] if ids is None else ids
    for name, fn in pols.items():
        out[name] = []
        for i in ids:
            out[name].append(dict(instance_id=i, **simulate_policy_traced(cell["by"][i], fn, cost, prior=cell["prior"])))
    return out


def seq_key(actions):
    return ">".join(actions)


def entropy_bits(counter):
    n = sum(counter.values())
    return -sum((c / n) * math.log2(c / n) for c in counter.values() if c)


def summarize(episodes):
    seqs = Counter(seq_key(e["actions"]) for e in episodes)
    n = len(episodes)
    first = Counter(e["actions"][0] if e["actions"] else "" for e in episodes)
    top = seqs.most_common(5)
    def mean(f):
        return sum(f(e) for e in episodes) / n if n else float("nan")
    return {
        "n_tasks": n,
        "n_distinct_seq": len(seqs),
        "entropy_bits": entropy_bits(seqs) if n else 0.0,
        "top1_share": top[0][1] / n if n else float("nan"),
        "top5": [(s, c) for s, c in top],
        "first_action_frac": {k: v / n for k, v in first.items()},
        "mean_utility": mean(lambda e: e["utility"]),
        "mean_n_actions": mean(lambda e: len(e["actions"])),
        "mean_L0": mean(lambda e: e["actions"].count("L0")),
        "mean_L2": mean(lambda e: e["actions"].count("L2")),
        "mean_L3": mean(lambda e: e["actions"].count("L3")),
        "mean_verify": mean(lambda e: e["actions"].count("verify")),
        "mean_regen": mean(lambda e: e["actions"].count("generate")),
        "frac_give_up": mean(lambda e: 1.0 if (e["actions"] and e["actions"][-1] in ("give_up", "give_up_cap")) else 0.0),
        "frac_exhausted": mean(lambda e: 1.0 if e["exhausted"] else 0.0),
        "frac_cap": mean(lambda e: 1.0 if "give_up_cap" in e["actions"] else 0.0),
        "frac_repeat_critic": mean(lambda e: 1.0 if _repeat_critic(e) else 0.0),
        "pass_rate": mean(lambda e: 1.0 if e["reward"] > 0 else 0.0),
    }


def _repeat_critic(e):
    seen = set()
    for a, p in zip(e["actions"], e["patch_idx"]):
        if a in CRITICS:
            if (a, p) in seen:
                return True
            seen.add((a, p))
    return False
