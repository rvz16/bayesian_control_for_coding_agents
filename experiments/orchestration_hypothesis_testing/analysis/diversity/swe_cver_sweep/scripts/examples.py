"""examples.json: BDP traces (calK_jointcomb) in chosen SWE cells at a chosen C_ver, one per distinct first action.
Per first action the trace shown is the MOST COMMON action sequence among tasks with that first action (ties: lowest
instance id) - not picked by outcome. Each example carries the task's free signals, the starting beliefs, the actions
with observations and the belief after each step, the utility, and the same task's utility under rule (ii), rule
(iii) and the perfect-information oracle.
usage: examples.py SWEEP_DIR"""
import csv
import json
import os
import pickle
import sys
from collections import Counter

OUT = sys.argv[1]
PICKS = [("swe_verified", "gpt_oss_20b", "cv20_s"), ("swe_lite", "qwen25_32b", "cv15_s"), ("swe_verified", "qwen25_32b", "cv20_s")]
V = "calK_jointcomb"

sweep = {(r["variant"], r["regime"], r["bench"], r["gen"]): r for r in csv.DictReader(open(os.path.join(OUT, "sweep.csv")))}
vecs = pickle.load(open(os.path.join(OUT, "logs", "vecs.pkl"), "rb"))
sig = {}
with open(os.path.join(OUT, "..", "per_task_prior", "signals.csv")) as f:
    for r in csv.DictReader(f):
        sig[(r["bench"], r["gen"], r["instance_id"], int(r["patch_pos"]))] = r

out = []
for b, g, reg in PICKS:
    R = pickle.load(open(os.path.join(OUT, "runs", reg, V, f"{b}__{g}.pkl"), "rb"))
    S = sweep[(V, reg, b, g)]
    vv = vecs[(V, reg, b, g)]
    pos = {i: k for k, i in enumerate(vv["ids"])}
    tr = [t for t in R["traces"] if t["policy"] == "bayesian_DP"]
    firsts = Counter(t["actions"][0] for t in tr)
    ex = []
    for a, n in firsts.most_common():
        ts = [t for t in tr if t["actions"][0] == a]
        seqc = Counter(tuple(t["actions"]) for t in ts)
        top = seqc.most_common(1)[0][0]
        t = sorted([t for t in ts if tuple(t["actions"]) == top], key=lambda t: t["instance_id"])[0]
        i = t["instance_id"]; k = pos[i]
        s0 = sig.get((b, g, i, 0), {})
        steps = []
        prev = t["b0"][0]
        for act, o, bel, p in zip(t["actions"], t["obs"], t["beliefs"], t["patch"]):
            steps.append(dict(action=act, patch=p, observation=o, belief_before=prev, belief_after=bel))
            prev = bel
        ex.append(dict(first_action=a, n_tasks_with_this_first_action=n,
                       mean_BDP_utility_these_tasks=round(sum(x["utility"] for x in ts) / n, 2),
                       mean_rule_ii_utility_these_tasks=round(sum(vv["ii"][pos[x["instance_id"]]] for x in ts) / n, 2),
                       sequence_share_within_first_action=f"{seqc[top]}/{n}", instance_id=i,
                       free_signals=dict(combtask_score=s0.get("combtask"), task_bin=t["tbin"],
                                         diff_chars_per_patch=[sig.get((b, g, i, j), {}).get("diff_chars") for j in range(len(t["Y_fresh"]))],
                                         swe_difficulty=s0.get("t_swe_difficulty") or None, n_code_blocks=s0.get("t_n_code_blocks")),
                       belief_task_bT=t["bT"], belief_start_per_patch_b0=t["b0"], steps=steps, Y_per_patch=t["Y_fresh"],
                       utility_BDP=t["utility"], utility_rule_ii=float(vv["ii"][k]), utility_rule_iii=float(vv["iii"][k]),
                       utility_perfect_oracle=float(vv["perfect"][k])))
    out.append(dict(cell=f"{b}/{g}", regime=reg, c_ver=float(S["c_ver"]), critic_costs="L0 1, L3 5 (no L2 on SWE)",
                    c_gen=10, reward=100, prior=R["prior"], n_eval=int(S["n_eval"]), rule_ii=S["rule_ii"], rule_iii=S["rule_iii"],
                    U_BDP=float(S["U_BDP"]), U_rule_ii=float(S["U_rule_ii"]), U_rule_iii=float(S["U_rule_iii"]),
                    U_perfect=float(S["U_perfect"]),
                    BDP_minus_ii=f"{S['BDP_minus_ii']} [{S['BDP_ci_lo_ii']}, {S['BDP_ci_hi_ii']}]",
                    BDP_minus_iii=f"{S['BDP_minus_iii']} [{S['BDP_ci_lo_iii']}, {S['BDP_ci_hi_iii']}]",
                    BDP_minus_BG=f"{S['BDP_minus_BG']} [{S['BDP_ci_lo_BG']}, {S['BDP_ci_hi_BG']}]",
                    BDP_n_distinct_seq=int(S["BDP_n_distinct_seq"]), first_action_counts=dict(firsts), examples=ex))
json.dump(out, open(os.path.join(OUT, "examples.json"), "w"), indent=1, default=str)
for c in out:
    print(c["cell"], c["regime"], c["first_action_counts"], c["BDP_minus_ii"], c["BDP_minus_iii"])
    for e in c["examples"]:
        print("  ", e["instance_id"], e["first_action"], e["n_tasks_with_this_first_action"], "bT", e["belief_task_bT"], "b0", e["belief_start_per_patch_b0"],
              " > ".join(f"{s['action']}{'' if s['observation'] is None else ':' + str(s['observation'])}({s['belief_after']})" for s in e["steps"]),
              "Y", e["Y_per_patch"], "U", e["utility_BDP"], "ii", e["utility_rule_ii"], "perf", e["utility_perfect_oracle"],
              "meanU", e["mean_BDP_utility_these_tasks"], "mean_ii", e["mean_rule_ii_utility_these_tasks"], e["free_signals"])
