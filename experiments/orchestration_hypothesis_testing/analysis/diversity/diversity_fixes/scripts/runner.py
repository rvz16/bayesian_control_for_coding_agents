"""Cell preparation + policy construction + replay for one (cell, regime, variant)."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from functools import lru_cache

from cells import load_cell
from chains import clean_chains, fit_repair_kernel, ident, root_chains, select_version
from data import load_records
from engine import (DP, Cost, Episode, Flags, Greedy, make_dp_policy, make_gate, make_gate_repair,
                    make_greedy_policy, make_repair_k, policy_always_verify, policy_fixed_pipeline,
                    policy_never_submit, simulate)
from protocol import REGIMES, build_cell

BASE_RULES = ["always_verify", "never_submit", "gate_syn", "gate_test", "gate_test_giveup", "gate_llm", "fixed_pipeline"]
CRIT_OF = {"syn": "L0", "test": "L2", "llm": "L3"}
REPAIR_RULES = ([f"gate_{c}_rep_{e}" for c in ("syn", "test", "llm") for e in ("verify", "giveup", "regen")]
                + [f"repair{k}_verify" for k in (1, 2, 3, 4)])


@lru_cache(maxsize=None)
def _cell(bench, gen, l1):
    C = load_cell(bench, gen)
    cell = build_cell(C["cal"], C["sr"], C["rx"], bench, kernel_from="selfrefine", cal_rounding="int",
                      iter_rounding="round", restrict_pool=False, extra_fields=("L1_lint",) if l1 else ())
    return C, cell


def cal_kernel(cell):
    """P(Y_k+1 | Y_k) over consecutive calibration patches of train instances, Beta(1,1)."""
    c = {"00": 0, "01": 0, "10": 0, "11": 0}
    for i in cell["train_ids"]:
        rs = cell["by"][i]
        for x, y in zip(rs, rs[1:]):
            c[f"{int(x['Y'])}{int(y['Y'])}"] += 1
    return {"P_fix": (c["01"] + 1) / (c["00"] + c["01"] + 2), "P_break": (c["10"] + 1) / (c["10"] + c["11"] + 2), "counts": c}


@lru_cache(maxsize=None)
def chain_data(bench, gen, method, mode):
    C, cell = _cell(bench, gen, False)
    ver, d = select_version(bench, gen, method, cell["by"], mode)
    if d is None:
        return {"version": None, "rooted": {}, "clean_stats": {}, "root_stats": {}, "n_chains": 0, "rep": None}
    chains, st = clean_chains(load_records(d), bench)
    rooted, rst = root_chains(chains, cell["by"])
    rep = fit_repair_kernel(rooted, cell["train_ids"])
    return {"version": f"{bench}_{gen}_{method}_records:{ver}", "rooted": rooted, "clean_stats": dict(st),
            "root_stats": dict(rst), "n_chains": len(chains), "rep": rep, "chains": chains}


def prepare(bench, gen, V):
    C, cell = _cell(bench, gen, bool(V.get("L1")))
    cd = chain_data(bench, gen, *V["chains"]) if V.get("chains") else None
    rooted = cd["rooted"] if cd else {}
    eps = {i: Episode(i, cell["by"][i], chain=rooted.get(i), ident=ident) for i in cell["by"]}
    return C, cell, cd, eps


def make_policies(cell, cd, bench, regime, V, include_rules=True, only_rules=False):
    rg = dict(REGIMES[regime])
    cost = Cost(**rg, c_rep=V.get("c_rep", 10.0))
    fl = Flags(**V["flags"])
    rep = cd["rep"] if cd else None
    critics = None if V.get("L1") else ["L0", "L2", "L3"]
    pols = {}
    if not only_rules:
        pols["always_verify"] = policy_always_verify
        ck = cal_kernel(cell)
        gr = Greedy(cell["prior"], cell["likes"]["critic_likelihoods"], cost, fl, rep=rep, critics=critics, cal_kernel=ck)
        dp = DP(cell["prior"], cell["likes"]["critic_likelihoods"], cell["kernel"], cost, fl, rep=rep, critics=critics, cal_kernel=ck)
        pols["bayesian_greedy"] = make_greedy_policy(gr)
        pols["bayesian_DP"] = make_dp_policy(dp)
    if include_rules or only_rules:
        swe = bench.startswith("swe")
        pols["always_verify"] = policy_always_verify
        pols["never_submit"] = policy_never_submit
        pols["gate_syn"] = make_gate("L0", "regen")
        pols["gate_llm"] = make_gate("L3", "regen")
        if not swe:
            pols["gate_test"] = make_gate("L2", "regen")
            pols["gate_test_giveup"] = make_gate("L2", "give_up")
        pols["fixed_pipeline"] = policy_fixed_pipeline
        if fl.repair:
            for c in ("syn", "test", "llm"):
                if swe and c == "test":
                    continue
                for e in ("verify", "giveup", "regen"):
                    pols[f"gate_{c}_rep_{e}"] = make_gate_repair(CRIT_OF[c], {"giveup": "give_up"}.get(e, e))
            for k in (1, 2, 3, 4):
                pols[f"repair{k}_verify"] = make_repair_k(k)
    return pols, cost, fl


def run(cell, eps, pols, cost, fl, ids):
    out = {}
    for name, fn in pols.items():
        out[name] = [dict(instance_id=i, **simulate(eps[i], fn, cost, fl, prior=cell["prior"])) for i in ids]
    return out
