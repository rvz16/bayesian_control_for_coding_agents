"""Run variants over all 54 cells x {fast, slow}. One pickle per variant in WORK.
usage: run_variants.py WORK_DIR [variant ...]   (default: all variants, in parallel)"""
import sys
sys.dont_write_bytecode = True
import math, os, pickle
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import cell_keys
from protocol import paper_regime
from runner import make_policies, prepare, run
from variants import VARIANTS

WORK = sys.argv[1]
os.makedirs(WORK, exist_ok=True)
RULE_CMP_IGNORE = {"bayesian_greedy", "bayesian_DP"}


def entropy_bits(c):
    n = sum(c.values())
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v) if n else 0.0


def summarize(E):
    n = len(E)
    seqs = Counter(">".join(e["actions"]) for e in E)
    seqs_o = Counter(">".join(f"{a}:{o}" for a, o in zip(e["actions"], e["obs"])) for e in E)
    first = Counter(e["actions"][0] if e["actions"] else "" for e in E)
    m = lambda f: sum(f(e) for e in E) / n if n else float("nan")
    cnt = lambda a: (lambda e: e["actions"].count(a))
    return dict(
        n_tasks=n, n_distinct_seq=len(seqs), n_distinct_seq_with_obs=len(seqs_o), entropy_bits=entropy_bits(seqs),
        top1_share=seqs.most_common(1)[0][1] / n, top5=seqs.most_common(5),
        first_action_frac={k: v / n for k, v in first.items()}, n_first_actions=len(first),
        mean_utility=m(lambda e: e["utility"]), mean_n_actions=m(lambda e: len(e["actions"])),
        mean_L0=m(cnt("L0")), mean_L1=m(cnt("L1")), mean_L2=m(cnt("L2")), mean_L3=m(cnt("L3")),
        mean_repair=m(cnt("repair")), mean_regen=m(cnt("generate")), mean_verify=m(cnt("verify")),
        frac_give_up=m(lambda e: 1.0 if (e["actions"] and e["actions"][-1] in ("give_up", "give_up_cap")) else 0.0),
        frac_exhausted=m(lambda e: 1.0 if e["exhausted"] else 0.0),
        frac_cap=m(lambda e: 1.0 if "give_up_cap" in e["actions"] else 0.0),
        frac_any_repair=m(lambda e: 1.0 if "repair" in e["actions"] else 0.0),
        pass_rate=m(lambda e: 1.0 if e["reward"] > 0 else 0.0),
        mean_cost=m(lambda e: e["cost"]),
    )


def one_variant(vname):
    V = VARIANTS[vname]
    rows, util_eval, util_train, traces, meta = [], {}, {}, [], {}
    for (b, g) in cell_keys():
        gg = g.replace("_local", "")
        C, cell, cd, eps = prepare(b, g, V)
        root_avail_eval = sum(1 for i in cell["pool"] if eps[i].chain_len(0) >= 2) / len(cell["pool"])
        meta[(b, gg)] = dict(
            n_eval=len(cell["pool"]), prior=cell["prior"],
            chain_version=cd["version"] if cd else None,
            n_rooted=len(cd["rooted"]) if cd else 0,
            n_replayable_eval=sum(1 for i in cell["pool"] if eps[i].chain_len(0) >= 2),
            frac_eval_repairable_root=root_avail_eval,
            rep={k: v for k, v in cd["rep"].items()} if cd and cd["rep"] else None)
        for regime in ("fast", "slow"):
            pols, cost, fl = make_policies(cell, cd, b, regime, V)
            out = run(cell, eps, pols, cost, fl, cell["pool"])
            rpols, _, _ = make_policies(cell, cd, b, regime, V, only_rules=True)
            tr = run(cell, eps, rpols, cost, fl, cell["train_ids"])
            util_train[(b, gg, regime)] = {p: sum(e["utility"] for e in E) / len(E) for p, E in tr.items()}
            util_eval[(b, gg, regime)] = {p: [e["utility"] for e in E] for p, E in out.items()}
            av = sum(util_eval[(b, gg, regime)]["always_verify"]) / len(cell["pool"])
            seqmap = {p: [tuple(e["actions"]) for e in E] for p, E in out.items()}
            for p, E in out.items():
                s = summarize(E)
                same_rule = ""
                if p in ("bayesian_DP", "bayesian_greedy"):
                    same_rule = ";".join(r for r in out if r not in RULE_CMP_IGNORE and seqmap[r] == seqmap[p])
                rows.append(dict(variant=vname, bench=b, gen=gg, regime=regime, paper_regime=(regime == paper_regime(b)),
                                 policy=p, delta_vs_av=s["mean_utility"] - av, identical_traces_to_rule=same_rule,
                                 frac_eval_repairable_root=root_avail_eval, **s))
                if regime == paper_regime(b) and p in ("bayesian_DP", "bayesian_greedy"):
                    for e in E:
                        ep = eps[e["instance_id"]]
                        traces.append(dict(variant=vname, cell=f"{b}/{gg}", bench=b, gen=gg, regime=regime, policy=p,
                                           instance_id=e["instance_id"], actions=e["actions"],
                                           beliefs=[None if x is None else round(float(x), 4) for x in e["beliefs"]],
                                           patch=e["patch_idx"], obs=e["obs"],
                                           Y_fresh=[int(r["Y"]) for r in ep.fresh],
                                           Y_chain=[int(r["Y"]) for r in ep.chain] if ep.chain else None,
                                           utility=e["utility"], cost=e["cost"], reward=e["reward"]))
    tmp = os.path.join(WORK, f".{vname}.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dict(rows=rows, util_eval=util_eval, util_train=util_train, traces=traces, meta=meta), f)
    os.replace(tmp, os.path.join(WORK, f"{vname}.pkl"))
    return vname, len(rows)


if __name__ == "__main__":
    names = sys.argv[2:] or list(VARIANTS)
    force = os.environ.get("FORCE") == "1"
    skip = [v for v in names if os.path.exists(os.path.join(WORK, f"{v}.pkl")) and not force]
    for v in skip:
        print("skip (exists)", v, flush=True)
    names = [v for v in names if v not in skip]
    if not names:
        sys.exit(0)
    with Pool(min(len(names), os.cpu_count() or 4)) as P:
        for v, n in P.imap_unordered(one_variant, names):
            print("done", v, n, flush=True)
