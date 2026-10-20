"""Hard regression gate: with every flag off, this engine must reproduce action_traces/traces.jsonl
(actions, beliefs (4 dp), patch_idx, obs, utility, cost, reward) for every policy, cell, regime and
instance, and action_traces/summary.csv delta_vs_av for BG/BDP. Exit code 1 on any mismatch."""
import sys
sys.dont_write_bytecode = True
import csv, json, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cells import cell_keys
from runner import make_policies, prepare, run
from variants import VARIANTS

# reference = action_traces/run_all.py output (traces.jsonl is not committed); ACTION_TRACES_DIR overrides
AT = os.environ.get("ACTION_TRACES_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "action_traces")
if not os.path.exists(os.path.join(AT, "traces.jsonl")):
    sys.exit(f"missing {os.path.join(AT, 'traces.jsonl')}: run action_traces/scripts/run_all.py first "
             "(or point ACTION_TRACES_DIR at its output dir)")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "regression_gate.json")

ref = {}
for line in open(os.path.join(AT, "traces.jsonl")):
    r = json.loads(line)
    ref[(r["cell"], r["regime"], r["policy"], r["instance_id"])] = r
ref_sum = {}
for r in csv.DictReader(open(os.path.join(AT, "summary.csv"))):
    ref_sum[(f"{r['bench']}/{r['gen']}", r["regime"], r["policy"])] = float(r["delta_vs_av"])

V = VARIANTS["baseline"]
n_cmp, bad, seen = 0, [], set()
delta_bad = []
for (b, g) in cell_keys():
    gg = g.replace("_local", "")
    C, cell, cd, eps = prepare(b, g, V)
    for regime in ("fast", "slow"):
        pols, cost, fl = make_policies(cell, cd, b, regime, V)
        out = run(cell, eps, pols, cost, fl, cell["pool"])
        av = sum(e["utility"] for e in out["always_verify"]) / len(out["always_verify"])
        for p, E in out.items():
            for e in E:
                k = (f"{b}/{gg}", regime, p, e["instance_id"])
                seen.add(k)
                r = ref.get(k)
                n_cmp += 1
                mine = dict(actions=e["actions"], beliefs=[None if x is None else round(float(x), 4) for x in e["beliefs"]],
                            patch_idx=e["patch_idx"], obs=e["obs"], utility=e["utility"], cost=e["cost"], reward=e["reward"])
                if r is None or any(mine[f] != r[f] for f in mine):
                    bad.append(dict(key=list(k), mine=mine, ref=None if r is None else {f: r[f] for f in mine}))
            if p in ("bayesian_greedy", "bayesian_DP"):
                d = round(sum(e["utility"] for e in E) / len(E) - av, 3)
                rd = ref_sum.get((f"{b}/{gg}", regime, p))
                if rd is None or abs(d - rd) > 1e-9:
                    delta_bad.append([b, gg, regime, p, d, rd])
missing = [list(k) for k in ref if k not in seen]
res = dict(n_episodes_compared=n_cmp, n_mismatch=len(bad), n_reference_episodes=len(ref), n_reference_not_reproduced=len(missing),
           n_cell_delta_mismatch=len(delta_bad), first_mismatches=bad[:5], missing_examples=missing[:5], delta_mismatch=delta_bad[:10],
           PASS=(not bad and not missing and not delta_bad))
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, indent=1))
if bad:
    print(json.dumps(bad[:3], indent=1))
sys.exit(0 if res["PASS"] else 1)
