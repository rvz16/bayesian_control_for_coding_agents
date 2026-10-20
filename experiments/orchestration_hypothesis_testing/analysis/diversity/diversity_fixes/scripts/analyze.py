"""Aggregate runs/<variant>.pkl into summary.csv, traces.jsonl, rules_vs_bdp.csv, variant_table.csv and
examples.json (paper cost regime, 25% eval split).

usage: analyze.py RUNS_DIR OUT_DIR
Rule comparisons follow action_traces/scripts/run_all.py:97-137 (oracle rule per cell picked on eval,
best single rule over all 54 cells picked on eval, per-benchmark rule picked on train), paired bootstrap
over eval instances, 2000 resamples, seed 42. For step-1 variants the rule set also contains the
repair-using rules (engine.make_gate_repair / make_repair_k)."""
import sys
sys.dont_write_bytecode = True
import csv, json, math, os, pickle
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variants import VARIANTS

RUNS, OUT = sys.argv[1], sys.argv[2]
AT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "action_traces", "versions.json")
MATCHED = {tuple(k.split("/")): v["reproduces_paper_BG_BDP"] for k, v in json.load(open(AT))["cells"].items()}
CTRL = ("bayesian_DP", "bayesian_greedy")
mean = lambda x: float(np.mean(x)) if len(x) else float("nan")


def boot_ci(d, B=2000, seed=42):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def entropy_bits(c):
    n = sum(c.values())
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v) if n else 0.0


order = [v for v in VARIANTS if os.path.exists(os.path.join(RUNS, f"{v}.pkl"))]
D = {v: pickle.load(open(os.path.join(RUNS, f"{v}.pkl"), "rb")) for v in order}

# ---------------------------------------------------------------- summary.csv
summary = []
for v in order:
    meta = D[v]["meta"]
    for r in D[v]["rows"]:
        m = meta[(r["bench"], r["gen"])]
        row = dict(r)
        row["matched_paper_cell"] = MATCHED[(r["bench"], r["gen"])]
        row["chain_version"] = m["chain_version"]
        row["n_replayable_eval"] = m["n_replayable_eval"]
        row["replayable_cell"] = m["n_replayable_eval"] > 0
        for k in ("top5", "first_action_frac"):
            row[k] = json.dumps(row[k])
        for k, x in list(row.items()):
            if isinstance(x, float):
                row[k] = round(x, 4)
        summary.append(row)
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

# ---------------------------------------------------------------- traces.jsonl
with open(os.path.join(OUT, "traces.jsonl"), "w") as f:
    for v in order:
        for t in D[v]["traces"]:
            f.write(json.dumps(t, separators=(",", ":")) + "\n")

# ---------------------------------------------------------------- rules_vs_bdp.csv
rule_rows = []
for v in order:
    UE, UT, meta = D[v]["util_eval"], D[v]["util_train"], D[v]["meta"]
    keys = [k for k in UE if k[2] == ("slow" if k[0].startswith("swe") else "fast")]
    rules_of = {k: [p for p in UE[k] if p not in CTRL] for k in keys}
    common = [r for r in rules_of[keys[0]] if all(r in rules_of[k] for k in keys)]
    sc = {r: mean([mean(UE[k][r]) for k in keys]) for r in common}
    single = max(sc, key=sc.get)
    bench_rule = {}
    for bb in sorted({k[0] for k in keys}):
        ks = [k for k in keys if k[0] == bb]
        rs = [r for r in rules_of[ks[0]] if all(r in rules_of[k] for k in ks)]
        s2 = {r: mean([UT[k][r] for k in ks]) for r in rs}
        bench_rule[bb] = max(s2, key=s2.get)
    for k in keys:
        b, g, regime = k
        U = UE[k]
        rules = rules_of[k]
        bdp, bg = np.array(U["bayesian_DP"]), np.array(U["bayesian_greedy"])
        oracle = max(rules, key=lambda r: mean(U[r]))
        oracle_base = max([r for r in rules if not ("_rep_" in r or r.startswith("repair"))], key=lambda r: mean(U[r]))
        m = meta[(b, g)]
        row = dict(variant=v, bench=b, gen=g, regime=regime, matched_paper_cell=MATCHED[(b, g)],
                   replayable_cell=m["n_replayable_eval"] > 0, n_replayable_eval=m["n_replayable_eval"], n_eval=len(bdp),
                   U_BDP=round(mean(bdp), 3), U_BG=round(mean(bg), 3), U_always_verify=round(mean(U["always_verify"]), 3))
        for label, r in [("oracle_rule", oracle), ("oracle_nonrepair_rule", oracle_base),
                         ("single_rule_evalpicked", single), ("bench_rule_trainpicked", bench_rule[b])]:
            for cname, arr in (("BDP", bdp), ("BG", bg)):
                d = arr - np.array(U[r]); lo, hi = boot_ci(d)
                row[f"{cname}_minus_{label}"] = round(float(d.mean()), 3)
                row[f"{cname}_ci95_{label}"] = f"[{lo:+.2f},{hi:+.2f}]"
                row[f"{cname}_ci_lo_{label}"] = round(lo, 3); row[f"{cname}_ci_hi_{label}"] = round(hi, 3)
            row[label] = r
        for r in rules:
            row[f"U_{r}"] = round(mean(U[r]), 3)
        rule_rows.append(row)
keys_all = []
for r in rule_rows:
    for k in r:
        if k not in keys_all:
            keys_all.append(k)
with open(os.path.join(OUT, "rules_vs_bdp.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys_all); w.writeheader(); w.writerows(rule_rows)

# ---------------------------------------------------------------- per-variant table
def div_stats(rows, traces):
    nseq = [r["n_distinct_seq"] for r in rows]
    ent = [r["entropy_bits"] for r in rows]
    pooled = {tuple(t["actions"]) for t in traces}
    mix = Counter(a for t in traces for a in t["actions"])
    tot = sum(mix.values())
    return dict(n_cells=len(rows), nseq_median=float(np.median(nseq)), nseq_mean=round(mean(nseq), 2), nseq_max=max(nseq),
                cells_nseq1=sum(1 for x in nseq if x == 1), pooled_distinct_seq=len(pooled),
                cells_first_action_varies=sum(1 for r in rows if r["n_first_actions"] > 1),
                entropy_median=round(float(np.median(ent)), 2), entropy_max=round(max(ent), 2),
                mean_ep_len=round(mean([r["mean_n_actions"] for r in rows]), 2),
                cells_identical_to_a_rule=sum(1 for r in rows if r["identical_traces_to_rule"]),
                action_mix=json.dumps({a: round(c / tot, 3) for a, c in mix.most_common()}))


def util_stats(rr, ctrl):
    out = dict(mean_delta_vs_av=round(mean([r[f"U_{'BDP' if ctrl == 'BDP' else 'BG'}"] - r["U_always_verify"] for r in rr]), 2))
    for label in ("oracle_rule", "oracle_nonrepair_rule", "single_rule_evalpicked", "bench_rule_trainpicked"):
        d = [r[f"{ctrl}_minus_{label}"] for r in rr]
        out[f"{label}_win"] = sum(1 for r in rr if r[f"{ctrl}_minus_{label}"] > 1 and r[f"{ctrl}_ci_lo_{label}"] > 0)
        out[f"{label}_gt1"] = sum(1 for x in d if x > 1)
        out[f"{label}_tie"] = sum(1 for x in d if abs(x) <= 1)
        out[f"{label}_worse"] = sum(1 for x in d if x < -1)
        out[f"{label}_ci_rule_better"] = sum(1 for r in rr if r[f"{ctrl}_ci_hi_{label}"] < 0)
        out[f"{label}_mean"] = round(mean(d), 2)
    return out


vt = []
for v in order:
    rows_v = [r for r in D[v]["rows"] if r["paper_regime"]]
    for subset in ("all54", "matched43", "replayable", "replayable_matched"):
        def keep(b, g):
            m = D[v]["meta"][(b, g)]
            return {"all54": True, "matched43": MATCHED[(b, g)], "replayable": m["n_replayable_eval"] > 0,
                    "replayable_matched": m["n_replayable_eval"] > 0 and MATCHED[(b, g)]}[subset]
        if subset.startswith("replayable") and not VARIANTS[v].get("chains"):
            continue
        rr = [r for r in rule_rows if r["variant"] == v and keep(r["bench"], r["gen"])]
        for ctrl, pol in (("BDP", "bayesian_DP"), ("BG", "bayesian_greedy")):
            sr = [r for r in rows_v if r["policy"] == pol and keep(r["bench"], r["gen"])]
            tr = [t for t in D[v]["traces"] if t["policy"] == pol and keep(t["bench"], t["gen"])]
            if not sr:
                continue
            vt.append(dict(variant=v, subset=subset, controller=ctrl, **div_stats(sr, tr), **util_stats(rr, ctrl),
                           cells=";".join(sorted(f"{r['bench']}/{r['gen']}" for r in sr)) if subset.startswith("replayable") else ""))
with open(os.path.join(OUT, "variant_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(vt[0].keys())); w.writeheader(); w.writerows(vt)

# ---------------------------------------------------------------- examples: most diverse BDP cell per variant
ex = {}
for v in order:
    rows_v = [r for r in D[v]["rows"] if r["paper_regime"] and r["policy"] == "bayesian_DP"]
    best = max(rows_v, key=lambda r: (r["entropy_bits"], r["n_distinct_seq"]))
    cell = f"{best['bench']}/{best['gen']}"
    tr = [t for t in D[v]["traces"] if t["policy"] == "bayesian_DP" and t["cell"] == cell]
    by = defaultdict(list)
    for t in tr:
        by[tuple(t["actions"])].append(t)
    picks = []
    for seq, ts in sorted(by.items(), key=lambda kv: -len(kv[1]))[:4]:
        t = ts[0]
        picks.append(dict(n_tasks_with_this_sequence=len(ts), instance_id=t["instance_id"], actions=t["actions"],
                          obs=t["obs"], beliefs=t["beliefs"], patch=t["patch"], Y_fresh=t["Y_fresh"], Y_chain=t["Y_chain"],
                          utility=t["utility"], mean_utility_of_sequence=round(mean([x["utility"] for x in ts]), 2)))
    ex[v] = dict(cell=cell, n_distinct_seq=best["n_distinct_seq"], entropy_bits=round(best["entropy_bits"], 3),
                 prior=round(D[v]["meta"][(best["bench"], best["gen"])]["prior"], 4),
                 rep_kernel={k: (round(x, 4) if isinstance(x, float) else x) for k, x in
                             (D[v]["meta"][(best["bench"], best["gen"])]["rep"] or {}).items()
                             if k in ("P_fix", "P_break", "n_pairs", "likes_rep")},
                 examples=picks)
json.dump(ex, open(os.path.join(OUT, "examples.json"), "w"), indent=1, default=str)
print("variants", len(order), "summary rows", len(summary), "rule rows", len(rule_rows), "variant_table rows", len(vt))
