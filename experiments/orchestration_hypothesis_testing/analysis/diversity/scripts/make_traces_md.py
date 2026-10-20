#!/usr/bin/env python3
"""Build AGENT_TRACES.md: example Bayes-DP traces per benchmark with task descriptions,
plus the distribution / diversity of action sequences.

Sources (all produced by the replay analyses in this folder):
  A  action_traces/traces.jsonl              paper controller, paper cost regime
  B  per_task_prior/traces.jsonl             calK_jointcomb (per-task signals), paper cost regime
  C  swe_cver_sweep/runs/cv20_s/calK_jointcomb/*.pkl   SWE at C_ver = 20 (critics syn 1 / llm 5)
"""
import collections, glob, json, math, pickle, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "AGENT_TRACES.md"   # usage: make_traces_md.py [OUT.md]
_INPUTS = {"action_traces/traces.jsonl": "action_traces/scripts/run_all.py action_traces",
           "per_task_prior/traces.jsonl": "per_task_prior/scripts/run_ptp.py runs + analyze_ptp.py runs .",
           "swe_cver_sweep/runs/cv20_s/calK_jointcomb": "swe_cver_sweep/scripts/sweep.py runs all"}
_missing = [f"  {k}  <- {v}" for k, v in _INPUTS.items() if not (ROOT / k).exists()]
if _missing:
    sys.exit("make_traces_md.py: regenerable inputs are missing (not committed; see README.md):\n" + "\n".join(_missing))

BENCH_NAME = {
    "swe_lite": "SWE-bench Lite", "swe_verified": "SWE-bench Verified", "lcb_hard": "LiveCodeBench hard",
    "lcb_medium": "LiveCodeBench medium", "lcb_easy": "LiveCodeBench easy", "codecontests": "CodeContests",
    "humanevalfix": "HumanEvalFix", "humaneval": "HumanEval+", "mbpp": "MBPP+",
}
ORDER = ["swe_lite", "swe_verified", "lcb_hard", "lcb_medium", "lcb_easy", "codecontests", "humanevalfix",
         "humaneval", "mbpp"]
GEN_NAME = {"gpt5_mini": "gpt-5-mini", "haiku45": "haiku-4.5", "sonnet45": "sonnet-4.5",
            "qwen3_coder": "qwen3-coder", "qwen25_32b": "qwen2.5-32B", "gpt_oss_20b": "gpt-oss-20b"}
ACT = {"L0": "syntax check", "L1": "lint", "L2": "public tests", "L3": "LLM judge",
       "generate": "regenerate", "verify": "verify", "give_up": "give up", "repair": "repair"}
CRITICS = ("L0", "L1", "L2", "L3")


def tok(a, o=None):
    if a in CRITICS:
        return f"{a}{'✓' if o else '✗'}" if o is not None else a
    if a == "verify":
        return "verify" if o is None else f"verify(Y={int(o)})"
    return {"generate": "regen", "give_up": "give_up"}.get(a, a)


def seq_str(actions):
    return " → ".join(tok(a) for a in actions)


def entropy(counter):
    n = sum(counter.values())
    return abs(-sum(c / n * math.log2(c / n) for c in counter.values())) if n else 0.0


# ---------------------------------------------------------------- load traces
def load_jsonl(path, keep):
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if keep(d):
                out.append(d)
    return out


A = load_jsonl(ROOT / "action_traces/traces.jsonl",
               lambda d: d["policy"] == "bayesian_DP" and d.get("paper_regime", True))
B = load_jsonl(ROOT / "per_task_prior/traces.jsonl",
               lambda d: d["variant"] == "calK_jointcomb" and d["policy"] == "bayesian_DP"
               and ((d["bench"].startswith("swe") and d["regime"] == "slow")
                    or (not d["bench"].startswith("swe") and d["regime"] == "fast")))
C = []
for f in sorted(glob.glob(str(ROOT / "swe_cver_sweep/runs/cv20_s/calK_jointcomb/*.pkl"))):
    d = pickle.load(open(f, "rb"))
    C += [t for t in d["traces"] if t["policy"] == "bayesian_DP"]
for t in A:
    t["Y_fresh"] = t.get("Y_patches")
    t["patch"] = t.get("patch_idx")

# ---------------------------------------------------------------- task descriptions
def clean(s, n=320):
    s = re.sub(r"<!--.*?-->", " ", s or "", flags=re.S)
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[: n - 1].rsplit(" ", 1)[0] + " …"


DESC = {}


def load_desc():
    try:
        import datasets
        datasets.disable_progress_bars()
    except Exception as e:  # pragma: no cover
        print("datasets unavailable:", e, file=sys.stderr)
        return
    def ds(*a, **k):
        try:
            return datasets.load_dataset(*a, **k)
        except Exception as e:
            print("skip", a, e, file=sys.stderr)
            return None
    for name, b in (("princeton-nlp/SWE-bench_Lite", "swe_lite"), ("princeton-nlp/SWE-bench_Verified", "swe_verified")):
        d = ds(name, split="test")
        for r in d or []:
            DESC[(b, r["instance_id"])] = (r["instance_id"], clean(r["problem_statement"]))
    for line in open(ROOT / "per_task_prior/cache/lcb_meta.jsonl"):
        r = json.loads(line)
        for b in ("lcb_hard", "lcb_medium", "lcb_easy"):
            DESC[(b, str(r["question_id"]))] = (f"{r['question_title']} ({r['platform']}, {r['difficulty']})",
                                                clean(r["question_content"]))
    # CodeContests test descriptions (name -> description), fetched once from the HF test parquet
    for name, text in json.load(open(ROOT / "cache/cc_test_desc.json")).items():
        DESC[("codecontests", name)] = (name, clean(text))
    d = ds("bigcode/humanevalpack", "python", split="test")
    for r in d or []:
        DESC[("humanevalfix", r["task_id"])] = (f"{r['task_id']} ({r.get('bug_type', '')} bug)",
                                                clean(r["docstring"] or r["prompt"]))
        DESC[("humaneval", r["task_id"].replace("Python/", "HumanEval/"))] = (r["task_id"].replace("Python/", "HumanEval/"),
                                                                             clean(r["docstring"] or r["prompt"]))
    d = ds("evalplus/mbppplus", split="test")
    for r in d or []:
        for k in (str(r["task_id"]), f"Mbpp/{r['task_id']}"):
            DESC[("mbpp", k)] = (f"Mbpp/{r['task_id']}", clean(r["prompt"]))


load_desc()


def describe(bench, iid):
    d = DESC.get((bench, str(iid)))
    if d is None and bench == "humaneval":
        d = DESC.get((bench, f"HumanEval/{iid}"))
    return d or (str(iid), "(description not found in local dataset cache)")


# ---------------------------------------------------------------- statistics
def stats(traces, bench):
    ts = [t for t in traces if t["bench"] == bench]
    if not ts:
        return None
    seqs = collections.Counter(tuple(t["actions"]) for t in ts)
    cells = collections.defaultdict(list)
    for t in ts:
        cells[t["cell"]].append(t)
    first_varies = sum(1 for v in cells.values() if len({t["actions"][0] for t in v}) > 1)
    per_cell = [len({tuple(t["actions"]) for t in v}) for v in cells.values()]
    mix = collections.Counter(a for t in ts for a in t["actions"])
    n_act = sum(mix.values())
    first = collections.Counter(t["actions"][0] for t in ts)
    return dict(n=len(ts), cells=len(cells), distinct=len(seqs), per_cell_mean=sum(per_cell) / len(per_cell),
                per_cell_max=max(per_cell), first_varies=first_varies, H=entropy(seqs), seqs=seqs, mix=mix,
                n_act=n_act, first=first, len=n_act / len(ts), U=sum(t["utility"] for t in ts) / len(ts),
                give_up=sum(t["actions"][-1] == "give_up" for t in ts) / len(ts),
                verify=sum("verify" in t["actions"] for t in ts) / len(ts))


def pct(x):
    return f"{100 * x:.0f}%"


def mix_str(s):
    order = ["verify", "give_up", "L0", "L1", "L2", "L3", "generate", "repair"]
    return ", ".join(f"{ACT[a]} {pct(s['mix'][a] / s['n_act'])}" for a in order if s["mix"].get(a))


# ---------------------------------------------------------------- examples
def pick_examples(traces, bench, k=4):
    """See below; afterwards also append the most common multi-step trace (>= 3 actions) if none was picked."""
    out = _pick(traces, bench, k)
    if all(len(seq) < 3 for seq, *_ in out):
        ts = [t for t in traces if t["bench"] == bench]
        long = collections.Counter(tuple(t["actions"]) for t in ts if len(t["actions"]) >= 3)
        if long:
            seq = long.most_common(1)[0][0]
            g = sorted((t for t in ts if tuple(t["actions"]) == seq), key=lambda t: str(t["instance_id"]))
            out.append((seq, long[seq], len(ts), g[0]))
    return out


def _pick(traces, bench, k=4):
    """One task per distinct sequence, most common sequences first; prefer different first actions."""
    ts = [t for t in traces if t["bench"] == bench]
    by_seq = collections.defaultdict(list)
    for t in ts:
        by_seq[tuple(t["actions"])].append(t)
    ranked = sorted(by_seq.items(), key=lambda kv: -len(kv[1]))
    chosen, firsts = [], set()
    for seq, group in ranked:                     # pass 1: new first action
        if seq[0] not in firsts:
            chosen.append((seq, group)); firsts.add(seq[0])
        if len(chosen) == k:
            break
    for seq, group in ranked:                     # pass 2: fill with longer traces
        if len(chosen) == k:
            break
        if all(seq != c[0] for c in chosen) and len(seq) > 2:
            chosen.append((seq, group))
    out = []
    for seq, group in chosen:
        g = sorted(group, key=lambda t: str(t["instance_id"]))
        out.append((seq, len(group), len(ts), g[0]))
    return out


def trace_table(t):
    rows = ["| step | action | result | belief after |", "|---|---|---|---|"]
    obs = t.get("obs") or [None] * len(t["actions"])
    for i, (a, o, b) in enumerate(zip(t["actions"], obs, t["beliefs"]), 1):
        if a in CRITICS:
            res = "PASS" if o else "FAIL"
        elif a == "verify":
            res = "correct (reward 100)" if o else "wrong (no reward)"
        elif a == "generate":
            res = f"new patch #{(t['patch'][i] if t.get('patch') and i < len(t['patch']) else '?')}"
        else:
            res = "stop"
        rows.append(f"| {i} | {ACT.get(a, a)} | {res} | {b:.3f} |")
    return "\n".join(rows)


def example_md(seq, n_seq, n_bench, t, bench, label):
    title, desc = describe(bench, t["instance_id"])
    b0 = t.get("b0", [None])[0] if isinstance(t.get("b0"), list) else t.get("b0")
    start = f"starting belief b0 = {b0:.3f}" if b0 is not None else ""
    if t.get("bT") is not None:
        start += f" (task-level {t['bT']:.3f})"
    gen = GEN_NAME.get(t["gen"], t["gen"])
    return (f"**{title}** · generator {gen} · {label}\n\n"
            f"> {desc}\n\n"
            f"Sequence `{seq_str(seq)}` — used on {n_seq}/{n_bench} tasks ({pct(n_seq / n_bench)}) of this benchmark. "
            f"{start}.\n\n{trace_table(t)}\n\n"
            f"**Result:** utility **{t['utility']:+.0f}** (reward {t.get('reward', 0):.0f} − cost {t.get('cost', 0):.0f}).\n")


# ---------------------------------------------------------------- write
L = []
w = L.append
w("# Agent traces of the Bayesian controller (Bayes-DP)\n")
w("Replay of the paper's cached runs (6 generators × 9 benchmarks, held-out 25% split, 3 cached patches per task). "
  "Each trace is what the controller did on one task: which critic it called, what the critic said, how its belief "
  "that the current patch is correct moved, and how the episode ended. Generated by `scripts/make_traces_md.py`; "
  "method and caveats in `action_traces/`, `per_task_prior/` and `swe_cver_sweep/` SUMMARY.md files.\n")
w("**Legend.** L0 = syntax check, L2 = public tests, L3 = LLM judge (critics); `regen` = take the next cached patch; "
  "`verify` = run the full hidden test suite (reward 100 if correct); `give_up` = stop with no reward. "
  "✓/✗ = critic PASS/FAIL. Belief = P(current patch is correct). Utility = 100·[verified correct] − Σ action costs.\n")
w("**Three settings compared.**\n")
w("- **A · paper controller** — as in the paper: one prior per (benchmark, generator), paper costs "
  "(SWE: verifier 90, critics 1/5; others: verifier 5, critics 1; regenerate 10).")
w("- **B · per-task beliefs** — same costs, but each task/patch starts from its own belief, computed from free "
  "signals (task text features, patch length), modelled jointly with the critics (`calK_jointcomb`).")
w("- **C · SWE with a cheaper verifier** — setting B on SWE-bench with verifier cost 20 instead of 90 "
  "(e.g. targeted tests instead of the full suite).\n")

# overall table
w("## 1. How diverse are the actions?\n")
w("| benchmark | setting | tasks | distinct sequences (pooled) | per setting: mean / max | settings where first action varies | entropy (bits) | give-up share | mean actions | mean utility |")
w("|---|---|---|---|---|---|---|---|---|---|")
for bench in ORDER:
    for lab, src in (("A", A), ("B", B)) + ((("C", C),) if bench.startswith("swe") else ()):
        s = stats(src, bench)
        if not s:
            continue
        w(f"| {BENCH_NAME[bench]} | {lab} | {s['n']} | {s['distinct']} | {s['per_cell_mean']:.1f} / {s['per_cell_max']} | "
          f"{s['first_varies']}/{s['cells']} | {s['H']:.2f} | {pct(s['give_up'])} | {s['len']:.2f} | {s['U']:+.1f} |")
tot = {lab: stats([dict(t, bench="all") for t in src], "all") for lab, src in (("A", A), ("B", B))}
w(f"| **all 9** | A | {tot['A']['n']} | {tot['A']['distinct']} | — | {tot['A']['first_varies']}/{tot['A']['cells']} | {tot['A']['H']:.2f} | {pct(tot['A']['give_up'])} | {tot['A']['len']:.2f} | {tot['A']['U']:+.1f} |")
w(f"| **all 9** | B | {tot['B']['n']} | {tot['B']['distinct']} | — | {tot['B']['first_varies']}/{tot['B']['cells']} | {tot['B']['H']:.2f} | {pct(tot['B']['give_up'])} | {tot['B']['len']:.2f} | {tot['B']['U']:+.1f} |")
w("")
w("**Action mix** (share of all actions taken):\n")
w("| benchmark | A · paper controller | B · per-task beliefs |")
w("|---|---|---|")
for bench in ORDER:
    sa, sb = stats(A, bench), stats(B, bench)
    w(f"| {BENCH_NAME[bench]} | {mix_str(sa)} | {mix_str(sb)} |")
sc = stats(C, "swe_verified"), stats(C, "swe_lite")
w(f"| SWE (C, verifier 20) | — | Lite: {mix_str(sc[1])}<br>Verified: {mix_str(sc[0])} |")
w("")
w("**Reading it.** In A the controller is close to a fixed recipe: every task in a setting gets the same first "
  "action (0/54 settings vary), SWE is always `give_up` (verifying at cost 90 needs belief > 0.9), MBPP+/HumanEval+ "
  "are always `verify` at once. B gives each task its own starting belief, so first actions differ in most "
  "settings and the number of distinct sequences roughly doubles — but utility gains over a well-chosen simple rule "
  "stay within noise (≈ +0.1 per task). C is the one regime where diverse behaviour also pays: on SWE with verifier "
  "cost 10–20 Bayes-DP beats the best train-picked rule (incl. signal-aware rules) by about +2–3 per task.\n")

# per benchmark
w("## 2. Per benchmark: sequence distribution and example tasks\n")
for bench in ORDER:
    w(f"### {BENCH_NAME[bench]}\n")
    for lab, src, name in (("A", A, "paper controller"), ("B", B, "per-task beliefs")) + \
            ((("C", C, "verifier cost 20"),) if bench.startswith("swe") else ()):
        s = stats(src, bench)
        w(f"**{lab} · {name}** — {s['n']} tasks, {s['distinct']} distinct sequences, entropy {s['H']:.2f} bits. Top sequences:\n")
        w("| share | tasks | sequence |")
        w("|---|---|---|")
        for seq, c in s["seqs"].most_common(8):
            w(f"| {pct(c / s['n'])} | {c} | `{seq_str(seq)}` |")
        rest = s["n"] - sum(c for _, c in s["seqs"].most_common(8))
        if rest:
            w(f"| {pct(rest / s['n'])} | {rest} | *({s['distinct'] - 8} other sequences)* |")
        w("")
    src, lab = (C, "setting C · SWE, verifier cost 20") if bench.startswith("swe") else (B, "setting B · per-task beliefs")
    w(f"**Example tasks** ({lab}; one task per distinct behaviour, lowest instance id — not cherry-picked by outcome):\n")
    for i, (seq, n_seq, n_b, t) in enumerate(pick_examples(src, bench), 1):
        w(f"{i}. " + example_md(seq, n_seq, n_b, t, bench, lab).replace("\n", "\n   ").rstrip() + "\n")
    w("")

OUT.write_text("\n".join(L))
print("wrote", OUT, "A", len(A), "B", len(B), "C", len(C), "desc", len(DESC))
