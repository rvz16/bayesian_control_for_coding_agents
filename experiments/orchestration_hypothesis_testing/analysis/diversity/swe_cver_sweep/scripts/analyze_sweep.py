"""Aggregate the verifier-cost sweep -> sweep.csv (per cell x C_ver), headroom.csv, curves.csv (pooled).

Fair rules (selection logic mirrors per_task_prior/scripts/analyze_ptp.py L84-104; that file is a script that runs on
import, so the ~15 lines are re-stated here and checked against its rules_vs_bdp.csv by check_analysis()):
  (ii)  per-(benchmark, regime) base rule picked on TRAIN (mean train utility over the benchmark's 6 cells)
  (iii) per-(benchmark, regime) rule picked on TRAIN among base rules U signal-thresholded (q, A, B) rules
  oracle_rule  per-cell base rule picked on EVAL (hindsight upper bound for rules)
Perfect-information oracle (`U_perfect`): knows Y of every cached patch of the task and pays the same action
costs (C_gen per regeneration, C_ver once, no critic cost): U = max(0, max_{k<3, Y_k=1} R - C_ver - k*C_gen).
Paired bootstrap per cell: analyze_util.boot_ci (2000, seed 42). Pooled curves: mean over cells of the per-cell
mean difference; CI by bootstrapping tasks within each cell (stratified, 2000, seed 42).
Robustness (iii-fine, `iiif`): same as (iii) but thresholds at train quantiles 0.05..0.95 (step 0.05), from
runs_fineq/ (calK_jointcomb, SWE only); its base-rule / controller utilities are asserted equal to runs/.
usage: analyze_sweep.py SWEEP_RUNS OUT_DIR PTP_DIR [FINEQ_RUNS]
"""
import csv
import glob
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "per_task_prior", "scripts")))
from common import MATCHED, boot_ci  # noqa: E402

RUNS, OUT, PTP = sys.argv[1:4]
FINEQ = sys.argv[4] if len(sys.argv) > 4 else None
FINE_QS = tuple(round(0.05 * i, 2) for i in range(1, 20))
CTRL = ("bayesian_DP", "bayesian_greedy")
R_, CGEN, H = 100.0, 10.0, 3
mean = lambda x: float(np.mean(x)) if len(x) else float("nan")
BENCH_ORDER = ["swe_lite", "swe_verified", "lcb_hard", "lcb_medium", "lcb_easy", "mbpp", "humaneval", "humanevalfix",
               "codecontests"]


def cver_of(reg):
    if reg == "slow": return 90.0
    if reg == "fast": return 5.0
    return float(re.match(r"cv(\d+)_", reg).group(1))


def crit_of(reg):
    return {"slow": "s", "fast": "f"}.get(reg, reg.split("_")[-1])


def thr_util(R, key, combo, split):
    src = R["thr_eval" if split == "eval" else "thr_train"].get(key)
    if src is not None:
        return src[combo]
    return (R["util_eval"][key] if split == "eval" else R["util_train"][key])[combo[1]]


def pick_fair(cells, key_of, qs=(0.25, 0.5, 0.75)):
    """cells: list of R for one benchmark & regime. Returns (bench_rule, bench_thr) exactly as analyze_ptp."""
    ks = [key_of(R) for R in cells]
    rules_of = {key_of(R): [p for p in R["util_eval"][key_of(R)] if p not in CTRL] for R in cells}
    Rk = {key_of(R): R for R in cells}
    rs = [r for r in rules_of[ks[0]] if all(r in rules_of[k] for k in ks)]
    s2 = {r: mean([Rk[k]["util_train"][k][r] for k in ks]) for r in rs}
    bench_rule = max(s2, key=s2.get)
    s3 = dict(s2)
    for c in [(q, A, B) for q in qs for A in rs for B in rs if A != B]:
        s3[c] = mean([thr_util(Rk[k], k, c, "train") for k in ks])
    bench_thr = max(s3, key=lambda c: (s3[c], isinstance(c, str)))
    return bench_rule, bench_thr


def perfect_oracle(R, key, ids):
    """Per-task utility of the perfect-information oracle, in the eval order `ids`."""
    cv = cver_of(key[2])
    Y = defaultdict(dict)
    for c in R["calib"]:
        Y[c["instance_id"]][c["patch_pos"]] = c["Y"]
    out = []
    for i in ids:
        ys = [Y[i][k] for k in sorted(Y[i])][:H]
        best = 0.0
        for k, y in enumerate(ys):
            if y == 1:
                best = max(best, R_ - cv - k * CGEN)
                break
        out.append(best)
    return np.array(out)


def ent(c):
    n = sum(c.values())
    return -sum((x / n) * math.log2(x / n) for x in c.values() if x) if n else 0.0


def cell_record(R, key, bench_rule, bench_thr, Rbase=None, fine=None):
    U = R["util_eval"][key]
    ids = [t["instance_id"] for t in R["traces"] if t["policy"] == "bayesian_DP" and t["regime"] == key[2]]
    assert len(ids) == len(U["bayesian_DP"])
    bdp, bg = np.array(U["bayesian_DP"]), np.array(U["bayesian_greedy"])
    rules = [p for p in U if p not in CTRL]
    orule = max(rules, key=lambda r: mean(U[r]))
    u_ii = np.array(U[bench_rule])
    u_iii = np.array(U[bench_thr] if isinstance(bench_thr, str) else thr_util(R, key, bench_thr, "eval"))
    u_or = np.array(U[orule])
    u_pi = perfect_oracle(R, key, ids)
    rows = {r["policy"]: r for r in R["rows"] if r["regime"] == key[2]}
    b, g, reg = key
    rec = dict(variant=R["variant"], critics=crit_of(reg), c_ver=cver_of(reg), regime=reg, bench=b, gen=g,
               matched_paper_cell=MATCHED[(b, g)], informative=R["informative"], n_eval=len(bdp), prior=round(R["prior"], 4),
               U_perfect=round(mean(u_pi), 3), U_BDP=round(mean(bdp), 3), U_BG=round(mean(bg), 3),
               U_always_verify=round(mean(U["always_verify"]), 3), U_never_submit=round(mean(U["never_submit"]), 3),
               rule_ii=bench_rule, U_rule_ii=round(mean(u_ii), 3),
               rule_iii=bench_thr if isinstance(bench_thr, str) else f"q{bench_thr[0]}:{bench_thr[1]}|{bench_thr[2]}",
               U_rule_iii=round(mean(u_iii), 3), oracle_rule_eval=orule, U_oracle_rule_eval=round(mean(u_or), 3))
    comps = [("ii", u_ii), ("iii", u_iii), ("oracle_rule", u_or)]
    if fine is not None:
        Rf, thr_f = fine
        assert Rf["util_eval"][key] == U, key
        u_iiif = np.array(U[thr_f] if isinstance(thr_f, str) else thr_util(Rf, key, thr_f, "eval"))
        rec["rule_iiif"] = thr_f if isinstance(thr_f, str) else f"q{thr_f[0]}:{thr_f[1]}|{thr_f[2]}"
        rec["U_rule_iiif"] = round(mean(u_iiif), 3)
        comps.append(("iiif", u_iiif))
    if Rbase is not None:
        comps.append(("base_BDP", np.array(Rbase["util_eval"][key]["bayesian_DP"])))
    for lab, arr in comps:
        for cn, mine in (("BDP", bdp), ("BG", bg)):
            if lab == "base_BDP" and cn == "BG":
                arr_ = np.array(Rbase["util_eval"][key]["bayesian_greedy"])
            else:
                arr_ = arr
            d = mine - arr_
            lo, hi = boot_ci(d)
            rec[f"{cn}_minus_{lab}"] = round(float(d.mean()), 3)
            rec[f"{cn}_ci_lo_{lab}"] = round(lo, 3); rec[f"{cn}_ci_hi_{lab}"] = round(hi, 3)
    for lab, arr in (("BDP", bdp), ("rule_ii", u_ii), ("rule_iii", u_iii), ("oracle_rule", u_or)):
        d = u_pi - arr
        lo, hi = boot_ci(d)
        rec[f"headroom_vs_{lab}"] = round(float(d.mean()), 3)
        rec[f"headroom_vs_{lab}_ci"] = f"[{lo:+.2f},{hi:+.2f}]"
    d = bdp - bg
    lo, hi = boot_ci(d)
    rec["BDP_minus_BG"] = round(float(d.mean()), 3); rec["BDP_ci_lo_BG"] = round(lo, 3); rec["BDP_ci_hi_BG"] = round(hi, 3)
    rec["BDP_beats_ii_and_iii_ci"] = bool(rec["BDP_ci_lo_ii"] > 0 and rec["BDP_ci_lo_iii"] > 0)
    rec["BDP_beats_ii_and_iii_ci_gt1"] = bool(rec["BDP_beats_ii_and_iii_ci"] and rec["BDP_minus_ii"] > 1 and rec["BDP_minus_iii"] > 1)
    rec["BDP_loses_ii_or_iii_ci"] = bool(rec["BDP_ci_hi_ii"] < 0 or rec["BDP_ci_hi_iii"] < 0)
    for cn, pol in (("BDP", "bayesian_DP"), ("BG", "bayesian_greedy")):
        r = rows[pol]
        rec[f"{cn}_n_distinct_seq"] = r["n_distinct_seq"]; rec[f"{cn}_n_first_actions"] = r["n_first_actions"]
        rec[f"{cn}_entropy_bits"] = round(r["entropy_bits"], 3); rec[f"{cn}_frac_give_up"] = round(r["frac_give_up"], 3)
        rec[f"{cn}_mean_verify"] = round(r["mean_verify"], 3); rec[f"{cn}_top1_share"] = round(r["top1_share"], 3)
        rec[f"{cn}_first_action_frac"] = json.dumps({k: round(v, 3) for k, v in r["first_action_frac"].items()})
    vecs = dict(BDP=bdp, BG=bg, ii=u_ii, iii=u_iii, oracle_rule=u_or, perfect=u_pi, ids=ids)
    if fine is not None:
        vecs["iiif"] = u_iiif
    return rec, vecs


def load(runs=None):
    D = defaultdict(dict)   # (regime, variant) -> {(b,g): R}
    for f in sorted(glob.glob(os.path.join(runs or RUNS, "*", "*", "*.pkl"))):
        reg = os.path.basename(os.path.dirname(os.path.dirname(f))); v = os.path.basename(os.path.dirname(f))
        R = pickle.load(open(f, "rb"))
        D[(reg, v)][(R["bench"], R["gen"])] = R
    return D


def analyse(D, base_variant="calK_const", Dfine=None):
    recs, vecs = [], {}
    for (reg, v), cells in sorted(D.items()):
        by_b = defaultdict(list)
        for (b, g), R in cells.items():
            by_b[b].append(R)
        for b, Rs in by_b.items():
            assert len(Rs) == 6, (reg, v, b, len(Rs))
            key_of = lambda R: (R["bench"], R["gen"], reg)
            Rs = sorted(Rs, key=lambda R: R["gen"])
            br, bt = pick_fair(Rs, key_of)
            Fs = (Dfine or {}).get((reg, v), {})
            Rsf = sorted([Fs[(R["bench"], R["gen"])] for R in Rs if (R["bench"], R["gen"]) in Fs], key=lambda R: R["gen"])
            btf = pick_fair(Rsf, key_of, FINE_QS)[1] if len(Rsf) == 6 else None
            for R in sorted(Rs, key=lambda R: R["gen"]):
                Rb = D.get((reg, base_variant), {}).get((R["bench"], R["gen"])) if v != base_variant else None
                fine = (Fs[(R["bench"], R["gen"])], btf) if btf is not None else None
                rec, vv = cell_record(R, key_of(R), br, bt, Rb, fine)
                recs.append(rec); vecs[(v, reg, b, R["gen"])] = vv
    return recs, vecs


def strat_ci(ds, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    ms = np.zeros(B)
    for d in ds:
        d = np.asarray(d, float)
        ms += d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    ms /= len(ds)
    return float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def curves(recs, vecs):
    """Pooled curves per (variant, critics, c_ver, scope)."""
    out = []
    groups = defaultdict(list)
    for r in recs:
        scopes = [r["bench"], "swe_pooled" if r["bench"].startswith("swe") else "nonswe_pooled"]
        for s in scopes:
            groups[(r["variant"], r["critics"], r["c_ver"], s)].append(r)
    for (v, cr, cv, s), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][2])):
        V = [vecs[(v, r["regime"], r["bench"], r["gen"])] for r in rs]
        row = dict(variant=v, critics=cr, c_ver=cv, scope=s, n_cells=len(rs), n_tasks=sum(len(x["BDP"]) for x in V),
                   U_perfect=round(mean([r["U_perfect"] for r in rs]), 3), U_BDP=round(mean([r["U_BDP"] for r in rs]), 3),
                   U_BG=round(mean([r["U_BG"] for r in rs]), 3), U_rule_ii=round(mean([r["U_rule_ii"] for r in rs]), 3),
                   U_rule_iii=round(mean([r["U_rule_iii"] for r in rs]), 3),
                   U_oracle_rule=round(mean([r["U_oracle_rule_eval"] for r in rs]), 3))
        extra = [("BDP_minus_iiif", "BDP", "iiif")] if all("iiif" in x for x in V) else []
        for name, a, bb in [("BDP_minus_ii", "BDP", "ii"), ("BDP_minus_iii", "BDP", "iii"), ("BG_minus_ii", "BG", "ii"),
                            ("BG_minus_iii", "BG", "iii"), ("BDP_minus_BG", "BDP", "BG"),
                            ("BDP_minus_oracle_rule", "BDP", "oracle_rule"), ("headroom_vs_BDP", "perfect", "BDP"),
                            ("headroom_vs_ii", "perfect", "ii"), ("headroom_vs_iii", "perfect", "iii"),
                            ("headroom_vs_oracle_rule", "perfect", "oracle_rule")] + extra:
            ds = [x[a] - x[bb] for x in V]
            lo, hi = strat_ci(ds)
            row[name] = round(mean([d.mean() for d in ds]), 3); row[f"{name}_lo"] = round(lo, 3); row[f"{name}_hi"] = round(hi, 3)
        if any("BDP_minus_base_BDP" in r for r in rs):
            row["BDP_minus_base_BDP"] = round(mean([r["BDP_minus_base_BDP"] for r in rs if "BDP_minus_base_BDP" in r]), 3)
        row["cells_beat_both_ci"] = sum(r["BDP_beats_ii_and_iii_ci"] for r in rs)
        row["cells_beat_both_ci_gt1"] = sum(r["BDP_beats_ii_and_iii_ci_gt1"] for r in rs)
        if extra:
            row["cells_beat_ii_iii_iiif_ci_gt1"] = sum(1 for r in rs if r["BDP_beats_ii_and_iii_ci_gt1"] and r["BDP_ci_lo_iiif"] > 0 and r["BDP_minus_iiif"] > 1)
        row["cells_lose_ii_or_iii_ci"] = sum(r["BDP_loses_ii_or_iii_ci"] for r in rs)
        row["list_beat_both_ci_gt1"] = ";".join(f"{r['bench']}/{r['gen']}" for r in rs if r["BDP_beats_ii_and_iii_ci_gt1"])
        row["list_lose_ii_or_iii_ci"] = ";".join(f"{r['bench']}/{r['gen']}" for r in rs if r["BDP_loses_ii_or_iii_ci"])
        row["BDP_nseq_mean"] = round(mean([r["BDP_n_distinct_seq"] for r in rs]), 2)
        row["BDP_nseq_max"] = max(r["BDP_n_distinct_seq"] for r in rs)
        row["BDP_cells_first_action_varies"] = sum(1 for r in rs if r["BDP_n_first_actions"] > 1)
        row["BDP_entropy_mean"] = round(mean([r["BDP_entropy_bits"] for r in rs]), 3)
        row["BDP_frac_give_up_mean"] = round(mean([r["BDP_frac_give_up"] for r in rs]), 3)
        row["BG_nseq_mean"] = round(mean([r["BG_n_distinct_seq"] for r in rs]), 2)
        out.append(row)
    return out


def check_analysis(recs):
    """My fair-rule selection on cv90_s / cv5_f must equal per_task_prior/rules_vs_bdp.csv (slow / fast)."""
    ref = {}
    with open(os.path.join(PTP, "rules_vs_bdp.csv")) as f:
        for r in csv.DictReader(f):
            if r["variant"] in ("calK_jointcomb", "calK_const"):
                ref[(r["variant"], r["bench"], r["gen"], r["regime"])] = r
    n = bad = 0; diffs = []
    for r in recs:
        reg = {"cv90_s": "slow", "cv5_f": "fast"}.get(r["regime"])
        if reg is None:
            continue
        k = (r["variant"], r["bench"], r["gen"], reg)
        if k not in ref:
            continue
        x = ref[k]; n += 1
        pairs = [(r["U_BDP"], float(x["U_BDP"])), (r["U_BG"], float(x["U_BG"])),
                 (r["BDP_minus_ii"], float(x["BDP_minus_bench_rule_trainpicked"])),
                 (r["BDP_minus_iii"], float(x["BDP_minus_thr_rule_trainpicked"])),
                 (r["BDP_ci_lo_ii"], float(x["BDP_ci_lo_bench_rule_trainpicked"])),
                 (r["BDP_ci_lo_iii"], float(x["BDP_ci_lo_thr_rule_trainpicked"])),
                 (r["BDP_minus_oracle_rule"], float(x["BDP_minus_oracle_rule"]))]
        ok = all(abs(a - b) < 1e-9 for a, b in pairs) and r["rule_ii"] == x["bench_rule_trainpicked"] and r["rule_iii"] == x["thr_rule_trainpicked"]
        if not ok:
            bad += 1; diffs.append(dict(key=k, mine=[a for a, _ in pairs] + [r["rule_ii"], r["rule_iii"]],
                                        ref=[b for _, b in pairs] + [x["bench_rule_trainpicked"], x["thr_rule_trainpicked"]]))
    return dict(rows_compared=n, mismatches=bad, PASS=(n > 0 and bad == 0), examples=diffs[:5])


def ptp_headroom():
    """Perfect-info headroom on per_task_prior's own runs (paper regime), to compare with SUMMARY §6 (4.88 / 5.59 / 2.40)."""
    out = {}
    for v in ("calK_jointcomb", "calK_const"):
        gaps = defaultdict(list)
        for f in sorted(glob.glob(os.path.join(PTP, "runs", v, "*.pkl"))):
            R = pickle.load(open(f, "rb"))
            reg = "slow" if R["bench"].startswith("swe") else "fast"
            key = (R["bench"], R["gen"], reg)
            ids = [t["instance_id"] for t in R["traces"] if t["policy"] == "bayesian_DP" and t["regime"] == reg]
            g = mean(perfect_oracle(R, key, ids)) - mean(R["util_eval"][key]["bayesian_DP"])
            gaps["all"].append(g); gaps["swe" if R["bench"].startswith("swe") else "nonswe"].append(g)
        out[v] = {k: round(mean(x), 3) for k, x in gaps.items()} | {"n_cells": len(gaps["all"])}
    return out


if __name__ == "__main__":
    D = load()
    Dfine = load(FINEQ) if FINEQ else None
    recs, vecs = analyse(D, Dfine=Dfine)
    recs.sort(key=lambda r: (r["variant"], r["critics"], BENCH_ORDER.index(r["bench"]), r["gen"], r["c_ver"]))
    swe = [r for r in recs if r["bench"].startswith("swe")]
    non = [r for r in recs if not r["bench"].startswith("swe")]
    write_csv(os.path.join(OUT, "sweep.csv"), swe)
    write_csv(os.path.join(OUT, "sweep_nonswe.csv"), non)
    HK = ["variant", "critics", "c_ver", "bench", "gen", "matched_paper_cell", "n_eval", "U_perfect", "U_BDP", "U_rule_ii",
          "U_rule_iii", "U_oracle_rule_eval", "headroom_vs_BDP", "headroom_vs_BDP_ci", "headroom_vs_rule_ii",
          "headroom_vs_rule_ii_ci", "headroom_vs_rule_iii", "headroom_vs_rule_iii_ci", "headroom_vs_oracle_rule",
          "headroom_vs_oracle_rule_ci", "BDP_minus_ii", "BDP_ci_lo_ii", "BDP_ci_hi_ii", "BDP_minus_iii", "BDP_ci_lo_iii",
          "BDP_ci_hi_iii", "BDP_minus_iiif", "BDP_ci_lo_iiif", "BDP_ci_hi_iiif"]
    write_csv(os.path.join(OUT, "headroom.csv"), [{k: r.get(k, "") for k in HK} for r in recs])
    cv = curves(recs, vecs)
    write_csv(os.path.join(OUT, "curves.csv"), cv)
    chk = check_analysis(recs)
    chk["perfect_info_headroom_on_per_task_prior_runs"] = ptp_headroom()
    json.dump(chk, open(os.path.join(OUT, "logs", "analysis_check.json"), "w"), indent=1, default=str)
    pickle.dump(vecs, open(os.path.join(OUT, "logs", "vecs.pkl"), "wb"))
    print("records", len(recs), "curves", len(cv), "check", json.dumps(chk)[:600])
