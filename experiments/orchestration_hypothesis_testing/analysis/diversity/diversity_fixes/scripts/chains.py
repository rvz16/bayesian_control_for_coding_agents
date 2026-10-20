"""Self-Refine / Reflexion repair chains: cleaning, rooting on calibration patch p0, version selection,
and the train-split repair kernel.

A chain is replayable for an instance only if its step 0 IS calibration patch p0: identical Y, identical
L0/L1/L2/L3 verdicts (None included) and identical size (code_chars == diff_chars, or len(diff|code) ==
diff_chars). Then `repair` from depth d moves to recorded chain step d+1 (its own recorded Y and critic
verdicts). Nothing beyond the recorded steps is ever replayed.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from cells import ITER, art_dir
from data import DL, load_records

SIG_FIELDS = ("L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review")
ERA_END = "2026-08-31T23:59:59Z"


def ident(r):
    """Patch identity used to decide whether a repair changed the patch."""
    if r.get("code"):
        return ("code", r["code"])
    if r.get("diff"):
        return ("diff", r["diff"])
    return ("chars", r.get("code_chars"))


def size(r):
    if r.get("code_chars") is not None:
        return int(r["code_chars"])
    if r.get("diff"):
        return len(r["diff"])
    if r.get("code"):
        return len(r["code"])
    return None


def matches_cal(step0, calrec):
    if step0.get("Y") is None or calrec.get("Y") is None:
        return False
    if int(step0["Y"]) != int(calrec["Y"]):
        return False
    if any(step0.get(f) != calrec.get(f) for f in SIG_FIELDS):
        return False
    return size(step0) is not None and size(step0) == calrec.get("diff_chars")


def clean_chains(recs, bench):
    """-> (chains: iid -> list[rec] (contiguous steps 0..L-1, Y and used critics non-None), stats Counter)."""
    need = ["L0_syntax", "L1_lint", "L3_llm_review"] + ([] if bench.startswith("swe") else ["L2_public_tests"])
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[str(r["instance_id"])][int(r["step"])].append(r)
    out, st = {}, Counter()
    for i, steps in by.items():
        key = lambda r: json.dumps([r.get("Y")] + [r.get(f) for f in SIG_FIELDS] + [ident(r)], default=str)
        if any(len({key(x) for x in v}) > 1 for v in steps.values()):
            st["drop_inconsistent_duplicate_steps"] += 1
            continue
        if any(len(v) > 1 for v in steps.values()):
            st["dedup_consistent_duplicates"] += 1
        ch = []
        for s in range(0, max(steps) + 1):
            if s not in steps:
                st["truncated_gap"] += 1
                break
            r = steps[s][0]
            if r.get("Y") is None or any(r.get(f) is None for f in need):
                st["truncated_missing_Y_or_critic"] += 1
                break
            ch.append(r)
        if ch:
            out[i] = ch
    return out, st


def root_chains(chains, cal_by):
    """Keep chains whose step 0 is calibration p0 of the same instance."""
    rooted, st = {}, Counter()
    for i, ch in chains.items():
        if i not in cal_by:
            st["chain_instance_not_in_calibration"] += 1
            continue
        p0 = cal_by[i][0]
        if matches_cal(ch[0], p0):
            rooted[i] = ch
        else:
            st["step0_not_p0"] += 1
            if any(matches_cal(ch[0], p) for p in cal_by[i][1:]):
                st["step0_matches_p1_or_p2_only"] += 1
    return rooted, st


def fit_repair_kernel(rooted, ids):
    """Kernel on train-split rooted chains. Returns dict with
    - P_fix/P_break over ALL consecutive pairs (spec kernel, Beta(1,1)),
    - the 'obs' model: P(unchanged | Y) and P_fix/P_break over CHANGED pairs only,
    - continuation probabilities rho[d] = P(step d+1 exists | step d exists), overall and by the kind of
      the transition that led into step d (same / chg). Depth with no train data -> 0."""
    c = Counter(); cs = Counter(); cc = Counter(); u = Counter()
    reach = Counter(); cont = Counter()
    reach_k = Counter(); cont_k = Counter()
    for i in ids:
        ch = rooted.get(i)
        if not ch:
            continue
        for d in range(len(ch)):
            reach[d] += 1
            if d + 1 < len(ch):
                cont[d] += 1
            if d >= 1:
                kind = "same" if ident(ch[d]) == ident(ch[d - 1]) else "chg"
                reach_k[(kind, d)] += 1
                if d + 1 < len(ch):
                    cont_k[(kind, d)] += 1
        for a, b in zip(ch, ch[1:]):
            ya, yb = int(a["Y"]), int(b["Y"])
            c[f"{ya}{yb}"] += 1
            same = ident(a) == ident(b)
            u[(ya, same)] += 1
            if not same:
                cc[f"{ya}{yb}"] += 1
    lap = lambda k, n: (k + 1) / (n + 2)
    # critic likelihoods on repaired patches (train chain steps >= 1), Beta(1,1)
    lk = {}
    for f in SIG_FIELDS:
        k = Counter()
        for i in ids:
            for r in (rooted.get(i) or [])[1:]:
                if r.get(f) is not None:
                    k[(int(r["Y"]), bool(r[f]))] += 1
        n1, n0 = k[(1, True)] + k[(1, False)], k[(0, True)] + k[(0, False)]
        if n1 + n0:
            lk[f] = (lap(k[(1, True)], n1), lap(k[(0, True)], n0))
    D = max(reach) + 2 if reach else 2
    rho = {d: (lap(cont[d], reach[d]) if reach[d] else 0.0) for d in range(D + 1)}
    rho_k = {k: {d: (lap(cont_k[(k, d)], reach_k[(k, d)]) if reach_k[(k, d)] else 0.0) for d in range(D + 1)}
             for k in ("same", "chg")}
    return {
        "P_fix": lap(c["01"], c["00"] + c["01"]), "P_break": lap(c["10"], c["10"] + c["11"]),
        "counts": dict(c), "n_pairs": sum(c.values()),
        "q_same_Y1": lap(u[(1, True)], u[(1, True)] + u[(1, False)]),
        "q_same_Y0": lap(u[(0, True)], u[(0, True)] + u[(0, False)]),
        "P_fix_chg": lap(cc["01"], cc["00"] + cc["01"]), "P_break_chg": lap(cc["10"], cc["10"] + cc["11"]),
        "counts_chg": dict(cc),
        "rho": rho, "rho_kind": rho_k,
        "n_train_chains": sum(1 for i in ids if i in rooted),
        "likes_rep": lk,
    }


def all_versions(bench, gen, method):
    """Paper-era versions (created <= 2026-08-31) of exactly `<bench>_<gen>_<method>_records`."""
    name = f"{bench}_{gen}_{method}_records"
    out = []
    for d in sorted(os.listdir(DL)):
        if d.startswith(name + "__v"):
            meta = json.load(open(os.path.join(DL, d, "_artifact_meta.json")))
            if meta["created"] <= ERA_END:
                out.append((d.rsplit("__", 1)[1], os.path.join(DL, d), meta["created"], meta.get("digest")))
    return sorted(out, key=lambda x: x[2])


def select_version(bench, gen, method, cal_by, mode):
    """mode 'at' = the action_traces version (cells.ITER); 'maxov' = paper-era version with the most
    replayable rooted chains (rooted & >=1 transition); ties -> action_traces version, then earliest."""
    at = ITER[(bench, gen)][0 if method == "selfrefine" else 1]
    if mode == "at":
        d = art_dir(f"{bench}_{gen}_{method}_records", at)
        return at, d
    best = None
    for v, d, created, _ in all_versions(bench, gen, method):
        ch, _ = clean_chains(load_records(d), bench)
        rooted, _ = root_chains(ch, cal_by)
        score = sum(1 for x in rooted.values() if len(x) >= 2)
        key = (score, v == at, -int(created.replace("-", "").replace(":", "").replace("T", "").replace("Z", "")))
        if best is None or key > best[0]:
            best = (key, v, d)
    return best[1], best[2]
