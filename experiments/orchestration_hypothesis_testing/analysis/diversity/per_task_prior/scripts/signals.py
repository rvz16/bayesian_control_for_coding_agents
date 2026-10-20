"""Step 1: pre-action signals for all 54 cells + AUROC vs Y on the eval split.

Outputs (in per_task_prior/):
  signals.csv     one row per cached patch: cell, split, Y, every signal (learned scores on TRAIN are out-of-fold)
  auroc.csv       per cell x signal: oriented eval AUROC + task-cluster bootstrap CI (2000, seed 42);
                  plus POOLED and per-benchmark pooled rows (within-cell stratified AUROC)
  cache/match_report.json, cache/learned_meta.json

Leakage rule. For target cell c = (bench, gen), every learned model is trained on patches of ALL generators of
the benchmark, restricted to tasks in c's TRAIN split (splits differ across generators on SWE-Verified and
CodeContests, so the model is always re-fit per target cell). Train-split scores of c are 5-fold out-of-fold
(folds over c's train tasks); eval-split scores come from a model fit on all of c's train tasks. No eval-split
task of c ever contributes a label. Orientation of raw features (sign) is chosen on train.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import csv
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, MATCHED, all_cells, fast_auc  # noqa: E402
import task_meta  # noqa: E402

from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

GENS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45", "qwen25_32b", "gpt_oss_20b"]
PATCH_SIGNALS = ["log_diff_chars", "empty_diff", "L1_lint"]
LEARNED = ["tasklr", "difficulty", "combtask"]
PAID = ["L0_syntax", "L2_public_tests", "L3_llm_review"]
NFOLD = 5
B = 2000


def load():
    cells = all_cells()
    by_bench = defaultdict(dict)
    for b, g, gg, cell in cells:
        for i, rs in cell["by"].items():
            by_bench[b].setdefault(i, rs)
    meta, report = task_meta.build(by_bench)
    return cells, meta, report


# ---------------------------------------------------------------- learned models
class Featurizer:
    def __init__(self, meta_b, train_tasks, use_text, use_feat):
        self.use_text, self.use_feat = use_text, use_feat
        M = [meta_b.get(i) for i in train_tasks]
        if use_text:
            self.tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
            self.tf.fit([(m or {}).get("text", "") or "" for m in M] + ["_"])
        if use_feat:
            self.num = sorted({k for m in M if m for k in m["num"] if not k.startswith("priv_")})
            vals = {k: np.array([m["num"].get(k) for m in M if m and m["num"].get(k) is not None], float) for k in self.num}
            self.num = [k for k in self.num if len(vals[k]) and np.std(vals[k]) > 0]
            self.med = {k: float(np.median(vals[k])) for k in self.num}
            self.mu = {k: float(np.mean(vals[k])) for k in self.num}
            self.sd = {k: float(np.std(vals[k])) for k in self.num}
            cnt = defaultdict(int)
            for m in M:
                for t in self._cat_tokens(m):
                    cnt[t] += 1
            self.cat = sorted(t for t, c in cnt.items() if c >= 3)
            self.cat_ix = {t: j for j, t in enumerate(self.cat)}

    @staticmethod
    def _cat_tokens(m):
        if not m:
            return []
        out = []
        for k, v in m["cat"].items():
            if k == "tags":
                out += [f"tag={t}" for t in (v or "").split()]
            else:
                out.append(f"{k}={v}")
        return out

    def transform(self, tasks, gens, meta_b):
        blocks = []
        if self.use_text:
            blocks.append(self.tf.transform([(meta_b.get(i) or {}).get("text", "") or "" for i in tasks]))
        if self.use_feat:
            X = np.zeros((len(tasks), len(self.num)))
            C = np.zeros((len(tasks), len(self.cat)))
            for r, i in enumerate(tasks):
                m = meta_b.get(i)
                for j, k in enumerate(self.num):
                    v = m["num"].get(k) if m else None
                    v = self.med[k] if v is None or (isinstance(v, float) and math.isnan(v)) else v
                    X[r, j] = (v - self.mu[k]) / self.sd[k]
                for t in self._cat_tokens(m):
                    if t in self.cat_ix:
                        C[r, self.cat_ix[t]] = 1.0
            blocks += [sparse.csr_matrix(X), sparse.csr_matrix(C)]
        G = np.zeros((len(tasks), len(GENS)))
        for r, g in enumerate(gens):
            G[r, GENS.index(g)] = 1.0
        blocks.append(sparse.csr_matrix(G))
        return sparse.hstack(blocks).tocsr()


def fit_predict(kind, meta_b, train_rows, pred_rows):
    """rows: list of (task, gen, Y). Returns logits for pred_rows."""
    use_text = kind in ("difficulty", "combtask")
    use_feat = kind in ("tasklr", "combtask")
    tasks = sorted({t for t, _, _ in train_rows})
    fz = Featurizer(meta_b, tasks, use_text, use_feat)
    X = fz.transform([t for t, _, _ in train_rows], [g for _, g, _ in train_rows], meta_b)
    y = np.array([y for _, _, y in train_rows])
    if y.min() == y.max():
        return np.full(len(pred_rows), 0.0)
    lr = LogisticRegression(C=1.0, max_iter=5000)
    lr.fit(X, y)
    Xp = fz.transform([t for t, _, _ in pred_rows], [g for _, g, _ in pred_rows], meta_b)
    return lr.decision_function(Xp)


def learned_for_cell(args):
    """Learned scores for one target cell. Returns {(iid, patch_id): {kind: logit}}."""
    tgt, cells_b, meta_b = args
    b, g, gg, cell = tgt
    train = set(cell["train_ids"])
    tr_sorted = sorted(train)
    fold = {t: k % NFOLD for k, t in zip(range(len(tr_sorted)), [tr_sorted[j] for j in np.random.default_rng(0).permutation(len(tr_sorted))])}
    pool_rows = []                                  # (task, gen, Y) over all generators, train tasks of target only
    for (b2, g2, gg2, c2) in cells_b:
        for i, rs in c2["by"].items():
            if i in train:
                pool_rows += [(i, gg2, int(r["Y"])) for r in rs]
    own = [(i, int(r.get("patch_id", k)), int(r["Y"])) for i, rs in cell["by"].items() for k, r in enumerate(rs)]
    out = defaultdict(dict)
    for kind in LEARNED:
        # eval: full-train model
        ev = [(i, pid, y) for i, pid, y in own if i not in train]
        s = fit_predict(kind, meta_b, pool_rows, [(i, gg, y) for i, _, y in ev])
        for (i, pid, _), v in zip(ev, s):
            out[(i, pid)][kind] = float(v)
        # train: out-of-fold
        for f in range(NFOLD):
            fit_rows = [r for r in pool_rows if fold[r[0]] != f]
            pr = [(i, pid, y) for i, pid, y in own if i in train and fold[i] == f]
            if not pr:
                continue
            s = fit_predict(kind, meta_b, fit_rows, [(i, gg, y) for i, _, y in pr])
            for (i, pid, _), v in zip(pr, s):
                out[(i, pid)][kind] = float(v)
    return (b, gg), dict(out)


# ---------------------------------------------------------------- AUROC with task-cluster bootstrap
def _ustat(s, y):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 0.0
    r = rankdata(s)
    return float(r[y == 1].sum() - n1 * (n1 + 1) / 2), float(n1 * n0)


def auroc_cell(args):
    key, rows, sig_names = args
    ev = [r for r in rows if r["split"] == "eval"]
    tr = [r for r in rows if r["split"] == "train"]
    tasks = sorted({r["instance_id"] for r in ev})
    tix = {t: j for j, t in enumerate(tasks)}
    rng = np.random.default_rng(42)
    boots = rng.integers(0, len(tasks), size=(B, len(tasks)))
    by_task = defaultdict(list)
    for k, r in enumerate(ev):
        by_task[tix[r["instance_id"]]].append(k)
    res = {}
    for s in sig_names:
        str_ = np.array([r[s] if r[s] is not None else np.nan for r in tr], float)
        ytr = np.array([r["Y"] for r in tr])
        ok = ~np.isnan(str_)
        a_tr = fast_auc(str_[ok], ytr[ok]) if ok.any() else float("nan")
        sign = -1.0 if (not math.isnan(a_tr) and a_tr < 0.5) else 1.0
        sev = np.array([r[s] if r[s] is not None else np.nan for r in ev], float) * sign
        yev = np.array([r["Y"] for r in ev])
        okv = ~np.isnan(sev)
        num, den = _ustat(sev[okv], yev[okv]) if okv.any() else (0.0, 0.0)
        bn, bd = np.zeros(B), np.zeros(B)
        for bi in range(B):
            idx = np.fromiter((k for t in boots[bi] for k in by_task[t]), int)
            idx = idx[okv[idx]]
            bn[bi], bd[bi] = _ustat(sev[idx], yev[idx])
        res[s] = dict(sign=sign, auc_train=a_tr, num=num, den=den, bnum=bn, bden=bd, n_eval=int(okv.sum()),
                      n_pos=int(yev[okv].sum()), n_unique=int(len(np.unique(sev[okv]))) if okv.any() else 0)
    return key, res


def main():
    cells, meta, report = load()
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    json.dump(report, open(os.path.join(ROOT, "cache", "match_report.json"), "w"), indent=1)
    print(json.dumps({b: round(r["match_rate"], 3) for b, r in report.items()}), flush=True)

    jobs = [((b, g, gg, cell), [c for c in cells if c[0] == b], meta[b]) for (b, g, gg, cell) in cells]
    with Pool(min(12, os.cpu_count() or 4)) as P:
        learned = dict(P.map(learned_for_cell, jobs))
    print("learned done", flush=True)

    task_feats = sorted({k for b in meta for m in meta[b].values() for k in m["num"]})
    rows_all = []
    for (b, g, gg, cell) in cells:
        tr = set(cell["train_ids"])
        L = learned[(b, gg)]
        for i, rs in cell["by"].items():
            m = meta[b].get(i)
            for k, r in enumerate(rs):
                pid = int(r.get("patch_id", k))
                row = dict(bench=b, gen=gg, cell=f"{b}/{gg}", matched_paper_cell=MATCHED[(b, gg)], instance_id=i,
                           patch_id=pid, patch_pos=k, split="train" if i in tr else "eval", Y=int(r["Y"]),
                           task_meta_matched=m is not None, diff_chars=r.get("diff_chars"),
                           log_diff_chars=math.log1p(r["diff_chars"]) if r.get("diff_chars") is not None else None,
                           empty_diff=float(r.get("diff_chars") == 0) if r.get("diff_chars") is not None else None,
                           L1_lint=float(bool(r["L1_lint"])) if r.get("L1_lint") is not None else None)
                for f in task_feats:
                    v = m["num"].get(f) if m else None
                    row[f"t_{f}"] = None if v is None else float(v)
                for kind in LEARNED:
                    row[kind] = L.get((i, pid), {}).get(kind)
                for f in PAID:
                    row[f] = None if r.get(f) is None else float(bool(r[f]))
                rows_all.append(row)
    with open(os.path.join(ROOT, "signals.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_all[0].keys())); w.writeheader(); w.writerows(rows_all)
    print("signals.csv", len(rows_all), flush=True)

    sig_names = PATCH_SIGNALS + [f"t_{f}" for f in task_feats] + LEARNED + PAID
    by_cell = defaultdict(list)
    for r in rows_all:
        by_cell[(r["bench"], r["gen"])].append(r)
    ajobs = []
    for key, rows in by_cell.items():
        present = [s for s in sig_names if any(r[s] is not None for r in rows)]
        ajobs.append((key, rows, present))
    with Pool(min(12, os.cpu_count() or 4)) as P:
        A = dict(P.map(auroc_cell, ajobs))
    out = []
    for (b, gg), res in A.items():
        for s, d in res.items():
            auc = d["num"] / d["den"] if d["den"] else float("nan")
            bs = d["bnum"] / np.where(d["bden"] > 0, d["bden"], np.nan)
            lo, hi = (float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))) if d["den"] else (float("nan"),) * 2
            out.append(dict(scope="cell", bench=b, gen=gg, cell=f"{b}/{gg}", matched_paper_cell=MATCHED[(b, gg)], signal=s,
                            signal_type=("patch" if s in PATCH_SIGNALS else "paid_critic" if s in PAID else "learned" if s in LEARNED
                                         else "privileged_task" if s.startswith("t_priv_") else "task"),
                            orientation=int(d["sign"]), auc_train=round(d["auc_train"], 4), auc_eval=round(auc, 4),
                            ci_lo=round(lo, 4), ci_hi=round(hi, 4), ci_excludes_half=bool(lo > 0.5 or hi < 0.5),
                            n_eval_patches=d["n_eval"], n_pos=d["n_pos"], n_unique_values=d["n_unique"],
                            match_rate=round(report[b]["match_rate"], 3)))
    # pooled (within-cell stratified) AUROC: all 54, matched 43, per benchmark
    sigs = sorted({s for res in A.values() for s in res})
    scopes = [("POOLED_all54", lambda b, g: True), ("POOLED_matched43", lambda b, g: MATCHED[(b, g)])]
    scopes += [(f"POOLED_{bb}", (lambda bb: lambda b, g: b == bb)(bb)) for bb in sorted({k[0] for k in A})]
    for name, keep in scopes:
        for s in sigs:
            ks = [k for k in A if keep(*k) and s in A[k] and A[k][s]["den"] > 0]
            if not ks:
                continue
            num = sum(A[k][s]["num"] for k in ks); den = sum(A[k][s]["den"] for k in ks)
            bn = sum(A[k][s]["bnum"] for k in ks); bd = sum(A[k][s]["bden"] for k in ks)
            bs = bn / bd
            lo, hi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
            out.append(dict(scope=name, bench="", gen="", cell=name, matched_paper_cell="", signal=s,
                            signal_type=next(r["signal_type"] for r in out if r["signal"] == s and r["scope"] == "cell"),
                            orientation="", auc_train="", auc_eval=round(num / den, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                            ci_excludes_half=bool(lo > 0.5 or hi < 0.5), n_eval_patches=sum(A[k][s]["n_eval"] for k in ks),
                            n_pos=sum(A[k][s]["n_pos"] for k in ks), n_unique_values="", match_rate="", n_cells=len(ks)))
    keys = []
    for r in out:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(ROOT, "auroc.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(out)
    print("auroc.csv", len(out), flush=True)


if __name__ == "__main__":
    main()
