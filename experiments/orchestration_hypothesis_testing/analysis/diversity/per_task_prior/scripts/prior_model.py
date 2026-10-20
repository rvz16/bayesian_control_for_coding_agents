"""Discretised pre-action signals -> per-task / per-patch starting beliefs (naive Bayes on top of the cell prior).

Signal spec (dict):
  task  : None | 'tasklr' | 'difficulty' | 'combtask' | 'const'   task-level score, shared by a task's patches
  patch : tuple of 'len' | 'lint'                                  patch-level signals, one draw per cached patch
  K     : number of quantile bins (task score; non-empty patch length)
Bins are fitted on the TRAIN split of the cell (learned scores on train are out-of-fold, see signals.py).
  task score : K quantile bins over train TASKS' scores, edges e_1<..<e_{K-1}, bin j = #(edges < s).
  len        : bin 0 = empty diff (diff_chars == 0) if any train patch is empty; bins 1..K = quantile bins of
               log1p(diff_chars) over train non-empty patches.
  lint       : L1_lint verdict (pass / fail / missing) as observed on train; dropped if constant on train.
Likelihoods P(bin | Y) = (n_{bin,Y} + 1) / (n_Y + K_eff) on train patches (Beta(1,1) / Laplace).
A bin with P(bin|Y=1) == P(bin|Y=0) leaves the belief bit-identical (skip), so a constant signal reproduces the
base controller exactly.
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np


def bin_update(b, l1, l0):
    """Bayes update of P(Y=1) by an observation with likelihoods l1 = P(o|Y=1), l0 = P(o|Y=0)."""
    if l1 == l0:
        return b
    num = l1 * b
    return num / max(num + l0 * (1 - b), 1e-12)


def _edges(vals, K):
    vals = np.asarray(vals, float)
    if K <= 1 or len(vals) == 0:
        return np.array([])
    e = np.unique(np.quantile(vals, np.linspace(0, 1, K + 1)[1:-1]))
    return e


def _bin(e, v):
    return int(np.searchsorted(e, v, side="left"))      # bin j: e_{j-1} < v <= e_j


def _lik(bins, ys, nb):
    bins = np.asarray(bins, int); ys = np.asarray(ys, int)
    n1, n0 = int((ys == 1).sum()), int((ys == 0).sum())
    l1 = np.array([((bins == j) & (ys == 1)).sum() + 1 for j in range(nb)], float) / (n1 + nb)
    l0 = np.array([((bins == j) & (ys == 0)).sum() + 1 for j in range(nb)], float) / (n0 + nb)
    return l1, l0


class SignalModel:
    def __init__(self, spec, prior, rows_by_patch, train_ids):
        """rows_by_patch: {(iid, patch_pos): signals.csv row (dict)}; prior: cell prior (Laplace)."""
        self.spec, self.prior = spec, prior
        K = spec.get("K", 4)
        train = set(train_ids)
        tr_rows = [r for (i, _), r in rows_by_patch.items() if i in train]
        ytr = [int(r["Y"]) for r in tr_rows]
        self.task_edges = None
        ts = spec.get("task")
        if ts and ts != "const":
            tvals = {}
            for r in tr_rows:
                tvals.setdefault(r["instance_id"], float(r[ts]))
            self.task_edges = _edges(list(tvals.values()), K)
            self.n_task_bins = len(self.task_edges) + 1
            self.lT1, self.lT0 = _lik([self._tbin(r) for r in tr_rows], ytr, self.n_task_bins)
        else:
            self.n_task_bins = 1
            self.lT1, self.lT0 = np.array([1.0]), np.array([1.0])
        self.pfeat = []                      # list of (name, edges/None, nb)
        for p in spec.get("patch", ()):
            if p == "len":
                dc = np.array([float(r["diff_chars"]) for r in tr_rows])
                has_empty = bool((dc == 0).any())
                e = _edges(np.log1p(dc[dc > 0]), K)
                self.pfeat.append(("len", (has_empty, e), (1 if has_empty else 0) + len(e) + 1))
            elif p.startswith("col:"):            # generic numeric patch-level column, K quantile bins on train
                e = _edges([float(r[p[4:]]) for r in tr_rows], K)
                self.pfeat.append((p, e, len(e) + 1))
            elif p == "lint":
                cats = sorted({self._lint_cat(r) for r in tr_rows})      # 0 / 1 / 2 = missing verdict
                if len(cats) >= 2:
                    self.pfeat.append(("lint", {c: j for j, c in enumerate(cats)}, len(cats)))
            else:
                raise ValueError(p)
        # joint patch bins (naive Bayes product across patch features)
        self.pcombos = list(product(*[range(nb) for _, _, nb in self.pfeat])) if self.pfeat else [()]
        cols = [[self._pbin_one(f, r) for r in tr_rows] for f in self.pfeat]
        liks = [_lik(c, ytr, f[2]) for c, f in zip(cols, self.pfeat)]
        self.pliks = liks
        L1 = np.ones(len(self.pcombos)); L0 = np.ones(len(self.pcombos))
        for ci, combo in enumerate(self.pcombos):
            for k, j in enumerate(combo):
                L1[ci] *= liks[k][0][j]; L0[ci] *= liks[k][1][j]
        self.pL1, self.pL0 = L1, L0
        self.combo_ix = {c: k for k, c in enumerate(self.pcombos)}

    # ---- binning
    @staticmethod
    def _lint_cat(r):
        v = r.get("L1_lint")
        return 2 if v in (None, "") else int(float(v))

    def _tbin(self, r):
        return _bin(self.task_edges, float(r[self.spec["task"]])) if self.task_edges is not None else 0

    def _pbin_one(self, f, r):
        name, e, nb = f
        if name == "len":
            has_empty, edges = e
            dc = float(r["diff_chars"])
            if has_empty:
                return 0 if dc == 0 else 1 + _bin(edges, math.log1p(dc))
            return _bin(edges, math.log1p(dc))
        if name == "lint":
            return e.get(self._lint_cat(r), 0)
        if name.startswith("col:"):
            return _bin(e, float(r[name[4:]]))
        raise ValueError(name)

    def patch_combo(self, r):
        return self.combo_ix[tuple(self._pbin_one(f, r) for f in self.pfeat)]

    # ---- beliefs
    def task_belief(self, r):
        j = self._tbin(r)
        return bin_update(self.prior, self.lT1[j], self.lT0[j]), j

    def patch_update(self, b, combo):
        return bin_update(b, self.pL1[combo], self.pL0[combo])

    def patch_lik(self):
        """(L1, L0) arrays over joint patch bins, or None when there is no informative patch signal."""
        if not self.pfeat:
            return None
        return self.pL1, self.pL0

    def describe(self):
        return dict(task=self.spec.get("task"), K=self.spec.get("K", 4), n_task_bins=self.n_task_bins,
                    task_edges=None if self.task_edges is None else [round(float(x), 4) for x in self.task_edges],
                    lik_task_Y1=[round(float(x), 4) for x in self.lT1], lik_task_Y0=[round(float(x), 4) for x in self.lT0],
                    patch=[f[0] for f in self.pfeat], n_patch_bins=[f[2] for f in self.pfeat],
                    lik_patch=[dict(Y1=[round(float(x), 4) for x in l[0]], Y0=[round(float(x), 4) for x in l[1]]) for l in self.pliks])


class JointSignalModel(SignalModel):
    """Relaxes the naive-Bayes assumptions that ci_check.py rejects:
      * patch-length bin conditional on the task bin:  P(lbin | Y, tbin)   (Laplace over length bins)
      * critic likelihoods conditional on the patch-length bin:  P(c pass | Y, lbin), shrunk towards the cell's
        pooled likelihood with prior strength M = 2 pseudo-patches:  (n_pass + M p_pool) / (n + M).
    spec: task = score column, patch = ('len',) (lint is not used here), K."""
    M = 2.0

    def __init__(self, spec, prior, rows_by_patch, train_ids, likes):
        assert tuple(spec.get("patch", ())) == ("len",)
        super().__init__(spec, prior, rows_by_patch, train_ids)
        train = set(train_ids)
        tr = [r for (i, _), r in rows_by_patch.items() if i in train]
        Y = np.array([int(r["Y"]) for r in tr])
        tb = np.array([self._tbin(r) for r in tr])
        lb = np.array([self._pbin_one(self.pfeat[0], r) for r in tr])
        nL = self.pfeat[0][2]
        self.nL = nL
        self.pl_t = {}           # tbin -> (L1[nL], L0[nL])
        for t in range(self.n_task_bins):
            m = tb == t
            self.pl_t[t] = _lik(lb[m], Y[m], nL)
        self.likes_l = {}        # lbin -> {field: (p1, p0)}
        for l in range(nL):
            d = {}
            for f, L in likes.items():
                vals = np.array([np.nan if r.get(f) in ("", None) else float(r[f]) for r in tr])
                out = []
                for y, pool in ((1, L["P_pass_given_Y1"]), (0, L["P_pass_given_Y0"])):
                    m = (lb == l) & (Y == y) & ~np.isnan(vals)
                    out.append((vals[m].sum() + self.M * pool) / (m.sum() + self.M))
                d[f] = tuple(out)
            self.likes_l[l] = d

    def lbin(self, r):
        return self._pbin_one(self.pfeat[0], r)

    def patch_update_joint(self, b, tbin, l):
        L1, L0 = self.pl_t[tbin]
        return bin_update(b, L1[l], L0[l])

    def describe(self):
        d = super().describe()
        d["joint"] = dict(lik_len_given_task={int(t): dict(Y1=[round(float(x), 4) for x in v[0]], Y0=[round(float(x), 4) for x in v[1]])
                                              for t, v in self.pl_t.items()},
                          critic_likes_by_len={int(l): {f: [round(float(p), 4) for p in v] for f, v in d2.items()}
                                               for l, d2 in self.likes_l.items()})
        return d
