"""Per-task / per-patch starting beliefs on top of diversity_fixes/scripts/engine.py (imported, not copied).

Execution (both controllers):
  episode start      b = b_T (+) s_P(p0)       b_T = prior (+) task signal; (+) = Bayes update by a free signal
  generate, 'calK'   b = push_calK(b) (+) s_P(next)   (base step0_calK: data-fitted fresh-patch kernel)
  generate, 'task'   b = b_T (+) s_P(next)            (base step0_g201: reset to the TASK posterior, not to b0)
  critics            unchanged Bayes updates
Planning (BDP): the DP is the engine's DP (state (t,g,d,a,mask) x belief grid, full within-patch lookahead) with
one change: Q_generate(b) = -c_gen + sum_j P(bin_j | b') V_next(b' (+) bin_j), b' = gen_belief(b), i.e. the DP
takes the expectation over the next patch's free signal. With mode 'task' the DP is solved per distinct b_T
(gen_belief = b_T). With no informative patch signal Q_generate is exactly the engine's.
"""
from __future__ import annotations

import numpy as np

from common import DF  # noqa: F401  (puts diversity_fixes/scripts on sys.path)
from engine import CRIT_FIELD, DP, Greedy, bayes, push


class SigDP(DP):
    def __init__(self, *a, patch_lik=None, **kw):
        super().__init__(*a, **kw)
        self.pl = patch_lik

    def gen_value(self, Vn, b):
        bp = self.gen_belief(b)
        if self.pl is None:
            return Vn[self.nn(bp)]
        bp = np.broadcast_to(np.asarray(bp, float), np.shape(b)).astype(float)
        L1, L0 = self.pl
        tot = np.zeros_like(bp)
        for l1, l0 in zip(L1, L0):
            pj = bp * l1 + (1 - bp) * l0
            if l1 == l0:
                bj = bp
            else:
                bj = (l1 * bp) / np.maximum(pj, 1e-12)
            tot = tot + pj * Vn[self.nn(bj)]
        return tot

    # copy of engine.DP._solve with the generate line routed through gen_value
    def _solve(self, s):
        t, g, d, a, m = s
        fl, cm, B = self.fl, self.cost, self.B
        H = fl.H
        Q = {}
        Q["verify"] = cm.reward * B - cm.c_ver
        Q["give_up"] = np.zeros_like(B)
        if self.regen_allowed(g):
            if t + 1 >= H:
                Q["generate"] = np.full_like(B, -cm.c_gen + 0.0)
            else:
                Vn = self.V(self.state_after_regen(t, g))[0]
                Q["generate"] = -cm.c_gen + self.gen_value(Vn, B)
        for c in self.critics:
            if fl.f0b and c in m:
                continue
            p1, p0 = (self.likes_rep if d >= 1 else self.likes)[c]
            p_pass = p1 * B + p0 * (1 - B)
            bp = bayes(B, p1, p0, True)
            bf = bayes(B, p1, p0, False)
            if fl.lookahead == "paper1":
                ba = lambda bn: self._best_after_paper(bn, t, g)
            else:
                Vc = self.V((t, g, d, a, m | {c}))[0]
                ba = lambda bn, Vc=Vc: Vc[self.nn(bn)]
            Q[c] = -cm.critic(c) + p_pass * ba(bp) + (1 - p_pass) * ba(bf)
        assert not (fl.repair and a == 1), "repair not supported in per_task_prior"
        names = [x for x in self.actions if x in Q]
        M = np.vstack([Q[x] for x in names])
        best = M.argmax(0)
        return M[best, np.arange(len(B))], names, M

    def _best_after_paper(self, bn, t, g):
        cm = self.cost
        opts = [cm.reward * bn - cm.c_ver, np.zeros_like(bn)]
        if t + 1 < self.fl.H and self.regen_allowed(g):
            Vn = self.V(self.state_after_regen(t, g))[0]
            opts.append(-cm.c_gen + self.gen_value(Vn, bn))
        return np.max(np.vstack(opts), 0)


class TaskInfo:
    """Per-episode free information: b_T, task bin, joint patch bin of each cached patch, b0 of each patch."""
    def __init__(self, bT, tbin, pcombos, sm):
        self.bT, self.tbin, self.pcombos = bT, tbin, pcombos
        self.b0 = [sm.patch_update(bT, c) for c in pcombos]


def _next_belief(b, S, info, sm, gen_belief):
    bp = gen_belief(b)
    k = S["patch_idx"] + 1
    if k < len(info.pcombos):
        return sm.patch_update(bp, info.pcombos[k])
    return bp


def make_sig_dp_policy(get_dp, infos, sm):
    """get_dp(info) -> SigDP planned for this task (one per distinct b_T in mode 'task', shared in mode 'calK')."""
    def _p(S, rec):
        info = infos[S["_ep"].iid]
        dp = get_dp(info)
        if "belief" not in S:
            S["belief"] = info.b0[0]
        b = S["belief"]
        a = dp.select(b, S)
        if a in CRIT_FIELD:
            p1, p0 = dp.likes[a]
            S["belief"] = bayes(b, p1, p0, bool(rec.get(CRIT_FIELD[a])))
        elif a == "generate":
            S["belief"] = _next_belief(b, S, info, sm, dp.gen_belief)
        return a
    return _p


def make_sig_greedy_policy(get_gr, infos, sm):
    """BG with the same free signals. Q_generate uses E[b_next] = gen_belief(b) (martingale over the next bin)."""
    def _p(S, rec):
        info = infos[S["_ep"].iid]
        gr = get_gr(info)
        if "belief" not in S:
            S["belief"] = info.b0[0]
        b = S["belief"]
        a = gr.select(b, S)
        if a in CRIT_FIELD:
            p1, p0 = gr.likes[a]
            S["belief"] = bayes(b, p1, p0, bool(rec.get(CRIT_FIELD[a])))
        elif a == "generate":
            if gr.fl.calK:
                g = lambda x: push(x, gr.calk["P_fix"], gr.calk["P_break"])
            elif gr.fl.f0a or gr.fl.f0c:
                g = lambda x: gr.prior
            else:
                g = lambda x: x
            S["belief"] = _next_belief(b, S, info, sm, g)
        return a
    return _p


def make_threshold_rule(fn_hi, fn_lo, s_of, tau):
    """Signal-thresholded rule: at the start of every cached patch k, follow rule `fn_hi` for that patch if
    s(task, k) >= tau else `fn_lo`. s is the free-signal posterior of the patch (b_T (+) s_P(k))."""
    def _p(S, rec):
        k = S["patch_idx"]
        if S.get("_thr_k") != k:
            S["_thr_k"] = k
            S["_thr_fn"] = fn_hi if s_of(S["_ep"].iid, k) >= tau else fn_lo
        return S["_thr_fn"](S, rec)
    return _p


__all__ = ["SigDP", "Greedy", "TaskInfo", "make_sig_dp_policy", "make_sig_greedy_policy", "make_threshold_rule"]


# ----------------------------------------------------------------------------------------------
# Joint variant: critic likelihoods depend on the current patch's length bin l, and the next patch's l' is drawn
# from P(l' | Y, task bin). DP state (t, g, l, mask) x belief grid, full within-patch lookahead (needs f0b).
# One JointDP per task bin (and per b_T in mode 'task').
# ----------------------------------------------------------------------------------------------
class JointDP(DP):
    def __init__(self, prior, likes, gen_kernel, cost, flags, *, critics, cal_kernel, jsm, tbin):
        super().__init__(prior, likes, gen_kernel, cost, flags, critics=critics, cal_kernel=cal_kernel)
        assert flags.f0b and flags.lookahead == "full" and not flags.repair
        self.jsm, self.tbin = jsm, tbin
        self.likes_l = {l: {c: v[CRIT_FIELD[c]] for c in self.critics} for l, v in jsm.likes_l.items()}
        self.L1, self.L0 = jsm.pl_t[tbin]

    def gen_value(self, t, g, b):
        bp = np.broadcast_to(np.asarray(self.gen_belief(b), float), np.shape(b)).astype(float)
        tot = np.zeros_like(bp)
        for l, (l1, l0) in enumerate(zip(self.L1, self.L0)):
            pj = bp * l1 + (1 - bp) * l0
            bj = bp if l1 == l0 else (l1 * bp) / np.maximum(pj, 1e-12)
            Vn = self.V((t + 1, g + 1, l, frozenset()))[0]
            tot = tot + pj * Vn[self.nn(bj)]
        return tot

    def _solve(self, s):
        t, g, l, m = s
        cm, B = self.cost, self.B
        Q = {"verify": cm.reward * B - cm.c_ver, "give_up": np.zeros_like(B)}
        if self.regen_allowed(g):
            Q["generate"] = (np.full_like(B, -cm.c_gen + 0.0) if t + 1 >= self.fl.H
                             else -cm.c_gen + self.gen_value(t, g, B))
        for c in self.critics:
            if c in m:
                continue
            p1, p0 = self.likes_l[l][c]
            p_pass = p1 * B + p0 * (1 - B)
            Vc = self.V((t, g, l, m | {c}))[0]
            Q[c] = -cm.critic(c) + p_pass * Vc[self.nn(bayes(B, p1, p0, True))] + (1 - p_pass) * Vc[self.nn(bayes(B, p1, p0, False))]
        names = [x for x in self.actions if x in Q]
        M = np.vstack([Q[x] for x in names])
        best = M.argmax(0)
        return M[best, np.arange(len(B))], names, M

    def select(self, b, S):
        s = (S["t"], S["g"], S["_lbin"], S["mask"])
        _, names, M = self.V(s)
        col = M[:, int(self.nn(b))]
        j = max(range(len(names)), key=lambda j: (col[j], -j))
        return names[j]


class JointInfo:
    def __init__(self, bT, tbin, lbins, jsm):
        self.bT, self.tbin, self.lbins = bT, tbin, lbins
        self.pcombos = lbins
        self.b0 = [jsm.patch_update_joint(bT, tbin, l) for l in lbins]


def make_joint_dp_policy(get_dp, infos, jsm):
    def _p(S, rec):
        info = infos[S["_ep"].iid]
        dp = get_dp(info)
        k = S["patch_idx"]
        S["_lbin"] = info.lbins[k]
        if "belief" not in S:
            S["belief"] = info.b0[0]
        b = S["belief"]
        a = dp.select(b, S)
        if a in CRIT_FIELD:
            p1, p0 = dp.likes_l[S["_lbin"]][a]
            S["belief"] = bayes(b, p1, p0, bool(rec.get(CRIT_FIELD[a])))
        elif a == "generate":
            bp = dp.gen_belief(b)
            S["belief"] = jsm.patch_update_joint(bp, info.tbin, info.lbins[k + 1]) if k + 1 < len(info.lbins) else bp
        return a
    return _p


def make_joint_greedy_policy(get_gr, infos, jsm):
    def _p(S, rec):
        info = infos[S["_ep"].iid]
        gr = get_gr(info)
        k = S["patch_idx"]
        l = info.lbins[k]
        gr.likes = {c: jsm.likes_l[l][CRIT_FIELD[c]] for c in gr.critics}
        if "belief" not in S:
            S["belief"] = info.b0[0]
        b = S["belief"]
        a = gr.select(b, S)
        if a in CRIT_FIELD:
            p1, p0 = gr.likes[a]
            S["belief"] = bayes(b, p1, p0, bool(rec.get(CRIT_FIELD[a])))
        elif a == "generate":
            bp = push(b, gr.calk["P_fix"], gr.calk["P_break"]) if gr.fl.calK else gr.prior
            S["belief"] = jsm.patch_update_joint(bp, info.tbin, info.lbins[k + 1]) if k + 1 < len(info.lbins) else bp
        return a
    return _p
