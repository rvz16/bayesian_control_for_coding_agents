"""Replay engine for BG / BDP with independently switchable evidence fixes (Step 0) and a data-backed
repair action (Step 1).

With every flag off this engine is a re-implementation of the paper code at d8876f5:
  analysis/controller.py  BayesianController._value/_build/select_action (L105-170),
                          simulate_policy (L177-229), make_bayesian_policy (L300-320),
                          policy_always_verify, policy_fixed_pipeline
  analysis/lcb_compare.py GreedyController (L56-105), make_greedy_policy (L108-125)
and the regression gate (run_gate.py) checks it reproduces action_traces/traces.jsonl exactly.

Flags
  f0a  BG: on `generate` (fresh cached patch) the belief resets to the prior b0 (paper BG keeps the stale
       post-FAIL belief, lcb_compare.py:108-124).
  f0b  A critic already observed on the current patch is masked until the patch changes. The DP state is
       (t, g, d, avail, mask, b): the critic mask is part of the state, a critic's continuation value is
       V(same t, mask + {critic}) (full within-patch DP), and a masked critic is never an action.
       `lookahead='paper1'` keeps the paper's one-critic lookahead (after a critic only verify / give_up /
       generate are considered, controller.py:127-134) and only restricts the action set by the mask.
  f0c  `generate` returns an independent fresh cached patch, so the belief resets to b0 in execution AND in
       the DP's planning transition for generate (paper propagates the old belief through the refinement
       kernel, controller.py:100-103, 116-118, 130, 317-318). For BG f0c == f0a.
  repair  Step 1: action `repair` moves to the next recorded chain step (only where the chain is rooted on
       the current patch and has a next step). Belief: kernel push b' = b(1-pb) + (1-b)pf with the
       train-split chain kernel ('kernel' model), or ('obs' model) the agent also observes whether the
       patch changed: P(same|Y) Bayes update with Y'=Y if unchanged, changed-only kernel if changed.
  calK  (diagnostic, not one of the requested fixes) regenerate moves the belief through a kernel fitted on
       consecutive calibration patches of the same train instance, P(Y_k+1 | Y_k), in planning and execution
       (BG: belief push and Q_generate = -c_gen + R*push(b) - c_ver). Fresh patches of one instance are
       strongly correlated, so reset-to-b0 (f0c) overvalues regeneration after a FAIL.
  rep_likes  (step-1 option) critic likelihoods on a repaired patch (chain depth d >= 1) are fitted on the
       train-split rooted chain steps d >= 1 instead of on calibration patches. Repaired patches are tuned
       against the visible feedback, so e.g. P(L2 pass | Y=0) is much higher on them than on fresh patches.
  budget  regenerate is unavailable when no fresh cached patch remains (both controllers know the cached
       budget); paper semantics (off) let `generate` run past the last patch and end the episode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CRIT_FIELD = {"L0": "L0_syntax", "L1": "L1_lint", "L2": "L2_public_tests", "L3": "L3_llm_review"}
CRIT_ORDER = ["L0", "L2", "L3", "L1"]          # paper order L0, L2, L3 (controller.py:140-145); L1 last


@dataclass
class Cost:
    c_gen: float = 10.0
    c_L0: float = 1.0
    c_L1: float = 1.0
    c_L2: float = 1.0
    c_L3: float = 1.0
    c_ver: float = 5.0
    reward: float = 100.0
    c_rep: float = 10.0

    def critic(self, c):
        return getattr(self, f"c_{c}")


@dataclass
class Flags:
    f0a: bool = False
    f0b: bool = False
    f0c: bool = False
    lookahead: str = "paper1"       # 'paper1' | 'full' (full requires f0b)
    repair: bool = False
    repair_model: str = "kernel"    # 'kernel' | 'obs'
    budget: bool = False
    calK: bool = False              # diagnostic: regenerate kernel fitted on consecutive CALIBRATION patches
    rep_likes: bool = False         # critic likelihoods on repaired patches fitted on train chain steps >= 1
    H: int = 3
    n_b: int = 51
    n_fresh: int = 3


# ----------------------------------------------------------------------------------------------
# Episode data
# ----------------------------------------------------------------------------------------------
class Episode:
    def __init__(self, iid, fresh, chain=None, ident=None):
        self.iid = iid
        self.fresh = fresh            # calibration records p0, p1, p2 (sorted by patch_id)
        self.chain = chain            # rooted chain on fresh[0] (chain[0] is p0) or None
        self.ident = ident

    def rec(self, k, d):
        return self.fresh[k] if d <= 0 else self.chain[d]

    def chain_len(self, k):
        return len(self.chain) if (self.chain is not None and k == 0) else 0

    def repair_changed(self, d):
        return self.ident(self.chain[d + 1]) != self.ident(self.chain[d])


# ----------------------------------------------------------------------------------------------
# Simulator (controller.py:simulate_policy L177-229 + repair; trace outputs as in action_traces)
# ----------------------------------------------------------------------------------------------
def simulate(ep: Episode, policy_fn, cost: Cost, flags: Flags, *, max_actions=64, prior=None):
    n = len(ep.fresh)
    S = {"step": 0, "patch_idx": 0, "max_patches": n, "verified": False, "given_up": False, "actions": [],
         # extended state
         "t": 0, "g": 0, "d": 0 if ep.chain_len(0) >= 1 else -1, "mask": frozenset(), "_ep": ep,
         "_flags": flags}
    _avail(S)
    beliefs, pids, obs = [], [], []
    cum_cost = 0.0
    reward = 0.0
    while not (S["verified"] or S["given_up"]) and S["step"] < n:
        if len(S["actions"]) >= max_actions:
            S["given_up"] = True
            S["actions"].append("give_up_cap")
            beliefs.append(S.get("belief", prior)); pids.append(_pid(S, flags)); obs.append(None)
            break
        k, d = S["patch_idx"], S["d"]
        rec = ep.rec(k, d)
        pid = _pid(S, flags)
        action = policy_fn(S, rec)
        S["actions"].append(action)
        o = None
        if action == "verify":
            cum_cost += cost.c_ver
            reward = cost.reward * rec["Y"]
            S["verified"] = True
            o = int(rec["Y"])
        elif action == "give_up":
            S["given_up"] = True
        elif action == "generate":
            if flags.budget:
                assert S["avail_regen"], "regenerate chosen with no fresh patch left"
            cum_cost += cost.c_gen
            S["patch_idx"] += 1
            S["step"] += 1
            S["t"] += 1; S["g"] += 1; S["d"] = -1; S["mask"] = frozenset()
            if S["patch_idx"] >= n:
                S["given_up"] = True
        elif action == "repair":
            assert S["avail_repair"], "repair chosen without a recorded next chain step"
            cum_cost += cost.c_rep
            changed = ep.repair_changed(d)
            S["d"] = d + 1; S["t"] += 1
            if changed or not flags.f0b:
                S["mask"] = frozenset()
            o = "chg" if changed else "same"
        elif action in CRIT_FIELD:
            if flags.f0b:
                assert action not in S["mask"], f"masked critic {action} re-queried"
                S["mask"] = S["mask"] | {action}
            cum_cost += cost.critic(action)
            o = bool(rec.get(CRIT_FIELD[action]))
        else:
            raise ValueError(action)
        _avail(S)
        beliefs.append(S.get("belief", prior)); pids.append(pid); obs.append(o)
    acts = S["actions"]
    return {"reward": reward, "cost": cum_cost, "utility": reward - cum_cost, "verified": S["verified"],
            "n_actions": len(acts), "actions": list(acts), "beliefs": beliefs, "patch_idx": pids, "obs": obs,
            "exhausted": (not S["verified"]) and bool(acts) and acts[-1] == "generate"}


def _pid(S, flags):
    if not flags.repair:
        return S["patch_idx"]
    return f"{S['patch_idx']}" + (f"r{S['d']}" if S["d"] >= 1 else "")


def _avail(S):
    ep, fl = S["_ep"], S["_flags"]
    S["avail_repair"] = bool(fl.repair and S["d"] >= 0 and S["d"] + 1 < ep.chain_len(S["patch_idx"]))
    S["avail_regen"] = (S["patch_idx"] + 1 < S["max_patches"]) if fl.budget else True


# ----------------------------------------------------------------------------------------------
# Bayes helpers (same arithmetic order as controller.py:90-103 so flags-off is bit-identical)
# ----------------------------------------------------------------------------------------------
def bayes(b, p1, p0, passed):
    if passed:
        num = p1 * b
        den = num + p0 * (1 - b)
    else:
        num = (1 - p1) * b
        den = num + (1 - p0) * (1 - b)
    return num / np.maximum(den, 1e-12) if isinstance(b, np.ndarray) else num / max(den, 1e-12)


def push(b, pf, pb):
    return b * (1 - pb) + (1 - b) * pf


# ----------------------------------------------------------------------------------------------
# DP controller
# ----------------------------------------------------------------------------------------------
class DP:
    """Backward induction over (t, g, d, avail, mask) x belief grid, memoised.
    t = patch-producing actions so far (paper `step`), g = regenerations so far, d = chain depth of the
    current patch (-1 = fresh patch without chain), avail = repair available on the current patch."""

    def __init__(self, prior, likes, gen_kernel, cost: Cost, flags: Flags, rep=None, critics=None, cal_kernel=None):
        self.prior = prior
        self.calk = cal_kernel
        self.likes = {c: (likes[CRIT_FIELD[c]]["P_pass_given_Y1"], likes[CRIT_FIELD[c]]["P_pass_given_Y0"])
                      for c in CRIT_ORDER if CRIT_FIELD[c] in likes and (critics is None or c in critics)}
        self.critics = [c for c in CRIT_ORDER if c in self.likes]
        self.likes_rep = _rep_likes(self.likes, rep, flags)
        self.pf = gen_kernel["P_fix_given_broken"]; self.pb = gen_kernel["P_break_given_correct"]
        self.cost, self.fl, self.rep = cost, flags, rep
        if flags.lookahead == "full":
            assert flags.f0b, "full within-patch lookahead needs the critic mask (f0b)"
        self.B = np.linspace(0.0, 1.0, flags.n_b)
        self.actions = ["verify", "give_up", "generate"] + self.critics + (["repair"] if flags.repair else [])
        self.memo = {}

    # nearest grid point; identical to int(np.argmin(np.abs(self.beliefs - x))) incl. ties -> lower index
    def nn(self, x):
        B = self.B
        x = np.asarray(x, float)
        j = np.clip(np.searchsorted(B, x, side="left"), 1, len(B) - 1)
        lo, hi = j - 1, j
        pick_lo = np.abs(B[lo] - x) <= np.abs(B[hi] - x)
        return np.where(pick_lo, lo, hi)

    def gen_belief(self, b):
        if self.fl.calK:
            return push(b, self.calk["P_fix"], self.calk["P_break"])
        if self.fl.f0c:
            return np.full_like(b, self.prior) if isinstance(b, np.ndarray) else self.prior
        return push(b, self.pf, self.pb)

    def regen_allowed(self, g):
        return (g + 1 < self.fl.n_fresh) if self.fl.budget else True

    def state_after_regen(self, t, g):
        return (t + 1, g + 1, -1, 0, frozenset())

    def V(self, s):
        if s not in self.memo:
            self.memo[s] = self._solve(s)
        return self.memo[s]

    def _solve(self, s):
        t, g, d, a, m = s
        fl, cm, B = self.fl, self.cost, self.B
        H = fl.H
        Q = {}
        Q["verify"] = cm.reward * B - cm.c_ver
        Q["give_up"] = np.zeros_like(B)
        can_regen = self.regen_allowed(g)
        if can_regen:
            if t + 1 >= H:
                Q["generate"] = np.full_like(B, -cm.c_gen + 0.0)
            else:
                Vn = self.V(self.state_after_regen(t, g))[0]
                Q["generate"] = -cm.c_gen + Vn[self.nn(self.gen_belief(B))]
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
        if fl.repair and a == 1 and t + 1 < H:
            Q["repair"] = self._q_repair(B, t, g, d, m)
        names = [x for x in self.actions if x in Q]
        M = np.vstack([Q[x] for x in names])
        best = M.argmax(0)
        return M[best, np.arange(len(B))], names, M

    def _best_after_paper(self, bn, t, g):
        cm = self.cost
        opts = [cm.reward * bn - cm.c_ver, np.zeros_like(bn)]
        if t + 1 < self.fl.H and self.regen_allowed(g):
            Vn = self.V(self.state_after_regen(t, g))[0]
            opts.append(-cm.c_gen + Vn[self.nn(self.gen_belief(bn))])
        return np.max(np.vstack(opts), 0)

    def _q_repair(self, B, t, g, d, m):
        cm, R = self.cost, self.rep
        if self.fl.repair_model == "kernel":
            rho = R["rho"].get(d + 1, 0.0)
            nb = push(B, R["P_fix"], R["P_break"])
            idx = self.nn(nb)
            v1 = self.V((t + 1, g, d + 1, 1, frozenset()))[0][idx] if rho > 0 else 0.0
            v0 = self.V((t + 1, g, d + 1, 0, frozenset()))[0][idx]
            return -cm.c_rep + rho * v1 + (1 - rho) * v0
        q1, q0 = R["q_same_Y1"], R["q_same_Y0"]
        ps = B * q1 + (1 - B) * q0
        b_same = (B * q1) / np.maximum(ps, 1e-12)
        b_chg = (B * (1 - q1) * (1 - R["P_break_chg"]) + (1 - B) * (1 - q0) * R["P_fix_chg"]) / np.maximum(1 - ps, 1e-12)
        out = -cm.c_rep + np.zeros_like(B)
        for kind, pk, nb, mask in (("same", ps, b_same, m), ("chg", 1 - ps, b_chg, frozenset())):
            rho = R["rho_kind"][kind].get(d + 1, 0.0)
            idx = self.nn(nb)
            v1 = self.V((t + 1, g, d + 1, 1, mask))[0][idx] if rho > 0 else 0.0
            v0 = self.V((t + 1, g, d + 1, 0, mask))[0][idx]
            out = out + pk * (rho * v1 + (1 - rho) * v0)
        return out

    def select(self, b, S):
        fl = self.fl
        m = S["mask"] if fl.f0b else frozenset()
        a = 1 if S.get("avail_repair") else 0
        d = S["d"] if fl.repair else -1
        s = (S["t"], S["g"], d, a, m)
        _, names, M = self.V(s)
        i = int(self.nn(b))
        col = M[:, i]
        ok = [j for j, x in enumerate(names) if self._available(x, S)]
        j = max(ok, key=lambda j: (col[j], -j))
        return names[j]

    def _available(self, x, S):
        if x == "generate" and self.fl.budget:
            return S["avail_regen"]
        if x == "repair":
            return S["avail_repair"]
        return True


def _rep_likes(likes, rep, flags):
    """Likelihoods used on repaired patches: chain-fitted (train steps >= 1) when flags.rep_likes, else the
    calibration ones. A critic without chain data keeps its calibration likelihood."""
    if not (flags.rep_likes and rep and rep.get("likes_rep")):
        return likes
    return {c: rep["likes_rep"].get(CRIT_FIELD[c], v) for c, v in likes.items()}


def make_dp_policy(dp: DP):
    def _p(S, rec):
        b = S.get("belief", dp.prior)
        a = dp.select(b, S)
        if a in CRIT_FIELD:
            p1, p0 = (dp.likes_rep if (dp.fl.repair and S["d"] >= 1) else dp.likes)[a]
            S["belief"] = bayes(b, p1, p0, bool(rec.get(CRIT_FIELD[a])))
        elif a == "generate":
            S["belief"] = dp.gen_belief(b)
        elif a == "repair":
            S["belief"] = repair_update(b, dp.rep, dp.fl, S)
        return a
    return _p


def repair_update(b, R, fl, S):
    if fl.repair_model == "kernel":
        return push(b, R["P_fix"], R["P_break"])
    changed = S["_ep"].repair_changed(S["d"])
    q1, q0 = R["q_same_Y1"], R["q_same_Y0"]
    if not changed:
        return b * q1 / max(b * q1 + (1 - b) * q0, 1e-12)
    den = b * (1 - q1) + (1 - b) * (1 - q0)
    return (b * (1 - q1) * (1 - R["P_break_chg"]) + (1 - b) * (1 - q0) * R["P_fix_chg"]) / max(den, 1e-12)


# ----------------------------------------------------------------------------------------------
# Greedy controller (lcb_compare.py:56-125) + fixes
# ----------------------------------------------------------------------------------------------
class Greedy:
    def __init__(self, prior, likes, cost: Cost, flags: Flags, rep=None, critics=None, cal_kernel=None):
        self.prior = prior
        self.calk = cal_kernel
        self.likes = {c: (likes[CRIT_FIELD[c]]["P_pass_given_Y1"], likes[CRIT_FIELD[c]]["P_pass_given_Y0"])
                      for c in CRIT_ORDER if CRIT_FIELD[c] in likes and (critics is None or c in critics)}
        self.critics = [c for c in CRIT_ORDER if c in self.likes]
        self.likes_rep = _rep_likes(self.likes, rep, flags)
        self.cost, self.fl, self.rep = cost, flags, rep

    def select(self, b, S):
        cm, fl = self.cost, self.fl
        Q = {"verify": cm.reward * b - cm.c_ver, "give_up": 0.0}
        if not fl.budget or S["avail_regen"]:
            if fl.calK:
                Q["generate"] = -cm.c_gen + cm.reward * push(b, self.calk["P_fix"], self.calk["P_break"]) - cm.c_ver
            else:
                Q["generate"] = -cm.c_gen + cm.reward * self.prior - cm.c_ver
        for c in self.critics:
            if fl.f0b and c in S["mask"]:
                continue
            p1, p0 = (self.likes_rep if (fl.repair and S["d"] >= 1) else self.likes)[c]
            p_pass = p1 * b + p0 * (1 - b)
            v_pass = max(cm.reward * bayes(b, p1, p0, True) - cm.c_ver, 0.0)
            v_fail = max(cm.reward * bayes(b, p1, p0, False) - cm.c_ver, 0.0)
            Q[c] = -cm.critic(c) + p_pass * v_pass + (1 - p_pass) * v_fail
        if fl.repair and S["avail_repair"]:
            R = self.rep
            if fl.repair_model == "kernel":
                py = push(b, R["P_fix"], R["P_break"])
            else:
                q1, q0 = R["q_same_Y1"], R["q_same_Y0"]
                py = b * q1 + b * (1 - q1) * (1 - R["P_break_chg"]) + (1 - b) * (1 - q0) * R["P_fix_chg"]
            Q["repair"] = -cm.c_rep + cm.reward * py - cm.c_ver
        return max(Q, key=lambda a: Q[a])


def make_greedy_policy(gr: Greedy):
    def _p(S, rec):
        b = S.get("belief", gr.prior)
        a = gr.select(b, S)
        if a in CRIT_FIELD:
            p1, p0 = (gr.likes_rep if (gr.fl.repair and S["d"] >= 1) else gr.likes)[a]
            S["belief"] = bayes(b, p1, p0, bool(rec.get(CRIT_FIELD[a])))
        elif a == "generate" and gr.fl.calK:
            S["belief"] = push(b, gr.calk["P_fix"], gr.calk["P_break"])
        elif a == "generate" and (gr.fl.f0a or gr.fl.f0c):
            S["belief"] = gr.prior
        elif a == "repair":
            S["belief"] = repair_update(b, gr.rep, gr.fl, S)
        return a
    return _p


# ----------------------------------------------------------------------------------------------
# Rules. Paper rules are re-implemented with the same state keys as action_traces/replay_traces.py.
# ----------------------------------------------------------------------------------------------
def policy_always_verify(S, rec):
    return "verify"


def policy_never_submit(S, rec):
    return "give_up"


def make_gate(critic, on_fail="regen"):
    f, key = CRIT_FIELD[critic], f"_gate_{critic}_done"

    def _p(S, rec):
        if not S.get(key):
            S[key] = True
            return critic
        S[key] = False
        if rec.get(f):
            return "verify"
        if on_fail == "give_up":
            return "give_up"
        if S["patch_idx"] + 1 < S.get("max_patches", 3):
            return "generate"
        return "give_up"
    return _p


def policy_fixed_pipeline(S, rec):           # controller.py:276-289
    if not S.get("L0_done"):
        S["L0_done"] = True
        return "L0"
    if not S.get("L3_done"):
        S["L3_done"] = True
        return "L3"
    S["L0_done"] = False
    S["L3_done"] = False
    if rec.get("L0_syntax") and rec.get("L3_llm_review"):
        return "verify"
    if S["patch_idx"] + 1 < S.get("max_patches", 5):
        return "generate"
    return "give_up"


def make_gate_repair(critic, at_end):
    """Run `critic` on the current patch (once per distinct patch content); PASS -> verify;
    FAIL -> repair while a recorded next chain step exists; at chain end (or no chain):
    at_end = 'verify' | 'give_up' | 'regen' (fresh patch if one remains, else give_up)."""
    f = CRIT_FIELD[critic]

    def _p(S, rec):
        key = (S["patch_idx"], S["d"])
        known = S.setdefault("_gr_known", {})
        if S.pop("_gr_same", False):          # last repair returned the identical patch: verdict is known
            known[key] = known.get(S.pop("_gr_prev"), None)
        if known.get(key) is None:
            known[key] = bool(rec.get(f))
            return critic
        if known[key]:
            return "verify"
        if S["avail_repair"]:
            if not S["_ep"].repair_changed(S["d"]):
                S["_gr_same"] = True; S["_gr_prev"] = key
            return "repair"
        if at_end == "verify":
            return "verify"
        if at_end == "regen" and S["patch_idx"] + 1 < S["max_patches"]:
            return "generate"
        return "give_up"
    return _p


def make_repair_k(k):
    """Repair k times (fewer if the chain ends), then verify."""
    def _p(S, rec):
        n = S.setdefault("_rk", 0)
        if n < k and S["avail_repair"]:
            S["_rk"] = n + 1
            return "repair"
        return "verify"
    return _p
