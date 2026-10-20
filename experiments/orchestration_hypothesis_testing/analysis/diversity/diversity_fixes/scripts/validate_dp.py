"""Internal-consistency check of the DP: simulate episodes drawn from the controller's OWN generative model
(Y ~ belief model, critic verdicts ~ fitted likelihoods, regenerate ~ planning transition, repair ~ fitted
kernel, chain continuation ~ rho) and compare the mean utility of the DP policy with its root value V(root).
A gap beyond Monte-Carlo noise + grid error would mean planning and execution are inconsistent."""
import sys
sys.dont_write_bytecode = True
import json, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import CRIT_FIELD, DP, Cost, Flags, bayes, push, repair_update
from protocol import REGIMES
from runner import _cell, cal_kernel, chain_data
from variants import VARIANTS

rng = np.random.default_rng(0)


def model_episode(dp, fl, R):
    """Draw one episode from the planner's model; return realised utility."""
    cm = dp.cost
    p_y = dp.prior
    y = int(rng.random() < p_y)
    b = dp.prior
    S = dict(t=0, g=0, d=0 if (fl.repair and R) else -1, mask=frozenset(), avail_repair=bool(fl.repair and R),
             avail_regen=True)
    if fl.repair and R:      # root availability ~ rho(0) under the model
        S["avail_repair"] = bool(rng.random() < R["rho"].get(0, 0.0))
    seen = {}
    u = 0.0
    for _ in range(64):
        S["avail_regen"] = (S["g"] + 1 < fl.n_fresh) if fl.budget else True
        a = dp.select(b, S)
        if a == "verify":
            return u - cm.c_ver + cm.reward * y
        if a == "give_up":
            return u
        if a == "generate":
            u -= cm.c_gen
            if not (S["t"] + 1 < fl.H):
                return u
            if fl.calK:
                py = dp.calk["P_fix"] if y == 0 else 1 - dp.calk["P_break"]
            elif fl.f0c:
                py = dp.prior
            else:
                py = dp.pf if y == 0 else 1 - dp.pb
            y = int(rng.random() < py); b = float(dp.gen_belief(b))
            S.update(t=S["t"] + 1, g=S["g"] + 1, d=-1, mask=frozenset(), avail_repair=False)
            continue
        if a == "repair":
            u -= cm.c_rep
            py = R["P_fix"] if y == 0 else 1 - R["P_break"]
            y = int(rng.random() < py); b = float(push(b, R["P_fix"], R["P_break"]))
            S.update(t=S["t"] + 1, d=S["d"] + 1, mask=frozenset())
            S["avail_repair"] = bool(rng.random() < R["rho"].get(S["d"], 0.0))
            continue
        p1, p0 = dp.likes[a]
        z = rng.random() < (p1 if y == 1 else p0)
        u -= cm.critic(a)
        b = float(bayes(b, p1, p0, bool(z)))
        S["mask"] = S["mask"] | {a}
    return u


out = {}
for vname in ["step0", "step0_g201", "step1_H5", "step1mx_H6", "step1mxK_H5"]:
    V = VARIANTS[vname]
    for (bench, gen) in [("lcb_medium", "gpt5_mini"), ("humanevalfix", "gpt5_mini"), ("codecontests", "qwen3_coder")]:
        C, cell = _cell(bench, gen, False)
        cd = chain_data(bench, gen, *V["chains"]) if V.get("chains") else None
        R = cd["rep"] if cd else None
        fl = Flags(**V["flags"])
        cost = Cost(**REGIMES["fast"], c_rep=V.get("c_rep", 10.0))
        dp = DP(cell["prior"], cell["likes"]["critic_likelihoods"], cell["kernel"], cost, fl, rep=R,
                critics=["L0", "L2", "L3"], cal_kernel=cal_kernel(cell))
        # root value: mixture over root repair availability (model rho(0))
        i0 = int(dp.nn(dp.prior))
        if fl.repair and R:
            r0 = R["rho"].get(0, 0.0)
            v_root = r0 * dp.V((0, 0, 0, 1, frozenset()))[0][i0] + (1 - r0) * dp.V((0, 0, 0, 0, frozenset()))[0][i0]
        else:
            v_root = dp.V((0, 0, -1, 0, frozenset()))[0][i0]
        sims = np.array([model_episode(dp, fl, R) for _ in range(20000)])
        se = sims.std() / np.sqrt(len(sims))
        out[f"{vname}/{bench}/{gen}"] = dict(V_root=round(float(v_root), 3), mc_mean=round(float(sims.mean()), 3), mc_se=round(float(se), 3))
        print(vname, bench, gen, out[f"{vname}/{bench}/{gen}"])
json.dump(out, open(sys.argv[1], "w"), indent=1)
