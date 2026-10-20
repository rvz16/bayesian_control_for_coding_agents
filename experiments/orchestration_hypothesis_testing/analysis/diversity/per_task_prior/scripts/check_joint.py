"""Unit check: JointDP with non-informative joint pieces (every length bin carries the cell's pooled critic
likelihoods, P(len bin | Y, task bin) identical for Y=0/1) must act exactly like the engine's DP (base calK, no
signal) on every eval episode of every cell, paper regime. usage: check_joint.py OUT_JSON"""
import sys
sys.dont_write_bytecode = True
import json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import gname
from cells import cell_keys
from engine import DP, make_dp_policy
from protocol import paper_regime
from runner import cal_kernel, make_policies, prepare, run
from prior_model import JointSignalModel
from ptp_engine import JointDP, JointInfo, make_joint_dp_policy
OUT = sys.argv[1]
sys.argv = [sys.argv[0], os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')]
import run_ptp  # noqa: E402

bad = n = 0
for b, g in cell_keys():
    V = dict(flags=run_ptp.BASES["calK"], L1=False)
    C, cell, cd, eps = prepare(b, g, V)
    rows = run_ptp.sigrows()[(b, gname(g))]
    likes = cell["likes"]["critic_likelihoods"]
    jsm = JointSignalModel(dict(task="combtask", patch=("len",), K=4), cell["prior"], rows, cell["train_ids"], likes)
    for l in jsm.likes_l:                      # neutralise
        jsm.likes_l[l] = {f: (v["P_pass_given_Y1"], v["P_pass_given_Y0"]) for f, v in likes.items()}
    for t in jsm.pl_t:
        u = np.full(jsm.nL, 1.0 / jsm.nL); jsm.pl_t[t] = (u, u.copy())
    infos = {i: JointInfo(cell["prior"], 0, [jsm.lbin(rows[(i, k)]) for k in range(len(ep.fresh))], jsm) for i, ep in eps.items()}
    regime = paper_regime(b)
    _, cost, fl = make_policies(cell, cd, b, regime, V, only_rules=True)
    ck = cal_kernel(cell)
    jdp = JointDP(cell["prior"], likes, cell["kernel"], cost, fl, critics=["L0", "L2", "L3"], cal_kernel=ck, jsm=jsm, tbin=0)
    dp = DP(cell["prior"], likes, cell["kernel"], cost, fl, critics=["L0", "L2", "L3"], cal_kernel=ck)
    out = run(cell, eps, {"j": make_joint_dp_policy(lambda info: jdp, infos, jsm), "d": make_dp_policy(dp)}, cost, fl, cell["pool"])
    for a, c in zip(out["j"], out["d"]):
        n += 1
        if a["actions"] != c["actions"] or a["utility"] != c["utility"] or [round(x, 10) for x in a["beliefs"]] != [round(x, 10) for x in c["beliefs"]]:
            bad += 1
res = dict(n_episodes=n, n_mismatch=bad, PASS=bad == 0)
json.dump(res, open(OUT, "w"), indent=1); print(res)
