#!/usr/bin/env python
"""L2 with ε-greedy verify-on-bail exploration.

Fix the broken naive L2: when DP picks `bail`, with probability ε force a
verify instead. This pays ε · bail_rate · c_v extra cost per task but
gives us unbiased Y observations for online θ updates.

Sweeps ε ∈ {0, 0.05, 0.1, 0.2, 0.5} on the 146 CC patches at c_v=20.
Reports Ū and final θ̂ vs frozen-θ baselines.
"""
from __future__ import annotations
import json, random, sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_cverify import CC_FITTED, load_patches_from_allure
from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update
from abbo.realworld.agents.code_contests import CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES
from abbo.realworld.agents.simple_agent import AgentCostConfig
from online_theta import OnlineCriticCalibration

SEED = 42
PRIOR = 0.5
COSTS = AgentCostConfig(c_llm_call=10, c_full_test=20, c_critic_test=1, reward=100)
R = 100
EPS_GRID = [0.0, 0.05, 0.1, 0.2, 0.5]


def simulate_one_patch_eps(patch, theta, costs, planner, prior, eps, rng):
    """Run one patch through DP with ε-greedy bail override.

    Returns (utility, observed_critics, Y_observed_or_None).
    observed_critics = list of (critic, passed) actually run during this patch.
    Y_observed_or_None = ground-truth Y if we verified, else None.
    """
    belief = prior
    crit_used = frozenset()
    ver_left = 1
    cost = 0.0
    step = 0
    observed = []
    max_steps = 12

    while step < max_steps:
        action, _q = planner.choose_action(
            belief=belief, gen_left=0,
            crit_used=crit_used, ver_left=ver_left,
        )
        # ε-greedy override on bail
        if action == "bail_out" and rng.random() < eps:
            action = "verify"  # forced exploration

        if action == "bail_out":
            return (R if False else 0) - cost, observed, None

        if action == "verify":
            cost += costs.c_full_test
            # Reveal Y from the patch's known label
            Y = int(patch.Y)
            return (R if Y == 1 else 0) - cost, observed, Y

        if action.startswith("critic:"):
            critic = action.split(":", 1)[1]
            passed = patch.critic_outcomes.get(critic, False)
            belief = bayes_update(belief, critic, passed, likelihoods=theta)
            cost += costs.c_critic_test
            crit_used = crit_used | frozenset([critic])
            observed.append((critic, passed))
            step += 1
            continue

        # generate disabled in simulator
        return -cost, observed, None

    return -cost, observed, None


def run_eps(patches, init_theta, eps, costs=COSTS, prior=PRIOR, seed=SEED):
    rng = random.Random(seed)
    cal = OnlineCriticCalibration(CC_CRITIC_NAMES, init_theta=init_theta)
    utilities = []
    n_forced = 0
    for p in patches:
        theta = cal.estimate()
        planner = DPPlanner(costs, max_generators=0, max_verifications=1,
                            critic_likelihoods=theta)
        planner.solve()
        u, observed, Y = simulate_one_patch_eps(
            p, theta, costs, planner, prior, eps, rng,
        )
        utilities.append(u)
        # Update theta only if we observed Y (verified)
        if Y is not None:
            for critic, passed in observed:
                cal.update(critic, passed, Y)
    return utilities, cal.estimate()


def main():
    rng = random.Random(SEED)
    cc_attach = ROOT / "allure-results" / "7e4c207c-f8d9-46c5-8663-3ccdf4e5ea11-attachment.json"
    patches = load_patches_from_allure(cc_attach)
    rng.shuffle(patches)
    n = len(patches)
    print(f"Loaded {n} CC patches; cost regime c_v={COSTS.c_full_test}\n")

    # Baselines (frozen θ)
    from online_theta import simulate_level1, utility
    l1_hand = simulate_level1(patches, CC_CRITIC_LIKELIHOODS, COSTS)
    l1_fitted = simulate_level1(patches, CC_FITTED, COSTS)
    print(f"Baselines (frozen θ):")
    print(f"  L1 hand:   Ū = {utility(l1_hand):+.3f}")
    print(f"  L1 fitted: Ū = {utility(l1_fitted):+.3f}\n")

    print("=== ε-greedy online L2 (starting from hand-tuned θ) ===\n")
    results = []
    for eps in EPS_GRID:
        utils, final_theta = run_eps(patches, CC_CRITIC_LIKELIHOODS, eps)
        U = sum(utils) / len(utils)
        results.append({"eps": eps, "U_final": U, "final_theta": final_theta})
        print(f"  ε = {eps:>4}: Ū = {U:+7.3f}  Δ vs L1-hand = {U - utility(l1_hand):+.3f}")

    print("\n=== ε-greedy online L2 (cold start from uniform θ) ===\n")
    UNIFORM = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in CC_CRITIC_NAMES}
    results_cold = []
    for eps in EPS_GRID:
        utils, final_theta = run_eps(patches, UNIFORM, eps)
        U = sum(utils) / len(utils)
        results_cold.append({"eps": eps, "U_final": U, "final_theta": final_theta})
        print(f"  ε = {eps:>4}: Ū = {U:+7.3f}  Δ vs L1-hand = {U - utility(l1_hand):+.3f}")

    # Final θ for the best ε on from-hand start
    best = max(results, key=lambda r: r["U_final"])
    print(f"\n### Best ε (from-hand start): {best['eps']} → Ū = {best['U_final']:+.3f}")
    print(f"\nFinal θ̂ comparison ('learned' = best ε from-hand):")
    print(f"{'critic':<15} {'L1 hand':<16} {'L1 fitted':<16} {'Learned (ε={})':<22}".format(best['eps']))
    for cn in CC_CRITIC_NAMES:
        h = CC_CRITIC_LIKELIHOODS[cn]
        f_ = CC_FITTED[cn]
        ln = best["final_theta"][cn]
        print(f"{cn:<15} "
              f"({h['p_pass_y1']:.2f}, {h['p_pass_y0']:.2f})   "
              f"({f_['p_pass_y1']:.2f}, {f_['p_pass_y0']:.2f})   "
              f"({ln['p_pass_y1']:.2f}, {ln['p_pass_y0']:.2f})")

    out = ROOT / "sim_results" / "online_theta_eps.json"
    out.write_text(json.dumps({
        "n_patches": n,
        "seed": SEED,
        "eps_grid": EPS_GRID,
        "from_hand_start": [{"eps": r["eps"], "U": r["U_final"]} for r in results],
        "cold_start":      [{"eps": r["eps"], "U": r["U_final"]} for r in results_cold],
        "best_eps_from_hand": best["eps"],
        "best_final_theta": best["final_theta"],
        "baselines": {
            "L1_hand": utility(l1_hand),
            "L1_fitted": utility(l1_fitted),
        },
    }, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
