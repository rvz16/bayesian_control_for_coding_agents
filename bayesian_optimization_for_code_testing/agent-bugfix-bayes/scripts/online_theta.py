#!/usr/bin/env python
"""L2: Online θ updates after verified outcomes.

Stream patches in some order. After each verify reveals Y, update
Beta-Binomial counts for every critic that ran on that patch, re-fit θ,
and re-solve DP for the next patch.

Compare:
  Level 1 (frozen θ_hand): theta never changes, same DP throughout.
  Level 1' (frozen θ_fitted): pre-trained on a calibration set.
  Level 2 (online from θ_hand): starts at hand-tuned, learns from runtime.
  Level 2' (online from uniform): starts at Beta(1,1), learns from scratch.

Metrics: cumulative Ū trajectory, final θ estimates, final policy diff.
"""
from __future__ import annotations
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_cverify import CC_FITTED, load_patches_from_allure
from abbo.realworld.agents.agent_simulator import (
    PatchOutcomes, simulate_dp,
)
from abbo.realworld.agents.bayes_agent import DPPlanner
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig

SEED = 42
PRIOR = 0.5
ALPHA, BETA = 1.0, 1.0  # Beta(1,1) Laplace smoothing


@dataclass
class CriticCounts:
    n_y1: int = 0
    k_pass_y1: int = 0
    n_y0: int = 0
    k_pass_y0: int = 0


class OnlineCriticCalibration:
    """Beta-Binomial counts updated after each verified outcome."""

    def __init__(self, critic_names, alpha=ALPHA, beta=BETA,
                 init_theta: dict | None = None):
        self.alpha = alpha
        self.beta = beta
        self.counts = {c: CriticCounts() for c in critic_names}
        self._init_theta = init_theta  # fallback when we have no observations

    def update(self, critic, passed, y_true):
        c = self.counts[critic]
        if y_true == 1:
            c.n_y1 += 1
            if passed:
                c.k_pass_y1 += 1
        else:
            c.n_y0 += 1
            if passed:
                c.k_pass_y0 += 1

    def estimate(self) -> dict[str, dict[str, float]]:
        theta = {}
        for name, c in self.counts.items():
            # If we've seen zero of a class, fall back to initial theta or 0.5
            if c.n_y1 + c.n_y0 == 0 and self._init_theta is not None:
                theta[name] = dict(self._init_theta.get(name, {"p_pass_y1": 0.5, "p_pass_y0": 0.5}))
                continue
            p1 = (self.alpha + c.k_pass_y1) / (self.alpha + self.beta + c.n_y1)
            p0 = (self.alpha + c.k_pass_y0) / (self.alpha + self.beta + c.n_y0)
            # clip + enforce minimal gap (matches calibration.py)
            p1 = max(0.02, min(0.98, p1))
            p0 = max(0.02, min(0.98, p0))
            if p1 - p0 < 0.05:
                mid = (p1 + p0) / 2
                p1 = min(0.98, mid + 0.025)
                p0 = max(0.02, mid - 0.025)
            theta[name] = {"p_pass_y1": p1, "p_pass_y0": p0}
        return theta


def simulate_level1(patches, theta, costs, prior=PRIOR):
    """Frozen-θ baseline: theta never changes."""
    planner = DPPlanner(costs, max_generators=0, max_verifications=1,
                        critic_likelihoods=theta)
    planner.solve()
    results = []
    for p in patches:
        r = simulate_dp(p, theta, costs, prior=prior,
                        max_verifications=1, planner=planner)
        results.append(r)
    return results


def simulate_level2(patches, init_theta, costs, prior=PRIOR,
                    critic_names=CC_CRITIC_NAMES):
    """Online: start at init_theta, update after each verified patch."""
    cal = OnlineCriticCalibration(critic_names, init_theta=init_theta)
    results = []
    theta_trajectory = []
    for i, p in enumerate(patches):
        theta = cal.estimate()
        planner = DPPlanner(costs, max_generators=0, max_verifications=1,
                            critic_likelihoods=theta)
        planner.solve()
        r = simulate_dp(p, theta, costs, prior=prior,
                        max_verifications=1, planner=planner)
        results.append(r)
        # Update counts from critics observed during this patch.
        # patch.critic_outcomes has the truth; r.actions has what was observed.
        for action in r.actions:
            if action.action.startswith("critic:"):
                critic = action.action.split(":", 1)[1]
                if critic in p.critic_outcomes:
                    cal.update(critic, p.critic_outcomes[critic], int(p.Y))
        # The verify reveals Y but doesn't observe critic outcomes the agent
        # didn't run — we only update critics the agent actually ran.
        theta_trajectory.append({
            "i": i, "patch": p.bug_id,
            "theta_snapshot": {k: dict(v) for k, v in theta.items()},
        })
    return results, theta_trajectory


def utility(results, R=100):
    return sum((R if r.fixed else 0) - r.total_cost for r in results) / max(1, len(results))


def main():
    print("=== L2: Online θ updates on CodeContests held-out (n=146) ===\n")

    cc_attach = ROOT / "allure-results" / "7e4c207c-f8d9-46c5-8663-3ccdf4e5ea11-attachment.json"
    patches = load_patches_from_allure(cc_attach)
    rng = random.Random(SEED)
    rng.shuffle(patches)
    print(f"Loaded {len(patches)} patches, shuffled with seed={SEED}")

    # Use c_v=20 so DP actively uses critics (prior sweep showed critics
    # only get called when c_v ≥ ~10; at c_v=5 DP just verifies directly).
    # This is the regime where online θ learning has signal to learn from.
    costs = AgentCostConfig(c_llm_call=10, c_full_test=20, c_critic_test=1, reward=100)
    print(f"Costs: c_v={costs.c_full_test}, c_critic={costs.c_critic_test}, "
          f"c_llm={costs.c_llm_call}, R={costs.reward}")

    # Baselines
    print("\n--- Running 4 variants ---")
    l1_hand = simulate_level1(patches, CC_CRITIC_LIKELIHOODS, costs)
    print(f"L1 hand:    Ū = {utility(l1_hand):+.3f}")
    l1_fitted = simulate_level1(patches, CC_FITTED, costs)
    print(f"L1 fitted:  Ū = {utility(l1_fitted):+.3f}")
    l2_from_hand, traj_hand = simulate_level2(patches, CC_CRITIC_LIKELIHOODS, costs)
    print(f"L2 from-hand: Ū = {utility(l2_from_hand):+.3f}")
    UNIFORM = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in CC_CRITIC_NAMES}
    l2_cold, traj_cold = simulate_level2(patches, UNIFORM, costs)
    print(f"L2 cold-start (Beta(1,1)): Ū = {utility(l2_cold):+.3f}")

    # Trajectory: utility-so-far at each i
    def cum_util(rs, R=100):
        out, total = [], 0.0
        for i, r in enumerate(rs, 1):
            total += (R if r.fixed else 0) - r.total_cost
            out.append(total / i)
        return out

    cum = {
        "L1_hand": cum_util(l1_hand),
        "L1_fitted": cum_util(l1_fitted),
        "L2_from_hand": cum_util(l2_from_hand),
        "L2_cold_start": cum_util(l2_cold),
    }

    # Snapshot trajectory at quartiles
    n = len(patches)
    qs = [n // 4, n // 2, 3 * n // 4, n - 1]
    print(f"\n### Cumulative Ū over time ===\n")
    print("| after_n | L1_hand | L1_fitted | L2_from_hand | L2_cold |")
    print("|---|---:|---:|---:|---:|")
    for q in qs:
        row = f"| {q+1} |"
        for k in ["L1_hand", "L1_fitted", "L2_from_hand", "L2_cold_start"]:
            row += f" {cum[k][q]:+.2f} |"
        print(row)

    # Final theta comparison
    print("\n### Theta comparison after streaming all 146 patches\n")
    final_theta = traj_hand[-1]["theta_snapshot"]
    cold_theta = traj_cold[-1]["theta_snapshot"]
    print("| critic | L1 hand | L1 fitted | L2-from-hand final | L2-cold final |")
    print("|---|---|---|---|---|")
    for cn in CC_CRITIC_NAMES:
        h = CC_CRITIC_LIKELIHOODS[cn]
        f_ = CC_FITTED[cn]
        ft = final_theta[cn]
        cd = cold_theta[cn]
        print(f"| {cn} | "
              f"({h['p_pass_y1']:.2f},{h['p_pass_y0']:.2f}) | "
              f"({f_['p_pass_y1']:.2f},{f_['p_pass_y0']:.2f}) | "
              f"({ft['p_pass_y1']:.2f},{ft['p_pass_y0']:.2f}) | "
              f"({cd['p_pass_y1']:.2f},{cd['p_pass_y0']:.2f}) |")

    out = ROOT / "sim_results" / "online_theta_l2.json"
    out.write_text(json.dumps({
        "n_patches": n,
        "seed": SEED,
        "utilities": {
            "L1_hand": utility(l1_hand),
            "L1_fitted": utility(l1_fitted),
            "L2_from_hand": utility(l2_from_hand),
            "L2_cold_start": utility(l2_cold),
        },
        "cumulative": cum,
        "final_theta_l2_from_hand": final_theta,
        "final_theta_l2_cold": cold_theta,
    }, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
