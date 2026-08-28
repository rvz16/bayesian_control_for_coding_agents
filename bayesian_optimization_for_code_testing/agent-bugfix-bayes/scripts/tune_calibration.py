#!/usr/bin/env python
"""Real tuning experiment (HP plan Section 9 / Experiment H).

Train/val/test split on CodeContests (146 patches → 50/50/46).
Tune Beta-Binomial calibration prior (α, β) and initial belief b₀.
Compare best-validation config to default on held-out test set.

Honest discipline:
  1. Tuning happens on val ONLY.
  2. Test set is evaluated ONCE, with the locked-in best config.
  3. Default config (α=1, β=1, b₀=0.5) is evaluated on the same test set
     for fair comparison.
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from abbo.realworld.agents.agent_simulator import (
    PatchOutcomes, simulate_dp,
)
from abbo.realworld.agents.bayes_agent import DPPlanner
from abbo.realworld.agents.calibration import calibrate_likelihoods, CalibrationSample
from abbo.realworld.agents.code_contests import CC_CRITIC_NAMES
from abbo.realworld.agents.simple_agent import AgentCostConfig
from sweep_cverify import load_patches_from_allure

SEED = 42
R = 100
# Use c_v=20 so DP actually exercises critics (otherwise tuning has nothing to do)
COSTS = AgentCostConfig(c_llm_call=10, c_full_test=20, c_critic_test=1, reward=R)

# Tuning grid
ALPHA_BETA_GRID = [
    (0.5, 0.5),     # Jeffreys-like
    (1.0, 1.0),     # default (uniform)
    (2.0, 2.0),     # mild shrinkage
    (5.0, 5.0),     # strong shrinkage
    (1.0, 5.0),     # skeptical
    (5.0, 1.0),     # optimistic
]
PRIOR_GRID = [0.35, 0.5, 0.65]


def patches_to_samples(patches: list[PatchOutcomes]) -> list[CalibrationSample]:
    samples = []
    for p in patches:
        for c, passed in p.critic_outcomes.items():
            samples.append(CalibrationSample(
                bug_id=p.bug_id, arm=p.arm, critic=c,
                critic_passed=passed, patch_correct=p.Y,
            ))
    return samples


def evaluate_config(patches, theta, prior, costs=COSTS):
    """Mean utility over patches under (theta, prior). Re-solves DP."""
    planner = DPPlanner(costs, max_generators=0, max_verifications=1,
                        critic_likelihoods=theta)
    planner.solve()
    total = 0.0
    for p in patches:
        r = simulate_dp(p, theta, costs, prior=prior, max_verifications=1, planner=planner)
        total += (R if r.fixed else 0) - r.total_cost
    return total / len(patches)


def main():
    rng = random.Random(SEED)
    cc_attach = ROOT / "allure-results" / "7e4c207c-f8d9-46c5-8663-3ccdf4e5ea11-attachment.json"
    patches = load_patches_from_allure(cc_attach)
    rng.shuffle(patches)
    n = len(patches)
    # 50 / 50 / 46 split
    train = patches[:50]
    val   = patches[50:100]
    test  = patches[100:]
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}  total: {n}")
    print(f"Costs: {COSTS}\n")

    train_samples = patches_to_samples(train)

    # Grid search on validation
    print("=== Validation grid search ===\n")
    results = []
    for α, β in ALPHA_BETA_GRID:
        theta, _ = calibrate_likelihoods(train_samples, prior_alpha=α, prior_beta=β)
        for b0 in PRIOR_GRID:
            U_val = evaluate_config(val, theta, b0)
            results.append({"alpha": α, "beta": β, "prior": b0, "U_val": U_val,
                            "theta": theta})
            tag = ""
            print(f"  (α={α}, β={β}, b₀={b0}): U_val = {U_val:+7.3f}")

    # Pick best by val
    best = max(results, key=lambda r: r["U_val"])
    default = next(r for r in results if r["alpha"] == 1 and r["beta"] == 1 and r["prior"] == 0.5)

    print(f"\nBest on validation: α={best['alpha']}, β={best['beta']}, b₀={best['prior']}, "
          f"U_val = {best['U_val']:+.3f}")
    print(f"Default config:     α=1, β=1, b₀=0.5, U_val = {default['U_val']:+.3f}")

    # Evaluate ONCE on test
    print("\n=== Final test evaluation (one-shot) ===\n")
    U_test_best = evaluate_config(test, best["theta"], best["prior"])
    U_test_default = evaluate_config(test, default["theta"], default["prior"])
    print(f"  Tuned   (α={best['alpha']}, β={best['beta']}, b₀={best['prior']}):  U_test = {U_test_best:+.3f}")
    print(f"  Default (α=1, β=1, b₀=0.5):                            U_test = {U_test_default:+.3f}")
    delta = U_test_best - U_test_default
    print(f"  Δ tuning benefit on test = {delta:+.3f}  ({'helps' if delta > 0.01 else 'no benefit' if abs(delta) <= 0.01 else 'hurts'})")

    # Save
    out = ROOT / "sim_results" / "tuning_experiment_h.json"
    out.write_text(json.dumps({
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "seed": SEED,
        "costs": {"R": R, "c_v": COSTS.c_full_test, "c_critic": COSTS.c_critic_test, "c_llm": COSTS.c_llm_call},
        "grid": {"alpha_beta": ALPHA_BETA_GRID, "prior": PRIOR_GRID},
        "validation_results": [
            {k: v for k, v in r.items() if k != "theta"} for r in results
        ],
        "best_val": {k: v for k, v in best.items() if k != "theta"},
        "default_val": {k: v for k, v in default.items() if k != "theta"},
        "test": {
            "U_test_tuned": U_test_best,
            "U_test_default": U_test_default,
            "delta": delta,
        },
    }, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
