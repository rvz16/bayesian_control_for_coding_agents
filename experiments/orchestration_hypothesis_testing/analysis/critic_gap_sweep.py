"""Critic-gap sensitivity sweep.

Holds prior + c_ver fixed at the most-realistic-benchmark point
(LCB-gpt5: prior=0.56, c_ver=30) and sweeps the synthetic gap of one
critic from 0.1 to 0.95. Tells us at what critic gap does Bayesian
start winning via inference (not give-up).

The give-up regime kicks in when c_ver > prior * reward
  = 0.56 * 100 = 56 ≫ c_ver=30.
So at this point give-up is NOT optimal (verify is positive: 56 - 30 = +26).
Any Bayesian win here is genuinely inference-driven.

We sweep:
  - L3_gap (the noisy critic) from 0.1 to 0.95, holding L0 + L2 fixed.
    This lets us isolate when ADDING a stronger noisy critic helps.
  - We also report the all-critics-disabled-except-L3 case to show
    threshold(L3) at each gap.

Output: critic_gap_sweep.json + ascii table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # orchestration_hypothesis_testing/
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from analysis.controller import (  # noqa: E402
    BayesianController, CostModel, simulate_policy,
    policy_always_verify, policy_threshold_L0, policy_threshold_L3,
    policy_fixed_pipeline, policy_best_of_N, make_bayesian_policy,
)
from regime_map_sweep import sample_trajectory  # noqa: E402


def make_likelihoods(p_pass_y1: float, p_pass_y0: float) -> dict:
    """Critic table holding L0 at LCB-gpt5 measured (gap=0.66),
    sweeping L3."""
    return {
        # LCB-gpt5 measured L0 (real, strong signal)
        "L0_syntax": {"P_pass_given_Y1": 0.98, "P_pass_given_Y0": 0.33,
                      "gap": 0.65},
        "L3_llm_review": {
            "P_pass_given_Y1": p_pass_y1,
            "P_pass_given_Y0": p_pass_y0,
            "gap": p_pass_y1 - p_pass_y0,
        },
    }


def evaluate_gap(prior: float, c_ver: float, p_y1: float, p_y0: float,
                 n_trajs: int = 1000, horizon: int = 5,
                 seed: int = 42) -> dict[str, float]:
    likelihoods = make_likelihoods(p_y1, p_y0)
    cost = CostModel(c_gen=10, c_L0=1, c_L3=5, c_ver=c_ver, reward=100)
    kernel = {"kernel_all": {"P_fix_given_broken": prior,
                             "P_break_given_correct": 1 - prior}}
    controller = BayesianController(prior, {"critic_likelihoods": likelihoods},
                                    kernel, cost, horizon=horizon)

    rng = np.random.default_rng(seed)
    policies = {
        "always_verify": policy_always_verify,
        "threshold_L3": policy_threshold_L3,
        "best_of_3": policy_best_of_N(3),
        "bayesian": make_bayesian_policy(controller),
    }
    sums = {n: 0.0 for n in policies}
    for _ in range(n_trajs):
        traj = sample_trajectory(rng, prior, likelihoods, horizon=horizon)
        for n, fn in policies.items():
            r = simulate_policy(traj, fn, cost)
            sums[n] += r["utility"]
    return {n: s / n_trajs for n, s in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior", type=float, default=0.56)
    parser.add_argument("--cver", type=float, default=30.0)
    # We hold P(pass|Y=1) high (= 0.95) and sweep P(pass|Y=0) from
    # 0.05 to 0.85 — equivalent to sweeping gap from 0.10 to 0.90.
    parser.add_argument("--py1", type=float, default=0.95)
    parser.add_argument("--py0-min", type=float, default=0.05)
    parser.add_argument("--py0-max", type=float, default=0.85)
    parser.add_argument("--py0-step", type=float, default=0.05)
    parser.add_argument("--n-trajs", type=int, default=1000)
    args = parser.parse_args()

    py0s = np.arange(args.py0_min, args.py0_max + 1e-9, args.py0_step).round(3)
    print(f"prior={args.prior}, c_ver={args.cver}, P(pass|Y=1)={args.py1}")
    print(f"sweeping P(pass|Y=0) from {args.py0_min} to {args.py0_max} -> gap from "
          f"{args.py1 - args.py0_max:.2f} to {args.py1 - args.py0_min:.2f}")

    results: dict = {}
    for py0 in py0s:
        gap = args.py1 - float(py0)
        results[f"{gap:.3f}"] = {
            "py1": args.py1, "py0": float(py0), "gap": float(gap),
            "policies": evaluate_gap(args.prior, args.cver,
                                     args.py1, float(py0), n_trajs=args.n_trajs),
        }
    args.output.write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}\n")

    # Print table
    pol_names = list(next(iter(results.values()))["policies"].keys())
    header = f"  {'gap':>6} | " + " ".join(f"{p:>14}" for p in pol_names) + "  | winner"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for gap_key, row in results.items():
        gap = row["gap"]
        utils = row["policies"]
        best = max(utils, key=lambda k: utils[k])
        bay_v = utils["bayesian"]
        thr_v = utils["threshold_L3"]
        cells = " ".join(f"{utils[p]:>14.2f}" for p in pol_names)
        margin = bay_v - max(v for k, v in utils.items() if k != "bayesian")
        marker = "*" if best == "bayesian" and margin > 0.5 else " "
        print(f"  {gap:>6.3f} | {cells}  | {marker}{best:<14} (BAY-2nd_best margin={margin:+.2f})")


if __name__ == "__main__":
    main()
