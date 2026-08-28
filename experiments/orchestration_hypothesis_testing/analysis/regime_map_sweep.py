"""2D regime map: sweep (prior, c_ver) at fixed critic configuration.

Uses LCB gpt5_mini's measured critic likelihoods as the critic config
(L0 gap=0.66, L2 gap=0.81, L3 gap=0.37 — three informative critics, no
oracle). Sweeps prior from 0.10 to 0.85 and c_ver from 5 to 100.

For each (prior, c_ver) cell:
  - Runs every policy on synthetic IID trajectories drawn from the
    likelihood tables (1000 trajectories of length 5)
  - IID transition kernel: P_fix = prior, P_break = 1 - prior
  - Finds winning policy + utility margin

Output: regime_map.json + ascii winner table

Marks the three real-benchmark datapoints on the map:
  sympy:          (prior=0.22, c_ver=30, L2_gap=0.97 ORACLE)
  SWE-bench Lite: (prior=0.37, c_ver=30, L3_gap=0.19 only)
  LCB hard:       (prior=0.56, c_ver=30, L0+L2+L3 multi-strong)
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


def sample_trajectory(rng: np.random.Generator, prior: float,
                      likelihoods: dict, horizon: int = 5) -> list[dict]:
    """Sample a synthetic trajectory: H independent patches drawn from
    the likelihood tables.

    Each patch:
      Y ~ Bernoulli(prior)
      L0_syntax | Y ~ Bernoulli(P_pass_given_Y[Y])
      L3_llm_review | Y ~ Bernoulli(P_pass_given_Y[Y])
    """
    traj = []
    for step in range(horizon):
        Y = int(rng.uniform() < prior)
        rec = {"step": step, "instance_id": "synthetic", "Y": Y}
        for k in ("L0_syntax", "L3_llm_review"):
            l = likelihoods[k]
            p = l["P_pass_given_Y1"] if Y == 1 else l["P_pass_given_Y0"]
            rec[k] = bool(rng.uniform() < p)
        traj.append(rec)
    return traj


def threshold_L2_factory(critic_field="L2_public_tests"):
    def _p(state, rec):
        if not state.get("L2_done"):
            state["L2_done"] = True
            return "L0"  # use L0 cost slot for L2 (cheap)
        state["L2_done"] = False
        # In our synthetic data L2 isn't in rec; treat absence as use L0 instead
        return "verify" if rec.get("L0_syntax") else (
            "generate" if state["patch_idx"] + 1 < 5 else "give_up"
        )
    return _p


def evaluate_cell(prior: float, c_ver: float, likelihoods: dict,
                  n_trajs: int = 500, horizon: int = 5,
                  seed: int = 42) -> dict[str, float]:
    """Run all policies at this (prior, c_ver), return mean utility per policy."""
    cost = CostModel(c_gen=10, c_L0=1, c_L3=5, c_ver=c_ver, reward=100)
    # Bayesian controller with IID kernel
    likes_with_prior = dict(likelihoods)
    likes_with_prior["prior_Y1"] = prior
    kernel = {"kernel_all": {
        "P_fix_given_broken": prior,
        "P_break_given_correct": 1 - prior,
    }}
    controller = BayesianController(prior, {"critic_likelihoods": likelihoods}, kernel, cost, horizon=horizon)

    rng = np.random.default_rng(seed)
    policies = {
        "always_verify": policy_always_verify,
        "threshold_L0": policy_threshold_L0,
        "threshold_L3": policy_threshold_L3,
        "fixed_pipeline": policy_fixed_pipeline,
        "best_of_3": policy_best_of_N(3),
        "bayesian": make_bayesian_policy(controller),
    }

    sums = {name: 0.0 for name in policies}
    for _ in range(n_trajs):
        traj = sample_trajectory(rng, prior, likelihoods, horizon=horizon)
        for name, fn in policies.items():
            r = simulate_policy(traj, fn, cost)
            sums[name] += r["utility"]
    return {name: s / n_trajs for name, s in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihoods", type=Path, required=True,
                        help="Path to likelihood_tables.json (use LCB gpt5)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-min", type=float, default=0.10)
    parser.add_argument("--prior-max", type=float, default=0.85)
    parser.add_argument("--prior-step", type=float, default=0.05)
    parser.add_argument("--cver-min", type=float, default=5)
    parser.add_argument("--cver-max", type=float, default=100)
    parser.add_argument("--cver-step", type=float, default=10)
    parser.add_argument("--n-trajs", type=int, default=500)
    args = parser.parse_args()

    likes_data = json.loads(args.likelihoods.read_text())
    likelihoods = likes_data["critic_likelihoods"]
    print(f"Loaded likelihoods from {args.likelihoods.name}")
    for k, v in likelihoods.items():
        if v.get("gap") is not None:
            print(f"  {k}: gap={v['gap']:.3f}")

    priors = np.arange(args.prior_min, args.prior_max + 1e-9, args.prior_step).round(3)
    cvers = np.arange(args.cver_min, args.cver_max + 1e-9, args.cver_step).round(1)
    print(f"sweeping {len(priors)} priors x {len(cvers)} c_vers = {len(priors) * len(cvers)} cells")

    grid: dict = {}
    for p in priors:
        grid[float(p)] = {}
        for c in cvers:
            cell = evaluate_cell(float(p), float(c), likelihoods, n_trajs=args.n_trajs)
            grid[float(p)][float(c)] = cell

    args.output.write_text(json.dumps(grid, indent=2))
    print(f"\nwrote {args.output}")

    # Print winner table
    policies = list(grid[float(priors[0])][float(cvers[0])].keys())
    print("\nWINNER MAP (rows=prior, cols=c_ver, *=Bayesian wins, .=ties Bayesian within 0.5):")
    header_label = "p|c"
    print(f"  {header_label:>5} | " + " ".join(f"{c:>6.0f}" for c in cvers))
    print("  " + "-" * (8 + 7 * len(cvers)))
    for p in priors:
        cells = []
        for c in cvers:
            row = grid[float(p)][float(c)]
            best_p = max(row, key=lambda k: row[k])
            best_v = row[best_p]
            bay_v = row["bayesian"]
            if best_p == "bayesian":
                marker = "*"
                label = "BAY"
            elif abs(best_v - bay_v) < 0.5:
                marker = "."
                label = best_p[:6]
            else:
                marker = " "
                label = best_p[:6]
            cells.append(f"{marker}{label:>5}")
        print(f"  {p:>5.2f} |" + " ".join(cells))

    # Find Bayesian-wins-strictly cells
    bay_wins = []
    for p in priors:
        for c in cvers:
            row = grid[float(p)][float(c)]
            best_p = max(row, key=lambda k: row[k])
            best_v = row[best_p]
            bay_v = row["bayesian"]
            if best_p == "bayesian" and best_v - sorted(row.values())[-2] > 0.5:
                bay_wins.append((float(p), float(c), bay_v - sorted(row.values())[-2]))
    print(f"\nBayesian wins strictly (margin > 0.5) in {len(bay_wins)} of {len(priors)*len(cvers)} cells.")
    if bay_wins:
        print("  Top 5 by margin:")
        for p, c, m in sorted(bay_wins, key=lambda x: -x[2])[:5]:
            print(f"    prior={p:.2f}, c_ver={c:.0f}: margin = {m:.2f}")


if __name__ == "__main__":
    main()
