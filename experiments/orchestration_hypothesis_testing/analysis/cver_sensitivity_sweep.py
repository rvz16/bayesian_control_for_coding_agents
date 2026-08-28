"""Sweep verifier cost c_ver across a range and replot policy utilities.

Reveals the regime where Bayesian beats baselines — at low c_ver, "always
verify" wins trivially; at high c_ver, critic-based policies (including
Bayesian) become valuable; the crossover point is the publishable claim.

Output: <output-dir>/cver_sweep.json + ascii table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis.controller import (  # noqa: E402
    BayesianController,
    CostModel,
    load_trajectories,
    policy_always_verify,
    policy_threshold_L0,
    policy_threshold_L3,
    policy_fixed_pipeline,
    policy_best_of_N,
    make_bayesian_policy,
    simulate_policy,
)


def run_sweep(out_dir: Path, generators: list[str], c_ver_values: list[float]) -> dict:
    """Run all policies at each c_ver value, for each generator."""
    results: dict = {}
    for gen in generators:
        gen_dir = out_dir / gen
        traj = load_trajectories(gen_dir / "iter_records.jsonl")
        likes = json.loads((gen_dir / "likelihood_tables.json").read_text())
        kernel = json.loads((gen_dir / "transition_kernel.json").read_text())
        prior = likes.get("prior_Y1", 0.5)

        per_cver: dict[str, dict] = {}
        for c_ver in c_ver_values:
            cost = CostModel(c_gen=10, c_L0=1, c_L3=5, c_ver=c_ver, reward=100)
            controller = BayesianController(prior, likes, kernel, cost)
            policies = {
                "always_verify": policy_always_verify,
                "threshold_L0": policy_threshold_L0,
                "threshold_L3": policy_threshold_L3,
                "fixed_pipeline": policy_fixed_pipeline,
                "best_of_3": policy_best_of_N(3),
                "bayesian": make_bayesian_policy(controller),
            }
            row = {}
            for name, fn in policies.items():
                utils = []
                for inst, t in traj.items():
                    r = simulate_policy(t, fn, cost)
                    utils.append(r["utility"])
                row[name] = float(np.mean(utils))
            per_cver[str(c_ver)] = row
        results[gen] = per_cver
    return results


def print_table(results: dict, c_ver_values: list[float]) -> None:
    for gen, per_cver in results.items():
        print(f"\n=== {gen} ===")
        policies = list(next(iter(per_cver.values())).keys())
        # Header
        print(f"  {'c_ver':>6} | " + " ".join(f"{p:>14}" for p in policies))
        print("  " + "-" * (8 + 15 * len(policies)))
        for c in c_ver_values:
            row = per_cver[str(c)]
            best = max(row, key=lambda p: row[p])
            cells = []
            for p in policies:
                marker = "*" if p == best else " "
                cells.append(f"{marker}{row[p]:>13.2f}")
            print(f"  {c:>6.0f} |" + "".join(cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--c-ver-min", type=float, default=5)
    parser.add_argument("--c-ver-max", type=float, default=120)
    parser.add_argument("--c-ver-step", type=float, default=10)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    gens = [g.strip() for g in args.generators.split(",") if g.strip()]
    c_ver_values = list(np.arange(args.c_ver_min, args.c_ver_max + 0.01, args.c_ver_step).round(2))
    print(f"sweeping c_ver over {c_ver_values}")
    results = run_sweep(out, gens, c_ver_values)
    sweep_path = out / "cver_sweep.json"
    sweep_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {sweep_path}")
    print_table(results, c_ver_values)


if __name__ == "__main__":
    main()
