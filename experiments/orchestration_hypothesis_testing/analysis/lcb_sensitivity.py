"""Tier D sensitivity analysis on empirical LCB calibration data.

Three pre-registered analyses produced from the existing v2 calibration
(no extra API calls):

  D1 (theta-sensitivity, §E4) — Perturb each P(z|Y) entry by ±10%, ±20%.
                                Refit the Bayesian DP & Greedy controllers
                                from the perturbed likelihood tables and
                                re-run all 8 policies. Confirms the win
                                survives realistic miscalibration.

  D2 (cost-model sweep, §F4)  — Sweep c_ver ∈ {10, 15, 20, 30, 40, 60, 100}
                                with empirical (not synthetic) likelihoods.
                                Plot policy utility curves vs c_ver.

  D3 (verifier efficiency, §G2) — verify_calls_per_solve as a secondary
                                  metric. Bayesian story: same pass rate at
                                  fewer verify calls.

Inputs per generator:
  data/lcb_calibration_v2/<gen>/critic_results.jsonl
  data/lcb_calibration_v2/<gen>/likelihood_tables.json

Outputs:
  data/lcb_calibration_v2/<gen>/sensitivity.json
  data/lcb_calibration_v2/<gen>/sensitivity.csv  (paper-friendly)
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.controller import (  # noqa: E402
    BayesianController, CostModel, simulate_policy,
    policy_always_verify, policy_threshold_L0, policy_threshold_L3,
    policy_fixed_pipeline, policy_best_of_N, make_bayesian_policy,
)
from analysis.lcb_compare import (  # noqa: E402
    GreedyController, make_greedy_policy, policy_threshold_L2,
    load_lcb_trajectories, paired_bootstrap_ci,
)


def perturb_likelihoods(likes: dict, frac: float, mode: str = "uniform") -> dict:
    """Return a copy of likelihoods with each P(z|Y) entry shifted by `frac`.

    mode='uniform': add +frac (clipped to [0.01, 0.99]).
    mode='alternating': flip sign per entry — shrinks gaps, the worst case
                        for a controller that relies on critic informativeness.
    """
    out = deepcopy(likes)
    cl = out["critic_likelihoods"]
    flip = 1
    for name, l in cl.items():
        for k in ("P_pass_given_Y1", "P_pass_given_Y0"):
            v = l[k]
            sign = flip if mode == "alternating" else 1
            new = max(0.01, min(0.99, v + sign * frac))
            l[k] = new
            flip = -flip
        l["gap"] = l["P_pass_given_Y1"] - l["P_pass_given_Y0"]
    return out


def run_policies(traj: dict, likes: dict, prior: float, cost: CostModel,
                 n_boot: int = 1000, baseline: str = "always_verify") -> dict:
    kernel = {"kernel_all": {"P_fix_given_broken": prior, "P_break_given_correct": 1 - prior}}
    dp = BayesianController(prior, likes, kernel, cost, horizon=3)
    greedy = GreedyController(prior, likes, cost)

    policies = {
        "always_verify": policy_always_verify,
        "threshold_L0": policy_threshold_L0,
        "threshold_L2": policy_threshold_L2,
        "threshold_L3": policy_threshold_L3,
        "fixed_pipeline": policy_fixed_pipeline,
        "best_of_3": policy_best_of_N(3),
        "bayesian_DP": make_bayesian_policy(dp),
        "bayesian_greedy": make_greedy_policy(greedy),
    }

    utils: dict[str, list[float]] = {n: [] for n in policies}
    rewards: dict[str, list[float]] = {n: [] for n in policies}
    n_verify: dict[str, list[int]] = {n: [] for n in policies}
    for inst, t in traj.items():
        for name, fn in policies.items():
            r = simulate_policy(t, fn, cost)
            utils[name].append(r["utility"])
            rewards[name].append(r["reward"])
            n_verify[name].append(1 if r.get("verified") else 0)

    base_u = utils[baseline]
    out = {}
    for name in policies:
        u = utils[name]
        r = rewards[name]
        v = n_verify[name]
        n_solved = sum(1 for rr in r if rr > 0)
        n_verified = sum(v)
        mean_diff, lo, hi = paired_bootstrap_ci(u, base_u, n_boot)
        out[name] = {
            "mean_utility": float(np.mean(u)),
            "pass_rate": float(np.mean([rr > 0 for rr in r])),
            "n_verified": n_verified,
            "n_solved": n_solved,
            "verify_per_solve": (n_verified / n_solved) if n_solved else None,
            "diff_vs_baseline": mean_diff,
            "ci95_lo": lo,
            "ci95_hi": hi,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    c_ver_grid = [10, 15, 20, 30, 40, 60, 100]
    reward_grid = [25, 50, 100, 200, 500]   # D4: reward sweep with c_ver fixed at 30
    perturb_grid = [
        ("clean", 0.0, "uniform"),
        ("plus_10", +0.10, "uniform"),
        ("minus_10", -0.10, "uniform"),
        ("plus_20", +0.20, "uniform"),
        ("minus_20", -0.20, "uniform"),
        ("alt_10", 0.10, "alternating"),
        ("alt_20", 0.20, "alternating"),
    ]

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen_dir = out_dir / gen
        rec_path = gen_dir / "critic_results.jsonl"
        like_path = gen_dir / "likelihood_tables.json"
        if not rec_path.exists() or not like_path.exists():
            print(f"[{gen}] missing data, skipping")
            continue
        traj = load_lcb_trajectories(rec_path)
        likes_clean = json.loads(like_path.read_text())
        prior = likes_clean.get("prior_Y1", 0.5)

        results = {
            "generator": gen, "prior": prior, "n_instances": len(traj),
            "D1_theta_sensitivity": {}, "D2_c_ver_sweep": {},
            "D3_verifier_efficiency": {}, "D4_reward_sweep": {},
        }

        # D1: theta-sensitivity (default c_ver = 30)
        cost_default = CostModel(c_gen=10, c_L0=1, c_L2=2, c_L3=5, c_ver=30, reward=100)
        for label, frac, mode in perturb_grid:
            likes_p = perturb_likelihoods(likes_clean, frac, mode)
            res = run_policies(traj, likes_p, prior, cost_default, args.n_boot)
            results["D1_theta_sensitivity"][label] = res

        # D2: c_ver sweep (clean likelihoods)
        for c_ver in c_ver_grid:
            cost_v = CostModel(c_gen=10, c_L0=1, c_L2=2, c_L3=5, c_ver=c_ver, reward=100)
            res = run_policies(traj, likes_clean, prior, cost_v, args.n_boot)
            results["D2_c_ver_sweep"][f"c_ver_{c_ver}"] = res

        # D4: reward sweep (clean likelihoods, c_ver fixed at 30)
        for reward in reward_grid:
            cost_r = CostModel(c_gen=10, c_L0=1, c_L2=2, c_L3=5, c_ver=30, reward=reward)
            res = run_policies(traj, likes_clean, prior, cost_r, args.n_boot)
            results["D4_reward_sweep"][f"reward_{reward}"] = res

        # D3: verifier efficiency at default — pulled from clean run
        clean = results["D1_theta_sensitivity"]["clean"]
        for name, r in clean.items():
            results["D3_verifier_efficiency"][name] = {
                "n_solved": r["n_solved"], "n_verified": r["n_verified"],
                "verify_per_solve": r["verify_per_solve"],
            }

        out_json = gen_dir / "sensitivity.json"
        out_json.write_text(json.dumps(results, indent=2))

        # Compact CSV view: D1 headline + D2 headline
        csv_lines = ["analysis,scenario,policy,utility,pass_rate,diff,ci_lo,ci_hi"]
        for label, _, _ in perturb_grid:
            for name, r in results["D1_theta_sensitivity"][label].items():
                csv_lines.append(
                    f"D1,{label},{name},{r['mean_utility']:.2f},{r['pass_rate']:.3f},"
                    f"{r['diff_vs_baseline']:.2f},{r['ci95_lo']:.2f},{r['ci95_hi']:.2f}"
                )
        for c_ver in c_ver_grid:
            for name, r in results["D2_c_ver_sweep"][f"c_ver_{c_ver}"].items():
                csv_lines.append(
                    f"D2,c_ver_{c_ver},{name},{r['mean_utility']:.2f},{r['pass_rate']:.3f},"
                    f"{r['diff_vs_baseline']:.2f},{r['ci95_lo']:.2f},{r['ci95_hi']:.2f}"
                )
        for reward in reward_grid:
            for name, r in results["D4_reward_sweep"][f"reward_{reward}"].items():
                csv_lines.append(
                    f"D4,reward_{reward},{name},{r['mean_utility']:.2f},{r['pass_rate']:.3f},"
                    f"{r['diff_vs_baseline']:.2f},{r['ci95_lo']:.2f},{r['ci95_hi']:.2f}"
                )
        (gen_dir / "sensitivity.csv").write_text("\n".join(csv_lines) + "\n")
        print(f"[{gen}] wrote sensitivity.{{json,csv}}")

        # Compact print: how does the headline survive?
        print(f"\n=== {gen} (prior={prior:.3f}) ===")
        print("D1 theta-sensitivity (bayesian_greedy diff vs always_verify):")
        for label, _, _ in perturb_grid:
            r = results["D1_theta_sensitivity"][label]["bayesian_greedy"]
            print(f"  {label:<12} util={r['mean_utility']:>+7.2f}  Δ={r['diff_vs_baseline']:>+6.2f}  "
                  f"CI=[{r['ci95_lo']:>+6.2f}, {r['ci95_hi']:>+6.2f}]")
        print("D2 c_ver sweep (bayesian_greedy vs threshold_L2):")
        for c_ver in c_ver_grid:
            d = results["D2_c_ver_sweep"][f"c_ver_{c_ver}"]
            bg, tL2 = d["bayesian_greedy"], d["threshold_L2"]
            print(f"  c_ver={c_ver:>3}  greedy={bg['mean_utility']:>+7.2f}  "
                  f"threshold_L2={tL2['mean_utility']:>+7.2f}  diff={bg['mean_utility']-tL2['mean_utility']:>+6.2f}")
        print("D3 verifier efficiency:")
        for name in ("always_verify", "threshold_L2", "bayesian_greedy", "bayesian_DP"):
            r = results["D3_verifier_efficiency"][name]
            vps = r["verify_per_solve"]
            vps_s = f"{vps:.2f}" if vps is not None else "n/a"
            print(f"  {name:<18} solved={r['n_solved']:>3}  verified={r['n_verified']:>3}  "
                  f"verify/solve={vps_s}")


if __name__ == "__main__":
    main()
