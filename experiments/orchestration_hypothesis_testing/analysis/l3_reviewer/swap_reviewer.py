"""Replay lcb_compare with a chosen L3 reviewer (from L3_sweep.jsonl)
substituted into both the trajectories and the likelihood tables.

Inputs per generator (must already exist):
  <gen>/critic_results.jsonl      — base records with L0/L2/Y
  <gen>/L3_sweep.jsonl            — multi-reviewer L3 columns

Output:
  <gen>/policy_comparison_l3_<reviewer>.json

Usage:
  python lcb_compare_swap_reviewer.py \\
    --output-dir data/lcb_calibration_v2 \\
    --generators gpt5_mini,qwen3_coder,haiku45 \\
    --reviewer gpt4omini
"""
from __future__ import annotations

import argparse
import json
import sys
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
    paired_bootstrap_ci,
)


def build_likes_with_reviewer(rows: list[dict], reviewer_label: str) -> dict:
    """Compute likelihood tables matching lcb_calibrate's schema, using the
    chosen reviewer's L3 column instead of the canonical L3_llm_review."""
    n = len(rows)
    n_y1 = sum(1 for r in rows if r["Y"] == 1)
    n_y0 = n - n_y1

    def likes_for(field: str, alt_field: str | None = None) -> dict:
        f = alt_field or field
        TP = sum(1 for r in rows if r["Y"] == 1 and r.get(f))
        FP = sum(1 for r in rows if r["Y"] == 0 and r.get(f))
        FN = n_y1 - TP
        TN = n_y0 - FP
        # Beta(1,1)
        p1 = (TP + 1) / (n_y1 + 2)
        p0 = (FP + 1) / (n_y0 + 2)
        return {"P_pass_given_Y1": p1, "P_pass_given_Y0": p0, "gap": p1 - p0,
                "TP": TP, "FN": FN, "FP": FP, "TN": TN}

    return {
        "prior_Y1": (n_y1 + 1) / (n + 2),  # Beta(1,1) on prior too
        "n_records": n,
        "n_resolved": n_y1,
        "critic_likelihoods": {
            "L0_syntax": likes_for("L0_syntax"),
            "L2_public_tests": likes_for("L2_public_tests"),
            # After merge_records_with_reviewer, L3_llm_review already holds
            # the swept reviewer's value. No alt_field needed.
            "L3_llm_review": likes_for("L3_llm_review"),
        },
        "smoothing": "Beta(1,1)",
    }


def merge_records_with_reviewer(crit_path: Path, sweep_path: Path,
                                 reviewer_label: str) -> list[dict]:
    """Join critic_results.jsonl with L3_sweep.jsonl on (inst, pid).
    Replace each row's L3_llm_review with the L3_<reviewer> from the sweep."""
    base = [json.loads(l) for l in open(crit_path) if l.strip()]
    # Dedup sweep (latest wins)
    sweep_latest: dict[tuple, dict] = {}
    for line in open(sweep_path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (str(r["instance_id"]), int(r["patch_id"]))
        prev = sweep_latest.get(key, {})
        merged = dict(prev)
        for k, v in r.items():
            if v is None and k in merged and merged[k] is not None:
                continue
            merged[k] = v
        sweep_latest[key] = merged

    out = []
    rev_field = f"L3_{reviewer_label}"
    for r in base:
        key = (str(r["instance_id"]), int(r["patch_id"]))
        sw = sweep_latest.get(key, {})
        new_l3 = sw.get(rev_field)
        new_r = dict(r)
        new_r["L3_llm_review"] = new_l3
        out.append(new_r)
    return out


def load_trajectories_from_records(records: list[dict]) -> dict[str, list[dict]]:
    by_inst: dict[str, list[dict]] = {}
    for r in records:
        by_inst.setdefault(str(r["instance_id"]), []).append(r)
    return {k: sorted(v, key=lambda r: r.get("patch_id", 0)) for k, v in by_inst.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--reviewer", required=True,
                        help="reviewer label as it appears in L3_sweep.jsonl (e.g., gpt4omini)")
    parser.add_argument("--c-ver", type=float, default=30.0)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    cost = CostModel(c_gen=10, c_L0=1, c_L2=2, c_L3=5, c_ver=args.c_ver, reward=100)
    out_dir = args.output_dir.resolve()

    for gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen_dir = out_dir / gen
        crit_path = gen_dir / "critic_results.jsonl"
        sweep_path = gen_dir / "L3_sweep.jsonl"
        if not crit_path.exists() or not sweep_path.exists():
            print(f"[{gen}] missing data, skipping")
            continue

        records = merge_records_with_reviewer(crit_path, sweep_path, args.reviewer)
        n_unset = sum(1 for r in records if r["L3_llm_review"] is None)
        if n_unset:
            print(f"[{gen}] WARN: {n_unset}/{len(records)} records have no {args.reviewer} review (treating as False)")
        # Coerce to bool so downstream Bayes update doesn't choke
        for r in records:
            if r["L3_llm_review"] is None:
                r["L3_llm_review"] = False

        likes = build_likes_with_reviewer(records, args.reviewer)
        prior = likes["prior_Y1"]
        traj = load_trajectories_from_records(records)

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
        for inst, t in traj.items():
            for name, fn in policies.items():
                r = simulate_policy(t, fn, cost)
                utils[name].append(r["utility"])
                rewards[name].append(r["reward"])

        base_u = utils["always_verify"]
        results = {"reviewer": args.reviewer, "prior": prior, "n_instances": len(traj),
                   "L3_gap_with_reviewer": likes["critic_likelihoods"]["L3_llm_review"]["gap"],
                   "policies": {}}
        for name in policies:
            u = utils[name]
            r = rewards[name]
            mean_u = float(np.mean(u))
            pass_rate = float(np.mean([rr > 0 for rr in r]))
            mean_diff, lo, hi = paired_bootstrap_ci(u, base_u, args.n_boot)
            results["policies"][name] = {
                "mean_utility": mean_u, "pass_rate": pass_rate,
                "diff_vs_always_verify": mean_diff,
                "ci95_lo": lo, "ci95_hi": hi,
            }
        out_path = gen_dir / f"policy_comparison_l3_{args.reviewer}.json"
        out_path.write_text(json.dumps(results, indent=2))

        print(f"\n=== {gen} | L3={args.reviewer} (prior={prior:.3f}, L3_gap={results['L3_gap_with_reviewer']:+.3f}, n={len(traj)}) ===")
        print(f"  {'policy':<20} {'utility':>9} {'pass':>7} {'diff':>10} {'95% CI':>22}")
        for name in sorted(results["policies"], key=lambda x: -results["policies"][x]["mean_utility"]):
            r = results["policies"][name]
            print(f"  {name:<20} {r['mean_utility']:>+9.2f} {r['pass_rate']*100:>6.1f}% "
                  f"{r['diff_vs_always_verify']:>+10.2f} [{r['ci95_lo']:>+6.2f}, {r['ci95_hi']:>+6.2f}]")


if __name__ == "__main__":
    main()
