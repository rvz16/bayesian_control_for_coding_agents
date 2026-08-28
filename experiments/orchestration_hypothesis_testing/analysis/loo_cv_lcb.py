"""Leave-one-out cross-validation for the Bayesian policies on LCB cells.

Addresses the train/test split methodological concern raised in
[#issuecomment-4386429236]: critic likelihoods are estimated from the same
sample we evaluate policies on. LOO-CV recomputes likelihoods n times
(each excluding 1 instance) and evaluates the policy on the held-out
instance. Aggregating gives an honest out-of-fold utility.

Only the BAYESIAN policies (greedy, DP) depend on the estimated likelihoods.
The threshold/best-of-N/always_verify policies use raw critic outcomes and
are unaffected by LOO; we recompute them too for consistency in the output
schema but expect their utilities to match the in-sample numbers.

Inputs (per generator):
  data/lcb_calibration_v2/<gen>/critic_results.jsonl

Output (per generator):
  data/lcb_calibration_v2/<gen>/policy_comparison_loo.json

Usage:
  python3 scripts/loo_cv_lcb.py \\
      --src-dir data/lcb_calibration_v2 \\
      --generators gpt5_mini,qwen3_coder,haiku45,sonnet45
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Import controller classes from the existing module
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from analysis.controller import (  # noqa: E402
    BayesianController, CostModel, simulate_policy,
    policy_always_verify, policy_threshold_L0, policy_threshold_L3,
    policy_fixed_pipeline, policy_best_of_N,
    make_bayesian_policy,
)

# Re-import GreedyController + threshold_L2 + greedy policy from lcb_compare
from analysis.lcb_compare import (  # noqa: E402
    GreedyController, make_greedy_policy, policy_threshold_L2,
)


# ----- Likelihood estimation (Beta(1,1)-smoothed counts) --------------------

def estimate_likelihoods(records: list[dict]) -> dict:
    """Mirror compute_likelihoods from lcb_calibrate.py — Beta(1,1) smoothing."""
    rows = [r for r in records if r.get("Y") in (0, 1)]
    n_y1 = sum(1 for r in rows if r["Y"] == 1)
    n_total = len(rows)
    prior = (n_y1 + 1) / (n_total + 2)
    likelihoods: dict[str, dict] = {}
    for k in ("L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"):
        tp = sum(1 for r in rows if r["Y"] == 1 and r.get(k) is True)
        fn = sum(1 for r in rows if r["Y"] == 1 and r.get(k) is False)
        fp = sum(1 for r in rows if r["Y"] == 0 and r.get(k) is True)
        tn = sum(1 for r in rows if r["Y"] == 0 and r.get(k) is False)
        n_y1_with = tp + fn
        n_y0_with = fp + tn
        if n_y1_with == 0 or n_y0_with == 0:
            continue
        p_y1 = (tp + 1) / (n_y1_with + 2)
        p_y0 = (fp + 1) / (n_y0_with + 2)
        likelihoods[k] = {
            "P_pass_given_Y1": p_y1, "P_pass_given_Y0": p_y0,
            "gap": p_y1 - p_y0, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        }
    return {"prior_Y1": prior, "critic_likelihoods": likelihoods}


# ----- LOO loop --------------------------------------------------------------

def load_records(path: Path) -> dict[str, list[dict]]:
    """Group critic_results.jsonl by instance_id, sort by patch_id."""
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "instance_id" not in r:
            continue
        by_inst[r["instance_id"]].append(r)
    for inst, rs in by_inst.items():
        rs.sort(key=lambda r: r.get("patch_id", 0))
    return dict(by_inst)


def run_loo_for_gen(src_dir: Path, gen: str, cost: CostModel, n_boot: int) -> dict:
    records_path = src_dir / gen / "critic_results.jsonl"
    if not records_path.exists():
        return {"skipped": f"no critic_results at {records_path}"}
    by_inst = load_records(records_path)
    insts = sorted(by_inst.keys())
    n = len(insts)
    if n < 3:
        return {"skipped": f"too few instances ({n})"}

    # Per-policy per-instance utility list
    pol_utils: dict[str, list[float]] = defaultdict(list)

    for held_out in insts:
        # Build calibration set excluding held_out
        calib_records = []
        for inst in insts:
            if inst == held_out:
                continue
            calib_records.extend(by_inst[inst])

        # Estimate likelihoods from calibration set
        like = estimate_likelihoods(calib_records)
        if not like["critic_likelihoods"]:
            continue
        prior = like["prior_Y1"]

        # Build controllers from out-of-fold likelihoods. BayesianController
        # requires a kernel; we don't have a measured transition kernel inside
        # a calibration-only LOO loop, so we synthesise the IID kernel
        # (post-regen belief = prior). This matches the kernel that
        # run_policies builds for its bayesian_DP path when no measured
        # kernel is available -- see lcb_sensitivity.run_policies line 78.
        iid_kernel = {"kernel_all": {"P_fix_given_broken": prior,
                                      "P_break_given_correct": 1 - prior}}
        dp = BayesianController(prior=prior, like_tables=like,
                                kernel=iid_kernel, cost=cost)
        greedy = GreedyController(prior=prior, like_tables=like, cost=cost)

        traj = by_inst[held_out]

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

        for name, fn in policies.items():
            try:
                r = simulate_policy(traj, fn, cost)
                pol_utils[name].append(r["utility"])
            except Exception:
                # Skip failing policy/trajectory; keep loop going
                continue

    if not pol_utils:
        return {"skipped": "no policy utilities collected"}

    # Aggregate
    summary = {}
    base = pol_utils.get("always_verify", [])
    rng = random.Random(42)
    for name, utils in pol_utils.items():
        if len(utils) == 0:
            continue
        mean = sum(utils) / len(utils)
        # Paired bootstrap vs always_verify
        if base and len(base) == len(utils):
            diffs = [utils[i] - base[i] for i in range(len(utils))]
            mean_diff = sum(diffs) / len(diffs)
            boot = []
            for _ in range(n_boot):
                idxs = [rng.randrange(len(diffs)) for _ in range(len(diffs))]
                boot.append(sum(diffs[i] for i in idxs) / len(diffs))
            boot.sort()
            lo = boot[int(0.025 * n_boot)]
            hi = boot[int(0.975 * n_boot)]
        else:
            mean_diff, lo, hi = 0.0, 0.0, 0.0
        summary[name] = {
            "mean_utility": mean,
            "diff_vs_always_verify": mean_diff,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_folds": len(utils),
        }

    return {
        "method": "leave-one-instance-out CV",
        "n_instances": n,
        "n_folds_completed": min(len(v) for v in pol_utils.values() if v),
        "policies": summary,
        "cost_model": {
            "c_gen": cost.c_gen, "c_L0": cost.c_L0,
            "c_L3": cost.c_L3, "c_ver": cost.c_ver, "reward": cost.reward,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", required=True, type=Path)
    parser.add_argument("--generators", default="gpt5_mini,qwen3_coder,haiku45,sonnet45")
    parser.add_argument("--c-ver", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out-suffix", default="_loo")
    args = parser.parse_args()

    cost = CostModel(c_gen=10, c_L0=1, c_L3=5, c_ver=args.c_ver, reward=100)
    if hasattr(cost, "c_L2"):
        cost.c_L2 = 2

    gens = [g.strip() for g in args.generators.split(",") if g.strip()]

    print(f"Source: {args.src_dir}")
    print(f"Cost model: c_gen={cost.c_gen} c_L0={cost.c_L0} c_L3={cost.c_L3} "
          f"c_ver={cost.c_ver} reward={cost.reward}")
    print()

    for gen in gens:
        print(f"=== {gen} ===")
        result = run_loo_for_gen(args.src_dir, gen, cost, args.n_boot)
        if "skipped" in result:
            print(f"  SKIPPED: {result['skipped']}")
            continue
        out_path = args.src_dir / gen / f"policy_comparison{args.out_suffix}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"policies": result["policies"],
                                         "method": result["method"],
                                         "n_instances": result["n_instances"],
                                         "cost_model": result["cost_model"]}, indent=2))

        # Print summary
        print(f"  n_instances={result['n_instances']} n_folds_completed={result['n_folds_completed']}")
        print(f"  {'policy':<20} {'mean_util':>10} {'Δ vs always_verify':>22} {'95% CI':>22}")
        # Sort by Δ desc
        items = sorted(result["policies"].items(),
                       key=lambda kv: -kv[1].get("diff_vs_always_verify", 0))
        for name, s in items:
            ci = f"[{s['ci95_lo']:+.2f}, {s['ci95_hi']:+.2f}]"
            print(f"  {name:<20} {s['mean_utility']:>+10.2f} {s['diff_vs_always_verify']:>+22.2f} {ci:>22}")
        print(f"  -> wrote {out_path.name}")
        print()


if __name__ == "__main__":
    main()
