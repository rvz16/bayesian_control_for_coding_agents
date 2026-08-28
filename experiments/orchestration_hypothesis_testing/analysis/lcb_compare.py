"""Baseline-vs-controller comparison on LCB calibration data.

Adapts run_baseline_vs_controller.py to LCB's schema:
  - Records grouped by instance_id, sorted by patch_id
  - Each instance = one trajectory of length 3 (independent samples)
  - L2_public_tests is the THIRD informative critic (not present in SWE-bench Lite)

Adds:
  - Greedy controller (pre-registered §F1, currently missing)
  - Paired-bootstrap CIs on policy utility differences (B=1000)

Inputs per generator:
  data/lcb_calibration_v2/<gen>/critic_results.jsonl
  data/lcb_calibration_v2/<gen>/likelihood_tables.json

Output: <gen>/policy_comparison.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # .../orchestration_hypothesis_testing
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from calibration.lcb import canonical_generator_key  # noqa: E402
from analysis.controller import (  # noqa: E402
    BayesianController, CostModel, simulate_policy,
    policy_always_verify, policy_threshold_L0, policy_threshold_L3,
    policy_fixed_pipeline, policy_best_of_N,
)


def policy_threshold_L2(state, rec):
    """Threshold on L2_public_tests (LCB-only critic)."""
    if not state.get("L2_done"):
        state["L2_done"] = True
        return "L0"  # approximate L2 cost as L0 (cheap, both ~free)
    state["L2_done"] = False
    return "verify" if rec.get("L2_public_tests") else (
        "generate" if state["patch_idx"] + 1 < state.get("max_patches", 3) else "give_up"
    )


# ---------------------------------------------------------------------------
# Greedy controller (pre-registered §F1)
# ---------------------------------------------------------------------------

class GreedyController:
    """Myopic 1-step lookahead: at each state, pick the action with the
    highest immediate Q-value (no backward induction over future states).

    For comparison against the DP controller. The slide-8 inversion claims
    Greedy can beat DP under theta-misspecification — we measure the gap
    here.
    """

    def __init__(self, prior: float, like_tables: dict, cost: CostModel) -> None:
        self.prior = prior
        self.likes = like_tables["critic_likelihoods"]
        self.cost = cost

    def _bayes_update(self, b: float, critic: str, observed_pass: bool) -> float:
        l = self.likes[critic]
        if observed_pass:
            num = l["P_pass_given_Y1"] * b
            den = num + l["P_pass_given_Y0"] * (1 - b)
        else:
            num = (1 - l["P_pass_given_Y1"]) * b
            den = num + (1 - l["P_pass_given_Y0"]) * (1 - b)
        return num / max(den, 1e-12)

    def select_action(self, b: float, step: int) -> str:
        cm = self.cost
        # Q values: greedy = look 1 step ahead only
        Q_verify = cm.reward * b - cm.c_ver
        Q_giveup = 0.0
        Q_generate = -cm.c_gen + cm.reward * self.prior - cm.c_ver  # assume verify after gen

        def Q_critic(name: str, c: float) -> float:
            l = self.likes[name]
            p_pass = l["P_pass_given_Y1"] * b + l["P_pass_given_Y0"] * (1 - b)
            b_pass = self._bayes_update(b, name, True)
            b_fail = self._bayes_update(b, name, False)
            v_pass = max(cm.reward * b_pass - cm.c_ver, 0.0)
            v_fail = max(cm.reward * b_fail - cm.c_ver, 0.0)
            return -c + p_pass * v_pass + (1 - p_pass) * v_fail

        actions = {
            "verify": Q_verify, "give_up": Q_giveup, "generate": Q_generate,
        }
        if "L0_syntax" in self.likes:
            actions["L0"] = Q_critic("L0_syntax", cm.c_L0)
        if "L2_public_tests" in self.likes:
            actions["L2"] = Q_critic("L2_public_tests", cm.c_L2)
        if "L3_llm_review" in self.likes:
            actions["L3"] = Q_critic("L3_llm_review", cm.c_L3)
        return max(actions, key=lambda a: actions[a])


def make_greedy_policy(controller: GreedyController):
    def _p(state, rec):
        b = state.get("belief", controller.prior)
        a = controller.select_action(b, state["step"])
        if a == "L0":
            obs = bool(rec.get("L0_syntax"))
            state["belief"] = controller._bayes_update(b, "L0_syntax", obs)
            return "L0"
        if a == "L2":
            obs = bool(rec.get("L2_public_tests"))
            state["belief"] = controller._bayes_update(b, "L2_public_tests", obs)
            return "L2"
        if a == "L3":
            obs = bool(rec.get("L3_llm_review"))
            state["belief"] = controller._bayes_update(b, "L3_llm_review", obs)
            return "L3"
        return a
    return _p


# ---------------------------------------------------------------------------
# Trajectory loading (LCB-specific)
# ---------------------------------------------------------------------------

def load_lcb_trajectories(records_path: Path) -> dict[str, list[dict]]:
    """Group LCB calibration records by instance_id, sort by patch_id."""
    by_inst: dict[str, list[dict]] = {}
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            by_inst.setdefault(r["instance_id"], []).append(r)
    return {k: sorted(v, key=lambda r: r.get("patch_id", 0)) for k, v in by_inst.items()}


# ---------------------------------------------------------------------------
# Paired bootstrap CI
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(util_a: list[float], util_b: list[float],
                         n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Returns (mean_diff, ci_lo, ci_hi) at 95%.

    Tests A - B (positive = A wins). Paired by instance index.
    """
    arr_a = np.array(util_a)
    arr_b = np.array(util_b)
    diffs = arr_a - arr_b
    mean = float(diffs.mean())
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(diffs[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return mean, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generators", required=True)
    parser.add_argument("--c-gen", type=float, default=5.0)
    parser.add_argument("--c-l0", type=float, default=1.0)
    parser.add_argument("--c-l2", type=float, default=2.0)
    parser.add_argument("--c-l3", type=float, default=5.0)
    parser.add_argument("--c-ver", type=float, default=30.0)
    parser.add_argument("--reward", type=float, default=100.0)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--kernel-file", type=str, default=None,
                        help="Optional gen=path[,gen=path] mapping to load measured "
                             "transition kernels (P_fix_given_broken, P_break_given_correct) "
                             "from JSON. Default: synthesized IID kernel matching prior.")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to policy_comparison.json filename "
                             "(e.g. '_kernel_measured' to keep ablation runs separate).")
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    cost = CostModel(
        c_gen=args.c_gen,
        c_L0=args.c_l0,
        c_L2=args.c_l2,
        c_L3=args.c_l3,
        c_ver=args.c_ver,
        reward=args.reward,
    )

    # Parse kernel-file mapping
    kernel_paths: dict[str, Path] = {}
    if args.kernel_file:
        for pair in args.kernel_file.split(","):
            if "=" not in pair:
                continue
            k, v = pair.split("=", 1)
            kernel_paths[canonical_generator_key(k)] = Path(v.strip())

    for gen in [canonical_generator_key(g) for g in args.generators.split(",") if g.strip()]:
        gen_dir = out_dir / gen
        rec_path = gen_dir / "critic_results.jsonl"
        like_path = gen_dir / "likelihood_tables.json"
        if not rec_path.exists() or not like_path.exists():
            print(f"[{gen}] missing data, skipping")
            continue

        traj = load_lcb_trajectories(rec_path)
        likes = json.loads(like_path.read_text())
        prior = likes.get("prior_Y1", 0.5)

        # Default: IID kernel synthesized to match the prior
        kernel = {"kernel_all": {
            "P_fix_given_broken": prior,
            "P_break_given_correct": 1 - prior,
        }}
        kernel_source = "iid_synth"
        if gen in kernel_paths:
            kp = kernel_paths[gen]
            if kp.exists():
                full = json.loads(kp.read_text())
                if "kernel_all" in full:
                    kernel = {"kernel_all": full["kernel_all"]}
                else:
                    kernel = {"kernel_all": full}
                kernel_source = f"measured ({kp})"
                pf = kernel["kernel_all"].get("P_fix_given_broken")
                pb = kernel["kernel_all"].get("P_break_given_correct")
                print(f"[{gen}] using measured kernel: P_fix={pf:.3f}, P_break={pb:.3f}")
            else:
                print(f"[{gen}] kernel file {kp} not found; falling back to IID")

        # Build controllers
        dp = BayesianController(prior, likes, kernel, cost, horizon=args.horizon)
        greedy = GreedyController(prior, likes, cost)

        from analysis.controller import make_bayesian_policy
        policies = {
            "always_verify": policy_always_verify,
            "threshold_L0": policy_threshold_L0,
            "threshold_L2": policy_threshold_L2,
            "threshold_L3": policy_threshold_L3,
            "fixed_pipeline": policy_fixed_pipeline,
            f"best_of_{args.best_of}": policy_best_of_N(args.best_of),
            "bayesian_DP": make_bayesian_policy(dp),
            "bayesian_greedy": make_greedy_policy(greedy),
        }

        # Run each policy on each instance
        utils_per_policy: dict[str, list[float]] = {n: [] for n in policies}
        rewards_per_policy: dict[str, list[float]] = {n: [] for n in policies}
        for inst, t in traj.items():
            for name, fn in policies.items():
                r = simulate_policy(t, fn, cost)
                utils_per_policy[name].append(r["utility"])
                rewards_per_policy[name].append(r["reward"])

        # Mean / pass rate / paired bootstrap vs always_verify
        results = {}
        baseline_utils = utils_per_policy["always_verify"]
        for name in policies:
            u = utils_per_policy[name]
            r = rewards_per_policy[name]
            mean_u = float(np.mean(u))
            mean_pass = float(np.mean([rr > 0 for rr in r]))
            mean_diff, lo, hi = paired_bootstrap_ci(u, baseline_utils, args.n_boot)
            results[name] = {
                "mean_utility": mean_u,
                "pass_rate": mean_pass,
                "diff_vs_always_verify": mean_diff,
                "ci95_lo": lo,
                "ci95_hi": hi,
            }
        out_name = f"policy_comparison{args.out_suffix}.json"
        (gen_dir / out_name).write_text(json.dumps(
            {"kernel_source": kernel_source,
             "kernel_used": kernel["kernel_all"],
             "policies": results}, indent=2))

        # Print table
        print(f"\n=== {gen} (prior={prior:.3f}, n={len(traj)}, kernel={kernel_source}) ===")
        print(f"  {'policy':<20} {'utility':>9} {'pass':>7} {'Δ vs always_verify (95% CI)':>32}")
        print("  " + "-" * 72)
        for name, r in sorted(results.items(), key=lambda x: -x[1]["mean_utility"]):
            print(f"  {name:<20} {r['mean_utility']:>9.2f} {r['pass_rate']*100:>6.1f}% "
                  f"   {r['diff_vs_always_verify']:>+6.2f} [{r['ci95_lo']:>+6.2f}, {r['ci95_hi']:>+6.2f}]")


if __name__ == "__main__":
    main()
