"""For each SWE iter cell, compute:
  - transition_kernel.json (P(fix|broken), P(break|correct))
  - policy_comparison.json (Bayesian DP/Greedy + threshold + best_of_N)

Reads iter_records.jsonl with Y backfilled. Skips records with Y=None.
"""
import json, glob, sys
from pathlib import Path
from collections import defaultdict

EXP_DIR = Path(__file__).resolve().parents[1]  # orchestration_hypothesis_testing/

# Cost model (same as LCB)
COST_GEN = 5
COST_L0 = 1
COST_L3 = 5
COST_VER = 30
REWARD = 100


def compute_kernel(records):
    """records sorted within each instance by step. Returns kernel dict."""
    by_inst = defaultdict(list)
    for r in records:
        by_inst[r["instance_id"]].append(r)
    for v in by_inst.values():
        v.sort(key=lambda x: x["step"])

    fix = persist_broken = brk = persist_correct = 0
    for traj in by_inst.values():
        for i in range(len(traj) - 1):
            yk, yk1 = traj[i].get("Y"), traj[i + 1].get("Y")
            if yk is None or yk1 is None:
                continue
            if yk == 0 and yk1 == 1:
                fix += 1
            elif yk == 0 and yk1 == 0:
                persist_broken += 1
            elif yk == 1 and yk1 == 0:
                brk += 1
            elif yk == 1 and yk1 == 1:
                persist_correct += 1
    n_broken = fix + persist_broken
    n_correct = brk + persist_correct
    return {
        "fix_count": fix,
        "persist_broken_count": persist_broken,
        "break_count": brk,
        "persist_correct_count": persist_correct,
        "P_fix_given_broken": fix / n_broken if n_broken else 0,
        "P_break_given_correct": brk / n_correct if n_correct else 0,
        "n_transitions": fix + persist_broken + brk + persist_correct,
    }


def policy_always_verify(traj):
    """Verify the step-0 patch."""
    if not traj:
        return 0
    y = traj[0].get("Y")
    if y is None:
        return None
    return REWARD * y - COST_GEN - COST_VER


def policy_threshold_l3(traj):
    """Step 0: run L3. If PASS, verify. If FAIL, take last step (skip verify)."""
    if not traj:
        return 0
    r0 = traj[0]
    y = r0.get("Y")
    if y is None:
        return None
    l3 = r0.get("L3_llm_review")
    if l3:
        return REWARD * y - COST_GEN - COST_L3 - COST_VER
    else:
        return -COST_GEN - COST_L3  # skipped


def policy_fixed_pipeline(traj):
    """Step 0: run L0+L3. If both PASS, verify."""
    if not traj:
        return 0
    r0 = traj[0]
    y = r0.get("Y")
    if y is None:
        return None
    if r0.get("L0_syntax") and r0.get("L3_llm_review"):
        return REWARD * y - COST_GEN - COST_L0 - COST_L3 - COST_VER
    else:
        return -COST_GEN - COST_L0 - COST_L3


def policy_bayesian_greedy(traj, prior, l3_pass_y1, l3_pass_y0):
    """Run L3, update belief, verify if posterior > 0.5."""
    if not traj:
        return 0
    r0 = traj[0]
    y = r0.get("Y")
    if y is None:
        return None
    l3 = r0.get("L3_llm_review")
    if l3:
        prior_o = prior * l3_pass_y1
        marginal = prior * l3_pass_y1 + (1 - prior) * l3_pass_y0
    else:
        prior_o = prior * (1 - l3_pass_y1)
        marginal = prior * (1 - l3_pass_y1) + (1 - prior) * (1 - l3_pass_y0)
    posterior = prior_o / marginal if marginal > 0 else 0
    if posterior > 0.5:
        return REWARD * y - COST_GEN - COST_L3 - COST_VER
    else:
        return -COST_GEN - COST_L3


def best_of_n(traj_list_per_inst, n=3):
    """Best-of-N: run n predictions, verify last one, but no decision logic."""
    # for SWE iter we only have 1 trajectory per instance, so this is just always_verify on the n-th step
    if not traj_list_per_inst:
        return 0
    last = traj_list_per_inst[-1]
    y = last.get("Y")
    if y is None:
        return None
    return REWARD * y - n * COST_GEN - COST_VER


def evaluate_cell(cell_dir: Path, calib_dir: Path):
    """Compute kernel + policy_comparison for one iter cell."""
    iter_path = cell_dir / "iter_records.jsonl"
    if not iter_path.exists():
        return None
    records = [json.loads(l) for l in open(iter_path)]
    by_inst = defaultdict(list)
    for r in records:
        by_inst[r["instance_id"]].append(r)
    for v in by_inst.values():
        v.sort(key=lambda x: x["step"])

    # 1. transition kernel
    kernel = compute_kernel(records)

    # 2. likelihoods from calib
    lik_path = calib_dir / "likelihood_tables.json"
    if not lik_path.exists():
        return {"kernel": kernel, "error": "no calib likelihoods"}
    lik = json.loads(lik_path.read_text())
    prior = lik["prior_Y1"]
    l3 = lik["critic_likelihoods"]["L3_llm_review"]
    l3_pass_y1 = l3["P_pass_given_Y1"]
    l3_pass_y0 = l3["P_pass_given_Y0"]

    # 3. evaluate policies on each instance's first-step record
    policies = {
        "always_verify": [],
        "threshold_L3": [],
        "fixed_pipeline": [],
        "bayesian_greedy": [],
        "best_of_3": [],
    }
    for traj in by_inst.values():
        u_av = policy_always_verify(traj)
        u_t3 = policy_threshold_l3(traj)
        u_fp = policy_fixed_pipeline(traj)
        u_bg = policy_bayesian_greedy(traj, prior, l3_pass_y1, l3_pass_y0)
        u_bn = best_of_n(traj, n=3)
        for k, v in zip(policies.keys(), [u_av, u_t3, u_fp, u_bg, u_bn]):
            if v is not None:
                policies[k].append(v)

    # 4. summarise
    out = {"kernel_source": "measured_iter", "policies": {}}
    av_mean = sum(policies["always_verify"]) / len(policies["always_verify"]) if policies["always_verify"] else 0
    n_av = len(policies["always_verify"])
    pass_rate_av = sum(1 for v in policies["always_verify"] if v > 0) / n_av if n_av else 0
    for name, vals in policies.items():
        n = len(vals)
        if n == 0:
            out["policies"][name] = {"n": 0}
            continue
        mean = sum(vals) / n
        pass_rate = sum(1 for v in vals if v > 0) / n
        diff = mean - av_mean if name != "always_verify" else 0
        out["policies"][name] = {
            "mean_utility": mean,
            "pass_rate": pass_rate,
            "diff_vs_always_verify": diff,
            "n_instances": n,
        }
    out["kernel"] = kernel
    return out


def main():
    n_done = 0
    for bench in ["lite", "verified"]:
        for cell_dir in sorted((EXP_DIR / f"data/swebench_{bench}_realbaselines").glob("*/*")):
            gen = cell_dir.parts[-2]
            method = cell_dir.parts[-1]
            # Find calibration likelihood for this gen+bench
            if gen == "qwen25_32b":
                calib_dir = EXP_DIR / f"data/swebench_{bench}_qwen32b/qwen25_32b"
            else:
                calib_dir = EXP_DIR / f"data/swebench_{bench}/{gen}"
            result = evaluate_cell(cell_dir, calib_dir)
            if result is None:
                continue
            (cell_dir / "transition_kernel.json").write_text(
                json.dumps(result.get("kernel", {}), indent=2))
            policy_path = cell_dir / "policy_comparison.json"
            policy_dict = {k: v for k, v in result.items() if k != "kernel"}
            policy_path.write_text(json.dumps(policy_dict, indent=2))
            print(f"  {bench}/{gen}/{method}: kernel + policy written")
            n_done += 1
    print(f"\nDone: {n_done} cells")


if __name__ == "__main__":
    main()
