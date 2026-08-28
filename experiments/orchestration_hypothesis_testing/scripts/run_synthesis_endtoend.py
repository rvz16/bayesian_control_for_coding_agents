#!/usr/bin/env python
"""End-to-end Bayesian agent on synthesis benchmarks (LCB / MBPP+ / HumanEval+).

Mirrors the train/test split + GrFt/DPFt structure of the abbo bug-fix runners
(run_codecontests_full.py, run_humaneval_full.py) but for synthesis tasks.

Workflow:
  1. Read split.json + likelihood_tables.json produced by
     synthesis_train_test_split.py (train-only theta fit + held-out test ids).
  2. For each test_id, replay all 9 policy variants using the cached
     critic outcomes in critic_results.jsonl (one patch per instance under
     --n-patches=1, more if you re-ran calibration with --n-patches >= 3).
  3. Apply the abbo DPPlanner / greedy-Q-lookahead with fitted (train) theta
     for the BAYESIAN variants (greedy_fitted = GrFt, dp_fitted = DPFt).
  4. Save per-cell results in the same JSON schema as
     bayesian_optimization_for_code_testing/agent-bugfix-bayes/sim_results/
     *_full_endtoend__*.json — wandb upload via existing upload_runs.py picks
     them up automatically.

Variants implemented (replay-style on cached patches):
  simple, best_of_3, threshold_L0, threshold_L2, threshold_L3,
  fixed_pipeline, greedy_hand, greedy_fitted (GrFt),
  dp_hand, dp_fitted (DPFt)

Variants NOT implemented (require live LLM): self_refine, reflexion.

Usage:
  python scripts/run_synthesis_endtoend.py \\
      --src-dir data/lcb_calibration_medium \\
      --benchmark lcb_medium \\
      --generator haiku45 \\
      --output sim_results/synthesis_endtoend__lcb_medium__haiku45.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABBO_SRC = ROOT.parent.parent / "bayesian_optimization_for_code_testing" \
                              / "agent-bugfix-bayes" / "src"
sys.path.insert(0, str(ABBO_SRC))

# DPPlanner + bayes_update from the abbo codebase
from abbo.realworld.agents.bayes_agent import DPPlanner, bayes_update  # noqa: E402

log = logging.getLogger("synth_endtoend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# ============================================================================
# Constants matching abbo bug-fix conventions
# ============================================================================
CRITIC_KEYS = ["L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"]
PRIOR = 0.5
SPLIT_SEED = 42  # must match synthesis_train_test_split.py default

# Cost vector for synthesis benchmarks (fast-oracle):
# c_ver=5, c_critic=1 each, c_gen=10, R=100
@dataclass
class Costs:
    c_gen: int = 10
    c_critic: int = 1
    c_ver: int = 5
    reward: int = 100


# ============================================================================
# Helpers — load cached calibration data + train-fitted theta
# ============================================================================
def load_records(jsonl_path: Path) -> dict[str, list[dict]]:
    """Group critic_results.jsonl rows by instance_id, sorted by patch_id."""
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        iid = str(r.get("instance_id") or r.get("question_id"))
        by_inst[iid].append(r)
    for iid in by_inst:
        by_inst[iid].sort(key=lambda r: int(r.get("patch_id", 0)))
    return dict(by_inst)


def load_likelihoods(gen_dir: Path) -> dict:
    """Return {critic_name: {p_pass_y1, p_pass_y0}} in the format the abbo
    bayes_update / DPPlanner expects."""
    j = json.loads((gen_dir / "likelihood_tables.json").read_text())
    out = {}
    for name, lk in j["critic_likelihoods"].items():
        out[name] = {
            "p_pass_y1": lk.get("P_pass_given_Y1", lk.get("p_pass_y1")),
            "p_pass_y0": lk.get("P_pass_given_Y0", lk.get("p_pass_y0")),
        }
    return out, j.get("prior_Y1", 0.5)


def load_split(gen_dir: Path) -> tuple[list[str], list[str]]:
    j = json.loads((gen_dir / "split.json").read_text())
    return [str(x) for x in j["train_ids"]], [str(x) for x in j["test_ids"]]


# ============================================================================
# Result dataclass — matches abbo Result schema (per-task per-variant)
# ============================================================================
@dataclass
class Result:
    task_id: str
    variant: str
    fixed: bool = False
    total_cost: float = 0.0
    n_llm_calls: int = 0
    n_critic_runs: int = 0
    n_full_tests: int = 0
    final_action: str = ""
    actions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "variant": self.variant,
            "fixed": self.fixed,
            "total_cost": self.total_cost,
            "n_llm_calls": self.n_llm_calls,
            "n_critic_runs": self.n_critic_runs,
            "n_full_tests": self.n_full_tests,
            "final_action": self.final_action,
            "actions": self.actions,
        }


# ============================================================================
# Variant runners — REPLAY style, consuming cached patches in order
# ============================================================================
def _get_patch(patches: list[dict], idx: int) -> dict | None:
    """Return the idx-th cached patch outcome, or None if not available."""
    if idx < len(patches):
        return patches[idx]
    return None


def run_simple(task_id: str, patches: list[dict], costs: Costs) -> Result:
    """Generate → verify. Returns first patch's outcome."""
    res = Result(task_id=task_id, variant="simple")
    p = _get_patch(patches, 0)
    if p is None:
        res.final_action = "no_patch"
        return res
    # generate
    res.n_llm_calls += 1
    res.total_cost += costs.c_gen
    res.actions.append({"step": 0, "action": "generate", "patch_id": 0})
    # verify
    res.n_full_tests += 1
    res.total_cost += costs.c_ver
    ok = bool(int(p.get("Y", 0)))
    res.fixed = ok
    res.final_action = "verify_pass" if ok else "verify_fail"
    res.actions.append({"step": 1, "action": "verify", "ok": ok})
    return res


def run_best_of_n(task_id: str, patches: list[dict], costs: Costs, n: int = 3) -> Result:
    """Generate N patches, verify each, accept first that passes (else last)."""
    res = Result(task_id=task_id, variant="best_of_3")
    fixed_any = False
    for k in range(min(n, len(patches))):
        p = patches[k]
        res.n_llm_calls += 1
        res.total_cost += costs.c_gen
        res.actions.append({"step": 2*k, "action": "generate", "patch_id": k})
        res.n_full_tests += 1
        res.total_cost += costs.c_ver
        ok = bool(int(p.get("Y", 0)))
        res.actions.append({"step": 2*k+1, "action": "verify", "ok": ok})
        if ok:
            fixed_any = True
            break
    res.fixed = fixed_any
    res.final_action = "verify_pass" if fixed_any else "exhausted"
    return res


def run_threshold(task_id: str, patches: list[dict], costs: Costs,
                  critics: list[str], variant_name: str) -> Result:
    """Generate → run each critic in `critics` order; if all pass, verify."""
    res = Result(task_id=task_id, variant=variant_name)
    p = _get_patch(patches, 0)
    if p is None:
        res.final_action = "no_patch"
        return res
    res.n_llm_calls += 1
    res.total_cost += costs.c_gen
    res.actions.append({"step": 0, "action": "generate", "patch_id": 0})
    for k, cn in enumerate(critics):
        res.n_critic_runs += 1
        res.total_cost += costs.c_critic
        passed = bool(p.get(cn, False))
        res.actions.append({"step": 1+k, "action": f"critic:{cn}", "passed": passed})
        if not passed:
            res.final_action = "critic_reject"
            return res
    # All critics passed → verify
    res.n_full_tests += 1
    res.total_cost += costs.c_ver
    ok = bool(int(p.get("Y", 0)))
    res.fixed = ok
    res.final_action = "verify_pass" if ok else "verify_fail"
    res.actions.append({"step": 1+len(critics), "action": "verify", "ok": ok})
    return res


def run_fixed_pipeline(task_id: str, patches: list[dict], costs: Costs) -> Result:
    """Generate → run ALL critics → verify regardless of critic outcomes."""
    res = Result(task_id=task_id, variant="fixed_pipeline")
    p = _get_patch(patches, 0)
    if p is None:
        res.final_action = "no_patch"
        return res
    res.n_llm_calls += 1
    res.total_cost += costs.c_gen
    res.actions.append({"step": 0, "action": "generate", "patch_id": 0})
    for k, cn in enumerate(CRITIC_KEYS):
        res.n_critic_runs += 1
        res.total_cost += costs.c_critic
        passed = bool(p.get(cn, False))
        res.actions.append({"step": 1+k, "action": f"critic:{cn}", "passed": passed})
    res.n_full_tests += 1
    res.total_cost += costs.c_ver
    ok = bool(int(p.get("Y", 0)))
    res.fixed = ok
    res.final_action = "verify_pass" if ok else "verify_fail"
    res.actions.append({"step": 1+len(CRITIC_KEYS), "action": "verify", "ok": ok})
    return res


def _q_critic_one_step(b: float, critic_name: str, theta: dict, costs: Costs) -> float:
    """One-step lookahead Q-value for running `critic_name` next."""
    lk = theta[critic_name]
    p_pass = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)
    b_pass = lk["p_pass_y1"] * b / max(p_pass, 1e-12)
    b_fail_denom = (1 - lk["p_pass_y1"]) * b + (1 - lk["p_pass_y0"]) * (1 - b)
    b_fail = (1 - lk["p_pass_y1"]) * b / max(b_fail_denom, 1e-12)
    return (
        -costs.c_critic
        + p_pass * max(0.0, -costs.c_ver + b_pass * costs.reward)
        + (1 - p_pass) * max(0.0, -costs.c_ver + b_fail * costs.reward)
    )


def run_greedy(task_id: str, patches: list[dict], theta: dict, costs: Costs,
               label: str, prior: float = PRIOR, max_gen: int = 3) -> Result:
    """One-step lookahead Bayesian agent. Replay-style on cached patches."""
    res = Result(task_id=task_id, variant=f"greedy_{label}")
    belief = prior
    gen_left = max_gen
    crit_used: set[str] = set()
    patch_idx = -1
    step = 0
    current_patch: dict | None = None

    while step < 12:
        # Compute Q-values for available actions
        Q_bail = 0.0
        Q_verify = -costs.c_ver + belief * costs.reward
        Q_critics = {}
        if current_patch is not None:
            for cn in theta:
                if cn not in crit_used:
                    Q_critics[cn] = _q_critic_one_step(belief, cn, theta, costs)
        best_c, best_q = (max(Q_critics.items(), key=lambda x: x[1])
                          if Q_critics else (None, -math.inf))
        Q_gen = -math.inf
        if gen_left > 0:
            # Post-generate belief approximation (same as abbo: optimistic transition)
            b_after = belief * 0.95 + (1 - belief) * 0.50
            Q_gen = -costs.c_gen - costs.c_ver + b_after * costs.reward

        # Force first action to be `generate` (no patch yet)
        if current_patch is None:
            action = "generate"
        else:
            choices = [("bail", Q_bail), ("verify", Q_verify)]
            if best_c:
                choices.append((f"critic:{best_c}", best_q))
            if gen_left > 0:
                choices.append(("generate", Q_gen))
            action, _q = max(choices, key=lambda x: x[1])

        if action == "bail":
            res.final_action = "bail"
            break

        if action == "verify":
            res.n_full_tests += 1
            res.total_cost += costs.c_ver
            ok = bool(int(current_patch.get("Y", 0)))
            res.actions.append({"step": step, "action": "verify", "ok": ok})
            if ok:
                res.fixed = True
                res.final_action = "verify_pass"
                break
            belief = 0.05  # failed verify → near-certain Y=0
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            passed = bool(current_patch.get(cn, False))
            res.n_critic_runs += 1
            res.total_cost += costs.c_critic
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used.add(cn)
            res.actions.append({"step": step, "action": action,
                                "passed": passed, "b": belief})
        else:  # generate
            patch_idx += 1
            new_patch = _get_patch(patches, patch_idx)
            if new_patch is None:
                # No more cached patches → bail
                res.final_action = "exhausted_patches"
                break
            current_patch = new_patch
            res.n_llm_calls += 1
            res.total_cost += costs.c_gen
            gen_left -= 1
            belief = belief * 0.95 + (1 - belief) * 0.50  # transition kernel
            crit_used = set()
            res.actions.append({"step": step, "action": "generate",
                                "patch_id": patch_idx, "b": belief})
        step += 1

    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    return res


def run_dp(task_id: str, patches: list[dict], theta: dict, costs: Costs,
           planner: DPPlanner, label: str, prior: float = PRIOR,
           max_gen: int = 3, max_ver: int = 2) -> Result:
    """Full-DP Bayesian agent via abbo DPPlanner. Replay-style."""
    res = Result(task_id=task_id, variant=f"dp_{label}")
    belief = prior
    gen_left = max_gen
    ver_left = max_ver
    crit_used: frozenset[str] = frozenset()
    patch_idx = -1
    step = 0
    current_patch: dict | None = None

    # Force first action to be `generate` (no patch yet)
    while step < 16:
        if current_patch is None:
            action = "generate:initial"
        else:
            action, _q = planner.choose_action(belief, gen_left, crit_used, ver_left)

        if action == "bail_out":
            res.final_action = "bail"
            break

        if action == "verify":
            res.n_full_tests += 1
            res.total_cost += costs.c_ver
            ok = bool(int(current_patch.get("Y", 0)))
            ver_left -= 1
            res.actions.append({"step": step, "action": "verify", "ok": ok})
            if ok:
                res.fixed = True
                res.final_action = "verify_pass"
                break
            belief = 0.05
        elif action.startswith("critic:"):
            cn = action.split(":", 1)[1]
            passed = bool(current_patch.get(cn, False))
            res.n_critic_runs += 1
            res.total_cost += costs.c_critic
            belief = bayes_update(belief, cn, passed, likelihoods=theta)
            crit_used = crit_used | frozenset([cn])
            res.actions.append({"step": step, "action": action,
                                "passed": passed, "b": belief})
        elif action.startswith("generate"):
            patch_idx += 1
            new_patch = _get_patch(patches, patch_idx)
            if new_patch is None:
                res.final_action = "exhausted_patches"
                break
            current_patch = new_patch
            res.n_llm_calls += 1
            res.total_cost += costs.c_gen
            gen_left -= 1
            belief = belief * 0.95 + (1 - belief) * 0.50
            crit_used = frozenset()
            res.actions.append({"step": step, "action": "generate",
                                "patch_id": patch_idx, "b": belief})
        step += 1

    if not res.fixed and not res.final_action:
        res.final_action = "exhausted"
    return res


# ============================================================================
# Main loop
# ============================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir", required=True, type=Path,
                   help="Calibration data dir (with <gen>/{critic_results,likelihood_tables,split}.json)")
    p.add_argument("--benchmark", required=True,
                   help="Benchmark slug for output metadata (e.g. lcb_medium, mbpp, humaneval)")
    p.add_argument("--generator", required=True,
                   help="Generator slug (must match a subdir of --src-dir)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output JSON path (e.g. sim_results/synthesis_endtoend__lcb_medium__haiku45.json)")
    p.add_argument("--hand-theta-from", default="",
                   help="Optional path to hand-tuned likelihood_tables.json for greedy_hand/dp_hand. "
                        "If empty, hand variants use the same fitted theta (effectively identical to fitted variants).")
    p.add_argument("--variants", default="all",
                   help="Comma-separated subset of variants to run. Default 'all' runs all 10. "
                        "Use 'fitted' shorthand for just greedy_fitted+dp_fitted (GrFt+DPFt). "
                        "Or list explicitly: greedy_fitted,dp_fitted,simple")
    args = p.parse_args()

    gen_dir = (args.src_dir / args.generator).resolve()
    cr_path = gen_dir / "critic_results.jsonl"
    if not cr_path.exists():
        raise SystemExit(f"missing {cr_path}")

    train_ids, test_ids = load_split(gen_dir)
    log.info("loaded split: %d train, %d test", len(train_ids), len(test_ids))

    fitted_theta, prior_y1 = load_likelihoods(gen_dir)
    log.info("loaded fitted theta: prior_Y1=%.3f, critics=%s",
             prior_y1, list(fitted_theta.keys()))

    # Hand-tuned theta (optional override). For synthesis we don't have a
    # canonical hand-tuned table; default to same as fitted so hand/fitted
    # variants are sentinel-different in name only unless user provides one.
    if args.hand_theta_from:
        hand_path = Path(args.hand_theta_from)
        hand_j = json.loads(hand_path.read_text())
        hand_theta = {n: {"p_pass_y1": lk.get("P_pass_given_Y1", lk.get("p_pass_y1")),
                          "p_pass_y0": lk.get("P_pass_given_Y0", lk.get("p_pass_y0"))}
                      for n, lk in hand_j["critic_likelihoods"].items()}
        log.info("loaded hand theta from %s", hand_path)
    else:
        hand_theta = fitted_theta
        log.info("no --hand-theta-from: using fitted theta for hand variants too")

    records_by_inst = load_records(cr_path)
    costs = Costs()
    dp_hand = DPPlanner(costs, max_generators=3, max_verifications=2,
                        critic_likelihoods=hand_theta)
    dp_hand.solve()
    dp_fitted = DPPlanner(costs, max_generators=3, max_verifications=2,
                          critic_likelihoods=fitted_theta)
    dp_fitted.solve()

    # Run all variants on test split only
    state: dict = {"results": {}, "n_train": len(train_ids), "n_test": len(test_ids),
                   "benchmark": args.benchmark, "generator": args.generator,
                   "split_seed": SPLIT_SEED, "prior_Y1": prior_y1,
                   "fitted_theta": fitted_theta,
                   "costs": {"c_gen": costs.c_gen, "c_critic": costs.c_critic,
                             "c_ver": costs.c_ver, "reward": costs.reward}}

    # Resolve --variants filter
    all_variants = ["simple", "best_of_3", "threshold_L0", "threshold_L2", "threshold_L3",
                    "fixed_pipeline", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted"]
    if args.variants == "all":
        wanted = set(all_variants)
    elif args.variants == "fitted":
        wanted = {"greedy_fitted", "dp_fitted"}
    else:
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        unknown = wanted - set(all_variants)
        if unknown:
            raise SystemExit(f"unknown variants: {sorted(unknown)}. Valid: {all_variants}")
    # Force `simple` to always run — it's the baseline for delta_vs_simple
    wanted.add("simple")
    log.info("running variants: %s", sorted(wanted))

    for tid in test_ids:
        patches = records_by_inst.get(tid, [])
        if not patches:
            log.warning("no cached patches for test instance %s — skipping", tid)
            continue

        results = []
        if "simple" in wanted:
            results.append(run_simple(tid, patches, costs))
        if "best_of_3" in wanted:
            results.append(run_best_of_n(tid, patches, costs, n=3))
        if "threshold_L0" in wanted:
            results.append(run_threshold(tid, patches, costs, ["L0_syntax"], "threshold_L0"))
        if "threshold_L2" in wanted:
            results.append(run_threshold(tid, patches, costs, ["L0_syntax", "L1_lint", "L2_public_tests"], "threshold_L2"))
        if "threshold_L3" in wanted:
            results.append(run_threshold(tid, patches, costs, ["L0_syntax", "L1_lint", "L2_public_tests", "L3_llm_review"], "threshold_L3"))
        if "fixed_pipeline" in wanted:
            results.append(run_fixed_pipeline(tid, patches, costs))
        if "greedy_hand" in wanted:
            results.append(run_greedy(tid, patches, hand_theta, costs, "hand"))
        if "greedy_fitted" in wanted:
            results.append(run_greedy(tid, patches, fitted_theta, costs, "fitted"))
        if "dp_hand" in wanted:
            results.append(run_dp(tid, patches, hand_theta, costs, dp_hand, "hand"))
        if "dp_fitted" in wanted:
            results.append(run_dp(tid, patches, fitted_theta, costs, dp_fitted, "fitted"))
        for r in results:
            state["results"][f"{tid}|{r.variant}"] = r.to_dict()

    # Aggregate per variant (Δ_π vs always_verify = simple in fast-oracle)
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for k, v in state["results"].items():
        by_variant[v["variant"]].append(v)
    summaries: list[dict] = []
    av_util = None
    for v_name in ["simple", "best_of_3", "threshold_L0", "threshold_L2", "threshold_L3",
                   "fixed_pipeline", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted"]:
        rs = by_variant.get(v_name, [])
        if not rs:
            continue
        n = len(rs)
        fix_rate = sum(1 for r in rs if r["fixed"]) / n
        mean_cost = sum(r["total_cost"] for r in rs) / n
        mean_util = costs.reward * fix_rate - mean_cost
        if v_name == "simple":
            av_util = mean_util
        entry = {
            "policy": v_name,
            "n_episodes": n,
            "mean_utility": round(mean_util, 4),
            "pass_rate": round(fix_rate, 4),
            "fix_rate": round(fix_rate, 4),
            "mean_cost": round(mean_cost, 4),
        }
        if av_util is not None:
            entry["delta_vs_simple"] = round(mean_util - av_util, 4)
        summaries.append(entry)
        print(f"  {v_name:<20} n={n}  fix={fix_rate:.3f}  cost={mean_cost:7.3f}  "
              f"Ū={mean_util:8.3f}  Δ={mean_util - av_util:+8.3f}"
              if av_util is not None else
              f"  {v_name:<20} n={n}  fix={fix_rate:.3f}  cost={mean_cost:7.3f}  Ū={mean_util:8.3f}")

    state["summaries"] = summaries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, indent=2))
    log.info("saved → %s", args.output)


if __name__ == "__main__":
    main()
