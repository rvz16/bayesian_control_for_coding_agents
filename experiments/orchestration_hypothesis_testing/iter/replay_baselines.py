"""Self-Refine and Reflexion baselines as policy replays on iter trajectories.

Reference impls:
  Self-Refine: github.com/madaan/self-refine
  Reflexion:   github.com/noahshinn/reflexion

CANONICAL DIFFERENCES (from analyzing the repos):
  Self-Refine:  generate -> model_self_critique (free-text) -> refine -> stop on
                substring match in critique. Memory: only most recent
                (code, feedback). No external test executor.
  Reflexion:    generate -> external_test_executor (binary + trace) -> if fail,
                model_self_reflect -> refine using accumulated reflections list.
                Stop on test-pass.

POLICY-REPLAY APPROXIMATIONS (no new API spend; reuse existing iter trajectories):

  selfrefine_last
    Spirit: model self-refines for N steps, takes the converged output.
    Replay: take the patch from the LAST step (step N-1), verify.
    Critic-stack-aware? NO (mirrors Self-Refine: no external critics).
    Cost: c_gen * N (generation per step) + c_ver (one verification at end).
    Reward: reward * Y[last_step].

  reflexion_first_pass
    Spirit: external evaluator returns binary pass/fail; stop when test passes.
    Replay (LCB / bugfix): walk steps, take first patch where
      L2_public_tests = True (treat L2 as the "external test"). If none,
      take last step.
      Cost: (k+1) * c_gen + (k+1) * c_L2 + c_ver  where k = step chosen.
      Reward: reward * Y[step_chosen].
    Replay (SWE-bench, no L2 in iter records): walk steps, take first patch where
      Y = 1 (treat verifier as the "external test"). If none, take last step.
      Cost: (k+1) * c_gen + (k+1) * c_ver  (verifier called each step).
      Reward: reward * Y[step_chosen].

PAIRED-BOOTSTRAP CI (B=1000) on diff_vs_always_verify, matching existing
policy_comparison.json schema.

Outputs per cell:
  <out-dir>/<benchmark>/<gen>/policy_comparison_iter_replay_baselines.json

Schema (per file):
  {
    "policies": {
      "selfrefine_last": {
        "mean_utility": float, "pass_rate": float,
        "diff_vs_always_verify": float, "ci95_lo": float, "ci95_hi": float,
        "n_instances": int
      },
      "reflexion_first_pass": {...},
      "always_verify": {...}      # recomputed for the same cost model + n_instances
    },
    "iter_dir": str, "n_instances": int, "cost_model": {...}
  }

Usage:
  python3 scripts/compute_iter_replay_baselines.py \\
    --iter-dir data/lcb_calibration_v2_iter   --variant lcb     --out-suffix _iter_replay_baselines

  python3 scripts/compute_iter_replay_baselines.py \\
    --iter-dir data/humanevalfix_iter         --variant bugfix  --c-gen 10 --c-l0 1 --c-l2 1 --c-l3 1 --c-ver 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Package root (parents[1]) on sys.path so imports like `from calibration.X import Y`,
# `from iter.X import Y`, etc. resolve to the new refactored layout.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from calibration.lcb import canonical_generator_key  # noqa: E402

GENERATORS = ["gpt5_mini", "qwen3_coder", "haiku45", "sonnet45"]


# ---------- Cost model ----------

DEFAULT_COSTS = {
    "c_gen": 5,
    "c_L0":  1,
    "c_L2":  2,
    "c_L3":  5,
    "c_ver": 30,
    "reward": 100,
}


# ---------- Trajectory loading ----------

def load_iter_trajectories(records_path: Path) -> dict[str, list[dict]]:
    """Group iter_records.jsonl by instance_id, sort by step."""
    if not records_path.exists():
        return {}
    by_inst: dict[str, list[dict]] = defaultdict(list)
    for line in open(records_path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "instance_id" not in r or "step" not in r:
            continue
        by_inst[r["instance_id"]].append(r)
    for inst, rs in by_inst.items():
        rs.sort(key=lambda r: r["step"])
    return dict(by_inst)


# ---------- Policy utility computers ----------

def utility_selfrefine_last(traj: list[dict], costs: dict, variant: str) -> float:
    """Take last refinement step's patch, verify.

    Cost accounting: step-0 patch is sunk (matches existing policy_comparison.json
    convention where always_verify cost = c_ver only). Refinement steps 1..N-1
    each cost c_gen. Final verify costs c_ver.
    """
    n_steps = len(traj)
    if n_steps == 0:
        return 0.0
    last = traj[-1]
    y_last = int(bool(last.get("Y") or 0))
    n_refines = max(0, n_steps - 1)  # steps 1..N-1
    cost = n_refines * costs["c_gen"] + costs["c_ver"]
    return costs["reward"] * y_last - cost


def utility_reflexion_first_pass(traj: list[dict], costs: dict, variant: str) -> float:
    """Walk steps, stop at first 'external test' pass.

    Cost accounting (sunk-step-0 convention):
      step-0 patch generation: sunk (free)
      refinement step k (1..N-1): c_gen each
      external-test check at each step traversed: c_L2 (LCB) or c_ver (SWE)
      final verify of the chosen patch: c_ver (LCB only — SWE's test IS the verifier)

    LCB / bugfix variant: external test = L2_public_tests
    SWE variant: external test = Y itself (verifier IS the test, paid per step)
    """
    n_steps = len(traj)
    if n_steps == 0:
        return 0.0

    chosen_idx = None
    if variant in {"lcb", "bugfix"}:
        for i, r in enumerate(traj):
            if bool(r.get("L2_public_tests")):
                chosen_idx = i
                break
    elif variant == "swe":
        for i, r in enumerate(traj):
            if int(bool(r.get("Y") or 0)) == 1:
                chosen_idx = i
                break
    else:
        raise ValueError(f"unknown variant: {variant}")

    # Fallback to last step if no test pass
    if chosen_idx is None:
        chosen_idx = n_steps - 1

    chosen = traj[chosen_idx]
    y_chosen = int(bool(chosen.get("Y") or 0))
    n_traversed = chosen_idx + 1
    n_refines = max(0, n_traversed - 1)  # generations beyond step 0

    if variant in {"lcb", "bugfix"}:
        # n_traversed L2 checks (one per step), plus final c_ver
        cost = n_refines * costs["c_gen"] + n_traversed * costs["c_L2"] + costs["c_ver"]
    else:  # swe
        # SWE: verifier IS the external test, paid per step. No additional c_ver
        # because the last "test" call resolves the question.
        cost = n_refines * costs["c_gen"] + n_traversed * costs["c_ver"]

    return costs["reward"] * y_chosen - cost


def utility_always_verify(traj: list[dict], costs: dict, variant: str) -> float:
    """Verify the step-0 patch (the original independent sample).

    Matches existing policy_comparison.json: cost = c_ver only (step-0 generation
    sunk). Reward = reward * Y_0.
    """
    if not traj:
        return 0.0
    first = traj[0]
    y0 = int(bool(first.get("Y") or 0))
    cost = costs["c_ver"]
    return costs["reward"] * y0 - cost


# ---------- Bootstrap CI ----------

def paired_bootstrap_ci(util_a: list[float], util_b: list[float],
                         n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Return mean(util_a - util_b) + 95% paired-bootstrap CI."""
    rng = random.Random(seed)
    n = len(util_a)
    diffs = [a - b for a, b in zip(util_a, util_b)]
    mean = sum(diffs) / n if n else 0.0
    boot_means = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        boot_means.append(sum(diffs[i] for i in idxs) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    return mean, lo, hi


# ---------- Per-cell driver ----------

def run_cell(iter_dir: Path, gen: str, variant: str, costs: dict,
             n_boot: int) -> dict | None:
    records_path = iter_dir / gen / "iter_records.jsonl"
    if not records_path.exists():
        return None
    trajectories = load_iter_trajectories(records_path)
    if not trajectories:
        return None
    insts = sorted(trajectories.keys())

    # Per-instance utilities
    util_av = []
    util_sr = []
    util_rx = []
    pass_av = []
    pass_sr = []
    pass_rx = []
    for inst in insts:
        traj = trajectories[inst]
        # always_verify (baseline)
        util_av.append(utility_always_verify(traj, costs, variant))
        pass_av.append(int(bool((traj[0] if traj else {}).get("Y") or 0)))
        # selfrefine_last
        util_sr.append(utility_selfrefine_last(traj, costs, variant))
        pass_sr.append(int(bool((traj[-1] if traj else {}).get("Y") or 0)))
        # reflexion_first_pass
        util_rx.append(utility_reflexion_first_pass(traj, costs, variant))
        # pass_rate of reflexion = whether the chosen patch had Y=1
        if not traj:
            pass_rx.append(0)
        else:
            chosen_idx = None
            if variant in {"lcb", "bugfix"}:
                for i, r in enumerate(traj):
                    if bool(r.get("L2_public_tests")):
                        chosen_idx = i; break
            else:
                for i, r in enumerate(traj):
                    if int(bool(r.get("Y") or 0)) == 1:
                        chosen_idx = i; break
            if chosen_idx is None:
                chosen_idx = len(traj) - 1
            pass_rx.append(int(bool(traj[chosen_idx].get("Y") or 0)))

    n = len(insts)

    def mk_stats(label: str, util: list[float], passes: list[int]) -> dict:
        mean_diff, lo, hi = paired_bootstrap_ci(util, util_av, n_boot=n_boot)
        return {
            "label": label,
            "mean_utility": sum(util) / n if n else 0.0,
            "pass_rate": sum(passes) / n if n else 0.0,
            "diff_vs_always_verify": mean_diff,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_instances": n,
        }

    # always_verify reference (Δ=0 by definition)
    av = {
        "label": "always_verify",
        "mean_utility": sum(util_av) / n if n else 0.0,
        "pass_rate": sum(pass_av) / n if n else 0.0,
        "diff_vs_always_verify": 0.0,
        "ci95_lo": 0.0,
        "ci95_hi": 0.0,
        "n_instances": n,
    }
    return {
        "always_verify": av,
        "selfrefine_last": mk_stats("selfrefine_last", util_sr, pass_sr),
        "reflexion_first_pass": mk_stats("reflexion_first_pass", util_rx, pass_rx),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iter-dir", required=True, type=Path,
                        help="dir containing <gen>/iter_records.jsonl")
    parser.add_argument("--variant", required=True, choices=["lcb", "swe", "bugfix"],
                        help="trajectory variant: 'lcb'/'bugfix' use L2_public_tests as ext test; 'swe' uses Y")
    parser.add_argument("--out-suffix", default="_iter_replay_baselines",
                        help="suffix for output JSON name (default: _iter_replay_baselines)")
    parser.add_argument("--generators", default=",".join(GENERATORS))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--c-gen", type=float, default=DEFAULT_COSTS["c_gen"])
    parser.add_argument("--c-l0", type=float, default=DEFAULT_COSTS["c_L0"])
    parser.add_argument("--c-l2", type=float, default=DEFAULT_COSTS["c_L2"])
    parser.add_argument("--c-l3", type=float, default=DEFAULT_COSTS["c_L3"])
    parser.add_argument("--c-ver", type=int, default=DEFAULT_COSTS["c_ver"])
    parser.add_argument("--reward", type=float, default=DEFAULT_COSTS["reward"])
    args = parser.parse_args()

    costs = dict(DEFAULT_COSTS)
    costs["c_gen"] = args.c_gen
    costs["c_L0"] = args.c_l0
    costs["c_L2"] = args.c_l2
    costs["c_L3"] = args.c_l3
    costs["c_ver"] = args.c_ver
    costs["reward"] = args.reward
    gens = [canonical_generator_key(g) for g in args.generators.split(",") if g.strip()]

    print(f"Iter dir: {args.iter_dir}")
    print(f"Variant:  {args.variant}")
    print(f"Costs:    {costs}")
    print()
    print(f"{'gen':12} {'n':>4}  {'av':>8}  {'srf':>8}  {'rx':>8}  "
          f"{'Δsrf':>8} {'srf_ci':>20}  {'Δrx':>8} {'rx_ci':>20}")
    for gen in gens:
        result = run_cell(args.iter_dir, gen, args.variant, costs, args.n_boot)
        if result is None:
            print(f"{gen:12}   skipped (no iter_records)")
            continue
        out_path = args.iter_dir / gen / f"policy_comparison{args.out_suffix}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "policies": result,
            "iter_dir": str(args.iter_dir),
            "variant": args.variant,
            "n_instances": result["selfrefine_last"]["n_instances"],
            "cost_model": costs,
        }, indent=2))
        srf = result["selfrefine_last"]
        rx  = result["reflexion_first_pass"]
        av  = result["always_verify"]
        print(f"{gen:12} {srf['n_instances']:>4}  "
              f"{av['mean_utility']:+8.2f}  {srf['mean_utility']:+8.2f}  {rx['mean_utility']:+8.2f}  "
              f"{srf['diff_vs_always_verify']:+8.2f} [{srf['ci95_lo']:+6.1f},{srf['ci95_hi']:+6.1f}]  "
              f"{rx['diff_vs_always_verify']:+8.2f} [{rx['ci95_lo']:+6.1f},{rx['ci95_hi']:+6.1f}]")


if __name__ == "__main__":
    main()
