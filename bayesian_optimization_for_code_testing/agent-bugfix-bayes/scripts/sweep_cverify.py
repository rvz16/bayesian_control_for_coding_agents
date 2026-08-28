#!/usr/bin/env python
"""Hyperparameter sweep: C_verify sensitivity on cached calibration data.

Implements HP plan Section 15 (recommended first experiment).

For each C_verify in a grid:
  1. Build AgentCostConfig with that c_full_test.
  2. Re-solve DP planners (dp_hand, dp_fitted) — DP value table depends on costs.
  3. Re-simulate Greedy + DP on every held-out patch using cached critic outcomes.
  4. Record per-variant Ū_π and Δ_π vs always-verify baseline.

This sweep does NOT call the LLM. It uses the simulator (frozen patches),
not the end-to-end runner. So we're measuring decision-quality sensitivity
to verify cost, holding patch outcomes constant.

Output:
  sim_results/cverify_sweep.json — all (benchmark, c_verify, variant) cells
  prints a markdown table per benchmark
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from abbo.realworld.agents.agent_simulator import (
    PatchOutcomes, run_simulation_grid, aggregate, format_grid,
)
from abbo.realworld.agents.code_contests import (
    CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.swe_bench import SWE_CRITIC_LIKELIHOODS, SWE_CRITIC_NAMES


# Cached fitted theta tables (from calibration runs)
HEF_FITTED = {
    "critic_early":  {"p_pass_y1": 0.9761904761904762, "p_pass_y0": 0.16666666666666666},
    "critic_lint":   {"p_pass_y1": 0.9535714285714286, "p_pass_y0": 0.9035714285714286},
    "critic_mid":    {"p_pass_y1": 0.9761904761904762, "p_pass_y0": 0.023809523809523808},
    "critic_syntax": {"p_pass_y1": 0.98,               "p_pass_y0": 0.9511904761904761},
}
HEF_HAND = {
    "critic_syntax": {"p_pass_y1": 0.99, "p_pass_y0": 0.90},
    "critic_lint":   {"p_pass_y1": 0.90, "p_pass_y0": 0.70},
    "critic_early":  {"p_pass_y1": 0.95, "p_pass_y0": 0.30},
    "critic_mid":    {"p_pass_y1": 0.95, "p_pass_y0": 0.20},
}

CC_FITTED = {
    "critic_early":  {"p_pass_y1": 0.9687500000000001, "p_pass_y0": 0.375},
    "critic_lint":   {"p_pass_y1": 0.728125,           "p_pass_y0": 0.678125},
    "critic_mid":    {"p_pass_y1": 0.9687500000000001, "p_pass_y0": 0.125},
    "critic_syntax": {"p_pass_y1": 0.9803571428571428, "p_pass_y0": 0.94375},
}

SWE_FITTED = {
    "critic_syntax": {"p_pass_y1": 0.914, "p_pass_y0": 0.864},
    "critic_lint":   {"p_pass_y1": 0.247, "p_pass_y0": 0.197},
    "critic_early":  {"p_pass_y1": 0.667, "p_pass_y0": 0.444},
    "critic_mid":    {"p_pass_y1": 0.667, "p_pass_y0": 0.222},
}


# C_verify grid (per HP plan section 15)
C_VERIFY_GRID = [1, 2, 5, 10, 20, 40]


def load_patches_from_allure(json_path: Path) -> list[PatchOutcomes]:
    """Load held-out patches with their critic outcomes from a calibration
    per_task_predictions.json attachment."""
    d = json.loads(json_path.read_text())
    patches = []
    for entry in d["fitted"]:
        # Schema differs between benchmarks: HEF uses 'task', SWE uses 'instance',
        # CC uses 'task'. All have 'arm', 'y', 'observations'.
        bug_id = entry.get("task") or entry.get("instance") or entry.get("task_id")
        patches.append(PatchOutcomes(
            bug_id=bug_id,
            arm=entry["arm"],
            critic_outcomes={o["c"]: o["p"] for o in entry["observations"]},
            Y=entry["y"],
        ))
    return patches


def sweep_one_benchmark(label, patches, hand_theta, fitted_theta, names):
    """Sweep C_verify on a single benchmark; return per-variant Δ curves."""
    print(f"\n=== {label}: n={len(patches)} patches ===")
    flat = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in names}
    theta_tables = {"hand": hand_theta, "fitted": fitted_theta, "flat": flat}

    by_c = {}
    for c_v in C_VERIFY_GRID:
        costs = AgentCostConfig(c_llm_call=10.0, c_full_test=float(c_v),
                                c_critic_test=1.0, reward=100.0)
        grid = run_simulation_grid(
            patches, theta_tables, costs=costs, prior=0.5, max_verifications=1,
        )
        # We use 'simple-like' = 'always verify' as baseline. Our simulator
        # doesn't have a Simple variant (no generate), so use greedy_flat as
        # the always-verify proxy (flat θ has gap=0 → always verifies).
        # The HP plan defines baseline as "always_verify"; greedy_flat with
        # c_verify same matches that exactly.
        baseline = grid["greedy_flat"].avg_utility
        cell = {}
        for v, m in grid.items():
            cell[v] = {
                "U_pi": round(m.avg_utility, 3),
                "delta": round(m.avg_utility - baseline, 3),
                "fix_rate": round(m.fix_rate, 3),
                "avg_cost": round(m.avg_cost, 3),
                "bail_rate": round(m.bail_rate, 3),
                "wasted_verify_rate": round(m.wasted_verify_rate, 3),
            }
        cell["_baseline_U"] = round(baseline, 3)
        by_c[c_v] = cell
        # Mini progress print: just dp_fitted vs baseline
        u = grid["dp_fitted"].avg_utility
        print(f"  c_v={c_v:>3}: dp_fitted Ū={u:+7.2f}, Δ vs always_verify={u - baseline:+6.2f}, "
              f"bail={grid['dp_fitted'].bail_rate:.2f}")
    return by_c


def print_markdown_table(label, by_c, variants):
    print(f"\n### {label}\n")
    header = "| C_verify |" + "".join(f" {v} Δ |" for v in variants)
    sep = "|---|" + "".join("---:|" for _ in variants)
    print(header)
    print(sep)
    for c_v in C_VERIFY_GRID:
        row = f"| {c_v} |"
        for v in variants:
            d = by_c[c_v][v]["delta"]
            row += f" {d:+.2f} |"
        print(row)


def main():
    benchmarks = {}

    # HumanEvalFix
    try:
        hef_patches = load_patches_from_allure(
            ROOT / "allure-results" / "8f63bbac-c552-43be-bcdf-fce9c25dec18-attachment.json"
        )
    except FileNotFoundError:
        # Try to find any HEF per_task_predictions
        import glob
        candidates = glob.glob(str(ROOT / "allure-results" / "*-attachment.json"))
        hef_patches = None
        for p in candidates:
            try:
                d = json.loads(Path(p).read_text())
                if "fitted" in d and isinstance(d["fitted"], list) and d["fitted"]:
                    e = d["fitted"][0]
                    if "task" in e and isinstance(e.get("observations"), list):
                        # heuristic: HEF tasks start with "Python/"
                        if str(e["task"]).startswith("Python/"):
                            hef_patches = load_patches_from_allure(Path(p))
                            print(f"  found HEF in: {p}")
                            break
            except Exception:
                continue

    if hef_patches:
        names = list(HEF_FITTED.keys())
        benchmarks["HumanEvalFix"] = sweep_one_benchmark(
            "HumanEvalFix", hef_patches, HEF_HAND, HEF_FITTED, names,
        )

    # CodeContests — cached at: per_task_predictions.json (find via id "7e4c207c")
    cc_attach = ROOT / "allure-results" / "7e4c207c-f8d9-46c5-8663-3ccdf4e5ea11-attachment.json"
    if cc_attach.exists():
        cc_patches = load_patches_from_allure(cc_attach)
        benchmarks["CodeContests"] = sweep_one_benchmark(
            "CodeContests", cc_patches,
            # Hand-tuned CC theta — pull from code_contests.py constant
            CC_CRITIC_LIKELIHOODS,
            CC_FITTED, CC_CRITIC_NAMES,
        )

    # SWE-Bench Lite — cached at: per_task_predictions.json (id "64dc6695")
    swe_attach = ROOT / "allure-results" / "64dc6695-dedb-4bb4-89ed-259736e24f37-attachment.json"
    if swe_attach.exists():
        swe_patches = load_patches_from_allure(swe_attach)
        benchmarks["SWE-Bench Lite"] = sweep_one_benchmark(
            "SWE-Bench Lite", swe_patches,
            SWE_CRITIC_LIKELIHOODS, SWE_FITTED, SWE_CRITIC_NAMES,
        )

    # Save
    out = ROOT / "sim_results" / "cverify_sweep.json"
    out.write_text(json.dumps({
        "c_verify_grid": C_VERIFY_GRID,
        "benchmarks": benchmarks,
        "cost_constants": {"c_llm_call": 10, "c_critic_test": 1, "reward": 100},
        "note": "Δ is utility vs greedy_flat (always-verify proxy) at the same c_verify.",
    }, indent=2))
    print(f"\nSaved: {out}")

    # Print markdown tables for headline
    for label, by_c in benchmarks.items():
        print_markdown_table(label, by_c, ["dp_fitted", "dp_hand", "greedy_fitted", "greedy_hand"])

    # Headline summary: at what C_verify does dp_fitted start to win?
    print("\n=== Crossover analysis: where dp_fitted starts to beat always-verify ===")
    for label, by_c in benchmarks.items():
        crossover = None
        for c_v in C_VERIFY_GRID:
            if by_c[c_v]["dp_fitted"]["delta"] > 0:
                crossover = c_v
                break
        print(f"  {label}: dp_fitted positive at c_v ≥ {crossover}")


if __name__ == "__main__":
    main()
