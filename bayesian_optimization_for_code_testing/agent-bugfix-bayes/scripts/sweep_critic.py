#!/usr/bin/env python
"""Experiment D: critic cost sweep.

Hold R=100, c_v=5, c_llm=10. Sweep c_critic ∈ {0.1, 0.5, 1, 2, 5}.
Tests whether critics are worth using at different price points.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_cverify import (
    HEF_HAND, HEF_FITTED, CC_FITTED, SWE_FITTED, load_patches_from_allure,
)
from abbo.realworld.agents.agent_simulator import run_simulation_grid
from abbo.realworld.agents.code_contests import CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES
from abbo.realworld.agents.swe_bench import SWE_CRITIC_LIKELIHOODS, SWE_CRITIC_NAMES
from abbo.realworld.agents.simple_agent import AgentCostConfig

C_CRITIC_GRID = [0.1, 0.5, 1, 2, 5]


def sweep_one(label, patches, hand_theta, fitted_theta, names):
    flat = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in names}
    theta_tables = {"hand": hand_theta, "fitted": fitted_theta, "flat": flat}
    cells = {}
    for c_c in C_CRITIC_GRID:
        costs = AgentCostConfig(c_llm_call=10.0, c_full_test=5.0,
                                c_critic_test=float(c_c), reward=100.0)
        grid = run_simulation_grid(
            patches, theta_tables, costs=costs, prior=0.5, max_verifications=1,
        )
        baseline = grid["greedy_flat"].avg_utility
        cells[c_c] = {}
        for v in ["dp_fitted", "dp_hand", "greedy_fitted", "greedy_hand"]:
            m = grid[v]
            cells[c_c][v] = {
                "U": round(m.avg_utility, 3),
                "delta": round(m.avg_utility - baseline, 3),
                "fix_rate": round(m.fix_rate, 3),
                "avg_critics": round(m.avg_critics_called, 3),
                "bail_rate": round(m.bail_rate, 3),
            }
    return cells


def main():
    print("=== Experiment D: critic cost sweep (R=100, c_v=5, c_llm=10) ===\n")
    benchmarks = {}
    import glob
    hef_p = None
    for p in glob.glob(str(ROOT / "allure-results" / "*-attachment.json")):
        try:
            d = json.loads(Path(p).read_text())
            if "fitted" in d and d["fitted"] and \
               str(d["fitted"][0].get("task", "")).startswith("Python/"):
                hef_p = load_patches_from_allure(Path(p))
                break
        except Exception:
            continue
    if hef_p:
        benchmarks["HumanEvalFix"] = sweep_one("HEF", hef_p, HEF_HAND, HEF_FITTED, list(HEF_FITTED.keys()))
    cc_attach = ROOT / "allure-results" / "7e4c207c-f8d9-46c5-8663-3ccdf4e5ea11-attachment.json"
    if cc_attach.exists():
        benchmarks["CodeContests"] = sweep_one(
            "CC", load_patches_from_allure(cc_attach), CC_CRITIC_LIKELIHOODS,
            CC_FITTED, CC_CRITIC_NAMES,
        )
    swe_attach = ROOT / "allure-results" / "64dc6695-dedb-4bb4-89ed-259736e24f37-attachment.json"
    if swe_attach.exists():
        benchmarks["SWE-Bench Lite"] = sweep_one(
            "SWE", load_patches_from_allure(swe_attach), SWE_CRITIC_LIKELIHOODS,
            SWE_FITTED, SWE_CRITIC_NAMES,
        )

    for label, cells in benchmarks.items():
        print(f"\n### {label}: dp_fitted under varying critic cost\n")
        print("| c_critic | Ū | Δ | avg_critics_called | bail_rate |")
        print("|---|---:|---:|---:|---:|")
        for c_c in C_CRITIC_GRID:
            d = cells[c_c]["dp_fitted"]
            print(f"| {c_c} | {d['U']:+.2f} | {d['delta']:+.2f} | "
                  f"{d['avg_critics']:.2f} | {d['bail_rate']:.2f} |")

    out = ROOT / "sim_results" / "critic_cost_sweep.json"
    out.write_text(json.dumps({"benchmarks": benchmarks, "grid": C_CRITIC_GRID}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
