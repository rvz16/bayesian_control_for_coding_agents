#!/usr/bin/env python
"""Experiment F: prior belief b₀ sweep."""
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

PRIOR_GRID = [0.25, 0.35, 0.5, 0.65, 0.75]


def sweep_one(label, patches, hand_theta, fitted_theta, names):
    flat = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in names}
    theta_tables = {"hand": hand_theta, "fitted": fitted_theta, "flat": flat}
    cells = {}
    costs = AgentCostConfig(c_llm_call=10.0, c_full_test=5.0,
                            c_critic_test=1.0, reward=100.0)
    for b0 in PRIOR_GRID:
        grid = run_simulation_grid(
            patches, theta_tables, costs=costs, prior=b0, max_verifications=1,
        )
        baseline = grid["greedy_flat"].avg_utility
        cells[b0] = {}
        for v in ["dp_fitted", "dp_hand", "greedy_fitted", "greedy_hand"]:
            m = grid[v]
            cells[b0][v] = {
                "U": round(m.avg_utility, 3),
                "delta": round(m.avg_utility - baseline, 3),
                "fix_rate": round(m.fix_rate, 3),
                "bail_rate": round(m.bail_rate, 3),
                "verify_rate": round(m.verify_rate, 3),
            }
    return cells


def main():
    print("=== Experiment F: prior belief b₀ sweep (defaults: R=100, c_v=5, c_critic=1) ===\n")
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
            "CC", load_patches_from_allure(cc_attach),
            CC_CRITIC_LIKELIHOODS, CC_FITTED, CC_CRITIC_NAMES,
        )
    swe_attach = ROOT / "allure-results" / "64dc6695-dedb-4bb4-89ed-259736e24f37-attachment.json"
    if swe_attach.exists():
        benchmarks["SWE-Bench Lite"] = sweep_one(
            "SWE", load_patches_from_allure(swe_attach),
            SWE_CRITIC_LIKELIHOODS, SWE_FITTED, SWE_CRITIC_NAMES,
        )

    for label, cells in benchmarks.items():
        print(f"\n### {label}: dp_fitted vs always_verify under varying b₀\n")
        print("| b₀ | Ū | Δ | fix_rate | bail_rate | verify_rate |")
        print("|---|---:|---:|---:|---:|---:|")
        for b0 in PRIOR_GRID:
            d = cells[b0]["dp_fitted"]
            print(f"| {b0} | {d['U']:+.2f} | {d['delta']:+.2f} | "
                  f"{d['fix_rate']:.2f} | {d['bail_rate']:.2f} | {d['verify_rate']:.2f} |")

    out = ROOT / "sim_results" / "prior_sweep.json"
    out.write_text(json.dumps({"benchmarks": benchmarks, "grid": PRIOR_GRID}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
