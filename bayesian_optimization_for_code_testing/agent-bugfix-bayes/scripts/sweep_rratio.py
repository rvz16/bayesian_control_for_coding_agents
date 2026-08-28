#!/usr/bin/env python
"""Experiment C: R/c_v ratio sweep — single-axis projection.

For each (R, c_v) pair, compute Δ_π = U_dp_fitted − U_always_verify and
plot against ρ = R/c_v. Multiple (R, c_v) pairs that share ρ should give
similar Δ if the regime story is correct.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sweep_cverify import (
    HEF_HAND, HEF_FITTED, CC_FITTED, SWE_FITTED,
    load_patches_from_allure,
)
from abbo.realworld.agents.agent_simulator import run_simulation_grid
from abbo.realworld.agents.code_contests import CC_CRITIC_LIKELIHOODS, CC_CRITIC_NAMES
from abbo.realworld.agents.swe_bench import SWE_CRITIC_LIKELIHOODS, SWE_CRITIC_NAMES
from abbo.realworld.agents.simple_agent import AgentCostConfig

R_GRID = [50, 100, 200]
CV_GRID = [1, 2, 5, 10, 20, 40]


def sweep_one(label, patches, hand_theta, fitted_theta, names):
    cells = []
    flat = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in names}
    theta_tables = {"hand": hand_theta, "fitted": fitted_theta, "flat": flat}
    for R in R_GRID:
        for c_v in CV_GRID:
            costs = AgentCostConfig(c_llm_call=10.0, c_full_test=float(c_v),
                                    c_critic_test=1.0, reward=float(R))
            grid = run_simulation_grid(
                patches, theta_tables, costs=costs, prior=0.5, max_verifications=1,
            )
            u_dp = grid["dp_fitted"].avg_utility
            u_av = grid["greedy_flat"].avg_utility   # always-verify proxy
            rho = R / c_v
            cells.append({
                "R": R, "c_v": c_v, "rho": round(rho, 2),
                "U_dp_fitted": round(u_dp, 3),
                "U_always_verify": round(u_av, 3),
                "delta": round(u_dp - u_av, 3),
                "bail_rate": round(grid["dp_fitted"].bail_rate, 3),
            })
    return cells


def main():
    print("=== Experiment C: R/c_v ratio sweep ===\n")
    benchmarks = {}
    # HEF
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
        benchmarks["HumanEvalFix"] = sweep_one(
            "HEF", hef_p, HEF_HAND, HEF_FITTED, list(HEF_FITTED.keys()),
        )

    cc_attach = ROOT / "allure-results" / "7e4c207c-f8d9-46c5-8663-3ccdf4e5ea11-attachment.json"
    if cc_attach.exists():
        cc_p = load_patches_from_allure(cc_attach)
        benchmarks["CodeContests"] = sweep_one(
            "CC", cc_p, CC_CRITIC_LIKELIHOODS, CC_FITTED, CC_CRITIC_NAMES,
        )

    swe_attach = ROOT / "allure-results" / "64dc6695-dedb-4bb4-89ed-259736e24f37-attachment.json"
    if swe_attach.exists():
        swe_p = load_patches_from_allure(swe_attach)
        benchmarks["SWE-Bench Lite"] = sweep_one(
            "SWE", swe_p, SWE_CRITIC_LIKELIHOODS, SWE_FITTED, SWE_CRITIC_NAMES,
        )

    # Print: cells sorted by ρ
    for label, cells in benchmarks.items():
        print(f"\n### {label}: Δ vs ρ = R/c_v\n")
        print("| R | c_v | ρ | Δ (dp_fitted vs always_verify) | bail_rate |")
        print("|---|---|---|---:|---:|")
        for c in sorted(cells, key=lambda x: x["rho"]):
            print(f"| {c['R']} | {c['c_v']} | {c['rho']} | "
                  f"{c['delta']:+.2f} | {c['bail_rate']:.2f} |")

    out = ROOT / "sim_results" / "rratio_sweep.json"
    out.write_text(json.dumps({"benchmarks": benchmarks,
                               "R_grid": R_GRID, "cv_grid": CV_GRID}, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
