"""Phase-1 simulation on SWE-bench Lite. Replays held-out patches through
the agent decision loop with hand-tuned vs fitted theta_hat. Compares DP vs Greedy.

Re-collects critic outcomes via Docker (cached images make this fast on a
warm cache). No LLM.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import allure
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.agent_simulator import (
    format_grid,
    patches_from_samples,
    run_simulation_grid,
    simulate_dp,
    simulate_greedy,
)
from abbo.realworld.agents.calibration import calibrate_likelihoods
from abbo.realworld.agents.simple_agent import AgentCostConfig
from abbo.realworld.agents.swe_bench import (
    SWE_CRITIC_LIKELIHOODS,
    SWE_CRITIC_NAMES,
    collect_calibration_samples_from_pairs,
    list_instance_ids,
)


N_TRAIN = 7
N_TEST = 4
SPLIT_SEED = 42
PRIOR_BELIEF = 0.5


def _docker_available() -> bool:
    import subprocess
    r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
    return r.returncode == 0


@pytest.mark.skipif(not _docker_available(), reason="Docker daemon not running")
@allure.parent_suite("swebench_simulation")
@allure.suite("phase_1_decision_quality")
@allure.title("Phase 1: DP vs Greedy × hand vs fitted on SWE-Bench Lite held-out")
def test_swebench_simulation_quality() -> None:
    rng = random.Random(SPLIT_SEED)
    all_ids = list_instance_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    test_ids = all_ids[N_TRAIN:N_TRAIN + N_TEST]

    allure.dynamic.description(
        f"SWE-bench Lite curated pool: {len(all_ids)} small-dep instances.\n"
        f"Train: {N_TRAIN}  Test: {N_TEST}  seed={SPLIT_SEED}\n"
        f"Critics: {SWE_CRITIC_NAMES}\n"
        f"Compares DP vs Greedy under hand-tuned, fitted, and flat theta tables.\n"
        f"Docker-cached run — should be much faster than the calibration run."
    )

    # --- Step 1: collect train + fit ----------------------------------------
    with allure.step(f"Collect train samples ({len(train_ids)} instances)"):
        train_samples, train_kept, _ = collect_calibration_samples_from_pairs(
            train_ids, verbose=True,
        )
        fitted_lk, _counts = calibrate_likelihoods(
            train_samples, prior_alpha=1.0, prior_beta=1.0,
        )

    # --- Step 2: collect test samples ---------------------------------------
    with allure.step(f"Collect test samples ({len(test_ids)} instances)"):
        test_samples, test_kept, _ = collect_calibration_samples_from_pairs(
            test_ids, verbose=True,
        )
        patches = patches_from_samples(test_samples)

    print(f"\n  fitted theta:  {fitted_lk}")
    print(f"  hand-tuned:    {SWE_CRITIC_LIKELIHOODS}")
    print(f"  n_held_out_patches: {len(patches)}")

    # --- Step 3: run grid ---------------------------------------------------
    flat_lk = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in SWE_CRITIC_NAMES}
    theta_tables = {
        "hand": SWE_CRITIC_LIKELIHOODS,
        "fitted": fitted_lk,
        "flat": flat_lk,
    }
    costs = AgentCostConfig()

    with allure.step("Run DP × Greedy × {hand, fitted, flat} simulation grid"):
        grid = run_simulation_grid(
            patches, theta_tables, costs=costs,
            prior=PRIOR_BELIEF, max_verifications=1,
        )

    table_text = format_grid(grid)
    print("\n" + table_text + "\n")

    metrics_dict = {label: asdict(m) for label, m in grid.items()}
    headline = {
        "deltas_dp_fitted_vs_dp_hand": {
            "fix_rate": grid["dp_fitted"].fix_rate - grid["dp_hand"].fix_rate,
            "avg_cost": grid["dp_hand"].avg_cost - grid["dp_fitted"].avg_cost,
            "avg_utility": grid["dp_fitted"].avg_utility - grid["dp_hand"].avg_utility,
            "wasted_verify_rate":
                grid["dp_hand"].wasted_verify_rate - grid["dp_fitted"].wasted_verify_rate,
        },
    }

    with allure.step("Save artifacts"):
        allure.attach(
            table_text, name="simulation_grid.txt",
            attachment_type=allure.attachment_type.TEXT,
        )
        allure.attach(
            json.dumps({
                "metrics": metrics_dict,
                "headline_deltas": headline,
                "fitted_lk": fitted_lk,
                "hand_tuned_lk": SWE_CRITIC_LIKELIHOODS,
                "n_train_kept": len(train_kept),
                "n_test_kept": len(test_kept),
                "n_test_patches": len(patches),
            }, indent=2),
            name="simulation_metrics.json",
            attachment_type=allure.attachment_type.JSON,
        )

        dp_results = []
        greedy_results = []
        for p in patches:
            dp_results.append(simulate_dp(
                p, fitted_lk, costs, prior=PRIOR_BELIEF, max_verifications=1,
            ))
            greedy_results.append(simulate_greedy(
                p, fitted_lk, costs, prior=PRIOR_BELIEF,
            ))

        def _serialize(r):
            return {
                "bug_id": r.bug_id, "arm": r.arm, "Y": r.Y,
                "final_decision": r.final_decision, "fixed": r.fixed,
                "total_cost": r.total_cost,
                "n_critics_called": r.n_critics_called,
                "actions": [asdict(a) for a in r.actions],
            }

        allure.attach(
            json.dumps({
                "dp_fitted": [_serialize(r) for r in dp_results],
                "greedy_fitted": [_serialize(r) for r in greedy_results],
            }, indent=2),
            name="per_patch_traces.json",
            attachment_type=allure.attachment_type.JSON,
        )

    assert len(patches) == len(test_kept) * 2
    assert "dp_fitted" in grid and "greedy_hand" in grid
