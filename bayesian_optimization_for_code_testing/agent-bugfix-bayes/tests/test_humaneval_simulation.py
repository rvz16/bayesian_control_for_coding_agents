"""Phase-1 simulation: replay HumanEvalFix held-out patches through the agent
decision loop with hand-tuned vs fitted theta_hat. Compares DP vs Greedy.

No LLM, no Docker — uses pre-collected critic outcomes.

Output artifacts (Allure):
    - simulation_grid.txt        — table of all 4 (agent × theta) variants
    - simulation_metrics.json    — full aggregate metrics
    - per_patch_traces.json      — per-patch action logs
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import allure

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from abbo.realworld.agents.agent_simulator import (
    aggregate,
    format_grid,
    patches_from_samples,
    run_simulation_grid,
    simulate_dp,
    simulate_greedy,
)
from abbo.realworld.agents.calibration import (
    calibrate_likelihoods,
)
from abbo.realworld.agents.humaneval_fix import (
    HE_CRITIC_LIKELIHOODS,
    HE_CRITIC_NAMES,
    collect_calibration_samples_from_pairs,
    list_task_ids,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig


N_TRAIN = 40
N_TEST = 124          # all 164 tasks: 40 train + 124 test = 248 held-out patches
SPLIT_SEED = 42
PRIOR_BELIEF = 0.5


@allure.parent_suite("humaneval_simulation")
@allure.suite("phase_1_decision_quality")
@allure.title("Phase 1: DP vs Greedy × hand vs fitted on HumanEvalFix held-out")
def test_humaneval_simulation_quality() -> None:
    """Run the agent decision loop with each theta on the held-out split."""
    rng = random.Random(SPLIT_SEED)
    all_ids = list_task_ids()
    rng.shuffle(all_ids)
    train_ids = all_ids[:N_TRAIN]
    test_ids = all_ids[N_TRAIN:N_TRAIN + N_TEST]

    allure.dynamic.description(
        f"HumanEvalFix Python split: {len(all_ids)} tasks total.\n"
        f"Train: {N_TRAIN}  Test: {N_TEST}  seed={SPLIT_SEED}\n"
        f"Critics: {HE_CRITIC_NAMES}\n"
        f"Compares DP vs Greedy under hand-tuned, fitted, and flat theta tables.\n"
        f"No LLM — uses pre-collected critic outcomes from the calibration data."
    )

    # --- Step 1: re-run calibration to get fitted theta ----------------------
    with allure.step(f"Collect train samples and fit theta_hat"):
        train_samples = collect_calibration_samples_from_pairs(train_ids, verbose=False)
        fitted_lk, _counts = calibrate_likelihoods(
            train_samples, prior_alpha=1.0, prior_beta=1.0,
        )

    # --- Step 2: collect test samples and convert to PatchOutcomes ----------
    with allure.step(f"Collect test samples ({len(test_ids)} tasks)"):
        test_samples = collect_calibration_samples_from_pairs(test_ids, verbose=False)
        patches = patches_from_samples(test_samples)

    print(f"\n  fitted theta:  {fitted_lk}")
    print(f"  hand-tuned:    {HE_CRITIC_LIKELIHOODS}")
    print(f"  n_held_out_patches: {len(patches)}")

    # --- Step 3: run all 4 (agent × theta) variants -------------------------
    flat_lk = {c: {"p_pass_y1": 0.5, "p_pass_y0": 0.5} for c in HE_CRITIC_NAMES}
    theta_tables = {
        "hand": HE_CRITIC_LIKELIHOODS,
        "fitted": fitted_lk,
        "flat": flat_lk,
    }
    costs = AgentCostConfig()

    with allure.step("Run DP × Greedy × {hand, fitted, flat} simulation grid"):
        grid = run_simulation_grid(
            patches, theta_tables, costs=costs,
            prior=PRIOR_BELIEF, max_verifications=1,
        )

    # --- Step 4: print + save artifacts -------------------------------------
    table_text = format_grid(grid)
    print("\n" + table_text + "\n")

    metrics_dict = {label: asdict(m) for label, m in grid.items()}

    # Build the headline 4-cell table the deck wants
    headline = {
        "deltas_dp_fitted_vs_dp_hand": {
            "fix_rate": grid["dp_fitted"].fix_rate - grid["dp_hand"].fix_rate,
            "avg_cost": grid["dp_hand"].avg_cost - grid["dp_fitted"].avg_cost,
            "avg_utility": grid["dp_fitted"].avg_utility - grid["dp_hand"].avg_utility,
            "wasted_verify_rate":
                grid["dp_hand"].wasted_verify_rate - grid["dp_fitted"].wasted_verify_rate,
        },
        "deltas_dp_fitted_vs_greedy_fitted": {
            "fix_rate":
                grid["dp_fitted"].fix_rate - grid["greedy_fitted"].fix_rate,
            "avg_cost":
                grid["greedy_fitted"].avg_cost - grid["dp_fitted"].avg_cost,
            "avg_utility":
                grid["dp_fitted"].avg_utility - grid["greedy_fitted"].avg_utility,
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
                "hand_tuned_lk": HE_CRITIC_LIKELIHOODS,
                "n_train": N_TRAIN,
                "n_test_patches": len(patches),
            }, indent=2),
            name="simulation_metrics.json",
            attachment_type=allure.attachment_type.JSON,
        )

        # Per-patch traces (run again, single agent each, to capture full action log)
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

    # --- Sanity assertions ---------------------------------------------------
    # Allow some skipped tasks (HumanEvalFix is mostly clean but a few may
    # fail to collect — keep this loose).
    assert len(patches) >= N_TEST, f"expected >= {N_TEST} patches, got {len(patches)}"
    assert "dp_fitted" in grid and "greedy_hand" in grid
    for m in grid.values():
        assert 0.0 <= m.fix_rate <= 1.0
        assert m.n_patches == len(patches)
