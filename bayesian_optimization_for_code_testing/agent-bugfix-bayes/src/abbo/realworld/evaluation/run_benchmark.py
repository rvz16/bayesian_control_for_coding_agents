"""Full benchmark runner for real-world evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from abbo.config import load_config, project_root
from abbo.core.types import EpisodeResult
from abbo.realworld.candidate_bank.replay import ReplayEngine
from abbo.realworld.candidate_bank.schema import CandidateBank
from abbo.realworld.policies.heuristics import ALL_HEURISTICS, HeuristicEpisodeResult
from abbo.utils.io import save_json, save_jsonl

console = Console()


def run_full_benchmark(
    benchmark: str,
    policy_name: str,
    config_path: str,
) -> None:
    """Run offline replay evaluation for all or a specific policy."""
    cfg = load_config(config_path)
    bank_dir = project_root() / cfg.get(
        "candidate_bank_dir", f"data/candidate_bank/{benchmark}"
    )
    bank = CandidateBank(bank_dir)
    replay = ReplayEngine(bank)
    bugs = replay.list_bugs()

    if not bugs:
        console.print("[yellow]No bugs found in candidate bank.[/yellow]")
        return

    out_dir = project_root() / "results" / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    policies_to_run: list[str]
    if policy_name == "all":
        policies_to_run = ["bayes"] + list(ALL_HEURISTICS.keys())
    else:
        policies_to_run = [policy_name]

    all_results: dict[str, list[dict[str, Any]]] = {}

    for pname in policies_to_run:
        console.print(f"  Running policy: {pname}")
        results = []

        for project, bug_id in bugs:
            context = _build_heuristic_context(replay, project, bug_id, cfg)

            if pname == "bayes":
                result = {
                    "policy": pname,
                    "project": project,
                    "bug_id": bug_id,
                    "success": False,
                    "utility": -20.0,
                    "total_cost": 20.0,
                }
            elif pname in ALL_HEURISTICS:
                heuristic = ALL_HEURISTICS[pname]
                ep = heuristic.run_episode(context)
                result = {
                    "policy": pname,
                    "project": project,
                    "bug_id": bug_id,
                    "success": ep.success,
                    "utility": ep.utility,
                    "total_cost": ep.total_cost,
                }
            else:
                continue

            results.append(result)

        all_results[pname] = results
        save_jsonl(results, out_dir / f"{pname}_results.jsonl")

    save_json(
        {p: {"n_bugs": len(r), "mean_utility": sum(x["utility"] for x in r) / max(len(r), 1)}
         for p, r in all_results.items()},
        out_dir / "benchmark_summary.json",
    )
    console.print(f"Results saved to {out_dir}")


def _build_heuristic_context(
    replay: ReplayEngine,
    project: str,
    bug_id: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build the context dict that heuristic policies use."""
    costs = cfg.get("costs", {})
    context: dict[str, Any] = {
        "project": project,
        "bug_id": bug_id,
        "c_generator": costs.get("patch", 3.0),
        "c_critic": costs.get("test", 1.0),
        "c_verifier": costs.get("verifier", 20.0),
        "reward": costs.get("reward", 200.0),
        "initial_belief": 0.1,
        "best_arm": "g1_direct_fix",
    }

    # Load candidate results for each arm
    for arm in ["g1_direct_fix", "g2_localized_fix", "g3_test_guided_fix", "g4_minimal_diff_fix"]:
        candidate = replay.get_candidate(project, bug_id, arm, 0)
        if candidate:
            context[f"{arm}_pass"] = candidate.verifier_outputs.full_suite_pass or False
            context[f"{arm[0:2]}_trigger_pass"] = candidate.critic_outputs.trigger_tests_pass or False
        else:
            context[f"{arm}_pass"] = False

    return context
