#!/usr/bin/env python3
"""Profile Bayes Greedy/DP overhead for the 2-model, 9-benchmark matrix.

The policy microbenchmarks use each cell's fitted likelihood table and the
same controller implementation used by ``run_fitted_live.py``. End-to-end
timings are read from the completed SAGE/UQ runs on Capstor. Those logs time
critics and verifiers explicitly, but not generation/router LLM calls, so the
LLM bucket is reported as a residual rather than as pure model latency.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_ROOT))

from analysis.controller import BayesianController, CostModel  # noqa: E402
from analysis.lcb_compare import GreedyController  # noqa: E402

CAPSTOR = Path("/capstor/store/cscs/swissai/a0142/agents_uq")
BENCHMARKS = (
    "lcb_easy",
    "lcb_medium",
    "lcb_hard",
    "humaneval",
    "mbpp",
    "humanevalfix",
    "codecontests",
    "swebench_lite",
    "swebench_verified",
)
MODELS = {
    "gpt-oss-20b": {
        "lcb": CAPSTOR / "lcb_llm_tool_agent_gpt_oss_20b/2652261_20260630_122731",
        "nonswe": CAPSTOR / "sage_uncertainty_nonswe_gpt_oss_20b/2652263_20260630_122732",
        "swe": CAPSTOR / "sage_uncertainty_swe_gpt_oss_20b/2656597_20260630_220737",
    },
    "Qwen2.5-Coder-32B": {
        "lcb": CAPSTOR / "lcb_llm_tool_agent_qwen25_32b/2652316_20260630_124848",
        "nonswe": CAPSTOR / "sage_uncertainty_nonswe_qwen25_32b/2652317_20260630_124842",
        "swe": CAPSTOR / "sage_uncertainty_swe_qwen25_32b/2656598_20260630_220804",
    },
}

HORIZON = 5
N_BELIEF = 51
DEFAULT_KERNEL = {
    "kernel_all": {
        "P_fix_given_broken": 0.50,
        "P_break_given_correct": 0.05,
    }
}
COST = CostModel(c_gen=5.0, c_L0=1.0, c_L2=2.0, c_L3=5.0, c_ver=30.0, reward=100.0)


def run_group(benchmark: str) -> str:
    if benchmark.startswith("lcb_"):
        return "lcb"
    if benchmark.startswith("swebench_"):
        return "swe"
    return "nonswe"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_likelihoods(path: Path, benchmark: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    cleaned = {}
    for name, values in (data.get("critic_likelihoods") or {}).items():
        if values.get("P_pass_given_Y1") is None or values.get("P_pass_given_Y0") is None:
            continue
        if benchmark.startswith("swebench_") and name == "L2_public_tests":
            continue
        cleaned[name] = values
    data["critic_likelihoods"] = cleaned
    return data


def median_ns_per_call(fn: Callable[[int], Any], calls: int, rounds: int = 9) -> float:
    samples = []
    fn(0)
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(rounds):
            start = time.perf_counter_ns()
            for i in range(calls):
                fn(i)
            samples.append((time.perf_counter_ns() - start) / calls)
    finally:
        if gc_enabled:
            gc.enable()
    return statistics.median(samples)


def profile_policy(likes: dict[str, Any], build_repeats: int, decision_calls: int) -> dict[str, float]:
    prior = float(likes.get("prior_Y1", 0.5))
    # Kernel values affect the selected policy, but not the number of grid
    # states/action values evaluated, which is what this timing measures.
    for _ in range(2):
        BayesianController(prior, likes, DEFAULT_KERNEL, COST, horizon=HORIZON, n_belief=N_BELIEF)

    build_samples = []
    for _ in range(build_repeats):
        start = time.perf_counter_ns()
        dp = BayesianController(
            prior,
            likes,
            DEFAULT_KERNEL,
            COST,
            horizon=HORIZON,
            n_belief=N_BELIEF,
        )
        build_samples.append(time.perf_counter_ns() - start)

    greedy = GreedyController(prior, likes, COST)
    beliefs = (0.03, 0.11, 0.27, 0.49, 0.73, 0.91)
    active_critics = [
        name
        for name in ("L0_syntax", "L2_public_tests", "L3_llm_review")
        if name in likes["critic_likelihoods"]
    ]
    critic = active_critics[0]

    greedy_ns = median_ns_per_call(
        lambda i: greedy.select_action(beliefs[i % len(beliefs)], i % HORIZON),
        decision_calls,
    )
    lookup_ns = median_ns_per_call(
        lambda i: dp.select_action(beliefs[i % len(beliefs)], i % HORIZON),
        decision_calls,
    )
    update_ns = median_ns_per_call(
        lambda i: dp._bayes_update(
            beliefs[i % len(beliefs)], critic, observed_pass=bool(i & 1)
        ),
        decision_calls,
    )

    return {
        "dp_build_ms": statistics.median(build_samples) / 1e6,
        "dp_build_p95_ms": sorted(build_samples)[max(0, int(0.95 * len(build_samples)) - 1)] / 1e6,
        "dp_lookup_us": lookup_ns / 1e3,
        "greedy_decision_us": greedy_ns / 1e3,
        "belief_update_us": update_ns / 1e3,
        "active_critics": len(active_critics),
        "dp_grid_states": N_BELIEF * HORIZON,
    }


def summarize_runtime(rows: list[dict[str, Any]]) -> dict[str, float]:
    buckets: defaultdict[str, float] = defaultdict(float)
    n_decisions = 0
    n_updates = 0
    total_wall = 0.0

    for row in rows:
        wall = float(row.get("wall_clock", 0.0) or 0.0)
        total_wall += wall
        actions = row.get("actions") or []
        n_decisions += max(0, len(actions) - 1)  # initial generation is automatic
        explicitly_timed = 0.0
        generations_seen = 0
        for action in actions:
            name = str(action.get("action", ""))
            elapsed = float(action.get("wall_clock_s", 0.0) or 0.0)
            explicitly_timed += elapsed
            if name in {"verify", "final_verify"}:
                buckets["verifier_s"] += elapsed
            elif name == "critic_L3":
                buckets["openrouter_l3_s"] += elapsed
            elif name.startswith("critic_L"):
                buckets["local_critics_s"] += elapsed
            else:
                buckets["other_timed_s"] += elapsed

            if name == "generate":
                if generations_seen > 0:
                    n_updates += 1
                generations_seen += 1
            elif name.startswith("critic_L") and action.get("passed") is not None:
                n_updates += 1

        # Generation and SAGE routing calls were not separately timed.
        buckets["llm_orchestration_residual_s"] += wall - explicitly_timed

    result = dict(buckets)
    result.update(
        n_instances=len(rows),
        n_decisions=n_decisions,
        n_belief_updates=n_updates,
        total_wall_s=total_wall,
        mean_wall_s=total_wall / len(rows) if rows else 0.0,
    )
    return result


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator > 0 else 0.0


def profile_cell(
    model: str,
    benchmark: str,
    root: Path,
    build_repeats: int,
    decision_calls: int,
) -> dict[str, Any]:
    readable = root / "readable" / benchmark
    final_path = readable / "final_logprob_bayes_quality.jsonl"
    likes_path = readable / "likelihood_tables.json"
    if not final_path.exists() or not likes_path.exists():
        raise FileNotFoundError(f"missing cell artifacts under {readable}")

    runtime = summarize_runtime(load_jsonl(final_path))
    policy = profile_policy(load_likelihoods(likes_path, benchmark), build_repeats, decision_calls)
    wall = runtime["total_wall_s"]
    decisions = runtime["n_decisions"]
    updates = runtime["n_belief_updates"]
    build_s = policy["dp_build_ms"] / 1e3
    lookup_s = decisions * policy["dp_lookup_us"] / 1e6
    greedy_s = decisions * policy["greedy_decision_us"] / 1e6
    update_s = updates * policy["belief_update_us"] / 1e6
    dp_current_s = runtime["n_instances"] * build_s + lookup_s + update_s
    dp_cached_s = build_s + lookup_s + update_s
    greedy_total_s = greedy_s + update_s

    out = {
        "model": model,
        "benchmark": benchmark,
        "source": str(readable),
        **runtime,
        **policy,
        "greedy_total_s": greedy_total_s,
        "greedy_overhead_pct": pct(greedy_total_s, wall),
        "dp_current_total_s": dp_current_s,
        "dp_current_overhead_pct": pct(dp_current_s, wall),
        "dp_cached_total_s": dp_cached_s,
        "dp_cached_overhead_pct": pct(dp_cached_s, wall),
    }
    for bucket in (
        "llm_orchestration_residual_s",
        "verifier_s",
        "openrouter_l3_s",
        "local_critics_s",
        "other_timed_s",
    ):
        out[f"{bucket.removesuffix('_s')}_pct"] = pct(float(out.get(bucket, 0.0)), wall)
    return out


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Fitted-live Bayes Greedy and DP computational overhead",
        "",
        "This profiles the compact 51-point fitted-live controller, not the ABBO "
        "`PerCriticCostDPPlanner` used for the earlier 1.45% LCB-Hard estimate.",
        "",
        "## Setup",
        "",
        "The policy code is profiled on CPU using the fitted critic likelihoods for each "
        "benchmark/model cell. The DP implementation builds a 51-point belief grid over "
        "a five-generation horizon (255 belief-step states). Non-SWE cells evaluate L0, "
        "L2, and L3; SWE cells evaluate L0 and L3. Timings are medians of repeated calls.",
        "",
        "End-to-end denominators come from the completed SAGE/UQ runs. Critic and verifier "
        "actions contain explicit timers. Generation and LLM routing calls do not, so "
        "`LLM residual` is `total wall-clock - explicitly timed actions`; it is real "
        "elapsed time but is not pure engine latency.",
        "",
        "`DP current` reconstructs the current fitted-live implementation, which builds "
        "the policy once per instance. `DP cached` builds one table per benchmark/model "
        "cell and then performs policy lookups.",
        "",
        "## Results",
        "",
        "| Model | Benchmark | n | Wall / instance (s) | Belief update (us) | "
        "Greedy (us/decision) | DP build (ms) | DP lookup (us) | Greedy overhead | "
        "DP current | DP cached | LLM residual | Verifier | OpenRouter L3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['benchmark']} | {row['n_instances']} | "
            f"{row['mean_wall_s']:.2f} | {row['belief_update_us']:.2f} | "
            f"{row['greedy_decision_us']:.2f} | {row['dp_build_ms']:.2f} | "
            f"{row['dp_lookup_us']:.2f} | "
            f"{row['greedy_overhead_pct']:.5f}% | {row['dp_current_overhead_pct']:.3f}% | "
            f"{row['dp_cached_overhead_pct']:.5f}% | "
            f"{row['llm_orchestration_residual_pct']:.1f}% | {row['verifier_pct']:.1f}% | "
            f"{row['openrouter_l3_pct']:.1f}% |"
        )

    lines.extend(["", "## Aggregate", ""])
    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        wall = sum(row["total_wall_s"] for row in model_rows)
        greedy = sum(row["greedy_total_s"] for row in model_rows)
        dp_current = sum(row["dp_current_total_s"] for row in model_rows)
        dp_cached = sum(row["dp_cached_total_s"] for row in model_rows)
        lines.append(
            f"- **{model}:** observed wall-clock {wall:,.1f} s; Greedy CPU overhead "
            f"{greedy:.4f} s ({pct(greedy, wall):.5f}%); DP current {dp_current:.3f} s "
            f"({pct(dp_current, wall):.3f}%); DP cached {dp_cached:.3f} s "
            f"({pct(dp_cached, wall):.5f}%)."
        )

    all_wall = sum(row["total_wall_s"] for row in rows)
    all_greedy = sum(row["greedy_total_s"] for row in rows)
    all_dp_current = sum(row["dp_current_total_s"] for row in rows)
    all_dp_cached = sum(row["dp_cached_total_s"] for row in rows)
    update_range = (
        min(row["belief_update_us"] for row in rows),
        max(row["belief_update_us"] for row in rows),
    )
    greedy_range = (
        min(row["greedy_decision_us"] for row in rows),
        max(row["greedy_decision_us"] for row in rows),
    )
    build_range = (
        min(row["dp_build_ms"] for row in rows),
        max(row["dp_build_ms"] for row in rows),
    )
    lookup_range = (
        min(row["dp_lookup_us"] for row in rows),
        max(row["dp_lookup_us"] for row in rows),
    )
    lines.extend(
        [
            "",
            f"Across all 18 cells, observed wall-clock is **{all_wall:,.1f} s**. "
            f"Greedy policy computation takes **{all_greedy:.4f} s "
            f"({pct(all_greedy, all_wall):.5f}%)**. DP takes **{all_dp_current:.3f} s "
            f"({pct(all_dp_current, all_wall):.3f}%)** with the current per-instance build, "
            f"or **{all_dp_cached:.3f} s ({pct(all_dp_cached, all_wall):.5f}%)** when cached "
            "once per cell.",
            "",
            "## Paper-ready summary",
            "",
            f"> Across nine benchmarks and two local models, a Bayesian belief update takes "
            f"{update_range[0]:.2f}-{update_range[1]:.2f} microseconds and Bayes-Greedy "
            f"action selection takes {greedy_range[0]:.2f}-{greedy_range[1]:.2f} "
            f"microseconds per step. The DP policy is computed over 255 belief-step states "
            f"in {build_range[0]:.2f}-{build_range[1]:.2f} ms; subsequent action selection "
            f"takes {lookup_range[0]:.2f}-{lookup_range[1]:.2f} microseconds. Across all "
            f"18 cells, policy computation accounts for {pct(all_greedy, all_wall):.5f}% "
            f"of measured wall-clock for Bayes Greedy and {pct(all_dp_current, all_wall):.3f}% "
            f"for the current per-instance DP construction ({pct(all_dp_cached, all_wall):.5f}% "
            "when the DP table is cached once per cell).",
            "",
            "## Reproduction",
            "",
            "```bash",
            "conda run -n agents python experiments/orchestration_hypothesis_testing/scripts/profile_bayes_overhead_matrix.py",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-repeats", type=int, default=31)
    parser.add_argument("--decision-calls", type=int, default=20_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXP_ROOT / "reports",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, roots in MODELS.items():
        for benchmark in BENCHMARKS:
            print(f"Profiling {model} / {benchmark}", flush=True)
            rows.append(
                profile_cell(
                    model,
                    benchmark,
                    roots[run_group(benchmark)],
                    args.build_repeats,
                    args.decision_calls,
                )
            )

    stem = args.output_dir / "fitted_live_bayes_overhead_2models_9benchmarks"
    (stem.with_suffix(".json")).write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(rows, stem.with_suffix(".csv"))
    write_markdown(rows, stem.with_suffix(".md"))
    print(f"Wrote {stem}.{{json,csv,md}}")


if __name__ == "__main__":
    main()
