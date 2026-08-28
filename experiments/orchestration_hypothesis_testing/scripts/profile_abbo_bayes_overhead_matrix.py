#!/usr/bin/env python3
"""Profile the ABBO PerCriticCostDPPlanner over the 18-cell matrix.

This is the planner used by ``collect_final_confidence_bayes_quality.py`` and
by the earlier LCB-Hard/120B overhead estimate. It is intentionally distinct
from the smaller 51 x horizon controller in ``analysis/controller.py``.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

EXP_ROOT = Path(__file__).resolve().parents[1]
ABBO_SRC = (
    EXP_ROOT.parents[1]
    / "bayesian_optimization_for_code_testing/agent-bugfix-bayes/src"
)
sys.path.insert(0, str(ABBO_SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from abbo.realworld.agents.bayes_agent import (  # noqa: E402
    DPPlanner,
    DPState,
    bayes_update,
    discretize,
    grid_belief,
)
from abbo.realworld.agents.simple_agent import AgentCostConfig  # noqa: E402
from profile_bayes_overhead_matrix import (  # noqa: E402
    BENCHMARKS,
    MODELS,
    load_jsonl,
    run_group,
    summarize_runtime,
)

KERNEL = {"p_fix_broken": 0.50, "p_break_correct": 0.05}
CRITIC_COSTS = {
    "L0_syntax": 1.0,
    "L1_lint": 1.0,
    "L2_public_tests": 2.0,
    "L3_llm_review": 5.0,
}
COSTS = AgentCostConfig(
    c_llm_call=5.0,
    c_critic_test=1.0,
    c_full_test=30.0,
    reward=100.0,
)


class PerCriticCostDPPlanner(DPPlanner):
    """Exact per-critic-cost planner from the confidence collector."""

    def __init__(self, *args: Any, critic_costs: dict[str, float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.critic_costs = critic_costs

    def _value(self, state: DPState) -> float:
        if state in self._cache:
            return self._cache[state][0]

        belief = grid_belief(state.belief_idx)
        best_value = 0.0
        best_action = "bail_out"

        if state.ver_left > 0:
            fail_state = DPState(
                belief_idx=discretize(0.05),
                gen_left=state.gen_left,
                crit_used=state.crit_used,
                ver_left=state.ver_left - 1,
            )
            after_fail = self._value(fail_state) if state.ver_left > 1 else 0.0
            value = (
                -self.costs.c_full_test
                + belief * self.costs.reward
                + (1 - belief) * after_fail
            )
            if value > best_value:
                best_value, best_action = value, "verify"

        if state.gen_left > 0:
            next_belief = (
                belief * (1 - self.transition_kernel["p_break_correct"])
                + (1 - belief) * self.transition_kernel["p_fix_broken"]
            )
            next_state = DPState(
                belief_idx=discretize(next_belief),
                gen_left=state.gen_left - 1,
                crit_used=state.crit_used,
                ver_left=state.ver_left,
            )
            value = -self.costs.c_llm_call + self._value(next_state)
            if value > best_value:
                best_value, best_action = value, "generate:measured_kernel"

        for critic_name in self._critic_names:
            if critic_name in state.crit_used:
                continue
            likelihood = self.critic_likelihoods[critic_name]
            p_pass = (
                likelihood["p_pass_y1"] * belief
                + likelihood["p_pass_y0"] * (1 - belief)
            )
            pass_belief = bayes_update(
                belief,
                critic_name,
                passed=True,
                likelihoods=self.critic_likelihoods,
            )
            fail_belief = bayes_update(
                belief,
                critic_name,
                passed=False,
                likelihoods=self.critic_likelihoods,
            )
            used = state.crit_used | frozenset([critic_name])
            pass_state = DPState(
                discretize(pass_belief), state.gen_left, used, state.ver_left
            )
            fail_state = DPState(
                discretize(fail_belief), state.gen_left, used, state.ver_left
            )
            value = (
                -self.critic_costs[critic_name]
                + p_pass * self._value(pass_state)
                + (1 - p_pass) * self._value(fail_state)
            )
            if value > best_value:
                best_value, best_action = value, f"critic:{critic_name}"

        self._cache[state] = (best_value, best_action)
        return best_value


def load_theta(path: Path) -> tuple[float, dict[str, dict[str, float]]]:
    data = json.loads(path.read_text())
    theta = {}
    for name, row in (data.get("critic_likelihoods") or {}).items():
        p1 = row.get("P_pass_given_Y1", row.get("p_pass_y1"))
        p0 = row.get("P_pass_given_Y0", row.get("p_pass_y0"))
        if p1 is not None and p0 is not None:
            theta[name] = {"p_pass_y1": float(p1), "p_pass_y0": float(p0)}
    if not theta:
        raise ValueError(f"no fitted critic likelihoods in {path}")
    # Some GPT-OSS cells have null L3 likelihoods. The production loader drops
    # incomplete critics, so profile the same available action space here.
    return float(data.get("prior_Y1", 0.5)), theta


def make_planner(theta: dict[str, dict[str, float]], generations: int, verifies: int) -> PerCriticCostDPPlanner:
    planner = PerCriticCostDPPlanner(
        COSTS,
        generations,
        verifies,
        critic_likelihoods=theta,
        transition_kernel=KERNEL,
        critic_costs=CRITIC_COSTS,
    )
    planner.solve()
    return planner


def profile_build(
    theta: dict[str, dict[str, float]],
    generations: int,
    verifies: int,
    repeats: int,
) -> tuple[float, float, int, PerCriticCostDPPlanner]:
    # One warm-up avoids import/cache setup contaminating the measurement.
    warmup = make_planner(theta, generations, verifies)
    del warmup
    gc.collect()

    timings = []
    cache_sizes = []
    planner = None
    for _ in range(repeats):
        start = time.perf_counter_ns()
        planner = make_planner(theta, generations, verifies)
        timings.append((time.perf_counter_ns() - start) / 1e6)
        cache_sizes.append(len(planner._cache))
        if len(timings) < repeats:
            del planner
            planner = None
            gc.collect()
    assert planner is not None
    return (
        statistics.median(timings),
        sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        int(statistics.median(cache_sizes)),
        planner,
    )


def q_critic_one_step(
    belief: float,
    critic_name: str,
    theta: dict[str, dict[str, float]],
) -> float:
    likelihood = theta[critic_name]
    p_pass = likelihood["p_pass_y1"] * belief + likelihood["p_pass_y0"] * (1 - belief)
    pass_belief = likelihood["p_pass_y1"] * belief / max(p_pass, 1e-12)
    fail_denominator = (
        (1 - likelihood["p_pass_y1"]) * belief
        + (1 - likelihood["p_pass_y0"]) * (1 - belief)
    )
    fail_belief = (1 - likelihood["p_pass_y1"]) * belief / max(fail_denominator, 1e-12)
    return (
        -CRITIC_COSTS[critic_name]
        + p_pass * max(0.0, -COSTS.c_full_test + pass_belief * COSTS.reward)
        + (1 - p_pass) * max(0.0, -COSTS.c_full_test + fail_belief * COSTS.reward)
    )


def choose_greedy(belief: float, theta: dict[str, dict[str, float]]) -> str:
    choices = [
        ("bail_out", 0.0),
        ("verify", -COSTS.c_full_test + belief * COSTS.reward),
    ]
    choices.extend(
        (f"critic:{name}", q_critic_one_step(belief, name, theta))
        for name in theta
    )
    next_belief = belief * 0.95 + (1 - belief) * 0.50
    choices.append(
        (
            "generate",
            -COSTS.c_llm_call - COSTS.c_full_test + next_belief * COSTS.reward,
        )
    )
    return max(choices, key=lambda item: item[1])[0]


def median_us_per_call(function, calls: int, rounds: int = 9) -> float:
    samples = []
    function(0)
    for _ in range(rounds):
        start = time.perf_counter_ns()
        for index in range(calls):
            function(index)
        samples.append((time.perf_counter_ns() - start) / calls / 1e3)
    return statistics.median(samples)


def profile_cell(
    model: str,
    benchmark: str,
    root: Path,
    matched_repeats: int,
    legacy_repeats: int,
    decision_calls: int,
) -> dict[str, Any]:
    readable = root / "readable" / benchmark
    runtime = summarize_runtime(
        load_jsonl(readable / "final_logprob_bayes_quality.jsonl")
    )
    prior, theta = load_theta(readable / "likelihood_tables.json")

    matched_ms, matched_p95, matched_states, matched = profile_build(
        theta, generations=5, verifies=2, repeats=matched_repeats
    )
    legacy_ms, legacy_p95, legacy_states, legacy = profile_build(
        theta, generations=20, verifies=10, repeats=legacy_repeats
    )
    beliefs = (0.03, 0.11, 0.27, 0.49, 0.73, 0.91)
    critic = next(iter(theta))
    greedy_us = median_us_per_call(
        lambda i: choose_greedy(beliefs[i % len(beliefs)], theta), decision_calls
    )
    update_us = median_us_per_call(
        lambda i: bayes_update(
            beliefs[i % len(beliefs)],
            critic,
            passed=bool(i & 1),
            likelihoods=theta,
        ),
        decision_calls,
    )
    matched_lookup_us = median_us_per_call(
        lambda i: matched.choose_action(
            beliefs[i % len(beliefs)],
            gen_left=i % 6,
            crit_used=frozenset(),
            ver_left=i % 3,
        ),
        decision_calls,
    )
    legacy_lookup_us = median_us_per_call(
        lambda i: legacy.choose_action(
            beliefs[i % len(beliefs)],
            gen_left=i % 21,
            crit_used=frozenset(),
            ver_left=i % 11,
        ),
        decision_calls,
    )

    wall = runtime["total_wall_s"]
    instances = runtime["n_instances"]
    decisions = runtime["n_decisions"]
    updates = runtime["n_belief_updates"]
    greedy_total = (decisions * greedy_us + updates * update_us) / 1e6

    def totals(build_ms: float, lookup_us: float) -> tuple[float, float]:
        step_s = (decisions * lookup_us + updates * update_us) / 1e6
        return instances * build_ms / 1e3 + step_s, build_ms / 1e3 + step_s

    matched_current, matched_cached = totals(matched_ms, matched_lookup_us)
    legacy_current, legacy_cached = totals(legacy_ms, legacy_lookup_us)
    def percent(value: float) -> float:
        return 100.0 * value / wall if wall else 0.0

    return {
        "model": model,
        "benchmark": benchmark,
        "source": str(readable),
        "n_instances": instances,
        "total_wall_s": wall,
        "mean_wall_s": runtime["mean_wall_s"],
        "n_decisions": decisions,
        "n_belief_updates": updates,
        "prior_y1": prior,
        "n_critics": len(theta),
        "belief_update_us": update_us,
        "greedy_decision_us": greedy_us,
        "greedy_total_s": greedy_total,
        "greedy_overhead_pct": percent(greedy_total),
        "matched_g": 5,
        "matched_v": 2,
        "matched_dp_build_ms": matched_ms,
        "matched_dp_build_p95_ms": matched_p95,
        "matched_dp_states": matched_states,
        "matched_dp_lookup_us": matched_lookup_us,
        "matched_dp_current_s": matched_current,
        "matched_dp_current_pct": percent(matched_current),
        "matched_dp_cached_s": matched_cached,
        "matched_dp_cached_pct": percent(matched_cached),
        "legacy_g": 20,
        "legacy_v": 10,
        "legacy_dp_build_ms": legacy_ms,
        "legacy_dp_build_p95_ms": legacy_p95,
        "legacy_dp_states": legacy_states,
        "legacy_dp_lookup_us": legacy_lookup_us,
        "legacy_dp_current_s": legacy_current,
        "legacy_dp_current_pct": percent(legacy_current),
        "legacy_dp_cached_s": legacy_cached,
        "legacy_dp_cached_pct": percent(legacy_cached),
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# ABBO Bayes Greedy and DP overhead",
        "",
        "This report profiles the same `PerCriticCostDPPlanner` used for the earlier "
        "LCB-Hard/gpt-oss-120b estimate of 1.45% overhead. It is not the smaller "
        "51-point fitted-live controller.",
        "",
        "Two budgets are reported: `G=5,V=2`, matching the completed two-model runs, "
        "and `G=20,V=10`, matching the earlier 120B LCB-Hard stress configuration. "
        "Both use `C=101`. `K` is the number of critics with non-null fitted "
        "likelihoods in that cell; the earlier 120B run had `K=4`.",
        "",
        "| Model | Benchmark | n | K | Wall/instance (s) | Belief (us) | Greedy (us) | "
        "G5/V2 build (ms) | States | Current overhead | Cached overhead | "
        "G20/V10 build (s) | States | Current overhead |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['benchmark']} | {row['n_instances']} | "
            f"{row['n_critics']} | "
            f"{row['mean_wall_s']:.2f} | {row['belief_update_us']:.2f} | "
            f"{row['greedy_decision_us']:.2f} | {row['matched_dp_build_ms']:.1f} | "
            f"{row['matched_dp_states']:,} | {row['matched_dp_current_pct']:.3f}% | "
            f"{row['matched_dp_cached_pct']:.4f}% | "
            f"{row['legacy_dp_build_ms'] / 1e3:.2f} | {row['legacy_dp_states']:,} | "
            f"{row['legacy_dp_current_pct']:.2f}% |"
        )

    wall = sum(row["total_wall_s"] for row in rows)
    greedy = sum(row["greedy_total_s"] for row in rows)
    matched_current = sum(row["matched_dp_current_s"] for row in rows)
    matched_cached = sum(row["matched_dp_cached_s"] for row in rows)
    legacy_current = sum(row["legacy_dp_current_s"] for row in rows)
    def percent(value: float) -> float:
        return 100.0 * value / wall
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"Across all 18 cells (`{wall:,.1f} s` observed wall-clock):",
            "",
            f"- Bayes Greedy: `{greedy:.3f} s` (`{percent(greedy):.5f}%`).",
            f"- DP at matched `G=5,V=2`, rebuilt per instance: `{matched_current:.1f} s` "
            f"(`{percent(matched_current):.3f}%`).",
            f"- DP at matched `G=5,V=2`, cached once per cell: `{matched_cached:.2f} s` "
            f"(`{percent(matched_cached):.5f}%`).",
            f"- DP at legacy `G=20,V=10`, rebuilt per instance: `{legacy_current:.1f} s` "
            f"(`{percent(legacy_current):.2f}%`).",
            "",
            "## Interpretation",
            "",
            "The earlier `1.45%` is still correct for its exact LCB-Hard/120B run: "
            "77 per-instance builds at about 1.49 s each over 7,922 s total wall-clock. "
            "The matrix-matched budget is smaller, so its DP table has fewer reachable "
            "states and lower construction cost. The legacy column is the direct "
            "same-planner/same-budget sensitivity comparison.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "conda run -n agents python experiments/orchestration_hypothesis_testing/scripts/profile_abbo_bayes_overhead_matrix.py",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-repeats", type=int, default=5)
    parser.add_argument("--legacy-repeats", type=int, default=3)
    parser.add_argument("--decision-calls", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "reports")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, roots in MODELS.items():
        for benchmark in BENCHMARKS:
            print(f"Profiling ABBO {model} / {benchmark}", flush=True)
            rows.append(
                profile_cell(
                    model,
                    benchmark,
                    roots[run_group(benchmark)],
                    args.matched_repeats,
                    args.legacy_repeats,
                    args.decision_calls,
                )
            )

    stem = args.output_dir / "abbo_bayes_overhead_2models_9benchmarks"
    stem.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(rows, stem.with_suffix(".csv"))
    write_markdown(rows, stem.with_suffix(".md"))
    print(f"Wrote {stem}.{{json,csv,md}}")


if __name__ == "__main__":
    main()
