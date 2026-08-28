"""Results summarization: tables, plots, and markdown report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from abbo.config import project_root
from abbo.core.stats import paired_bootstrap
from abbo.utils.io import load_json, load_jsonl, save_csv, save_json


def generate_all_outputs(benchmark: str, out_dir: Path) -> None:
    """Generate all summary outputs for a benchmark."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all results
    results_by_policy: dict[str, list[dict[str, Any]]] = {}
    for f in out_dir.glob("*_results.jsonl"):
        policy = f.stem.replace("_results", "")
        results_by_policy[policy] = load_jsonl(f)

    if not results_by_policy:
        # Try loading summary only
        summary_path = out_dir / "benchmark_summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            _write_report(benchmark, summary, {}, out_dir)
        return

    # Compute metrics per policy
    summary: dict[str, dict[str, Any]] = {}
    for policy, results in results_by_policy.items():
        n = len(results)
        if n == 0:
            continue
        resolved = sum(1 for r in results if r.get("success", False))
        utilities = [r.get("utility", 0.0) for r in results]
        costs = [r.get("total_cost", 0.0) for r in results]

        summary[policy] = {
            "n_bugs": n,
            "resolved_rate": resolved / n,
            "mean_utility": sum(utilities) / n,
            "mean_cost": sum(costs) / n,
        }

    # Save main table
    table_rows = [
        {"policy": p, **m} for p, m in sorted(summary.items())
    ]
    save_csv(table_rows, out_dir / "main_table.csv")

    # Bootstrap comparisons
    bootstrap_results: dict[str, Any] = {}
    if "bayes" in results_by_policy:
        bayes_utils = {
            (r["project"], r["bug_id"]): r["utility"]
            for r in results_by_policy["bayes"]
        }
        for policy, results in results_by_policy.items():
            if policy == "bayes":
                continue
            heur_utils = {
                (r["project"], r["bug_id"]): r["utility"]
                for r in results
            }
            common_bugs = sorted(set(bayes_utils) & set(heur_utils))
            if len(common_bugs) < 3:
                continue
            a = [bayes_utils[b] for b in common_bugs]
            b = [heur_utils[b] for b in common_bugs]
            ci = paired_bootstrap(a, b, seed=42)
            bootstrap_results[f"bayes_vs_{policy}"] = {
                "mean_diff": ci.mean_diff,
                "ci_lower": ci.ci_lower,
                "ci_upper": ci.ci_upper,
                "p_value": ci.p_value,
                "n_bugs": len(common_bugs),
            }

    save_json(bootstrap_results, out_dir / "bootstrap_summary.json")

    # Write report
    _write_report(benchmark, summary, bootstrap_results, out_dir)


def _write_report(
    benchmark: str,
    summary: dict[str, Any],
    bootstrap: dict[str, Any],
    out_dir: Path,
) -> None:
    """Write the article-ready markdown report."""
    lines = [
        f"# {benchmark.title()} Benchmark Report",
        "",
        "## Setup",
        "",
        f"Benchmark: {benchmark}",
        "Evaluation: offline replay from pre-built candidate bank.",
        "Utility: reward * indicator(full_suite_pass) - total_cost.",
        "",
        "## Policies",
        "",
        "| Policy | Description |",
        "|--------|-------------|",
        "| bayes | Bayesian decision-theoretic controller |",
        "| h1_one_shot | Direct fix once, then verify |",
        "| h2_fixed_workflow | trigger_tests -> localized_fix -> verify |",
        "| h3_two_stage_fixed | direct_fix -> trigger_tests -> conditional regen -> verify |",
        "| h4_threshold | Hand-tuned threshold policy |",
        "| h5_single_critic_then_map | One cheap critic, best generator arm, verify |",
        "",
        "## Calibration Method",
        "",
        "Critic likelihoods P(z|Y,d) estimated with Laplace smoothing from",
        "the calibration split. Generator transitions P(Y'|Y,g) estimated",
        "as global rates from parent-child candidate pairs.",
        "",
        "## Utility Definition",
        "",
        "U = reward_success * 1{full_suite_pass} - generator_costs - critic_costs - verifier_costs",
        "",
        "## Results",
        "",
    ]

    if summary:
        lines.append("| Policy | Resolved Rate | Mean Utility | Mean Cost | N Bugs |")
        lines.append("|--------|---------------|--------------|-----------|--------|")
        for policy, m in sorted(summary.items()):
            if isinstance(m, dict):
                lines.append(
                    f"| {policy} | {m.get('resolved_rate', 0):.3f} | "
                    f"{m.get('mean_utility', 0):.2f} | "
                    f"{m.get('mean_cost', 0):.2f} | "
                    f"{m.get('n_bugs', 0)} |"
                )

    lines.extend(["", "## Bootstrap Confidence Intervals", ""])
    if bootstrap:
        for comp, ci in bootstrap.items():
            lines.append(
                f"- **{comp}**: mean diff = {ci['mean_diff']:.3f}, "
                f"95% CI [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}], "
                f"p = {ci['p_value']:.4f} (n={ci['n_bugs']} bugs)"
            )
    else:
        lines.append("No bootstrap comparisons available.")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Candidate bank was built with stub generators; real LLM calls needed for meaningful results.",
        "- Transition model estimates are global; per-repo conditioning not yet implemented.",
        "- Belief grid discretization (0.01) may lose precision at extreme values.",
        "",
        "## Reproducibility",
        "",
        "- All evaluation uses offline replay (no live API calls).",
        "- Seeds are fixed for bootstrap sampling.",
        "- Configuration is in YAML files under configs/.",
        "",
    ])

    (out_dir / "report.md").write_text("\n".join(lines))
