"""Command-line interface for the ABBO toolkit.

Usage::

    python -m abbo.cli run-synthetic --episodes 100000 --seed 7
    python -m abbo.cli run-synthetic-exact
    python -m abbo.cli calibrate-toy --trials 10000
    python -m abbo.cli run-toy --episodes 10000 --seed 7
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="abbo", help="Agent Bug-fix Bayesian Orchestration toolkit.")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _results_dir(benchmark: str) -> Path:
    d = PROJECT_ROOT / "results" / benchmark
    d.mkdir(parents=True, exist_ok=True)
    return d


# -----------------------------------------------------------------------
# Synthetic commands
# -----------------------------------------------------------------------

@app.command()
def run_synthetic_exact() -> None:
    """Run exact symbolic computation on the synthetic benchmark."""
    from abbo.synthetic.env import SyntheticEnv
    from abbo.synthetic.exact_math import compute_exact_utilities, extract_tree
    from abbo.utils.io import save_json

    env = SyntheticEnv()
    console.print("[bold]Exact Synthetic Benchmark[/bold]")
    utilities = compute_exact_utilities(env)

    table = Table(title="Exact Expected Utilities")
    table.add_column("Policy", style="cyan")
    table.add_column("Exact Value", style="green")
    table.add_column("Decimal", style="yellow")
    for name, val in utilities.items():
        table.add_row(name, str(val), f"{float(val):.6f}")
    console.print(table)

    tree = extract_tree(env)
    console.print("\n[bold]Optimal Decision Tree[/bold]")
    _print_tree(tree)

    out_dir = _results_dir("synthetic")
    save_json(
        {k: {"fraction": str(v), "decimal": float(v)} for k, v in utilities.items()},
        out_dir / "exact_utilities.json",
    )
    save_json(tree, out_dir / "optimal_tree.json")
    console.print(f"\nResults saved to {out_dir}")


def _print_tree(node: dict, indent: int = 0) -> None:
    prefix = "  " * indent
    if node["action"] == "test":
        console.print(f"{prefix}Run [cyan]{node['test']}[/cyan]")
        console.print(f"{prefix}  if fail:")
        _print_tree(node["fail"], indent + 2)
        console.print(f"{prefix}  if pass:")
        _print_tree(node["pass"], indent + 2)
    else:
        console.print(
            f"{prefix}Apply [green]{node['patch']}[/green] "
            f"(target={node['target_class']}), then verify"
        )


@app.command()
def run_synthetic(
    episodes: int = typer.Option(100_000, help="Number of MC episodes"),
    seed: int = typer.Option(7, help="Random seed"),
) -> None:
    """Run Monte Carlo simulation on the synthetic benchmark."""
    from abbo.synthetic.env import SyntheticEnv
    from abbo.synthetic.policies import (
        BayesOptimalPolicy,
        OneTestMAPPolicy,
        OneTestIfFailElseRulePolicy,
        TunedThresholdHeuristic,
    )
    from abbo.synthetic.simulate import compare_policies
    from abbo.utils.io import save_csv, save_json

    env = SyntheticEnv()
    policies = [
        BayesOptimalPolicy(),
        OneTestMAPPolicy("T1"),
        OneTestIfFailElseRulePolicy(),
        TunedThresholdHeuristic(0.7),
    ]

    console.print(
        f"[bold]Monte Carlo Synthetic Benchmark[/bold]  "
        f"episodes={episodes}, seed={seed}"
    )
    rows = compare_policies(env, policies, episodes, seed)

    table = Table(title="Monte Carlo Results")
    table.add_column("Policy", style="cyan")
    table.add_column("Mean Utility", style="green")
    table.add_column("95% CI", style="yellow")
    table.add_column("Resolved Rate", style="magenta")
    for row in rows:
        table.add_row(
            row["policy"],
            f"{row['mean_utility']:.4f}",
            f"[{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]",
            f"{row['resolved_rate']:.4f}",
        )
    console.print(table)

    out_dir = _results_dir("synthetic")
    save_csv(rows, out_dir / "main_table.csv")
    save_json(rows, out_dir / "mc_results.json")
    console.print(f"\nResults saved to {out_dir}")


# -----------------------------------------------------------------------
# Toy commands
# -----------------------------------------------------------------------

@app.command()
def calibrate_toy(
    trials: int = typer.Option(10_000, help="Calibration trials per cell"),
) -> None:
    """Calibrate the toy benchmark's diagnostic failure matrix."""
    from abbo.toy.diagnostics import calibrate_all, print_calibration_report
    from abbo.toy.env import ToyEnv
    from abbo.utils.io import save_json

    console.print(f"[bold]Toy Benchmark Calibration[/bold]  trials={trials}")
    toy_env = ToyEnv()
    matrix = calibrate_all(toy_env, n_trials=trials, seed=42)
    print_calibration_report(matrix)

    out_dir = _results_dir("toy")
    save_json(
        {t: {c: float(v) for c, v in row.items()} for t, row in matrix.items()},
        out_dir / "calibration_matrix.json",
    )
    console.print(f"\nCalibration saved to {out_dir}")


@app.command()
def run_toy(
    episodes: int = typer.Option(10_000, help="Number of MC episodes"),
    seed: int = typer.Option(7, help="Random seed"),
) -> None:
    """Run Monte Carlo simulation on the toy benchmark."""
    from abbo.synthetic.env import SyntheticEnv
    from abbo.synthetic.policies import BayesOptimalPolicy, OneTestMAPPolicy
    from abbo.synthetic.simulate import compare_policies
    from abbo.toy.env import ToyEnv
    from abbo.utils.io import save_csv, save_json

    console.print(
        f"[bold]Toy Benchmark Simulation[/bold]  episodes={episodes}, seed={seed}"
    )
    # The toy env uses the same interface as SyntheticEnv
    toy = ToyEnv()
    synth_env = toy.as_synthetic_env()
    policies = [BayesOptimalPolicy(), OneTestMAPPolicy("T1")]
    rows = compare_policies(synth_env, policies, episodes, seed)

    table = Table(title="Toy Benchmark Results")
    table.add_column("Policy", style="cyan")
    table.add_column("Mean Utility", style="green")
    table.add_column("Resolved Rate", style="magenta")
    for row in rows:
        table.add_row(
            row["policy"],
            f"{row['mean_utility']:.4f}",
            f"{row['resolved_rate']:.4f}",
        )
    console.print(table)

    out_dir = _results_dir("toy")
    save_csv(rows, out_dir / "main_table.csv")
    save_json(rows, out_dir / "mc_results.json")
    console.print(f"\nResults saved to {out_dir}")


# -----------------------------------------------------------------------
# Real-world commands
# -----------------------------------------------------------------------

@app.command()
def build_candidate_bank(
    benchmark: str = typer.Option(..., help="Benchmark name (bugsinpy, defects4j)"),
    config: str = typer.Option(..., help="Path to benchmark config YAML"),
) -> None:
    """Build candidate bank for a real-world benchmark (requires API keys)."""
    from abbo.config import load_config
    from abbo.realworld.candidate_bank.build import build_bank

    console.print(f"[bold]Building candidate bank[/bold]  benchmark={benchmark}")
    cfg = load_config(config)
    build_bank(cfg, benchmark)
    console.print("Candidate bank build complete.")


@app.command()
def fit_bayes_model(
    benchmark: str = typer.Option(..., help="Benchmark name"),
    config: str = typer.Option(..., help="Path to benchmark config YAML"),
) -> None:
    """Fit Bayesian calibration model from candidate bank data."""
    from abbo.config import load_config
    from abbo.realworld.calibration.fit import fit_and_save

    console.print(f"[bold]Fitting Bayesian model[/bold]  benchmark={benchmark}")
    cfg = load_config(config)
    fit_and_save(cfg, benchmark)
    console.print("Model fitting complete.")


@app.command()
def run_benchmark(
    benchmark: str = typer.Option(..., help="Benchmark name"),
    policy: str = typer.Option("all", help="Policy name or 'all'"),
    config: str = typer.Option(None, help="Path to config YAML"),
) -> None:
    """Run offline replay evaluation on a real-world benchmark."""
    from abbo.realworld.evaluation.run_benchmark import run_full_benchmark

    console.print(
        f"[bold]Running benchmark[/bold]  benchmark={benchmark}, policy={policy}"
    )
    config_path = config or str(PROJECT_ROOT / "configs" / f"{benchmark}_pilot.yaml")
    run_full_benchmark(benchmark, policy, config_path)
    console.print("Benchmark evaluation complete.")


@app.command()
def summarize_results(
    benchmark: str = typer.Option(..., help="Benchmark name"),
) -> None:
    """Generate summary tables, plots, and report for a benchmark."""
    from abbo.realworld.evaluation.summarize import generate_all_outputs

    console.print(f"[bold]Summarizing results[/bold]  benchmark={benchmark}")
    out_dir = _results_dir(benchmark)
    generate_all_outputs(benchmark, out_dir)
    console.print(f"Summary saved to {out_dir}")


# -----------------------------------------------------------------------
# Real LLM code fixing
# -----------------------------------------------------------------------

@app.command()
def fix_bug(
    bug: str = typer.Option("bug_a", help="Bug to fix: bug_a, bug_b, bug_c, or all"),
    arm: str = typer.Option("all", help="Generator arm or 'all'"),
    model: str = typer.Option("qwen2.5:7b", help="Ollama model name"),
    temperature: float = typer.Option(0.2, help="LLM temperature"),
) -> None:
    """Fix a toy bug using a local LLM via Ollama."""
    from abbo.realworld.agents.code_fixer import (
        ARM_PROMPTS,
        BUGGY_SOURCES,
        fix_bug as do_fix,
        fix_bug_all_arms,
    )
    from abbo.realworld.agents.llm_provider import LLMConfig, is_ollama_available
    from abbo.utils.io import save_json

    if not is_ollama_available():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    llm_config = LLMConfig(model=model, temperature=temperature)
    console.print(f"[bold]LLM Code Fixing[/bold]  model={model}")

    bugs = list(BUGGY_SOURCES.keys()) if bug == "all" else [bug]
    arms = list(ARM_PROMPTS.keys()) if arm == "all" else [arm]

    all_results = []

    for bug_id in bugs:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Bug: {bug_id}[/bold]")

        for arm_name in arms:
            console.print(f"\n  [yellow]Arm: {arm_name}[/yellow]  (calling {model}...)")
            attempt = do_fix(bug_id, arm_name, llm_config)

            # Show results
            status = "[green]FIXED[/green]" if attempt.tests_passed else "[red]FAILED[/red]"
            console.print(f"  Result: {status}  ({attempt.wall_clock_seconds:.1f}s)")
            console.print(f"  Failing before: {len(attempt.failing_tests_before)} tests")
            console.print(f"  Failing after:  {len(attempt.failing_tests_after)} tests")

            if attempt.diff:
                console.print(f"\n  [bold]Diff:[/bold]")
                for line in attempt.diff.splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        console.print(f"  [green]{line}[/green]")
                    elif line.startswith("-") and not line.startswith("---"):
                        console.print(f"  [red]{line}[/red]")
                    else:
                        console.print(f"  {line}")
            elif not attempt.tests_passed:
                console.print(f"  [red]Error: {attempt.error}[/red]")

            all_results.append({
                "bug_id": bug_id,
                "arm": arm_name,
                "model": attempt.model,
                "fixed": attempt.tests_passed,
                "failing_before": len(attempt.failing_tests_before),
                "failing_after": len(attempt.failing_tests_after),
                "wall_clock_seconds": attempt.wall_clock_seconds,
            })

    # Summary table
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    table = Table(title="Fix Results Summary")
    table.add_column("Bug", style="cyan")
    table.add_column("Arm", style="yellow")
    table.add_column("Result", style="green")
    table.add_column("Failing Before", style="red")
    table.add_column("Failing After", style="red")
    table.add_column("Time (s)", style="white")
    for r in all_results:
        table.add_row(
            r["bug_id"],
            r["arm"],
            "[green]FIXED[/green]" if r["fixed"] else "[red]FAILED[/red]",
            str(r["failing_before"]),
            str(r["failing_after"]),
            f"{r['wall_clock_seconds']:.1f}",
        )
    console.print(table)

    out_dir = _results_dir("fix_results")
    save_json(all_results, out_dir / "fix_results.json")
    console.print(f"\nResults saved to {out_dir}")


@app.command()
def list_models() -> None:
    """List available Ollama models."""
    from abbo.realworld.agents.llm_provider import is_ollama_available, list_ollama_models

    if not is_ollama_available():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    models = list_ollama_models()
    table = Table(title="Available Ollama Models")
    table.add_column("Model", style="cyan")
    for m in models:
        table.add_row(m)
    console.print(table)


@app.command()
def run_bayes_demo(
    bug: str = typer.Option("all", help="Bug to fix: bug_a, bug_b, bug_c, or all"),
    model: str = typer.Option("qwen2.5:7b", help="Ollama model name"),
    temperature: float = typer.Option(0.2, help="LLM temperature"),
) -> None:
    """Run Bayesian orchestration vs heuristic baselines on toy bugs."""
    from abbo.realworld.agents.bayes_fix_runner import (
        BUGGY_SOURCES,
        CostConfig,
        run_comparison,
    )
    from abbo.realworld.agents.llm_provider import LLMConfig, is_ollama_available
    from abbo.utils.io import save_json

    if not is_ollama_available():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    llm_config = LLMConfig(model=model, temperature=temperature)
    cost_config = CostConfig()
    bugs = list(BUGGY_SOURCES.keys()) if bug == "all" else [bug]

    all_results = []

    for bug_id in bugs:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Bug: {bug_id}[/bold]")
        results = run_comparison(bug_id, llm_config, cost_config)
        all_results.extend(results)

    # Summary table
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    table = Table(title="Bayesian vs Heuristic Comparison")
    table.add_column("Bug", style="cyan")
    table.add_column("Policy", style="yellow")
    table.add_column("Fixed", style="green")
    table.add_column("Cost", style="red")
    table.add_column("Reward", style="green")
    table.add_column("Utility", style="bold white")
    table.add_column("Steps", style="dim")
    for r in all_results:
        table.add_row(
            r.bug_id,
            r.policy_name,
            "[green]YES[/green]" if r.fixed else "[red]NO[/red]",
            f"{r.total_cost:.1f}",
            f"{r.reward:.1f}",
            f"{r.utility:.1f}",
            str(len(r.actions)),
        )
    console.print(table)

    out_dir = _results_dir("bayes_demo")
    save_json(
        [
            {
                "bug_id": r.bug_id,
                "policy": r.policy_name,
                "fixed": r.fixed,
                "cost": r.total_cost,
                "reward": r.reward,
                "utility": r.utility,
                "steps": len(r.actions),
                "actions": [
                    {
                        "step": a.step,
                        "type": a.action_type,
                        "label": a.label,
                        "cost": a.cost,
                        "observation": a.observation,
                        "belief_before": a.belief_before,
                        "belief_after": a.belief_after,
                    }
                    for a in r.actions
                ],
            }
            for r in all_results
        ],
        out_dir / "bayes_comparison.json",
    )
    console.print(f"\nResults saved to {out_dir}")


@app.command()
def run_hard_demo(
    bug: str = typer.Option("all", help="Bug: bug_d..bug_h, or 'all' for all hard bugs"),
    model: str = typer.Option("qwen2.5:7b", help="Ollama model name"),
    temperature: float = typer.Option(0.4, help="LLM temperature (higher = harder)"),
) -> None:
    """Run Bayesian orchestration vs heuristics on HARD bugs."""
    from abbo.realworld.agents.bayes_fix_runner import (
        CostConfig,
        run_comparison_hard,
    )
    from abbo.realworld.agents.hard_bugs import BUGGY_SOURCES_HARD
    from abbo.realworld.agents.llm_provider import LLMConfig, is_ollama_available
    from abbo.utils.io import save_json

    if not is_ollama_available():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    llm_config = LLMConfig(model=model, temperature=temperature)
    cost_config = CostConfig()
    bugs = list(BUGGY_SOURCES_HARD.keys()) if bug == "all" else [bug]

    all_results = []

    for bug_id in bugs:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Hard Bug: {bug_id}[/bold]")
        results = run_comparison_hard(bug_id, llm_config, cost_config)
        all_results.extend(results)

    # Summary table
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    table = Table(title="Hard Bugs: Bayesian vs Heuristic")
    table.add_column("Bug", style="cyan")
    table.add_column("Policy", style="yellow")
    table.add_column("Fixed", style="green")
    table.add_column("Cost", style="red")
    table.add_column("Reward", style="green")
    table.add_column("Utility", style="bold white")
    table.add_column("Steps", style="dim")
    for r in all_results:
        table.add_row(
            r.bug_id,
            r.policy_name,
            "[green]YES[/green]" if r.fixed else "[red]NO[/red]",
            f"{r.total_cost:.1f}",
            f"{r.reward:.1f}",
            f"{r.utility:.1f}",
            str(len(r.actions)),
        )
    console.print(table)

    # Compute per-policy averages
    from collections import defaultdict
    policy_stats: dict[str, list[float]] = defaultdict(list)
    policy_fixes: dict[str, list[bool]] = defaultdict(list)
    for r in all_results:
        policy_stats[r.policy_name].append(r.utility)
        policy_fixes[r.policy_name].append(r.fixed)

    console.print()
    avg_table = Table(title="Average Results by Policy")
    avg_table.add_column("Policy", style="yellow")
    avg_table.add_column("Fix Rate", style="green")
    avg_table.add_column("Avg Utility", style="bold white")
    for policy_name in ["bayes", "h1_one_shot", "h3_generate_twice"]:
        if policy_name in policy_stats:
            avg_u = sum(policy_stats[policy_name]) / len(policy_stats[policy_name])
            fix_rate = sum(policy_fixes[policy_name]) / len(policy_fixes[policy_name])
            avg_table.add_row(
                policy_name,
                f"{fix_rate:.0%}",
                f"{avg_u:.1f}",
            )
    console.print(avg_table)

    out_dir = _results_dir("hard_demo")
    save_json(
        [
            {
                "bug_id": r.bug_id,
                "policy": r.policy_name,
                "fixed": r.fixed,
                "cost": r.total_cost,
                "reward": r.reward,
                "utility": r.utility,
                "steps": len(r.actions),
            }
            for r in all_results
        ],
        out_dir / "hard_comparison.json",
    )
    console.print(f"\nResults saved to {out_dir}")


@app.command()
def run_mixed_demo(
    model: str = typer.Option("qwen2.5:7b", help="Ollama model name"),
    temperature: float = typer.Option(0.2, help="LLM temperature"),
) -> None:
    """Run Bayesian vs heuristics on a MIX of easy + hard bugs."""
    from abbo.realworld.agents.bayes_fix_runner import (
        CostConfig,
        run_comparison,
        run_comparison_hard,
    )
    from abbo.realworld.agents.llm_provider import LLMConfig, is_ollama_available
    from abbo.utils.io import save_json

    if not is_ollama_available():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    llm_config = LLMConfig(model=model, temperature=temperature)
    cost_config = CostConfig()

    # Mix: 3 easy bugs + 2 hard bugs
    all_results = []

    # Easy bugs (high fix rate)
    for bug_id in ["bug_a", "bug_b"]:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Easy Bug: {bug_id}[/bold]")
        results = run_comparison(bug_id, llm_config, cost_config)
        all_results.extend(results)

    # Hard bugs (low fix rate)
    for bug_id in ["bug_f", "bug_h"]:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Hard Bug: {bug_id}[/bold]")
        results = run_comparison_hard(bug_id, llm_config, cost_config)
        all_results.extend(results)

    # Summary
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    table = Table(title="Mixed Benchmark: Bayesian vs Heuristic")
    table.add_column("Bug", style="cyan")
    table.add_column("Difficulty", style="dim")
    table.add_column("Policy", style="yellow")
    table.add_column("Fixed", style="green")
    table.add_column("Cost", style="red")
    table.add_column("Utility", style="bold white")
    for r in all_results:
        diff = "easy" if r.bug_id in ("bug_a", "bug_b", "bug_c") else "hard"
        table.add_row(
            r.bug_id, diff, r.policy_name,
            "[green]YES[/green]" if r.fixed else "[red]NO[/red]",
            f"{r.total_cost:.1f}",
            f"{r.utility:.1f}",
        )
    console.print(table)

    # Per-policy summary
    from collections import defaultdict
    stats: dict[str, list[float]] = defaultdict(list)
    fixes: dict[str, list[bool]] = defaultdict(list)
    for r in all_results:
        stats[r.policy_name].append(r.utility)
        fixes[r.policy_name].append(r.fixed)

    console.print()
    avg_table = Table(title="Overall Results")
    avg_table.add_column("Policy", style="yellow")
    avg_table.add_column("Fix Rate", style="green")
    avg_table.add_column("Total Utility", style="bold white")
    avg_table.add_column("Avg Utility", style="bold white")
    for policy_name in sorted(stats.keys()):
        total = sum(stats[policy_name])
        avg = total / len(stats[policy_name])
        fix_rate = sum(fixes[policy_name]) / len(fixes[policy_name])
        avg_table.add_row(
            policy_name,
            f"{fix_rate:.0%}",
            f"{total:.1f}",
            f"{avg:.1f}",
        )
    console.print(avg_table)

    out_dir = _results_dir("mixed_demo")
    save_json(
        [
            {
                "bug_id": r.bug_id,
                "policy": r.policy_name,
                "fixed": r.fixed,
                "cost": r.total_cost,
                "utility": r.utility,
            }
            for r in all_results
        ],
        out_dir / "mixed_comparison.json",
    )
    console.print(f"\nResults saved to {out_dir}")


@app.command()
def compare_agents(
    bug: str = typer.Option("all", help="Bug: bug_r1..bug_r8, or 'all'"),
    trials: int = typer.Option(1, help="Number of trials per bug (for statistics)"),
    model: str = typer.Option("qwen2.5:7b", help="Ollama model name"),
    temperature: float = typer.Option(0.3, help="LLM temperature"),
    max_retries: int = typer.Option(3, help="Max retries for simple agent"),
) -> None:
    """Compare Simple Agent (Claude Code style) vs Bayesian POMDP Agent.

    This is the main experiment: same bugs, same LLM, fair comparison.
    The simple agent uses a retry loop (generate → test → retry).
    The Bayesian agent uses belief tracking and adaptive decisions.
    """
    from collections import defaultdict

    from abbo.realworld.agents.bayes_agent import run_bayes_agent, AgentCostConfig
    from abbo.realworld.agents.llm_provider import LLMConfig, is_ollama_available
    from abbo.realworld.agents.real_bugs import BUGGY_SOURCES_REAL, BUG_METADATA
    from abbo.realworld.agents.simple_agent import (
        run_simple_agent,
        run_simple_agent_escalating,
    )
    from abbo.utils.io import save_json

    if not is_ollama_available():
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    llm_config = LLMConfig(model=model, temperature=temperature)
    cost_config = AgentCostConfig()
    bugs = sorted(BUGGY_SOURCES_REAL.keys()) if bug == "all" else [bug]

    console.print(f"[bold]Comparing agents on {len(bugs)} bugs, {trials} trial(s) each[/bold]")
    console.print(
        f"Model: {model}, Temperature: {temperature}, "
        f"Costs: LLM={cost_config.c_llm_call}, "
        f"FullTest={cost_config.c_full_test}, "
        f"Critic={cost_config.c_critic_test}, "
        f"Reward={cost_config.reward}\n"
    )

    all_results: list[dict] = []

    for bug_id in bugs:
        meta = BUG_METADATA[bug_id]
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(
            f"[bold]{bug_id}[/bold] ({meta['difficulty']}): {meta['description']}"
        )

        for trial in range(trials):
            if trials > 1:
                console.print(f"\n  [dim]--- Trial {trial + 1}/{trials} ---[/dim]")

            # 1. Simple Agent (retry loop)
            console.print(f"\n[bold yellow]  Simple Agent (retry loop):[/bold yellow]")
            simple_result = run_simple_agent(
                bug_id, llm_config, cost_config, max_retries=max_retries,
            )

            # 2. Simple Agent with escalating prompts
            console.print(f"\n[bold yellow]  Simple Agent (escalating prompts):[/bold yellow]")
            escalating_result = run_simple_agent_escalating(
                bug_id, llm_config, cost_config, max_retries=max_retries,
            )

            # 3. Bayesian POMDP Agent
            console.print(f"\n[bold magenta]  Bayesian POMDP Agent:[/bold magenta]")
            bayes_result = run_bayes_agent(
                bug_id, llm_config, cost_config,
            )

            for r in [simple_result, escalating_result, bayes_result]:
                all_results.append({
                    "bug_id": bug_id,
                    "difficulty": meta["difficulty"],
                    "trial": trial,
                    "policy": r.policy_name,
                    "fixed": r.fixed,
                    "cost": r.total_cost,
                    "reward": r.reward,
                    "utility": r.utility,
                    "llm_calls": r.llm_calls,
                    "test_runs": getattr(r, "test_runs", 0),
                    "critic_runs": getattr(r, "critic_runs", 0),
                    "wall_seconds": r.wall_clock_seconds,
                })

    # ---- Summary tables ----
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")

    # Per-bug table
    table = Table(title="Simple Agent vs Bayesian POMDP Agent")
    table.add_column("Bug", style="cyan")
    table.add_column("Difficulty", style="dim")
    table.add_column("Agent", style="yellow")
    table.add_column("Fixed", style="green")
    table.add_column("Cost", style="red")
    table.add_column("Utility", style="bold white")
    table.add_column("LLM Calls", style="dim")
    table.add_column("Tests", style="dim")

    for r in all_results:
        table.add_row(
            r["bug_id"], r["difficulty"], r["policy"],
            "[green]YES[/green]" if r["fixed"] else "[red]NO[/red]",
            f"{r['cost']:.1f}",
            f"{r['utility']:.1f}",
            str(r["llm_calls"]),
            f"{r['test_runs']}+{r['critic_runs']}c",
        )
    console.print(table)

    # Per-policy aggregate
    policy_stats: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        policy_stats[r["policy"]].append(r)

    console.print()
    agg_table = Table(title="Aggregate Results by Agent")
    agg_table.add_column("Agent", style="yellow")
    agg_table.add_column("Fix Rate", style="green")
    agg_table.add_column("Avg Cost", style="red")
    agg_table.add_column("Avg Utility", style="bold white")
    agg_table.add_column("Avg LLM Calls", style="dim")
    agg_table.add_column("Total Utility", style="bold cyan")

    for policy in ["simple_agent", "simple_agent_escalating", "bayes_pomdp"]:
        if policy not in policy_stats:
            continue
        runs = policy_stats[policy]
        fix_rate = sum(1 for r in runs if r["fixed"]) / len(runs)
        avg_cost = sum(r["cost"] for r in runs) / len(runs)
        avg_util = sum(r["utility"] for r in runs) / len(runs)
        avg_llm = sum(r["llm_calls"] for r in runs) / len(runs)
        total_util = sum(r["utility"] for r in runs)
        agg_table.add_row(
            policy,
            f"{fix_rate:.0%}",
            f"{avg_cost:.1f}",
            f"{avg_util:.1f}",
            f"{avg_llm:.1f}",
            f"{total_util:.1f}",
        )
    console.print(agg_table)

    # Per-difficulty breakdown
    console.print()
    diff_table = Table(title="Results by Difficulty Level")
    diff_table.add_column("Difficulty", style="dim")
    diff_table.add_column("Agent", style="yellow")
    diff_table.add_column("Fix Rate", style="green")
    diff_table.add_column("Avg Utility", style="bold white")

    for difficulty in ["easy", "medium", "hard"]:
        for policy in ["simple_agent", "simple_agent_escalating", "bayes_pomdp"]:
            runs = [r for r in all_results
                    if r["difficulty"] == difficulty and r["policy"] == policy]
            if not runs:
                continue
            fix_rate = sum(1 for r in runs if r["fixed"]) / len(runs)
            avg_util = sum(r["utility"] for r in runs) / len(runs)
            diff_table.add_row(difficulty, policy, f"{fix_rate:.0%}", f"{avg_util:.1f}")
    console.print(diff_table)

    # Save results
    out_dir = _results_dir("agent_comparison")
    save_json(all_results, out_dir / "comparison_results.json")
    console.print(f"\nResults saved to {out_dir}")


@app.command()
def bayes_analysis() -> None:
    """Show expected utility of Bayes vs heuristics across fix probabilities."""
    from abbo.realworld.agents.bayes_fix_runner import CostConfig, compute_expected_utilities

    cost_config = CostConfig()
    console.print("[bold]Expected Utility Analysis: Bayes vs Heuristics[/bold]")
    console.print(
        f"Costs: generator={cost_config.c_generator}, "
        f"critic={cost_config.c_critic}, "
        f"verifier={cost_config.c_verifier}, "
        f"reward={cost_config.reward}\n"
    )

    table = Table(title="Expected Utility by Fix Probability")
    table.add_column("P(fix)", style="bold")
    table.add_column("Bayes EU", style="cyan")
    table.add_column("Bayes Path", style="dim")
    table.add_column("H1 EU", style="yellow")
    table.add_column("H2 EU", style="yellow")
    table.add_column("H3 EU", style="yellow")
    table.add_column("Bayes Δ", style="green")

    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        res = compute_expected_utilities(p, cost_config)
        best_heuristic = max(
            res["h1_one_shot"]["expected_utility"],
            res["h2_fixed_workflow"]["expected_utility"],
            res["h3_generate_twice"]["expected_utility"],
        )
        delta = res["bayes"]["expected_utility"] - best_heuristic
        table.add_row(
            f"{p:.1f}",
            f"{res['bayes']['expected_utility']:.1f}",
            res["bayes"].get("path", ""),
            f"{res['h1_one_shot']['expected_utility']:.1f}",
            f"{res['h2_fixed_workflow']['expected_utility']:.1f}",
            f"{res['h3_generate_twice']['expected_utility']:.1f}",
            f"{delta:+.1f}",
        )
    console.print(table)
    console.print(
        "\n[dim]Bayes Δ = Bayes EU - best heuristic EU. "
        "Positive means Bayes outperforms.[/dim]"
    )


if __name__ == "__main__":
    app()
