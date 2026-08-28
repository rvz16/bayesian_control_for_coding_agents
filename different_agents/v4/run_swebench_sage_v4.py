"""Evaluation script for SAGE Agent v4 on SWE-bench.

SWE-bench is the ideal benchmark for SAGE + tool calling because:
- Real tool usage (search, read, edit, test)
- Multi-step reasoning required
- Uncertainty matters (which file? what change?)

Usage:
    # With SWE-bench lite (easier subset)
    python examples/run_swebench_sage_v4.py \
        --service-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset princeton-nlp/SWE-bench_Lite \
        --limit 10

    # With full SWE-bench
    python examples/run_swebench_sage_v4.py \
        --service-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --dataset princeton-nlp/SWE-bench \
        --limit 10
"""

import argparse
import json
import os
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

from datasets import load_dataset

# Project root is 2 levels up from different_agents/v4/
ROOT = Path(__file__).resolve().parents[2]
V4_DIR = Path(__file__).resolve().parent
SHARED_DIR = ROOT / "different_agents" / "shared"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(SHARED_DIR))

from langgraph_sage_agent_v4_swe import (
    GraphDeps,
    build_graph,
    create_initial_state,
    SWEToolExecutor,
    DEFAULT_CONFIG,
)


# ============================================================================
# SWE-bench Setup
# ============================================================================

def setup_repo(repo: str, base_commit: str, workdir: Path) -> Path:
    """Clone repository and checkout base commit."""
    repo_name = repo.split("/")[-1]
    repo_path = workdir / repo_name

    if repo_path.exists():
        shutil.rmtree(repo_path)

    print(f"Cloning {repo}...")
    subprocess.run(
        f"git clone https://github.com/{repo}.git {repo_path}",
        shell=True,
        check=True,
        capture_output=True,
    )

    print(f"Checking out {base_commit[:8]}...")
    subprocess.run(
        f"git checkout {base_commit}",
        shell=True,
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return repo_path


def apply_patch_and_test(repo_path: Path, patch: str, test_command: str) -> dict:
    """Apply patch and run tests to evaluate."""
    try:
        # Apply patch
        if patch.strip():
            result = subprocess.run(
                f"git apply --check",
                shell=True,
                cwd=repo_path,
                input=patch,
                text=True,
                capture_output=True,
            )

            if result.returncode == 0:
                subprocess.run(
                    f"git apply",
                    shell=True,
                    cwd=repo_path,
                    input=patch,
                    text=True,
                    check=True,
                )

        # Run tests
        result = subprocess.run(
            test_command,
            shell=True,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=300,
        )

        return {
            "applied": True,
            "tests_passed": result.returncode == 0,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
        }

    except subprocess.TimeoutExpired:
        return {"applied": True, "tests_passed": False, "error": "Timeout"}
    except Exception as e:
        return {"applied": False, "tests_passed": False, "error": str(e)}


# ============================================================================
# Evaluation
# ============================================================================

def run_evaluation(
    llm,
    dataset_name: str,
    limit: int,
    config: dict,
    output_dir: str = "results",
    workdir: Optional[Path] = None,
    sandbox: bool = True,
) -> dict:
    """Run SWE-bench evaluation."""

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="test")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create work directory for repos
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="swebench_"))
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"SAGE v4 SWE-bench Evaluation")
    print(f"Dataset: {dataset_name}")
    print(f"Limit: {limit}")
    print(f"Workdir: {workdir}")
    print(f"Sandbox: {sandbox}")
    print(f"{'='*70}\n")

    # Results tracking
    results = {
        "examples": [],
        "metrics": {
            "total": 0,
            "resolved": 0,
            "partial": 0,
            "failed": 0,
        },
        "tool_stats": {
            "total_tool_calls": 0,
            "by_tool": {},
        },
        "uncertainty_stats": {
            "avg_decision_uncertainty": 0.0,
            "avg_epistemic_uncertainty": 0.0,
        },
        "config": config,
    }

    decision_uncertainties = []
    epistemic_uncertainties = []

    # Process examples
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        instance_id = item["instance_id"]
        repo = item["repo"]
        base_commit = item["base_commit"]
        issue_text = item["problem_statement"]
        test_patch = item.get("test_patch", "")
        hints = item.get("hints_text", "")

        print(f"\n{'='*70}")
        print(f"Example {i+1}/{limit}")
        print(f"Instance: {instance_id}")
        print(f"Repo: {repo}")
        print(f"{'='*70}")
        print(f"Issue: {issue_text[:300]}...")

        try:
            # Setup repository
            repo_path = setup_repo(repo, base_commit, workdir)

            # Create tool executor
            tool_executor = SWEToolExecutor(
                repo_path=str(repo_path),
                sandbox=sandbox,
            )

            # Build graph
            deps = GraphDeps(
                llm=llm,
                tool_executor=tool_executor,
                config=config,
            )
            graph = build_graph(deps)

            # Create initial state
            initial_state = create_initial_state(
                instance_id=instance_id,
                issue_description=f"{issue_text}\n\nHints: {hints}" if hints else issue_text,
                repo_path=str(repo_path),
                base_commit=base_commit,
                test_command="pytest",  # Default, could be extracted from metadata
            )

            # Run agent
            final_state = graph.invoke(initial_state)

            # Extract results
            patch = final_state.get("final_patch", "")
            trajectory = final_state.get("trajectory", [])
            status = final_state.get("status", "unknown")

            print(f"\nAgent finished: {status}")
            print(f"Iterations: {final_state.get('iteration', 0)}")
            print(f"Files found: {len(final_state.get('files_found', []))}")
            print(f"Files read: {len(final_state.get('files_read', {}))}")
            print(f"Edits made: {len(final_state.get('edits_made', []))}")

            # Evaluate result (if not sandbox)
            if not sandbox and patch:
                eval_result = apply_patch_and_test(repo_path, patch, "pytest")
                resolved = eval_result.get("tests_passed", False)
            else:
                eval_result = {"sandbox": True}
                resolved = False  # Can't verify in sandbox

            # Determine outcome
            if status == "done" and final_state.get("tests_passed"):
                outcome = "resolved"
                results["metrics"]["resolved"] += 1
            elif status == "done" and final_state.get("edits_made"):
                outcome = "partial"
                results["metrics"]["partial"] += 1
            else:
                outcome = "failed"
                results["metrics"]["failed"] += 1

            results["metrics"]["total"] += 1

            # Track tool usage
            for step in trajectory:
                tool = step.get("tool", "unknown")
                results["tool_stats"]["total_tool_calls"] += 1
                results["tool_stats"]["by_tool"][tool] = \
                    results["tool_stats"]["by_tool"].get(tool, 0) + 1

            # Track uncertainty
            decision_uncertainties.append(final_state.get("tool_decision_uncertainty", 0))
            epistemic_uncertainties.append(final_state.get("epistemic_uncertainty", 0))

            # Save example result
            example_result = {
                "instance_id": instance_id,
                "repo": repo,
                "outcome": outcome,
                "status": status,
                "iterations": final_state.get("iteration", 0),
                "files_found": final_state.get("files_found", []),
                "files_read": list(final_state.get("files_read", {}).keys()),
                "edits_made": len(final_state.get("edits_made", [])),
                "tests_passed": final_state.get("tests_passed", False),
                "patch_length": len(patch),
                "trajectory_length": len(trajectory),
                "epistemic_uncertainty": final_state.get("epistemic_uncertainty", 0),
                "tool_decision_uncertainty": final_state.get("tool_decision_uncertainty", 0),
                "confidence_score": final_state.get("confidence_score", 0),
            }
            results["examples"].append(example_result)

            print(f"\nOutcome: {outcome}")
            print(f"Epistemic uncertainty: {final_state.get('epistemic_uncertainty', 0):.2f}")

        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

            results["metrics"]["total"] += 1
            results["metrics"]["failed"] += 1

            results["examples"].append({
                "instance_id": instance_id,
                "outcome": "error",
                "error": str(e),
            })

    # Calculate averages
    if decision_uncertainties:
        results["uncertainty_stats"]["avg_decision_uncertainty"] = \
            sum(decision_uncertainties) / len(decision_uncertainties)
    if epistemic_uncertainties:
        results["uncertainty_stats"]["avg_epistemic_uncertainty"] = \
            sum(epistemic_uncertainties) / len(epistemic_uncertainties)

    # Calculate rates
    n = results["metrics"]["total"]
    if n > 0:
        results["metrics"]["resolve_rate"] = results["metrics"]["resolved"] / n
        results["metrics"]["partial_rate"] = results["metrics"]["partial"] / n
        results["metrics"]["fail_rate"] = results["metrics"]["failed"] / n

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {results['metrics']['total']}")
    print(f"Resolved: {results['metrics']['resolved']} ({results['metrics'].get('resolve_rate', 0):.1%})")
    print(f"Partial: {results['metrics']['partial']} ({results['metrics'].get('partial_rate', 0):.1%})")
    print(f"Failed: {results['metrics']['failed']} ({results['metrics'].get('fail_rate', 0):.1%})")

    print(f"\nTool Usage:")
    for tool, count in sorted(results["tool_stats"]["by_tool"].items(), key=lambda x: -x[1]):
        print(f"  {tool}: {count}")

    print(f"\nUncertainty Stats:")
    print(f"  Avg decision uncertainty: {results['uncertainty_stats']['avg_decision_uncertainty']:.3f}")
    print(f"  Avg epistemic uncertainty: {results['uncertainty_stats']['avg_epistemic_uncertainty']:.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"sage_v4_swebench_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAGE v4 SWE-bench Evaluation")

    parser.add_argument(
        "--service-url",
        default="http://localhost:8000/v1",
        help="LLM service URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        default="princeton-nlp/SWE-bench_Lite",
        help="SWE-bench dataset name",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for repositories",
    )
    parser.add_argument(
        "--no-sandbox",
        action="store_true",
        help="Actually apply changes (not sandbox mode)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum tool iterations per task",
    )
    parser.add_argument(
        "--enable-resampling",
        action="store_true",
        default=True,
        help="Enable decision resampling",
    )

    args = parser.parse_args()

    # Initialize LLM client
    import openai

    class SimpleLLMClient:
        def __init__(self, base_url: str, model: str, api_key: str = None):
            # Try to get API key from various sources
            if api_key is None:
                api_key = os.environ.get("OPENROUTER_API_KEY") or \
                          os.environ.get("OPENAI_API_KEY") or \
                          "not-needed"
            self.client = openai.OpenAI(
                base_url=base_url,
                api_key=api_key,
            )
            self.model = model

        def generate(self, prompt: str, temperature: float = 0.7) -> str:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content

        def complete(self, prompt: str) -> str:
            """Alias for generate to match LLMClient protocol."""
            return self.generate(prompt)

    llm = SimpleLLMClient(args.service_url, args.model)
    print(f"Using SimpleLLMClient with {args.model}")

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["max_iterations"] = args.max_iterations
    config["enable_resampling"] = args.enable_resampling

    # Run evaluation
    workdir = Path(args.workdir) if args.workdir else None

    results = run_evaluation(
        llm=llm,
        dataset_name=args.dataset,
        limit=args.limit,
        config=config,
        output_dir=args.output_dir,
        workdir=workdir,
        sandbox=not args.no_sandbox,
    )

    return results


if __name__ == "__main__":
    main()
