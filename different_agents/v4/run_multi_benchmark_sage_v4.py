"""Evaluation script for SAGE Agent v4 on code generation benchmarks.

SAGE v4 uses:
- Natural code generation (no tool call format)
- Tool calling for decisions (validation strategy, recovery strategy)
- SAGE uncertainty for decision-making

Usage:
    python examples/run_multi_benchmark_sage_v4.py \
        --service-url http://localhost:8000/v1 \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --limit 20 \
        --benchmarks humaneval mbpp gsm8k
"""

import argparse
import json
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset

# Project root is 2 levels up from different_agents/v4/
ROOT = Path(__file__).resolve().parents[2]
V4_DIR = Path(__file__).resolve().parent
SHARED_DIR = ROOT / "different_agents" / "shared"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V4_DIR))
sys.path.insert(0, str(SHARED_DIR))

from langgraph_sage_agent_v4 import (
    GraphDeps,
    build_graph,
    create_initial_state,
    DEFAULT_CONFIG,
)


# ============================================================================
# Code Evaluation Functions
# ============================================================================

def evaluate_code(code: str, test: str, entry_point: str, timeout: float = 5.0) -> dict:
    """Execute generated code and check if it passes tests.

    Returns:
        dict with keys: passed (bool), errors (list), output (str)
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    try:
        # Extract just the function from generated code
        clean_code = extract_function_code(code, entry_point)

        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            # Create namespace and execute code
            namespace = {}
            exec(clean_code, namespace)

            # Execute tests
            exec(test, namespace)

            signal.alarm(0)  # Cancel timeout

            return {
                "passed": True,
                "errors": [],
                "output": "All tests passed",
            }

        except TimeoutError:
            return {
                "passed": False,
                "errors": ["Execution timed out"],
                "output": "",
            }
        except AssertionError as e:
            signal.alarm(0)
            return {
                "passed": False,
                "errors": [f"Assertion failed: {str(e)}"],
                "output": "",
            }
        except Exception as e:
            signal.alarm(0)
            return {
                "passed": False,
                "errors": [f"{type(e).__name__}: {str(e)}"],
                "output": "",
            }

    except Exception as e:
        return {
            "passed": False,
            "errors": [f"Code extraction failed: {str(e)}"],
            "output": "",
        }


def extract_function_code(generated: str, entry_point: str) -> str:
    """Extract function code from generated response."""

    # Try to find code block
    code_block_match = re.search(r'```python\n(.*?)```', generated, re.DOTALL)
    if code_block_match:
        code = code_block_match.group(1)
    else:
        code_block_match = re.search(r'```\n(.*?)```', generated, re.DOTALL)
        if code_block_match:
            code = code_block_match.group(1)
        else:
            code = generated

    # Find the function definition
    func_pattern = rf'(def\s+{re.escape(entry_point)}\s*\([^)]*\).*?)(?=\ndef\s|\nclass\s|\Z)'
    func_match = re.search(func_pattern, code, re.DOTALL)

    if func_match:
        return func_match.group(1).strip()

    # If no match, return the whole code
    return code.strip()


def evaluate_gsm8k_answer(generated: str, expected: str) -> dict:
    """Check if GSM8K answer is correct.

    Returns:
        dict with keys: passed (bool), errors (list), output (str)
    """
    try:
        # Extract number from generated answer
        # Look for #### format first
        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', str(generated))
        if match:
            generated_num = float(match.group(1).replace(',', ''))
        else:
            # Try to find any number near the end
            numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', str(generated))
            if numbers:
                generated_num = float(numbers[-1].replace(',', ''))
            else:
                return {
                    "passed": False,
                    "errors": ["No number found in generated answer"],
                    "output": f"Generated: {generated}",
                }

        # Extract expected number
        expected_match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', str(expected))
        if expected_match:
            expected_num = float(expected_match.group(1).replace(',', ''))
        else:
            expected_num = float(str(expected).replace(',', ''))

        # Compare
        is_correct = abs(generated_num - expected_num) < 1e-6

        return {
            "passed": is_correct,
            "errors": [] if is_correct else [f"Expected {expected_num}, got {generated_num}"],
            "output": f"Generated: {generated_num}, Expected: {expected_num}",
        }

    except Exception as e:
        return {
            "passed": False,
            "errors": [f"Answer comparison failed: {str(e)}"],
            "output": "",
        }


# ============================================================================
# Dataset Loading
# ============================================================================

def load_humaneval(limit: int) -> list[dict]:
    """Load HumanEval dataset."""
    dataset = load_dataset("openai/openai_humaneval", split="test")

    examples = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        examples.append({
            "task": item["prompt"],
            "test_cases": item["test"],
            "entry_point": item["entry_point"],
            "canonical_solution": item["canonical_solution"],
            "benchmark": "humaneval",
            "task_id": item["task_id"],
        })

    return examples


def load_mbpp(limit: int) -> list[dict]:
    """Load MBPP dataset."""
    dataset = load_dataset("mbpp", "sanitized", split="test")

    examples = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break

        # Get prompt text (field name varies)
        prompt_text = item.get("prompt") or item.get("text", "")

        # Build test string from test_list
        test_string = "\n".join(item.get("test_list", []))

        # Extract entry point from test
        entry_point_match = re.search(r'assert\s+(\w+)\s*\(', test_string)
        entry_point = entry_point_match.group(1) if entry_point_match else "solution"

        examples.append({
            "task": prompt_text,
            "test_cases": test_string,
            "entry_point": entry_point,
            "canonical_solution": item.get("code", ""),
            "benchmark": "mbpp",
            "task_id": str(item.get("task_id", i)),
        })

    return examples


def load_gsm8k(limit: int) -> list[dict]:
    """Load GSM8K dataset."""
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    examples = []
    for i, item in enumerate(dataset):
        if i >= limit:
            break
        examples.append({
            "task": item["question"],
            "test_cases": "",  # No code tests
            "entry_point": "",
            "canonical_solution": item["answer"],
            "benchmark": "gsm8k",
            "task_id": str(i),
        })

    return examples


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def run_evaluation(
    llm,
    benchmarks: list[str],
    limit: int,
    config: dict,
    output_dir: str = "results",
) -> dict:
    """Run evaluation on specified benchmarks."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build graph
    deps = GraphDeps(llm=llm, config=config)
    graph = build_graph(deps)

    # Load datasets
    all_examples = []
    for benchmark in benchmarks:
        if benchmark == "humaneval":
            all_examples.extend(load_humaneval(limit))
        elif benchmark == "mbpp":
            all_examples.extend(load_mbpp(limit))
        elif benchmark == "gsm8k":
            all_examples.extend(load_gsm8k(limit))

    print(f"\n{'='*70}")
    print(f"SAGE v4 Evaluation")
    print(f"Benchmarks: {benchmarks}")
    print(f"Total examples: {len(all_examples)}")
    print(f"{'='*70}\n")

    # Results tracking
    results = {
        "examples": [],
        "metrics": {
            "total": 0,
            "correct": 0,
            "failed": 0,
            "by_benchmark": {},
        },
        "uncertainty_stats": {
            "epistemic_sum": 0.0,
            "aleatoric_sum": 0.0,
            "validation_decision_uncertainty_sum": 0.0,
            "recovery_decision_uncertainty_sum": 0.0,
        },
        "decision_stats": {
            "validation_strategies": {},
            "recovery_strategies": {},
        },
    }

    for benchmark in benchmarks:
        results["metrics"]["by_benchmark"][benchmark] = {
            "total": 0,
            "correct": 0,
        }

    # Process each example
    for i, example in enumerate(all_examples):
        print(f"\n{'='*70}")
        print(f"Example {i+1}/{len(all_examples)} [{example['benchmark']}]")
        print(f"Task ID: {example.get('task_id', 'N/A')}")
        print(f"{'='*70}")
        print(f"Task: {example['task'][:200]}...")

        try:
            # Create initial state
            initial_state = create_initial_state(
                task=example["task"],
                benchmark=example["benchmark"],
                test_cases=example["test_cases"],
                expected_output=example["canonical_solution"],
                entry_point=example["entry_point"],
            )

            # Run graph
            final_state = graph.invoke(initial_state)

            # Extract generated code
            generated_code = final_state.get("final_code") or final_state.get("code", "")

            print(f"\nGenerated code preview:")
            print(generated_code[:500] + "..." if len(generated_code) > 500 else generated_code)

            # Evaluate result
            if example["benchmark"] == "gsm8k":
                eval_result = evaluate_gsm8k_answer(
                    generated_code,
                    example["canonical_solution"]
                )
            else:
                eval_result = evaluate_code(
                    generated_code,
                    example["test_cases"],
                    example["entry_point"],
                )

            is_correct = eval_result["passed"]

            # Update metrics
            results["metrics"]["total"] += 1
            results["metrics"]["by_benchmark"][example["benchmark"]]["total"] += 1

            if is_correct:
                results["metrics"]["correct"] += 1
                results["metrics"]["by_benchmark"][example["benchmark"]]["correct"] += 1
            else:
                results["metrics"]["failed"] += 1

            # Update uncertainty stats
            results["uncertainty_stats"]["epistemic_sum"] += final_state.get("epistemic_uncertainty", 0)
            results["uncertainty_stats"]["aleatoric_sum"] += final_state.get("aleatoric_uncertainty", 0)
            results["uncertainty_stats"]["validation_decision_uncertainty_sum"] += final_state.get("validation_decision_uncertainty", 0)
            results["uncertainty_stats"]["recovery_decision_uncertainty_sum"] += final_state.get("recovery_decision_uncertainty", 0)

            # Update decision stats
            val_strategy = final_state.get("validation_strategy", "unknown")
            results["decision_stats"]["validation_strategies"][val_strategy] = \
                results["decision_stats"]["validation_strategies"].get(val_strategy, 0) + 1

            rec_strategy = final_state.get("recovery_strategy", "none")
            if rec_strategy:
                results["decision_stats"]["recovery_strategies"][rec_strategy] = \
                    results["decision_stats"]["recovery_strategies"].get(rec_strategy, 0) + 1

            # Save example result
            example_result = {
                "task_id": example.get("task_id"),
                "benchmark": example["benchmark"],
                "task": example["task"][:500],
                "generated_code": generated_code[:2000],
                "is_correct": is_correct,
                "errors": eval_result.get("errors", []),
                "epistemic_uncertainty": final_state.get("epistemic_uncertainty", 0),
                "aleatoric_uncertainty": final_state.get("aleatoric_uncertainty", 0),
                "validation_strategy": val_strategy,
                "validation_decision_uncertainty": final_state.get("validation_decision_uncertainty", 0),
                "recovery_strategy": rec_strategy,
                "generation_attempts": final_state.get("generation_attempts", 1),
                "total_attempts": final_state.get("total_attempts", 1),
                "status": final_state.get("status", "unknown"),
            }
            results["examples"].append(example_result)

            print(f"\nResult: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
            print(f"Epistemic uncertainty: {final_state.get('epistemic_uncertainty', 0):.2f}")
            print(f"Validation strategy: {val_strategy}")
            if eval_result.get("errors"):
                print(f"Errors: {eval_result['errors'][:2]}")

        except Exception as e:
            print(f"\nERROR: {type(e).__name__}: {str(e)}")
            traceback.print_exc()

            results["metrics"]["total"] += 1
            results["metrics"]["failed"] += 1
            results["metrics"]["by_benchmark"][example["benchmark"]]["total"] += 1

            results["examples"].append({
                "task_id": example.get("task_id"),
                "benchmark": example["benchmark"],
                "is_correct": False,
                "errors": [str(e)],
                "status": "error",
            })

    # Calculate averages
    n = results["metrics"]["total"]
    if n > 0:
        results["uncertainty_stats"]["epistemic_avg"] = results["uncertainty_stats"]["epistemic_sum"] / n
        results["uncertainty_stats"]["aleatoric_avg"] = results["uncertainty_stats"]["aleatoric_sum"] / n
        results["uncertainty_stats"]["validation_decision_uncertainty_avg"] = \
            results["uncertainty_stats"]["validation_decision_uncertainty_sum"] / n

        results["metrics"]["accuracy"] = results["metrics"]["correct"] / n

        for benchmark in results["metrics"]["by_benchmark"]:
            bm = results["metrics"]["by_benchmark"][benchmark]
            if bm["total"] > 0:
                bm["accuracy"] = bm["correct"] / bm["total"]

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total examples: {results['metrics']['total']}")
    print(f"Overall accuracy: {results['metrics'].get('accuracy', 0):.2%}")
    print()

    for benchmark in benchmarks:
        bm = results["metrics"]["by_benchmark"][benchmark]
        print(f"{benchmark}: {bm['correct']}/{bm['total']} = {bm.get('accuracy', 0):.2%}")

    print()
    print("Uncertainty Stats:")
    print(f"  Avg epistemic uncertainty: {results['uncertainty_stats'].get('epistemic_avg', 0):.3f}")
    print(f"  Avg aleatoric uncertainty: {results['uncertainty_stats'].get('aleatoric_avg', 0):.3f}")
    print(f"  Avg validation decision uncertainty: {results['uncertainty_stats'].get('validation_decision_uncertainty_avg', 0):.3f}")

    print()
    print("Validation Strategy Distribution:")
    for strategy, count in results["decision_stats"]["validation_strategies"].items():
        print(f"  {strategy}: {count} ({count/n:.1%})")

    print()
    print("Recovery Strategy Distribution:")
    for strategy, count in results["decision_stats"]["recovery_strategies"].items():
        print(f"  {strategy}: {count}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"sage_v4_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SAGE v4 Code Generation Evaluation")

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
        "--limit",
        type=int,
        default=20,
        help="Number of examples per benchmark",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["humaneval", "mbpp", "gsm8k"],
        choices=["humaneval", "mbpp", "gsm8k"],
        help="Benchmarks to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--enable-resampling",
        action="store_true",
        default=True,
        help="Enable resampling for decisions",
    )
    parser.add_argument(
        "--decision-samples",
        type=int,
        default=3,
        help="Number of samples for decision resampling",
    )

    args = parser.parse_args()

    # Initialize LLM client
    # Try different client implementations
    try:
        from sage_agent.llm.tts_client import TTSLLMClient
        llm = TTSLLMClient(
            base_url=args.service_url,
            model=args.model,
            tts_strategy="self_consistency",
            tts_budget=8,
        )
        print(f"Using TTSLLMClient with {args.model}")
    except ImportError:
        try:
            from sage_agent.llm.openai_client import OpenAILLMClient
            llm = OpenAILLMClient(
                base_url=args.service_url,
                model=args.model,
            )
            print(f"Using OpenAILLMClient with {args.model}")
        except ImportError:
            from sage_agent.llm.base import LLMClient

            class SimpleLLMClient(LLMClient):
                def __init__(self, base_url: str, model: str):
                    import openai
                    self.client = openai.OpenAI(
                        base_url=base_url,
                        api_key="not-needed",
                    )
                    self.model = model

                def generate(self, prompt: str, temperature: float = 0.7) -> str:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                    return response.choices[0].message.content

            llm = SimpleLLMClient(args.service_url, args.model)
            print(f"Using SimpleLLMClient with {args.model}")

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["temperature"] = args.temperature
    config["enable_resampling_decisions"] = args.enable_resampling
    config["decision_samples"] = args.decision_samples

    # Run evaluation
    results = run_evaluation(
        llm=llm,
        benchmarks=args.benchmarks,
        limit=args.limit,
        config=config,
        output_dir=args.output_dir,
    )

    return results


if __name__ == "__main__":
    main()
