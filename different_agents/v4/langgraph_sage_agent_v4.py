"""SAGE Agent v4: Hybrid Natural Generation + Tool-Based Decisions

Key Innovation:
- Code generation is NATURAL (not forced into tool call format)
- Tool calling used for DECISIONS (validation strategy, recovery strategy)
- SAGE uncertainty applied where it helps (decision-making, not generation)

Architecture:
1. Generate code naturally (what LLMs do best)
2. SAGE decides: how thoroughly to test?
3. Execute tests programmatically
4. If failed, SAGE decides: how to recover?
5. Execute recovery strategy

This combines the best of both approaches:
- High-quality natural code generation
- Uncertainty-aware decision making
- Structured validation and recovery
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TypedDict, Literal, Annotated, Any
from collections import Counter

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from sage_agent.core.schema import ToolSchema, ParameterDomain
from sage_agent.core.belief import ToolCandidate
from sage_agent.llm.base import LLMClient

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Core thresholds
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_attempts": 3,
    "max_validation_retries": 2,

    # Decision-making features
    "enable_sgr_for_decisions": True,  # SGR for tool decisions
    "enable_resampling_decisions": True,  # Resample uncertain decisions
    "decision_samples": 3,  # How many samples for decision resampling
    "decision_agreement_threshold": 0.7,

    # Testing strategy
    "default_test_strategy": "adaptive",  # "quick", "comprehensive", "adaptive"
    "trust_threshold": 0.9,  # Confidence to skip testing

    # Recovery strategy
    "max_recovery_attempts": 2,
    "enable_reflexion": True,

    # Generation parameters
    "temperature": 0.7,
    "max_tokens": 2048,
}


# ============================================================================
# State Definition
# ============================================================================

class CodeGenState(TypedDict):
    """State for code generation agent."""

    # Input
    task: str
    test_cases: str  # For HumanEval/MBPP
    expected_output: str  # For GSM8K
    entry_point: str  # Function name to test
    benchmark: str  # "humaneval", "mbpp", "gsm8k"

    # Generation
    code: str
    generation_confidence: float
    generation_attempts: int

    # Validation decision (SAGE tool calling)
    validation_strategy: str  # "quick_test", "comprehensive_test", "trust_code"
    validation_decision_uncertainty: float
    validation_samples: list[str]  # If resampled

    # Testing
    test_result: dict  # {"passed": bool, "errors": list, "output": str}

    # Recovery decision (SAGE tool calling)
    recovery_strategy: str  # "regenerate", "reflect", "debug", "escalate"
    recovery_decision_uncertainty: float

    # Tracking
    total_attempts: int
    epistemic_uncertainty: float
    aleatoric_uncertainty: float

    # Output
    status: Literal["generating", "validating", "testing", "recovering", "done", "failed"]
    final_code: str
    confidence_score: float
    warning: str


# ============================================================================
# Decision Tools (SAGE uses these, not for code generation!)
# ============================================================================

# Validation strategy tools
VALIDATION_TOOLS = {
    "quick_test": ToolSchema(
        name="quick_test",
        description="Run basic test cases only (fast, covers common scenarios)",
        parameters={},
        required=frozenset(),
    ),
    "comprehensive_test": ToolSchema(
        name="comprehensive_test",
        description="Run all tests including edge cases (thorough, slower)",
        parameters={},
        required=frozenset(),
    ),
    "trust_code": ToolSchema(
        name="trust_code",
        description="Skip testing, use code as-is (only if very confident)",
        parameters={},
        required=frozenset(),
    ),
}

# Recovery strategy tools
RECOVERY_TOOLS = {
    "regenerate_fresh": ToolSchema(
        name="regenerate_fresh",
        description="Generate completely new solution from scratch",
        parameters={},
        required=frozenset(),
    ),
    "reflect_and_revise": ToolSchema(
        name="reflect_and_revise",
        description="Analyze failures deeply and revise approach",
        parameters={},
        required=frozenset(),
    ),
    "debug_incrementally": ToolSchema(
        name="debug_incrementally",
        description="Fix specific errors step by step",
        parameters={},
        required=frozenset(),
    ),
    "escalate": ToolSchema(
        name="escalate",
        description="Give up, task too hard",
        parameters={},
        required=frozenset(),
    ),
}


# ============================================================================
# Helper Functions
# ============================================================================

def extract_uncertainty_from_response(response: str) -> tuple[float, float]:
    """Extract epistemic and aleatoric uncertainty from LLM response.

    Simple heuristic:
    - Epistemic: Look for uncertainty markers ("maybe", "perhaps", "could be")
    - Aleatoric: Look for task ambiguity markers ("unclear", "ambiguous")
    """
    epistemic_markers = ["maybe", "perhaps", "could", "might", "possibly", "unsure"]
    aleatoric_markers = ["unclear", "ambiguous", "multiple ways", "depends"]

    response_lower = response.lower()

    epistemic_count = sum(1 for marker in epistemic_markers if marker in response_lower)
    aleatoric_count = sum(1 for marker in aleatoric_markers if marker in response_lower)

    # Normalize to [0, 1]
    epistemic = min(epistemic_count * 0.2, 0.9)
    aleatoric = min(aleatoric_count * 0.2, 0.9)

    return epistemic, aleatoric


def compute_agreement(samples: list[str]) -> float:
    """Compute agreement rate between samples."""
    if len(samples) <= 1:
        return 1.0

    # Compare samples (simple: exact match)
    counter = Counter(samples)
    most_common_count = counter.most_common(1)[0][1]
    agreement = most_common_count / len(samples)

    return agreement


# ============================================================================
# Graph Nodes
# ============================================================================

@dataclass
class GraphDeps:
    """Dependencies for graph nodes."""
    llm: LLMClient
    config: dict = field(default_factory=lambda: DEFAULT_CONFIG.copy())


def generate_code_node(state: CodeGenState, deps: GraphDeps) -> dict:
    """Generate code naturally (NO tool calling)."""

    print(f"\n{'='*60}")
    print(f"GENERATING CODE (Attempt {state.get('generation_attempts', 0) + 1})")
    print(f"{'='*60}")

    # Build natural prompt based on benchmark
    if state["benchmark"] in ["humaneval", "mbpp"]:
        prompt = f"""Write a Python function to solve this task:

{state['task']}

Write ONLY the function implementation. Do not include test cases or examples.
Function name should be: {state.get('entry_point', 'solution')}
"""
    else:  # GSM8K
        prompt = f"""Solve this math problem:

{state['task']}

Provide your solution with step-by-step reasoning, then give the final answer as a number.
Format your final answer as: #### [number]
"""

    # Natural generation
    response = deps.llm.generate(prompt, temperature=deps.config["temperature"])

    # Extract uncertainty heuristically
    epistemic, aleatoric = extract_uncertainty_from_response(response)

    confidence = 1.0 - max(epistemic, aleatoric)

    print(f"Generated code (confidence: {confidence:.2f})")
    print(f"Epistemic uncertainty: {epistemic:.2f}")
    print(f"Aleatoric uncertainty: {aleatoric:.2f}")

    return {
        "code": response,
        "generation_confidence": confidence,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "status": "validating",
    }


def decide_validation_strategy_node(state: CodeGenState, deps: GraphDeps) -> dict:
    """SAGE tool calling: decide how thoroughly to test."""

    print(f"\n{'='*60}")
    print(f"DECIDING VALIDATION STRATEGY")
    print(f"{'='*60}")

    # Build decision prompt
    prompt = f"""You are deciding how to validate generated code.

Task: {state['task']}
Generated code confidence: {state.get('generation_confidence', 0.5):.2f}
Current epistemic uncertainty: {state.get('epistemic_uncertainty', 0.5):.2f}

Available strategies:
1. quick_test: Run basic test cases only (fast)
2. comprehensive_test: Run all tests including edge cases (thorough)
3. trust_code: Skip testing entirely (only if very confident)

Which strategy should we use? Consider:
- If confidence is high (>0.9) and uncertainty low (<0.2): trust_code
- If confidence is medium (0.7-0.9): quick_test
- If confidence is low (<0.7) or uncertainty high: comprehensive_test

Choose the most appropriate strategy and explain your reasoning briefly.
Format: {{"strategy": "...", "reasoning": "..."}}
"""

    if deps.config["enable_resampling_decisions"] and state.get("epistemic_uncertainty", 0) > 0.5:
        # Resample decision if uncertain
        print("High uncertainty - resampling validation decision")
        samples = []
        for i in range(deps.config["decision_samples"]):
            response = deps.llm.generate(prompt, temperature=0.8)
            try:
                parsed = json.loads(response)
                samples.append(parsed["strategy"])
            except:
                # Fallback parsing
                if "quick_test" in response:
                    samples.append("quick_test")
                elif "comprehensive_test" in response:
                    samples.append("comprehensive_test")
                elif "trust_code" in response:
                    samples.append("trust_code")
                else:
                    samples.append("quick_test")

        # Majority vote
        counter = Counter(samples)
        chosen_strategy = counter.most_common(1)[0][0]
        agreement = compute_agreement(samples)
        decision_uncertainty = 1.0 - agreement

        print(f"Samples: {samples}")
        print(f"Agreement: {agreement:.2f}")
        print(f"Chosen: {chosen_strategy}")
    else:
        # Single sample
        response = deps.llm.generate(prompt, temperature=0.7)
        try:
            parsed = json.loads(response)
            chosen_strategy = parsed["strategy"]
            decision_uncertainty = 0.3  # Default
        except:
            # Fallback
            if "comprehensive" in response.lower():
                chosen_strategy = "comprehensive_test"
            elif "trust" in response.lower():
                chosen_strategy = "trust_code"
            else:
                chosen_strategy = "quick_test"
            decision_uncertainty = 0.3

        samples = [chosen_strategy]

    print(f"Validation strategy: {chosen_strategy}")
    print(f"Decision uncertainty: {decision_uncertainty:.2f}")

    return {
        "validation_strategy": chosen_strategy,
        "validation_decision_uncertainty": decision_uncertainty,
        "validation_samples": samples,
    }


def execute_tests_node(state: CodeGenState, deps: GraphDeps) -> dict:
    """Execute tests programmatically (NOT tool calling)."""

    print(f"\n{'='*60}")
    print(f"EXECUTING TESTS ({state['validation_strategy']})")
    print(f"{'='*60}")

    if state["validation_strategy"] == "trust_code":
        print("Trusting code without testing")
        return {
            "test_result": {
                "passed": True,
                "trusted": True,
                "errors": [],
                "output": "Skipped testing (trusted)"
            },
            "status": "done",
        }

    # For actual testing, we'll integrate with the evaluation harness
    # For now, return placeholder that will be filled by eval script
    print("Tests will be executed by evaluation harness")

    return {
        "test_result": {
            "passed": None,  # To be filled by eval
            "errors": [],
            "output": "",
            "deferred": True,  # Signal that testing is deferred
        },
        "status": "testing",
    }


def decide_recovery_strategy_node(state: CodeGenState, deps: GraphDeps) -> dict:
    """SAGE tool calling: decide how to fix failures."""

    print(f"\n{'='*60}")
    print(f"DECIDING RECOVERY STRATEGY")
    print(f"{'='*60}")

    errors = state.get("test_result", {}).get("errors", [])
    attempts = state.get("total_attempts", 0)

    # Build decision prompt
    prompt = f"""You are deciding how to recover from test failures.

Task: {state['task']}
Code that failed:
{state['code']}

Test errors:
{errors}

Current attempts: {attempts}
Max attempts: {deps.config['max_attempts']}

Available strategies:
1. regenerate_fresh: Start over with new approach
2. reflect_and_revise: Deep analysis of failures, then revise
3. debug_incrementally: Fix specific errors one by one
4. escalate: Give up (too hard or too many attempts)

Which strategy should we use? Consider:
- If attempts >= max_attempts: escalate
- If errors suggest wrong approach: regenerate_fresh
- If errors are specific bugs: debug_incrementally
- If complex reasoning needed: reflect_and_revise

Choose the most appropriate strategy.
Format: {{"strategy": "...", "reasoning": "..."}}
"""

    if deps.config["enable_resampling_decisions"]:
        # Resample recovery decision
        samples = []
        for i in range(deps.config["decision_samples"]):
            response = deps.llm.generate(prompt, temperature=0.8)
            try:
                parsed = json.loads(response)
                samples.append(parsed["strategy"])
            except:
                # Fallback parsing
                if "regenerate" in response.lower():
                    samples.append("regenerate_fresh")
                elif "reflect" in response.lower():
                    samples.append("reflect_and_revise")
                elif "debug" in response.lower():
                    samples.append("debug_incrementally")
                else:
                    samples.append("escalate")

        counter = Counter(samples)
        chosen_strategy = counter.most_common(1)[0][0]
        agreement = compute_agreement(samples)
        decision_uncertainty = 1.0 - agreement

        print(f"Recovery samples: {samples}")
        print(f"Agreement: {agreement:.2f}")
    else:
        response = deps.llm.generate(prompt, temperature=0.7)
        try:
            parsed = json.loads(response)
            chosen_strategy = parsed["strategy"]
        except:
            chosen_strategy = "regenerate_fresh"
        decision_uncertainty = 0.3

    print(f"Recovery strategy: {chosen_strategy}")
    print(f"Decision uncertainty: {decision_uncertainty:.2f}")

    return {
        "recovery_strategy": chosen_strategy,
        "recovery_decision_uncertainty": decision_uncertainty,
        "status": "recovering",
    }


def execute_recovery_node(state: CodeGenState, deps: GraphDeps) -> dict:
    """Execute chosen recovery strategy."""

    print(f"\n{'='*60}")
    print(f"EXECUTING RECOVERY: {state['recovery_strategy']}")
    print(f"{'='*60}")

    strategy = state["recovery_strategy"]

    if strategy == "escalate":
        print("Escalating - giving up")
        return {
            "status": "failed",
            "final_code": state["code"],
            "warning": f"Failed after {state.get('total_attempts', 0)} attempts",
        }

    elif strategy == "regenerate_fresh":
        print("Regenerating from scratch")
        # Reset and go back to generation
        return {
            "code": "",
            "status": "generating",
            "total_attempts": state.get("total_attempts", 0) + 1,
        }

    elif strategy == "reflect_and_revise":
        print("Reflecting and revising")
        # Build reflexion prompt
        prompt = f"""Your previous solution failed tests. Analyze and revise.

Original task: {state['task']}

Your previous code:
{state['code']}

Test failures:
{state.get('test_result', {}).get('errors', [])}

Reflect on what went wrong, then write a corrected solution.
"""

        revised_code = deps.llm.generate(prompt, temperature=0.7)

        return {
            "code": revised_code,
            "status": "validating",
            "total_attempts": state.get("total_attempts", 0) + 1,
        }

    elif strategy == "debug_incrementally":
        print("Debugging incrementally")
        # Build debug prompt
        errors = state.get("test_result", {}).get("errors", [])
        first_error = errors[0] if errors else "Unknown error"

        prompt = f"""Fix this specific error in the code:

Code:
{state['code']}

Error:
{first_error}

Write the corrected code.
"""

        debugged_code = deps.llm.generate(prompt, temperature=0.5)

        return {
            "code": debugged_code,
            "status": "validating",
            "total_attempts": state.get("total_attempts", 0) + 1,
        }

    else:
        # Fallback
        return {"status": "failed", "warning": f"Unknown recovery strategy: {strategy}"}


def finalize_node(state: CodeGenState, deps: GraphDeps) -> dict:
    """Finalize successful result."""

    print(f"\n{'='*60}")
    print(f"SUCCESS - FINALIZING")
    print(f"{'='*60}")

    return {
        "status": "done",
        "final_code": state["code"],
        "confidence_score": state.get("generation_confidence", 0.5),
    }


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_generation(state: CodeGenState) -> str:
    """Route after code generation."""
    if state.get("generation_attempts", 0) >= 3:
        return "finalize"  # Give up after 3 generation attempts
    return "decide_validation"


def route_after_validation_decision(state: CodeGenState) -> str:
    """Route after validation strategy decision."""
    return "execute_tests"


def route_after_testing(state: CodeGenState) -> str:
    """Route after tests execute."""
    test_result = state.get("test_result", {})

    if test_result.get("deferred"):
        # Tests deferred to eval harness
        return "finalize"

    if test_result.get("passed") or test_result.get("trusted"):
        return "finalize"
    else:
        return "decide_recovery"


def route_after_recovery_decision(state: CodeGenState) -> str:
    """Route after recovery strategy decision."""
    return "execute_recovery"


def route_after_recovery(state: CodeGenState) -> str:
    """Route after recovery execution."""
    if state["status"] == "failed":
        return END
    elif state["status"] == "generating":
        return "generate_code"
    elif state["status"] == "validating":
        return "decide_validation"
    else:
        return "finalize"


def route_after_finalize(state: CodeGenState) -> str:
    """Route after finalization."""
    return END


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph(deps: GraphDeps) -> StateGraph:
    """Build the SAGE v4 graph."""

    graph = StateGraph(CodeGenState)

    # Add nodes
    graph.add_node("generate_code", lambda s: generate_code_node(s, deps))
    graph.add_node("decide_validation", lambda s: decide_validation_strategy_node(s, deps))
    graph.add_node("execute_tests", lambda s: execute_tests_node(s, deps))
    graph.add_node("decide_recovery", lambda s: decide_recovery_strategy_node(s, deps))
    graph.add_node("execute_recovery", lambda s: execute_recovery_node(s, deps))
    graph.add_node("finalize", lambda s: finalize_node(s, deps))

    # Set entry point
    graph.set_entry_point("generate_code")

    # Add edges with routing
    graph.add_conditional_edges(
        "generate_code",
        route_after_generation,
        {
            "decide_validation": "decide_validation",
            "finalize": "finalize",
        }
    )

    graph.add_conditional_edges(
        "decide_validation",
        route_after_validation_decision,
        {
            "execute_tests": "execute_tests",
        }
    )

    graph.add_conditional_edges(
        "execute_tests",
        route_after_testing,
        {
            "finalize": "finalize",
            "decide_recovery": "decide_recovery",
        }
    )

    graph.add_conditional_edges(
        "decide_recovery",
        route_after_recovery_decision,
        {
            "execute_recovery": "execute_recovery",
        }
    )

    graph.add_conditional_edges(
        "execute_recovery",
        route_after_recovery,
        {
            "generate_code": "generate_code",
            "decide_validation": "decide_validation",
            END: END,
        }
    )

    graph.add_conditional_edges(
        "finalize",
        route_after_finalize,
        {
            END: END,
        }
    )

    return graph.compile()


# ============================================================================
# Helper to Create Initial State
# ============================================================================

def create_initial_state(
    task: str,
    benchmark: str = "humaneval",
    test_cases: str = "",
    expected_output: str = "",
    entry_point: str = "solution",
) -> CodeGenState:
    """Create initial state for code generation."""
    return CodeGenState(
        task=task,
        test_cases=test_cases,
        expected_output=expected_output,
        entry_point=entry_point,
        benchmark=benchmark,
        code="",
        generation_confidence=0.0,
        generation_attempts=0,
        validation_strategy="",
        validation_decision_uncertainty=0.0,
        validation_samples=[],
        test_result={},
        recovery_strategy="",
        recovery_decision_uncertainty=0.0,
        total_attempts=0,
        epistemic_uncertainty=0.0,
        aleatoric_uncertainty=0.0,
        status="generating",
        final_code="",
        confidence_score=0.0,
        warning="",
    )
