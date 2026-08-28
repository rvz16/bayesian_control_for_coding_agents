"""SAGE Agent v4 for SWE-bench: Real Tool Calling for Software Engineering Tasks

This is the ideal use case for SAGE + tool calling:
- Agent must SEARCH for relevant files (tool calling)
- Agent must READ files to understand context (tool calling)
- Agent must EDIT code to fix issues (natural generation + tool calling)
- Agent must RUN tests to verify (tool calling)
- Agent must DECIDE which action to take next (SAGE uncertainty)
- Agent can ASK USER for clarification when too uncertain (SAGE question asking)

Architecture:
1. Understand issue (natural language processing)
2. Search for relevant files (SAGE tool calling with uncertainty)
3. Read and analyze files (SAGE tool calling)
4. Generate patch (NATURAL generation - what LLMs do best)
5. Apply edit via tool (SAGE tool calling)
6. Run tests (SAGE tool calling)
7. Iterate or submit

SAGE Features:
- Question Asking: Ask user when uncertainty is too high
- SAUP: Situation Awareness Uncertainty Propagation (Zhao et al., 2024)
  * Propagates uncertainty across multi-step reasoning using weighted RMS
  * Uses situational weights based on semantic distance surrogates
  * Formula: U_agent = sqrt(1/N * Σ((W_i * U_i)^2))
- Reflexion: Reflect on failures before retry
- Resampling: Multiple samples for uncertain decisions

SAGE uncertainty is applied to:
- Which tool to call next
- Tool arguments (which file, what query)
- When to stop searching and start editing
- When to submit vs. retry
- When to ask user for help
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import TypedDict, Literal, Any, Optional
from collections import Counter
from pathlib import Path

from langgraph.graph import StateGraph, END

from sage_agent.core.types import ToolSchema
from sage_agent.core.domains import ParameterDomain
from sage_agent.llm.llm import LLMClient

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # Core thresholds
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_iterations": 15,  # Max tool calls before giving up
    "max_search_iterations": 5,
    "max_edit_attempts": 3,

    # Decision-making (Resampling)
    "enable_resampling": True,
    "decision_samples": 3,
    "agreement_threshold": 0.7,

    # SAUP (Situation Awareness Uncertainty Propagation)
    # Based on: "SAUP: Situation Awareness Uncertainty Propagation on LLM Agent"
    "enable_saup": True,
    "saup_surrogate_type": "distance",  # "position", "distance", or "hybrid"
    "saup_escalation_threshold": 0.85,  # Trajectory uncertainty to trigger escalation
    "max_high_uncertainty_steps": 3,  # Consecutive high-uncertainty steps before escalation

    # Question Asking (Ask User)
    "enable_ask_user": True,
    "ask_user_threshold": 0.8,  # Uncertainty threshold to ask user
    "max_questions_per_task": 3,  # Don't ask too many questions

    # Reflexion
    "enable_reflexion": True,
    "reflexion_on_test_failure": True,
    "reflexion_on_edit_failure": True,
    "max_reflexion_attempts": 2,

    # Search settings
    "max_files_to_read": 10,
    "max_file_lines": 500,

    # Edit settings
    "require_test_pass": True,
    "allow_multiple_edits": True,

    # Generation
    "temperature": 0.7,
    "patch_temperature": 0.5,  # Lower for code edits
}


# ============================================================================
# Tool Definitions (Real tools for SWE-bench!)
# ============================================================================

# Tool descriptions (for prompts)
TOOL_DESCRIPTIONS = {
    "search_codebase": "Search the codebase for files matching a query. Use to find relevant files for the issue.",
    "read_file": "Read the contents of a file. Use to understand code before editing.",
    "edit_file": "Edit a file by replacing old content with new content.",
    "create_file": "Create a new file with given content.",
    "run_tests": "Run tests to verify the fix. Use after making edits.",
    "submit_patch": "Submit the final patch. Use when confident the fix is complete.",
    "think": "Think about the problem without taking action. Use to reason about approach.",
    "ask_user": "Ask the user a clarifying question when you are too uncertain. Use sparingly.",
}

SWE_TOOLS = {
    "search_codebase": ToolSchema(
        name="search_codebase",
        parameters={
            "query": ParameterDomain.continuous(),  # Search query
            "file_pattern": ParameterDomain.continuous(),  # e.g., "*.py", "test_*.py"
        },
        required=frozenset({"query"}),
    ),

    "read_file": ToolSchema(
        name="read_file",
        parameters={
            "file_path": ParameterDomain.continuous(),
            "start_line": ParameterDomain.continuous(),  # Line number (int)
            "end_line": ParameterDomain.continuous(),  # Line number (int)
        },
        required=frozenset({"file_path"}),
    ),

    "edit_file": ToolSchema(
        name="edit_file",
        parameters={
            "file_path": ParameterDomain.continuous(),
            "old_content": ParameterDomain.continuous(),  # Content to replace
            "new_content": ParameterDomain.continuous(),  # Replacement content
        },
        required=frozenset({"file_path", "old_content", "new_content"}),
    ),

    "create_file": ToolSchema(
        name="create_file",
        parameters={
            "file_path": ParameterDomain.continuous(),
            "content": ParameterDomain.continuous(),
        },
        required=frozenset({"file_path", "content"}),
    ),

    "run_tests": ToolSchema(
        name="run_tests",
        parameters={
            "test_command": ParameterDomain.continuous(),  # e.g., "pytest tests/"
            "specific_test": ParameterDomain.continuous(),  # Optional: specific test file/function
        },
        required=frozenset(),
    ),

    "submit_patch": ToolSchema(
        name="submit_patch",
        parameters={
            "explanation": ParameterDomain.continuous(),  # Why this fix works
        },
        required=frozenset(),
    ),

    "think": ToolSchema(
        name="think",
        parameters={
            "thought": ParameterDomain.continuous(),
        },
        required=frozenset({"thought"}),
    ),

    "ask_user": ToolSchema(
        name="ask_user",
        parameters={
            "question": ParameterDomain.continuous(),  # The question to ask
            "options": ParameterDomain.continuous(),  # Optional: suggested answers
            "context": ParameterDomain.continuous(),  # Why you're asking
        },
        required=frozenset({"question"}),
    ),
}


# ============================================================================
# State Definition
# ============================================================================

class SWEState(TypedDict):
    """State for SWE-bench agent."""

    # Input
    instance_id: str
    issue_description: str
    repo_path: str
    base_commit: str
    test_command: str

    # Exploration state
    files_found: list[str]
    files_read: dict[str, str]  # file_path -> content
    search_queries: list[str]

    # Edit state
    edits_made: list[dict]  # List of {file_path, old_content, new_content}
    current_patch: str

    # Test state
    test_results: list[dict]  # List of {passed, output, errors}
    tests_passed: bool

    # Trajectory (for SAGE uncertainty tracking)
    trajectory: list[dict]  # List of {tool, args, result, uncertainty}
    iteration: int

    # SAGE uncertainty
    epistemic_uncertainty: float
    tool_decision_uncertainty: float
    current_tool_samples: list[str]

    # SAUP (Trajectory-level uncertainty)
    saup_trajectory_uncertainty: float  # Accumulated uncertainty across trajectory
    saup_uncertainty_history: list[float]  # History of step uncertainties
    consecutive_high_uncertainty_steps: int  # Count of high uncertainty in a row

    # Question Asking
    questions_asked: list[dict]  # List of {question, answer, context}
    user_responses: list[str]
    questions_count: int

    # Reflexion
    reflexion_memories: list[dict]  # List of {failure, analysis, lesson}
    reflexion_count: int
    last_failure_context: dict  # Context of most recent failure for reflexion

    # Output
    status: Literal["exploring", "editing", "testing", "submitting", "asking", "reflecting", "done", "failed"]
    final_patch: str
    explanation: str
    confidence_score: float


# ============================================================================
# Tool Execution (Actual execution on filesystem)
# ============================================================================

class SWEToolExecutor:
    """Execute SWE tools on actual filesystem."""

    def __init__(self, repo_path: str, sandbox: bool = True):
        self.repo_path = Path(repo_path)
        self.sandbox = sandbox  # If True, don't actually write files

    def execute(self, tool_name: str, args: dict, user_input_callback=None) -> dict:
        """Execute a tool and return result."""
        if tool_name == "search_codebase":
            return self._search_codebase(args)
        elif tool_name == "read_file":
            return self._read_file(args)
        elif tool_name == "edit_file":
            return self._edit_file(args)
        elif tool_name == "create_file":
            return self._create_file(args)
        elif tool_name == "run_tests":
            return self._run_tests(args)
        elif tool_name == "submit_patch":
            return self._submit_patch(args)
        elif tool_name == "think":
            return {"success": True, "thought": args.get("thought", "")}
        elif tool_name == "ask_user":
            return self._ask_user(args, user_input_callback)
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    def _ask_user(self, args: dict, callback=None) -> dict:
        """Ask user a clarifying question."""
        question = args.get("question", "")
        options = args.get("options", "")
        context = args.get("context", "")

        print(f"\n{'='*60}")
        print("AGENT QUESTION FOR USER")
        print(f"{'='*60}")
        print(f"Question: {question}")
        if options:
            print(f"Options: {options}")
        if context:
            print(f"Context: {context}")
        print(f"{'='*60}")

        # If callback provided, use it; otherwise simulate or wait
        if callback:
            answer = callback(question, options, context)
        else:
            # In automated mode, return a default or simulate
            answer = "[No user response - automated mode]"

        return {
            "success": True,
            "question": question,
            "answer": answer,
            "context": context,
        }

    def _search_codebase(self, args: dict) -> dict:
        """Search for files matching query."""
        query = args.get("query", "")
        file_pattern = args.get("file_pattern", "*.py")

        try:
            # Use grep to search
            cmd = f"grep -r -l '{query}' --include='{file_pattern}' ."
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            return {
                "success": True,
                "files": files[:20],  # Limit results
                "count": len(files),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "files": []}

    def _read_file(self, args: dict) -> dict:
        """Read file contents."""
        file_path = args.get("file_path", "")
        start_line = args.get("start_line", 1)
        end_line = args.get("end_line", 500)

        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            with open(full_path, "r") as f:
                lines = f.readlines()

            # Extract requested range
            content = "".join(lines[start_line - 1:end_line])

            return {
                "success": True,
                "content": content,
                "total_lines": len(lines),
                "lines_read": f"{start_line}-{min(end_line, len(lines))}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _edit_file(self, args: dict) -> dict:
        """Edit file by replacing content."""
        file_path = args.get("file_path", "")
        old_content = args.get("old_content", "")
        new_content = args.get("new_content", "")

        try:
            full_path = self.repo_path / file_path
            if not full_path.exists():
                return {"success": False, "error": f"File not found: {file_path}"}

            with open(full_path, "r") as f:
                content = f.read()

            if old_content not in content:
                return {
                    "success": False,
                    "error": "Old content not found in file. Check exact match.",
                }

            new_file_content = content.replace(old_content, new_content, 1)

            if not self.sandbox:
                with open(full_path, "w") as f:
                    f.write(new_file_content)

            return {
                "success": True,
                "message": f"Edited {file_path}",
                "sandbox": self.sandbox,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_file(self, args: dict) -> dict:
        """Create new file."""
        file_path = args.get("file_path", "")
        content = args.get("content", "")

        try:
            full_path = self.repo_path / file_path
            if full_path.exists():
                return {"success": False, "error": f"File already exists: {file_path}"}

            if not self.sandbox:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, "w") as f:
                    f.write(content)

            return {
                "success": True,
                "message": f"Created {file_path}",
                "sandbox": self.sandbox,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _run_tests(self, args: dict) -> dict:
        """Run tests."""
        test_command = args.get("test_command", "pytest")
        specific_test = args.get("specific_test", "")

        if specific_test:
            test_command = f"{test_command} {specific_test}"

        try:
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=120,
            )

            passed = result.returncode == 0

            return {
                "success": True,
                "passed": passed,
                "stdout": result.stdout[-2000:],  # Last 2000 chars
                "stderr": result.stderr[-1000:],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Test timeout", "passed": False}
        except Exception as e:
            return {"success": False, "error": str(e), "passed": False}

    def _submit_patch(self, args: dict) -> dict:
        """Submit final patch."""
        explanation = args.get("explanation", "")

        # Generate diff
        try:
            result = subprocess.run(
                "git diff",
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            patch = result.stdout

            return {
                "success": True,
                "patch": patch,
                "explanation": explanation,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ============================================================================
# Graph Nodes
# ============================================================================

@dataclass
class GraphDeps:
    """Dependencies for graph nodes."""
    llm: LLMClient
    tool_executor: SWEToolExecutor
    config: dict = field(default_factory=lambda: DEFAULT_CONFIG.copy())


def decide_next_action_node(state: SWEState, deps: GraphDeps) -> dict:
    """SAGE tool calling: decide which tool to use next."""

    print(f"\n{'='*60}")
    print(f"DECIDING NEXT ACTION (Iteration {state.get('iteration', 0) + 1})")
    print(f"{'='*60}")

    # Check iteration limit
    if state.get("iteration", 0) >= deps.config["max_iterations"]:
        print("Max iterations reached, forcing submit")
        return {
            "status": "submitting",
            "tool_decision_uncertainty": 0.0,
        }

    # Build context for decision
    context = build_decision_context(state)

    # Build decision prompt
    prompt = f"""You are solving a GitHub issue. Decide the next action.

{context}

Available tools:
1. search_codebase(query, file_pattern) - Find relevant files
2. read_file(file_path, start_line, end_line) - Read file contents
3. edit_file(file_path, old_content, new_content) - Edit a file
4. create_file(file_path, content) - Create new file
5. run_tests(test_command, specific_test) - Run tests
6. submit_patch(explanation) - Submit final solution
7. think(thought) - Reason about the problem

Current phase: {state.get('status', 'exploring')}

Guidelines:
- If you haven't found relevant files yet, use search_codebase
- If you found files but haven't read them, use read_file
- If you understand the issue and know what to change, use edit_file
- After editing, use run_tests to verify
- If tests pass and you're confident, use submit_patch
- Use think if you need to reason about complex issues

What should be the next action? Respond with JSON:
{{"tool": "tool_name", "args": {{"arg1": "value1", ...}}, "reasoning": "why this action"}}
"""

    # Initialize samples list
    samples = []

    # Resample if enabled and uncertain
    if deps.config["enable_resampling"] and state.get("epistemic_uncertainty", 0) > 0.5:
        for i in range(deps.config["decision_samples"]):
            response = deps.llm.generate(prompt, temperature=0.8)
            parsed = parse_tool_decision(response)
            if parsed:
                samples.append(parsed["tool"])

        # Majority vote
        if samples:
            counter = Counter(samples)
            chosen_tool = counter.most_common(1)[0][0]
            agreement = counter.most_common(1)[0][1] / len(samples)
            decision_uncertainty = 1.0 - agreement

            # Get full args from best matching sample
            for i in range(deps.config["decision_samples"]):
                response = deps.llm.generate(prompt, temperature=0.7)
                parsed = parse_tool_decision(response)
                if parsed and parsed["tool"] == chosen_tool:
                    tool_args = parsed.get("args", {})
                    break
            else:
                tool_args = {}

            print(f"Resampled: {samples}")
            print(f"Chosen: {chosen_tool} (agreement: {agreement:.2f})")
        else:
            chosen_tool = "think"
            tool_args = {"thought": "Need to analyze the situation"}
            decision_uncertainty = 0.9
    else:
        # Single sample
        response = deps.llm.generate(prompt, temperature=0.7)
        parsed = parse_tool_decision(response)

        if parsed:
            chosen_tool = parsed["tool"]
            tool_args = parsed.get("args", {})
            decision_uncertainty = 0.3
        else:
            chosen_tool = "think"
            tool_args = {"thought": "Parsing failed, need to reconsider"}
            decision_uncertainty = 0.7

        # Track single sample
        samples = [chosen_tool]

    print(f"Next action: {chosen_tool}")
    print(f"Args: {json.dumps(tool_args, indent=2)[:200]}")
    print(f"Decision uncertainty: {decision_uncertainty:.2f}")

    # Determine new status based on tool
    new_status = state.get("status", "exploring")
    if chosen_tool == "edit_file":
        new_status = "editing"
    elif chosen_tool == "run_tests":
        new_status = "testing"
    elif chosen_tool == "submit_patch":
        new_status = "submitting"

    return {
        "chosen_tool": chosen_tool,
        "chosen_args": tool_args,
        "tool_decision_uncertainty": decision_uncertainty,
        "current_tool_samples": samples if deps.config["enable_resampling"] else [chosen_tool],
        "status": new_status,
    }


def execute_tool_node(state: SWEState, deps: GraphDeps) -> dict:
    """Execute the chosen tool."""

    print(f"\n{'='*60}")
    print(f"EXECUTING TOOL: {state.get('chosen_tool', 'unknown')}")
    print(f"{'='*60}")

    tool_name = state.get("chosen_tool", "think")
    tool_args = state.get("chosen_args", {})

    # Execute tool
    result = deps.tool_executor.execute(tool_name, tool_args)

    print(f"Result: {json.dumps(result, indent=2)[:500]}")

    # Update state based on tool
    updates = {
        "iteration": state.get("iteration", 0) + 1,
        "trajectory": state.get("trajectory", []) + [{
            "tool": tool_name,
            "args": tool_args,
            "result": result,
            "uncertainty": state.get("tool_decision_uncertainty", 0),
        }],
    }

    # Compute SAUP after each action
    if deps.config.get("enable_saup", True):
        saup_updates = compute_saup_uncertainty(state, deps)
        updates.update(saup_updates)

    if tool_name == "search_codebase" and result.get("success"):
        files_found = state.get("files_found", []) + result.get("files", [])
        updates["files_found"] = list(set(files_found))  # Dedupe
        updates["search_queries"] = state.get("search_queries", []) + [tool_args.get("query", "")]

    elif tool_name == "read_file" and result.get("success"):
        files_read = state.get("files_read", {}).copy()
        files_read[tool_args.get("file_path", "")] = result.get("content", "")
        updates["files_read"] = files_read

    elif tool_name == "edit_file":
        if result.get("success"):
            edits_made = state.get("edits_made", []) + [{
                "file_path": tool_args.get("file_path"),
                "old_content": tool_args.get("old_content"),
                "new_content": tool_args.get("new_content"),
            }]
            updates["edits_made"] = edits_made
        else:
            # Edit failed - track for reflexion
            updates["epistemic_uncertainty"] = min(
                state.get("epistemic_uncertainty", 0) + 0.2, 1.0
            )
            updates["last_failure_context"] = {
                "type": "edit_failure",
                "details": result.get("error", "Unknown error"),
                "file_path": tool_args.get("file_path", ""),
                "attempted_edit": tool_args,
            }

    elif tool_name == "run_tests":
        test_results = state.get("test_results", []) + [result]
        updates["test_results"] = test_results
        updates["tests_passed"] = result.get("passed", False)

        # Track test failure for reflexion
        if not result.get("passed", False):
            updates["last_failure_context"] = {
                "type": "test_failure",
                "details": result.get("stderr", "") or result.get("stdout", ""),
                "test_output": result,
            }

    elif tool_name == "submit_patch":
        if result.get("success"):
            updates["final_patch"] = result.get("patch", "")
            updates["explanation"] = tool_args.get("explanation", "")
            updates["status"] = "done"

    elif tool_name == "ask_user":
        # Store question and response
        questions = state.get("questions_asked", []) + [{
            "question": tool_args.get("question", ""),
            "answer": result.get("answer", ""),
            "context": tool_args.get("context", ""),
        }]
        updates["questions_asked"] = questions
        user_responses = state.get("user_responses", []) + [result.get("answer", "")]
        updates["user_responses"] = user_responses
        updates["status"] = "exploring"  # Go back to exploring with new info

        # Reduce uncertainty after getting user input
        updates["epistemic_uncertainty"] = max(
            state.get("epistemic_uncertainty", 0.5) - 0.2, 0.1
        )

    return updates


def generate_patch_node(state: SWEState, deps: GraphDeps) -> dict:
    """Generate code patch using NATURAL generation (not tool calling)."""

    print(f"\n{'='*60}")
    print(f"GENERATING PATCH (Natural Generation)")
    print(f"{'='*60}")

    # Build context from files read
    file_context = ""
    for file_path, content in state.get("files_read", {}).items():
        file_context += f"\n### {file_path}\n```\n{content[:1000]}\n```\n"

    prompt = f"""Generate a code fix for this issue.

## Issue Description
{state.get('issue_description', '')}

## Relevant Files
{file_context}

## Previous Edits Made
{json.dumps(state.get('edits_made', []), indent=2)}

## Test Results
{json.dumps(state.get('test_results', [])[-1] if state.get('test_results') else {}, indent=2)}

Generate the minimal code change needed to fix the issue.
Respond with:
1. The file to edit
2. The exact old code to replace (must match exactly!)
3. The new code

Format:
FILE: <path>
OLD:
```
<exact old code>
```
NEW:
```
<new code>
```
"""

    # Natural generation with lower temperature for code
    response = deps.llm.generate(prompt, temperature=deps.config["patch_temperature"])

    # Parse response
    patch_info = parse_patch_response(response)

    if patch_info:
        print(f"Generated patch for: {patch_info['file_path']}")
        return {
            "chosen_tool": "edit_file",
            "chosen_args": patch_info,
        }
    else:
        print("Failed to parse patch, will think about it")
        return {
            "chosen_tool": "think",
            "chosen_args": {"thought": "Need to reconsider the patch approach"},
            "epistemic_uncertainty": min(state.get("epistemic_uncertainty", 0) + 0.1, 1.0),
        }


def finalize_node(state: SWEState, deps: GraphDeps) -> dict:
    """Finalize and return result."""

    print(f"\n{'='*60}")
    print(f"FINALIZING - Status: {state.get('status', 'unknown')}")
    print(f"{'='*60}")

    if state.get("status") == "done":
        confidence = 1.0 - state.get("epistemic_uncertainty", 0.5)
        if state.get("tests_passed"):
            confidence = min(confidence + 0.2, 1.0)

        return {
            "status": "done",
            "confidence_score": confidence,
        }
    else:
        return {
            "status": "failed",
            "confidence_score": 0.0,
        }


# ============================================================================
# SAUP: Situation Awareness Uncertainty Propagation
# Based on: "SAUP: Situation Awareness Uncertainty Propagation on LLM Agent"
# (Zhao et al., 2024) - arXiv:2412.01033
#
# Key insight: Uncertainty accumulates across multi-step reasoning.
# SAUP propagates and aggregates uncertainty using:
# 1. Per-step uncertainty (U_i) - from token probabilities or resampling
# 2. Situational weights (W_i) - based on semantic distance metrics
# 3. Weighted RMS aggregation: U_agent = sqrt(1/N * Σ((W_i * U_i)^2))
# ============================================================================

import math
from typing import List, Tuple


def compute_semantic_distance(text_a: str, text_b: str) -> float:
    """Compute semantic distance between two texts.

    In the paper, this uses RoBERTa fine-tuned on SQuAD v2.
    Here we use a simpler approximation based on:
    - Token overlap (Jaccard similarity)
    - Length ratio
    - Keyword matching

    Returns distance in [0, 1] where 0 = identical, 1 = completely different.
    """
    if not text_a or not text_b:
        return 0.5  # Unknown distance

    # Normalize texts
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())

    if not tokens_a or not tokens_b:
        return 0.5

    # Jaccard similarity
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    jaccard = intersection / union if union > 0 else 0

    # Convert similarity to distance
    distance = 1.0 - jaccard

    return min(max(distance, 0.0), 1.0)


def compute_saup_situational_weight(
    question: str,
    thinking_action: str,
    observation: str,
    step_position: int,
    total_steps: int,
    surrogate_type: str = "distance"
) -> float:
    """Compute situational weight for a step using SAUP surrogates.

    From the paper, three surrogate methods:
    1. SAUP-P (Position): Weight based on step position (closer to end = higher weight)
    2. SAUP-D (Distance): Weight based on semantic distances D_a and D_o
    3. SAUP-PD (Hybrid): Combines both

    Args:
        question: The original question/issue description
        thinking_action: The agent's thinking + action at this step
        observation: The observation/result from the environment
        step_position: Current step number (1-indexed)
        total_steps: Total steps so far
        surrogate_type: "position", "distance", or "hybrid"

    Returns:
        Situational weight W_i in [0, 1]
    """
    if surrogate_type == "position":
        # SAUP-P: Later steps get higher weight (closer to final answer)
        # Normalized so weights sum to reasonable values
        if total_steps == 0:
            return 1.0
        position_weight = step_position / total_steps
        return position_weight

    elif surrogate_type == "distance":
        # SAUP-D: Based on semantic distances
        # D_a = distance(question, thinking+action+observation)
        # D_o = distance(thinking+action, observation)
        combined_context = f"{thinking_action} {observation}"
        d_a = compute_semantic_distance(question, combined_context)
        d_o = compute_semantic_distance(thinking_action, observation)

        # Weight = D_a + D_o (higher distance = higher uncertainty contribution)
        # But we want W in [0, 1], so normalize
        weight = (d_a + d_o) / 2.0
        return weight

    elif surrogate_type == "hybrid":
        # SAUP-PD: Combine position and distance
        position_weight = step_position / max(total_steps, 1)

        combined_context = f"{thinking_action} {observation}"
        d_a = compute_semantic_distance(question, combined_context)
        d_o = compute_semantic_distance(thinking_action, observation)
        distance_weight = (d_a + d_o) / 2.0

        # Combine with equal weighting
        weight = 0.5 * position_weight + 0.5 * distance_weight
        return weight

    else:
        # Default: uniform weight
        return 1.0


def compute_saup_uncertainty(state: SWEState, deps: GraphDeps) -> dict:
    """Compute trajectory-level uncertainty using TRUE SAUP propagation.

    From "SAUP: Situation Awareness Uncertainty Propagation on LLM Agent":

    The overall agent uncertainty is computed as weighted RMS:
        U_agent = sqrt(1/N * Σ((W_i * U_i)^2))

    Where:
    - U_i = per-step uncertainty (from token probs or resampling disagreement)
    - W_i = situational weight (from semantic distance surrogates)
    - N = number of steps

    This captures how uncertainty accumulates across the reasoning trajectory,
    with situational awareness weighting based on how well the agent stays
    on the correct reasoning path.
    """
    config = deps.config
    current_tool = state.get("chosen_tool", "think")
    current_args = state.get("chosen_args", {})
    current_base_uncertainty = state.get("tool_decision_uncertainty", 0.5)
    trajectory = state.get("trajectory", [])
    history = state.get("saup_uncertainty_history", [])
    question = state.get("issue_description", "")

    # Get surrogate type from config (default to "distance" as in paper)
    surrogate_type = config.get("saup_surrogate_type", "distance")

    # Build thinking/action string for current step
    thinking_action = f"{current_tool}: {json.dumps(current_args)[:200]}"

    # Get observation from last trajectory step (if any)
    if trajectory:
        last_result = trajectory[-1].get("result", {})
        observation = json.dumps(last_result)[:500] if last_result else ""
    else:
        observation = ""

    # Compute situational weight for current step
    total_steps = len(trajectory) + 1
    current_weight = compute_saup_situational_weight(
        question=question,
        thinking_action=thinking_action,
        observation=observation,
        step_position=total_steps,
        total_steps=total_steps,
        surrogate_type=surrogate_type
    )

    # Add current step to history
    new_history_entry = {
        "tool": current_tool,
        "uncertainty": current_base_uncertainty,
        "situational_weight": current_weight,
        "step": total_steps,
    }
    new_history = history + [new_history_entry]

    # Compute weighted RMS over all steps (Equation 1 from paper)
    # U_agent = sqrt(1/N * Σ((W_i * U_i)^2))
    if new_history:
        N = len(new_history)
        sum_weighted_sq = sum(
            (entry["situational_weight"] * entry["uncertainty"]) ** 2
            for entry in new_history
        )
        trajectory_uncertainty = math.sqrt(sum_weighted_sq / N)
    else:
        trajectory_uncertainty = current_base_uncertainty

    # Ensure in valid range
    trajectory_uncertainty = min(max(trajectory_uncertainty, 0.0), 1.0)

    # Track consecutive high-uncertainty steps (for escalation logic)
    threshold = config.get("uncertainty_threshold", 0.3)
    consecutive = state.get("consecutive_high_uncertainty_steps", 0)
    weighted_current = current_weight * current_base_uncertainty

    if weighted_current > threshold:
        consecutive += 1
    else:
        consecutive = max(0, consecutive - 1)

    print(f"SAUP (Situation Awareness Uncertainty Propagation):")
    print(f"  Step: {total_steps}, Tool: {current_tool}")
    print(f"  Base uncertainty (U_i): {current_base_uncertainty:.3f}")
    print(f"  Situational weight (W_i): {current_weight:.3f}")
    print(f"  Weighted uncertainty (W_i * U_i): {weighted_current:.3f}")
    print(f"  Trajectory uncertainty (RMS): {trajectory_uncertainty:.3f}")
    print(f"  Surrogate type: {surrogate_type}")
    print(f"  Consecutive high-unc steps: {consecutive}")

    return {
        "saup_trajectory_uncertainty": trajectory_uncertainty,
        "saup_uncertainty_history": new_history,
        "consecutive_high_uncertainty_steps": consecutive,
        "epistemic_uncertainty": trajectory_uncertainty,  # Use trajectory-level as main
    }


def should_escalate_saup(state: SWEState, deps: GraphDeps) -> bool:
    """Check if SAUP indicates we should escalate (ask user or give up)."""
    config = deps.config

    if not config.get("enable_saup", True):
        return False

    trajectory_unc = state.get("saup_trajectory_uncertainty", 0)
    consecutive = state.get("consecutive_high_uncertainty_steps", 0)

    # Escalate if trajectory uncertainty too high
    if trajectory_unc > config.get("saup_escalation_threshold", 0.85):
        print(f"SAUP escalation: trajectory uncertainty {trajectory_unc:.2f} > threshold")
        return True

    # Escalate if too many consecutive high-uncertainty steps
    if consecutive >= config.get("max_high_uncertainty_steps", 3):
        print(f"SAUP escalation: {consecutive} consecutive high-uncertainty steps")
        return True

    return False


# ============================================================================
# Question Asking (Ask User)
# ============================================================================

def ask_user_node(state: SWEState, deps: GraphDeps) -> dict:
    """Generate and handle a question to the user."""

    print(f"\n{'='*60}")
    print(f"ASKING USER (Question {state.get('questions_count', 0) + 1})")
    print(f"{'='*60}")

    # Build question based on current uncertainty
    context = build_decision_context(state)
    trajectory_unc = state.get("saup_trajectory_uncertainty", 0.5)

    prompt = f"""You are solving a GitHub issue but are uncertain about the next step.
Current uncertainty level: {trajectory_unc:.2f}

{context}

Generate a clarifying question for the user. The question should:
1. Be specific and actionable
2. Help reduce your uncertainty about the approach
3. Not ask for the solution directly

Format:
{{"question": "Your question", "options": ["option1", "option2"], "context": "Why you're asking"}}
"""

    response = deps.llm.generate(prompt, temperature=0.5)

    try:
        parsed = json.loads(response)
        question = parsed.get("question", "What should I focus on?")
        options = parsed.get("options", [])
        question_context = parsed.get("context", "")
    except:
        question = "I'm uncertain about the approach. What aspect should I focus on?"
        options = ["Find more files", "Try a different approach", "Proceed with current understanding"]
        question_context = "High uncertainty in decision-making"

    # Record question (answer will be filled by execution)
    return {
        "chosen_tool": "ask_user",
        "chosen_args": {
            "question": question,
            "options": ", ".join(options) if options else "",
            "context": question_context,
        },
        "status": "asking",
        "questions_count": state.get("questions_count", 0) + 1,
    }


def should_ask_user(state: SWEState, deps: GraphDeps) -> bool:
    """Check if we should ask the user for help."""
    config = deps.config

    if not config.get("enable_ask_user", True):
        return False

    # Don't ask too many questions
    if state.get("questions_count", 0) >= config.get("max_questions_per_task", 3):
        return False

    # Check SAUP-triggered escalation
    if should_escalate_saup(state, deps):
        return True

    # Check decision uncertainty
    decision_unc = state.get("tool_decision_uncertainty", 0)
    if decision_unc > config.get("ask_user_threshold", 0.8):
        print(f"Decision uncertainty {decision_unc:.2f} > threshold, asking user")
        return True

    return False


# ============================================================================
# Reflexion: Learn from Failures
# ============================================================================

def reflexion_node(state: SWEState, deps: GraphDeps) -> dict:
    """Reflect on a failure and learn from it."""

    print(f"\n{'='*60}")
    print(f"REFLEXION (Attempt {state.get('reflexion_count', 0) + 1})")
    print(f"{'='*60}")

    failure_context = state.get("last_failure_context", {})
    failure_type = failure_context.get("type", "unknown")
    failure_details = failure_context.get("details", "")

    # Build reflexion prompt
    prompt = f"""Reflect on this failure and learn from it.

## Issue
{state.get('issue_description', '')[:500]}

## What Failed
Type: {failure_type}
Details: {failure_details}

## Previous Actions
{json.dumps(state.get('trajectory', [])[-5:], indent=2, default=str)[:1000]}

## Previous Reflexions
{json.dumps(state.get('reflexion_memories', []), indent=2)[:500]}

Analyze what went wrong and what you should do differently.
Format:
{{
    "failure_analysis": "What specifically went wrong",
    "root_cause": "Why it went wrong",
    "lesson_learned": "What to do differently",
    "next_action": "Concrete next step"
}}
"""

    response = deps.llm.generate(prompt, temperature=0.5)

    try:
        parsed = json.loads(response)
    except:
        parsed = {
            "failure_analysis": "Failed to parse reflexion",
            "root_cause": "Unknown",
            "lesson_learned": "Try a different approach",
            "next_action": "search for more context",
        }

    print(f"Failure analysis: {parsed.get('failure_analysis', '')[:200]}")
    print(f"Lesson learned: {parsed.get('lesson_learned', '')[:200]}")
    print(f"Next action: {parsed.get('next_action', '')[:200]}")

    # Store reflexion memory
    new_memory = {
        "failure_type": failure_type,
        "analysis": parsed.get("failure_analysis", ""),
        "lesson": parsed.get("lesson_learned", ""),
        "iteration": state.get("iteration", 0),
    }

    memories = state.get("reflexion_memories", []) + [new_memory]

    # Reduce uncertainty after reflexion (we learned something)
    new_uncertainty = max(state.get("epistemic_uncertainty", 0.5) - 0.1, 0.2)

    return {
        "reflexion_memories": memories,
        "reflexion_count": state.get("reflexion_count", 0) + 1,
        "epistemic_uncertainty": new_uncertainty,
        "status": "exploring",  # Go back to exploring with new knowledge
        "last_failure_context": {},  # Clear failure context
    }


def should_reflect(state: SWEState, deps: GraphDeps) -> bool:
    """Check if we should trigger reflexion."""
    config = deps.config

    if not config.get("enable_reflexion", True):
        return False

    # Check if we have a failure context
    failure_context = state.get("last_failure_context", {})
    if not failure_context:
        return False

    # Don't reflect too many times
    if state.get("reflexion_count", 0) >= config.get("max_reflexion_attempts", 2):
        return False

    failure_type = failure_context.get("type", "")

    # Reflect on test failures
    if failure_type == "test_failure" and config.get("reflexion_on_test_failure", True):
        return True

    # Reflect on edit failures
    if failure_type == "edit_failure" and config.get("reflexion_on_edit_failure", True):
        return True

    return False


# ============================================================================
# Helper Functions
# ============================================================================

def build_decision_context(state: SWEState) -> str:
    """Build context string for decision-making."""
    parts = [
        f"## Issue\n{state.get('issue_description', 'No description')[:500]}",
    ]

    if state.get("files_found"):
        parts.append(f"\n## Files Found ({len(state['files_found'])})\n" +
                     "\n".join(f"- {f}" for f in state["files_found"][:10]))

    if state.get("files_read"):
        parts.append(f"\n## Files Read ({len(state['files_read'])})\n" +
                     "\n".join(f"- {f}" for f in state["files_read"].keys()))

    if state.get("edits_made"):
        parts.append(f"\n## Edits Made ({len(state['edits_made'])})\n" +
                     "\n".join(f"- {e['file_path']}" for e in state["edits_made"]))

    if state.get("test_results"):
        last_test = state["test_results"][-1]
        parts.append(f"\n## Last Test Result\nPassed: {last_test.get('passed', 'N/A')}")

    return "\n".join(parts)


def parse_tool_decision(response: str) -> Optional[dict]:
    """Parse tool decision from LLM response."""
    try:
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # Fallback: look for tool name
        for tool in SWE_TOOLS.keys():
            if tool in response.lower():
                return {"tool": tool, "args": {}}

        return None
    except:
        return None


def parse_patch_response(response: str) -> Optional[dict]:
    """Parse patch from natural generation response."""
    try:
        # Look for FILE:
        file_match = re.search(r'FILE:\s*(.+)', response)
        if not file_match:
            return None
        file_path = file_match.group(1).strip()

        # Look for OLD: ... ``` ... ```
        old_match = re.search(r'OLD:\s*```[^\n]*\n(.*?)```', response, re.DOTALL)
        if not old_match:
            return None
        old_content = old_match.group(1)

        # Look for NEW: ... ``` ... ```
        new_match = re.search(r'NEW:\s*```[^\n]*\n(.*?)```', response, re.DOTALL)
        if not new_match:
            return None
        new_content = new_match.group(1)

        return {
            "file_path": file_path,
            "old_content": old_content,
            "new_content": new_content,
        }
    except:
        return None


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_decision(state: SWEState, deps: GraphDeps) -> str:
    """Route after deciding next action."""
    tool = state.get("chosen_tool", "think")

    # Check if we should ask user first (SAGE question asking)
    if should_ask_user(state, deps):
        return "ask_user"

    if state.get("status") == "submitting":
        return "execute_tool"  # Execute submit
    elif tool == "edit_file" and not state.get("chosen_args", {}).get("old_content"):
        # Need to generate patch content
        return "generate_patch"
    else:
        return "execute_tool"


def route_after_execution(state: SWEState, deps: GraphDeps) -> str:
    """Route after executing tool."""
    if state.get("status") == "done":
        return "finalize"
    elif state.get("status") == "failed":
        return "finalize"
    elif state.get("iteration", 0) >= 15:
        return "finalize"

    # Check if we should reflect on failure (Reflexion)
    if should_reflect(state, deps):
        return "reflexion"

    # Check if SAUP indicates we should ask user
    if should_ask_user(state, deps):
        return "ask_user"

    return "decide_action"


def route_after_patch(state: SWEState) -> str:
    """Route after generating patch."""
    return "execute_tool"


def route_after_ask_user(state: SWEState) -> str:
    """Route after asking user."""
    return "execute_tool"  # Execute the ask_user tool


def route_after_reflexion(state: SWEState) -> str:
    """Route after reflexion."""
    return "decide_action"  # Go back to deciding with new knowledge


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph(deps: GraphDeps) -> StateGraph:
    """Build the SAGE v4 SWE-bench graph with full SAGE features.

    Nodes:
    - decide_action: Decide next tool (with SAGE resampling)
    - execute_tool: Execute chosen tool
    - generate_patch: Natural code generation for patches
    - ask_user: Ask user for clarification (SAGE question asking)
    - reflexion: Reflect on failures (Reflexion)
    - finalize: Complete the task

    SAGE Features:
    - Resampling: Multiple samples for uncertain decisions
    - SAUP: Trajectory-level uncertainty tracking
    - Question Asking: Ask user when too uncertain
    - Reflexion: Learn from failures
    """

    graph = StateGraph(SWEState)

    # Add nodes
    graph.add_node("decide_action", lambda s: decide_next_action_node(s, deps))
    graph.add_node("execute_tool", lambda s: execute_tool_node(s, deps))
    graph.add_node("generate_patch", lambda s: generate_patch_node(s, deps))
    graph.add_node("ask_user", lambda s: ask_user_node(s, deps))
    graph.add_node("reflexion", lambda s: reflexion_node(s, deps))
    graph.add_node("finalize", lambda s: finalize_node(s, deps))

    # Set entry point
    graph.set_entry_point("decide_action")

    # Add edges with routing that considers SAGE features
    graph.add_conditional_edges(
        "decide_action",
        lambda s: route_after_decision(s, deps),
        {
            "execute_tool": "execute_tool",
            "generate_patch": "generate_patch",
            "ask_user": "ask_user",
        }
    )

    graph.add_conditional_edges(
        "execute_tool",
        lambda s: route_after_execution(s, deps),
        {
            "decide_action": "decide_action",
            "finalize": "finalize",
            "reflexion": "reflexion",
            "ask_user": "ask_user",
        }
    )

    graph.add_conditional_edges(
        "generate_patch",
        route_after_patch,
        {
            "execute_tool": "execute_tool",
        }
    )

    graph.add_conditional_edges(
        "ask_user",
        route_after_ask_user,
        {
            "execute_tool": "execute_tool",
        }
    )

    graph.add_conditional_edges(
        "reflexion",
        route_after_reflexion,
        {
            "decide_action": "decide_action",
        }
    )

    graph.add_edge("finalize", END)

    return graph.compile()


# ============================================================================
# Create Initial State
# ============================================================================

def create_initial_state(
    instance_id: str,
    issue_description: str,
    repo_path: str,
    base_commit: str = "",
    test_command: str = "pytest",
) -> SWEState:
    """Create initial state for SWE-bench task."""
    return SWEState(
        # Input
        instance_id=instance_id,
        issue_description=issue_description,
        repo_path=repo_path,
        base_commit=base_commit,
        test_command=test_command,

        # Exploration state
        files_found=[],
        files_read={},
        search_queries=[],

        # Edit state
        edits_made=[],
        current_patch="",

        # Test state
        test_results=[],
        tests_passed=False,

        # Trajectory
        trajectory=[],
        iteration=0,

        # SAGE uncertainty
        epistemic_uncertainty=0.5,
        tool_decision_uncertainty=0.5,
        current_tool_samples=[],

        # SAUP (Trajectory-level uncertainty)
        saup_trajectory_uncertainty=0.0,
        saup_uncertainty_history=[],
        consecutive_high_uncertainty_steps=0,

        # Question Asking
        questions_asked=[],
        user_responses=[],
        questions_count=0,

        # Reflexion
        reflexion_memories=[],
        reflexion_count=0,
        last_failure_context={},

        # Output
        status="exploring",
        final_patch="",
        explanation="",
        confidence_score=0.0,
    )
