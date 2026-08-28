"""Canonical LangGraph Implementation of SAGE-Agent Algorithm 1.

This module provides a LangGraph-based implementation of the SAGE-Agent
decision loop from Section 5 of the paper:
"Structured Uncertainty Guided Clarification for LLM Agents"

Graph Structure:
━━━━━━━━━━━━━━━━

    ┌─────────────────────────────────────────────────────────┐
    │                                                         │
    │  ┌──────────┐     ┌──────────────────┐                  │
    │  │  START   │────►│ generate_cands   │◄─────────────┐   │
    │  └──────────┘     └────────┬─────────┘              │   │
    │                            │                        │   │
    │                            ▼                        │   │
    │                   ┌──────────────────┐              │   │
    │                   │ compute_probs    │              │   │
    │                   └────────┬─────────┘              │   │
    │                            │                        │   │
    │              ┌─────────────┼─────────────┐          │   │
    │              │             │             │          │   │
    │              ▼             ▼             ▼          │   │
    │         [confident]   [uncertain]   [escalate]      │   │
    │              │             │             │          │   │
    │              │             ▼             │          │   │
    │              │    ┌──────────────────┐   │          │   │
    │              │    │ generate_qs      │   │          │   │
    │              │    └────────┬─────────┘   │          │   │
    │              │             │             │          │   │
    │              │             ▼             │          │   │
    │              │    ┌──────────────────┐   │          │   │
    │              │    │ select_question  │   │          │   │
    │              │    └────────┬─────────┘   │          │   │
    │              │             │             │          │   │
    │              │    ┌────────┴────────┐    │          │   │
    │              │    │                 │    │          │   │
    │              │    ▼                 ▼    │          │   │
    │              │  [ask]           [execute]│          │   │
    │              │    │                 │    │          │   │
    │              │    ▼                 │    │          │   │
    │              │ ┌──────────────┐     │    │          │   │
    │              │ │ ask_question │     │    │          │   │
    │              │ └──────┬───────┘     │    │          │   │
    │              │        │             │    │          │   │
    │              │        ▼             │    │          │   │
    │              │ ┌──────────────┐     │    │          │   │
    │              │ │update_belief │─────┼────┼──────────┘   │
    │              │ └──────────────┘     │    │              │
    │              │                      │    │              │
    │              ▼                      ▼    ▼              │
    │         ┌──────────────────┐   ┌──────────────┐         │
    │         │     execute      │   │   escalate   │         │
    │         └────────┬─────────┘   └──────┬───────┘         │
    │                  │                    │                 │
    │                  ▼                    ▼                 │
    │              ┌──────┐            ┌──────────┐           │
    │              │ END  │            │   END    │           │
    │              └──────┘            └──────────┘           │
    │                                                         │
    └─────────────────────────────────────────────────────────┘

The graph implements Algorithm 1 from the paper with these nodes:
1. generate_candidates: Generate tool call candidates from user input
2. compute_probabilities: Compute belief-based probabilities π_c(t)
3. generate_questions: Generate clarifying questions
4. select_question: Score questions by EVPI - Cost and select best
5. ask_question: Ask user the selected question
6. update_belief: Update domains based on response
7. execute: Execute the tool call
8. escalate: Escalate to human operator
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    TypedDict,
)

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from ..core.belief import BeliefState
from ..core.domains import ParameterDomain
from ..core.evpi import compute_evpi
from ..core.sage_algorithm import SAGEConfig, TerminationReason
from ..core.types import (
    Aspect,
    CandidateGenerator,
    ConstraintExtractor,
    ExecutionResult,
    Question,
    QuestionAsker,
    QuestionGenerator,
    ToolCall,
    ToolCallCandidate,
    ToolExecutor,
    ToolSchema,
    UNK,
)


class SAGEGraphState(TypedDict):
    """State for SAGE-Agent LangGraph implementation.

    This TypedDict defines all state maintained during the decision loop.
    State is passed between nodes and updated at each step.

    Attributes:
        user_input: Original user request
        observations: List of question responses collected
        domains: Current belief domains {tool_name: {param: domain}}
        candidates: Generated tool call candidates
        probabilities: Normalized probabilities π_c(t)
        best_idx: Index of highest probability candidate
        max_prob: Maximum probability max_c π_c(t)
        questions: Generated clarifying questions
        best_question: Selected question q*
        best_score: Score(q*) = EVPI(q*) - Cost(q*)
        aspect_counts: Count of times each aspect was queried
        step: Current step t (question count)
        status: Current execution status
        tool_call: Final tool call (if executed)
        execution_result: Result from tool execution
        termination_reason: Why the loop terminated
        error: Error message if any
    """
    user_input: str
    observations: List[str]
    domains: Dict[str, Dict[str, ParameterDomain]]
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    best_idx: int
    max_prob: float
    questions: List[Question]
    best_question: Optional[Question]
    best_score: float
    aspect_counts: Dict[str, int]
    step: int
    status: Literal["pending", "executing", "done", "escalated"]
    tool_call: Optional[ToolCall]
    execution_result: Optional[ExecutionResult]
    termination_reason: Optional[str]
    error: Optional[str]


@dataclass
class SAGEGraphConfig:
    """Configuration for SAGE-Agent LangGraph.

    Extends SAGEConfig with graph-specific options.

    Attributes:
        tau_exec: Execution confidence threshold (default: 0.85)
        alpha: Termination threshold factor (default: 0.1)
        lambda_redundancy: Redundancy weight (default: 0.5)
        epsilon: Small probability for infinite domains (default: 1e-4)
        max_questions: Maximum clarification questions (default: 6)
        min_confidence_for_execution: Minimum confidence to execute (default: 0.3)
        enable_escalation: Whether to escalate on uncertainty (default: True)
        recursion_limit: LangGraph recursion limit (default: 50)
    """
    tau_exec: float = 0.85
    alpha: float = 0.1
    lambda_redundancy: float = 0.5
    epsilon: float = 1e-4
    max_questions: int = 6
    min_confidence_for_execution: float = 0.3
    enable_escalation: bool = True
    recursion_limit: int = 50

    def to_sage_config(self) -> SAGEConfig:
        """Convert to core SAGEConfig."""
        return SAGEConfig(
            tau_exec=self.tau_exec,
            alpha=self.alpha,
            lambda_redundancy=self.lambda_redundancy,
            epsilon=self.epsilon,
            max_questions=self.max_questions,
            min_confidence_for_execution=self.min_confidence_for_execution,
            enable_escalation=self.enable_escalation,
        )


@dataclass
class SAGEGraphDeps:
    """Dependencies for SAGE-Agent graph nodes.

    All external dependencies are injected through this dataclass,
    making the graph nodes pure functions of state and deps.
    """
    tool_schemas: Dict[str, ToolSchema]
    candidate_generator: CandidateGenerator
    question_generator: QuestionGenerator
    question_asker: QuestionAsker
    tool_executor: ToolExecutor
    constraint_extractor: Optional[ConstraintExtractor]
    config: SAGEGraphConfig


def build_sage_graph(deps: SAGEGraphDeps) -> "StateGraph":
    """Build the SAGE-Agent LangGraph.

    Args:
        deps: All dependencies for graph nodes

    Returns:
        Compiled LangGraph StateGraph

    Raises:
        ImportError: If langgraph is not installed
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "langgraph is required for SAGEGraph. "
            "Install with: pip install sage-agent[langgraph]"
        )

    graph = StateGraph(SAGEGraphState)

    # =========================================================================
    # Node Definitions
    # =========================================================================

    def generate_candidates_node(state: SAGEGraphState) -> dict:
        """Generate tool call candidates from user input and observations."""
        candidates = deps.candidate_generator.generate_candidates(
            state["user_input"],
            state["observations"],
            deps.tool_schemas,
        )
        # Filter to known tools
        candidates = [c for c in candidates if c.tool_name in deps.tool_schemas]

        if not candidates:
            return {
                "candidates": [],
                "error": "No candidates generated",
                "status": "done",
                "termination_reason": TerminationReason.NO_CANDIDATES.name,
            }

        return {"candidates": candidates}

    def compute_probabilities_node(state: SAGEGraphState) -> dict:
        """Compute belief-based probabilities π_c(t) for each candidate."""
        if not state["candidates"]:
            return {"probabilities": [], "max_prob": 0.0, "best_idx": 0}

        belief = BeliefState(
            domains=state["domains"],
            epsilon=deps.config.epsilon,
        )

        weights = [
            belief.candidate_weight(c, deps.tool_schemas[c.tool_name])
            for c in state["candidates"]
        ]
        probabilities = belief.normalize(weights)
        max_prob = max(probabilities) if probabilities else 0.0
        best_idx = probabilities.index(max_prob) if probabilities else 0

        return {
            "probabilities": probabilities,
            "max_prob": max_prob,
            "best_idx": best_idx,
        }

    def confidence_router(
        state: SAGEGraphState,
    ) -> Literal["confident", "uncertain", "escalate"]:
        """Route based on confidence level.

        - confident: max_c π_c(t) ≥ τ_exec → execute
        - uncertain: need more information → generate questions
        - escalate: too many steps or low confidence → human
        """
        if state.get("error"):
            return "confident"  # Will terminate with error

        if not state["candidates"]:
            return "escalate"

        # Check max questions reached
        if state["step"] >= deps.config.max_questions:
            if state["max_prob"] >= deps.config.min_confidence_for_execution:
                return "confident"
            if deps.config.enable_escalation:
                return "escalate"
            return "confident"

        # Check confidence threshold (paper τ_exec = 0.85)
        if state["max_prob"] >= deps.config.tau_exec:
            return "confident"

        return "uncertain"

    def generate_questions_node(state: SAGEGraphState) -> dict:
        """Generate clarifying questions for uncertain candidates."""
        questions = deps.question_generator.generate_questions(
            state["user_input"],
            state["candidates"],
            state["observations"],
            deps.tool_schemas,
        )

        if not questions:
            return {"questions": [], "best_question": None, "best_score": 0.0}

        return {"questions": questions}

    def select_question_node(state: SAGEGraphState) -> dict:
        """Score questions and select best by EVPI - Cost.

        Score(q) = EVPI(q, B(t)) - Cost(q, t)
        where Cost(q, t) = λ · Σ_{a∈A(q)} n_a(t)
        """
        if not state["questions"]:
            return {"best_question": None, "best_score": 0.0}

        scored: List[tuple[float, Question]] = []
        for question in state["questions"]:
            evpi = compute_evpi(
                state["candidates"],
                state["probabilities"],
                question.aspects,
            )
            # Redundancy cost: λ · Σ_{a∈A(q)} n_a(t)
            cost = sum(
                state["aspect_counts"].get(f"{a.tool_name}:{a.param_name}", 0)
                for a in question.aspects
            )
            score = evpi - deps.config.lambda_redundancy * cost
            scored.append((score, question))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_question = scored[0]

        return {"best_question": best_question, "best_score": best_score}

    def question_router(
        state: SAGEGraphState,
    ) -> Literal["ask", "execute"]:
        """Decide whether to ask question or execute.

        Execute if Score(q*) < α · max_c π_c(t) (cost exceeds benefit)
        """
        if state["best_question"] is None:
            return "execute"

        # Check termination condition from paper
        if state["best_score"] < deps.config.alpha * state["max_prob"]:
            # Cost exceeds benefit, but check required params
            best = state["candidates"][state["best_idx"]]
            schema = deps.tool_schemas[best.tool_name]
            has_unknowns = any(
                best.arguments.get(p, UNK) == UNK for p in schema.required
            )
            if not has_unknowns:
                return "execute"

        return "ask"

    def ask_question_node(state: SAGEGraphState) -> dict:
        """Ask the selected question and record response."""
        if state["best_question"] is None:
            return {}

        response = deps.question_asker.ask(state["best_question"])

        # Update observations
        new_observations = list(state["observations"]) + [response]

        # Update aspect counts
        new_counts = dict(state["aspect_counts"])
        for aspect in state["best_question"].aspects:
            key = f"{aspect.tool_name}:{aspect.param_name}"
            new_counts[key] = new_counts.get(key, 0) + 1

        return {
            "observations": new_observations,
            "aspect_counts": new_counts,
            "step": state["step"] + 1,
        }

    def update_belief_node(state: SAGEGraphState) -> dict:
        """Update belief domains based on user response."""
        if state["best_question"] is None or not state["observations"]:
            return {}

        if deps.constraint_extractor is None:
            return {}

        response = state["observations"][-1]
        new_domains = {
            tool: dict(params) for tool, params in state["domains"].items()
        }

        for aspect in state["best_question"].aspects:
            tool = deps.tool_schemas[aspect.tool_name]
            current = new_domains[tool.name][aspect.param_name]
            updated = deps.constraint_extractor.update_domain(current, response)
            new_domains[tool.name][aspect.param_name] = updated

            # Apply domain refiner if available
            if tool.domain_refiner is not None:
                refined = tool.domain_refiner.refine(
                    tool, new_domains[tool.name], {}
                )
                new_domains[tool.name] = dict(refined)

        return {"domains": new_domains}

    def execute_node(state: SAGEGraphState) -> dict:
        """Execute the best candidate tool call."""
        if state.get("error") or not state["candidates"]:
            return {
                "status": "done",
                "termination_reason": TerminationReason.NO_CANDIDATES.name,
            }

        candidate = state["candidates"][state["best_idx"]]
        schema = deps.tool_schemas[candidate.tool_name]

        # Create tool call
        tool_call = ToolCall(
            tool_name=candidate.tool_name,
            arguments=dict(candidate.arguments),
        )

        # Validate required parameters
        try:
            schema.validate_call(tool_call.arguments)
        except ValueError as exc:
            return {
                "tool_call": tool_call,
                "execution_result": ExecutionResult(success=False, error=str(exc)),
                "status": "done",
                "error": str(exc),
            }

        # Execute
        result = deps.tool_executor.execute(tool_call)

        # Determine termination reason
        if state["max_prob"] >= deps.config.tau_exec:
            reason = TerminationReason.CONFIDENT_EXECUTION
        elif state["step"] >= deps.config.max_questions:
            reason = TerminationReason.MAX_QUESTIONS_REACHED
        else:
            reason = TerminationReason.COST_EXCEEDS_BENEFIT

        return {
            "tool_call": tool_call,
            "execution_result": result,
            "status": "done",
            "termination_reason": reason.name,
        }

    def escalate_node(state: SAGEGraphState) -> dict:
        """Escalate to human operator."""
        return {
            "status": "escalated",
            "termination_reason": TerminationReason.ESCALATED.name,
            "error": state.get("error") or "Escalated due to high uncertainty",
        }

    # =========================================================================
    # Graph Construction
    # =========================================================================

    # Add nodes
    graph.add_node("generate_candidates", generate_candidates_node)
    graph.add_node("compute_probabilities", compute_probabilities_node)
    graph.add_node("generate_questions", generate_questions_node)
    graph.add_node("select_question", select_question_node)
    graph.add_node("ask_question", ask_question_node)
    graph.add_node("update_belief", update_belief_node)
    graph.add_node("execute", execute_node)
    graph.add_node("escalate", escalate_node)

    # Entry point
    graph.set_entry_point("generate_candidates")

    # generate_candidates -> compute_probabilities
    graph.add_edge("generate_candidates", "compute_probabilities")

    # compute_probabilities -> confidence router
    graph.add_conditional_edges(
        "compute_probabilities",
        confidence_router,
        {
            "confident": "execute",
            "uncertain": "generate_questions",
            "escalate": "escalate",
        },
    )

    # generate_questions -> select_question
    graph.add_edge("generate_questions", "select_question")

    # select_question -> question router
    graph.add_conditional_edges(
        "select_question",
        question_router,
        {
            "ask": "ask_question",
            "execute": "execute",
        },
    )

    # ask_question -> update_belief -> generate_candidates (loop)
    graph.add_edge("ask_question", "update_belief")
    graph.add_edge("update_belief", "generate_candidates")

    # Terminal nodes
    graph.add_edge("execute", END)
    graph.add_edge("escalate", END)

    return graph


def create_initial_state(
    user_input: str,
    tool_schemas: Mapping[str, ToolSchema],
) -> SAGEGraphState:
    """Create initial state for SAGE graph execution.

    Args:
        user_input: User's natural language request
        tool_schemas: Available tool schemas

    Returns:
        Initial SAGEGraphState
    """
    return SAGEGraphState(
        user_input=user_input,
        observations=[],
        domains={
            name: dict(schema.parameters)
            for name, schema in tool_schemas.items()
        },
        candidates=[],
        probabilities=[],
        best_idx=0,
        max_prob=0.0,
        questions=[],
        best_question=None,
        best_score=0.0,
        aspect_counts={},
        step=0,
        status="pending",
        tool_call=None,
        execution_result=None,
        termination_reason=None,
        error=None,
    )


class SAGEGraph:
    """High-level wrapper for SAGE-Agent LangGraph.

    Provides a simple interface matching the core SAGEAgent API.

    Example:
        ```python
        graph = SAGEGraph(
            tool_schemas=[booking_tool],
            candidate_generator=gen,
            question_generator=q_gen,
            question_asker=asker,
            tool_executor=executor,
        )

        result = graph.run("Book a flight to NYC")
        print(f"Executed: {result['tool_call']}")
        print(f"Status: {result['status']}")
        ```
    """

    def __init__(
        self,
        tool_schemas: Iterable[ToolSchema],
        candidate_generator: CandidateGenerator,
        question_generator: QuestionGenerator,
        question_asker: QuestionAsker,
        tool_executor: ToolExecutor,
        constraint_extractor: Optional[ConstraintExtractor] = None,
        config: Optional[SAGEGraphConfig] = None,
    ) -> None:
        """Initialize SAGE Graph.

        Args:
            tool_schemas: Available tools with parameter schemas
            candidate_generator: Generates tool call candidates
            question_generator: Generates clarification questions
            question_asker: Interface for asking questions
            tool_executor: Executes tool calls
            constraint_extractor: Updates domains from responses
            config: Graph configuration
        """
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "langgraph is required for SAGEGraph. "
                "Install with: pip install sage-agent[langgraph]"
            )

        self.tool_schemas: Dict[str, ToolSchema] = {
            t.name: t for t in tool_schemas
        }
        self.config = config or SAGEGraphConfig()

        self._deps = SAGEGraphDeps(
            tool_schemas=self.tool_schemas,
            candidate_generator=candidate_generator,
            question_generator=question_generator,
            question_asker=question_asker,
            tool_executor=tool_executor,
            constraint_extractor=constraint_extractor,
            config=self.config,
        )

        self._graph = build_sage_graph(self._deps).compile()

    def run(self, user_input: str) -> SAGEGraphState:
        """Execute SAGE-Agent decision loop.

        Args:
            user_input: User's natural language request

        Returns:
            Final graph state with tool_call, execution_result, and status
        """
        initial = create_initial_state(user_input, self.tool_schemas)
        return self._graph.invoke(
            initial,
            {"recursion_limit": self.config.recursion_limit},
        )

    @property
    def compiled_graph(self):
        """Access the compiled LangGraph for advanced usage."""
        return self._graph
