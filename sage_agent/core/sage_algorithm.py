"""Canonical SAGE-Agent Algorithm 1 Implementation.

This module implements Algorithm 1 from Section 5 of the paper:
"Structured Uncertainty Guided Clarification for LLM Agents"

Algorithm 1: SAGE-Agent Decision Loop
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1: Initialize belief B(0), aspect counts n(0), t ← 0
2: while t < T_max do
3:     Generate candidates C from user request and observations
4:     Compute probabilities: π_c(t) ∝ Π_c(t) for each c ∈ C
5:     if max_c π_c(t) ≥ τ_exec then
6:         Execute argmax_c π_c(t) and return
7:     end if
8:     Generate candidate questions Q
9:     For each q ∈ Q: compute Score(q) = EVPI(q, B(t)) - Cost(q, t)
10:    Select q* = argmax_q Score(q)
11:    if Score(q*) < α · max_c π_c(t) then
12:        Execute argmax_c π_c(t) and return (cost exceeds benefit)
13:    end if
14:    Ask q*, receive response r
15:    Update domains D(t+1) based on r
16:    Update aspect counts: n_a(t+1) ← n_a(t) + 1 for a ∈ A(q*)
17:    t ← t + 1
18: end while
19: Final execution or escalation based on confidence

Key Formulas:
- Candidate viability: Π_i(t) ∝ ∏_j p(θ_{i,j} | T_i, observations_t)
- Parameter certainty: p(θ) = 1 if specified, 1/|D| if finite, ε if infinite
- EVPI: EVPI(q, B(t)) = E_r[max_c π_c(t|q,r)] - max_c π_c(t)
- Redundancy cost: Cost(q, t) = λ · Σ_{a∈A(q)} n_a(t)
- Termination: Stop when Score(q*) < α · max_c π_c(t)

Paper hyperparameters:
- τ_exec = 0.85 (execution confidence threshold)
- α = 0.1 (termination threshold factor)
- λ = 0.5 (redundancy weight)
- ε = 1e-4 (epsilon for infinite domains)
- T_max = 6 (maximum clarification questions)
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

from .belief import BeliefState
from .domains import ParameterDomain
from .evpi import compute_evpi
from .types import (
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


class TerminationReason(Enum):
    """Reason why the SAGE agent terminated."""
    CONFIDENT_EXECUTION = auto()      # max_c π_c(t) ≥ τ_exec
    COST_EXCEEDS_BENEFIT = auto()     # Score(q*) < α · max_c π_c(t)
    MAX_QUESTIONS_REACHED = auto()    # t ≥ T_max
    NO_QUESTIONS_AVAILABLE = auto()   # Question generator returned empty
    NO_CANDIDATES = auto()            # No valid candidates generated
    ESCALATED = auto()                # Escalated to human operator


@dataclass(frozen=True)
class SAGEConfig:
    """Configuration for SAGE-Agent matching paper hyperparameters.

    All default values match those specified in the paper.

    Attributes:
        tau_exec: Execution confidence threshold τ_exec (default: 0.85)
                  Agent executes when max_c π_c(t) ≥ τ_exec
        alpha: Termination threshold factor α (default: 0.1)
               Stop questioning when Score(q*) < α · max_c π_c(t)
        lambda_redundancy: Redundancy weight λ (default: 0.5)
                          Cost(q, t) = λ · Σ_{a∈A(q)} n_a(t)
        epsilon: Small probability for infinite domains ε (default: 1e-4)
        max_questions: Maximum clarification questions T_max (default: 6)
        min_confidence_for_execution: Minimum confidence to execute at termination
                                     If below this, escalate instead (default: 0.3)
        enable_escalation: Whether to escalate when uncertain (default: True)
    """
    tau_exec: float = 0.85
    alpha: float = 0.1
    lambda_redundancy: float = 0.5
    epsilon: float = 1e-4
    max_questions: int = 6
    min_confidence_for_execution: float = 0.3
    enable_escalation: bool = True


@dataclass
class SAGEStep:
    """Record of a single step in the SAGE decision loop.

    Captures the state at each iteration for debugging and analysis.

    Attributes:
        t: Time step (question count)
        candidates: Generated tool call candidates
        probabilities: Normalized probabilities π_c(t) for each candidate
        max_probability: Maximum probability max_c π_c(t)
        best_candidate_idx: Index of highest probability candidate
        questions: Generated candidate questions (may be empty)
        question_scores: Score(q) = EVPI(q) - Cost(q) for each question
        selected_question: Best question q* selected for asking
        selected_score: Score of selected question
        response: User response to the question (None if not asked)
    """
    t: int
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    max_probability: float
    best_candidate_idx: int
    questions: List[Question] = field(default_factory=list)
    question_scores: List[float] = field(default_factory=list)
    selected_question: Optional[Question] = None
    selected_score: Optional[float] = None
    response: Optional[str] = None


@dataclass
class SAGEResult:
    """Final result of SAGE-Agent execution.

    Attributes:
        tool_call: The tool call that was executed (None if escalated)
        execution_result: Result from tool execution (None if escalated)
        escalated: Whether the request was escalated to human
        termination_reason: Why the agent terminated
        final_probability: Confidence in the final decision max_c π_c(t)
        steps: Record of all decision steps
        total_questions: Number of clarification questions asked
    """
    tool_call: Optional[ToolCall]
    execution_result: Optional[ExecutionResult]
    escalated: bool
    termination_reason: TerminationReason
    final_probability: float
    steps: List[SAGEStep] = field(default_factory=list)
    total_questions: int = 0


class SAGEAgent:
    """Canonical SAGE-Agent implementing Algorithm 1 from the paper.

    This class provides a clean implementation of the SAGE decision loop
    for structured uncertainty-guided clarification in tool-calling agents.

    The agent maintains a belief state over tool call candidates and uses
    EVPI (Expected Value of Perfect Information) to select optimal
    clarification questions, balancing information gain against redundancy.

    Example:
        ```python
        agent = SAGEAgent(
            tool_schemas=[booking_tool, search_tool],
            candidate_generator=llm_candidate_gen,
            question_generator=llm_question_gen,
            question_asker=user_interface,
            tool_executor=api_executor,
        )

        result = agent.run("Book a flight to NYC next week")
        if result.escalated:
            print("Request escalated to human operator")
        else:
            print(f"Executed: {result.tool_call}")
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
        config: Optional[SAGEConfig] = None,
    ) -> None:
        """Initialize SAGE-Agent.

        Args:
            tool_schemas: Available tools with parameter schemas
            candidate_generator: Generates tool call candidates from user input
            question_generator: Generates clarification questions
            question_asker: Interface for asking questions to user
            tool_executor: Executes tool calls
            constraint_extractor: Updates domains based on user responses
            config: Agent configuration (uses paper defaults if None)
        """
        self.tool_schemas: Dict[str, ToolSchema] = {
            tool.name: tool for tool in tool_schemas
        }
        self.candidate_generator = candidate_generator
        self.question_generator = question_generator
        self.question_asker = question_asker
        self.tool_executor = tool_executor
        self.constraint_extractor = constraint_extractor
        self.config = config or SAGEConfig()

        # Store base domains for reset between runs
        self._base_domains: Dict[str, Dict[str, ParameterDomain]] = {
            tool.name: dict(tool.parameters) for tool in self.tool_schemas.values()
        }

    def run(self, user_input: str) -> SAGEResult:
        """Execute Algorithm 1: SAGE-Agent Decision Loop.

        Args:
            user_input: User's natural language request

        Returns:
            SAGEResult with tool call, execution result, and decision trace
        """
        # Line 1: Initialize belief B(0), aspect counts n(0), t ← 0
        belief = BeliefState(
            domains=copy.deepcopy(self._base_domains),
            epsilon=self.config.epsilon,
        )
        aspect_counts: Dict[Aspect, int] = defaultdict(int)
        observations: List[str] = []
        steps: List[SAGEStep] = []
        t = 0

        # Line 2: while t < T_max do
        while t < self.config.max_questions:
            # Line 3: Generate candidates C
            candidates = self.candidate_generator.generate_candidates(
                user_input, observations, self.tool_schemas
            )

            if not candidates:
                return SAGEResult(
                    tool_call=None,
                    execution_result=None,
                    escalated=self.config.enable_escalation,
                    termination_reason=TerminationReason.NO_CANDIDATES,
                    final_probability=0.0,
                    steps=steps,
                    total_questions=t,
                )

            # Validate candidates reference known tools
            candidates = [
                c for c in candidates if c.tool_name in self.tool_schemas
            ]
            if not candidates:
                return SAGEResult(
                    tool_call=None,
                    execution_result=None,
                    escalated=self.config.enable_escalation,
                    termination_reason=TerminationReason.NO_CANDIDATES,
                    final_probability=0.0,
                    steps=steps,
                    total_questions=t,
                )

            # Line 4: Compute probabilities π_c(t) ∝ Π_c(t)
            weights = [
                belief.candidate_weight(c, self.tool_schemas[c.tool_name])
                for c in candidates
            ]
            probabilities = belief.normalize(weights)
            max_prob = max(probabilities)
            best_idx = probabilities.index(max_prob)
            best_candidate = candidates[best_idx]

            # Create step record
            step = SAGEStep(
                t=t,
                candidates=list(candidates),
                probabilities=list(probabilities),
                max_probability=max_prob,
                best_candidate_idx=best_idx,
            )

            # Line 5-7: if max_c π_c(t) ≥ τ_exec then execute
            if max_prob >= self.config.tau_exec:
                steps.append(step)
                return self._execute(
                    best_candidate,
                    max_prob,
                    TerminationReason.CONFIDENT_EXECUTION,
                    steps,
                    t,
                )

            # Line 8: Generate candidate questions Q
            questions = self.question_generator.generate_questions(
                user_input, candidates, observations, self.tool_schemas
            )

            if not questions:
                # No questions available, execute best candidate
                steps.append(step)
                return self._execute(
                    best_candidate,
                    max_prob,
                    TerminationReason.NO_QUESTIONS_AVAILABLE,
                    steps,
                    t,
                )

            # Line 9: Compute Score(q) = EVPI(q) - Cost(q) for each question
            scored_questions: List[Tuple[float, Question]] = []
            for question in questions:
                evpi = compute_evpi(candidates, probabilities, question.aspects)
                cost = self._redundancy_cost(question, aspect_counts)
                score = evpi - cost
                scored_questions.append((score, question))

            # Line 10: Select q* = argmax_q Score(q)
            scored_questions.sort(key=lambda x: x[0], reverse=True)
            best_score, best_question = scored_questions[0]

            # Update step with question info
            step.questions = questions
            step.question_scores = [s for s, _ in scored_questions]
            step.selected_question = best_question
            step.selected_score = best_score

            # Line 11-13: if Score(q*) < α · max_c π_c(t) then execute
            if best_score < self.config.alpha * max_prob:
                # Cost exceeds benefit, execute if we have required params
                if not self._has_required_unknowns(best_candidate):
                    steps.append(step)
                    return self._execute(
                        best_candidate,
                        max_prob,
                        TerminationReason.COST_EXCEEDS_BENEFIT,
                        steps,
                        t,
                    )
                # Has required unknowns, must continue asking

            # Line 14: Ask q*, receive response r
            response = self.question_asker.ask(best_question)
            step.response = response
            observations.append(response)

            # Line 15: Update domains D(t+1) based on r
            for aspect in best_question.aspects:
                self._update_domain(belief, aspect, response)

            # Line 16: Update aspect counts
            for aspect in best_question.aspects:
                aspect_counts[aspect] += 1

            steps.append(step)

            # Line 17: t ← t + 1
            t += 1

        # Line 18-19: Max questions reached, final execution or escalation
        # Generate final candidates with all observations
        candidates = self.candidate_generator.generate_candidates(
            user_input, observations, self.tool_schemas
        )
        candidates = [c for c in candidates if c.tool_name in self.tool_schemas]

        if not candidates:
            return SAGEResult(
                tool_call=None,
                execution_result=None,
                escalated=self.config.enable_escalation,
                termination_reason=TerminationReason.MAX_QUESTIONS_REACHED,
                final_probability=0.0,
                steps=steps,
                total_questions=t,
            )

        weights = [
            belief.candidate_weight(c, self.tool_schemas[c.tool_name])
            for c in candidates
        ]
        probabilities = belief.normalize(weights)
        max_prob = max(probabilities)
        best_idx = probabilities.index(max_prob)
        best_candidate = candidates[best_idx]

        # Check if confidence is sufficient for execution
        if max_prob < self.config.min_confidence_for_execution:
            if self.config.enable_escalation:
                return SAGEResult(
                    tool_call=None,
                    execution_result=None,
                    escalated=True,
                    termination_reason=TerminationReason.ESCALATED,
                    final_probability=max_prob,
                    steps=steps,
                    total_questions=t,
                )

        return self._execute(
            best_candidate,
            max_prob,
            TerminationReason.MAX_QUESTIONS_REACHED,
            steps,
            t,
        )

    def _execute(
        self,
        candidate: ToolCallCandidate,
        probability: float,
        reason: TerminationReason,
        steps: List[SAGEStep],
        questions_asked: int,
    ) -> SAGEResult:
        """Execute a tool call candidate.

        Args:
            candidate: Tool call to execute
            probability: Confidence in this candidate
            reason: Why we're executing now
            steps: Decision trace
            questions_asked: Number of questions asked

        Returns:
            SAGEResult with execution outcome
        """
        tool_schema = self.tool_schemas[candidate.tool_name]
        tool_call = ToolCall(
            tool_name=candidate.tool_name,
            arguments=dict(candidate.arguments),
        )

        # Validate required parameters
        try:
            tool_schema.validate_call(tool_call.arguments)
        except ValueError as exc:
            # Missing required parameters
            result = ExecutionResult(success=False, error=str(exc))
            return SAGEResult(
                tool_call=tool_call,
                execution_result=result,
                escalated=False,
                termination_reason=reason,
                final_probability=probability,
                steps=steps,
                total_questions=questions_asked,
            )

        # Execute the tool
        result = self.tool_executor.execute(tool_call)
        return SAGEResult(
            tool_call=tool_call,
            execution_result=result,
            escalated=False,
            termination_reason=reason,
            final_probability=probability,
            steps=steps,
            total_questions=questions_asked,
        )

    def _redundancy_cost(
        self,
        question: Question,
        aspect_counts: Mapping[Aspect, int],
    ) -> float:
        """Compute redundancy cost: Cost(q, t) = λ · Σ_{a∈A(q)} n_a(t).

        Penalizes asking about aspects that have already been queried,
        encouraging the agent to explore different aspects.

        Args:
            question: Question being scored
            aspect_counts: How many times each aspect has been queried

        Returns:
            Redundancy cost for this question
        """
        total_count = sum(aspect_counts.get(a, 0) for a in question.aspects)
        return self.config.lambda_redundancy * total_count

    def _update_domain(
        self,
        belief: BeliefState,
        aspect: Aspect,
        response: str,
    ) -> None:
        """Update belief domain based on user response.

        Args:
            belief: Current belief state
            aspect: The aspect that was queried
            response: User's response
        """
        if self.constraint_extractor is None:
            return

        tool = self.tool_schemas[aspect.tool_name]
        current_domain = belief.domains[tool.name][aspect.param_name]
        updated_domain = self.constraint_extractor.update_domain(
            current_domain, response
        )
        belief.domains[tool.name][aspect.param_name] = updated_domain

        # Apply domain refiner if available
        if tool.domain_refiner is not None:
            refined = tool.domain_refiner.refine(
                tool,
                belief.domains[tool.name],
                self._current_assignments(belief, tool.name),
            )
            belief.domains[tool.name] = dict(refined)

    def _current_assignments(
        self,
        belief: BeliefState,
        tool_name: str,
    ) -> Mapping[str, object]:
        """Get current known parameter assignments from belief state.

        Parameters with domain size 1 are considered "known".

        Args:
            belief: Current belief state
            tool_name: Tool to get assignments for

        Returns:
            Mapping of parameter names to known values (or UNK)
        """
        assignments: Dict[str, object] = {}
        for param, domain in belief.domains[tool_name].items():
            if domain.is_finite() and domain.size() == 1:
                # Domain reduced to single value - parameter is known
                values = domain.values
                if values:
                    assignments[param] = next(iter(values))
                else:
                    assignments[param] = UNK
            else:
                assignments[param] = UNK
        return assignments

    def _has_required_unknowns(self, candidate: ToolCallCandidate) -> bool:
        """Check if candidate has unknown required parameters.

        Args:
            candidate: Tool call candidate

        Returns:
            True if any required parameter is UNK
        """
        tool_schema = self.tool_schemas[candidate.tool_name]
        for param in tool_schema.required:
            if candidate.arguments.get(param, UNK) == UNK:
                return True
        return False
