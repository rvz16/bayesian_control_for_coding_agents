"""POMDP Formulation for SAGE-Agent.

This module provides the formal POMDP (Partially Observable Markov Decision Process)
definitions from Section 3 of the paper:
"Structured Uncertainty Guided Clarification for LLM Agents"

POMDP Tuple: M = (S, A, T, O, Z, R, γ)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Definition 1 (State Space S):
    s = (θ*, {D_j(t)}) where:
    - θ* is the true user intent (unknown)
    - D_j(t) is the domain of parameter j at time t

Definition 2 (Structured Candidate Space C):
    C = {c_i = (T_i, {θ_{i,j}}) : T_i ∈ Tools}
    where each candidate is a tool with argument assignments

Definition 3 (Belief State B(t)):
    B(t) = {(c_i, Π_i(t)) : c_i ∈ C}
    where Π_i(t) ∈ [0, 1] is the candidate viability

Action Space A:
    - Execute: a_exec(c) - Execute tool call candidate c
    - Query: a_query(q) - Ask clarification question q
    - Escalate: a_esc - Escalate to human operator

Transition Function T(s'|s, a):
    - For a_exec: Terminal state (episode ends)
    - For a_query: Domain refinement D_j(t+1) ⊆ D_j(t)

Observation Model Z(o|s, a):
    - For a_query: o = user_response ∈ natural language
    - Processed by constraint extractor to update domains

Reward Function R(s, a):
    - R_exec: +1 if correct execution, 0 otherwise
    - R_query: -λ (cost of asking)
    - Certainty-weighted (Section 6.2): R(a) = Cert(a) × R_base(a)

Paper reference: arXiv:2511.08798, Section 3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from .belief import BeliefState
from .domains import ParameterDomain
from .types import (
    Aspect,
    Question,
    ToolCall,
    ToolCallCandidate,
    ToolSchema,
    UNK,
)


# =============================================================================
# Action Space (Definition from Section 3.1)
# =============================================================================

class ActionType(Enum):
    """Types of actions available in the SAGE POMDP.

    The agent can take three types of actions:
    1. EXECUTE: Commit to a tool call and execute it
    2. QUERY: Ask a clarification question to gather information
    3. ESCALATE: Transfer control to a human operator
    """
    EXECUTE = auto()
    QUERY = auto()
    ESCALATE = auto()


@dataclass(frozen=True)
class Action:
    """Base class for POMDP actions."""
    action_type: ActionType


@dataclass(frozen=True)
class ExecuteAction(Action):
    """Execute a tool call candidate.

    This action terminates the episode. The agent commits to a specific
    tool call and receives a reward based on correctness.

    Attributes:
        candidate: The tool call candidate to execute
    """
    candidate: ToolCallCandidate
    action_type: ActionType = field(default=ActionType.EXECUTE, init=False)


@dataclass(frozen=True)
class QueryAction(Action):
    """Ask a clarification question.

    This action gathers information to reduce uncertainty. The agent
    asks a question and receives a natural language response that
    is processed to update the belief state.

    Attributes:
        question: The clarification question to ask
    """
    question: Question
    action_type: ActionType = field(default=ActionType.QUERY, init=False)


@dataclass(frozen=True)
class EscalateAction(Action):
    """Escalate to human operator.

    This action is taken when the agent cannot resolve uncertainty
    within acceptable bounds. Control is transferred to a human.

    Attributes:
        reason: Explanation for escalation
    """
    reason: str = "High uncertainty"
    action_type: ActionType = field(default=ActionType.ESCALATE, init=False)


# =============================================================================
# State Space (Definition 1)
# =============================================================================

@dataclass(frozen=True)
class POMDPState:
    """POMDP state representation (Definition 1).

    s = (θ*, {D_j(t)}) where:
    - θ* is the true user intent (hidden, represented by ground_truth)
    - D_j(t) are the current parameter domains

    In practice, θ* is unknown to the agent. The state is used for:
    1. Simulation: When we know the ground truth for evaluation
    2. Formal analysis: For theoretical POMDP computations

    Attributes:
        ground_truth: The true intended tool call (hidden from agent)
        domains: Current parameter domains {tool: {param: domain}}
        observations: History of user responses
        time_step: Current step t in the decision loop
    """
    ground_truth: Optional[ToolCall]  # Hidden from agent, used in simulation
    domains: Dict[str, Dict[str, ParameterDomain]]
    observations: Tuple[str, ...] = field(default_factory=tuple)
    time_step: int = 0

    def with_observation(self, response: str) -> "POMDPState":
        """Create new state with additional observation."""
        return POMDPState(
            ground_truth=self.ground_truth,
            domains=self.domains,
            observations=self.observations + (response,),
            time_step=self.time_step + 1,
        )


# =============================================================================
# Observation Space
# =============================================================================

@dataclass(frozen=True)
class Observation:
    """Observation from the environment.

    In the SAGE POMDP, observations are user responses to clarification
    questions. The observation is processed by a constraint extractor
    to update the belief state.

    Attributes:
        response: Natural language response from user
        question: The question that was asked (context)
        extracted_values: Values extracted by constraint extractor
    """
    response: str
    question: Question
    extracted_values: Dict[Aspect, object] = field(default_factory=dict)


# =============================================================================
# Transition Model (Section 3.2)
# =============================================================================

class TransitionModel(Protocol):
    """Transition function T(s'|s, a).

    Models how the state changes based on actions:
    - EXECUTE: Transitions to terminal state
    - QUERY: Domains may shrink based on user response
    - ESCALATE: Transitions to terminal state
    """

    def transition(
        self,
        state: POMDPState,
        action: Action,
        observation: Optional[Observation],
    ) -> POMDPState:
        """Compute next state given current state, action, and observation."""
        ...


@dataclass
class DomainRefinementTransition:
    """Transition model based on domain refinement.

    For QUERY actions, the domains shrink based on the user's response.
    The constraint extractor processes the response to extract constraints.
    """
    constraint_extractor: Any  # ConstraintExtractor

    def transition(
        self,
        state: POMDPState,
        action: Action,
        observation: Optional[Observation],
    ) -> POMDPState:
        """Apply domain refinement transition."""
        if action.action_type == ActionType.EXECUTE:
            # Terminal state after execution
            return state

        if action.action_type == ActionType.ESCALATE:
            # Terminal state after escalation
            return state

        if action.action_type == ActionType.QUERY and observation is not None:
            # Refine domains based on observation
            new_domains = {
                tool: dict(params) for tool, params in state.domains.items()
            }

            query_action: QueryAction = action  # type: ignore
            for aspect in query_action.question.aspects:
                if aspect.tool_name in new_domains:
                    current = new_domains[aspect.tool_name].get(aspect.param_name)
                    if current is not None and self.constraint_extractor:
                        refined = self.constraint_extractor.update_domain(
                            current, observation.response
                        )
                        new_domains[aspect.tool_name][aspect.param_name] = refined

            return POMDPState(
                ground_truth=state.ground_truth,
                domains=new_domains,
                observations=state.observations + (observation.response,),
                time_step=state.time_step + 1,
            )

        return state


# =============================================================================
# Observation Model
# =============================================================================

class ObservationModel(Protocol):
    """Observation function Z(o|s, a).

    Models the probability of receiving observation o given state s
    and action a. In practice, this is the likelihood that a user
    gives a particular response to a clarification question.
    """

    def observation_probability(
        self,
        observation: Observation,
        state: POMDPState,
        action: Action,
    ) -> float:
        """Compute probability of observation given state and action."""
        ...


@dataclass
class UniformObservationModel:
    """Simple observation model assuming uniform responses.

    For theoretical analysis, assumes user responses are uniformly
    sampled from the domain. In practice, responses are deterministic
    given the true intent θ*.
    """

    def observation_probability(
        self,
        observation: Observation,
        state: POMDPState,
        action: Action,
    ) -> float:
        """Compute observation probability.

        If ground truth is known (simulation), probability is 1 for
        consistent observations and 0 otherwise.
        """
        if action.action_type != ActionType.QUERY:
            return 1.0  # No observation for non-query actions

        if state.ground_truth is None:
            return 1.0  # Unknown ground truth, accept any observation

        # Check consistency with ground truth
        query_action: QueryAction = action  # type: ignore
        for aspect in query_action.question.aspects:
            if aspect.tool_name == state.ground_truth.tool_name:
                true_value = state.ground_truth.arguments.get(aspect.param_name)
                if true_value is not None:
                    # Check if extracted value matches ground truth
                    extracted = observation.extracted_values.get(aspect)
                    if extracted is not None and extracted != true_value:
                        return 0.0

        return 1.0


# =============================================================================
# Reward Model (Section 3.3 and 6.2)
# =============================================================================

class RewardModel(Protocol):
    """Reward function R(s, a).

    Defines the reward for taking action a in state s.
    """

    def reward(
        self,
        state: POMDPState,
        action: Action,
        next_state: Optional[POMDPState] = None,
    ) -> float:
        """Compute reward for state-action pair."""
        ...


@dataclass
class SAGERewardModel:
    """SAGE-Agent reward model with certainty weighting.

    From Section 6.2:
        Cert(a_t) = max_c π_c(t) if EXECUTE
        Cert(a_t) = 1 - max_c π_c(t) if QUERY
        R(a_t) = Cert(a_t) × R_base(a_t)

    Attributes:
        execution_reward: Base reward for correct execution (+1)
        query_cost: Cost per clarification question (λ)
        incorrect_penalty: Penalty for incorrect execution
    """
    execution_reward: float = 1.0
    query_cost: float = 0.1
    incorrect_penalty: float = -1.0

    def reward(
        self,
        state: POMDPState,
        action: Action,
        next_state: Optional[POMDPState] = None,
        belief: Optional[BeliefState] = None,
        candidates: Optional[Sequence[ToolCallCandidate]] = None,
        tool_schemas: Optional[Mapping[str, ToolSchema]] = None,
    ) -> float:
        """Compute certainty-weighted reward."""
        if action.action_type == ActionType.ESCALATE:
            return 0.0  # Neutral reward for escalation

        # Compute certainty from belief state
        certainty = 0.5  # Default if belief not provided
        if belief is not None and candidates and tool_schemas:
            weights = [
                belief.candidate_weight(c, tool_schemas[c.tool_name])
                for c in candidates
                if c.tool_name in tool_schemas
            ]
            probs = belief.normalize(weights)
            max_prob = max(probs) if probs else 0.0

            if action.action_type == ActionType.EXECUTE:
                certainty = max_prob
            elif action.action_type == ActionType.QUERY:
                certainty = 1.0 - max_prob

        if action.action_type == ActionType.QUERY:
            return certainty * (-self.query_cost)

        if action.action_type == ActionType.EXECUTE:
            exec_action: ExecuteAction = action  # type: ignore
            # Check correctness if ground truth known
            if state.ground_truth is not None:
                is_correct = (
                    exec_action.candidate.tool_name == state.ground_truth.tool_name
                    and all(
                        exec_action.candidate.arguments.get(k) == v
                        for k, v in state.ground_truth.arguments.items()
                        if v != UNK
                    )
                )
                base = self.execution_reward if is_correct else self.incorrect_penalty
            else:
                base = self.execution_reward  # Assume correct if unknown

            return certainty * base

        return 0.0


# =============================================================================
# Value Functions
# =============================================================================

@dataclass
class ValueEstimate:
    """Value function estimate V(b) for a belief state.

    In the SAGE framework, we use EVPI as a proxy for value:
    V(b) ≈ max_c π_c(t) + EVPI_gain

    Attributes:
        value: Estimated value
        max_probability: Maximum candidate probability
        evpi: Expected value of perfect information
        question: Best question to ask (if querying is valuable)
    """
    value: float
    max_probability: float
    evpi: float = 0.0
    best_question: Optional[Question] = None
    recommended_action: Optional[ActionType] = None


def compute_value(
    belief: BeliefState,
    candidates: Sequence[ToolCallCandidate],
    questions: Sequence[Question],
    tool_schemas: Mapping[str, ToolSchema],
    tau_exec: float = 0.85,
    alpha: float = 0.1,
    lambda_redundancy: float = 0.5,
) -> ValueEstimate:
    """Compute value estimate for current belief state.

    Uses the decision criteria from Algorithm 1:
    - Execute if max_c π_c(t) ≥ τ_exec
    - Otherwise, compute EVPI and decide query vs execute

    Args:
        belief: Current belief state
        candidates: Tool call candidates
        questions: Available clarification questions
        tool_schemas: Tool definitions
        tau_exec: Execution confidence threshold
        alpha: Termination factor
        lambda_redundancy: Redundancy cost weight

    Returns:
        ValueEstimate with value and recommended action
    """
    from .evpi import compute_evpi

    # Compute candidate probabilities
    weights = [
        belief.candidate_weight(c, tool_schemas[c.tool_name])
        for c in candidates
        if c.tool_name in tool_schemas
    ]
    probs = belief.normalize(weights)
    max_prob = max(probs) if probs else 0.0

    # High confidence: execute immediately
    if max_prob >= tau_exec:
        return ValueEstimate(
            value=max_prob,
            max_probability=max_prob,
            evpi=0.0,
            recommended_action=ActionType.EXECUTE,
        )

    # Compute EVPI for each question
    best_evpi = 0.0
    best_question = None

    for question in questions:
        evpi = compute_evpi(list(candidates), list(probs), question.aspects)
        if evpi > best_evpi:
            best_evpi = evpi
            best_question = question

    # Apply termination criterion
    score = best_evpi - lambda_redundancy * 0  # Simplified: no redundancy tracking here
    if score < alpha * max_prob:
        # Cost exceeds benefit: execute
        return ValueEstimate(
            value=max_prob,
            max_probability=max_prob,
            evpi=best_evpi,
            best_question=best_question,
            recommended_action=ActionType.EXECUTE,
        )

    # Query is valuable
    expected_value = max_prob + best_evpi  # Approximate future value
    return ValueEstimate(
        value=expected_value,
        max_probability=max_prob,
        evpi=best_evpi,
        best_question=best_question,
        recommended_action=ActionType.QUERY,
    )


# =============================================================================
# POMDP Definition (Full Tuple)
# =============================================================================

@dataclass
class POMDP:
    """Complete POMDP formulation M = (S, A, T, O, Z, R, γ).

    This class encapsulates all POMDP components for the SAGE-Agent
    clarification problem.

    Attributes:
        tool_schemas: Defines state space structure (parameters and domains)
        transition_model: State transition function T(s'|s,a)
        observation_model: Observation function Z(o|s,a)
        reward_model: Reward function R(s,a)
        discount: Discount factor γ (default: 1.0 for finite horizon)
        max_steps: Maximum steps T_max (horizon)
        tau_exec: Execution threshold from paper
        alpha: Termination factor from paper
        lambda_redundancy: Redundancy weight from paper
    """
    tool_schemas: Mapping[str, ToolSchema]
    transition_model: TransitionModel
    observation_model: ObservationModel
    reward_model: RewardModel
    discount: float = 1.0
    max_steps: int = 6
    tau_exec: float = 0.85
    alpha: float = 0.1
    lambda_redundancy: float = 0.5

    def initial_state(
        self,
        ground_truth: Optional[ToolCall] = None,
    ) -> POMDPState:
        """Create initial POMDP state.

        Args:
            ground_truth: True user intent (for simulation)

        Returns:
            Initial state with full domains
        """
        domains = {
            name: dict(schema.parameters)
            for name, schema in self.tool_schemas.items()
        }
        return POMDPState(
            ground_truth=ground_truth,
            domains=domains,
            observations=(),
            time_step=0,
        )

    def initial_belief(self) -> BeliefState:
        """Create initial belief state with full domains."""
        domains = {
            name: dict(schema.parameters)
            for name, schema in self.tool_schemas.items()
        }
        return BeliefState(domains=domains)

    def is_terminal(self, state: POMDPState, last_action: Optional[Action]) -> bool:
        """Check if state is terminal."""
        if last_action is None:
            return False
        if last_action.action_type in (ActionType.EXECUTE, ActionType.ESCALATE):
            return True
        if state.time_step >= self.max_steps:
            return True
        return False


def create_sage_pomdp(
    tool_schemas: Iterable[ToolSchema],
    constraint_extractor: Any = None,
    **config,
) -> POMDP:
    """Create a SAGE-Agent POMDP with default models.

    Args:
        tool_schemas: Available tools
        constraint_extractor: For domain refinement transitions
        **config: Override default hyperparameters

    Returns:
        Configured POMDP instance
    """
    schemas = {t.name: t for t in tool_schemas}

    return POMDP(
        tool_schemas=schemas,
        transition_model=DomainRefinementTransition(constraint_extractor),
        observation_model=UniformObservationModel(),
        reward_model=SAGERewardModel(
            execution_reward=config.get("execution_reward", 1.0),
            query_cost=config.get("query_cost", 0.1),
            incorrect_penalty=config.get("incorrect_penalty", -1.0),
        ),
        discount=config.get("discount", 1.0),
        max_steps=config.get("max_steps", 6),
        tau_exec=config.get("tau_exec", 0.85),
        alpha=config.get("alpha", 0.1),
        lambda_redundancy=config.get("lambda_redundancy", 0.5),
    )
