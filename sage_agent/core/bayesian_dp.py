"""Bayesian Dynamic Programming planner for SAGE-Agent question selection.

Adapts the multi-step lookahead approach from the Bayesian diagnostic controller
(article_implementation/bayesian/example.py) to the SAGE candidate/question framework.

Instead of myopic EVPI (one-step-ahead), this planner uses DP recursion to consider
the full remaining question budget when selecting which question to ask. At each step,
it simulates both outcome branches (answer narrows domain vs. doesn't) and recurses
to find the question that maximizes expected value over all remaining steps.

Key mapping from example.py -> SAGE:
    - Hypotheses H -> Tool-call candidates with belief weights
    - Diagnostic tests T -> Clarification questions targeting aspects
    - P(fail|H,T) -> Domain-based: P(candidate compatible with narrowed domain)
    - Binary outcome -> "answer resolves aspect" vs "answer doesn't resolve"
    - Budget -> max_questions - current_step
    - C_TEST -> Per-question cost (varies by question complexity)
    - C_PATCH + C_VER -> Execution cost (risk of wrong tool call)
    - R -> Reward for correct execution

Cost model (analogous to example.py's C_TEST=1, C_PATCH=3, C_VER=20, R=200):

    In example.py all tests cost the same (C_TEST=1). In reality, different
    clarification questions have different costs to the user:

    - Simple single-aspect question ("From which city?") is cheap
    - Multi-aspect question ("Where from, when, which class?") is expensive
    - Questions about open-ended (infinite) domains are harder to answer
    - Repeated questions about the same aspect are frustrating

    Similarly, execution has a cost (C_PATCH + C_VER in example.py):
    - Wrong tool call with side effects (booking, payment) is very costly
    - Wrong read-only tool call (search, lookup) is cheap to retry
    - Higher confidence → lower expected execution cost
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .belief import BeliefState
from .domains import ParameterDomain
from .evpi import compute_evpi
from .types import Aspect, Question, ToolCallCandidate, ToolSchema, UNK


@dataclass
class CostModel:
    """Cost model for the DP planner, mapping example.py's constants to SAGE.

    example.py uses flat costs:
        C_TEST = 1, C_PATCH = 3, C_VER = 20, R = 200

    Here we differentiate costs by question complexity and execution risk:
        - query_base_cost: minimum cost per question (like C_TEST)
        - query_aspect_cost: additional cost per aspect in the question
        - query_open_domain_cost: extra cost when aspect has infinite domain
        - execution_cost: cost of executing (like C_PATCH + C_VER)
        - reward: reward for correct execution (like R)
    """
    # Question costs (analogous to C_TEST = 1 in example.py)
    query_base_cost: float = 0.02       # simple yes/no or single-value question
    query_aspect_cost: float = 0.03     # per additional aspect in the question
    query_open_domain_cost: float = 0.02  # extra for open-ended (infinite) domains

    # Execution costs (analogous to C_PATCH=3 + C_VER=20 in example.py)
    execution_cost: float = 0.15        # risk cost of executing a tool call

    # Reward (analogous to R=200 in example.py)
    reward: float = 1.0                 # reward for correct execution


# Default cost model preserving backward-compatible behavior
DEFAULT_COST_MODEL = CostModel()

# Preset: high-stakes actions (booking, payment, deletion)
HIGH_STAKES_COST_MODEL = CostModel(
    query_base_cost=0.01,        # questions are cheap compared to mistakes
    query_aspect_cost=0.02,
    query_open_domain_cost=0.01,
    execution_cost=0.40,         # wrong execution is very costly
    reward=1.0,
)

# Preset: low-stakes actions (search, lookup, info retrieval)
LOW_STAKES_COST_MODEL = CostModel(
    query_base_cost=0.05,        # questions are relatively more expensive
    query_aspect_cost=0.04,
    query_open_domain_cost=0.03,
    execution_cost=0.03,         # wrong execution is cheap to retry
    reward=1.0,
)


def compute_question_cost(
    question: Question,
    tool_schemas: Mapping[str, ToolSchema],
    cost_model: CostModel,
) -> float:
    """Compute the cost of asking a specific question.

    Analogous to C_TEST in example.py, but varies by question complexity:
    - More aspects = more complex = more costly to user
    - Open-ended (infinite) domains are harder to answer

    Args:
        question: The question to cost.
        tool_schemas: Tool schemas for domain lookup.
        cost_model: Cost parameters.

    Returns:
        Cost of asking this question (always positive).
    """
    n_aspects = len(question.aspects)
    cost = cost_model.query_base_cost + cost_model.query_aspect_cost * max(0, n_aspects - 1)

    # Extra cost for open-ended domains (user must type free-form answer)
    for aspect in question.aspects:
        schema = tool_schemas.get(aspect.tool_name)
        if schema is None:
            continue
        domain = schema.parameters.get(aspect.param_name)
        if domain is not None and not domain.is_finite():
            cost += cost_model.query_open_domain_cost

    return cost


def get_terminal_value(
    probabilities: Sequence[float],
    cost_model: CostModel | None = None,
) -> float:
    """Compute the expected value of executing the best candidate now.

    Analogous to example.py's get_terminal_value(belief):
        V_term(b) = R * max(b) - C_PATCH - C_VER

    In example.py this is: 200 * max(belief) - 3 - 20 = 200*max(b) - 23
    Here: reward * max(probs) - execution_cost

    The execution_cost represents the risk of executing the wrong tool call.
    Higher confidence → higher expected reward, lower effective risk.

    Args:
        probabilities: Normalized probability distribution over candidates.
        cost_model: Cost model parameters. If None, uses backward-compatible
            behavior (just returns max(probabilities)).

    Returns:
        Terminal value = reward * max(probs) - execution_cost.
    """
    if not probabilities:
        return 0.0

    max_prob = max(probabilities)

    if cost_model is None:
        return max_prob

    return cost_model.reward * max_prob - cost_model.execution_cost


def _simulate_question_outcome(
    belief: BeliefState,
    candidates: Sequence[ToolCallCandidate],
    question: Question,
    tool_schemas: Mapping[str, ToolSchema],
    outcome_resolves: bool,
) -> Tuple[BeliefState, List[float]]:
    """Simulate the effect of a question outcome on the belief state.

    For each aspect in the question, we simulate two branches:
    - outcome_resolves=True: The answer narrows the domain to match the best
      candidate's value for that aspect (like "fail" in example.py confirming
      a hypothesis).
    - outcome_resolves=False: The answer doesn't narrow the domain (like "pass"
      meaning the hypothesis wasn't confirmed).

    Args:
        belief: Current belief state.
        candidates: Current candidates.
        question: The question being evaluated.
        tool_schemas: Tool schema definitions.
        outcome_resolves: Whether the answer resolves the queried aspect.

    Returns:
        (new_belief, new_probabilities) after the simulated update.
    """
    new_belief = BeliefState(
        domains=copy.deepcopy(belief.domains),
        epsilon=belief.epsilon,
    )

    if outcome_resolves:
        # Simulate: the answer narrows the domain to a single value
        # Use the best candidate's value as the "resolved" value
        for aspect in question.aspects:
            tool_name = aspect.tool_name
            param_name = aspect.param_name
            if tool_name not in new_belief.domains:
                continue

            current_domain = new_belief.domains[tool_name].get(param_name)
            if current_domain is None or not current_domain.is_finite():
                continue

            # Find the most common value among candidates for this aspect
            value_counts: Dict[object, float] = {}
            weights = [
                belief.candidate_weight(c, tool_schemas[c.tool_name])
                for c in candidates
                if c.tool_name in tool_schemas
            ]
            total_w = sum(weights)
            if total_w <= 0:
                continue

            for c, w in zip(candidates, weights):
                if c.tool_name != tool_name:
                    continue
                val = c.arguments.get(param_name, UNK)
                if val != UNK and current_domain.contains(val):
                    value_counts[val] = value_counts.get(val, 0.0) + w / total_w

            if value_counts:
                # Narrow to the most likely value
                best_val = max(value_counts, key=lambda v: value_counts[v])
                new_belief.domains[tool_name][param_name] = ParameterDomain.from_values(
                    [best_val]
                )
    # else: outcome_resolves=False -> domain stays the same (no information gained)

    # Recompute probabilities under new belief
    new_weights = [
        new_belief.candidate_weight(c, tool_schemas[c.tool_name])
        for c in candidates
        if c.tool_name in tool_schemas
    ]
    new_probs = new_belief.normalize(new_weights)

    return new_belief, new_probs


def _compute_resolution_probability(
    belief: BeliefState,
    candidates: Sequence[ToolCallCandidate],
    question: Question,
    tool_schemas: Mapping[str, ToolSchema],
) -> float:
    """Estimate P(question resolves the aspect) given current belief.

    Analogous to P(Z=fail | b, T) in example.py:
        P(resolve) = sum_H P(resolve | H) * b(H)

    A question "resolves" when the user's answer narrows the domain to a single
    value. The probability depends on how concentrated the belief already is
    over the queried aspect.

    Args:
        belief: Current belief state.
        candidates: Current candidates.
        question: The question being evaluated.
        tool_schemas: Tool schema definitions.

    Returns:
        Estimated probability that the question resolves the aspect.
    """
    weights = [
        belief.candidate_weight(c, tool_schemas[c.tool_name])
        for c in candidates
        if c.tool_name in tool_schemas
    ]
    total = sum(weights)
    if total <= 0:
        return 0.5  # Uniform prior

    probs = [w / total for w in weights]

    # For each aspect, compute how concentrated the candidates are
    resolution_probs = []
    for aspect in question.aspects:
        # Group candidate probability by their value for this aspect
        value_mass: Dict[object, float] = {}
        for c, p in zip(candidates, probs):
            if c.tool_name != aspect.tool_name:
                continue
            val = c.arguments.get(aspect.param_name, UNK)
            if val != UNK:
                value_mass[val] = value_mass.get(val, 0.0) + p

        if value_mass:
            # P(resolve) ~ max concentration = probability of most likely value
            # If one value dominates, the answer will likely match it
            resolution_probs.append(max(value_mass.values()))
        else:
            resolution_probs.append(0.5)

    return sum(resolution_probs) / len(resolution_probs) if resolution_probs else 0.5


def _dp_value(
    belief: BeliefState,
    candidates: Sequence[ToolCallCandidate],
    probabilities: Sequence[float],
    questions: Sequence[Question],
    tool_schemas: Mapping[str, ToolSchema],
    budget: int,
    cost_model: CostModel,
) -> float:
    """Recursive DP value computation.

    Analogous to example.py's get_optimal_policy_value(belief, depth):
        V(b, d) = max over questions q of:
            -C_q + P(resolve|b,q) * V(b_resolve, d-1)
                 + P(~resolve|b,q) * V(b_no_resolve, d-1)

    Unlike example.py where C_TEST is the same for all tests, here each
    question has its own cost computed by compute_question_cost():
    simple single-aspect questions are cheap, multi-aspect or open-domain
    questions are expensive.

    Args:
        belief: Current belief state.
        candidates: Current candidates.
        probabilities: Current probability distribution.
        questions: Available questions to ask.
        tool_schemas: Tool schema definitions.
        budget: Remaining question budget.
        cost_model: Cost model with per-question and execution costs.

    Returns:
        Expected value of the optimal policy from this state.
    """
    terminal_val = get_terminal_value(probabilities, cost_model)

    if budget <= 0 or not questions:
        return terminal_val

    best_value = terminal_val  # Can always choose to execute now

    for question in questions:
        # Per-question cost (unlike flat C_TEST=1 in example.py)
        q_cost = compute_question_cost(question, tool_schemas, cost_model)

        # P(resolve | b, question) - analogous to P(fail | b, T) in example.py
        p_resolve = _compute_resolution_probability(
            belief, candidates, question, tool_schemas
        )
        p_no_resolve = 1.0 - p_resolve

        # Simulate both outcomes - analogous to computing b_fail, b_pass
        belief_resolve, probs_resolve = _simulate_question_outcome(
            belief, candidates, question, tool_schemas, outcome_resolves=True
        )
        belief_no_resolve, probs_no_resolve = _simulate_question_outcome(
            belief, candidates, question, tool_schemas, outcome_resolves=False
        )

        # Recurse - analogous to get_optimal_policy_value(b_fail, depth-1)
        v_resolve = _dp_value(
            belief_resolve, candidates, probs_resolve,
            questions, tool_schemas, budget - 1, cost_model,
        )
        v_no_resolve = _dp_value(
            belief_no_resolve, candidates, probs_no_resolve,
            questions, tool_schemas, budget - 1, cost_model,
        )

        # Bellman equation: -C_q + E[V_next]
        expected_value = -q_cost + (p_resolve * v_resolve + p_no_resolve * v_no_resolve)

        if expected_value > best_value:
            best_value = expected_value

    return best_value


def compute_question_value_dp(
    belief: BeliefState,
    candidates: Sequence[ToolCallCandidate],
    probabilities: Sequence[float],
    questions: Sequence[Question],
    tool_schemas: Mapping[str, ToolSchema],
    budget: int,
    aspect_counts: Optional[Mapping[Aspect, int]] = None,
    redundancy_weight: float = 0.5,
    query_cost: float = 0.05,
    cost_model: Optional[CostModel] = None,
) -> List[Tuple[float, Question]]:
    """Score questions using DP lookahead over the remaining budget.

    This is the main entry point, replacing the myopic EVPI scoring loop
    in SageAgent.run(). For each question, it computes the expected value
    considering the full remaining question budget via DP recursion.

    Cost model (maps to example.py):
        example.py:  V = -C_TEST + E[V_next],  V_term = R*max(b) - C_PATCH - C_VER
        here:        V = -C_q    + E[V_next],  V_term = R*max(p) - C_exec

        C_q varies per question (simple=cheap, multi-aspect=expensive),
        unlike example.py's flat C_TEST=1 for all diagnostic tests.

    Args:
        belief: Current belief state.
        candidates: Current tool call candidates.
        probabilities: Current probability distribution over candidates.
        questions: Available clarification questions.
        tool_schemas: Tool schema definitions.
        budget: Remaining question budget (max_questions - current_step).
        aspect_counts: How many times each aspect has been queried (for redundancy).
        redundancy_weight: Weight for redundancy cost penalty.
        query_cost: Flat cost per question (used only if cost_model is None,
            for backward compatibility).
        cost_model: Differentiated cost model. If provided, overrides query_cost
            with per-question costs and adds execution cost to terminal value.
            Use HIGH_STAKES_COST_MODEL for actions with side effects (booking,
            payment) or LOW_STAKES_COST_MODEL for read-only actions (search).

    Returns:
        List of (score, question) tuples, sorted descending by score.
        Compatible with the scoring interface in SageAgent.run().
    """
    if not questions or not candidates:
        return []

    if aspect_counts is None:
        aspect_counts = {}

    # Build cost model: use provided, or construct from flat query_cost
    if cost_model is None:
        cost_model = CostModel(
            query_base_cost=query_cost,
            query_aspect_cost=0.0,
            query_open_domain_cost=0.0,
            execution_cost=0.0,
            reward=1.0,
        )

    scored: List[Tuple[float, Question]] = []

    for question in questions:
        # Per-question cost (varies by complexity, unlike flat C_TEST in example.py)
        q_cost = compute_question_cost(question, tool_schemas, cost_model)

        # P(resolve | b, question)
        p_resolve = _compute_resolution_probability(
            belief, candidates, question, tool_schemas
        )
        p_no_resolve = 1.0 - p_resolve

        # Simulate outcomes
        belief_resolve, probs_resolve = _simulate_question_outcome(
            belief, candidates, question, tool_schemas, outcome_resolves=True
        )
        belief_no_resolve, probs_no_resolve = _simulate_question_outcome(
            belief, candidates, question, tool_schemas, outcome_resolves=False
        )

        # DP value of each branch (with budget-1 remaining)
        v_resolve = _dp_value(
            belief_resolve, candidates, probs_resolve,
            questions, tool_schemas, budget - 1, cost_model,
        )
        v_no_resolve = _dp_value(
            belief_no_resolve, candidates, probs_no_resolve,
            questions, tool_schemas, budget - 1, cost_model,
        )

        # Expected value of asking this question
        # Bellman: V(ask q) = -C_q + P(resolve)*V(resolve) + P(~resolve)*V(~resolve)
        dp_value = -q_cost + (p_resolve * v_resolve + p_no_resolve * v_no_resolve)

        # Subtract current terminal value to get marginal improvement
        current_terminal = get_terminal_value(probabilities, cost_model)
        marginal_value = dp_value - current_terminal

        # Apply redundancy penalty (same as original SAGE)
        redundancy_cost = 0.0
        for aspect in question.aspects:
            redundancy_cost += float(aspect_counts.get(aspect, 0))
        redundancy_cost *= redundancy_weight

        score = marginal_value - redundancy_cost
        scored.append((score, question))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored
