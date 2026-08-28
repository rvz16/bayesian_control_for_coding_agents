"""Belief state for SAGE-Agent POMDP formulation.

This module implements the belief state from Section 3.2 of the paper:
"Structured Uncertainty Guided Clarification for LLM Agents"

The key insight is that we maintain beliefs over the structured candidate space
rather than unstructured text sequences, enabling precise uncertainty quantification.

Definition 3 (Belief State): At time t, we maintain B(t) = {(c_i, Π_i(t)) : c_i ∈ C}
where Π_i(t) ∈ [0, 1] represents candidate viability.

Candidate viability is computed as:
    Π_i(t) ∝ p(θ_i | T_i, observations_t) = ∏_j p(θ_{i,j} | T_i, observations_t)

Where parameter certainty p(θ_{i,j} | T_i, obs_t) is:
    - 1 if specified (value is known)
    - |D_{i,j}(t)|^{-1} if unspecified with finite domain
    - ε (small value ~1e-4) if domain is infinite/continuous

CRITICAL: The domains used for computation must be the BELIEF DOMAINS (self.domains)
which are updated based on user responses, NOT the original schema domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

from .domains import ParameterDomain
from .types import ToolCallCandidate, ToolSchema, UNK


@dataclass
class BeliefState:
    """Belief state maintaining domains refined by observations.

    Attributes:
        domains: Current belief about parameter domains per tool.
                 These are updated as user responses constrain valid values.
        epsilon: Small probability for infinite/continuous domains (default: 1e-4).
    """
    domains: Dict[str, Dict[str, ParameterDomain]]
    epsilon: float = 1e-4

    def candidate_weight(
        self, candidate: ToolCallCandidate, tool_schema: ToolSchema
    ) -> float:
        """Compute candidate viability weight per Section 3.2.

        Π_i(t) ∝ ∏_j p(θ_{i,j} | T_i, observations_t)

        Where:
        - p(θ) = 1.0 if value is specified (not UNK)
        - p(θ) = 1 / |D_{i,j}(t)| if unspecified with finite domain
        - p(θ) = ε if domain is infinite/continuous

        Args:
            candidate: Tool call candidate with potentially unspecified arguments
            tool_schema: Schema defining the tool's parameters and requirements

        Returns:
            Unnormalized weight representing candidate viability
        """
        # CRITICAL: Use belief domains (updated from observations), not original schema
        tool_domains = self.domains.get(tool_schema.name, {})
        weight = 1.0

        for param_name in tool_schema.parameters:
            # Get current domain from belief state (with updates from user responses)
            # Fall back to schema domain if not in belief state
            domain = tool_domains.get(param_name, tool_schema.parameters[param_name])
            value = candidate.arguments.get(param_name, UNK)

            if value != UNK:
                # Specified value: check validity, p(θ) = 1 if valid, 0 if invalid
                if not domain.contains(value):
                    return 0.0
                weight *= 1.0
            else:
                # Unspecified value: weight by inverse domain size
                size = domain.size()
                if size is None:
                    # Continuous/infinite domain: use small epsilon
                    weight *= self.epsilon
                elif size == 0:
                    # Empty domain: impossible candidate
                    return 0.0
                else:
                    # Finite domain: uniform probability 1/|D|
                    weight *= 1.0 / float(size)

        return weight

    def normalize(self, weights: Iterable[float]) -> list[float]:
        weights_list = list(weights)
        total = sum(weights_list)
        if total <= 0:
            if not weights_list:
                return []
            uniform = 1.0 / float(len(weights_list))
            return [uniform for _ in weights_list]
        return [w / total for w in weights_list]
