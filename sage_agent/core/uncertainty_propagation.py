"""Uncertainty propagation for multi-step LLM agent reasoning.

This module implements uncertainty propagation inspired by SAUP 
(Semantic-Aware Uncertainty Propagation) from:
"Uncertainty Propagation on LLM Agent" (ACL 2025)

The key insight is that uncertainty should propagate through multi-step
reasoning chains, accumulating and potentially amplifying across steps.

For SAGE-Agent, this means:
1. Candidate generation uncertainty affects question selection
2. Question answering uncertainty affects domain refinement
3. Multiple clarification rounds compound uncertainty

Usage:
    from sage_agent.core.uncertainty_propagation import UncertaintyPropagator
    
    propagator = UncertaintyPropagator(
        base_uncertainty=0.1,
        propagation_mode="multiplicative"
    )
    
    # Update after each step
    propagator.observe(step_uncertainty=0.3, step_type="candidate_generation")
    propagator.observe(step_uncertainty=0.2, step_type="question_answering")
    
    # Get accumulated uncertainty for decision making
    total_uncertainty = propagator.accumulated_uncertainty
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PropagationMode(Enum):
    """Mode for propagating uncertainty across steps."""
    
    # U_total = 1 - (1 - u1)(1 - u2)... (independent errors compound)
    MULTIPLICATIVE = "multiplicative"
    
    # U_total = max(u1, u2, ...) (worst case dominates)
    MAX = "max"
    
    # U_total = (u1 + u2 + ...) / n (simple average)
    AVERAGE = "average"
    
    # U_total = w1*u1 + w2*u2 + ... (weighted by recency)
    RECENCY_WEIGHTED = "recency_weighted"
    
    # Bayesian update treating uncertainties as error probabilities
    BAYESIAN = "bayesian"


@dataclass
class UncertaintyObservation:
    """A single uncertainty observation from one reasoning step."""
    uncertainty: float
    step_type: str
    step_index: int
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class UncertaintyPropagator:
    """Propagates uncertainty across multi-step agent reasoning.
    
    This class tracks uncertainty observations from each step in the
    agent's reasoning chain and computes accumulated uncertainty for
    decision making.
    
    Attributes:
        base_uncertainty: Prior uncertainty before any observations
        propagation_mode: How to combine uncertainties across steps
        decay_factor: For recency weighting, how much older observations decay
        step_type_weights: Optional weights for different step types
    """
    
    base_uncertainty: float = 0.1
    propagation_mode: PropagationMode = PropagationMode.MULTIPLICATIVE
    decay_factor: float = 0.9
    step_type_weights: Dict[str, float] = field(default_factory=dict)
    _observations: List[UncertaintyObservation] = field(default_factory=list)
    _step_counter: int = 0
    
    def observe(
        self,
        step_uncertainty: float,
        step_type: str = "generic",
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Record an uncertainty observation from a reasoning step.
        
        Args:
            step_uncertainty: Uncertainty from this step (0=certain, 1=uncertain)
            step_type: Type of step (e.g., "candidate_generation", "question_answering")
            metadata: Optional additional information about the step
        """
        observation = UncertaintyObservation(
            uncertainty=max(0.0, min(1.0, step_uncertainty)),
            step_type=step_type,
            step_index=self._step_counter,
            metadata=metadata or {},
        )
        self._observations.append(observation)
        self._step_counter += 1
    
    def reset(self) -> None:
        """Clear all observations and reset state."""
        self._observations.clear()
        self._step_counter = 0
    
    @property
    def accumulated_uncertainty(self) -> float:
        """Compute accumulated uncertainty across all observations.
        
        Returns:
            Combined uncertainty based on propagation mode
        """
        if not self._observations:
            return self.base_uncertainty
        
        uncertainties = [obs.uncertainty for obs in self._observations]
        
        if self.propagation_mode == PropagationMode.MULTIPLICATIVE:
            return self._multiplicative_propagation(uncertainties)
        elif self.propagation_mode == PropagationMode.MAX:
            return max(uncertainties)
        elif self.propagation_mode == PropagationMode.AVERAGE:
            return sum(uncertainties) / len(uncertainties)
        elif self.propagation_mode == PropagationMode.RECENCY_WEIGHTED:
            return self._recency_weighted_propagation()
        elif self.propagation_mode == PropagationMode.BAYESIAN:
            return self._bayesian_propagation(uncertainties)
        else:
            return self._multiplicative_propagation(uncertainties)
    
    def _multiplicative_propagation(self, uncertainties: List[float]) -> float:
        """Compute uncertainty assuming independent error probabilities.
        
        If each step has uncertainty u_i (probability of error), then
        the probability of at least one error is:
        1 - (1-u1)(1-u2)...(1-un)
        """
        certainty = 1.0 - self.base_uncertainty
        for u in uncertainties:
            certainty *= (1.0 - u)
        return 1.0 - certainty
    
    def _recency_weighted_propagation(self) -> float:
        """Weight recent observations more heavily."""
        if not self._observations:
            return self.base_uncertainty
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        n = len(self._observations)
        for i, obs in enumerate(self._observations):
            # More recent observations (higher index) get higher weight
            recency_weight = self.decay_factor ** (n - 1 - i)
            
            # Apply step type weight if available
            type_weight = self.step_type_weights.get(obs.step_type, 1.0)
            weight = recency_weight * type_weight
            
            weighted_sum += weight * obs.uncertainty
            total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        return self.base_uncertainty
    
    def _bayesian_propagation(self, uncertainties: List[float]) -> float:
        """Bayesian update treating each observation as evidence.
        
        Uses a simple beta-like update where high uncertainty observations
        increase the posterior uncertainty estimate.
        """
        # Start with prior
        alpha = 1.0  # pseudo-counts for "correct"
        beta = self.base_uncertainty / (1.0 - self.base_uncertainty + 1e-10)
        
        for u in uncertainties:
            # Treat uncertainty as evidence of error
            alpha += (1.0 - u)
            beta += u
        
        # Posterior mean of beta distribution
        return beta / (alpha + beta)
    
    @property
    def step_uncertainties(self) -> List[Tuple[str, float]]:
        """Get list of (step_type, uncertainty) for all observations."""
        return [(obs.step_type, obs.uncertainty) for obs in self._observations]
    
    @property
    def num_steps(self) -> int:
        """Number of observations recorded."""
        return len(self._observations)
    
    def get_uncertainty_breakdown(self) -> Dict[str, float]:
        """Get average uncertainty per step type.
        
        Returns:
            Dict mapping step_type to average uncertainty
        """
        by_type: Dict[str, List[float]] = {}
        for obs in self._observations:
            by_type.setdefault(obs.step_type, []).append(obs.uncertainty)
        
        return {
            step_type: sum(uncertainties) / len(uncertainties)
            for step_type, uncertainties in by_type.items()
        }
    
    def should_escalate(
        self,
        escalation_threshold: float = 0.8,
        max_high_uncertainty_steps: int = 3,
        high_uncertainty_threshold: float = 0.6,
    ) -> bool:
        """Determine if the agent should escalate to human.
        
        Uses multiple criteria:
        1. Accumulated uncertainty exceeds threshold
        2. Too many high-uncertainty steps in a row
        
        Args:
            escalation_threshold: Accumulated uncertainty threshold
            max_high_uncertainty_steps: Max consecutive high-uncertainty steps
            high_uncertainty_threshold: What counts as "high uncertainty"
            
        Returns:
            True if escalation is recommended
        """
        # Check accumulated uncertainty
        if self.accumulated_uncertainty >= escalation_threshold:
            return True
        
        # Check for consecutive high-uncertainty steps
        consecutive_high = 0
        for obs in reversed(self._observations):
            if obs.uncertainty >= high_uncertainty_threshold:
                consecutive_high += 1
                if consecutive_high >= max_high_uncertainty_steps:
                    return True
            else:
                break
        
        return False


def create_sage_propagator(
    structured_weight: float = 0.7,
    llm_weight: float = 0.3,
) -> UncertaintyPropagator:
    """Create a propagator configured for SAGE-Agent.
    
    Uses multiplicative propagation with step-type weights that
    prioritize structured uncertainty from the POMDP belief state.
    
    Args:
        structured_weight: Weight for structured/belief uncertainty
        llm_weight: Weight for LLM/sampling uncertainty
        
    Returns:
        Configured UncertaintyPropagator
    """
    return UncertaintyPropagator(
        base_uncertainty=0.1,
        propagation_mode=PropagationMode.RECENCY_WEIGHTED,
        decay_factor=0.85,
        step_type_weights={
            "candidate_generation": structured_weight,
            "question_generation": structured_weight,
            "belief_update": structured_weight,
            "llm_parsing": llm_weight,
            "constraint_extraction": llm_weight,
        },
    )

