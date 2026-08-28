"""Core components for SAGE-Agent.

This module contains the fundamental building blocks:
- POMDP formulation (pomdp.py)
- Belief state (belief.py)
- EVPI computation (evpi.py)
- Algorithm 1 implementation (sage_algorithm.py)
- Domain definitions (domains.py)
- Type definitions (types.py)
"""

from .types import (
    Aspect,
    CandidateGenerator,
    ConstraintExtractor,
    DomainRefiner,
    ErrorQuestionGenerator,
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

from .domains import ParameterDomain

from .belief import BeliefState

from .evpi import compute_evpi, partition_candidates

from .sage_algorithm import (
    SAGEAgent,
    SAGEConfig,
    SAGEResult,
    SAGEStep,
    TerminationReason,
)

from .pomdp import (
    # Action types
    Action,
    ActionType,
    ExecuteAction,
    QueryAction,
    EscalateAction,
    # State and observation
    POMDPState,
    Observation,
    # Models
    TransitionModel,
    DomainRefinementTransition,
    ObservationModel,
    UniformObservationModel,
    RewardModel,
    SAGERewardModel,
    # Value function
    ValueEstimate,
    compute_value,
    # Full POMDP
    POMDP,
    create_sage_pomdp,
)

from .constraints import (
    SimpleConstraintExtractor,
    LLMConstraintExtractor,
    HybridConstraintExtractor,
)

__all__ = [
    # Types
    "Aspect",
    "CandidateGenerator",
    "ConstraintExtractor",
    "DomainRefiner",
    "ErrorQuestionGenerator",
    "ExecutionResult",
    "Question",
    "QuestionAsker",
    "QuestionGenerator",
    "ToolCall",
    "ToolCallCandidate",
    "ToolExecutor",
    "ToolSchema",
    "UNK",
    # Domains
    "ParameterDomain",
    # Belief
    "BeliefState",
    # EVPI
    "compute_evpi",
    "partition_candidates",
    # Algorithm 1
    "SAGEAgent",
    "SAGEConfig",
    "SAGEResult",
    "SAGEStep",
    "TerminationReason",
    # POMDP
    "Action",
    "ActionType",
    "ExecuteAction",
    "QueryAction",
    "EscalateAction",
    "POMDPState",
    "Observation",
    "TransitionModel",
    "DomainRefinementTransition",
    "ObservationModel",
    "UniformObservationModel",
    "RewardModel",
    "SAGERewardModel",
    "ValueEstimate",
    "compute_value",
    "POMDP",
    "create_sage_pomdp",
    # Constraints
    "SimpleConstraintExtractor",
    "LLMConstraintExtractor",
    "HybridConstraintExtractor",
]
