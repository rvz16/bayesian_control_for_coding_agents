"""SAGE-Agent: Structured Uncertainty Guided Clarification for LLM Agents.

This package implements the SAGE-Agent method from the paper
"Structured Uncertainty Guided Clarification for LLM Agents" (arXiv:2511.08798).

Key components:
    - SAGEAgent: Canonical Algorithm 1 implementation
    - SAGEConfig: Paper hyperparameters (τ=0.85, α=0.1, λ=0.5)
    - BeliefState: POMDP belief over tool call candidates
    - EVPI: Expected Value of Perfect Information for question selection

Example:
    ```python
    from sage_agent import SAGEAgent, SAGEConfig, ToolSchema, ParameterDomain

    tools = [ToolSchema(
        name="book_flight",
        parameters={"dest": ParameterDomain.from_values(["NYC", "LAX"])},
        required=frozenset(["dest"]),
    )]

    agent = SAGEAgent(
        tool_schemas=tools,
        candidate_generator=gen,
        question_generator=q_gen,
        question_asker=asker,
        tool_executor=executor,
    )

    result = agent.run("Book me a flight")
    ```
"""

from .core.agent import AgentResult, SageAgent, SageAgentConfig
from .core.sage_algorithm import (
    SAGEAgent,
    SAGEConfig,
    SAGEResult,
    SAGEStep,
    TerminationReason,
)
from .core.belief import BeliefState
from .core.evpi import compute_evpi, partition_candidates
from .core.bayesian_dp import (
    compute_question_value_dp,
    get_terminal_value,
    compute_question_cost,
    CostModel,
    DEFAULT_COST_MODEL,
    HIGH_STAKES_COST_MODEL,
    LOW_STAKES_COST_MODEL,
)
from .core.pomdp import (
    # POMDP Actions
    Action,
    ActionType as POMDPActionType,
    ExecuteAction,
    QueryAction,
    EscalateAction,
    # POMDP State and Observation
    POMDPState,
    Observation,
    # POMDP Models
    TransitionModel,
    DomainRefinementTransition,
    ObservationModel,
    UniformObservationModel,
    RewardModel,
    SAGERewardModel,
    # Value Function
    ValueEstimate,
    compute_value,
    # Complete POMDP
    POMDP,
    create_sage_pomdp,
)
from .core.constraints import (
    SimpleConstraintExtractor,
    LLMConstraintExtractor,
    HybridConstraintExtractor,
)
from .core.uncertainty_propagation import (
    UncertaintyPropagator,
    UncertaintyObservation,
    PropagationMode,
    create_sage_propagator,
)
from .core.advanced_reasoning import (
    DecomposedUncertainty,
    UncertaintyDecomposer,
    ChainOfThoughtVerifier,
    ChainVerificationResult,
    VerificationResult,
    ReflexionAgent,
    ReflexionResult,
    ReflectionMemory,
    create_enhanced_uncertainty_pipeline,
)
from .core.domains import ParameterDomain
from .tools.sgr_tools import (
    GeneratePlanTool,
    ReasoningTool,
    ClarificationTool,
    FinalAnswerTool,
)
from .sim.clarifybench import ClarifyBenchSimulator, SimulationResult, SimulationScenario

# GRPO requires torch - make import lazy/optional
try:
    from .training.grpo import (
        GRPOConfig,
        GRPOStepResult,
        GRPOTrainer,
        # SAGE-specific GRPO (new)
        ActionType,
        CertaintyWeightedReward,
        SAGEAction,
        SAGEGRPOConfig,
        SAGEGRPOTrainer,
    )
    _HAS_TORCH = True
except ImportError:
    GRPOConfig = None  # type: ignore
    GRPOStepResult = None  # type: ignore
    GRPOTrainer = None  # type: ignore
    ActionType = None  # type: ignore
    CertaintyWeightedReward = None  # type: ignore
    SAGEAction = None  # type: ignore
    SAGEGRPOConfig = None  # type: ignore
    SAGEGRPOTrainer = None  # type: ignore
    _HAS_TORCH = False

from .llm.llm import LLMClient

# Optional LLM clients
try:
    from .llm.openrouter import OpenRouterClient
except ImportError:
    OpenRouterClient = None  # type: ignore

try:
    from .llm.tts_service import TTSServiceClient
except ImportError:
    TTSServiceClient = None  # type: ignore

# Optional LangGraph integration
try:
    from .langgraph import SAGEGraph, SAGEGraphConfig, SAGEGraphState, build_sage_graph
    _HAS_LANGGRAPH = True
except ImportError:
    SAGEGraph = None  # type: ignore
    SAGEGraphConfig = None  # type: ignore
    SAGEGraphState = None  # type: ignore
    build_sage_graph = None  # type: ignore
    _HAS_LANGGRAPH = False
from .metrics.metrics import (
    MetricResult,
    CalibrationResult,
    ExtendedMetricResult,
    evaluate_metrics,
    evaluate_extended_metrics,
    compute_calibration,
    compute_uncertainty_aware_accuracy,
)
from .llm.prompts import build_candidate_prompt, build_question_prompt, tool_schemas_to_json
from .wiring.wiring import (
    LLMBackedCandidateGenerator,
    LLMBackedQuestionGenerator,
    ToolRegistryExecutor,
)
from .core.types import (
    Aspect,
    CandidateGenerator,
    ConstraintExtractor,
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

__all__ = [
    # Canonical SAGE-Agent (Algorithm 1)
    "SAGEAgent",
    "SAGEConfig",
    "SAGEResult",
    "SAGEStep",
    "TerminationReason",
    # Belief State
    "BeliefState",
    "compute_evpi",
    "partition_candidates",
    # Bayesian DP Planner
    "compute_question_value_dp",
    "get_terminal_value",
    # POMDP Formulation (Section 3)
    "Action",
    "POMDPActionType",
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
    # Legacy Agent (extended features)
    "AgentResult",
    "SageAgent",
    "SageAgentConfig",
    # Types
    "Aspect",
    "CandidateGenerator",
    "ConstraintExtractor",
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
    # Training (optional torch)
    "GRPOConfig",
    "GRPOStepResult",
    "GRPOTrainer",
    # SAGE-specific GRPO (Section 6.2)
    "ActionType",
    "CertaintyWeightedReward",
    "SAGEAction",
    "SAGEGRPOConfig",
    "SAGEGRPOTrainer",
    # Constraint Extraction
    "HybridConstraintExtractor",
    "LLMConstraintExtractor",
    "SimpleConstraintExtractor",
    # Wiring
    "LLMBackedCandidateGenerator",
    "LLMBackedQuestionGenerator",
    "ToolRegistryExecutor",
    # LLM Clients
    "LLMClient",
    "OpenRouterClient",
    "TTSServiceClient",
    "build_candidate_prompt",
    "build_question_prompt",
    "tool_schemas_to_json",
    # LangGraph (optional)
    "SAGEGraph",
    "SAGEGraphConfig",
    "SAGEGraphState",
    "build_sage_graph",
    # Metrics
    "MetricResult",
    "CalibrationResult",
    "ExtendedMetricResult",
    "compute_calibration",
    "compute_uncertainty_aware_accuracy",
    "evaluate_extended_metrics",
    "evaluate_metrics",
    # Domains
    "ParameterDomain",
    # Simulation
    "ClarifyBenchSimulator",
    "SimulationResult",
    "SimulationScenario",
    # Uncertainty Propagation (SAUP)
    "UncertaintyPropagator",
    "UncertaintyObservation",
    "PropagationMode",
    "create_sage_propagator",
    # Advanced Reasoning
    "DecomposedUncertainty",
    "UncertaintyDecomposer",
    "ChainOfThoughtVerifier",
    "ChainVerificationResult",
    "VerificationResult",
    "ReflexionAgent",
    "ReflexionResult",
    "ReflectionMemory",
    "create_enhanced_uncertainty_pipeline",
    # SGR Tools (Schema-Guided Reasoning)
    "GeneratePlanTool",
    "ReasoningTool",
    "ClarificationTool",
    "FinalAnswerTool",
]
