"""SGR (Schema-Guided Reasoning) Tools.

Standalone implementations of SGR tools for structured planning and reasoning.
These replace the external sgr-agent-core dependency with equivalent functionality.

Tools:
- GeneratePlanTool: Generate structured research/task plans
- ReasoningTool: Step-by-step reasoning with state tracking
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GeneratePlanTool(BaseModel):
    """Generate a structured research or task plan.

    Useful to split complex requests into manageable steps with clear goals
    and search strategies.

    Example:
        plan = GeneratePlanTool(
            reasoning="User needs flight booking, requires origin/dest/date",
            research_goal="Book optimal flight based on user preferences",
            planned_steps=[
                "Clarify departure city",
                "Clarify travel date",
                "Search available flights",
                "Present options to user"
            ],
            search_strategies=[
                "Ask clarifying questions for missing info",
                "Use flight search API with confirmed parameters"
            ]
        )
    """

    reasoning: str = Field(
        description="Justification for the research/task approach"
    )
    research_goal: str = Field(
        description="Primary research or task objective"
    )
    planned_steps: List[str] = Field(
        description="List of 3-4 planned steps to achieve the goal",
        min_length=3,
        max_length=4,
    )
    search_strategies: List[str] = Field(
        default_factory=list,
        description="Information search or data gathering strategies",
        min_length=0,
        max_length=3,
    )

    def execute(self) -> str:
        """Execute the plan tool, returning JSON representation."""
        return self.model_dump_json(
            indent=2,
            exclude={"reasoning"},  # Exclude internal reasoning from output
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class ReasoningTool(BaseModel):
    """Agent reasoning tool for step-by-step thinking and adaptive planning.

    This tool helps structure the agent's reasoning process, tracking:
    - Current reasoning chain
    - Situation assessment
    - Plan status and remaining steps
    - Task completion state

    Usage: Use this tool before any other tool execution to structure thinking.

    Example:
        reasoning = ReasoningTool(
            reasoning_steps=[
                "User wants to book a flight but origin is missing",
                "Need to ask clarifying question about departure city"
            ],
            current_situation="User requested flight to NYC, departure unknown",
            plan_status="Step 1 of 3: Gathering required parameters",
            enough_data=False,
            remaining_steps=[
                "Ask for departure city",
                "Ask for travel date",
                "Execute flight search"
            ],
            task_completed=False
        )
    """

    # Reasoning chain - step-by-step thinking process
    reasoning_steps: List[str] = Field(
        description="Step-by-step reasoning (brief, 1 sentence each)",
        min_length=2,
        max_length=3,
    )

    # Situation assessment
    current_situation: str = Field(
        description="Current research/task situation (2-3 sentences MAX)",
        max_length=300,
    )
    plan_status: str = Field(
        description="Status of current plan (1 sentence)",
        max_length=150,
    )
    enough_data: bool = Field(
        default=False,
        description="Sufficient data collected for task completion?",
    )

    # Next step planning
    remaining_steps: List[str] = Field(
        description="1-3 remaining steps (brief, action-oriented)",
        min_length=1,
        max_length=3,
    )
    task_completed: bool = Field(
        default=False,
        description="Is the task/research finished?",
    )

    def execute(self) -> str:
        """Execute the reasoning tool, returning JSON representation."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    def is_complete(self) -> bool:
        """Check if task is marked as complete."""
        return self.task_completed

    def has_sufficient_data(self) -> bool:
        """Check if enough data has been collected."""
        return self.enough_data


class ClarificationTool(BaseModel):
    """Tool for generating clarification questions.

    Used when the agent needs more information from the user to proceed.

    Example:
        clarification = ClarificationTool(
            question="Which city will you be departing from?",
            reason="Origin city is required for flight booking",
            options=["New York", "Los Angeles", "Chicago", "Other"]
        )
    """

    question: str = Field(
        description="The clarification question to ask the user"
    )
    reason: str = Field(
        description="Why this information is needed"
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Optional list of suggested answers",
    )
    required: bool = Field(
        default=True,
        description="Whether this information is required to proceed",
    )

    def execute(self) -> str:
        """Execute the clarification tool, returning JSON representation."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class FinalAnswerTool(BaseModel):
    """Tool for providing the final answer/result.

    Used when the agent has completed its task and wants to present results.

    Example:
        answer = FinalAnswerTool(
            answer="I've booked flight AA123 from BOS to NYC on March 15th",
            confidence=0.95,
            sources=["Flight API response", "User confirmation"]
        )
    """

    answer: str = Field(
        description="The final answer or result"
    )
    confidence: float = Field(
        default=1.0,
        description="Confidence in the answer (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Sources or evidence supporting the answer",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the answer",
    )

    def execute(self) -> str:
        """Execute the final answer tool, returning JSON representation."""
        return self.model_dump_json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
