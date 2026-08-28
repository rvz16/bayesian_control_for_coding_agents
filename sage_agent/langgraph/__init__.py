"""LangGraph integration for SAGE-Agent.

This module provides LangGraph-based implementations of the SAGE-Agent
decision loop from the paper "Structured Uncertainty Guided Clarification
for LLM Agents".

Example:
    ```python
    from sage_agent.langgraph import SAGEGraph, SAGEGraphConfig

    graph = SAGEGraph(
        tool_schemas=[booking_tool],
        candidate_generator=llm_gen,
        question_generator=q_gen,
        question_asker=user_interface,
        tool_executor=executor,
    )

    result = graph.run("Book a flight to NYC")
    ```
"""

from .sage_graph import (
    SAGEGraph,
    SAGEGraphConfig,
    SAGEGraphState,
    build_sage_graph,
)

__all__ = [
    "SAGEGraph",
    "SAGEGraphConfig",
    "SAGEGraphState",
    "build_sage_graph",
]
