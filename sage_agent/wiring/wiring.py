from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping

from ..llm.llm import LLMClient, PromptCandidateGenerator, PromptQuestionGenerator
from ..core.types import (
    CandidateGenerator,
    ExecutionResult,
    Question,
    QuestionGenerator,
    ToolCall,
    ToolCallCandidate,
    ToolExecutor,
    ToolSchema,
)


@dataclass
class LLMBackedCandidateGenerator(CandidateGenerator):
    llm: LLMClient

    def generate_candidates(
        self,
        user_input: str,
        observations: Iterable[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[ToolCallCandidate]:
        generator = PromptCandidateGenerator(self.llm)
        return generator.generate(user_input, observations, tool_schemas.values())


@dataclass
class LLMBackedQuestionGenerator(QuestionGenerator):
    llm: LLMClient

    def generate_questions(
        self,
        user_input: str,
        candidates: Iterable[ToolCallCandidate],
        observations: Iterable[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[Question]:
        generator = PromptQuestionGenerator(self.llm)
        return generator.generate(user_input, observations, tool_schemas.values(), candidates)


@dataclass
class ToolRegistryExecutor(ToolExecutor):
    registry: Mapping[str, Callable[[Mapping[str, object]], object]]

    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        func = self.registry.get(tool_call.tool_name)
        if func is None:
            return ExecutionResult(success=False, error="Unknown tool")
        try:
            output = func(tool_call.arguments)
        except Exception as exc:
            return ExecutionResult(success=False, error=str(exc))
        return ExecutionResult(success=True, output=output)
