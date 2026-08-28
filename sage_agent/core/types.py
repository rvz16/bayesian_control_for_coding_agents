from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol

from .domains import ParameterDomain


UNK = "<UNK>"


@dataclass(frozen=True)
class Aspect:
    tool_name: str
    param_name: str


@dataclass(frozen=True)
class Question:
    text: str
    aspects: tuple[Aspect, ...]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSchema:
    name: str
    parameters: Mapping[str, ParameterDomain]
    required: frozenset[str]
    domain_refiner: Optional["DomainRefiner"] = None

    def is_required(self, param_name: str) -> bool:
        return param_name in self.required

    def validate_call(self, arguments: Mapping[str, object]) -> None:
        missing = [p for p in self.required if p not in arguments or arguments[p] == UNK]
        if missing:
            raise ValueError(f"Missing required parameters for tool {self.name}: {missing}")


@dataclass(frozen=True)
class ToolCallCandidate:
    tool_name: str
    arguments: Mapping[str, object]

    def argument_value(self, param_name: str) -> object:
        return self.arguments.get(param_name, UNK)


@dataclass(frozen=True)
class ToolCall:
    tool_name: str
    arguments: Mapping[str, object]


class DomainRefiner(Protocol):
    def refine(
        self,
        tool: ToolSchema,
        domains: Mapping[str, ParameterDomain],
        assignments: Mapping[str, object],
    ) -> Mapping[str, ParameterDomain]:
        ...


class CandidateGenerator(Protocol):
    def generate_candidates(
        self,
        user_input: str,
        observations: Iterable[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[ToolCallCandidate]:
        ...


class QuestionGenerator(Protocol):
    def generate_questions(
        self,
        user_input: str,
        candidates: Iterable[ToolCallCandidate],
        observations: Iterable[str],
        tool_schemas: Mapping[str, ToolSchema],
    ) -> List[Question]:
        ...


class QuestionAsker(Protocol):
    def ask(self, question: Question) -> str:
        ...


class ConstraintExtractor(Protocol):
    def update_domain(self, domain: ParameterDomain, response: str) -> ParameterDomain:
        ...


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    output: Optional[object] = None
    error: Optional[str] = None


class ToolExecutor(Protocol):
    def execute(self, tool_call: ToolCall) -> ExecutionResult:
        ...


class ErrorQuestionGenerator(Protocol):
    def generate_error_question(
        self,
        error: str,
        last_call: ToolCall,
        tool_schemas: Mapping[str, ToolSchema],
    ) -> Optional[Question]:
        ...
