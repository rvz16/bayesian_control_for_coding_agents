from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Sequence

from ..core.agent import SageAgent
from ..metrics.metrics import MetricResult, evaluate_metrics
from ..core.types import Question, QuestionAsker, ToolCall


class UserSimulator(Protocol):
    def answer(self, question: Question, scenario_id: Optional[str] = None) -> Optional[str]:
        ...


@dataclass
class SimulationScenario:
    scenario_id: str
    requests: Sequence[str]
    ground_truth: Sequence[ToolCall]


@dataclass
class SimulationTurn:
    request: str
    tool_call: Optional[ToolCall]
    questions_asked: int


@dataclass
class SimulationResult:
    scenario_id: str
    turns: List[SimulationTurn]
    metrics: MetricResult


class SimulatedQuestionAsker(QuestionAsker):
    def __init__(self, simulator: UserSimulator, scenario_id: Optional[str]) -> None:
        self.simulator = simulator
        self.scenario_id = scenario_id
        self.count = 0

    def ask(self, question: Question) -> str:
        self.count += 1
        response = self.simulator.answer(question, self.scenario_id)
        if response is None:
            return ""
        return response


class ClarifyBenchSimulator:
    def __init__(self, agent: SageAgent, user_simulator: UserSimulator) -> None:
        self.agent = agent
        self.user_simulator = user_simulator

    def run(self, scenario: SimulationScenario) -> SimulationResult:
        if len(scenario.requests) != len(scenario.ground_truth):
            raise ValueError("Scenario requests and ground truth must align")

        turns: List[SimulationTurn] = []
        predictions: List[ToolCall] = []
        question_counts: List[int] = []

        for request, truth in zip(scenario.requests, scenario.ground_truth):
            asker = SimulatedQuestionAsker(self.user_simulator, scenario.scenario_id)
            self.agent.question_asker = asker
            result = self.agent.run(request)
            predictions.append(result.tool_call or ToolCall("", {}))
            question_counts.append(asker.count)
            turns.append(
                SimulationTurn(
                    request=request,
                    tool_call=result.tool_call,
                    questions_asked=asker.count,
                )
            )

        metrics = evaluate_metrics(predictions, list(scenario.ground_truth), question_counts)
        return SimulationResult(
            scenario_id=scenario.scenario_id,
            turns=turns,
            metrics=metrics,
        )
