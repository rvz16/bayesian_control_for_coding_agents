from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

from .belief import BeliefState
from .constraints import SimpleConstraintExtractor
from .evpi import compute_evpi
from .uncertainty_propagation import create_sage_propagator
from .types import (
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


@dataclass(frozen=True)
class SageAgentConfig:
    max_questions: int = 6
    redundancy_weight: float = 0.5
    tau_execute: float = 0.85
    alpha: float = 0.1
    epsilon: float = 1e-4
    use_tts_uncertainty: bool = False
    structured_uncertainty_weight: float = 0.7
    llm_uncertainty_modulation: float = 0.5
    uncertainty_threshold: float = 1.0
    use_propagation: bool = False
    use_reflexion: bool = False
    max_reflexion_attempts: int = 0
    use_dp_planning: bool = False
    dp_query_cost: float = 0.05
    dp_cost_model: object = None  # Optional CostModel from bayesian_dp


@dataclass
class AgentStep:
    candidates: List[ToolCallCandidate]
    probabilities: List[float]
    question: Optional[Question]
    score: Optional[float]
    response: Optional[str]
    structured_uncertainty: Optional[float] = None
    llm_uncertainty: Optional[float] = None
    combined_uncertainty: Optional[float] = None
    propagated_uncertainty: Optional[float] = None
    reflexion_note: Optional[str] = None


@dataclass
class AgentResult:
    tool_call: Optional[ToolCall]
    execution_result: ExecutionResult
    steps: List[AgentStep] = field(default_factory=list)


class SageAgent:
    def __init__(
        self,
        tool_schemas: Iterable[ToolSchema],
        candidate_generator: CandidateGenerator,
        question_generator: QuestionGenerator,
        question_asker: QuestionAsker,
        tool_executor: ToolExecutor,
        constraint_extractor: Optional[ConstraintExtractor] = None,
        error_question_generator: Optional[ErrorQuestionGenerator] = None,
        config: Optional[SageAgentConfig] = None,
    ) -> None:
        self.tool_schemas = {tool.name: tool for tool in tool_schemas}
        self.candidate_generator = candidate_generator
        self.question_generator = question_generator
        self.question_asker = question_asker
        self.tool_executor = tool_executor
        self.constraint_extractor = constraint_extractor or SimpleConstraintExtractor()
        self.error_question_generator = error_question_generator
        self.config = config or SageAgentConfig()
        self._base_domains = {
            tool.name: {k: v for k, v in tool.parameters.items()} for tool in tool_schemas
        }
        self._belief = self._new_belief()
        self._propagator = (
            create_sage_propagator(
                structured_weight=self.config.structured_uncertainty_weight,
                llm_weight=1.0 - self.config.structured_uncertainty_weight,
            )
            if self.config.use_propagation
            else None
        )

    def _new_belief(self) -> BeliefState:
        return BeliefState(
            domains=copy.deepcopy(self._base_domains),
            epsilon=self.config.epsilon,
        )

    def run(self, user_input: str) -> AgentResult:
        self._belief = self._new_belief()
        observations: List[str] = []
        steps: List[AgentStep] = []
        aspect_counts: Dict[Aspect, int] = defaultdict(int)
        t = 0
        reflexion_attempts = 0
        if self._propagator is not None:
            self._propagator.reset()

        while True:
            candidates = self.candidate_generator.generate_candidates(
                user_input, observations, self.tool_schemas
            )
            if not candidates:
                raise ValueError("Candidate generator returned no candidates")
            for candidate in candidates:
                if candidate.tool_name not in self.tool_schemas:
                    raise ValueError(
                        f"Unknown tool in candidate: {candidate.tool_name}"
                    )

            weights = [
                self._belief.candidate_weight(candidate, self.tool_schemas[candidate.tool_name])
                for candidate in candidates
            ]
            llm_uncertainty = self._get_llm_uncertainty(self.candidate_generator)
            if self.config.use_tts_uncertainty and llm_uncertainty is not None:
                modulation = self.config.llm_uncertainty_modulation
                uncertainty_factor = 1.0 - (llm_uncertainty * modulation)
                weights = [w * uncertainty_factor for w in weights]
            probabilities = self._belief.normalize(weights)
            max_prob = max(probabilities)
            best_idx = probabilities.index(max_prob)
            best_candidate = candidates[best_idx]
            structured_uncertainty = 1.0 - max_prob
            combined_uncertainty = structured_uncertainty
            if llm_uncertainty is not None:
                sw = self.config.structured_uncertainty_weight
                combined_uncertainty = sw * structured_uncertainty + (1.0 - sw) * llm_uncertainty

            propagated_uncertainty = None
            if self._propagator is not None:
                self._propagator.observe(
                    structured_uncertainty, "candidate_generation", {"max_prob": max_prob}
                )
                if llm_uncertainty is not None:
                    self._propagator.observe(llm_uncertainty, "llm_parsing", {"stage": "candidate"})
                propagated_uncertainty = self._propagator.accumulated_uncertainty
                combined_uncertainty = propagated_uncertainty

            if max_prob >= self.config.tau_execute or t >= self.config.max_questions:
                if combined_uncertainty <= self.config.uncertainty_threshold:
                    result = self._execute_candidate(best_candidate, steps)
                    if (
                        result.execution_result.success
                        or not self.config.use_reflexion
                        or reflexion_attempts >= self.config.max_reflexion_attempts
                    ):
                        return result
                    reflection = self._generate_reflection(
                        user_input, result.tool_call, result.execution_result.error or ""
                    )
                    if reflection:
                        observations.append(reflection)
                        steps.append(
                            AgentStep(
                                candidates=[best_candidate],
                                probabilities=[max_prob],
                                question=None,
                                score=None,
                                response=None,
                                structured_uncertainty=structured_uncertainty,
                                llm_uncertainty=llm_uncertainty,
                                combined_uncertainty=combined_uncertainty,
                                propagated_uncertainty=propagated_uncertainty,
                                reflexion_note=reflection,
                            )
                        )
                    reflexion_attempts += 1
                    continue
                return self._execute_candidate(best_candidate, steps)

            questions = self.question_generator.generate_questions(
                user_input, candidates, observations, self.tool_schemas
            )
            if not questions:
                return self._execute_candidate(best_candidate, steps)

            if self.config.use_dp_planning:
                from .bayesian_dp import compute_question_value_dp
                scored_questions = compute_question_value_dp(
                    belief=self._belief,
                    candidates=candidates,
                    probabilities=probabilities,
                    questions=questions,
                    tool_schemas=self.tool_schemas,
                    budget=self.config.max_questions - t,
                    aspect_counts=aspect_counts,
                    redundancy_weight=self.config.redundancy_weight,
                    query_cost=self.config.dp_query_cost,
                    cost_model=self.config.dp_cost_model,
                )
            else:
                scored_questions = []
                for question in questions:
                    evpi = compute_evpi(candidates, probabilities, question.aspects)
                    cost = self._redundancy_cost(question, aspect_counts)
                    scored_questions.append((evpi - cost, question))
                scored_questions.sort(key=lambda item: item[0], reverse=True)
            best_score, best_question = scored_questions[0]
            if best_score < self.config.alpha * max_prob:
                if not self._has_required_unknowns(best_candidate):
                    return self._execute_candidate(best_candidate, steps)

            response = self.question_asker.ask(best_question)
            observations.append(response)
            for aspect in best_question.aspects:
                aspect_counts[aspect] += 1
                self._update_domain(aspect, response)

            steps.append(
                AgentStep(
                    candidates=candidates,
                    probabilities=probabilities,
                    question=best_question,
                    score=best_score,
                    response=response,
                    structured_uncertainty=structured_uncertainty,
                    llm_uncertainty=llm_uncertainty,
                    combined_uncertainty=combined_uncertainty,
                    propagated_uncertainty=propagated_uncertainty,
                )
            )
            if self._propagator is not None:
                self._propagator.observe(
                    structured_uncertainty, "belief_update", {"response": response}
                )
            t += 1

    def _update_domain(self, aspect: Aspect, response: str) -> None:
        tool = self.tool_schemas[aspect.tool_name]
        current = self._belief.domains[tool.name][aspect.param_name]
        updated = self.constraint_extractor.update_domain(current, response)
        self._belief.domains[tool.name][aspect.param_name] = updated
        if tool.domain_refiner is not None:
            refined = tool.domain_refiner.refine(
                tool,
                self._belief.domains[tool.name],
                self._current_assignments(tool.name),
            )
            self._belief.domains[tool.name] = dict(refined)

    def _current_assignments(self, tool_name: str) -> Mapping[str, object]:
        assignments: Dict[str, object] = {}
        for param, domain in self._belief.domains[tool_name].items():
            if domain.is_finite() and domain.size() == 1:
                assignments[param] = next(iter(domain.values or []))
            else:
                assignments[param] = UNK
        return assignments

    def _redundancy_cost(self, question: Question, aspect_counts: Mapping[Aspect, int]) -> float:
        cost = 0.0
        for aspect in question.aspects:
            cost += float(aspect_counts.get(aspect, 0))
        return cost * self.config.redundancy_weight

    def _execute_candidate(
        self, candidate: ToolCallCandidate, steps: List[AgentStep]
    ) -> AgentResult:
        tool_schema = self.tool_schemas[candidate.tool_name]
        tool_call = ToolCall(tool_name=candidate.tool_name, arguments=candidate.arguments)
        try:
            tool_schema.validate_call(tool_call.arguments)
        except ValueError as exc:
            result = ExecutionResult(success=False, error=str(exc))
            return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

        result = self.tool_executor.execute(tool_call)
        if result.success or self.error_question_generator is None:
            return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

        error_question = self.error_question_generator.generate_error_question(
            result.error or "Execution failed", tool_call, self.tool_schemas
        )
        if error_question is None:
            return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

        response = self.question_asker.ask(error_question)
        steps.append(
            AgentStep(
                candidates=[candidate],
                probabilities=[1.0],
                question=error_question,
                score=None,
                response=response,
            )
        )
        for aspect in error_question.aspects:
            self._update_domain(aspect, response)
        return AgentResult(tool_call=tool_call, execution_result=result, steps=steps)

    def _has_required_unknowns(self, candidate: ToolCallCandidate) -> bool:
        tool_schema = self.tool_schemas[candidate.tool_name]
        for param in tool_schema.required:
            if candidate.arguments.get(param, UNK) == UNK:
                return True
        return False

    def _get_llm_uncertainty(self, generator: object) -> Optional[float]:
        if not self.config.use_tts_uncertainty:
            return None
        llm = getattr(generator, "llm", None)
        if llm is None:
            return None
        return getattr(llm, "last_uncertainty", None)

    def _generate_reflection(
        self,
        user_input: str,
        tool_call: Optional[ToolCall],
        error: str,
    ) -> str:
        llm = getattr(self.candidate_generator, "llm", None)
        if llm is None or not hasattr(llm, "complete"):
            return ""
        call_desc = (
            f"{tool_call.tool_name}({dict(tool_call.arguments)})"
            if tool_call is not None
            else "<none>"
        )
        prompt = (
            "Analyze why this tool call failed and suggest how to improve the next attempt.\n\n"
            f"User input: {user_input}\n"
            f"Tool call: {call_desc}\n"
            f"Error: {error}\n\n"
            "Reflection:"
        )
        return llm.complete(prompt).strip()
