from __future__ import annotations

import json
import re
from dataclasses import dataclass
import ast
import os
from typing import Iterable, List, Mapping, Optional, Protocol

from .prompts import build_candidate_prompt, build_question_prompt
from ..core.types import Aspect, Question, ToolCallCandidate, ToolSchema


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str:
        ...


@dataclass
class PromptCandidateGenerator:
    llm: LLMClient

    def generate(self, user_input: str, observations: Iterable[str], tool_schemas: Iterable[ToolSchema]) -> List[ToolCallCandidate]:
        prompt = build_candidate_prompt(user_input, tool_schemas, observations)
        try:
            response = self.llm.complete(prompt)
        except Exception as exc:
            if os.getenv("SAGE_DEBUG_LLM") == "1":
                print(f"\n[DEBUG] candidate_generator error: {exc}")
            return []
        if os.getenv("SAGE_DEBUG_LLM") == "1":
            print("\n[DEBUG] candidate_generator response:")
            print(response)
        payload = _parse_json_list(response)
        if os.getenv("SAGE_DEBUG_LLM") == "1":
            print("[DEBUG] candidate_generator parsed payload:")
            print(payload)
        candidates: List[ToolCallCandidate] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            tool_name = item.get("tool")
            arguments = item.get("arguments", {})
            if isinstance(arguments, dict) and set(arguments.keys()) == {"arguments"}:
                inner = arguments.get("arguments")
                if isinstance(inner, dict):
                    arguments = inner
            if isinstance(tool_name, str):
                tool_name = tool_name.strip()
            if not isinstance(arguments, dict) or not isinstance(tool_name, str):
                continue
            normalized_args = _normalize_arguments(arguments)
            if os.getenv("SAGE_DEBUG_LLM") == "1" and normalized_args != arguments:
                print(f"[DEBUG] normalized arguments for {tool_name}: {normalized_args}")
            arguments = normalized_args
            candidates.append(ToolCallCandidate(tool_name=tool_name, arguments=arguments))
        return candidates


@dataclass
class PromptQuestionGenerator:
    llm: LLMClient

    def generate(
        self,
        user_input: str,
        observations: Iterable[str],
        tool_schemas: Iterable[ToolSchema],
        candidates: Iterable[ToolCallCandidate],
    ) -> List[Question]:
        candidates_payload = [
            {"tool": c.tool_name, "arguments": dict(c.arguments)} for c in candidates
        ]
        prompt = build_question_prompt(user_input, tool_schemas, observations, candidates_payload)
        try:
            response = self.llm.complete(prompt)
        except Exception as exc:
            if os.getenv("SAGE_DEBUG_LLM") == "1":
                print(f"\n[DEBUG] question_generator error: {exc}")
            return []
        if os.getenv("SAGE_DEBUG_LLM") == "1":
            print("\n[DEBUG] question_generator response:")
            print(response)
        payload = _parse_json_list(response)
        questions: List[Question] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = item.get("question")
            aspects_payload = item.get("aspects", [])
            if not isinstance(text, str) or not isinstance(aspects_payload, list):
                continue
            aspects: List[Aspect] = []
            for aspect in aspects_payload:
                tool_name = aspect.get("tool")
                param = aspect.get("param")
                if isinstance(tool_name, str) and isinstance(param, str):
                    aspects.append(Aspect(tool_name=tool_name, param_name=param))
            if aspects:
                questions.append(Question(text=text, aspects=tuple(aspects)))
        return questions


def _parse_json_list(text: str) -> List[Mapping[str, object]]:
    cleaned = text.strip()
    strict = os.getenv("SAGE_STRICT_JSON") == "1"
    if not strict:
        if "<UNK>" in cleaned:
            cleaned = re.sub(r'(?<!")<UNK>(?!")', '"<UNK>"', cleaned)
        if cleaned.lower().startswith("answer:"):
            cleaned = cleaned.split(":", 1)[1].strip()
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return _coerce_payload_to_dicts(payload)
    except json.JSONDecodeError:
        if strict:
            return []
        pass

    if not strict:
        match = re.search(r"\[[\s\S]*\]", cleaned)
        if match:
            try:
                payload = json.loads(match.group(0))
                if isinstance(payload, dict):
                    return [payload]
                if isinstance(payload, list):
                    return _coerce_payload_to_dicts(payload)
            except json.JSONDecodeError:
                pass
        # Try slicing from first '[' to last ']'
        if "[" in cleaned and "]" in cleaned:
            sliced = cleaned[cleaned.find("["): cleaned.rfind("]") + 1]
            try:
                payload = json.loads(sliced)
                if isinstance(payload, dict):
                    return [payload]
                if isinstance(payload, list):
                    return _coerce_payload_to_dicts(payload)
            except json.JSONDecodeError:
                pass
        # Try trimming trailing brackets/braces (common model glitch).
        for trimmed in _trim_trailing_brackets(cleaned):
            try:
                payload = json.loads(trimmed)
                if isinstance(payload, dict):
                    return [payload]
                if isinstance(payload, list):
                    return _coerce_payload_to_dicts(payload)
            except json.JSONDecodeError:
                continue
        # Fallback: try Python literal eval (handles some trailing commas/booleans).
        try:
            payload = ast.literal_eval(_pythonize_json(cleaned))
            if isinstance(payload, dict):
                return [payload]
            if isinstance(payload, list):
                return _coerce_payload_to_dicts(payload)
        except (SyntaxError, ValueError):
            pass
    return []


def _coerce_payload_to_dicts(payload: List[object]) -> List[Mapping[str, object]]:
    parsed: List[Mapping[str, object]] = []
    # Handle ["tool", {args}] or [["tool", {args}], ...] formats.
    if len(payload) == 2 and isinstance(payload[0], str) and isinstance(payload[1], dict):
        parsed.append(_pair_to_item(payload[0], payload[1]))
        return parsed
    for item in payload:
        if isinstance(item, dict):
            parsed.append(item)
            continue
        if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], dict):
            parsed.append(_pair_to_item(item[0], item[1]))
            continue
        if isinstance(item, str):
            try:
                parsed_item = json.loads(item)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed_item, dict):
                parsed.append(parsed_item)
                continue
            if (
                isinstance(parsed_item, list)
                and len(parsed_item) == 2
                and isinstance(parsed_item[0], str)
                and isinstance(parsed_item[1], dict)
            ):
                parsed.append(_pair_to_item(parsed_item[0], parsed_item[1]))
    return parsed


def _pair_to_item(tool_name: str, maybe_args: Mapping[str, object]) -> Mapping[str, object]:
    if "tool" in maybe_args and "arguments" in maybe_args:
        return {"tool": tool_name, "arguments": maybe_args.get("arguments", {})}
    return {"tool": tool_name, "arguments": dict(maybe_args)}


def _trim_trailing_brackets(text: str) -> List[str]:
    variants: List[str] = []
    trimmed = text.strip()
    for _ in range(3):
        if trimmed and trimmed[-1] in "]}":
            trimmed = trimmed[:-1].rstrip()
            variants.append(trimmed)
        else:
            break
    return variants


def _normalize_arguments(arguments: Mapping[str, object]) -> Dict[str, object]:
    if os.getenv("SAGE_DISABLE_NORMALIZE") == "1":
        return dict(arguments)
    normalized: Dict[str, object] = {}
    for key, value in arguments.items():
        if isinstance(value, list) and len(value) == 1:
            value = value[0]
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("<unk>", "unk", "unknown", "dontcare", "don't care", "n/a", "none"):
                normalized[key] = "<UNK>"
                continue
            if lowered in ("true", "false"):
                normalized[key] = lowered == "true"
                continue
            if re.fullmatch(r"-?\d+", lowered):
                try:
                    normalized[key] = int(lowered)
                    continue
                except ValueError:
                    pass
            if re.fullmatch(r"-?\d+\.\d+", lowered):
                try:
                    normalized[key] = float(lowered)
                    continue
                except ValueError:
                    pass
            normalized[key] = value.strip()
            continue
        normalized[key] = value
    return normalized


def _pythonize_json(text: str) -> str:
    """Convert JSON booleans/null to Python for literal_eval fallback."""
    converted = re.sub(r"\btrue\b", "True", text, flags=re.IGNORECASE)
    converted = re.sub(r"\bfalse\b", "False", converted, flags=re.IGNORECASE)
    converted = re.sub(r"\bnull\b", "None", converted, flags=re.IGNORECASE)
    return converted
