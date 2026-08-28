from __future__ import annotations

import json
from typing import Iterable, Mapping

from ..core.types import ToolSchema


def tool_schemas_to_json(tool_schemas: Iterable[ToolSchema]) -> str:
    tools_payload = []
    for tool in tool_schemas:
        params_payload = {}
        for name, domain in tool.parameters.items():
            if domain.is_finite():
                params_payload[name] = sorted(list(domain.values or []))
            else:
                params_payload[name] = "<continuous>"
        tools_payload.append(
            {
                "name": tool.name,
                "required": sorted(list(tool.required)),
                "parameters": params_payload,
            }
        )
    return json.dumps(tools_payload, ensure_ascii=True)


def build_candidate_prompt(
    user_input: str, tool_schemas: Iterable[ToolSchema], observations: Iterable[str]
) -> str:
    tools_json = tool_schemas_to_json(tool_schemas)
    obs_text = "\n".join(observations)
    return (
        "You are an agent that proposes candidate tool calls.\n"
        "Return a JSON list. Each item must be: "
        "{\"tool\": string, \"arguments\": {param: value or <UNK>}}.\n"
        "Use <UNK> for unknowns. Only use tool names and parameter values from the schema.\n"
        f"User input: {user_input}\n"
        f"Observations: {obs_text or '<none>'}\n"
        f"Tool schemas: {tools_json}\n"
        "JSON list only."
    )


def build_question_prompt(
    user_input: str,
    tool_schemas: Iterable[ToolSchema],
    observations: Iterable[str],
    candidates: Iterable[Mapping[str, object]],
) -> str:
    tools_json = tool_schemas_to_json(tool_schemas)
    obs_text = "\n".join(observations)
    candidates_json = json.dumps(list(candidates), ensure_ascii=True)
    return (
        "You are an agent that generates clarifying questions for tool calling on different domain tasks.\n"
        "Return a JSON list. Each item must be: "
        "{\"question\": string, \"aspects\": [{\"tool\": string, \"param\": string}]}\n"
        "Only ask about parameters that are <UNK> in candidates.\n"
        f"User input: {user_input}\n"
        f"Observations: {obs_text or '<none>'}\n"
        f"Candidates: {candidates_json}\n"
        f"Tool schemas: {tools_json}\n"
        "!!!!USE JSON LIST only!!!!"
    )
