from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import Aspect, ToolCallCandidate, UNK


def partition_candidates(
    candidates: Sequence[ToolCallCandidate],
    aspects: Sequence[Aspect],
) -> Dict[Tuple[object, ...], List[int]]:
    groups: Dict[Tuple[object, ...], List[int]] = defaultdict(list)
    for idx, candidate in enumerate(candidates):
        key_parts = []
        for aspect in aspects:
            if candidate.tool_name != aspect.tool_name:
                key_parts.append(None)
                continue
            value = candidate.arguments.get(aspect.param_name, UNK)
            if isinstance(value, list):
                key_parts.append(tuple(value))
            elif isinstance(value, set):
                key_parts.append(tuple(sorted(value)))
            else:
                key_parts.append(value)
        groups[tuple(key_parts)].append(idx)
    return groups


def compute_evpi(
    candidates: Sequence[ToolCallCandidate],
    probabilities: Sequence[float],
    aspects: Sequence[Aspect],
) -> float:
    if not candidates:
        return 0.0
    if len(candidates) != len(probabilities):
        raise ValueError("Candidates and probabilities must align")

    partitions = partition_candidates(candidates, aspects)
    max_overall = max(probabilities) if probabilities else 0.0
    score = 0.0
    for indices in partitions.values():
        score += max(probabilities[i] for i in indices)
    return score - max_overall
