"""Unit tests for post-hoc SAGE UHead data plumbing. No model loading."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from experiments.orchestration_hypothesis_testing.scripts.analyze_lcb_llm_tool_agent_logs import (  # noqa: E402
    load_uhead,
)
from experiments.orchestration_hypothesis_testing.scripts.score_sage_uhead import (  # noqa: E402
    completion_ids,
    infer_harmony_start_date,
)


class FakeTokenizer:
    _sage_uhead_token_ids_by_bytes = {b"a": 7, b"bc": 11}


def test_infer_harmony_start_date_from_run_directory():
    root = pathlib.Path("/runs/lcb_hard_20260821_143000/results")
    assert infer_harmony_start_date(root) == "2026-08-21"


def test_completion_ids_reconstruct_saved_token_bytes():
    row = {
        "completion_tokens": 2,
        "logprobs": {
            "content": [
                {"bytes": [97]},
                {"bytes": [98, 99]},
            ]
        },
    }
    assert completion_ids(FakeTokenizer(), row) == [7, 11]


def test_completion_ids_reject_token_count_mismatch():
    row = {
        "completion_tokens": 2,
        "logprobs": {"content": [{"bytes": [97]}]},
    }
    with pytest.raises(ValueError, match="completion token mismatch"):
        completion_ids(FakeTokenizer(), row)


def test_load_uhead_indexes_each_generation(tmp_path):
    path = tmp_path / "scores.jsonl"
    rows = [
        {
            "instance_id": "x",
            "generation_index": 0,
            "uhead_confidence": 0.8,
            "uhead_uncertainty": 0.25,
        },
        {
            "instance_id": "x",
            "generation_index": 1,
            "uhead_confidence": 0.6,
            "uhead_uncertainty": 2 / 3,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    scores = load_uhead(path)
    assert scores[("x", 0)]["uhead_confidence"] == pytest.approx(0.8)
    assert scores[("x", 1)]["uhead_uncertainty"] == pytest.approx(2 / 3)
