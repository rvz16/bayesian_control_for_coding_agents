"""Unit tests for the CostTracker. No API calls."""
from __future__ import annotations

import json
import pathlib
import sys
import threading

import pytest

# Tests live in tests/; the package root (orchestration_hypothesis_testing/)
# is the parent of tests/. Add it to sys.path so we can import from _common/
# directly. Also keep scripts/ on the path for the legacy cost_tracker shim
# (used by callers we haven't migrated yet -- removed in refactor phase 6).
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))
sys.path.insert(0, str(_PKG_ROOT / "scripts"))

from _common.cost import CostTracker, extract_usage, project_cost  # noqa: E402


def test_under_cap_can_proceed():
    t = CostTracker(name="m", cap_usd=1.0)
    assert t.can_proceed()
    t.record(cost_usd=0.3, prompt_tokens=10, completion_tokens=5)
    assert t.can_proceed()
    assert t.total_usd == pytest.approx(0.3)
    assert t.n_calls == 1


def test_caps_at_threshold():
    t = CostTracker(name="m", cap_usd=1.0)
    t.record(cost_usd=0.6, prompt_tokens=0, completion_tokens=0)
    assert t.can_proceed()
    t.record(cost_usd=0.5, prompt_tokens=0, completion_tokens=0)
    assert not t.can_proceed()  # 1.1 > 1.0
    assert t.capped


def test_remaining_never_negative():
    t = CostTracker(name="m", cap_usd=1.0)
    t.record(cost_usd=5.0, prompt_tokens=0, completion_tokens=0)
    assert t.remaining == 0.0
    assert t.capped


def test_concurrent_writes(tmp_path):
    t = CostTracker(name="m", cap_usd=100.0, log_path=tmp_path / "log.jsonl")
    n_per_thread = 50
    n_threads = 8

    def worker():
        for _ in range(n_per_thread):
            t.record(cost_usd=0.01, prompt_tokens=1, completion_tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert t.n_calls == n_per_thread * n_threads
    assert t.total_usd == pytest.approx(0.01 * n_per_thread * n_threads)
    # Log file has one line per recorded call
    lines = (tmp_path / "log.jsonl").read_text().strip().splitlines()
    assert len(lines) == n_per_thread * n_threads
    last = json.loads(lines[-1])
    assert last["cumulative_usd"] == pytest.approx(t.total_usd)


def test_skipped_counter():
    t = CostTracker(name="m", cap_usd=0.1)
    t.record(cost_usd=0.2, prompt_tokens=0, completion_tokens=0)
    assert t.capped
    t.note_skipped(7)
    snap = t.snapshot()
    assert snap["n_skipped"] == 7
    assert snap["capped"]


def test_log_writes_jsonl_with_metadata(tmp_path):
    log_path = tmp_path / "log.jsonl"
    t = CostTracker(name="haiku45", cap_usd=10.0, log_path=log_path)
    t.record(
        cost_usd=0.0123,
        prompt_tokens=400,
        completion_tokens=120,
        instance_id="django__django-12284",
        patch_id=1,
        extra={"extraction_path": "change_blocks"},
    )
    rec = json.loads(log_path.read_text().strip())
    assert rec["model"] == "haiku45"
    assert rec["instance_id"] == "django__django-12284"
    assert rec["patch_id"] == 1
    assert rec["cost_usd"] == pytest.approx(0.0123)
    assert rec["prompt_tokens"] == 400
    assert rec["completion_tokens"] == 120
    assert rec["extraction_path"] == "change_blocks"
    assert rec["cumulative_usd"] == pytest.approx(0.0123)


def test_extract_usage_from_dict():
    fake = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost": 0.0042,
        }
    }
    cost, p, c = extract_usage(fake)
    assert cost == pytest.approx(0.0042)
    assert p == 100
    assert c == 50


def test_extract_usage_missing_cost():
    fake = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    cost, p, c = extract_usage(fake)
    assert cost == 0.0
    assert p == 10 and c == 5


def test_extract_usage_no_usage_field():
    cost, p, c = extract_usage({})
    assert cost == 0.0 and p == 0 and c == 0


def test_extract_usage_from_pydantic_like_obj():
    class FakeResp:
        def model_dump(self):
            return {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.0001}}
    cost, p, c = extract_usage(FakeResp())
    assert cost == pytest.approx(0.0001)
    assert p == 7 and c == 3


def test_project_cost():
    # 1 call costs $0.02 → projecting 150 calls = $3
    assert project_cost(0.02, 150, probe_calls=1) == pytest.approx(3.0)
    # Probe of 4 calls totalling $0.08 → 150 calls = $3
    assert project_cost(0.08, 150, probe_calls=4) == pytest.approx(3.0)
    # Zero probe calls → no projection
    assert project_cost(1.0, 100, probe_calls=0) == 0.0
