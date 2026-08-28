"""Unit tests for _common/telemetry.py. No API calls."""
from __future__ import annotations

import json
import pathlib
import sys
import threading

import pytest

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from _common.telemetry import (  # noqa: E402
    ACTION_TYPES,
    TelemetryLogger,
    _ActionTelemetry,
    write_action,
)


# ---------------------------------------------------------------------------
# Construction + basic record
# ---------------------------------------------------------------------------

def test_construct_and_record_calibration_shape(tmp_path):
    """Backward-compat shape: mirrors what calibration/lcb.py currently writes."""
    log_path = tmp_path / "action_telemetry.jsonl"
    tele = TelemetryLogger(log_path, dataset="mbpp", model_name="haiku45")
    tele.record(
        action_type="generate", runtime_s=1.2345678,
        instance_id="Mbpp/61", patch_id=0, api_cost_usd=0.0012,
    )
    tele.close()
    rec = json.loads(log_path.read_text().strip())
    assert rec["dataset"] == "mbpp"
    assert rec["model_name"] == "haiku45"
    assert rec["instance_id"] == "Mbpp/61"
    assert rec["patch_id"] == 0
    assert rec["action_type"] == "generate"
    assert rec["runtime_seconds"] == pytest.approx(1.2346)  # rounded to 4 dp
    assert rec["passed"] is None
    assert rec["api_cost_usd"] == pytest.approx(0.0012)
    assert "ts" in rec  # always present


def test_record_omits_unset_optional_fields(tmp_path):
    """Slim JSONL: step / belief_before / run_id / extra are omitted unless
    explicitly set. Keeps the calibration JSONL compact."""
    log_path = tmp_path / "log.jsonl"
    tele = TelemetryLogger(log_path, dataset="d", model_name="m")
    tele.record(
        action_type="critic_L0", runtime_s=0.01, instance_id="x",
    )
    tele.close()
    rec = json.loads(log_path.read_text().strip())
    # Required keys always present
    for k in ("ts", "dataset", "model_name", "instance_id", "action_type",
              "runtime_seconds", "passed", "api_cost_usd"):
        assert k in rec, f"missing required key {k!r}"
    # Optional keys absent when not passed
    for k in ("patch_id", "step", "belief_before", "run_id", "extra"):
        assert k not in rec, f"unexpected optional key {k!r} in slim record"


def test_record_includes_iter_and_bdp_optional_fields(tmp_path):
    """Iter / BDP-aware code path: step + belief_before are recorded when
    supplied. Critical for the latency-vs-policy joins in analysis."""
    log_path = tmp_path / "log.jsonl"
    tele = TelemetryLogger(log_path, dataset="lcb", model_name="haiku45",
                           run_id="wandb-abc123")
    tele.record(
        action_type="refine", runtime_s=2.0,
        instance_id="lcb/42", step=3, belief_before=0.78,
        extra={"reflection_summary": "off-by-one"},
    )
    tele.close()
    rec = json.loads(log_path.read_text().strip())
    assert rec["step"] == 3
    assert rec["belief_before"] == pytest.approx(0.78)
    assert rec["run_id"] == "wandb-abc123"
    assert rec["extra"] == {"reflection_summary": "off-by-one"}
    assert "patch_id" not in rec  # not supplied


def test_record_passed_field_round_trip(tmp_path):
    """`passed` is the single most-queried field downstream. Confirm
    True/False/None round-trip cleanly through JSON."""
    log_path = tmp_path / "log.jsonl"
    tele = TelemetryLogger(log_path, dataset="d", model_name="m")
    for p in (True, False, None):
        tele.record(action_type="verify", runtime_s=0.0,
                    instance_id="i", passed=p)
    tele.close()
    lines = log_path.read_text().splitlines()
    rs = [json.loads(line) for line in lines]
    assert [r["passed"] for r in rs] == [True, False, None]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_thread_safe_writes(tmp_path):
    """8 threads × 100 writes each. Every line should JSON-parse and the
    total line count match exactly."""
    log_path = tmp_path / "log.jsonl"
    tele = TelemetryLogger(log_path, dataset="d", model_name="m")
    n_per_thread = 100
    n_threads = 8

    def worker(tid: int):
        for i in range(n_per_thread):
            tele.record(
                action_type="critic_L0", runtime_s=0.001,
                instance_id=f"t{tid}_i{i}",
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads: t.start()
    for t in threads: t.join()
    tele.close()

    lines = log_path.read_text().splitlines()
    assert len(lines) == n_per_thread * n_threads
    for line in lines:
        json.loads(line)  # no JSONDecodeError


# ---------------------------------------------------------------------------
# Close semantics
# ---------------------------------------------------------------------------

def test_close_is_idempotent(tmp_path):
    tele = TelemetryLogger(tmp_path / "log.jsonl", dataset="d", model_name="m")
    tele.close()
    tele.close()  # no exception


def test_record_after_close_raises(tmp_path):
    tele = TelemetryLogger(tmp_path / "log.jsonl", dataset="d", model_name="m")
    tele.close()
    with pytest.raises(RuntimeError, match="closed"):
        tele.record(action_type="generate", runtime_s=0.0, instance_id="i")


def test_context_manager(tmp_path):
    log_path = tmp_path / "log.jsonl"
    with TelemetryLogger(log_path, dataset="d", model_name="m") as tele:
        tele.record(action_type="generate", runtime_s=0.1, instance_id="i")
    # File handle should be closed after exit
    assert tele._closed
    # File has the row
    assert json.loads(log_path.read_text().strip())["action_type"] == "generate"


# ---------------------------------------------------------------------------
# write_action() convenience wrapper
# ---------------------------------------------------------------------------

def test_write_action_matches_record(tmp_path):
    """write_action(logger, ...) must produce a row indistinguishable from
    logger.record(...). Verifies the abbo-compat shim doesn't drift."""
    log_path = tmp_path / "log.jsonl"
    tele = TelemetryLogger(log_path, dataset="d", model_name="m")
    write_action(tele, action_type="generate", runtime_s=1.5,
                 instance_id="i1", patch_id=0, api_cost_usd=0.01)
    tele.record(action_type="generate", runtime_s=1.5,
                instance_id="i1", patch_id=0, api_cost_usd=0.01)
    tele.close()
    a, b = (json.loads(line) for line in log_path.read_text().splitlines())
    # Strip timestamps which differ by microseconds
    a.pop("ts"); b.pop("ts")
    assert a == b


# ---------------------------------------------------------------------------
# Back-compat alias
# ---------------------------------------------------------------------------

def test_action_telemetry_alias_is_telemetry_logger():
    """_ActionTelemetry is kept as an alias so the existing imports in
    calibration/mbpp.py + humaneval.py keep working through the migration
    window (and also via the back-compat re-export in calibration/lcb.py)."""
    assert _ActionTelemetry is TelemetryLogger


# ---------------------------------------------------------------------------
# Path handling
# ---------------------------------------------------------------------------

def test_creates_parent_directories(tmp_path):
    """The writer should mkdir -p its parent so iter/refine.py doesn't have
    to do it before each gen subdir is created."""
    nested = tmp_path / "a" / "b" / "c" / "log.jsonl"
    tele = TelemetryLogger(nested, dataset="d", model_name="m")
    tele.record(action_type="generate", runtime_s=0.0, instance_id="i")
    tele.close()
    assert nested.exists()


# ---------------------------------------------------------------------------
# ACTION_TYPES sanity
# ---------------------------------------------------------------------------

def test_action_types_contains_expected_set():
    expected = {"generate", "refine", "reflect",
                "critic_L0", "critic_L1", "critic_L2", "critic_L3",
                "verify"}
    assert ACTION_TYPES == expected


# ---------------------------------------------------------------------------
# `extra` field semantics — `is not None` (not truthiness) so an explicit
# empty dict is written, matching the pattern used for other optional fields.
# ---------------------------------------------------------------------------

def test_record_extra_empty_dict_is_written(tmp_path):
    """An explicit empty `extra={}` is written as `"extra": {}`, not
    silently dropped. Truthiness-based drop was a bug because every other
    optional field (`patch_id`, `step`, `belief_before`, `run_id`) uses
    `is not None` — `extra` should be consistent."""
    log_path = tmp_path / "log.jsonl"
    tele = TelemetryLogger(log_path, dataset="d", model_name="m")
    tele.record(action_type="generate", runtime_s=0.0,
                instance_id="i", extra={})
    tele.close()
    rec = json.loads(log_path.read_text().strip())
    assert "extra" in rec
    assert rec["extra"] == {}


def test_record_extra_none_is_omitted(tmp_path):
    """The other side of the contract: extra=None (the default) means
    'no extra field in the row'. Pins the slim-row behavior for the
    common calibration case where the call-site doesn't pass `extra=`."""
    log_path = tmp_path / "log.jsonl"
    tele = TelemetryLogger(log_path, dataset="d", model_name="m")
    tele.record(action_type="generate", runtime_s=0.0, instance_id="i")
    tele.close()
    rec = json.loads(log_path.read_text().strip())
    assert "extra" not in rec
