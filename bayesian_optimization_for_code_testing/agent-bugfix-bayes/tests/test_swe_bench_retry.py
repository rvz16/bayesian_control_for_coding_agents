"""Tests for the docker-retry / timeout-tolerance layer in swe_bench.py.

These DO NOT require a real Docker daemon. We replace _sh with a closure
that returns scripted CompletedProcess results, exercising the retry
loop and asserting that:

  * a transient failure followed by success is recovered
  * persistent failure raises after the configured attempt count
  * subprocess.TimeoutExpired never bubbles up — _sh converts it
  * the retry helper sleeps between attempts (using a fake sleep so the
    test runs fast)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Make src importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from abbo.realworld.agents import swe_bench as sb


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_result(rc: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["fake"], returncode=rc,
                                       stdout="", stderr=stderr)


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    """Make time.sleep a no-op so the retry test runs instantly."""
    monkeypatch.setattr(sb.time, "sleep", lambda *_a, **_k: None)


# ----------------------------------------------------------------------
# _sh: TimeoutExpired tolerance
# ----------------------------------------------------------------------

def test_sh_converts_timeout_to_failure_result(monkeypatch):
    """A subprocess.TimeoutExpired must not propagate. _sh should return
    a result with returncode=-1 and a 'timeout' marker in stderr so that
    retry-aware callers can react."""

    def fake_run(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd=["fake"], timeout=1)

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    r = sb._sh(["fake"], timeout=1)
    assert r.returncode == -1
    assert "timeout" in r.stderr.lower()


# ----------------------------------------------------------------------
# _retry: transient → success
# ----------------------------------------------------------------------

def test_retry_recovers_on_second_attempt():
    """First attempt fails, second succeeds. _retry returns success."""
    calls = {"n": 0}
    seq = [_make_result(1, "transient"), _make_result(0)]

    def attempt():
        calls["n"] += 1
        return seq.pop(0)

    r = sb._retry(attempt, attempts=3, label="test", verbose=False)
    assert r.returncode == 0
    assert calls["n"] == 2


def test_retry_gives_up_after_attempts():
    """All attempts fail → _retry returns the last failure (does NOT raise)."""
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        return _make_result(99, "persistent")

    r = sb._retry(attempt, attempts=3, label="test", verbose=False)
    assert r.returncode == 99
    assert calls["n"] == 3
    assert "persistent" in r.stderr


def test_retry_first_call_succeeds_no_extra_calls():
    """First call succeeds → no retries done."""
    calls = {"n": 0}

    def attempt():
        calls["n"] += 1
        return _make_result(0)

    r = sb._retry(attempt, attempts=3, label="test", verbose=False)
    assert r.returncode == 0
    assert calls["n"] == 1


# ----------------------------------------------------------------------
# pull_image: end-to-end with mocked _sh
# ----------------------------------------------------------------------

def test_pull_image_succeeds_after_pull_failure(monkeypatch):
    """`docker image inspect` says missing; first `docker pull` times out;
    second `docker pull` succeeds. pull_image() must NOT raise."""
    state = {"calls": 0}
    pull_seq = [
        _make_result(-1, "timeout after 3600s: "),   # first pull: timeout
        _make_result(0),                              # second pull: ok
    ]

    def fake_sh(cmd, timeout=600, check=False):
        state["calls"] += 1
        # docker image inspect
        if "inspect" in cmd:
            return _make_result(1, "no such image")
        # docker pull
        if "pull" in cmd:
            return pull_seq.pop(0)
        return _make_result(0)

    monkeypatch.setattr(sb, "_sh", fake_sh)
    # Should not raise — second pull succeeds
    sb.pull_image("seaborn-3010", verbose=False)
    assert state["calls"] >= 3  # 1 inspect + 2 pulls (with retry)


def test_pull_image_raises_after_all_attempts(monkeypatch):
    """All `docker pull` attempts return failure → pull_image raises with
    the count in the message so the caller knows it wasn't a one-off."""

    def fake_sh(cmd, timeout=600, check=False):
        if "inspect" in cmd:
            return _make_result(1, "no such image")
        if "pull" in cmd:
            return _make_result(1, "Hub rate limit")
        return _make_result(0)

    monkeypatch.setattr(sb, "_sh", fake_sh)
    with pytest.raises(RuntimeError) as excinfo:
        sb.pull_image("seaborn-3010", verbose=False)
    msg = str(excinfo.value).lower()
    assert "pull failed" in msg
    assert str(sb.PULL_RETRY_ATTEMPTS) in str(excinfo.value)


# ----------------------------------------------------------------------
# start_container: end-to-end with mocked _sh
# ----------------------------------------------------------------------

def test_start_container_succeeds_after_one_retry(monkeypatch):
    """First `docker run` fails (e.g. name conflict / port race), second
    succeeds. start_container should not raise."""
    run_seq = [
        _make_result(125, "name in use"),
        _make_result(0),
    ]

    def fake_sh(cmd, timeout=600, check=False):
        # docker rm -f — always passes (we don't care about its rc)
        if "rm" in cmd:
            return _make_result(0)
        # docker run -d ...
        if cmd[1] == "run":
            return run_seq.pop(0)
        return _make_result(0)

    monkeypatch.setattr(sb, "_sh", fake_sh)
    cname = sb.start_container("seaborn-3010", verbose=False)
    assert cname == sb._container_name("seaborn-3010")
    assert run_seq == []  # both run attempts consumed


def test_start_container_raises_after_persistent_failure(monkeypatch):
    """Every `docker run` fails → start_container raises with the count
    in the message."""
    def fake_sh(cmd, timeout=600, check=False):
        if "rm" in cmd:
            return _make_result(0)
        if cmd[1] == "run":
            return _make_result(1, "containerd: image not found")
        return _make_result(0)

    monkeypatch.setattr(sb, "_sh", fake_sh)
    with pytest.raises(RuntimeError) as excinfo:
        sb.start_container("seaborn-3010", verbose=False)
    msg = str(excinfo.value).lower()
    assert "start failed" in msg
    assert str(sb.START_RETRY_ATTEMPTS) in str(excinfo.value)
