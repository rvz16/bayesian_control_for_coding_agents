"""Unit tests for the shared transition-kernel utilities. No API calls."""
from __future__ import annotations

import json
import pathlib
import sys
import threading

import pytest

# Tests live in tests/; the package root (orchestration_hypothesis_testing/) is
# the parent of tests/. Add it to sys.path so we can import from _common/
# directly without installing the package.
_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_ROOT))

from _common.kernel import (  # noqa: E402
    DEFAULT_KERNEL,
    OnlineKernelCalibration,
    compute_transition_kernel_from_pairs,
    kernel_update,
    pairs_from_trajectories,
    resolve_kernel,
)


# ---------------------------------------------------------------------------
# kernel_update — belief propagation
# ---------------------------------------------------------------------------

def test_kernel_update_lowercase_keys():
    k = {"p_fix_broken": 0.6, "p_break_correct": 0.1}
    # b'=b*(1-p_break)+(1-b)*p_fix
    assert kernel_update(0.5, k) == pytest.approx(0.5 * 0.9 + 0.5 * 0.6)


def test_kernel_update_uppercase_keys():
    # Accepts the post-hoc schema directly (used by some callers reading
    # transition_kernel.json's kernel_all blob).
    k = {"P_fix_given_broken": 0.6, "P_break_given_correct": 0.1}
    assert kernel_update(0.5, k) == pytest.approx(0.5 * 0.9 + 0.5 * 0.6)


def test_kernel_update_belief_extremes():
    k = {"p_fix_broken": 0.4, "p_break_correct": 0.2}
    # b=0 → b' = p_fix
    assert kernel_update(0.0, k) == pytest.approx(0.4)
    # b=1 → b' = 1 - p_break
    assert kernel_update(1.0, k) == pytest.approx(0.8)


def test_kernel_update_rejects_missing_keys():
    with pytest.raises(ValueError, match="missing"):
        kernel_update(0.5, {"foo": 0.1})


# ---------------------------------------------------------------------------
# compute_transition_kernel_from_pairs — post-hoc Beta-smoothed estimate
# ---------------------------------------------------------------------------

def test_compute_kernel_empty_pairs_returns_uniform_laplace():
    out = compute_transition_kernel_from_pairs([])
    # Beta(1,1) on zero data → p_fix = 1/(0+2) = 0.5, ditto p_break
    assert out["P_fix_given_broken"] == pytest.approx(0.5)
    assert out["P_break_given_correct"] == pytest.approx(0.5)
    assert out["n_pairs"] == 0
    assert out["raw_counts"] == {"0->0": 0, "0->1": 0, "1->0": 0, "1->1": 0}


def test_compute_kernel_counts_and_smoothing():
    # Pairs: 3× 0→1 (fixes), 1× 0→0 (stay broken),
    #        1× 1→0 (breaks), 2× 1→1 (stay correct)
    pairs = [(0, 1)] * 3 + [(0, 0)] + [(1, 0)] + [(1, 1)] * 2
    out = compute_transition_kernel_from_pairs(pairs)
    # n_broken = 4, k_fix = 3 → (3+1)/(4+2) = 4/6 = 0.6666...
    assert out["P_fix_given_broken"] == pytest.approx(4 / 6)
    # n_correct = 3, k_break = 1 → (1+1)/(3+2) = 2/5
    assert out["P_break_given_correct"] == pytest.approx(2 / 5)
    # Laplace symmetry: stay + change = 1 within each regime
    assert (out["P_fix_given_broken"] + out["P_stay_broken"]
            == pytest.approx(1.0))
    assert (out["P_break_given_correct"] + out["P_stay_correct"]
            == pytest.approx(1.0))
    assert out["n_pairs"] == 7
    assert out["n_broken_observed"] == 4
    assert out["n_correct_observed"] == 3
    assert out["raw_counts"] == {"0->0": 1, "0->1": 3, "1->0": 1, "1->1": 2}
    assert out["smoothing"] == "Beta(1.0,1.0)"


def test_compute_kernel_drops_non_binary_pairs():
    pairs = [(0, 1), (0, None), (None, 1), (0, 2), (1, 1)]
    out = compute_transition_kernel_from_pairs(pairs)
    # Only (0,1) and (1,1) survive
    assert out["n_pairs"] == 2
    assert out["raw_counts"]["0->1"] == 1
    assert out["raw_counts"]["1->1"] == 1


def test_compute_kernel_custom_prior():
    # Strong prior toward p_fix = 0.9 even with no data
    out = compute_transition_kernel_from_pairs([], alpha=9.0, beta=1.0)
    assert out["P_fix_given_broken"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# pairs_from_trajectories — trajectory unrolling helper
# ---------------------------------------------------------------------------

def test_pairs_from_trajectories_basic():
    trajs = [
        [{"Y": 0}, {"Y": 1}, {"Y": 1}],   # 2 pairs: (0,1) and (1,1)
        [{"Y": 1}, {"Y": 0}],             # 1 pair: (1,0)
        [{"Y": 1}],                       # no pairs (single step)
    ]
    pairs = pairs_from_trajectories(trajs)
    assert sorted(pairs) == [(0, 1), (1, 0), (1, 1)]


def test_pairs_from_trajectories_drops_missing_y():
    trajs = [
        [{"Y": 0}, {"Y": None}, {"Y": 1}],   # (0,None) and (None,1) both dropped
        [{"Y": 1}, {"Y": 0}],
    ]
    pairs = pairs_from_trajectories(trajs)
    assert pairs == [(1, 0)]


def test_pairs_from_trajectories_custom_key():
    trajs = [[{"y_hat": 0}, {"y_hat": 1}]]
    assert pairs_from_trajectories(trajs, y_key="y_hat") == [(0, 1)]


# ---------------------------------------------------------------------------
# OnlineKernelCalibration — Beta-Binomial running estimator
# ---------------------------------------------------------------------------

def test_online_kernel_prior_only():
    ok = OnlineKernelCalibration(init_kernel={"p_fix_broken": 0.4,
                                              "p_break_correct": 0.07})
    # No updates yet — falls back to init_kernel
    assert ok.get() == {"p_fix_broken": 0.4, "p_break_correct": 0.07}


def test_online_kernel_accepts_uppercase_init():
    # init_kernel from kernel_all blob in transition_kernel.json
    ok = OnlineKernelCalibration(init_kernel={
        "P_fix_given_broken": 0.42,
        "P_break_given_correct": 0.08,
    })
    assert ok.get()["p_fix_broken"] == pytest.approx(0.42)
    assert ok.get()["p_break_correct"] == pytest.approx(0.08)


def test_online_kernel_update_records_transitions():
    ok = OnlineKernelCalibration()
    ok.update(0, 1)  # fix
    ok.update(0, 1)  # fix
    ok.update(0, 0)  # stay broken
    ok.update(1, 0)  # break
    ok.update(1, 1)  # stay correct
    s = ok.summary()
    assert s["n_broken_observed"] == 3
    assert s["k_fix"] == 2
    assert s["n_correct_observed"] == 2
    assert s["k_break"] == 1
    # Posterior: p_fix = (2+1)/(3+2) = 0.6; p_break = (1+1)/(2+2) = 0.5
    est = ok.get()
    assert est["p_fix_broken"] == pytest.approx(0.6)
    assert est["p_break_correct"] == pytest.approx(0.5)


def test_online_kernel_falls_back_per_regime():
    # If we only observe broken->* transitions, p_break should still come
    # from init_kernel (no n_correct samples yet).
    ok = OnlineKernelCalibration(init_kernel={"p_fix_broken": 0.5,
                                              "p_break_correct": 0.03})
    for _ in range(10):
        ok.update(0, 1)
    est = ok.get()
    # p_fix moved toward 1 (with Laplace from 0.5 toward (10+1)/(10+2))
    assert est["p_fix_broken"] == pytest.approx(11 / 12)
    # p_break unchanged — still the prior
    assert est["p_break_correct"] == pytest.approx(0.03)


def test_online_kernel_posterior_approaches_truth():
    # True p_fix = 0.7, p_break = 0.1 over many samples — posterior mean
    # should land close to truth.
    import random
    rng = random.Random(42)
    ok = OnlineKernelCalibration()
    for _ in range(2000):
        # Half broken, half correct seeds
        y_before = rng.choice([0, 1])
        if y_before == 0:
            y_after = 1 if rng.random() < 0.7 else 0
        else:
            y_after = 0 if rng.random() < 0.1 else 1
        ok.update(y_before, y_after)
    est = ok.get()
    assert abs(est["p_fix_broken"] - 0.7) < 0.03
    assert abs(est["p_break_correct"] - 0.1) < 0.03


def test_online_kernel_thread_safe():
    """8 workers × 100 deterministic updates each. With the (i%2, (i+1)%2)
    pattern, every even i is a (0, 1) "fix" and every odd i is a (1, 0)
    "break". A race that miscategorizes transitions would still hit the
    total count — so this test also checks the per-regime breakdown."""
    ok = OnlineKernelCalibration()
    n_per_thread = 100
    n_threads = 8

    def worker():
        for i in range(n_per_thread):
            # Deterministic per-thread sequence: alternates fix / break
            ok.update(i % 2, (i + 1) % 2)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads: t.start()
    for t in threads: t.join()

    s = ok.summary()
    total = s["n_broken_observed"] + s["n_correct_observed"]
    assert total == n_per_thread * n_threads
    # Each worker emits exactly n_per_thread/2 of each (0,1) and (1,0). No
    # (0,0) or (1,1) — so n_broken == k_fix, and n_correct == k_break.
    expected_per_regime = (n_per_thread // 2) * n_threads
    assert s["n_broken_observed"] == expected_per_regime
    assert s["k_fix"] == expected_per_regime           # all 0→ are 0→1
    assert s["n_correct_observed"] == expected_per_regime
    assert s["k_break"] == expected_per_regime         # all 1→ are 1→0


def test_online_kernel_summary_serializable():
    ok = OnlineKernelCalibration()
    ok.update(0, 1)
    ok.update(1, 0)
    # summary() must be JSON-serializable so we can persist it at end-of-run
    s = ok.summary()
    json.dumps(s)  # no exception


# ---------------------------------------------------------------------------
# resolve_kernel — file-based + mode dispatch
# ---------------------------------------------------------------------------

def test_resolve_kernel_hardcoded_mode(tmp_path):
    k, src, ok = resolve_kernel(tmp_path, mode="hardcoded")
    assert k == DEFAULT_KERNEL
    assert src == "hardcoded"
    assert ok is None


def test_resolve_kernel_measured_with_file(tmp_path):
    # Write a transition_kernel.json with the production schema
    (tmp_path / "transition_kernel.json").write_text(json.dumps({
        "generator": "test",
        "kernel_all": {
            "P_fix_given_broken": 0.42,
            "P_break_given_correct": 0.08,
            "raw_counts": {"0->0": 5, "0->1": 5, "1->0": 1, "1->1": 9},
            "n_pairs": 20,
            "smoothing": "Beta(1,1)",
        },
    }))
    k, src, ok = resolve_kernel(tmp_path, mode="measured")
    assert k == {"p_fix_broken": 0.42, "p_break_correct": 0.08}
    assert src == "measured"
    assert ok is None


def test_resolve_kernel_measured_no_file_falls_back(tmp_path):
    k, src, ok = resolve_kernel(tmp_path, mode="measured")
    assert k == DEFAULT_KERNEL
    assert src == "default"
    assert ok is None


def test_resolve_kernel_online_returns_estimator(tmp_path):
    (tmp_path / "transition_kernel.json").write_text(json.dumps({
        "kernel_all": {"P_fix_given_broken": 0.6, "P_break_given_correct": 0.05},
    }))
    k, src, ok = resolve_kernel(tmp_path, mode="online")
    assert k == {"p_fix_broken": 0.6, "p_break_correct": 0.05}
    assert src == "measured"
    assert isinstance(ok, OnlineKernelCalibration)
    # Estimator starts at the measured kernel
    assert ok.get() == {"p_fix_broken": 0.6, "p_break_correct": 0.05}


def test_resolve_kernel_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown kernel mode"):
        resolve_kernel(pathlib.Path("/tmp"), mode="garbage")


def test_resolve_kernel_explicit_path(tmp_path):
    custom = tmp_path / "my_kernel.json"
    custom.write_text(json.dumps({
        "P_fix_given_broken": 0.55, "P_break_given_correct": 0.02,
    }))
    # No kernel_all wrapper — should still work
    k, src, _ = resolve_kernel(tmp_path, mode="measured", kernel_path=custom)
    assert k == {"p_fix_broken": 0.55, "p_break_correct": 0.02}
    assert src == "measured"


def test_resolve_kernel_malformed_json_raises(tmp_path):
    """A corrupt transition_kernel.json should fail loudly, not silently
    fall back to the default kernel — silent fallback would mask data
    corruption in long-running pipelines."""
    (tmp_path / "transition_kernel.json").write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        resolve_kernel(tmp_path, mode="measured")


def test_resolve_kernel_missing_required_keys_raises(tmp_path):
    """A JSON file that exists but doesn't carry the required P_fix /
    P_break keys should also fail loudly. resolve_kernel currently surfaces
    this as a KeyError from the dict access; this test pins that behavior."""
    (tmp_path / "transition_kernel.json").write_text(json.dumps({
        "kernel_all": {"foo": 0.5}  # neither P_fix_given_broken nor P_break_given_correct
    }))
    with pytest.raises(KeyError):
        resolve_kernel(tmp_path, mode="measured")


# ---------------------------------------------------------------------------
# Validation — input-sanity checks added to OnlineKernelCalibration
# ---------------------------------------------------------------------------

def test_online_kernel_init_rejects_malformed_init_kernel():
    """A malformed init_kernel should raise at construction, not later when
    .get() falls back to the empty-regime defaults."""
    with pytest.raises(ValueError, match="missing required keys"):
        OnlineKernelCalibration(init_kernel={"foo": 0.5})


def test_online_kernel_init_uppercase_only_succeeds():
    """An init_kernel with only the uppercase keys should be normalized to
    lowercase by the constructor — no exception, and .get() returns the
    expected values."""
    ok = OnlineKernelCalibration(init_kernel={
        "P_fix_given_broken": 0.7,
        "P_break_given_correct": 0.03,
    })
    assert ok.get() == {"p_fix_broken": 0.7, "p_break_correct": 0.03}


@pytest.mark.parametrize("y_before,y_after", [
    (None, 1),       # caller bug: forgot to verify
    (2, 1),          # caller bug: wrong dtype
    (0, "ok"),       # caller bug: stringy comparison
    (-1, 0),         # caller bug: signed-int leak
    (0, None),
])
def test_online_kernel_update_rejects_non_binary(y_before, y_after):
    """Pre-validation prevents the silent-corruption bug surface: any
    y != 0 used to fall into the "correct" regime. Now raises ValueError."""
    ok = OnlineKernelCalibration()
    with pytest.raises(ValueError, match="y_before and y_after"):
        ok.update(y_before, y_after)
    # Counts unchanged after the failed call
    assert ok.summary()["n_broken_observed"] == 0
    assert ok.summary()["n_correct_observed"] == 0


@pytest.mark.parametrize("y_before,y_after,exp_broken,exp_fix,exp_correct,exp_break", [
    (True, False, 0, 0, 1, 1),   # bool: True == 1, False == 0 → (1→0) = break
    (False, True, 1, 1, 0, 0),   # bool: (0→1) = fix
    (0.0, 1.0, 1, 1, 0, 0),       # float zero/one accepted (== 0/1)
    (1.0, 0.0, 0, 0, 1, 1),
])
def test_online_kernel_update_accepts_bool_and_float_zero_one(
    y_before, y_after, exp_broken, exp_fix, exp_correct, exp_break,
):
    """The validation uses `==` equality, so bool (True/False) and
    float 0.0/1.0 are accepted in addition to int 0/1 — they all
    round-trip to the same counts. Pins this so the error message
    in update() remains accurate ("must equal 0 or 1", not "must
    be int 0 or 1")."""
    ok = OnlineKernelCalibration()
    ok.update(y_before, y_after)  # no exception
    s = ok.summary()
    assert s["n_broken_observed"] == exp_broken
    assert s["k_fix"] == exp_fix
    assert s["n_correct_observed"] == exp_correct
    assert s["k_break"] == exp_break


# ---------------------------------------------------------------------------
# Asymmetric prior on a populated regime — the stay+change Beta posteriors
# are independent, so with alpha != beta they DO NOT have to sum to 1.
# Docstring claims this; test pins the property numerically so a future
# refactor doesn't silently drop independent smoothing in favor of "1 −
# p_fix" shortcuts.
# ---------------------------------------------------------------------------

def test_compute_kernel_asymmetric_prior_stay_plus_change_drifts_from_one():
    """With alpha=2, beta=8 (strong prior toward stay), independent Beta
    smoothing on each transition gives P_fix + P_stay_broken ≠ 1 — by
    design. Verify numerically that the function honors the asymmetric
    prior on populated counts (not just the all-empty case)."""
    # 1 fix + 1 stay-broken transition with prior (alpha=2, beta=8).
    # Independent smoothing:
    #   P_fix       = (1 + 2) / (2 + 2 + 8) = 3/12 = 0.25
    #   P_stay_brok = (1 + 2) / (2 + 2 + 8) = 3/12 = 0.25
    # Sum is 0.5, NOT 1 — different priors push both probabilities down.
    pairs = [(0, 1), (0, 0)]
    out = compute_transition_kernel_from_pairs(pairs, alpha=2.0, beta=8.0)
    assert out["P_fix_given_broken"] == pytest.approx(0.25)
    assert out["P_stay_broken"] == pytest.approx(0.25)
    # Independent posteriors DO NOT sum to 1 under asymmetric priors
    assert (out["P_fix_given_broken"] + out["P_stay_broken"]
            == pytest.approx(0.5))
