#!/usr/bin/env python3
"""Collect final LLM logprob scores, Bayesian state, and quality labels.

This is a non-invasive companion to ``run_synthesis_live.py``. It uses the
same synthesis benchmark dispatcher, costs, transition kernel handling, and
ABBO ``DPPlanner`` loop, but adds logprob collection for every generation.

Outputs, per generator:

    final_logprob_bayes_quality.csv
        llm_perplexity,llm_log_seq_prob,bayes_state,quality

    generation_trajectory_scores.csv
        one row per generated candidate, including both logprob scores and
        the Bayesian state before/after that generation.

Definitions requested for this experiment:
  - llm_perplexity: mean token logprob.
  - llm_log_seq_prob: sum token logprob.
  - bayes_state: synthesis-live Bayesian belief before the final quality label.
  - quality: 1 if the final candidate passes the verifier, else 0.

Example:
  python scripts/collect_final_confidence_bayes_quality.py \
      --src-dir data/mbpp_cv/fold0 \
      --benchmark mbpp \
      --generators qwen3_coder \
      --policies bayesian_DP \
      --n-instances 50
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


def _configure_writable_hf_cache() -> None:
    """Keep HuggingFace downloads in the user's writable cache."""
    root = Path(
        os.environ.get("ORCH_HF_CACHE_DIR", str(Path.home() / "hf_cache"))
    ).expanduser()
    paths = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "hub",
        "HF_DATASETS_CACHE": root / "datasets",
        "XDG_CACHE_HOME": root / "xdg",
    }
    force = os.environ.get("ORCH_FORCE_HF_CACHE", "") == "1"
    for name, path in paths.items():
        current = os.environ.get(name, "")
        if force or not current:
            os.environ[name] = str(path)
        Path(os.environ[name]).expanduser().mkdir(parents=True, exist_ok=True)

    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None:
        for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            if hasattr(hub_constants, name):
                setattr(hub_constants, name, os.environ[name])
    datasets_config = sys.modules.get("datasets.config")
    if datasets_config is not None and hasattr(datasets_config, "HF_DATASETS_CACHE"):
        datasets_config.HF_DATASETS_CACHE = Path(os.environ["HF_DATASETS_CACHE"])


_configure_writable_hf_cache()

import run_synthesis_live as synth
from cost_tracker import CostTracker
from lcb_calibrate import (
    GENERATORS,
    _make_client,
    canonical_generator_key,
    cost_for_call,
    extract_code,
)


POLICIES = ("bayesian_DP", "bayesian_greedy")
BENCHMARKS = ("mbpp", "humaneval", "lcb_easy", "lcb_medium", "lcb_hard")


@dataclass
class LogprobStats:
    mean_token_prob: float | None = None
    seq_prob: float | None = None
    mean_logprob: float | None = None
    seq_logprob: float | None = None
    n_logprob_tokens: int = 0
    logprobs_supported: bool = False
    logprobs_error: str = ""

    def confidence(self, mode: str) -> float | None:
        if mode == "seq_prob":
            return self.seq_prob
        if mode == "mean_token_prob":
            return self.mean_token_prob
        if mode == "perplexity":
            return self.mean_logprob
        if mode == "log_seq_prob":
            return self.seq_logprob
        raise ValueError(f"unknown confidence mode: {mode}")


@dataclass
class GeneratedCode:
    code: str | None
    api_cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    confidence: float | None
    logprob_stats: LogprobStats
    raw_response_chars: int = 0


@dataclass
class GenerationTraceRow:
    benchmark: str
    generator: str
    model_id: str
    policy: str
    instance_id: str
    patch_idx: int
    action_step: int
    bayes_state_at_generation: float
    bayes_state_after_generation: float
    llm_perplexity: float | None
    llm_log_seq_prob: float | None
    llm_mean_token_prob: float | None
    llm_seq_prob: float | None
    n_logprob_tokens: int
    logprobs_supported: bool
    logprobs_error: str
    prompt_tokens: int
    completion_tokens: int
    api_cost_usd: float
    candidate_chars: int
    raw_response_chars: int
    is_final_candidate: bool = False
    final_quality: int | None = None
    final_bayes_state: float | None = None
    final_action: str = ""


@dataclass
class FinalSample:
    benchmark: str
    generator: str
    model_id: str
    policy: str
    instance_id: str
    llm_confidence: float | None
    llm_perplexity: float | None
    llm_log_seq_prob: float | None
    bayes_state: float
    bayes_state_before_final_label: float
    bayes_state_after_final_label: float
    quality: int
    final_action: str
    label_source: str
    decision_cost: float
    api_cost_usd: float
    n_llm_calls: int
    n_critic_runs: int
    n_full_tests: int
    n_label_verifier_runs: int
    wall_clock: float
    prior_Y1: float
    initial_prior: float
    kernel_source: str
    critic_costs: dict[str, float]
    confidence_mode: str
    logprob_stats: dict[str, Any]
    actions: list[dict[str, Any]]
    generation_trace: list[dict[str, Any]]


class LogprobRequestState:
    """Remember provider logprob support to avoid repeated failed retries."""

    def __init__(
        self,
        *,
        request_logprobs: bool,
        fallback_without_logprobs: bool,
        top_logprobs: int | None,
        provider_order: list[str],
        provider_only: list[str],
        require_parameters: bool,
        allow_fallbacks: bool | None,
        require_logprobs_in_response: bool,
        retry_attempts: int,
        retry_sleep: float,
    ) -> None:
        self.enabled = request_logprobs
        self.fallback_without_logprobs = fallback_without_logprobs
        self.top_logprobs = top_logprobs
        self.provider_order = provider_order
        self.provider_only = provider_only
        self.require_parameters = require_parameters
        self.allow_fallbacks = allow_fallbacks
        self.require_logprobs_in_response = require_logprobs_in_response
        self.retry_attempts = retry_attempts
        self.retry_sleep = retry_sleep
        self.disabled_reason = ""


class LogprobUnavailableError(RuntimeError):
    """Raised when a required token-logprob response cannot be obtained."""


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from dicts, pydantic/openai objects, or model_extra."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    if value is not default:
        return value
    extra = getattr(obj, "model_extra", None)
    if isinstance(extra, dict):
        return extra.get(name, default)
    return default


def _first(items: Any) -> Any:
    if not items:
        return None
    if isinstance(items, list | tuple):
        return items[0] if items else None
    try:
        return items[0]
    except (TypeError, KeyError, IndexError):
        return None


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except TypeError:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    return repr(obj)


def _response_debug_summary(resp: Any) -> dict[str, Any]:
    data = _jsonable(resp)
    if not isinstance(data, dict):
        return {"response_type": type(resp).__name__, "response": data}
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    if not isinstance(choice, dict):
        choice = {}
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    return {
        "id": data.get("id"),
        "model": data.get("model"),
        "provider": data.get("provider"),
        "object": data.get("object"),
        "choice_keys": sorted(choice.keys()),
        "logprobs_type": type(logprobs).__name__,
        "logprobs_keys": sorted(logprobs.keys()) if isinstance(logprobs, dict) else None,
        "logprobs_content_len": len(content) if isinstance(content, list) else None,
        "finish_reason": choice.get("finish_reason"),
    }


def safe_stem(text: Any, max_len: int = 160) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (stem or "item")[:max_len]


def problem_id(problem: dict[str, Any]) -> str:
    for key in ("question_id", "instance_id", "task_id", "id"):
        if key in problem:
            return str(problem[key])
    raise KeyError("problem has no question_id/instance_id/task_id/id")


def logprob_stats_from_response(resp: Any, error: str = "") -> LogprobStats:
    logps: list[float] = []

    choices = _field(resp, "choices", [])
    choice = _first(choices)
    logprobs_obj = _field(choice, "logprobs")

    # Chat-completions format:
    # choices[0].logprobs.content[*].logprob
    content = _field(logprobs_obj, "content")
    if content:
        for item in content:
            lp = _safe_float(_field(item, "logprob"))
            if lp is None or lp <= -9998:
                continue
            logps.append(lp)

    # Legacy completions/OpenRouter passthrough format:
    # choices[0].logprobs.token_logprobs
    if not logps:
        token_logprobs = _field(logprobs_obj, "token_logprobs")
        if token_logprobs:
            for item in token_logprobs:
                lp = _safe_float(item)
                if lp is None or lp <= -9998:
                    continue
                logps.append(lp)

    if not logps:
        return LogprobStats(logprobs_supported=False, logprobs_error=error)

    seq_logprob = float(sum(logps))
    mean_logprob = seq_logprob / len(logps)
    seq_prob = 0.0 if seq_logprob < -745 else float(math.exp(seq_logprob))
    return LogprobStats(
        mean_token_prob=float(math.exp(mean_logprob)),
        seq_prob=seq_prob,
        mean_logprob=mean_logprob,
        seq_logprob=seq_logprob,
        n_logprob_tokens=len(logps),
        logprobs_supported=True,
        logprobs_error=error,
    )


def create_completion(
    client: Any,
    *,
    model_id: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    logprob_state: LogprobRequestState,
) -> tuple[Any, str]:
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if logprob_state.enabled:
        kwargs["logprobs"] = True
        if logprob_state.top_logprobs is not None:
            kwargs["top_logprobs"] = logprob_state.top_logprobs

    provider: dict[str, Any] = {}
    if logprob_state.provider_order:
        provider["order"] = logprob_state.provider_order
    if logprob_state.provider_only:
        provider["only"] = logprob_state.provider_only
    if logprob_state.require_parameters:
        provider["require_parameters"] = True
    if logprob_state.allow_fallbacks is not None:
        provider["allow_fallbacks"] = logprob_state.allow_fallbacks
    if provider:
        kwargs["extra_body"] = {"provider": provider}

    try:
        return client.chat.completions.create(**kwargs), ""
    except Exception as exc:
        if (
            not logprob_state.enabled
            or not logprob_state.fallback_without_logprobs
            or logprob_state.require_logprobs_in_response
        ):
            raise
        logprob_state.enabled = False
        logprob_state.disabled_reason = f"{type(exc).__name__}: {exc}"
        kwargs.pop("logprobs", None)
        kwargs.pop("top_logprobs", None)
        kwargs.pop("extra_body", None)
        return client.chat.completions.create(**kwargs), logprob_state.disabled_reason


def write_missing_logprobs_debug(
    *,
    resp: Any,
    raw_dir: Path,
    instance_id: str,
    policy: str,
    llm_call_idx: int,
    attempt: int,
    model_id: str,
    logprob_state: LogprobRequestState,
    reason: str,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    provider: dict[str, Any] = {}
    if logprob_state.provider_order:
        provider["order"] = logprob_state.provider_order
    if logprob_state.provider_only:
        provider["only"] = logprob_state.provider_only
    if logprob_state.require_parameters:
        provider["require_parameters"] = True
    if logprob_state.allow_fallbacks is not None:
        provider["allow_fallbacks"] = logprob_state.allow_fallbacks

    payload = {
        "reason": reason,
        "model_id": model_id,
        "request": {
            "logprobs": bool(logprob_state.enabled),
            "top_logprobs": logprob_state.top_logprobs,
            "provider": provider,
        },
        "summary": _response_debug_summary(resp),
        "response": _jsonable(resp),
    }
    path = (
        raw_dir
        / f"{safe_stem(instance_id)}__{policy}__g{llm_call_idx}"
        / f"missing_logprobs_attempt{attempt}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def generate_code_with_confidence(
    *,
    problem: dict[str, Any],
    prompt_fn: Callable[..., str],
    previous_code: str | None,
    feedback: str | None,
    client: Any,
    model_id: str,
    gen_key: str,
    policy: str,
    instance_id: str,
    llm_call_idx: int,
    tracker: CostTracker,
    raw_dir: Path,
    temperature: float,
    max_tokens: int,
    confidence_mode: str,
    logprob_state: LogprobRequestState,
) -> GeneratedCode:
    if tracker.capped:
        tracker.note_skipped()
        return GeneratedCode(None, 0.0, 0, 0, None, LogprobStats())

    prompt = prompt_fn(problem, prev_code=previous_code, feedback=feedback)
    attempts = 1
    if logprob_state.enabled and logprob_state.require_logprobs_in_response:
        attempts = max(1, logprob_state.retry_attempts)

    resp = None
    stats = LogprobStats()
    logprob_error = ""
    last_error = ""
    for attempt in range(1, attempts + 1):
        try:
            resp, logprob_error = create_completion(
                client,
                model_id=model_id,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                logprob_state=logprob_state,
            )
            stats = logprob_stats_from_response(resp, error=logprob_error)
            if logprob_state.enabled and not stats.logprobs_supported and not stats.logprobs_error:
                stats.logprobs_error = "missing_logprobs_in_response"
            if (
                not logprob_state.enabled
                or not logprob_state.require_logprobs_in_response
                or stats.logprobs_supported
            ):
                break
            last_error = stats.logprobs_error or "missing_logprobs_in_response"
            write_missing_logprobs_debug(
                resp=resp,
                raw_dir=raw_dir,
                instance_id=instance_id,
                policy=policy,
                llm_call_idx=llm_call_idx,
                attempt=attempt,
                model_id=model_id,
                logprob_state=logprob_state,
                reason=last_error,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if not (logprob_state.enabled and logprob_state.require_logprobs_in_response):
                raise

        if attempt < attempts:
            wait_s = max(0.0, logprob_state.retry_sleep) * attempt
            print(
                f"[{gen_key}/{policy}/{instance_id}/g{llm_call_idx}] "
                f"logprobs unavailable on attempt {attempt}/{attempts}: "
                f"{last_error}; retrying in {wait_s:.1f}s",
                flush=True,
            )
            time.sleep(wait_s)

    if resp is None:
        raise LogprobUnavailableError(last_error or "logprobs request failed")
    if (
        logprob_state.enabled
        and logprob_state.require_logprobs_in_response
        and not stats.logprobs_supported
    ):
        raise LogprobUnavailableError(last_error or "missing_logprobs_in_response")

    text = resp.choices[0].message.content or ""
    code = extract_code(text) or text
    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    api_cost = cost_for_call(model_id, prompt_tokens, completion_tokens)
    confidence = stats.confidence(confidence_mode)

    tracker.record(
        api_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        instance_id=instance_id,
        patch_id=llm_call_idx,
        extra={
            "kind": "generate",
            "policy": policy,
            "confidence": confidence,
            "confidence_mode": confidence_mode,
            "llm_perplexity": stats.mean_logprob,
            "llm_log_seq_prob": stats.seq_logprob,
            "logprobs_supported": stats.logprobs_supported,
        },
    )

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{safe_stem(instance_id)}__{policy}__g{llm_call_idx}.txt"
    raw_path.write_text(text)
    return GeneratedCode(
        code=code,
        api_cost_usd=api_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        confidence=confidence,
        logprob_stats=stats,
        raw_response_chars=len(text),
    )


def make_generation_trace_row(
    *,
    benchmark: str,
    gen_key: str,
    model_id: str,
    policy: str,
    instance_id: str,
    patch_idx: int,
    action_step: int,
    belief_before: float,
    belief_after: float,
    generated: GeneratedCode,
) -> GenerationTraceRow:
    stats = generated.logprob_stats
    return GenerationTraceRow(
        benchmark=benchmark,
        generator=gen_key,
        model_id=model_id,
        policy=policy,
        instance_id=instance_id,
        patch_idx=patch_idx,
        action_step=action_step,
        bayes_state_at_generation=float(belief_before),
        bayes_state_after_generation=float(belief_after),
        llm_perplexity=stats.mean_logprob,
        llm_log_seq_prob=stats.seq_logprob,
        llm_mean_token_prob=stats.mean_token_prob,
        llm_seq_prob=stats.seq_prob,
        n_logprob_tokens=stats.n_logprob_tokens,
        logprobs_supported=stats.logprobs_supported,
        logprobs_error=stats.logprobs_error,
        prompt_tokens=generated.prompt_tokens,
        completion_tokens=generated.completion_tokens,
        api_cost_usd=generated.api_cost_usd,
        candidate_chars=len(generated.code or ""),
        raw_response_chars=generated.raw_response_chars,
    )


def normalize_theta(table: dict[str, Any]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, lk in table.get("critic_likelihoods", {}).items():
        p1 = lk.get("p_pass_y1", lk.get("P_pass_given_Y1"))
        p0 = lk.get("p_pass_y0", lk.get("P_pass_given_Y0"))
        if p1 is None or p0 is None:
            continue
        out[name] = {"p_pass_y1": float(p1), "p_pass_y0": float(p0)}
    return out


def load_hand_theta(path: str) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    theta = normalize_theta(json.loads(Path(path).read_text()))
    if not theta:
        raise SystemExit(f"no usable critic likelihoods in {path}")
    return theta


CRITIC_ALIASES = {
    "L0": "L0_syntax",
    "L1": "L1_lint",
    "L2": "L2_public_tests",
    "L3": "L3_llm_review",
}


def filter_theta_critics(
    theta: dict[str, dict[str, float]],
    allowlist_raw: str,
) -> dict[str, dict[str, float]]:
    if not allowlist_raw:
        return theta
    wanted = []
    for raw in parse_csv(allowlist_raw):
        wanted.append(CRITIC_ALIASES.get(raw, raw))
    out = {name: theta[name] for name in wanted if name in theta}
    missing = [name for name in wanted if name not in theta]
    if missing:
        raise SystemExit(
            f"--critic-allowlist contains missing critics {missing}; "
            f"available: {sorted(theta)}"
        )
    if not out:
        raise SystemExit("--critic-allowlist removed every critic")
    return out


CRITIC_COST_KEYS = {
    "L0_syntax": "L0",
    "L1_lint": "L1",
    "L2_public_tests": "L2",
    "L3_llm_review": "L3",
}


def q_critic_one_step(
    belief: float,
    critic_name: str,
    theta: dict[str, dict[str, float]],
    costs: synth.Costs,
    critic_costs: dict[str, float],
) -> float:
    lk = theta[critic_name]
    p_pass = lk["p_pass_y1"] * belief + lk["p_pass_y0"] * (1 - belief)
    b_pass = lk["p_pass_y1"] * belief / max(p_pass, 1e-12)
    b_fail_denom = (
        (1 - lk["p_pass_y1"]) * belief
        + (1 - lk["p_pass_y0"]) * (1 - belief)
    )
    b_fail = (1 - lk["p_pass_y1"]) * belief / max(b_fail_denom, 1e-12)
    fallback_cost = min(critic_costs.values()) if critic_costs else 1.0
    c_critic = critic_costs.get(critic_name, fallback_cost)
    return (
        -c_critic
        + p_pass * max(0.0, -costs.c_ver + b_pass * costs.reward)
        + (1 - p_pass) * max(0.0, -costs.c_ver + b_fail * costs.reward)
    )


class PerCriticCostDPPlanner(synth.DPPlanner):
    """DPPlanner variant with a separate cost for each critic action."""

    def __init__(
        self,
        *args: Any,
        critic_costs: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.critic_costs = critic_costs or {}

    def _value(self, state: Any) -> float:
        from abbo.realworld.agents.bayes_agent import (
            DPState,
            _GENERATOR_ARMS,
            bayes_update,
            discretize,
            generator_transition,
            grid_belief,
        )

        if state in self._cache:
            return self._cache[state][0]

        b = grid_belief(state.belief_idx)
        best_val = 0.0
        best_action = "bail_out"

        if state.ver_left > 0:
            b_fail_idx = discretize(0.05)
            next_after_fail = DPState(
                belief_idx=b_fail_idx,
                gen_left=state.gen_left,
                crit_used=state.crit_used,
                ver_left=state.ver_left - 1,
            )
            v_after_fail = self._value(next_after_fail) if state.ver_left > 1 else 0.0
            q_ver = (
                -self.costs.c_full_test
                + b * self.costs.reward
                + (1 - b) * v_after_fail
            )
            if q_ver > best_val:
                best_val = q_ver
                best_action = "verify"

        if state.gen_left > 0:
            if self.transition_kernel is not None:
                p_fix = self.transition_kernel["p_fix_broken"]
                p_break = self.transition_kernel["p_break_correct"]
                b_next = b * (1 - p_break) + (1 - b) * p_fix
                next_state = DPState(
                    belief_idx=discretize(b_next),
                    gen_left=state.gen_left - 1,
                    crit_used=state.crit_used,
                    ver_left=state.ver_left,
                )
                q_gen = -self.costs.c_llm_call + self._value(next_state)
                if q_gen > best_val:
                    best_val = q_gen
                    best_action = "generate:measured_kernel"
            else:
                for arm in _GENERATOR_ARMS:
                    b_next = generator_transition(b, arm)
                    next_state = DPState(
                        belief_idx=discretize(b_next),
                        gen_left=state.gen_left - 1,
                        crit_used=state.crit_used,
                        ver_left=state.ver_left,
                    )
                    q_gen = -self.costs.c_llm_call + self._value(next_state)
                    if q_gen > best_val:
                        best_val = q_gen
                        best_action = f"generate:{arm}"

        for critic_name in self._critic_names:
            if critic_name in state.crit_used:
                continue
            lk = self.critic_likelihoods[critic_name]
            p_pass = lk["p_pass_y1"] * b + lk["p_pass_y0"] * (1 - b)

            b_if_pass = bayes_update(
                b, critic_name, passed=True, likelihoods=self.critic_likelihoods
            )
            b_if_fail = bayes_update(
                b, critic_name, passed=False, likelihoods=self.critic_likelihoods
            )
            next_crit_used = state.crit_used | frozenset([critic_name])
            next_pass = DPState(
                belief_idx=discretize(b_if_pass),
                gen_left=state.gen_left,
                crit_used=next_crit_used,
                ver_left=state.ver_left,
            )
            next_fail = DPState(
                belief_idx=discretize(b_if_fail),
                gen_left=state.gen_left,
                crit_used=next_crit_used,
                ver_left=state.ver_left,
            )
            q_crit = (
                -self.critic_costs.get(critic_name, self.costs.c_critic_test)
                + p_pass * self._value(next_pass)
                + (1 - p_pass) * self._value(next_fail)
            )
            if q_crit > best_val:
                best_val = q_crit
                best_action = f"critic:{critic_name}"

        self._cache[state] = (best_val, best_action)
        return best_val


def make_planner(
    *,
    costs: synth.Costs,
    max_generations: int,
    max_verifications: int,
    theta: dict[str, dict[str, float]],
    kernel: dict[str, float],
    critic_costs: dict[str, float],
) -> Any:
    from abbo.realworld.agents.simple_agent import AgentCostConfig

    abbo_costs = AgentCostConfig(
        c_llm_call=costs.c_gen,
        c_critic_test=min(critic_costs.values()) if critic_costs else 1.0,
        c_full_test=costs.c_ver,
        reward=costs.reward,
    )
    planner = PerCriticCostDPPlanner(
        abbo_costs,
        max_generations,
        max_verifications,
        critic_likelihoods=theta,
        transition_kernel=kernel,
        critic_costs=critic_costs,
    )
    planner.solve()
    return planner


def choose_greedy_action(
    *,
    belief: float,
    code_present: bool,
    gen_left: int,
    crit_used: frozenset[str],
    theta: dict[str, dict[str, float]],
    kernel: dict[str, float],
    costs: synth.Costs,
    critic_costs: dict[str, float],
) -> str:
    if not code_present:
        return "generate"

    choices: list[tuple[str, float]] = [
        ("bail_out", 0.0),
        ("verify", -costs.c_ver + belief * costs.reward),
    ]
    for critic_name in theta:
        if critic_name in crit_used:
            continue
        q_critic = q_critic_one_step(belief, critic_name, theta, costs, critic_costs)
        choices.append((f"critic:{critic_name}", q_critic))

    if gen_left > 0:
        b_after = synth.kernel_update(belief, kernel)
        q_generate = -costs.c_gen - costs.c_ver + b_after * costs.reward
        choices.append(("generate", q_generate))

    return max(choices, key=lambda item: item[1])[0]


def run_one_sample(
    *,
    problem: dict[str, Any],
    prompt_fn: Callable[..., str],
    critics_fn: Callable[..., dict[str, bool]],
    verify_fn: Callable[..., tuple[bool, str]],
    benchmark: str,
    gen_key: str,
    model_id: str,
    policy: str,
    client: Any,
    critic_client: Any | None,
    theta: dict[str, dict[str, float]],
    kernel: dict[str, float],
    kernel_source: str,
    costs: synth.Costs,
    critic_costs: dict[str, float],
    tracker: CostTracker,
    run_dir: Path,
    max_generations: int,
    max_verifications: int,
    max_actions: int,
    temperature: float,
    max_tokens: int,
    confidence_mode: str,
    logprob_state: LogprobRequestState,
    label_nonverified: bool,
    prior_y1: float,
    initial_prior: float,
    action_log_path: Path | None = None,
) -> FinalSample:
    start = time.perf_counter()
    inst_id = problem_id(problem)
    raw_dir = run_dir / gen_key / "raw_responses"

    planner = None
    if policy == "bayesian_DP":
        planner = make_planner(
            costs=costs,
            max_generations=max_generations,
            max_verifications=max_verifications,
            theta=theta,
            kernel=kernel,
            critic_costs=critic_costs,
        )
    elif policy != "bayesian_greedy":
        raise ValueError(f"unknown policy: {policy}")

    actions: list[dict[str, Any]] = []
    generation_trace: list[GenerationTraceRow] = []
    code: str | None = None
    feedback: str | None = None
    cached_critics: dict[str, bool] = {}
    crit_used: frozenset[str] = frozenset()
    belief = float(initial_prior)
    gen_left = int(max_generations)
    ver_left = int(max_verifications)
    api_cost_usd = 0.0
    decision_cost = 0.0
    final_action = ""
    label_source = "missing"
    quality: int | None = None
    bayes_state_before_final_label: float | None = None
    bayes_state_after_final_label: float | None = None
    n_llm_calls = 0
    n_critic_runs = 0
    n_full_tests = 0
    n_label_verifier_runs = 0
    last_confidence: float | None = None
    last_logprob_stats = LogprobStats()

    def record_action(row: dict[str, Any]) -> None:
        actions.append(row)
        if action_log_path is None:
            return
        append_jsonl(action_log_path, {
            "ts": time.time(),
            "benchmark": benchmark,
            "generator": gen_key,
            "model_id": model_id,
            "policy": policy,
            "instance_id": inst_id,
            **row,
        })

    for step in range(max_actions):
        if code is None:
            action = "generate:initial"
        elif policy == "bayesian_DP":
            action, _q = planner.choose_action(belief, gen_left, crit_used, ver_left)
        else:
            action = choose_greedy_action(
                belief=belief,
                code_present=code is not None,
                gen_left=gen_left,
                crit_used=crit_used,
                theta=theta,
                kernel=kernel,
                costs=costs,
                critic_costs=critic_costs,
            )

        if action == "bail_out":
            final_action = "bail"
            record_action({"step": step, "action": "bail_out", "belief": belief})
            break

        if action == "verify":
            belief_before_label = belief
            ok, info = verify_fn(code, problem)
            quality = int(bool(ok))
            label_source = f"policy_verify:{info}"
            bayes_state_before_final_label = float(belief_before_label)
            bayes_state_after_final_label = float(belief)
            n_full_tests += 1
            decision_cost += costs.c_ver
            ver_left -= 1
            action_row = {
                "step": step,
                "action": "verify",
                "belief": belief_before_label,
                "belief_before_label": belief_before_label,
                "ok": bool(ok),
                "info": info,
            }
            if ok:
                final_action = "verify_pass"
                record_action(action_row)
                break
            feedback = f"Hidden tests failed ({info})."
            belief = 0.05
            bayes_state_after_final_label = float(belief)
            action_row["belief_after_label"] = belief
            record_action(action_row)
            cached_critics = {}
            crit_used = frozenset()
            continue

        if action.startswith("critic:"):
            critic_name = action.split(":", 1)[1]
            if critic_name not in theta:
                record_action({
                    "step": step,
                    "action": action,
                    "belief": belief,
                    "skipped": True,
                    "reason": "missing_likelihood",
                })
                continue
            if not cached_critics:
                cached_critics = critics_fn(
                    code,
                    problem,
                    llm_client=critic_client if critic_client is not None else client,
                )
            passed = bool(cached_critics.get(critic_name, False))
            belief = synth.bayes_update(belief, critic_name, passed, likelihoods=theta)
            n_critic_runs += 1
            fallback_cost = min(critic_costs.values()) if critic_costs else 1.0
            critic_cost = critic_costs.get(critic_name, fallback_cost)
            decision_cost += critic_cost
            crit_used = crit_used | frozenset([critic_name])
            record_action({
                "step": step,
                "action": action,
                "passed": passed,
                "belief": belief,
                "cost": critic_cost,
            })
            continue

        if action.startswith("generate"):
            if gen_left <= 0:
                final_action = "exhausted"
                record_action({
                    "step": step,
                    "action": action,
                    "belief": belief,
                    "skipped": True,
                    "reason": "max_generations_reached",
                })
                break

            belief_before = belief
            generated = generate_code_with_confidence(
                problem=problem,
                prompt_fn=prompt_fn,
                previous_code=code,
                feedback=feedback,
                client=client,
                model_id=model_id,
                gen_key=gen_key,
                policy=policy,
                instance_id=inst_id,
                llm_call_idx=n_llm_calls,
                tracker=tracker,
                raw_dir=raw_dir,
                temperature=temperature,
                max_tokens=max_tokens,
                confidence_mode=confidence_mode,
                logprob_state=logprob_state,
            )
            if generated.code is None:
                final_action = "api_cost_cap"
                quality = 0
                label_source = "no_candidate"
                bayes_state_before_final_label = float(belief)
                bayes_state_after_final_label = float(belief)
                record_action({
                    "step": step,
                    "action": action,
                    "belief": belief,
                    "skipped": True,
                    "reason": "api_cost_cap",
                })
                break

            code = generated.code
            gen_left -= 1
            n_llm_calls += 1
            api_cost_usd += generated.api_cost_usd
            decision_cost += costs.c_gen
            belief = synth.kernel_update(belief, kernel)
            cached_critics = {}
            crit_used = frozenset()
            quality = None
            label_source = "missing"
            bayes_state_before_final_label = None
            bayes_state_after_final_label = None
            last_confidence = generated.confidence
            last_logprob_stats = generated.logprob_stats
            generation_trace.append(make_generation_trace_row(
                benchmark=benchmark,
                gen_key=gen_key,
                model_id=model_id,
                policy=policy,
                instance_id=inst_id,
                patch_idx=n_llm_calls - 1,
                action_step=step,
                belief_before=belief_before,
                belief_after=belief,
                generated=generated,
            ))
            record_action({
                "step": step,
                "action": "generate",
                "belief_before": belief_before,
                "belief": belief,
                "llm_confidence": last_confidence,
                "llm_perplexity": generated.logprob_stats.mean_logprob,
                "llm_log_seq_prob": generated.logprob_stats.seq_logprob,
                "api_cost_usd": generated.api_cost_usd,
                "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens,
                "candidate_chars": len(code),
            })
            continue

        final_action = f"unknown_action:{action}"
        quality = 0
        label_source = final_action
        bayes_state_before_final_label = float(belief)
        bayes_state_after_final_label = float(belief)
        record_action({"step": step, "action": action, "belief": belief, "unknown": True})
        break

    if not final_action:
        final_action = "exhausted"

    if quality is None:
        if label_nonverified and code is not None:
            bayes_state_before_final_label = float(belief)
            ok, info = verify_fn(code, problem)
            quality = int(bool(ok))
            label_source = f"label_verifier:{info}"
            bayes_state_after_final_label = float(belief)
            n_label_verifier_runs += 1
            record_action({
                "step": max_actions,
                "action": "label_verifier",
                "belief": belief,
                "belief_before_label": belief,
                "ok": bool(ok),
                "info": info,
            })
        else:
            quality = 0
            label_source = "unverified_terminal"
            bayes_state_before_final_label = float(belief)
            bayes_state_after_final_label = float(belief)
            record_action({
                "step": max_actions,
                "action": "unverified_terminal",
                "belief": belief,
            })

    if bayes_state_before_final_label is None:
        bayes_state_before_final_label = float(belief)
    if bayes_state_after_final_label is None:
        bayes_state_after_final_label = float(belief)

    if generation_trace:
        generation_trace[-1].is_final_candidate = True
        generation_trace[-1].final_quality = int(quality)
        generation_trace[-1].final_bayes_state = float(bayes_state_before_final_label)
        generation_trace[-1].final_action = final_action

    return FinalSample(
        benchmark=benchmark,
        generator=gen_key,
        model_id=model_id,
        policy=policy,
        instance_id=inst_id,
        llm_confidence=last_confidence,
        llm_perplexity=last_logprob_stats.mean_logprob,
        llm_log_seq_prob=last_logprob_stats.seq_logprob,
        bayes_state=float(bayes_state_before_final_label),
        bayes_state_before_final_label=float(bayes_state_before_final_label),
        bayes_state_after_final_label=float(bayes_state_after_final_label),
        quality=int(quality),
        final_action=final_action,
        label_source=label_source,
        decision_cost=float(decision_cost),
        api_cost_usd=float(api_cost_usd),
        n_llm_calls=n_llm_calls,
        n_critic_runs=n_critic_runs,
        n_full_tests=n_full_tests,
        n_label_verifier_runs=n_label_verifier_runs,
        wall_clock=time.perf_counter() - start,
        prior_Y1=float(prior_y1),
        initial_prior=float(initial_prior),
        kernel_source=kernel_source,
        critic_costs=critic_costs,
        confidence_mode=confidence_mode,
        logprob_stats=asdict(last_logprob_stats),
        actions=actions,
        generation_trace=[asdict(row) for row in generation_trace],
    )


def append_final_scores_csv(path: Path, sample: FinalSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["llm_perplexity", "llm_log_seq_prob", "bayes_state", "quality"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            "llm_perplexity": "" if sample.llm_perplexity is None else sample.llm_perplexity,
            "llm_log_seq_prob": (
                "" if sample.llm_log_seq_prob is None else sample.llm_log_seq_prob
            ),
            "bayes_state": sample.bayes_state,
            "quality": sample.quality,
        })


def append_generation_trace_csv(path: Path, sample: FinalSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = [
        "benchmark",
        "generator",
        "model_id",
        "policy",
        "instance_id",
        "patch_idx",
        "action_step",
        "bayes_state_at_generation",
        "bayes_state_after_generation",
        "llm_perplexity",
        "llm_log_seq_prob",
        "llm_mean_token_prob",
        "llm_seq_prob",
        "n_logprob_tokens",
        "logprobs_supported",
        "logprobs_error",
        "prompt_tokens",
        "completion_tokens",
        "api_cost_usd",
        "candidate_chars",
        "raw_response_chars",
        "is_final_candidate",
        "final_quality",
        "final_bayes_state",
        "final_action",
    ]
    with open(path, "a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in sample.generation_trace:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def append_jsonl(path: Path, rows: list[dict[str, Any]] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, dict):
        iterable = [rows]
    else:
        iterable = rows
    with open(path, "a", buffering=1) as fp:
        for row in iterable:
            fp.write(json.dumps(row) + "\n")
        fp.flush()
        os.fsync(fp.fileno())


def load_done_keys(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        inst = rec.get("instance_id")
        pol = rec.get("policy")
        if inst and pol:
            done.add((str(inst), str(pol)))
    return done


def load_existing_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_records": len(samples), "by_policy": {}}
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        by_policy.setdefault(str(row.get("policy")), []).append(row)
    for policy, rows in sorted(by_policy.items()):
        perplexities = [
            float(r["llm_perplexity"])
            for r in rows
            if r.get("llm_perplexity") is not None
        ]
        log_seq_probs = [
            float(r["llm_log_seq_prob"])
            for r in rows
            if r.get("llm_log_seq_prob") is not None
        ]
        n_generation_rows = sum(len(r.get("generation_trace") or []) for r in rows)
        out["by_policy"][policy] = {
            "n": len(rows),
            "n_with_final_logprobs": len(perplexities),
            "n_generation_rows": n_generation_rows,
            "mean_llm_perplexity": (
                sum(perplexities) / len(perplexities) if perplexities else None
            ),
            "mean_llm_log_seq_prob": (
                sum(log_seq_probs) / len(log_seq_probs) if log_seq_probs else None
            ),
            "mean_bayes_state": sum(float(r["bayes_state"]) for r in rows) / len(rows),
            "quality_rate": sum(int(r["quality"]) for r in rows) / len(rows),
        }
    return out


def parse_cap_map(raw: str, generators: list[str]) -> dict[str, float]:
    if "=" not in raw:
        cap = float(raw)
        return {gen: cap for gen in generators}
    out = {gen: math.inf for gen in generators}
    for pair in raw.split(","):
        if not pair.strip():
            continue
        key, value = pair.split("=", 1)
        gen = canonical_generator_key(key.strip())
        out[gen] = float(value)
    return out


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_critic_costs(args: argparse.Namespace) -> dict[str, float]:
    c_l0 = args.c_l0 if args.c_l0 is not None else args.c_critic
    c_l1 = args.c_l1 if args.c_l1 is not None else c_l0
    c_l2 = args.c_l2 if args.c_l2 is not None else args.c_critic
    c_l3 = args.c_l3 if args.c_l3 is not None else args.c_critic
    return {
        "L0_syntax": float(c_l0),
        "L1_lint": float(c_l1),
        "L2_public_tests": float(c_l2),
        "L3_llm_review": float(c_l3),
    }


def make_costs(args: argparse.Namespace, critic_costs: dict[str, float]) -> synth.Costs:
    """Construct run_synthesis_live.Costs across old and new local schemas."""
    import inspect

    params = inspect.signature(synth.Costs).parameters
    kwargs: dict[str, float] = {
        "c_gen": float(args.c_gen),
        "c_ver": float(args.c_ver),
        "reward": float(args.reward),
    }
    if "c_L0" in params:
        kwargs["c_L0"] = critic_costs["L0_syntax"]
    if "c_L1" in params:
        kwargs["c_L1"] = critic_costs["L1_lint"]
    if "c_L2" in params:
        kwargs["c_L2"] = critic_costs["L2_public_tests"]
    if "c_L3" in params:
        kwargs["c_L3"] = critic_costs["L3_llm_review"]
    if "c_critic" in params:
        kwargs["c_critic"] = float(args.c_critic)
    return synth.Costs(**kwargs)


def default_src_dir(benchmark: str) -> Path:
    return synth.ROOT / "data" / f"{benchmark}_full"


def load_ids_from_critic_results(gen_dir: Path) -> list[str]:
    """Fallback sample ids when no synthesis split has been prepared."""
    path = gen_dir / "critic_results.jsonl"
    ids: list[str] = []
    seen: set[str] = set()
    if not path.exists():
        return ids
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            inst_id = problem_id(row)
        except KeyError:
            continue
        if inst_id not in seen:
            seen.add(inst_id)
            ids.append(inst_id)
    return ids


def load_train_test_ids(gen_dir: Path, require_split: bool) -> tuple[list[str], list[str], str]:
    split_path = gen_dir / "split.json"
    if split_path.exists():
        train_ids, test_ids = synth.load_split(gen_dir)
        overlap = sorted(set(train_ids) & set(test_ids))
        if overlap:
            raise SystemExit(
                f"{split_path} has overlapping train/test ids; examples: {overlap[:5]}"
            )
        return train_ids, test_ids, "split.json"
    if require_split:
        raise SystemExit(
            f"{gen_dir} is missing split.json. Run scripts/synthesis_train_test_split.py "
            "or remove --require-split to use all ids from critic_results.jsonl."
        )
    ids = load_ids_from_critic_results(gen_dir)
    if not ids:
        raise SystemExit(
            f"{gen_dir} is missing split.json and has no readable critic_results.jsonl. "
            "Run calibration first, or prepare a synthesis split."
        )
    return [], ids, "critic_results.jsonl"


def ensure_synthesis_inputs(src_dir: Path, gen: str, require_split: bool) -> Path:
    gen_dir = (src_dir / gen).resolve()
    missing = [name for name in ("likelihood_tables.json",) if not (gen_dir / name).exists()]
    if require_split and not (gen_dir / "split.json").exists():
        missing.append("split.json")
    if missing:
        raise SystemExit(
            f"{gen_dir} is missing {', '.join(missing)}. "
            "Run calibration first; for held-out synthesis-live evaluation, also run "
            "scripts/synthesis_train_test_split.py."
        )
    return gen_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src-dir", type=Path, default=None,
                   help="Source dir with <gen>/likelihood_tables.json. If split.json exists, "
                        "the collector uses its test_ids; otherwise it falls back to all ids "
                        "from critic_results.jsonl. Default: data/<benchmark>_full.")
    p.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    p.add_argument("--generators", required=True)
    p.add_argument("--policies", default="bayesian_DP",
                   help="Comma-separated subset of bayesian_DP,bayesian_greedy.")
    p.add_argument("--output-dir", type=Path,
                   default=synth.ROOT / "data" / "final_confidence_bayes_quality")
    p.add_argument("--n-instances", "--n-test", dest="n_instances", type=int, default=0,
                   help="Optional cap on selected instances (0 = all).")
    p.add_argument("--require-split", action="store_true",
                   help="Require run_synthesis_live-style split.json instead of falling back "
                        "to critic_results.jsonl ids.")
    p.add_argument("--hand-theta-from", default="",
                   help="Optional likelihood_tables.json for hand theta.")
    p.add_argument("--theta-source", default="calibrated",
                   choices=["calibrated", "fitted", "hand"],
                   help="Which critic likelihood table to use inside bayesian_DP/"
                        "bayesian_greedy. 'calibrated' and legacy 'fitted' both mean "
                        "<gen>/likelihood_tables.json; this is not the policy name.")
    p.add_argument("--critic-allowlist", default="",
                   help="Optional comma-separated critic subset, e.g. L2 or "
                        "L0_syntax,L2_public_tests. Default uses all critics, matching "
                        "run_synthesis_live.")
    p.add_argument("--kernel-mode", default="measured",
                   choices=["measured", "hardcoded"],
                   help="Transition kernel source, matching run_synthesis_live except "
                        "online updating is intentionally not used for this collector.")
    p.add_argument("--initial-prior", default="fixed_0.5",
                   choices=["fixed_0.5", "fitted"],
                   help="Starting Bayesian belief, same default as run_synthesis_live.")
    p.add_argument("--max-generations", type=int, default=synth.MAX_GENERATORS)
    p.add_argument("--max-verifications", type=int, default=synth.MAX_VERIFICATIONS)
    p.add_argument("--max-actions", type=int, default=16)
    p.add_argument("--temperature", type=float, default=synth.TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=synth.MAX_TOKENS)
    p.add_argument("--max-api-cost-usd-per-model", default="20.0")
    p.add_argument("--c-gen", type=float, default=10.0)
    p.add_argument("--c-critic", type=float, default=1.0,
                   help="Fallback critic cost used when per-critic costs are omitted.")
    p.add_argument("--c-l0", type=float, default=None,
                   help="Cost for L0_syntax. Defaults to --c-critic.")
    p.add_argument("--c-l1", type=float, default=None,
                   help="Cost for L1_lint. Defaults to --c-l0, then --c-critic.")
    p.add_argument("--c-l2", type=float, default=None,
                   help="Cost for L2_public_tests. Defaults to --c-critic.")
    p.add_argument("--c-l3", type=float, default=None,
                   help="Cost for L3_llm_review. Defaults to --c-critic.")
    p.add_argument("--c-ver", type=float, default=5.0)
    p.add_argument("--reward", type=float, default=100.0)
    p.add_argument("--confidence-mode", default="mean_token_prob",
                   choices=["mean_token_prob", "seq_prob", "perplexity", "log_seq_prob"],
                   help="Legacy value written to llm_confidence in metadata. Final CSVs "
                        "always store both perplexity=mean logprob and "
                        "log_seq_prob=sum logprob.")
    p.add_argument("--no-logprobs", action="store_true",
                   help="Do not request generation logprobs; confidence will be empty.")
    p.add_argument("--no-logprob-fallback", action="store_true",
                   help="Fail instead of retrying without logprobs if unsupported.")
    p.add_argument("--top-logprobs", type=int, default=-1,
                   help="Optional top_logprobs value sent with logprobs=True. Default -1 omits it.")
    p.add_argument("--openrouter-provider-order", default="",
                   help="Comma-separated OpenRouter provider order, e.g. DeepSeek,Fireworks.")
    p.add_argument("--openrouter-provider-only", default="",
                   help="Comma-separated OpenRouter provider allowlist.")
    p.add_argument("--openrouter-require-parameters", action="store_true",
                   help="Ask OpenRouter to route only to providers supporting requested params.")
    p.add_argument("--openrouter-no-fallbacks", action="store_true",
                   help="Set OpenRouter provider.allow_fallbacks=false.")
    p.add_argument("--require-logprobs-in-response", action="store_true",
                   help="Fail a generation call if the provider response omits token logprobs.")
    p.add_argument("--logprob-retry-attempts", type=int, default=8,
                   help="Generation attempts before failing when required logprobs are absent.")
    p.add_argument("--logprob-retry-sleep", type=float, default=10.0,
                   help="Base seconds between required-logprob retries; wait grows linearly.")
    p.add_argument("--no-label-nonverified", action="store_true",
                   help="Do not run a label-only verifier for bail/exhausted candidates.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_dir = (args.src_dir or default_src_dir(args.benchmark)).resolve()
    benchmark = args.benchmark

    generators: list[str] = []
    for raw_gen in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen = canonical_generator_key(raw_gen)
        if gen not in generators:
            generators.append(gen)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    unknown_policies = sorted(set(policies) - set(POLICIES))
    if unknown_policies:
        raise SystemExit(f"unknown policies: {unknown_policies}; allowed: {', '.join(POLICIES)}")

    input_dirs = {
        gen: ensure_synthesis_inputs(src_dir, gen, args.require_split)
        for gen in generators
    }

    load_fn, prompt_fn, critics_fn, verify_fn = synth._benchmark_loader(benchmark)
    problems = load_fn()
    by_id: dict[str, dict[str, Any]] = {}
    for problem in problems:
        try:
            by_id[problem_id(problem)] = problem
        except KeyError:
            continue

    output_dir = (args.output_dir / benchmark).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    critic_costs = build_critic_costs(args)
    costs = make_costs(args, critic_costs)
    caps = parse_cap_map(args.max_api_cost_usd_per_model, generators)
    hand_theta = load_hand_theta(args.hand_theta_from)
    label_nonverified = not args.no_label_nonverified

    for gen in generators:
        gen_dir = input_dirs[gen]
        train_ids, test_ids, id_source = load_train_test_ids(gen_dir, args.require_split)
        fitted_theta, prior_y1 = synth.load_theta(gen_dir)
        if args.theta_source == "hand":
            if not hand_theta:
                raise SystemExit("--theta-source hand requires --hand-theta-from")
            theta = hand_theta
        else:
            theta = fitted_theta
        theta = filter_theta_critics(theta, args.critic_allowlist)

        if args.n_instances > 0:
            test_ids = test_ids[:args.n_instances]

        if args.kernel_mode == "hardcoded":
            kernel = synth.DEFAULT_KERNEL.copy()
            kernel_source = "hardcoded"
        else:
            kernel, kernel_source = synth.load_kernel(gen_dir)

        initial_prior = float(prior_y1) if args.initial_prior == "fitted" else synth.PRIOR
        model_id = GENERATORS[gen][0]
        synth.LLM_MODEL = model_id
        client = _make_client(gen)
        critic_client = _make_client(None) if "L3_llm_review" in theta else None

        gen_out_dir = output_dir / gen
        gen_out_dir.mkdir(parents=True, exist_ok=True)
        table_path = gen_out_dir / "final_logprob_bayes_quality.csv"
        trajectory_path = gen_out_dir / "generation_trajectory_scores.csv"
        trajectory_meta_path = gen_out_dir / "generation_trajectory_scores.jsonl"
        action_log_path = gen_out_dir / "controller_actions.jsonl"
        meta_path = gen_out_dir / "final_logprob_bayes_quality.jsonl"
        summary_path = gen_out_dir / "final_logprob_bayes_quality_summary.json"
        done = load_done_keys(meta_path)
        existing_rows = load_existing_rows(meta_path)

        (gen_out_dir / "sample.json").write_text(json.dumps([
            {"instance_id": tid} for tid in test_ids
        ], indent=2))

        tracker = CostTracker(
            name=gen,
            cap_usd=caps[gen],
            log_path=gen_out_dir / "final_confidence_cost_log.jsonl",
        )
        logprob_state = LogprobRequestState(
            request_logprobs=not args.no_logprobs,
            fallback_without_logprobs=not args.no_logprob_fallback,
            top_logprobs=None if args.top_logprobs < 0 else args.top_logprobs,
            provider_order=parse_csv(args.openrouter_provider_order),
            provider_only=parse_csv(args.openrouter_provider_only),
            require_parameters=args.openrouter_require_parameters,
            allow_fallbacks=False if args.openrouter_no_fallbacks else None,
            require_logprobs_in_response=args.require_logprobs_in_response,
            retry_attempts=args.logprob_retry_attempts,
            retry_sleep=args.logprob_retry_sleep,
        )

        print(
            f"[{benchmark}/{gen}] model={model_id} train={len(train_ids)} "
            f"test={len(test_ids)} ids={id_source} prior={prior_y1:.3f} "
            f"initial={initial_prior:.3f} kernel={kernel_source} c_ver={costs.c_ver}",
            flush=True,
        )
        print(
            f"[{benchmark}/{gen}] critic_costs="
            f"L0:{critic_costs['L0_syntax']} "
            f"L1:{critic_costs['L1_lint']} "
            f"L2:{critic_costs['L2_public_tests']} "
            f"L3:{critic_costs['L3_llm_review']}",
            flush=True,
        )

        for i, tid in enumerate(test_ids, 1):
            problem = by_id.get(str(tid))
            if problem is None:
                print(f"[{benchmark}/{gen}] missing test_id={tid}; skipping", flush=True)
                continue
            for policy in policies:
                if (str(tid), policy) in done:
                    continue
                if tracker.capped:
                    print(f"[{gen}] API cost cap reached; stopping generator", flush=True)
                    break
                print(f"[{benchmark}/{gen}/{policy}] {i}/{len(test_ids)} {tid}", flush=True)
                try:
                    sample = run_one_sample(
                        problem=problem,
                        prompt_fn=prompt_fn,
                        critics_fn=critics_fn,
                        verify_fn=verify_fn,
                        benchmark=benchmark,
                        gen_key=gen,
                        model_id=model_id,
                        policy=policy,
                        client=client,
                        critic_client=critic_client,
                        theta=theta,
                        kernel=kernel,
                        kernel_source=kernel_source,
                        costs=costs,
                        critic_costs=critic_costs,
                        tracker=tracker,
                        run_dir=output_dir,
                        max_generations=args.max_generations,
                        max_verifications=args.max_verifications,
                        max_actions=args.max_actions,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        confidence_mode=args.confidence_mode,
                        logprob_state=logprob_state,
                        label_nonverified=label_nonverified,
                        prior_y1=prior_y1,
                        initial_prior=initial_prior,
                        action_log_path=action_log_path,
                    )
                except Exception as exc:
                    append_jsonl(action_log_path, {
                        "ts": time.time(),
                        "benchmark": benchmark,
                        "generator": gen,
                        "model_id": model_id,
                        "policy": policy,
                        "instance_id": str(tid),
                        "action": "exception",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    if args.require_logprobs_in_response and isinstance(exc, LogprobUnavailableError):
                        print(
                            f"[{benchmark}/{gen}/{policy}] required logprobs failed "
                            f"for {tid}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        raise
                    if args.require_logprobs_in_response:
                        print(
                            f"[{benchmark}/{gen}/{policy}] sample failed "
                            f"for {tid}: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        raise
                    sample = FinalSample(
                        benchmark=benchmark,
                        generator=gen,
                        model_id=model_id,
                        policy=policy,
                        instance_id=str(tid),
                        llm_confidence=None,
                        llm_perplexity=None,
                        llm_log_seq_prob=None,
                        bayes_state=initial_prior,
                        bayes_state_before_final_label=initial_prior,
                        bayes_state_after_final_label=initial_prior,
                        quality=0,
                        final_action=f"exception:{type(exc).__name__}",
                        label_source=str(exc),
                        decision_cost=0.0,
                        api_cost_usd=0.0,
                        n_llm_calls=0,
                        n_critic_runs=0,
                        n_full_tests=0,
                        n_label_verifier_runs=0,
                        wall_clock=0.0,
                        prior_Y1=prior_y1,
                        initial_prior=initial_prior,
                        kernel_source=kernel_source,
                        critic_costs=critic_costs,
                        confidence_mode=args.confidence_mode,
                        logprob_stats=asdict(LogprobStats(logprobs_error=str(exc))),
                        actions=[{"error": str(exc)}],
                        generation_trace=[],
                    )

                append_final_scores_csv(table_path, sample)
                append_generation_trace_csv(trajectory_path, sample)
                append_jsonl(trajectory_meta_path, sample.generation_trace)
                append_jsonl(meta_path, asdict(sample))
                row_dict = asdict(sample)
                existing_rows.append(row_dict)
                done.add((str(tid), policy))
                print(
                    "  -> "
                    f"q={sample.quality} "
                    f"b={sample.bayes_state:.3f} "
                    f"perp={sample.llm_perplexity} "
                    f"log_seq={sample.llm_log_seq_prob} "
                    f"final={sample.final_action}",
                    flush=True,
                )
            if tracker.capped:
                break

        summary_path.write_text(json.dumps(summarize_samples(existing_rows), indent=2))
        (gen_out_dir / "final_confidence_cost_summary.json").write_text(
            json.dumps(tracker.snapshot(), indent=2)
        )
        if logprob_state.disabled_reason:
            print(
                f"[{gen}] logprobs disabled after provider error: "
                f"{logprob_state.disabled_reason}",
                flush=True,
            )
        print(f"[{gen}] wrote {table_path}", flush=True)
        print(f"[{gen}] wrote {trajectory_path}", flush=True)


if __name__ == "__main__":
    main()
