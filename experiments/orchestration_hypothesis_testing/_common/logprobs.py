"""Helpers for requesting and persisting OpenAI/vLLM chat logprobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def should_request_vllm_logprobs(base_url: str | None) -> bool:
    if not base_url:
        return False
    raw = os.environ.get("REQUEST_LOGPROBS")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def top_logprobs_from_env() -> int:
    try:
        return max(0, int(os.environ.get("TOP_LOGPROBS", "0") or "0"))
    except ValueError:
        return 0


def make_chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    request_logprobs: bool = False,
    top_logprobs: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if request_logprobs:
        kwargs["logprobs"] = True
        if top_logprobs is not None and top_logprobs > 0:
            kwargs["top_logprobs"] = int(top_logprobs)
    return kwargs


def _plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return _plain(obj.model_dump())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def extract_completion_logprobs(resp: Any, *, requested: bool = True) -> dict[str, Any]:
    """Return a compact, JSON-serializable view of choice[0].logprobs.

    vLLM follows the OpenAI chat-completions shape:
    choices[0].logprobs.content[i] = {token, logprob, top_logprobs, ...}.
    The fallback also understands the older completion-style
    {tokens, token_logprobs} shape.
    """
    data = _plain(resp)
    choices = data.get("choices") if isinstance(data, dict) else None
    choice = choices[0] if choices else {}
    raw = choice.get("logprobs") if isinstance(choice, dict) else None

    payload: dict[str, Any] = {
        "requested": bool(requested),
        "supported": False,
        "n_tokens": 0,
        "seq_logprob": None,
        "mean_logprob": None,
        "tokens": [],
    }
    if not requested:
        return payload
    if not isinstance(raw, dict) or not raw:
        payload["raw_logprobs"] = raw
        return payload

    tokens: list[dict[str, Any]] = []
    vals: list[float] = []
    content = raw.get("content") or []
    if isinstance(content, list) and content:
        for item in content:
            if not isinstance(item, dict):
                continue
            lp = item.get("logprob")
            row = {
                "token": item.get("token"),
                "logprob": lp,
                "bytes": item.get("bytes"),
            }
            if item.get("top_logprobs") is not None:
                row["top_logprobs"] = item.get("top_logprobs")
            tokens.append(row)
            if isinstance(lp, (int, float)):
                vals.append(float(lp))
    else:
        raw_tokens = raw.get("tokens") or []
        raw_lps = raw.get("token_logprobs") or []
        for tok, lp in zip(raw_tokens, raw_lps):
            tokens.append({"token": tok, "logprob": lp})
            if isinstance(lp, (int, float)):
                vals.append(float(lp))

    payload["supported"] = bool(tokens)
    payload["n_tokens"] = len(vals)
    if vals:
        seq = float(sum(vals))
        payload["seq_logprob"] = seq
        payload["mean_logprob"] = seq / len(vals)
    payload["tokens"] = tokens
    return payload


def logprob_summary_fields(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "logprobs_requested": bool(payload.get("requested")),
        "logprobs_supported": bool(payload.get("supported")),
        "n_logprob_tokens": int(payload.get("n_tokens") or 0),
        "seq_logprob": payload.get("seq_logprob"),
        "mean_logprob": payload.get("mean_logprob"),
    }


def write_logprob_sidecar(
    raw_dir: Path,
    stem: str,
    payload: dict[str, Any],
    *,
    model: str,
    prompt: str,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "model": model,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "logprobs": payload,
    }
    (raw_dir / f"{stem}.logprobs.json").write_text(json.dumps(record, indent=2))
