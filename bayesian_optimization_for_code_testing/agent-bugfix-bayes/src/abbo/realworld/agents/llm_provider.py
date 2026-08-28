"""LLM provider abstraction — supports Ollama, OpenAI-compatible, and OpenRouter.

Providers:
- "ollama":     Ollama's /api/generate endpoint (localhost:11434)
- "openai":     Any OpenAI-compatible /v1/chat/completions endpoint (vLLM, TGI, etc.)
- "openrouter": https://openrouter.ai/api/v1/chat/completions (OpenAI-compatible
                with Authorization header). Reads OPENROUTER_API_KEY from env.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError


@dataclass
class LLMConfig:
    """Configuration for an LLM provider."""

    provider: str = "ollama"
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout: int = 300


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wall_clock_seconds: float = 0.0


def call_llm(prompt: str, config: LLMConfig | None = None) -> LLMResponse:
    """Call an LLM via the configured provider.

    Args:
        prompt: The prompt text to send.
        config: LLM configuration.

    Returns:
        LLMResponse with the generated text and metadata.
    """
    if config is None:
        config = LLMConfig()

    start = time.perf_counter()

    if config.provider == "ollama":
        return _call_ollama(prompt, config, start)
    elif config.provider == "openai":
        return _call_openai(prompt, config, start)
    elif config.provider == "openrouter":
        return _call_openrouter(prompt, config, start)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _provider_default_base_url(provider: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api"
    if provider == "openai":
        return "http://localhost:8000"
    return "http://localhost:11434"


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def build_llm_config_from_env(
    *,
    default_provider: str = "ollama",
    default_model: str = "qwen2.5:7b",
    default_base_url: str | None = None,
    default_temperature: float = 0.2,
    default_max_tokens: int = 4096,
    default_timeout: int = 300,
) -> LLMConfig:
    """Build an LLMConfig from ABBO_LLM_* env vars with sensible defaults."""
    provider = os.environ.get("ABBO_LLM_PROVIDER", "").strip() or default_provider
    model = os.environ.get("ABBO_LLM_MODEL", "").strip() or default_model
    base_url = os.environ.get("ABBO_LLM_BASE_URL", "").strip()
    if not base_url:
        base_url = default_base_url or _provider_default_base_url(provider)
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        temperature=_get_env_float("ABBO_LLM_TEMPERATURE", default_temperature),
        max_tokens=_get_env_int("ABBO_LLM_MAX_TOKENS", default_max_tokens),
        timeout=_get_env_int("ABBO_LLM_TIMEOUT", default_timeout),
    )


def llm_response_error(response: LLMResponse) -> str | None:
    """Return the provider error message if the response is a transport/config failure."""
    text = (response.text or "").strip()
    if not text.startswith("Error"):
        return None
    if response.prompt_tokens or response.completion_tokens:
        return None
    return text


def call_llm_or_raise(prompt: str, config: LLMConfig | None = None) -> LLMResponse:
    """Call the configured LLM and raise on transport/configuration failures."""
    response = call_llm(prompt, config)
    err = llm_response_error(response)
    if err:
        raise RuntimeError(err)
    return response


def _call_ollama(prompt: str, config: LLMConfig, start: float) -> LLMResponse:
    """Call Ollama's /api/generate endpoint."""
    url = f"{config.base_url}/api/generate"
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    }

    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        return LLMResponse(
            text=f"Error calling Ollama: {e}",
            model=config.model,
            wall_clock_seconds=time.perf_counter() - start,
        )

    elapsed = time.perf_counter() - start
    return LLMResponse(
        text=data.get("response", ""),
        model=data.get("model", config.model),
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        wall_clock_seconds=elapsed,
    )


def _call_openai(prompt: str, config: LLMConfig, start: float) -> LLMResponse:
    """Call an OpenAI-compatible /v1/chat/completions endpoint (vLLM, TGI, etc.)."""
    url = f"{config.base_url}/v1/chat/completions"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        return LLMResponse(
            text=f"Error calling OpenAI endpoint: {e}",
            model=config.model,
            wall_clock_seconds=time.perf_counter() - start,
        )

    elapsed = time.perf_counter() - start
    text = ""
    if data.get("choices"):
        text = data["choices"][0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        model=data.get("model", config.model),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        wall_clock_seconds=elapsed,
    )


_KEY_NAMES = ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPEN_ROUTER")


def _load_openrouter_key() -> str:
    """Look up the OpenRouter API key in env, then in nearby .env files.

    Accepted variable names: OPENROUTER_API_KEY, OPEN_ROUTER_API_KEY, OPEN_ROUTER.
    .env files searched: project root and 3 parents up.
    """
    for k in _KEY_NAMES:
        v = os.environ.get(k, "").strip()
        if v:
            return v

    # Walk up to find a .env
    here = os.path.dirname(os.path.abspath(__file__))
    seen = set()
    for _ in range(8):
        if here in seen:
            break
        seen.add(here)
        env_path = os.path.join(here, ".env")
        if os.path.isfile(env_path):
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, _, v = line.partition("=")
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k in _KEY_NAMES and v:
                            os.environ.setdefault(k, v)
                            return v
            except Exception:
                pass
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return ""


def _call_openrouter(prompt: str, config: LLMConfig, start: float) -> LLMResponse:
    """Call OpenRouter's /api/v1/chat/completions (OpenAI-compatible + auth).

    Reads OPENROUTER_API_KEY from environment. OpenRouter requires a Bearer
    token; HTTP-Referer + X-Title are optional but recommended for
    accounting in the OpenRouter dashboard.
    """
    api_key = _load_openrouter_key()
    if not api_key:
        return LLMResponse(
            text="Error: OpenRouter API key not set "
                 "(checked OPENROUTER_API_KEY, OPEN_ROUTER_API_KEY, OPEN_ROUTER in env and .env)",
            model=config.model,
            wall_clock_seconds=time.perf_counter() - start,
        )

    base_url = config.base_url
    if base_url == "http://localhost:11434":  # default Ollama URL — fix for openrouter
        base_url = "https://openrouter.ai/api"

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/rvz16/agents_with_uncertainty_research",
            "X-Title": "abbo-bayes-pomdp",
        },
        method="POST",
    )

    # Retry-with-backoff for transient rate limits / upstream blips.
    delays = [0, 10, 30, 60, 120]  # cumulative ~220s; covers free-tier 429 windows
    last_err = ""
    data = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            with urlopen(req, timeout=config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except URLError as e:
            last_err = f"URLError: {e}"
            # Retry on rate-limit (429) and 5xx; bail on auth/404
            code = getattr(e, "code", None)
            if code in (401, 403, 404):
                break
            continue
        except Exception as e:
            last_err = f"Parse error: {e}"
            continue

    if data is None:
        return LLMResponse(
            text=f"Error calling OpenRouter: {last_err}",
            model=config.model,
            wall_clock_seconds=time.perf_counter() - start,
        )

    elapsed = time.perf_counter() - start
    text = ""
    if data.get("choices"):
        text = data["choices"][0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    return LLMResponse(
        text=text,
        model=data.get("model", config.model),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        wall_clock_seconds=elapsed,
    )


def is_openai_endpoint_available(base_url: str) -> bool:
    """Check if an OpenAI-compatible endpoint is reachable."""
    try:
        req = Request(f"{base_url}/v1/models")
        with urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running."""
    try:
        req = Request(f"{base_url}/api/tags")
        with urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """List available Ollama models."""
    try:
        req = Request(f"{base_url}/api/tags")
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
