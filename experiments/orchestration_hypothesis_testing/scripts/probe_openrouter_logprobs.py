#!/usr/bin/env python3
"""Probe which OpenRouter providers actually return parseable logprobs.

OpenRouter advertises logprobs support per provider, but some providers return
an empty/absent ``choices[0].logprobs`` for a given model even with
``provider.require_parameters=true``. This samples N completions, records which
provider served each and whether logprobs came back in the standard OpenAI
place, so we can pin the run to a confirmed-good provider.

Usage:
  export OPENROUTER_API_KEY=...
  python scripts/probe_openrouter_logprobs.py                     # deepseek-v4-flash, 12 calls
  PROBE_MODEL=deepseek/deepseek-v4-pro PROBE_N=8 python scripts/probe_openrouter_logprobs.py
"""
from __future__ import annotations

import os
from collections import Counter

from openai import OpenAI


def served_provider(resp) -> str:
    for attr in ("provider",):
        val = getattr(resp, attr, None)
        if val:
            return str(val)
    extra = getattr(resp, "model_extra", None) or {}
    return str(extra.get("provider") or "?")


def logprob_token_count(resp) -> int:
    try:
        lp = resp.choices[0].logprobs
    except Exception:
        return 0
    if not lp:
        return 0
    content = getattr(lp, "content", None) or []
    return len(content)


def main() -> None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")
    model = os.environ.get("PROBE_MODEL", "deepseek/deepseek-v4-flash")
    n = int(os.environ.get("PROBE_N", "12"))
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

    provider: dict = {"require_parameters": True}
    order = os.environ.get("OPENROUTER_PROVIDER_ORDER", "").strip()
    if order:
        provider["order"] = [p.strip() for p in order.split(",") if p.strip()]
        provider["allow_fallbacks"] = (
            os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )

    seen: Counter[str] = Counter()
    good: Counter[str] = Counter()
    print(f"model={model}  calls={n}  provider_routing={provider}")
    for i in range(n):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user",
                           "content": "Write a Python function summing a list of ints."}],
                temperature=0.7,
                max_tokens=80,
                logprobs=True,
                top_logprobs=5,
                extra_body={"provider": provider},
            )
        except Exception as exc:
            print(f"{i:2d}: ERROR {type(exc).__name__}: {exc}")
            continue
        prov = served_provider(resp)
        cnt = logprob_token_count(resp)
        seen[prov] += 1
        if cnt > 0:
            good[prov] += 1
        print(f"{i:2d}: provider={prov:<16} logprob_tokens={cnt}")

    print("\n=== per-provider: with_logprobs / total ===")
    for prov in sorted(seen):
        print(f"  {prov:<16} {good[prov]}/{seen[prov]}")
    reliable = [p for p in seen if good[p] == seen[p] and seen[p] > 0]
    print("\nreliable (always returned logprobs):", ", ".join(reliable) or "(none)")


if __name__ == "__main__":
    main()
