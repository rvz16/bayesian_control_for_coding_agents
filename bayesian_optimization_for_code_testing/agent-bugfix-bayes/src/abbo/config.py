"""Configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from abbo.utils.io import load_yaml


def resolve_env_vars(value: str) -> str:
    """Replace ``${VAR}`` patterns with environment variable values."""
    if not isinstance(value, str):
        return value
    result = value
    while "${" in result:
        start = result.index("${")
        end = result.index("}", start)
        var = result[start + 2 : end]
        result = result[:start] + os.environ.get(var, "") + result[end + 1 :]
    return result


def _resolve_dict(d: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _resolve_dict(v)
        elif isinstance(v, str):
            out[k] = resolve_env_vars(v)
        elif isinstance(v, list):
            out[k] = [
                _resolve_dict(x) if isinstance(x, dict)
                else resolve_env_vars(x) if isinstance(x, str)
                else x
                for x in v
            ]
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file with environment variable resolution."""
    raw = load_yaml(path)
    return _resolve_dict(raw)


def project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    here = Path(__file__).resolve().parent
    # src/abbo/config.py -> go up to src/ then up to project root
    return here.parent.parent


def results_dir(benchmark: str) -> Path:
    """Return the results directory for a benchmark, creating it if needed."""
    d = project_root() / "results" / benchmark
    d.mkdir(parents=True, exist_ok=True)
    return d
