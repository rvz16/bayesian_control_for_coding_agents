"""Artifact management for experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_artifact(name: str, data: Any, path: Path) -> Path:
    """Save an artifact to disk.  Returns the saved path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, indent=2, default=str))
    elif isinstance(data, str):
        path.write_text(data)
    elif isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(str(data))
    return path


def collect_episode_artifacts(
    episode_data: dict[str, Any],
    out_dir: Path,
) -> dict[str, Path]:
    """Collect and save all artifacts for an episode.  Returns artifact paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for key, value in episode_data.items():
        if value is None:
            continue
        ext = ".json" if isinstance(value, (dict, list)) else ".txt"
        p = save_artifact(key, value, out_dir / f"{key}{ext}")
        paths[key] = p

    return paths
