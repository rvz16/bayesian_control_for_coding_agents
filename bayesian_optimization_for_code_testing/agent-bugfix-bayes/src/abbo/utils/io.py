"""File I/O helpers for YAML, JSON, and CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Save a dict to a YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def save_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of dicts to a CSV file."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV file as a list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def save_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Save records as JSON Lines."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON Lines file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
