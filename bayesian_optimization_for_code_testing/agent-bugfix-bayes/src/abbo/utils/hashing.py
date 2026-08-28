"""Content hashing for prompt versioning and deduplication."""

from __future__ import annotations

import hashlib


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest of *text*."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = 12) -> str:
    """Return a truncated SHA-256 hex digest of *text*."""
    return sha256_hex(text)[:length]
