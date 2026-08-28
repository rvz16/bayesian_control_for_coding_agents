"""Timing context manager for measuring wall-clock durations."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Stores the elapsed wall-clock time of a timed block."""

    label: str
    elapsed_seconds: float = 0.0


@contextmanager
def timed(label: str = ""):
    """Context manager that measures wall-clock seconds.

    Usage::

        with timed("my_op") as t:
            do_work()
        print(t.elapsed_seconds)
    """
    result = TimingResult(label=label)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start
