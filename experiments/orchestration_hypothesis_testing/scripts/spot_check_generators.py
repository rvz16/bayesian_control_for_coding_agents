#!/usr/bin/env python3
"""Spot-check generator base rate on SWE-bench Lite.

Goal: pick R1's default generator (per PRE_REGISTRATION.md S4) by measuring
the empirical base rate `P(Y=1)` of three candidates on a 20-instance random
sample of SWE-bench Lite. The pre-registration requires the chosen generator
to land base rate in `[0.30, 0.70]`; outside that window the controller has
no real decisions to make.

Pipeline
--------
1. Sample N_INSTANCES (default 20) from SWE-bench Lite with seed=42.
2. For each generator x instance x patch_id (default 3 per instance):
     - Read oracle files (the files the gold patch touches)
     - Prompt the generator with problem + oracle file content
     - Parse SEARCH/REPLACE blocks into a unified diff
3. Submit predictions to SWE-bench Docker harness for ground-truth Y.
4. Compute base rate per generator and write summary.

Usage
-----
    # full run
    python spot_check_generators.py

    # quick smoke test with 2 instances
    python spot_check_generators.py --n-instances 2 --n-patches 1

    # only generate (skip eval); useful to verify generation works first
    python spot_check_generators.py --skip-eval

    # only evaluate existing predictions (after a generate-only run)
    python spot_check_generators.py --skip-generate
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
    RateLimitError,
)

# Project root (the worktree we're running from) plus the orchestration package
# root that contains _common/, calibration/, and iter/.
ROOT = Path(__file__).resolve().parents[3]
ORCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ORCH_ROOT))
sys.path.insert(0, str(ROOT))

# Local import: thread-safe cost ledger
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from _common.cost import CostTracker, extract_usage, project_cost  # noqa: E402
from _common.logprobs import (  # noqa: E402
    extract_completion_logprobs,
    logprob_summary_fields,
    should_request_vllm_logprobs,
    top_logprobs_from_env,
    write_logprob_sidecar,
)

# Provider exceptions that should NOT be retried; we abort immediately.
# A wrong API key won't fix itself by retrying 599 more times.
FATAL_API_EXCEPTIONS = (AuthenticationError, PermissionDeniedError)
# These are retryable but we still surface them; for our spot-check we just
# treat them as the call failing — no retry loop.
RECOVERABLE_API_EXCEPTIONS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
)

# Load .env: try worktree root first, then walk up looking for a non-empty
# .env. When running inside a git worktree the OPENROUTER_API_KEY typically
# lives in the parent checkout's .env, not the worktree's.
def _load_env_chain() -> None:
    candidates = [ROOT / ".env"]
    cur = ROOT.parent
    for _ in range(5):
        candidates.append(cur / ".env")
        if cur.parent == cur:
            break
        cur = cur.parent
    for env_path in candidates:
        if env_path.exists() and env_path.stat().st_size > 0:
            load_dotenv(env_path, override=False)


_load_env_chain()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spot_check")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "spot_check"

# Generators under test. Each entry maps short name ->
# (model_id, base_url, enable_thinking).
#
# - base_url None = use OpenRouter; otherwise a vLLM-served local OpenAI-
#   compatible endpoint.
# - enable_thinking is forwarded as `chat_template_kwargs` for Qwen3 family
#   models. None means "leave the model's default", True/False forces it.
GENERATORS: dict[str, tuple[str, str | None, bool | None]] = {
    "haiku45":           ("anthropic/claude-haiku-4.5", None,                          None),
    "sonnet45":          ("anthropic/claude-sonnet-4.5", None,                         None),
    "qwen3_coder":       ("qwen/qwen3-coder",           None,                          None),
    "gpt5_mini":         ("openai/gpt-5-mini",          None,                          None),
    "qwen25_7b":         ("Qwen/Qwen2.5-7B-Instruct",   os.environ.get("QWEN25_7B_BASE_URL", "http://127.0.0.1:8001/v1"),   None),
    "qwen3_8b":          ("Qwen/Qwen3-8B",              os.environ.get("QWEN3_8B_BASE_URL", "http://127.0.0.1:8002/v1"),    False),
    "qwen3_8b_thinking": ("Qwen/Qwen3-8B",              os.environ.get("QWEN3_8B_BASE_URL", "http://127.0.0.1:8002/v1"),    True),
    "qwen25_32b":        ("Qwen/Qwen2.5-Coder-32B-Instruct", os.environ.get("QWEN25_32B_BASE_URL", "http://127.0.0.1:8003/v1"), None),
    "gpt_oss_20b_local": ("openai/gpt-oss-20b",         os.environ.get("GPT_OSS_20B_BASE_URL", "http://127.0.0.1:8000/v1"), None),
}

# Per-model cost cap (USD) — used by --max-cost-usd-per-model when not given a
# single uniform value. Sized at ~1.5× the linear-projection estimate from
# Haiku's measured cost ($0.0197/gen × 150 gens = $2.95 for n=50, k=3) and
# OpenRouter list prices for the others. Local vLLM models pay $0; their cap
# is set high so it never triggers.
DEFAULT_COST_CAPS_USD: dict[str, float] = {
    "haiku45":     5.0,
    "sonnet45":   12.0,
    "qwen3_coder": 2.0,
    "gpt5_mini":   2.0,
    "qwen25_7b":  1000.0,
    "qwen25_32b": 1000.0,
    "qwen3_8b":   1000.0,
    "qwen3_8b_thinking": 1000.0,
    "gpt_oss_20b_local": 1000.0,
}

DEFAULT_N_INSTANCES = 20
DEFAULT_N_PATCHES = 3
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 42
DEFAULT_MAX_WORKERS_GEN = 8     # parallel API calls
DEFAULT_MAX_WORKERS_EVAL = 4    # parallel Docker containers
LLM_TIMEOUT_S = 240             # thinking-mode generations can run several minutes
# No real truncation: 10K lines covers all SWE-bench Lite files except the very
# largest django files. Truncating earlier corrupts SEARCH matching because the
# model edits code that isn't in its prompt. Models with small context windows
# (e.g. Qwen2.5-7B on OpenRouter at 32K) will fail with HTTP 400 on big files;
# that's an informative outcome we report rather than paper over.
MAX_FILE_LINES = 10000

PROMPT_TEMPLATE = """\
You are an expert software engineer fixing a bug in the {repo} repository.

## Issue Description
{problem_statement}

{hints_section}

## Files that likely need changes

{file_contents}

## Task
Fix the issue by modifying the file(s) above. Output your changes as a SEARCH/REPLACE
block for each change. The required format is:

<<<CHANGE FILE_PATH_HERE
SEARCH
EXACT_ORIGINAL_LINES_HERE
REPLACE
NEW_LINES_HERE
CHANGE>>>

CRITICAL FORMAT REQUIREMENTS:
1. Replace `FILE_PATH_HERE` with the actual file path of the file you are editing —
   one of the paths shown in the "Files that likely need changes" section above
   (for example `astropy/io/ascii/rst.py`). Do NOT copy the literal text
   `FILE_PATH_HERE`, and do NOT use a placeholder like `path/to/file.py`.
2. Replace `EXACT_ORIGINAL_LINES_HERE` with lines copied **verbatim** from the
   file shown above — same indentation, same whitespace, same line breaks.
   Do NOT paraphrase or summarize the original code.
3. Replace `NEW_LINES_HERE` with the corrected lines.
4. Use exactly three angle brackets: `<<<CHANGE` to open and `CHANGE>>>` to close.
   Do NOT use `<CHANGE ...>` with a single bracket. Do NOT use XML-style
   `</SEARCH>` or `</REPLACE>` closing tags.
5. The keywords `SEARCH`, `REPLACE`, and `CHANGE>>>` must each appear on their
   own line, with no other text on those lines.

You can output multiple CHANGE blocks if needed. Keep changes minimal and focused.
Do NOT include unchanged files. Do NOT output entire files.
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationRecord:
    generator_key: str
    generator_model: str
    instance_id: str
    repo: str
    patch_id: int
    diff: str
    response_chars: int
    error: str = ""
    timestamp: str = ""
    # Cost telemetry (filled when the provider returns usage metadata; local
    # vLLM endpoints leave these at 0).
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extraction_path: str = ""


@dataclass
class GeneratorSummary:
    generator_key: str
    generator_model: str
    n_instances: int
    n_patches_attempted: int
    n_patches_nonempty: int
    n_patches_evaluated: int
    n_correct: int
    base_rate: float
    base_rate_per_instance: float  # mean of per-instance pass-rate
    by_repo: dict[str, dict[str, float | int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Patch generation (oracle retrieval, repo-free)
# ---------------------------------------------------------------------------

def get_changed_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a unified diff."""
    files: list[str] = []
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            files.append(line[6:])
    return files


def _strip_fence(text: str) -> str:
    """Strip a single surrounding ```...``` fence (or leading/trailing fence
    fragments) from a code block body."""
    text = text.strip("\n")
    # Whole-body fenced: ```lang\n...\n```
    m = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n([\s\S]*?)\n?```\s*$", text)
    if m:
        return m.group(1).strip("\n")
    # Trailing-only fence
    text = re.sub(r"\n?```\s*$", "", text).rstrip("\n")
    # Leading-only fence
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*\n", "", text)
    return text


_PLACEHOLDER_PATH_RE = re.compile(
    r"^\s*(?:path/to/file|<filename>|YOUR_FILE|your_file_path|<path>|"
    r"FILE_PATH_HERE|FILE_PATH|<file>|<file_path>)\b",
    re.IGNORECASE,
)


def _normalize_path_token(fpath: str) -> str:
    """Strip noise the LLM sometimes wraps the path in.

    Examples seen in the spot-check raw_responses:
      `path="sympy/printing/latex.py"` -> `sympy/printing/latex.py`
      `'astropy/io/ascii/rst.py'`      -> `astropy/io/ascii/rst.py`
      `"django/forms/widgets.py"`     -> `django/forms/widgets.py`
    """
    p = fpath.strip()
    p = re.sub(r'^(?:path|file|filepath|file_path)\s*=\s*', "", p, flags=re.IGNORECASE)
    p = p.strip().lstrip('"\'').rstrip('"\'>')
    return p


def parse_change_blocks(response: str) -> list[tuple[str, str, str]]:
    """Parse SEARCH/REPLACE blocks, tolerant of multiple sloppy variants.

    Accepts:
      <<<CHANGE path/to/file
      SEARCH
      ...
      REPLACE
      ...
      CHANGE>>>

    and the looser forms small models emit (often inside ```diff fences),
    including bodies whose SEARCH/REPLACE payloads are themselves fenced:

      CHANGE path/to/file
      SEARCH
      ```python
      ...
      ```
      REPLACE
      ```python
      ...
      ```

    and the XML-style form Sonnet 4.5 sometimes emits:

      <CHANGE path/to/file>
      SEARCH
      ...
      </SEARCH>
      <REPLACE>
      ...
      </REPLACE>
      </CHANGE>
    """
    out: list[tuple[str, str, str]] = []

    # Strategy 1: canonical <<<CHANGE ... CHANGE>>>, plus the single-bracket
    # variant Sonnet 4.5 emits (`<CHANGE path>` opener, same `CHANGE>>>` close).
    canonical = re.compile(
        r"(?:<<<CHANGE\s+(?P<p1>\S+?)|<CHANGE\s+(?P<p2>\S+?)>)\s*\n"
        r"(?P<body>[\s\S]*?)CHANGE>>>",
        re.MULTILINE,
    )
    consumed: list[tuple[int, int]] = []
    for match in canonical.finditer(response):
        fpath = _normalize_path_token(match.group("p1") or match.group("p2") or "")
        if not fpath:
            continue
        body = match.group("body")
        parts = re.split(r"^\s*SEARCH\s*$", body, maxsplit=1, flags=re.MULTILINE)
        if len(parts) < 2:
            # Fallback: model omitted SEARCH keyword (gpt5_mini does this in
            # refinement). If REPLACE keyword is present, treat content before
            # REPLACE as SEARCH.
            replace_split = re.split(r"^\s*REPLACE\s*$", body, maxsplit=1, flags=re.MULTILINE)
            if len(replace_split) < 2:
                continue
            search_text = _strip_fence(replace_split[0]).strip("\n")
            replace_text = _strip_fence(replace_split[1]).strip("\n")
            if search_text:
                out.append((fpath, search_text, replace_text))
                consumed.append(match.span())
            continue
        rest = parts[1]
        parts2 = re.split(r"^\s*REPLACE\s*$", rest, maxsplit=1, flags=re.MULTILINE)
        if len(parts2) < 2:
            continue
        out.append((
            fpath,
            _strip_fence(parts2[0]).strip("\n"),
            _strip_fence(parts2[1]).strip("\n"),
        ))
        consumed.append(match.span())

    # Strategy 3: XML-style with explicit closing tags, no `CHANGE>>>`.
    # Observed in Sonnet 4.5 outputs (~20% of its failed extractions):
    #   <CHANGE path/to/file>
    #   SEARCH
    #   ...
    #   </SEARCH>
    #   <REPLACE>
    #   ...
    #   </REPLACE>
    #   </CHANGE>
    xml_style = re.compile(
        r"<CHANGE\s+([^>\n]+?)>\s*\n"
        r"\s*(?:<SEARCH>|SEARCH)\s*\n"
        r"(?P<search>[\s\S]*?)\n?</SEARCH>\s*\n"
        r"\s*(?:<REPLACE>|REPLACE)\s*\n"
        r"(?P<replace>[\s\S]*?)\n?</REPLACE>\s*\n"
        r"\s*</CHANGE>",
        re.MULTILINE,
    )

    def _overlaps_consumed(start: int, end: int) -> bool:
        return any(start < e and s < end for s, e in consumed)

    for match in xml_style.finditer(response):
        if _overlaps_consumed(match.start(), match.end()):
            continue
        fpath = _normalize_path_token(match.group(1))
        if not fpath:
            continue
        out.append((
            fpath,
            _strip_fence(match.group("search")).strip("\n"),
            _strip_fence(match.group("replace")).strip("\n"),
        ))
        consumed.append(match.span())

    # Strategy 2: loose `CHANGE path ... SEARCH ... REPLACE ...` blocks.
    # We split the response on lines that start a new `CHANGE <path>` and
    # parse each chunk for a SEARCH/REPLACE pair. Skip ranges already
    # consumed by the canonical parser. The line may be inside a ``` fence
    # (Qwen2.5-7B does this), so we accept an optional leading ``` prefix.

    # Accept opener forms: bare `CHANGE path`, `<<<CHANGE path`, and
    # `<CHANGE path>` (Sonnet 4.5). The `>?` allows the closing bracket of
    # the single-bracket form; the path is captured lazily so that bracket
    # is not consumed.
    chunks = list(
        re.finditer(
            r"^\s*(?:```[a-zA-Z0-9_+-]*\s*)?(?:<<<\s*|<\s*)?CHANGE\s+(\S+?)>?\s*$",
            response, re.MULTILINE,
        )
    )
    for i, m in enumerate(chunks):
        if _overlaps_consumed(m.start(), m.end()):
            continue
        fpath = _normalize_path_token(m.group(1))
        if not fpath:
            continue
        body_start = m.end()
        body_end = chunks[i + 1].start() if i + 1 < len(chunks) else len(response)
        body = response[body_start:body_end]
        # Trim trailing closing-fence ``` if present
        body = re.sub(r"\n?```\s*$", "", body, flags=re.MULTILINE)
        # XML-style search/replace closers can appear here too — strip them so
        # the SEARCH/REPLACE keyword splits below still work.
        body = re.sub(r"</SEARCH>\s*\n", "", body)
        body = re.sub(r"</REPLACE>\s*\n", "", body)
        body = re.sub(r"</CHANGE>\s*$", "", body, flags=re.MULTILINE)
        parts = re.split(r"^\s*<?SEARCH>?\s*$", body, maxsplit=1, flags=re.MULTILINE)
        if len(parts) < 2:
            continue
        parts2 = re.split(r"^\s*<?REPLACE>?\s*$", parts[1], maxsplit=1, flags=re.MULTILINE)
        if len(parts2) < 2:
            continue
        out.append((
            fpath,
            _strip_fence(parts2[0]).strip("\n"),
            _strip_fence(parts2[1]).strip("\n"),
        ))

    # Strategy 4: pathless XML pairs that the model emits without any
    # `<CHANGE path>` opener. Sonnet 4.5 does this — drops `<SEARCH>...</SEARCH>
    # <REPLACE>...</REPLACE>` directly into the prose, expecting the reader
    # to know which file from context. We emit these with `fpath=""` so
    # ``apply_change_blocks`` can infer the path by searching oracle files
    # for the SEARCH text.
    pathless_xml = re.compile(
        r"<SEARCH>\s*\n(?P<search>[\s\S]*?)\n?</SEARCH>\s*\n"
        r"\s*<REPLACE>\s*\n(?P<replace>[\s\S]*?)\n?</REPLACE>",
        re.MULTILINE,
    )
    for match in pathless_xml.finditer(response):
        if _overlaps_consumed(match.start(), match.end()):
            continue
        out.append((
            "",  # sentinel: apply_change_blocks will infer the path
            _strip_fence(match.group("search")).strip("\n"),
            _strip_fence(match.group("replace")).strip("\n"),
        ))
        consumed.append(match.span())

    return out


def make_diff(original: str, modified: str, file_path: str) -> str:
    """Compute unified diff; ensure trailing newline so `patch` is happy."""
    import difflib

    if original and not original.endswith("\n"):
        original += "\n"
    if modified and not modified.endswith("\n"):
        modified += "\n"
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
    )
    text = "".join(diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text


def _common_indent(lines: list[str]) -> str:
    """Longest common leading-whitespace prefix across non-blank lines.

    Blank/whitespace-only lines are ignored (their leading whitespace is
    not a meaningful indent signal). Returns "" if no non-blank lines.
    """
    indents: list[str] = []
    for line in lines:
        if line.strip():
            stripped_len = len(line) - len(line.lstrip())
            indents.append(line[:stripped_len])
    if not indents:
        return ""
    common = indents[0]
    for ind in indents[1:]:
        i = 0
        while i < len(common) and i < len(ind) and common[i] == ind[i]:
            i += 1
        common = common[:i]
        if not common:
            break
    return common


def _try_indent_tolerant_replace(
    original: str, search_text: str, replace_text: str
) -> str | None:
    """SEARCH/REPLACE with leading-whitespace (indentation) tolerance.

    Models often emit a SEARCH block at a shallower indent than the file
    actually has (e.g. 8 spaces vs 12). We dedent both SEARCH and each
    candidate line-window in the file by their respective common indents
    and compare line-by-line (with trailing whitespace stripped). On the
    first match we re-indent REPLACE to the window's indent before
    substituting, preserving the file's nesting level.

    Returns the modified file text on success, else None.
    """
    file_lines = original.split("\n")
    s_lines = search_text.split("\n")
    if s_lines and s_lines[-1] == "":
        s_lines = s_lines[:-1]
    if not s_lines:
        return None

    s_indent = _common_indent(s_lines)

    def _dedent(line: str, indent: str) -> str:
        return line[len(indent):] if indent and line.startswith(indent) else line

    s_norm = [_dedent(l, s_indent).rstrip() for l in s_lines]

    n = len(s_lines)
    for i in range(len(file_lines) - n + 1):
        window = file_lines[i : i + n]
        w_indent = _common_indent(window)
        w_norm = [_dedent(l, w_indent).rstrip() for l in window]
        if w_norm != s_norm:
            continue
        r_lines = replace_text.split("\n")
        r_trailing_nl = bool(r_lines) and r_lines[-1] == ""
        if r_trailing_nl:
            r_lines = r_lines[:-1]
        r_lines = [_dedent(l, s_indent) for l in r_lines]
        r_lines = [(w_indent + l) if l.strip() else l for l in r_lines]
        new_lines = file_lines[:i] + r_lines + file_lines[i + n:]
        return "\n".join(new_lines)
    return None


def _resolve_oracle_path(fpath: str, oracle_files: dict[str, str]) -> str:
    """Map a model-emitted path to an actual oracle key.

    Tiers, in order:
      1. Exact match.
      2. Either path is a suffix of the other (handles repo-prefix stripping).
      3. Basename uniquely identifies one oracle file.
    Returns the original fpath unchanged if no resolution succeeds (the caller
    will then bail out via ``if not original: continue``).
    """
    if fpath in oracle_files:
        return fpath
    # Tier 2: suffix match
    for opath in oracle_files:
        if opath.endswith(fpath) or fpath.endswith(opath):
            return opath
    # Tier 3: basename uniquely identifies a file
    basename = fpath.rsplit("/", 1)[-1]
    if basename and basename != fpath:
        candidates = [p for p in oracle_files if p.rsplit("/", 1)[-1] == basename]
        if len(candidates) == 1:
            return candidates[0]
    return fpath


def _try_fuzzy_replace(
    original: str,
    search_text: str,
    replace_text: str,
    threshold: float = 0.92,
) -> str | None:
    """SEARCH/REPLACE with fuzzy line-window match (last-resort tier).

    Slides a window of the SEARCH block's length across the file and scores
    each window against SEARCH using ``difflib.SequenceMatcher.ratio()``.
    Only accepts the best window if its similarity ratio is >= ``threshold``
    (default 0.92). This guards against accepting hallucinated SEARCH blocks
    (e.g. paraphrased docstrings) — those typically score well below 0.5.

    On success, replaces the matched window with REPLACE, re-indenting it to
    the window's leading indent (same convention as the indent-tolerant
    matcher above). Returns the modified file text on success, else None.
    """
    from difflib import SequenceMatcher

    file_lines = original.split("\n")
    s_lines = search_text.split("\n")
    if s_lines and s_lines[-1] == "":
        s_lines = s_lines[:-1]
    n = len(s_lines)
    if n == 0 or n > len(file_lines):
        return None

    s_norm_str = "\n".join(l.rstrip() for l in s_lines)
    best_score = 0.0
    best_i = -1
    # Bail out early: a quick heuristic — count overlap of the first SEARCH
    # line with file lines; if no file line is close enough, skip the
    # expensive O(n*m) sweep. This is a perf shortcut, not correctness.
    s_first = s_lines[0].rstrip().lstrip()
    has_anchor = bool(s_first) and any(
        s_first in l or l.lstrip().startswith(s_first[: max(8, len(s_first) // 2)])
        for l in file_lines
    )
    if not has_anchor:
        return None

    for i in range(len(file_lines) - n + 1):
        window = file_lines[i : i + n]
        w_norm_str = "\n".join(l.rstrip() for l in window)
        # Fast pre-check: if the lengths differ by more than 50%, skip.
        if len(w_norm_str) and abs(len(w_norm_str) - len(s_norm_str)) / max(
            len(w_norm_str), len(s_norm_str)
        ) > 0.5:
            continue
        ratio = SequenceMatcher(None, s_norm_str, w_norm_str).ratio()
        if ratio > best_score:
            best_score = ratio
            best_i = i
            if ratio >= 0.999:
                break  # near-exact, no need to keep searching

    if best_score < threshold or best_i < 0:
        return None

    # Replace the matched window with replace_text, re-indenting to the
    # window's common indent.
    s_indent = _common_indent(s_lines)
    window = file_lines[best_i : best_i + n]
    w_indent = _common_indent(window)
    r_lines = replace_text.split("\n")
    r_trailing_nl = bool(r_lines) and r_lines[-1] == ""
    if r_trailing_nl:
        r_lines = r_lines[:-1]
    r_lines = [
        (l[len(s_indent):] if s_indent and l.startswith(s_indent) else l)
        for l in r_lines
    ]
    r_lines = [(w_indent + l) if l.strip() else l for l in r_lines]
    new_lines = file_lines[:best_i] + r_lines + file_lines[best_i + n :]
    return "\n".join(new_lines)


def _infer_path_from_search(
    search_text: str,
    oracle_files: dict[str, str],
) -> str:
    """Find which oracle file uniquely contains the SEARCH text.

    Used when the parser couldn't extract a path (Strategy 4 pathless XML
    blocks) or when the model emitted a placeholder path with the real
    path elsewhere. Tries the full matcher chain (strict, rstrip, indent,
    fuzzy) so we don't miss matches due to whitespace drift.

    Returns the matched path if exactly one oracle file contains the
    SEARCH text, else "" (caller will skip the block).
    """
    if not search_text.strip():
        return ""
    matches: list[str] = []
    for opath, content in oracle_files.items():
        if not content:
            continue
        if search_text in content:
            matches.append(opath)
            continue
        stripped_orig = "\n".join(l.rstrip() for l in content.split("\n"))
        stripped_search = "\n".join(l.rstrip() for l in search_text.split("\n"))
        if stripped_search in stripped_orig:
            matches.append(opath)
            continue
        if _try_indent_tolerant_replace(content, search_text, "X") is not None:
            matches.append(opath)
            continue
        if _try_fuzzy_replace(content, search_text, "X") is not None:
            matches.append(opath)
    # Only accept unambiguous single-file matches. If multiple oracle
    # files contain the same SEARCH text we'd be guessing, so bail.
    return matches[0] if len(matches) == 1 else ""


def apply_change_blocks(
    oracle_files: dict[str, str],
    blocks: list[tuple[str, str, str]],
) -> dict[str, str]:
    """Apply SEARCH/REPLACE blocks. Returns modified file contents.

    Four matching tiers, in order:
      1. Strict literal substring.
      2. Trailing-whitespace tolerant (rstrip per line).
      3. Indentation tolerant (dedent SEARCH and the candidate window to
         their common indents, re-indent REPLACE to the window's indent).
      4. Fuzzy SequenceMatcher with similarity threshold 0.92 (last resort,
         guards against hallucinated SEARCH text).

    Also handles two path-resolution recoveries:
      - fpath == ""        (Strategy 4 pathless XML pair): scan oracle
        files for SEARCH match and use the unique hit.
      - placeholder path   (e.g. "FILE_PATH_HERE"): same scan, useful
        when the model emitted a placeholder path with the real path on
        the next line (qwen3_coder mistake).
    """
    modified: dict[str, str] = {}
    for fpath, search_text, replace_text in blocks:
        # Path inference: missing or placeholder paths get re-resolved by
        # finding which oracle file uniquely contains the SEARCH text.
        # Without this we'd silently drop these blocks (the most common
        # remaining failure for sonnet45 + qwen3_coder).
        if not fpath or _PLACEHOLDER_PATH_RE.match(fpath):
            inferred = _infer_path_from_search(search_text, oracle_files)
            if not inferred:
                continue
            fpath = inferred
        resolved = _resolve_oracle_path(fpath, oracle_files)
        original = modified.get(resolved, oracle_files.get(resolved, ""))
        if not original:
            continue
        if search_text in original:
            modified[resolved] = original.replace(search_text, replace_text, 1)
            continue
        stripped_orig = "\n".join(l.rstrip() for l in original.split("\n"))
        stripped_search = "\n".join(l.rstrip() for l in search_text.split("\n"))
        if stripped_search in stripped_orig:
            modified[resolved] = original.replace(
                search_text.rstrip(), replace_text.rstrip(), 1
            )
            continue
        indent_match = _try_indent_tolerant_replace(
            original, search_text, replace_text
        )
        if indent_match is not None:
            modified[resolved] = indent_match
            continue
        fuzzy_match = _try_fuzzy_replace(original, search_text, replace_text)
        if fuzzy_match is not None:
            modified[resolved] = fuzzy_match
    return modified


def build_diff(
    oracle_files: dict[str, str],
    modified_files: dict[str, str],
) -> str:
    """Build a unified diff covering all modified files."""
    diffs: list[str] = []
    for fpath, modified in modified_files.items():
        original = oracle_files.get(fpath, "")
        if original == modified:
            continue
        diff = make_diff(original, modified, fpath)
        if diff:
            diffs.append(diff)
    return "\n".join(diffs)


def fetch_oracle_files(repo: str, base_commit: str, file_paths: list[str]) -> dict[str, str]:
    """Read oracle files via raw.githubusercontent.com (no clone needed)."""
    import urllib.request
    import urllib.error

    out: dict[str, str] = {}
    for fpath in file_paths:
        url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{fpath}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            lines = text.split("\n")
            if len(lines) > MAX_FILE_LINES:
                text = "\n".join(lines[:MAX_FILE_LINES]) + (
                    f"\n... (truncated, {len(lines)} lines total)"
                )
            out[fpath] = text
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            log.warning("  oracle fetch failed for %s: %s", fpath, exc)
    return out


def make_client(base_url: str | None) -> OpenAI:
    """Build an OpenAI-compatible client.

    `base_url=None` -> OpenRouter (requires OPENROUTER_API_KEY).
    Anything else  -> a local vLLM endpoint, no real auth needed.
    """
    if base_url is None:
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
            "SAGE_OPENROUTER_API_KEY"
        )
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY not set in environment or .env")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return OpenAI(api_key="EMPTY", base_url=base_url)


DEFAULT_MAX_TOKENS = 4000        # focused diffs are well under this
# vLLM serves Qwen3-8B at max_model_len=32768. Big oracle prompts run
# 15-22K tokens, leaving ~10-17K for output. 10K covers a 4-6K thinking
# section + a 4K diff while leaving 22K of headroom for the prompt.
THINKING_MAX_TOKENS = 10000


def generate_one(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    seed: int,
    enable_thinking: bool | None = None,
    max_tokens: int | None = None,
    request_logprobs: bool = False,
    top_logprobs: int = 0,
) -> tuple[str, float, int, int, dict]:
    """Single chat completion call.

    Returns (response_text, cost_usd, prompt_tokens, completion_tokens).
    Cost is whatever the provider reported in `usage.cost` (OpenRouter
    populates this; local vLLM does not). Tokens come from `usage.*`.

    enable_thinking: forwarded to Qwen3-family models via vLLM's
        chat_template_kwargs. None lets the model use its default
        (thinking ON for Qwen3); explicit True/False forces it.
    """
    if max_tokens is None:
        max_tokens = THINKING_MAX_TOKENS if enable_thinking else DEFAULT_MAX_TOKENS

    extra: dict = {}
    if model.lower().startswith("qwen/qwen3") and enable_thinking is not None:
        extra["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "timeout": LLM_TIMEOUT_S,
        **extra,
    }
    if request_logprobs:
        kwargs["logprobs"] = True
        if top_logprobs > 0:
            kwargs["top_logprobs"] = top_logprobs
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    cost, prompt_tok, completion_tok = extract_usage(resp)
    generation_logprobs = extract_completion_logprobs(
        resp, requested=request_logprobs)
    return text, cost, prompt_tok, completion_tok, generation_logprobs


def strip_think_blocks(text: str) -> str:
    """Strip <think>...</think> blocks before patch parsing.

    Qwen3 thinking-mode responses look like:
        <think>
        ... reasoning ...
        </think>
        <<<CHANGE path
        ...
    The reasoning text frequently contains pseudo-CHANGE-like fragments
    that confuse the patch parsers. Removing the thinking section first
    lets the regular parser see only the final answer.
    """
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text)


def parse_raw_diff(response: str) -> str:
    """Fallback: extract a raw unified diff from the response.

    Models that don't follow the CHANGE-block instruction often emit either
    ```diff blocks or bare `--- a/... +++ b/...` headers. We try both.
    """
    fenced = re.search(r"```(?:diff|patch)?\s*\n([\s\S]*?)```", response, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
        if "+++ b/" in text or "--- a/" in text:
            if not text.endswith("\n"):
                text += "\n"
            return text
    # Bare diff: take from first '--- a/' to end-of-text
    bare = re.search(r"(--- a/[\s\S]+?)(?:\n```|\Z)", response)
    if bare:
        text = bare.group(1).rstrip()
        if not text.endswith("\n"):
            text += "\n"
        return text
    return ""


def parse_full_file_blocks(response: str) -> dict[str, str]:
    """Fallback: parse <<<FILE path ... FILE>>> blocks (full-file rewrites)."""
    blocks: dict[str, str] = {}
    pattern = re.compile(r"<<<FILE\s+(.+?)\s*\n([\s\S]*?)FILE>>>", re.MULTILINE)
    for match in pattern.finditer(response):
        fpath = match.group(1).strip()
        content = match.group(2)
        if content.endswith("\n"):
            content = content[:-1]
        blocks[fpath] = content
    return blocks


# ---------------------------------------------------------------------------
# Phase 1: Generation
# ---------------------------------------------------------------------------

def sample_instances(seed: int, n: int,
                     dataset_name: str = "princeton-nlp/SWE-bench_Lite",
                     language_filter: str | None = None) -> list[dict]:
    """Sample n instances from a SWE-bench dataset; n=0 means all.

    language_filter: when set (e.g., 'python'), keep only instances where
        repo_language matches. Required for SWE-bench Pro (multi-language).
    """
    ds = load_dataset(dataset_name, split="test")
    indices = list(range(len(ds)))
    if language_filter:
        wanted = language_filter.lower()
        indices = [i for i in indices
                   if (ds[i].get("repo_language") or "").lower() == wanted]
    rng = random.Random(seed)
    rng.shuffle(indices)
    chosen = indices[:n] if n and n > 0 else indices
    return [dict(ds[i]) for i in sorted(chosen)]


def make_prompt(instance: dict, oracle_files: dict[str, str]) -> str:
    file_contents = "\n\n".join(
        f"### {fpath}\n```python\n{content}\n```"
        for fpath, content in oracle_files.items()
    ) if oracle_files else "(no files available)"
    hints = instance.get("hints_text", "")
    hints_section = f"## Hints\n{hints}" if hints else ""
    return PROMPT_TEMPLATE.format(
        repo=instance["repo"],
        problem_statement=instance["problem_statement"],
        hints_section=hints_section,
        file_contents=file_contents,
    )


class FatalAuthError(SystemExit):
    """Raised on the first 401/403 to abort the entire run cleanly."""


def _extract_diff_from_response(
    response: str,
    oracle_files: dict[str, str],
) -> tuple[str, str, int]:
    """Run the three-tier extractor. Returns (diff, extraction_path, n_blocks)."""
    response_for_parse = strip_think_blocks(response)
    blocks = parse_change_blocks(response_for_parse)
    n_blocks = len(blocks)
    if blocks:
        modified = apply_change_blocks(oracle_files, blocks)
        diff = build_diff(oracle_files, modified)
        if diff:
            return diff, "change_blocks", n_blocks
    full_blocks = parse_full_file_blocks(response_for_parse)
    if full_blocks:
        diffs: list[str] = []
        for fpath, modified_content in full_blocks.items():
            resolved = fpath if fpath in oracle_files else next(
                (o for o in oracle_files if o.endswith(fpath) or fpath.endswith(o)),
                fpath,
            )
            original = oracle_files.get(resolved, "")
            if original:
                d = make_diff(original, modified_content, resolved)
                if d:
                    diffs.append(d)
        if diffs:
            return "\n".join(diffs), "full_file_blocks", n_blocks
    raw = parse_raw_diff(response_for_parse)
    if raw:
        return raw, "raw_diff", n_blocks
    return "", "none", n_blocks


def generate_for_generator(
    generator_key: str,
    generator_model: str,
    base_url: str | None,
    enable_thinking: bool | None,
    instances: list[dict],
    n_patches: int,
    temperature: float,
    base_seed: int,
    output_dir: Path,
    max_workers: int,
    cost_tracker: CostTracker | None = None,
) -> Path:
    """Generate all patches for one generator and write predictions JSONL.

    Returns path to predictions JSONL (one record per (instance, patch_id)).

    cost_tracker, if provided, will be updated after each paid call. Workers
    early-exit (returning a "skipped" record) once `cost_tracker.capped` is
    true, so we don't burn money beyond the configured cap.
    """
    out_dir = output_dir / generator_key
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    raw_records_path = out_dir / "generation_records.jsonl"

    # Resume: skip records already present
    completed: set[tuple[str, int]] = set()
    if predictions_path.exists():
        for line in predictions_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                m = re.match(r".+__p(\d+)$", rec["model_name_or_path"])
                if m:
                    completed.add((rec["instance_id"], int(m.group(1))))
            except (json.JSONDecodeError, KeyError):
                continue

    client = make_client(base_url)
    request_logprobs = should_request_vllm_logprobs(base_url)
    top_logprobs = top_logprobs_from_env()

    log.info("[%s] fetching oracle files for %d instances", generator_key, len(instances))
    oracle_per_instance: dict[str, dict[str, str]] = {}
    for inst in instances:
        gold_files = get_changed_files_from_patch(inst.get("patch", ""))
        oracle_per_instance[inst["instance_id"]] = fetch_oracle_files(
            inst["repo"], inst["base_commit"], gold_files
        )

    tasks: list[tuple[dict, int, int]] = []
    for inst in instances:
        for pid in range(n_patches):
            if (inst["instance_id"], pid) in completed:
                continue
            tasks.append((inst, pid, base_seed + pid * 10_000))

    log.info(
        "[%s] %d (instance, patch_id) pairs to generate (%d already done)",
        generator_key, len(tasks), len(completed),
    )
    if cost_tracker is not None:
        log.info(
            "[%s] cost cap = $%.2f (remaining $%.2f)",
            generator_key, cost_tracker.cap_usd, cost_tracker.remaining,
        )

    import threading
    write_lock = threading.Lock()

    raw_responses_dir = out_dir / "raw_responses"
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    def _do_one(task: tuple[dict, int, int]) -> GenerationRecord:
        inst, pid, seed = task

        # Cost-cap pre-check: skip cleanly if already over budget.
        if cost_tracker is not None and not cost_tracker.can_proceed():
            cost_tracker.note_skipped(1)
            return GenerationRecord(
                generator_key=generator_key,
                generator_model=generator_model,
                instance_id=inst["instance_id"],
                repo=inst["repo"],
                patch_id=pid,
                diff="",
                response_chars=0,
                error=f"skipped: cost cap ${cost_tracker.cap_usd:.2f} reached",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        oracle_files = oracle_per_instance[inst["instance_id"]]
        prompt = make_prompt(inst, oracle_files)
        try:
            response, cost_usd, p_tok, c_tok, generation_logprobs = generate_one(
                client, generator_model, prompt, temperature, seed,
                enable_thinking=enable_thinking,
                request_logprobs=request_logprobs,
                top_logprobs=top_logprobs,
            )
        except FATAL_API_EXCEPTIONS as exc:
            # Mark tracker as capped so siblings stop, but raise so the
            # outer loop also notices and bails out of the whole generator.
            if cost_tracker is not None:
                cost_tracker.note_skipped(1)
            log.error("[%s] FATAL auth error: %s — aborting generator", generator_key, exc)
            raise FatalAuthError(2) from exc
        except RECOVERABLE_API_EXCEPTIONS as exc:
            return GenerationRecord(
                generator_key=generator_key,
                generator_model=generator_model,
                instance_id=inst["instance_id"],
                repo=inst["repo"],
                patch_id=pid,
                diff="",
                response_chars=0,
                error=f"api_error: {type(exc).__name__}: {exc}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:
            return GenerationRecord(
                generator_key=generator_key,
                generator_model=generator_model,
                instance_id=inst["instance_id"],
                repo=inst["repo"],
                patch_id=pid,
                diff="",
                response_chars=0,
                error=f"exception: {type(exc).__name__}: {exc}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        diff, extraction_path, n_blocks = _extract_diff_from_response(
            response, oracle_files
        )

        # Record cost AFTER we know the outcome so the audit log has full context
        if cost_tracker is not None and (cost_usd > 0 or request_logprobs):
            extra = {
                "extraction_path": extraction_path,
                "diff_chars": len(diff),
            }
            if request_logprobs:
                extra.update(logprob_summary_fields(generation_logprobs))
            cost_tracker.record(
                cost_usd=cost_usd,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                instance_id=inst["instance_id"],
                patch_id=pid,
                extra=extra,
            )

        err = "" if diff else (
            f"empty diff (response chars={len(response)}, change_blocks={n_blocks})"
        )

        if not diff:
            rp = raw_responses_dir / f"{inst['instance_id']}_p{pid}.txt"
            rp.write_text(response)
        if request_logprobs:
            write_logprob_sidecar(raw_responses_dir, f"{inst['instance_id']}_p{pid}",
                                  generation_logprobs,
                                  model=generator_model, prompt=prompt)

        return GenerationRecord(
            generator_key=generator_key,
            generator_model=generator_model,
            instance_id=inst["instance_id"],
            repo=inst["repo"],
            patch_id=pid,
            diff=diff,
            response_chars=len(response),
            error=err if not diff else f"ok (via {extraction_path})",
            timestamp=datetime.now(timezone.utc).isoformat(),
            cost_usd=cost_usd,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            extraction_path=extraction_path,
        )

    aborted_due_to_auth = False
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_do_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except FatalAuthError:
                aborted_due_to_auth = True
                # Cancel anything not yet started; in-flight calls will finish.
                for f in futures:
                    f.cancel()
                break
            cost_str = f" ${rec.cost_usd:.4f}" if rec.cost_usd else ""
            log.info(
                "[%s] %s p%d -> diff=%d chars%s (%s)",
                generator_key, rec.instance_id, rec.patch_id, len(rec.diff),
                cost_str, rec.error or "ok",
            )
            with write_lock:
                with open(raw_records_path, "a") as f:
                    f.write(json.dumps(dataclasses.asdict(rec)) + "\n")
                pred = {
                    "instance_id": rec.instance_id,
                    "model_name_or_path": f"{generator_key}__p{rec.patch_id}",
                    "model_patch": rec.diff,
                }
                with open(predictions_path, "a") as f:
                    f.write(json.dumps(pred) + "\n")
    if aborted_due_to_auth:
        raise FatalAuthError(2)

    if cost_tracker is not None:
        snap = cost_tracker.snapshot()
        log.info(
            "[%s] DONE: $%.4f / $%.2f cap (%d calls, %d skipped)",
            generator_key, snap["total_usd"], snap["cap_usd"],
            snap["n_calls"], snap["n_skipped"],
        )

    return predictions_path


# ---------------------------------------------------------------------------
# Phase 2: Evaluation via SWE-bench Docker harness
# ---------------------------------------------------------------------------

PODMAN_COMPAT_SHIM = '''
"""Podman/Docker-SDK compatibility for SWE-bench evaluation.

SWE-bench's CLI does not expose the TestSpec architecture knob. Clariden jobs
can run on aarch64 nodes, where x86_64 images fail without binfmt/qemu, so this
shim can force native arm64 TestSpecs and normalizes Docker platform strings for
podman's Docker API.
"""
import os
import re
import sys

if "podman" in os.environ.get("DOCKER_HOST", "").lower() or os.environ.get("SWEBENCH_PODMAN_COMPAT"):
    import docker
    from docker.api import build as _build_api
    from docker.api import container as _container_api
    from docker.models import containers as _containers_model
    from docker.models import images as _images_model
    from swebench.harness import dockerfiles as _dockerfiles
    from swebench.harness.dockerfiles import python as _python_dockerfiles
    from swebench.harness.test_spec import test_spec as _test_spec

    _FORCED_ARCH = os.environ.get("SWEBENCH_ARCH")
    if _FORCED_ARCH == "aarch64":
        _FORCED_ARCH = "arm64"
    _DEFAULT_PLATFORM = os.environ.get(
        "SWEBENCH_DOCKER_PLATFORM",
        "linux/arm64/v8" if _FORCED_ARCH == "arm64" else "linux/amd64",
    )
    _UBUNTU_IMAGE = os.environ.get("SWEBENCH_UBUNTU_IMAGE")
    if _UBUNTU_IMAGE:
        _python_dockerfiles._DOCKERFILE_BASE_PY = (
            _python_dockerfiles._DOCKERFILE_BASE_PY
            .replace("ubuntu:{ubuntu_version}", _UBUNTU_IMAGE)
        )
        _dockerfiles._DOCKERFILE_BASE["py"] = _python_dockerfiles._DOCKERFILE_BASE_PY

    def _patch_arm64_test_spec(spec):
        if getattr(spec, "arch", None) != "arm64":
            return spec

        if spec.repo in {"scikit-learn/scikit-learn", "pydata/xarray"}:
            apt = (
                "apt-get update && apt-get install -y "
                "gfortran libopenblas-dev liblapack-dev pkg-config "
                "&& rm -rf /var/lib/apt/lists/*"
            )
            if apt not in spec.env_script_list:
                spec.env_script_list.insert(1, apt)

        if spec.repo == "pydata/xarray":
            drop_cdms2 = "sed -i '/^[[:space:]]*- cdms2$/d' environment.yml"
            if drop_cdms2 not in spec.env_script_list:
                for i, cmd in enumerate(spec.env_script_list):
                    if cmd == "conda env update -f environment.yml":
                        spec.env_script_list.insert(i, drop_cdms2)
                        break

        if spec.repo == "scikit-learn/scikit-learn":
            pip_pin = "python -m pip install 'pip<23.1'"
            if pip_pin not in spec.env_script_list:
                spec.env_script_list.append(pip_pin)

        if spec.repo == "sympy/sympy":
            reset_commit = None
            for cmd in spec.repo_script_list:
                match = re.match(r"git reset --hard ([0-9a-f]+)$", cmd)
                if match:
                    reset_commit = match.group(1)
                    break
            for i, cmd in enumerate(spec.repo_script_list):
                if cmd.startswith("git clone -o origin --branch ") and "https://github.com/sympy/sympy /testbed" in cmd:
                    fallback = "git clone -o origin https://github.com/sympy/sympy /testbed"
                    if reset_commit:
                        fallback = (
                            "git init /testbed && cd /testbed && "
                            "git remote add origin https://github.com/sympy/sympy && "
                            f"git fetch --depth 1 origin {reset_commit} && "
                            "git checkout FETCH_HEAD"
                        )
                    spec.repo_script_list[i] = (
                        f"{cmd} || "
                        f"(rm -rf /testbed && {fallback})"
                    )
                    break

        for i, cmd in enumerate(spec.env_script_list):
            if "conda create -n testbed python=3.5" in cmd:
                spec.env_script_list[i] = cmd.replace("python=3.5", "python=3.6")

        if any("lxml" in cmd or "pikepdf" in cmd for cmd in spec.env_script_list):
            apt = (
                "apt-get update && apt-get install -y "
                "libxml2-dev libxslt1-dev zlib1g-dev "
                "&& rm -rf /var/lib/apt/lists/*"
            )
            if apt not in spec.env_script_list:
                spec.env_script_list.insert(1, apt)

        if spec.repo == "astropy/astropy":
            for i, cmd in enumerate(spec.env_script_list):
                if "conda create -n testbed python=3.6 setuptools==38.2.4 -y" in cmd:
                    spec.env_script_list[i] = cmd.replace(" setuptools==38.2.4", "")
                    setuptools_pin = "python -m pip install 'setuptools==38.2.4'"
                    if setuptools_pin not in spec.env_script_list:
                        spec.env_script_list.insert(i + 2, setuptools_pin)
                    break
            for i, cmd in enumerate(spec.env_script_list):
                if cmd.startswith("python -m pip install ") and "--no-clean" not in cmd:
                    spec.env_script_list[i] = cmd.replace(
                        "python -m pip install ",
                        "python -m pip install --no-clean ",
                        1,
                    )
            for i, cmd in enumerate(spec.repo_script_list):
                if cmd == "python -m pip install -e .[test] --verbose":
                    spec.repo_script_list[i:i + 1] = [
                        "python -m pip install jinja2",
                        "python -m pip install 'Cython<3'",
                        "python -m pip install extension-helpers",
                        "python -m pip install --no-build-isolation -e .[test] --verbose",
                    ]
                    break

        if spec.repo == "matplotlib/matplotlib":
            for i, cmd in enumerate(spec.repo_script_list):
                if cmd.startswith("tar -xvzf ") and "--no-same-owner" not in cmd:
                    spec.repo_script_list[i] = cmd.replace(
                        "tar -xvzf ",
                        "tar --no-same-owner -xvzf ",
                        1,
                    )

        return spec

    def _normalize_platform(platform):
        platform = platform or _DEFAULT_PLATFORM
        if platform in {"linux/x86_64", "linux/x64"}:
            return _DEFAULT_PLATFORM if _FORCED_ARCH == "arm64" else "linux/amd64"
        if platform in {"linux/aarch64", "linux/arm64"}:
            return "linux/arm64/v8"
        return platform

    if _FORCED_ARCH:
        _orig_make_test_spec = _test_spec.make_test_spec
        def _patched_make_test_spec(
            instance,
            namespace=None,
            base_image_tag="latest",
            env_image_tag="latest",
            instance_image_tag="latest",
            arch="x86_64",
        ):
            return _patch_arm64_test_spec(
                _orig_make_test_spec(
                    instance,
                    namespace=namespace,
                    base_image_tag=base_image_tag,
                    env_image_tag=env_image_tag,
                    instance_image_tag=instance_image_tag,
                    arch=_FORCED_ARCH,
                )
            )
        _test_spec.make_test_spec = _patched_make_test_spec  # type: ignore[assignment]
        for _module_name in (
            "swebench.harness.docker_build",
            "swebench.harness.prepare_images",
            "swebench.harness.reporting",
            "swebench.harness.run_evaluation",
        ):
            _module = sys.modules.get(_module_name)
            if _module is not None and hasattr(_module, "make_test_spec"):
                setattr(_module, "make_test_spec", _patched_make_test_spec)

    def _is_platform_error(exc):
        msg = str(exc).lower()
        return "platform" in msg and (
            "unexpected" in msg
            or "unsupported" in msg
            or "invalid" in msg
            or "client is newer than server" in msg
        )

    _orig_pull = _images_model.ImageCollection.pull
    def _patched_pull(self, repository, tag=None, all_tags=False, **kwargs):
        kwargs["platform"] = _normalize_platform(kwargs.get("platform"))
        try:
            return _orig_pull(self, repository, tag=tag, all_tags=all_tags, **kwargs)
        except Exception as exc:
            if not _is_platform_error(exc):
                raise
            kwargs.pop("platform", None)
            return _orig_pull(self, repository, tag=tag, all_tags=all_tags, **kwargs)
    _images_model.ImageCollection.pull = _patched_pull  # type: ignore[assignment]

    _orig_create = _container_api.ContainerApiMixin.create_container_from_config
    def _patched_create(self, config, name=None, platform=None):
        platform = _normalize_platform(platform or config.get("platform"))
        config.pop("platform", None)
        host_cfg = config.get("HostConfig")
        if isinstance(host_cfg, dict):
            host_cfg.pop("Platform", None)
        try:
            return _orig_create(self, config, name, platform)
        except Exception as exc:
            if not _is_platform_error(exc):
                raise
            config.pop("platform", None)
            if isinstance(host_cfg, dict):
                host_cfg.pop("Platform", None)
            return _orig_create(self, config, name, None)
    _container_api.ContainerApiMixin.create_container_from_config = _patched_create  # type: ignore[assignment]

    _orig_model_create = _containers_model.ContainerCollection.create
    def _patched_model_create(self, *args, **kwargs):
        name = str(kwargs.get("name") or "")
        disable_network = os.environ.get("SWEBENCH_DISABLE_EVAL_NETWORK", "1") == "1"
        if disable_network and name.startswith("sweb.eval."):
            kwargs.setdefault("network_mode", "none")
        return _orig_model_create(self, *args, **kwargs)
    _containers_model.ContainerCollection.create = _patched_model_create  # type: ignore[assignment]

    _orig_container_stop = _containers_model.Container.stop
    def _patched_container_stop(self, **kwargs):
        name = str(getattr(self, "name", "") or "")
        if name.startswith("sweb.eval."):
            timeout = int(os.environ.get("SWEBENCH_EVAL_STOP_TIMEOUT", "2"))
            kwargs["timeout"] = min(int(kwargs.get("timeout", timeout)), timeout)
        return _orig_container_stop(self, **kwargs)
    _containers_model.Container.stop = _patched_container_stop  # type: ignore[assignment]

    _orig_build = _build_api.BuildApiMixin.build
    def _patched_build(self, *args, **kwargs):
        kwargs["platform"] = _normalize_platform(kwargs.get("platform"))
        try:
            return _orig_build(self, *args, **kwargs)
        except Exception as exc:
            if not _is_platform_error(exc):
                raise
            kwargs.pop("platform", None)
            return _orig_build(self, *args, **kwargs)
    _build_api.BuildApiMixin.build = _patched_build  # type: ignore[assignment]
'''


def _write_podman_shim(target_dir: Path) -> Path:
    """Write the podman-compat shim and a sitecustomize that imports it."""
    target_dir.mkdir(parents=True, exist_ok=True)
    shim_path = target_dir / "podman_compat_shim.py"
    shim_path.write_text(PODMAN_COMPAT_SHIM)
    site = target_dir / "sitecustomize.py"
    site.write_text("import podman_compat_shim  # noqa: F401\n")
    return target_dir


def _fmt_mtime(ts: float | None) -> str:
    if ts is None:
        return "-"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _latest_direct_file_mtime(paths: Iterable[Path]) -> float | None:
    latest: float | None = None
    for path in paths:
        try:
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if latest is None or mtime > latest:
            latest = mtime
    return latest


def _count_named_files(instance_dirs: list[Path], name: str) -> int:
    return sum(1 for instance_dir in instance_dirs if (instance_dir / name).exists())


def _swe_eval_dir(work_dir: Path) -> Path:
    if (work_dir / "logs" / "run_evaluation").exists():
        return work_dir
    if (work_dir / "eval" / "logs" / "run_evaluation").exists():
        return work_dir / "eval"
    if work_dir.name == "eval":
        return work_dir
    return work_dir / "eval"


def _swe_build_progress(work_dir: Path) -> str:
    eval_dir = _swe_eval_dir(work_dir)
    build_root = eval_dir / "logs" / "build_images"
    env_logs = list((build_root / "env").glob("*/build_image.log"))
    instance_logs = list((build_root / "instances").glob("*/build_image.log"))
    latest = _latest_direct_file_mtime([*env_logs, *instance_logs])
    return (
        f"build_env_logs={len(env_logs)} "
        f"build_instance_logs={len(instance_logs)} "
        f"latest_build={_fmt_mtime(latest)}"
    )


def _swe_eval_progress_snapshot(work_dir: Path, run_id: str) -> str:
    """Summarize SWE-bench harness artifacts created for a run_id."""
    eval_dir = _swe_eval_dir(work_dir)
    run_root = eval_dir / "logs" / "run_evaluation" / run_id
    aggregate = "yes" if list(eval_dir.glob(f"*.{run_id}.json")) else "no"
    if not run_root.exists():
        return f"run_id={run_id} launched_dirs=0 aggregate={aggregate} {_swe_build_progress(work_dir)}"

    model_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not model_dirs:
        return f"run_id={run_id} launched_dirs=0 aggregate={aggregate} {_swe_build_progress(work_dir)}"

    chunks: list[str] = []
    for model_dir in model_dirs:
        instance_dirs = sorted(path for path in model_dir.iterdir() if path.is_dir())
        direct_files: list[Path] = []
        for instance_dir in instance_dirs:
            try:
                direct_files.extend(path for path in instance_dir.iterdir() if path.is_file())
            except OSError:
                continue
        latest = _latest_direct_file_mtime(direct_files)
        chunks.append(
            f"{model_dir.name}: "
            f"launched_dirs={len(instance_dirs)} "
            f"run_logs={_count_named_files(instance_dirs, 'run_instance.log')} "
            f"patch_diff={_count_named_files(instance_dirs, 'patch.diff')} "
            f"eval_sh={_count_named_files(instance_dirs, 'eval.sh')} "
            f"test_output={_count_named_files(instance_dirs, 'test_output.txt')} "
            f"reports={_count_named_files(instance_dirs, 'report.json')} "
            f"aggregate={aggregate} "
            f"latest_file={_fmt_mtime(latest)}"
        )
    return " | ".join([f"run_id={run_id}", _swe_build_progress(work_dir), *chunks])


def _docker_api_ping(env: dict[str, str]) -> bool:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import docker; c=docker.from_env(); c.ping()",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log.warning("Docker API ping failed:\n%s", proc.stderr[-1000:])
    return proc.returncode == 0


def _ensure_podman_service(env: dict[str, str]) -> None:
    """Make the rootless Podman Docker API socket available for SWE harness."""
    docker_host = env.get("DOCKER_HOST", "")
    if not ("podman" in docker_host.lower() or env.get("SWEBENCH_PODMAN_COMPAT")):
        return
    if _docker_api_ping(env):
        return

    podman_root = env.get("PODMAN_ROOT")
    podman_runroot = env.get("PODMAN_RUNROOT")
    podman_tmpdir = env.get("PODMAN_TMPDIR") or env.get("SWEBENCH_LOCAL_TMPDIR")
    podman_log = env.get("PODMAN_SERVICE_LOG")
    if not (podman_root and podman_runroot and podman_tmpdir and podman_log):
        raise RuntimeError("Podman Docker API is unavailable and restart env vars are missing")

    socket_path = docker_host.removeprefix("unix://")
    if not socket_path:
        raise RuntimeError(f"Unsupported DOCKER_HOST for podman restart: {docker_host}")
    Path(socket_path).parent.mkdir(parents=True, exist_ok=True)
    for path in (podman_root, podman_runroot, podman_tmpdir):
        Path(path).mkdir(parents=True, exist_ok=True)
    try:
        Path(socket_path).unlink()
    except FileNotFoundError:
        pass

    log.warning("Restarting Podman Docker API service at %s", docker_host)
    log_fh = open(podman_log, "a", buffering=1)
    subprocess.Popen(
        [
            "podman",
            "--root",
            podman_root,
            "--runroot",
            podman_runroot,
            "--tmpdir",
            podman_tmpdir,
            "--events-backend=file",
            "system",
            "service",
            "--time=0",
            docker_host,
        ],
        env={**env, "TMPDIR": podman_tmpdir},
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    for _ in range(30):
        if Path(socket_path).is_socket() and _docker_api_ping(env):
            return
        time.sleep(1)
    raise RuntimeError(f"Podman Docker API socket did not become ready: {docker_host}")


def run_swebench_eval(
    predictions_path: Path,
    run_id: str,
    max_workers: int,
    work_dir: Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
) -> Path:
    """Invoke `python -m swebench.harness.run_evaluation` and return path to report.

    The harness is launched with cwd=work_dir so its report files land there
    in a predictable location; consequently every path we hand it must be
    absolute, otherwise the harness resolves relative paths against work_dir
    instead of the script's cwd.
    """
    work_dir = work_dir.resolve()
    predictions_path = predictions_path.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build env for the harness subprocess. If DOCKER_HOST points at podman,
    # inject our compat shim via PYTHONPATH (sitecustomize auto-imports it).
    env = os.environ.copy()
    docker_host = env.get("DOCKER_HOST", "")
    if "podman" in docker_host.lower() or env.get("SWEBENCH_PODMAN_COMPAT"):
        shim_dir = _write_podman_shim(work_dir / ".podman_compat")
        env["PYTHONPATH"] = (
            str(shim_dir) + os.pathsep + env.get("PYTHONPATH", "")
        )
        env.setdefault("SWEBENCH_DOCKER_PLATFORM", "linux/amd64")
        local_tmpdir = env.get("SWEBENCH_LOCAL_TMPDIR") or env.get("PODMAN_TMPDIR")
        if local_tmpdir:
            Path(local_tmpdir).mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = local_tmpdir
        _ensure_podman_service(env)

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset_name,
        "--predictions_path", str(predictions_path),
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        # The harness default namespace ("swebench") treats instance images as
        # remote images and tries to pull per-instance tags from Docker Hub.
        # Those tags are not published for all sampled instances, so build them
        # locally under podman instead.
        "--namespace", "none",
        # cache_level=instance: keep instance images across pid runs; same
        # instance_id with patch_id 0/1/2 reuses one built image.
        "--cache_level", "instance",
    ]
    log.info("eval: %s", " ".join(cmd))
    try:
        progress_interval = float(env.get("SWE_PROGRESS_INTERVAL", "60"))
    except ValueError:
        progress_interval = 60.0
    stop_progress = threading.Event()
    progress_thread: threading.Thread | None = None
    if progress_interval > 0:
        log.info("swe progress initial: %s", _swe_eval_progress_snapshot(work_dir, run_id))

        def _log_progress() -> None:
            while not stop_progress.wait(progress_interval):
                log.info("swe progress: %s", _swe_eval_progress_snapshot(work_dir, run_id))

        progress_thread = threading.Thread(
            target=_log_progress,
            name=f"swe-progress-{run_id}",
            daemon=True,
        )
        progress_thread.start()
    try:
        proc = subprocess.run(
            cmd, cwd=work_dir, env=env, capture_output=True, text=True,
        )
    finally:
        stop_progress.set()
        if progress_thread is not None:
            progress_thread.join(timeout=2)
        if progress_interval > 0:
            log.info("swe progress final: %s", _swe_eval_progress_snapshot(work_dir, run_id))
    # Persist full stdout/stderr per run so we can root-cause harness/Docker
    # issues without re-running. The in-process log only keeps a 2K tail.
    eval_logs_dir = work_dir / "eval_logs"
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    (eval_logs_dir / f"{run_id}.stdout.log").write_text(proc.stdout)
    (eval_logs_dir / f"{run_id}.stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        log.warning("eval exit=%d", proc.returncode)
        log.warning("stderr tail:\n%s", proc.stderr[-2000:])
    log.info("eval stdout tail:\n%s", proc.stdout[-2000:])
    # Harness writes a report file named `<model>.<run_id>.json` in work_dir.
    candidates = sorted(work_dir.glob(f"*.{run_id}.json"))
    if not candidates:
        raise RuntimeError(
            f"No SWE-bench report file matching *.{run_id}.json in {work_dir}"
        )
    return candidates[-1]


def load_report(report_path: Path) -> dict:
    return json.loads(report_path.read_text())


def is_infra_failed_swebench_report(report_path: Path) -> bool:
    """True for reports where harness infra failed before any test completed."""
    try:
        report = load_report(report_path)
    except Exception:
        return True
    return (
        int(report.get("submitted_instances") or 0) > 0
        and int(report.get("completed_instances") or 0) == 0
        and int(report.get("error_instances") or 0) > 0
    )


def parse_resolved(report: dict) -> set[str]:
    """Extract the set of resolved (model_patch passed) instance_ids.

    SWE-bench harness schema_version=2 reports use:
      - `resolved_ids`: list[str] of instance IDs that passed
      - `resolved_instances`: int count (NOT a list)
    Older schemas put the IDs directly under `resolved_instances`.
    """
    for key in ("resolved_ids", "resolved_instances"):
        raw = report.get(key)
        if raw is None:
            continue
        if isinstance(raw, list):
            return set(raw)
        if isinstance(raw, dict):
            return set(raw.keys())
    return set()


# ---------------------------------------------------------------------------
# Phase 3: Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    generator_key: str,
    generator_model: str,
    instances: list[dict],
    predictions_path: Path,
    eval_report_path: Path,
    n_patches: int,
) -> GeneratorSummary:
    report = load_report(eval_report_path)
    resolved = parse_resolved(report)
    log.info("[%s] resolved set size=%d", generator_key, len(resolved))

    # Build per-(instance, patch_id) outcome
    by_repo_total: dict[str, int] = {}
    by_repo_pass: dict[str, int] = {}
    n_attempted = 0
    n_nonempty = 0
    n_correct = 0
    per_instance_pass_rate: list[float] = []

    instance_repo = {inst["instance_id"]: inst["repo"] for inst in instances}

    # Reload predictions to know how many were nonempty
    nonempty_keys: set[tuple[str, int]] = set()
    for line in predictions_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.match(r".+__p(\d+)$", rec["model_name_or_path"])
        if not m:
            continue
        pid = int(m.group(1))
        if rec.get("model_patch"):
            nonempty_keys.add((rec["instance_id"], pid))

    for inst in instances:
        inst_id = inst["instance_id"]
        repo = inst["repo"]
        passes = 0
        for pid in range(n_patches):
            n_attempted += 1
            by_repo_total[repo] = by_repo_total.get(repo, 0) + 1
            # Resolved keys in the report are the instance_id strings (harness
            # de-duplicates on (instance_id, model_name_or_path))
            # SWE-bench's report aggregates per instance_id under the model name.
            # Since we used distinct model names per patch_id, each report
            # corresponds to ONE patch_id. We loaded the per-pid report below.
            # So check resolved set for inst_id presence.
            ok = (inst_id in resolved) and ((inst_id, pid) in nonempty_keys)
            if (inst_id, pid) in nonempty_keys:
                n_nonempty += 1
            if ok:
                n_correct += 1
                passes += 1
                by_repo_pass[repo] = by_repo_pass.get(repo, 0) + 1
        per_instance_pass_rate.append(passes / n_patches if n_patches else 0.0)

    by_repo = {
        repo: {
            "total": by_repo_total[repo],
            "passed": by_repo_pass.get(repo, 0),
            "rate": by_repo_pass.get(repo, 0) / by_repo_total[repo],
        }
        for repo in sorted(by_repo_total)
    }

    return GeneratorSummary(
        generator_key=generator_key,
        generator_model=generator_model,
        n_instances=len(instances),
        n_patches_attempted=n_attempted,
        n_patches_nonempty=n_nonempty,
        n_patches_evaluated=n_attempted,  # all attempted go through harness
        n_correct=n_correct,
        base_rate=n_correct / n_attempted if n_attempted else 0.0,
        base_rate_per_instance=(
            sum(per_instance_pass_rate) / len(per_instance_pass_rate)
            if per_instance_pass_rate else 0.0
        ),
        by_repo=by_repo,
    )


def aggregate_per_pid(
    generator_key: str,
    generator_model: str,
    instances: list[dict],
    predictions_path: Path,
    work_dir: Path,
    n_patches: int,
) -> GeneratorSummary:
    """Look up one report per patch_id and aggregate.

    The harness was invoked once per patch_id with run_id `{generator_key}_p{pid}`.
    """
    by_repo_total: dict[str, int] = {}
    by_repo_pass: dict[str, int] = {}
    n_attempted = 0
    n_nonempty = 0
    n_correct = 0
    per_instance_pass_rate: list[float] = []

    # Map (instance_id, pid) -> nonempty
    nonempty_keys: set[tuple[str, int]] = set()
    for line in predictions_path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.match(r".+__p(\d+)$", rec["model_name_or_path"])
        if not m:
            continue
        pid = int(m.group(1))
        if rec.get("model_patch"):
            nonempty_keys.add((rec["instance_id"], pid))

    # Load resolved sets per pid
    resolved_per_pid: dict[int, set[str]] = {}
    for pid in range(n_patches):
        run_id = f"{generator_key}_p{pid}"
        candidates = sorted(work_dir.glob(f"*.{run_id}.json"))
        if not candidates:
            log.warning("[%s] no report for pid=%d", generator_key, pid)
            resolved_per_pid[pid] = set()
            continue
        report = load_report(candidates[-1])
        resolved_per_pid[pid] = parse_resolved(report)
        log.info(
            "[%s] pid=%d resolved=%d/%d",
            generator_key, pid, len(resolved_per_pid[pid]), len(instances),
        )

    for inst in instances:
        inst_id = inst["instance_id"]
        repo = inst["repo"]
        passes = 0
        for pid in range(n_patches):
            n_attempted += 1
            by_repo_total[repo] = by_repo_total.get(repo, 0) + 1
            if (inst_id, pid) in nonempty_keys:
                n_nonempty += 1
            ok = inst_id in resolved_per_pid.get(pid, set())
            if ok:
                n_correct += 1
                passes += 1
                by_repo_pass[repo] = by_repo_pass.get(repo, 0) + 1
        per_instance_pass_rate.append(passes / n_patches if n_patches else 0.0)

    by_repo = {
        repo: {
            "total": by_repo_total[repo],
            "passed": by_repo_pass.get(repo, 0),
            "rate": by_repo_pass.get(repo, 0) / by_repo_total[repo],
        }
        for repo in sorted(by_repo_total)
    }

    return GeneratorSummary(
        generator_key=generator_key,
        generator_model=generator_model,
        n_instances=len(instances),
        n_patches_attempted=n_attempted,
        n_patches_nonempty=n_nonempty,
        n_patches_evaluated=n_attempted,
        n_correct=n_correct,
        base_rate=n_correct / n_attempted if n_attempted else 0.0,
        base_rate_per_instance=(
            sum(per_instance_pass_rate) / len(per_instance_pass_rate)
            if per_instance_pass_rate else 0.0
        ),
        by_repo=by_repo,
    )


# ---------------------------------------------------------------------------
# Per-pid prediction file split (harness expects one row per (instance, model))
# ---------------------------------------------------------------------------

def split_predictions_by_pid(
    predictions_path: Path, n_patches: int
) -> dict[int, Path]:
    """Split a multi-pid predictions file into per-pid files.

    The harness identifies predictions by (instance_id, model_name_or_path).
    Since we encoded patch_id in the model name as `{key}__p{pid}`, splitting
    the JSONL into one file per pid lets us use distinct run_ids and report
    files (`<model>.{run_id}.json`).
    """
    out: dict[int, Path] = {}
    buckets: dict[int, list[str]] = {pid: [] for pid in range(n_patches)}
    for line in predictions_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.match(r".+__p(\d+)$", rec.get("model_name_or_path", ""))
        if not m:
            continue
        pid = int(m.group(1))
        if pid in buckets:
            buckets[pid].append(line)
    for pid, lines in buckets.items():
        if not lines:
            continue
        path = predictions_path.with_name(f"predictions_p{pid}.jsonl")
        path.write_text("\n".join(lines) + "\n")
        out[pid] = path
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_per_model_caps(spec: str | None, defaults: dict[str, float]) -> dict[str, float]:
    """Parse `--max-cost-usd-per-model` value.

    Accepted forms:
      - bare float ("5.0")            -> apply uniformly to every selected model
      - csv "key=val,key=val,..."     -> per-model override; missing keys fall
                                          back to DEFAULT_COST_CAPS_USD
      - empty/None                    -> use DEFAULT_COST_CAPS_USD as-is
    """
    caps = dict(defaults)
    if not spec:
        return caps
    spec = spec.strip()
    if "=" not in spec:
        try:
            uniform = float(spec)
        except ValueError as exc:
            raise SystemExit(f"--max-cost-usd-per-model: cannot parse {spec!r}") from exc
        for k in caps:
            caps[k] = uniform
        return caps
    for part in spec.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise SystemExit(f"--max-cost-usd-per-model: bad token {part!r}, want key=value")
        k, v = part.split("=", 1)
        try:
            caps[k.strip()] = float(v.strip())
        except ValueError as exc:
            raise SystemExit(f"--max-cost-usd-per-model: bad value for {k!r}: {v!r}") from exc
    return caps


def _make_tracker(key: str, base_url: str | None, cap_usd: float, output_dir: Path) -> CostTracker:
    """Build a per-model cost tracker (skipped for local vLLM since cost=0)."""
    log_path = output_dir / key / "cost_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return CostTracker(name=key, cap_usd=cap_usd, log_path=log_path)


def _run_probe(
    selected: dict[str, tuple[str, str | None, bool | None]],
    instances: list[dict],
    n_patches: int,
    temperature: float,
    base_seed: int,
    output_dir: Path,
    caps: dict[str, float],
) -> dict[str, dict[str, float | int | bool]]:
    """Generate exactly 1 patch on 1 instance per selected model.

    Returns a per-model dict with measured probe cost + projected total cost
    for `n_patches × len(instances)` calls. Uses the first instance from the
    sample (deterministic given seed).
    """
    if not instances:
        raise SystemExit("--probe-only: empty sample")
    probe_inst = instances[0]
    probe_inst_id = probe_inst["instance_id"]
    n_total = len(instances) * n_patches
    log.info(
        "PROBE: 1 generation on %s for each model; projecting to %d total calls",
        probe_inst_id, n_total,
    )

    out: dict[str, dict[str, float | int | bool]] = {}
    for key, (model, base_url, enable_thinking) in selected.items():
        cap = caps.get(key, 0.0)
        tracker = _make_tracker(key, base_url, cap_usd=cap, output_dir=output_dir / "_probe")
        log.info("=== probe: %s (%s) cap=$%.2f ===", key, model, cap)
        try:
            generate_for_generator(
                generator_key=key,
                generator_model=model,
                base_url=base_url,
                enable_thinking=enable_thinking,
                instances=[probe_inst],
                n_patches=1,
                temperature=temperature,
                base_seed=base_seed,
                output_dir=output_dir / "_probe",
                max_workers=1,
                cost_tracker=tracker,
            )
        except FatalAuthError:
            log.error("[%s] PROBE aborted on auth error; skipping projection", key)
            out[key] = {
                "probe_cost_usd": 0.0,
                "projected_total_usd": float("nan"),
                "cap_usd": cap,
                "exceeds_cap": True,
                "auth_error": True,
            }
            continue
        snap = tracker.snapshot()
        probe_cost = float(snap["total_usd"])
        projected = project_cost(probe_cost, n_total_calls=n_total, probe_calls=1)
        exceeds = projected > cap and cap < 1000.0  # local-vLLM caps are 1000
        out[key] = {
            "probe_cost_usd": probe_cost,
            "projected_total_usd": projected,
            "cap_usd": cap,
            "exceeds_cap": exceeds,
            "auth_error": False,
        }
    return out


def _print_probe_table(probe: dict[str, dict[str, float | int | bool]], n_total: int) -> None:
    print()
    print("=" * 78)
    print(f"PROBE PROJECTIONS (n_total_calls = {n_total})")
    print("=" * 78)
    print(f"{'model':<14} {'probe_cost':>12} {'projected':>12} {'cap':>10} {'verdict':>14}")
    grand_projected = 0.0
    for key, d in probe.items():
        projected = float(d["projected_total_usd"])
        cap = float(d["cap_usd"])
        verdict = "AUTH-FAIL" if d.get("auth_error") else (
            "EXCEEDS CAP" if d["exceeds_cap"] else "ok"
        )
        print(
            f"{key:<14} ${d['probe_cost_usd']:>10.4f} ${projected:>10.2f} ${cap:>8.2f} {verdict:>14}"
        )
        if not d.get("auth_error"):
            grand_projected += projected
    print("-" * 78)
    print(f"{'TOTAL':<14} {'':>12} ${grand_projected:>10.2f}")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-instances", type=int, default=DEFAULT_N_INSTANCES)
    parser.add_argument("--n-patches", type=int, default=DEFAULT_N_PATCHES)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite",
                        help="HuggingFace dataset name. Default: SWE-bench Lite. "
                             "Set to 'princeton-nlp/SWE-bench_Verified' for Verified, "
                             "or 'ScaleAI/SWE-bench_Pro' for Pro.")
    parser.add_argument("--language-filter", default=None,
                        help="Filter instances by repo_language (e.g., 'python'). "
                             "Required for SWE-bench Pro (multi-language). "
                             "If unset, no filtering.")
    parser.add_argument("--max-workers-gen", type=int, default=DEFAULT_MAX_WORKERS_GEN)
    parser.add_argument("--max-workers-eval", type=int, default=DEFAULT_MAX_WORKERS_EVAL)
    parser.add_argument("--generators", default=",".join(GENERATORS),
                        help="Comma-separated generator keys to run")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--probe-only", action="store_true",
        help="Run 1 generation per generator, project total cost, then exit. "
             "No harness eval, no aggregation.",
    )
    parser.add_argument(
        "--max-cost-usd-per-model", default=None,
        help="Hard cost cap per generator (USD). Either a single float "
             "(applies to all) or 'key=val,key=val,...' for per-model overrides. "
             "Defaults are per DEFAULT_COST_CAPS_USD.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_keys = [k.strip() for k in args.generators.split(",") if k.strip()]
    selected: dict[str, tuple[str, str | None, bool | None]] = {}
    for k in selected_keys:
        if k not in GENERATORS:
            raise SystemExit(f"unknown generator key: {k}")
        selected[k] = GENERATORS[k]

    caps = _parse_per_model_caps(args.max_cost_usd_per_model, DEFAULT_COST_CAPS_USD)

    log.info("output dir: %s", output_dir)
    log.info("generators: %s", {
        k: f"{m} via {u or 'OpenRouter'}" + (
            f" (thinking={t})" if t is not None else ""
        )
        for k, (m, u, t) in selected.items()
    })
    log.info("cost caps (USD): %s", {k: caps[k] for k in selected})
    log.info("n_instances=%d n_patches=%d seed=%d", args.n_instances, args.n_patches, args.seed)

    instances = sample_instances(args.seed, args.n_instances,
                                  dataset_name=args.dataset,
                                  language_filter=args.language_filter)
    log.info(
        "sampled %d instances; repos=%s",
        len(instances),
        sorted({i["repo"] for i in instances}),
    )
    (output_dir / "sample.json").write_text(
        json.dumps(
            [{"instance_id": i["instance_id"], "repo": i["repo"]} for i in instances],
            indent=2,
        )
    )

    # Persist a reproducibility snapshot so we can re-derive what was launched
    # without depending on shell history. Captures CLI args, generator config,
    # caps, and current git SHA.
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"
    run_config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "cli_args": vars(args),
        "generators": {
            k: {"model": m, "base_url": u, "enable_thinking": t}
            for k, (m, u, t) in selected.items()
        },
        "cost_caps_usd": {k: caps[k] for k in selected},
        "n_total_calls_planned": len(instances) * args.n_patches * len(selected),
        "prompt_template_sha256": __import__("hashlib").sha256(
            PROMPT_TEMPLATE.encode("utf-8")
        ).hexdigest(),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # ---------- Probe-only short-circuit ----------
    if args.probe_only:
        probe = _run_probe(
            selected=selected, instances=instances,
            n_patches=args.n_patches, temperature=args.temperature,
            base_seed=args.seed, output_dir=output_dir, caps=caps,
        )
        n_total = len(instances) * args.n_patches
        _print_probe_table(probe, n_total)
        (output_dir / "probe_report.json").write_text(
            json.dumps({"n_total_calls": n_total, "models": probe}, indent=2)
        )
        log.info("probe-only mode: exiting without bulk generation or eval")
        return

    # ---------- Phase 1 ----------
    if not args.skip_generate:
        for key, (model, base_url, enable_thinking) in selected.items():
            log.info(
                "=== generating: %s (%s via %s%s) cap=$%.2f ===",
                key, model, base_url or "OpenRouter",
                f", thinking={enable_thinking}" if enable_thinking is not None else "",
                caps.get(key, 0.0),
            )
            tracker = _make_tracker(
                key=key, base_url=base_url,
                cap_usd=caps.get(key, 0.0), output_dir=output_dir,
            )
            try:
                generate_for_generator(
                    generator_key=key,
                    generator_model=model,
                    base_url=base_url,
                    enable_thinking=enable_thinking,
                    instances=instances,
                    n_patches=args.n_patches,
                    temperature=args.temperature,
                    base_seed=args.seed,
                    output_dir=output_dir,
                    max_workers=args.max_workers_gen,
                    cost_tracker=tracker,
                )
            except FatalAuthError:
                log.error("[%s] generator aborted on auth error; halting all phase-1 work", key)
                # Save what we have for this generator and stop the run.
                snap = tracker.snapshot()
                (output_dir / key / "cost_summary.json").write_text(json.dumps(snap, indent=2))
                raise SystemExit(2)
            snap = tracker.snapshot()
            (output_dir / key / "cost_summary.json").write_text(json.dumps(snap, indent=2))

    # ---------- Phase 2 ----------
    if not args.skip_eval:
        eval_dir = output_dir / "eval"
        for key in selected:
            predictions_path = output_dir / key / "predictions.jsonl"
            if not predictions_path.exists():
                log.warning("[%s] no predictions file at %s; skipping eval", key, predictions_path)
                continue
            per_pid_files = split_predictions_by_pid(predictions_path, args.n_patches)
            for pid, path in per_pid_files.items():
                run_id = f"{key}_p{pid}"
                report_glob = list(eval_dir.glob(f"*.{run_id}.json"))
                if report_glob:
                    latest_report = sorted(report_glob)[-1]
                    if not is_infra_failed_swebench_report(latest_report):
                        log.info("[%s] pid=%d already evaluated, skipping", key, pid)
                        continue
                    log.warning(
                        "[%s] pid=%d existing report looks like harness infra "
                        "failure (%s); re-running eval",
                        key, pid, latest_report,
                    )
                log.info("=== eval: %s pid=%d ===", key, pid)
                run_swebench_eval(
                    predictions_path=path,
                    run_id=run_id,
                    max_workers=args.max_workers_eval,
                    work_dir=eval_dir,
                    dataset_name=args.dataset,
                )

    # ---------- Phase 3 ----------
    summaries: list[GeneratorSummary] = []
    eval_dir = output_dir / "eval"
    for key, (model, _base_url, _enable_thinking) in selected.items():
        predictions_path = output_dir / key / "predictions.jsonl"
        if not predictions_path.exists() or not eval_dir.exists():
            log.warning("[%s] missing data; skipping summary", key)
            continue
        summary = aggregate_per_pid(
            generator_key=key,
            generator_model=model,
            instances=instances,
            predictions_path=predictions_path,
            work_dir=eval_dir,
            n_patches=args.n_patches,
        )
        summaries.append(summary)

    summary_payload = {
        "n_instances": args.n_instances,
        "n_patches": args.n_patches,
        "temperature": args.temperature,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generators": [dataclasses.asdict(s) for s in summaries],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2))

    # Pretty print
    print()
    print("=" * 78)
    print("SPOT-CHECK SUMMARY")
    print("=" * 78)
    print(f"sample: n_instances={args.n_instances} n_patches={args.n_patches} seed={args.seed}")
    print(f"{'generator':<14} {'model':<32} {'attempted':>9} {'nonempty':>8} {'pass':>5} "
          f"{'rate':>6} {'inst-rate':>10}")
    for s in summaries:
        print(
            f"{s.generator_key:<14} {s.generator_model:<32} "
            f"{s.n_patches_attempted:>9} {s.n_patches_nonempty:>8} "
            f"{s.n_correct:>5} {s.base_rate:>6.2%} {s.base_rate_per_instance:>10.2%}"
        )
    print("=" * 78)
    print("regime check (PRE_REGISTRATION.md S4):")
    for s in summaries:
        in_window = 0.30 <= s.base_rate <= 0.70
        verdict = "IN  [0.30,0.70]" if in_window else "OUT [0.30,0.70]"
        print(f"  {s.generator_key:<14} base_rate={s.base_rate:.3f}  -> {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
