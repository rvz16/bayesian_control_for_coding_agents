"""Realistic bug definitions for comparing Simple Agent vs Bayesian POMDP Agent.

Uses a log parser module — a common real-world component — with bugs at
graduated difficulty levels designed to hit ~30-60% fix rate on 7B models.

Bug difficulty is tuned so the Bayesian agent's adaptive decisions matter:
- Easy bugs: both agents fix them, but Bayesian saves cost by verifying sooner
- Medium bugs: Bayesian uses cheap critics to decide strategy adaptively
- Hard bugs: Bayesian bails out early, saving cost on unfixable bugs
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Clean source: a log parser with timestamp handling, filtering, aggregation
# ---------------------------------------------------------------------------

LOG_PARSER_CLEAN = '''\
"""Log parser: parse, filter, and summarize log records."""

import re
from datetime import datetime

LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
LOG_RE = re.compile(
    r"\\[(\\d{4}-\\d{1,2}-\\d{1,2} \\d{1,2}:\\d{2}:\\d{2})\\]"
    r" (DEBUG|INFO|WARNING|ERROR|CRITICAL)"
    r" (\\S+):(\\d+) - (.*)"
)

def parse_line(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    m = LOG_RE.match(line)
    if not m:
        return None
    ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    return {"ts": ts, "level": m.group(2), "src": m.group(3),
            "lineno": int(m.group(4)), "msg": m.group(5).strip()}

def parse_log(text):
    return [r for line in text.splitlines() if (r := parse_line(line)) is not None]

def filter_by_level(records, min_level):
    threshold = LEVELS.get(min_level.upper(), 0)
    return [r for r in records if LEVELS.get(r["level"], 0) >= threshold]

def filter_by_time(records, start=None, end=None):
    out = []
    for r in records:
        if start and r["ts"] < start:
            continue
        if end and r["ts"] > end:
            continue
        out.append(r)
    return out

def filter_by_source(records, source, exact=False):
    if exact:
        return [r for r in records if r["src"] == source]
    return [r for r in records if source in r["src"]]

def count_by_level(records):
    counts = {}
    for r in records:
        counts[r["level"]] = counts.get(r["level"], 0) + 1
    return counts

def count_by_source(records):
    counts = {}
    for r in records:
        counts[r["src"]] = counts.get(r["src"], 0) + 1
    return counts

def error_windows(records, n_before=2, n_after=2):
    windows = []
    for i, r in enumerate(records):
        if r["level"] in ("ERROR", "CRITICAL"):
            s = max(0, i - n_before)
            e = min(len(records), i + n_after + 1)
            windows.append(records[s:e])
    return windows

def error_rate(records):
    if not records:
        return 0.0
    errs = sum(1 for r in records if r["level"] in ("ERROR", "CRITICAL"))
    return errs / len(records)

def latest_by_source(records):
    latest = {}
    for r in records:
        if r["src"] not in latest or r["ts"] > latest[r["src"]]["ts"]:
            latest[r["src"]] = r
    return latest

def top_sources(records, n=5):
    counts = count_by_source(records)
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

def severity_score(records):
    """Weighted severity score."""
    weights = {"DEBUG": 0, "INFO": 1, "WARNING": 3, "ERROR": 5, "CRITICAL": 10}
    return sum(weights.get(r["level"], 0) for r in records)

def group_by_minute(records):
    """Group records into per-minute buckets."""
    groups = {}
    for r in records:
        key = r["ts"].replace(second=0, microsecond=0)
        groups.setdefault(key, []).append(r)
    return groups

def deduplicate(records):
    """Remove records with duplicate messages, keep first occurrence."""
    seen = set()
    result = []
    for r in records:
        if r["msg"] not in seen:
            seen.add(r["msg"])
            result.append(r)
    return result

def search_messages(records, pattern):
    """Return records whose msg matches a regex pattern."""
    import re as _re
    pat = _re.compile(pattern, _re.IGNORECASE)
    return [r for r in records if pat.search(r["msg"])]

def time_range(records):
    """Return (earliest, latest) timestamps."""
    if len(records) == 0:
        return None, None
    timestamps = [r["ts"] for r in records]
    return min(timestamps), max(timestamps)

def has_critical(records):
    """Check whether any CRITICAL record exists."""
    return any(r["level"] == "CRITICAL" for r in records)

def filter_by_lineno(records, min_line=0, max_line=99999):
    """Filter records by source line number range (inclusive)."""
    return [r for r in records if r["lineno"] >= min_line and r["lineno"] <= max_line]

def merge_logs(records1, records2):
    """Merge two parsed log lists, sorted by timestamp."""
    combined = records1 + records2
    return sorted(combined, key=lambda r: r["ts"])

def running_error_count(records):
    """Cumulative count of ERROR and CRITICAL records."""
    counts = []
    total = 0
    for r in records:
        if r["level"] in ("ERROR", "CRITICAL"):  # severity check
            total += 1
        counts.append(total)
    return counts

def format_record(record):
    """Format a parsed record back to the original log line format."""
    ts_str = record["ts"].strftime("%Y-%m-%d %H:%M:%S")
    return f"[{ts_str}] {record[\'level\']} {record[\'src\']}:{record[\'lineno\']} - {record[\'msg\']}"

def unique_levels(records):
    """Return sorted list of unique log levels present."""
    return sorted({r["level"] for r in records})

def records_after(records, timestamp):
    """Return records with timestamp strictly after the given time."""
    return [r for r in records if r["ts"] > timestamp]

def source_line_map(records):
    """Map each source file to the set of line numbers seen."""
    smap = {}
    for r in records:
        smap.setdefault(r["src"], set()).add(r["lineno"])
    return smap
'''


# ---------------------------------------------------------------------------
# Buggy variants — graduated difficulty
# ---------------------------------------------------------------------------

BUGGY_SOURCES_REAL: dict[str, str] = {}

# Bug R1 (EASY): filter_by_level uses > instead of >=
BUGGY_SOURCES_REAL["bug_r1"] = LOG_PARSER_CLEAN.replace(
    'LEVELS.get(r["level"], 0) >= threshold',
    'LEVELS.get(r["level"], 0) > threshold',
)

# Bug R2 (EASY): filter_by_source exact match uses 'in' instead of '=='
BUGGY_SOURCES_REAL["bug_r2"] = LOG_PARSER_CLEAN.replace(
    '        return [r for r in records if r["src"] == source]',
    '        return [r for r in records if source in r["src"]]',
)

# Bug R3 (MEDIUM): filter_by_time end boundary inverted
BUGGY_SOURCES_REAL["bug_r3"] = LOG_PARSER_CLEAN.replace(
    'if end and r["ts"] > end:',
    'if end and r["ts"] < end:',
)

# Bug R4 (MEDIUM): error_windows off-by-one (missing +1)
BUGGY_SOURCES_REAL["bug_r4"] = LOG_PARSER_CLEAN.replace(
    'e = min(len(records), i + n_after + 1)',
    'e = min(len(records), i + n_after)',
)

# Bug R5 (MEDIUM): error_rate counts only ERROR, not CRITICAL
BUGGY_SOURCES_REAL["bug_r5"] = LOG_PARSER_CLEAN.replace(
    'errs = sum(1 for r in records if r["level"] in ("ERROR", "CRITICAL"))',
    'errs = sum(1 for r in records if r["level"] == "ERROR")',
)

# Bug R6 (HARD): latest_by_source uses < instead of > (returns oldest)
BUGGY_SOURCES_REAL["bug_r6"] = LOG_PARSER_CLEAN.replace(
    'r["ts"] > latest[r["src"]]["ts"]',
    'r["ts"] < latest[r["src"]]["ts"]',
)

# Bug R7 (HARD): top_sources sorts ascending instead of descending
BUGGY_SOURCES_REAL["bug_r7"] = LOG_PARSER_CLEAN.replace(
    "return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]",
    "return sorted(counts.items(), key=lambda x: x[1])[:n]",
)

# Bug R8 (HARD): LEVELS dict missing CRITICAL entry
BUGGY_SOURCES_REAL["bug_r8"] = LOG_PARSER_CLEAN.replace(
    'LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}',
    'LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}',
)

# Bug R9 (EASY): filter_by_time start boundary inverted (< vs >)
BUGGY_SOURCES_REAL["bug_r9"] = LOG_PARSER_CLEAN.replace(
    'if start and r["ts"] < start:',
    'if start and r["ts"] > start:',
)

# Bug R10 (EASY): count_by_level off-by-one (default 1 instead of 0)
BUGGY_SOURCES_REAL["bug_r10"] = LOG_PARSER_CLEAN.replace(
    'counts[r["level"]] = counts.get(r["level"], 0) + 1',
    'counts[r["level"]] = counts.get(r["level"], 1) + 1',
)

# Bug R11 (EASY): parse_log splits on whitespace instead of newlines
BUGGY_SOURCES_REAL["bug_r11"] = LOG_PARSER_CLEAN.replace(
    'return [r for line in text.splitlines() if (r := parse_line(line)) is not None]',
    'return [r for line in text.split() if (r := parse_line(line)) is not None]',
)

# Bug R12 (EASY): LEVELS dict wrong WARNING value (2 -> 3, same as ERROR)
BUGGY_SOURCES_REAL["bug_r12"] = LOG_PARSER_CLEAN.replace(
    '"WARNING": 2, "ERROR": 3,',
    '"WARNING": 3, "ERROR": 3,',
)

# Bug R13 (MEDIUM): parse_line swaps level and src groups
BUGGY_SOURCES_REAL["bug_r13"] = LOG_PARSER_CLEAN.replace(
    '"level": m.group(2), "src": m.group(3),',
    '"level": m.group(3), "src": m.group(2),',
)

# Bug R14 (MEDIUM): error_windows start index uses n_after instead of n_before
BUGGY_SOURCES_REAL["bug_r14"] = LOG_PARSER_CLEAN.replace(
    's = max(0, i - n_before)',
    's = max(0, i - n_after)',
)

# Bug R15 (MEDIUM): error_rate divides by len-1 (off-by-one)
BUGGY_SOURCES_REAL["bug_r15"] = LOG_PARSER_CLEAN.replace(
    'return errs / len(records)',
    'return errs / (len(records) - 1)',
)

# Bug R16 (MEDIUM): count_by_source uses r["level"] instead of r["src"]
BUGGY_SOURCES_REAL["bug_r16"] = LOG_PARSER_CLEAN.replace(
    'counts[r["src"]] = counts.get(r["src"], 0) + 1',
    'counts[r["level"]] = counts.get(r["level"], 0) + 1',
)

# Bug R17 (HARD): latest_by_source uses level as dict key instead of src
BUGGY_SOURCES_REAL["bug_r17"] = LOG_PARSER_CLEAN.replace(
    'latest[r["src"]] = r',
    'latest[r["level"]] = r',
)

# Bug R18 (HARD): error_windows only catches ERROR, not CRITICAL
BUGGY_SOURCES_REAL["bug_r18"] = LOG_PARSER_CLEAN.replace(
    'if r["level"] in ("ERROR", "CRITICAL"):\n            s = max',
    'if r["level"] == "ERROR":\n            s = max',
)

# Bug R19 (HARD): error_rate inverted fraction (len/errs instead of errs/len)
BUGGY_SOURCES_REAL["bug_r19"] = LOG_PARSER_CLEAN.replace(
    'return errs / len(records)',
    'return len(records) / errs',
)

# Bug R20 (HARD): filter_by_level uses .lower() instead of .upper()
BUGGY_SOURCES_REAL["bug_r20"] = LOG_PARSER_CLEAN.replace(
    'threshold = LEVELS.get(min_level.upper(), 0)',
    'threshold = LEVELS.get(min_level.lower(), 0)',
)

# ---- NEW BUGS (r21-r50) — from expanded log parser functions ----

# Bug R21 (EASY): running_error_count starts at 1 instead of 0
BUGGY_SOURCES_REAL["bug_r21"] = LOG_PARSER_CLEAN.replace(
    'total = 0',
    'total = 1',
)

# Bug R22 (EASY): parse_line lineno not converted to int (stays string)
BUGGY_SOURCES_REAL["bug_r22"] = LOG_PARSER_CLEAN.replace(
    '"lineno": int(m.group(4))',
    '"lineno": m.group(4)',
)

# Bug R23 (EASY): severity_score wrong weight for WARNING (3 -> 1, same as INFO)
BUGGY_SOURCES_REAL["bug_r23"] = LOG_PARSER_CLEAN.replace(
    '"WARNING": 3, "ERROR": 5',
    '"WARNING": 1, "ERROR": 5',
)

# Bug R24 (EASY): has_critical checks for ERROR instead of CRITICAL
BUGGY_SOURCES_REAL["bug_r24"] = LOG_PARSER_CLEAN.replace(
    'any(r["level"] == "CRITICAL" for r in records)',
    'any(r["level"] == "ERROR" for r in records)',
)

# Bug R25 (EASY): format_record wrong date separator (- vs /)
BUGGY_SOURCES_REAL["bug_r25"] = LOG_PARSER_CLEAN.replace(
    'record["ts"].strftime("%Y-%m-%d %H:%M:%S")',
    'record["ts"].strftime("%Y/%m/%d %H:%M:%S")',
)

# Bug R26 (EASY): filter_by_lineno off-by-one on lower bound (>= vs >)
BUGGY_SOURCES_REAL["bug_r26"] = LOG_PARSER_CLEAN.replace(
    'r["lineno"] >= min_line',
    'r["lineno"] > min_line',
)

# Bug R27 (EASY): deduplicate checks level instead of msg for uniqueness
BUGGY_SOURCES_REAL["bug_r27"] = LOG_PARSER_CLEAN.replace(
    'if r["msg"] not in seen',
    'if r["level"] not in seen',
)

# Bug R28 (EASY): search_messages searches src field instead of msg
BUGGY_SOURCES_REAL["bug_r28"] = LOG_PARSER_CLEAN.replace(
    'return [r for r in records if pat.search(r["msg"])]',
    'return [r for r in records if pat.search(r["src"])]',
)

# Bug R29 (EASY): group_by_minute forgets to append record to group
BUGGY_SOURCES_REAL["bug_r29"] = LOG_PARSER_CLEAN.replace(
    'groups.setdefault(key, []).append(r)',
    'groups.setdefault(key, [])',
)

# Bug R30 (EASY): time_range returns single None instead of tuple
BUGGY_SOURCES_REAL["bug_r30"] = LOG_PARSER_CLEAN.replace(
    'return None, None',
    'return None',
)

# Bug R31 (MEDIUM): search_messages missing IGNORECASE flag
BUGGY_SOURCES_REAL["bug_r31"] = LOG_PARSER_CLEAN.replace(
    'pat = _re.compile(pattern, _re.IGNORECASE)',
    'pat = _re.compile(pattern)',
)

# Bug R32 (MEDIUM): time_range returns (latest, earliest) — swapped
BUGGY_SOURCES_REAL["bug_r32"] = LOG_PARSER_CLEAN.replace(
    'return min(timestamps), max(timestamps)',
    'return max(timestamps), min(timestamps)',
)

# Bug R33 (MEDIUM): running_error_count only counts ERROR, not CRITICAL
BUGGY_SOURCES_REAL["bug_r33"] = LOG_PARSER_CLEAN.replace(
    'if r["level"] in ("ERROR", "CRITICAL"):  # severity check',
    'if r["level"] == "ERROR":  # severity check',
)

# Bug R34 (MEDIUM): merge_logs duplicates first list instead of combining
BUGGY_SOURCES_REAL["bug_r34"] = LOG_PARSER_CLEAN.replace(
    'combined = records1 + records2',
    'combined = records1 + records1',
)

# Bug R35 (MEDIUM): merge_logs sorts by source file instead of timestamp
BUGGY_SOURCES_REAL["bug_r35"] = LOG_PARSER_CLEAN.replace(
    'return sorted(combined, key=lambda r: r["ts"])',
    'return sorted(combined, key=lambda r: r["src"])',
)

# Bug R36 (MEDIUM): severity_score CRITICAL weight same as ERROR (10 -> 5)
BUGGY_SOURCES_REAL["bug_r36"] = LOG_PARSER_CLEAN.replace(
    '"CRITICAL": 10}',
    '"CRITICAL": 5}',
)

# Bug R37 (MEDIUM): filter_by_lineno off-by-one on upper bound (<= vs <)
BUGGY_SOURCES_REAL["bug_r37"] = LOG_PARSER_CLEAN.replace(
    'r["lineno"] <= max_line',
    'r["lineno"] < max_line',
)

# Bug R38 (MEDIUM): running_error_count off-by-one in cumulative count
BUGGY_SOURCES_REAL["bug_r38"] = LOG_PARSER_CLEAN.replace(
    'counts.append(total)',
    'counts.append(total - 1)',
)

# Bug R39 (MEDIUM): format_record swaps level and src fields
BUGGY_SOURCES_REAL["bug_r39"] = LOG_PARSER_CLEAN.replace(
    "return f\"[{ts_str}] {record['level']} {record['src']}:{record['lineno']} - {record['msg']}\"",
    "return f\"[{ts_str}] {record['src']} {record['level']}:{record['lineno']} - {record['msg']}\"",
)

# Bug R40 (MEDIUM): filter_by_source non-exact uses == instead of in
BUGGY_SOURCES_REAL["bug_r40"] = LOG_PARSER_CLEAN.replace(
    'return [r for r in records if source in r["src"]]',
    'return [r for r in records if source == r["src"]]',
)

# Bug R41 (HARD): group_by_minute uses exact timestamp as key (no bucketing)
BUGGY_SOURCES_REAL["bug_r41"] = LOG_PARSER_CLEAN.replace(
    'key = r["ts"].replace(second=0, microsecond=0)',
    'key = r["ts"]',
)

# Bug R42 (HARD): deduplicate appends msg string instead of full record
BUGGY_SOURCES_REAL["bug_r42"] = LOG_PARSER_CLEAN.replace(
    'result.append(r)',
    'result.append(r["msg"])',
)

# Bug R43 (HARD): time_range extracts line numbers instead of timestamps
BUGGY_SOURCES_REAL["bug_r43"] = LOG_PARSER_CLEAN.replace(
    'timestamps = [r["ts"] for r in records]',
    'timestamps = [r["lineno"] for r in records]',
)

# Bug R44 (HARD): deduplicate adds src to seen set instead of msg
BUGGY_SOURCES_REAL["bug_r44"] = LOG_PARSER_CLEAN.replace(
    'seen.add(r["msg"])',
    'seen.add(r["src"])',
)

# Bug R45 (HARD): running_error_count increments by 2 instead of 1
BUGGY_SOURCES_REAL["bug_r45"] = LOG_PARSER_CLEAN.replace(
    'total += 1',
    'total += 2',
)

# Bug R46 (HARD): unique_levels returns source files instead of levels
BUGGY_SOURCES_REAL["bug_r46"] = LOG_PARSER_CLEAN.replace(
    'sorted({r["level"] for r in records})',
    'sorted({r["src"] for r in records})',
)

# Bug R47 (HARD): records_after returns records BEFORE instead of after
BUGGY_SOURCES_REAL["bug_r47"] = LOG_PARSER_CLEAN.replace(
    'return [r for r in records if r["ts"] > timestamp]',
    'return [r for r in records if r["ts"] < timestamp]',
)

# Bug R48 (HARD): source_line_map adds level instead of line number
BUGGY_SOURCES_REAL["bug_r48"] = LOG_PARSER_CLEAN.replace(
    'smap.setdefault(r["src"], set()).add(r["lineno"])',
    'smap.setdefault(r["src"], set()).add(r["level"])',
)

# Bug R49 (HARD): parse_line extracts wrong regex group for msg
BUGGY_SOURCES_REAL["bug_r49"] = LOG_PARSER_CLEAN.replace(
    '"msg": m.group(5).strip()',
    '"msg": m.group(4).strip()',
)

# Bug R50 (HARD): time_range treats single-element list as empty
BUGGY_SOURCES_REAL["bug_r50"] = LOG_PARSER_CLEAN.replace(
    'if len(records) == 0:',
    'if len(records) <= 1:',
)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

LOG_PARSER_TEST_CONTENT = '''"""Tests for the log parser module."""

import sys
sys.path.insert(0, "{src_dir}")

from datetime import datetime
from log_parser import (
    parse_line, parse_log, filter_by_level, filter_by_time,
    filter_by_source, count_by_level, count_by_source,
    error_windows, error_rate, latest_by_source, top_sources,
    severity_score, group_by_minute, deduplicate, search_messages,
    time_range, has_critical, filter_by_lineno, merge_logs,
    running_error_count, format_record, unique_levels,
    records_after, source_line_map,
)

SAMPLE_LOG = """\\
[2024-01-15 08:00:00] DEBUG server.py:10 - Starting server
[2024-01-15 08:00:01] INFO server.py:20 - Listening on 8080
[2024-01-15 08:00:05] INFO auth.py:15 - User admin logged in
[2024-01-15 08:01:00] WARNING db.py:45 - Pool running low
[2024-01-15 08:01:30] ERROR db.py:52 - Connection timeout
[2024-01-15 08:01:31] INFO db.py:60 - Retrying connection
[2024-01-15 08:02:00] ERROR server.py:88 - Handler failed
[2024-01-15 08:02:01] CRITICAL server.py:90 - Shutting down
[2024-01-15 08:05:00] INFO server.py:95 - Server restarted
[2024-01-15 09:00:00] DEBUG auth.py:10 - Session cleanup
"""


# ===== Parsing tests (critic group: parsing_check) =====

def test_parse_line_basic():
    r = parse_line("[2024-01-15 08:00:00] DEBUG server.py:10 - Starting")
    assert r is not None
    assert r["level"] == "DEBUG"
    assert r["src"] == "server.py"

def test_parse_log_count():
    assert len(parse_log(SAMPLE_LOG)) == 10

def test_parse_line_blank():
    assert parse_line("") is None
    assert parse_line("# comment") is None


# ===== Filtering tests (critic group: filtering_check) =====

def test_filter_by_level_warning():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "WARNING")
    assert len(filtered) == 4  # WARNING + 2 ERROR + CRITICAL

def test_filter_by_level_error():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "ERROR")
    assert len(filtered) == 3  # 2 ERROR + 1 CRITICAL

def test_filter_by_level_critical():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "CRITICAL")
    assert len(filtered) == 1

def test_filter_by_level_includes_threshold():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_level(recs, "ERROR")
    error_recs = [r for r in filtered if r["level"] == "ERROR"]
    assert len(error_recs) == 2

def test_filter_by_time_inclusive():
    recs = parse_log(SAMPLE_LOG)
    start = datetime(2024, 1, 15, 8, 1, 0)
    end = datetime(2024, 1, 15, 8, 2, 0)
    filtered = filter_by_time(recs, start=start, end=end)
    assert any(r["ts"] == start for r in filtered)
    assert any(r["ts"] == end for r in filtered)

def test_filter_by_time_end_boundary():
    recs = parse_log(SAMPLE_LOG)
    end = datetime(2024, 1, 15, 8, 1, 30)
    filtered = filter_by_time(recs, end=end)
    assert any(r["ts"] == end for r in filtered)

def test_filter_by_source_exact():
    recs = parse_log(SAMPLE_LOG)
    assert len(filter_by_source(recs, "server.py", exact=True)) == 5
    assert len(filter_by_source(recs, "server", exact=True)) == 0

def test_filter_by_source_substring():
    recs = parse_log(SAMPLE_LOG)
    assert len(filter_by_source(recs, "server")) == 5  # "server" in "server.py"


# ===== Aggregation tests (critic group: aggregation_check) =====

def test_count_by_level():
    recs = parse_log(SAMPLE_LOG)
    c = count_by_level(recs)
    assert c["DEBUG"] == 2
    assert c["INFO"] == 4
    assert c["ERROR"] == 2
    assert c["CRITICAL"] == 1

def test_error_rate_calculation():
    recs = parse_log(SAMPLE_LOG)
    assert abs(error_rate(recs) - 0.3) < 0.01  # 3 errors out of 10

def test_error_rate_empty():
    assert error_rate([]) == 0.0


# ===== Context & summary tests (critic group: context_check) =====

def test_error_windows_count():
    recs = parse_log(SAMPLE_LOG)
    wins = error_windows(recs, n_before=2, n_after=2)
    assert len(wins) == 3  # 2 ERROR + 1 CRITICAL

def test_error_windows_includes_after():
    recs = parse_log(SAMPLE_LOG)
    wins = error_windows(recs, n_before=0, n_after=2)
    assert wins[0][0]["level"] == "ERROR"
    assert len(wins[0]) == 3  # error + 2 after

def test_latest_by_source():
    recs = parse_log(SAMPLE_LOG)
    latest = latest_by_source(recs)
    assert latest["server.py"]["ts"] == datetime(2024, 1, 15, 8, 5, 0)
    assert latest["auth.py"]["ts"] == datetime(2024, 1, 15, 9, 0, 0)
    assert latest["db.py"]["ts"] == datetime(2024, 1, 15, 8, 1, 31)

def test_top_sources_order():
    recs = parse_log(SAMPLE_LOG)
    top = top_sources(recs)
    assert top[0][0] == "server.py"  # most common (5 records)
    assert top[0][1] == 5


# ===== Score & classification tests (critic group: score_check) =====

def test_severity_score():
    recs = parse_log(SAMPLE_LOG)
    # 2*DEBUG(0) + 4*INFO(1) + 1*WARNING(3) + 2*ERROR(5) + 1*CRITICAL(10) = 27
    assert severity_score(recs) == 27

def test_has_critical_true():
    recs = parse_log(SAMPLE_LOG)
    assert has_critical(recs) is True

def test_has_critical_false():
    recs = parse_log(SAMPLE_LOG)
    no_crit = [r for r in recs if r["level"] != "CRITICAL"]
    assert has_critical(no_crit) is False

def test_unique_levels():
    recs = parse_log(SAMPLE_LOG)
    levels = unique_levels(recs)
    assert levels == ["CRITICAL", "DEBUG", "ERROR", "INFO", "WARNING"]


# ===== Dedup & search tests (critic group: dedup_search_check) =====

def test_deduplicate():
    recs = parse_log(SAMPLE_LOG)
    dup = recs + [dict(recs[0])]  # add duplicate of first record
    deduped = deduplicate(dup)
    assert len(deduped) == len(recs)
    assert all(isinstance(r, dict) and "msg" in r for r in deduped)

def test_search_messages():
    recs = parse_log(SAMPLE_LOG)
    results = search_messages(recs, "timeout")
    assert len(results) == 1  # "Connection timeout"
    assert results[0]["src"] == "db.py"

def test_search_messages_case_insensitive():
    recs = parse_log(SAMPLE_LOG)
    results = search_messages(recs, "TIMEOUT")
    assert len(results) == 1  # case insensitive match


# ===== Time & merge tests (critic group: time_merge_check) =====

def test_time_range():
    recs = parse_log(SAMPLE_LOG)
    earliest, latest = time_range(recs)
    assert earliest == datetime(2024, 1, 15, 8, 0, 0)
    assert latest == datetime(2024, 1, 15, 9, 0, 0)

def test_time_range_empty():
    assert time_range([]) == (None, None)

def test_time_range_single():
    recs = parse_log(SAMPLE_LOG)[:1]
    earliest, latest = time_range(recs)
    assert earliest is not None
    assert earliest == latest

def test_merge_logs_sorted():
    recs = parse_log(SAMPLE_LOG)
    half = len(recs) // 2
    merged = merge_logs(recs[:half], recs[half:])
    assert len(merged) == len(recs)
    for i in range(len(merged) - 1):
        assert merged[i]["ts"] <= merged[i + 1]["ts"]
    assert merged[-1]["ts"] == recs[-1]["ts"]

def test_records_after():
    recs = parse_log(SAMPLE_LOG)
    cutoff = datetime(2024, 1, 15, 8, 2, 0)
    after = records_after(recs, cutoff)
    assert all(r["ts"] > cutoff for r in after)
    assert len(after) == 3  # 08:02:01, 08:05:00, 09:00:00


# ===== Line number & format tests (critic group: lineno_format_check) =====

def test_filter_by_lineno():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_lineno(recs, min_line=10, max_line=20)
    assert len(filtered) == 4  # server.py:10, server.py:20, auth.py:15, auth.py:10
    assert any(r["lineno"] == 20 for r in filtered)
    assert any(r["lineno"] == 10 for r in filtered)

def test_filter_by_lineno_inclusive():
    recs = parse_log(SAMPLE_LOG)
    filtered = filter_by_lineno(recs, min_line=10, max_line=10)
    assert all(r["lineno"] == 10 for r in filtered)
    assert len(filtered) == 2  # server.py:10 and auth.py:10

def test_format_record():
    recs = parse_log(SAMPLE_LOG)
    formatted = format_record(recs[0])
    assert formatted.startswith("[2024-01-15 08:00:00]")
    assert "] DEBUG server.py:" in formatted  # exact field order

def test_running_error_count():
    recs = parse_log(SAMPLE_LOG)
    counts = running_error_count(recs)
    assert len(counts) == len(recs)
    assert counts[-1] == 3  # 2 ERROR + 1 CRITICAL
    assert all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1))

def test_group_by_minute():
    recs = parse_log(SAMPLE_LOG)
    groups = group_by_minute(recs)
    assert len(groups) == 5  # 08:00, 08:01, 08:02, 08:05, 09:00
    total = sum(len(v) for v in groups.values())
    assert total == 10

def test_source_line_map():
    recs = parse_log(SAMPLE_LOG)
    smap = source_line_map(recs)
    assert "server.py" in smap
    assert 10 in smap["server.py"]
    assert 20 in smap["server.py"]
'''


# ---------------------------------------------------------------------------
# Critic test subsets (cheap diagnostic tests for the Bayesian agent)
# ---------------------------------------------------------------------------

REAL_CRITIC_TESTS = {
    "parsing_check": [
        "test_parse_line_basic",
        "test_parse_log_count",
        "test_parse_line_blank",
    ],
    "filtering_check": [
        "test_filter_by_level_warning",
        "test_filter_by_level_error",
        "test_filter_by_source_exact",
        "test_filter_by_source_substring",
        "test_filter_by_level_includes_threshold",
    ],
    "aggregation_check": [
        "test_count_by_level",
        "test_error_rate_calculation",
    ],
    "context_check": [
        "test_error_windows_count",
        "test_error_windows_includes_after",
        "test_latest_by_source",
        "test_top_sources_order",
    ],
    "score_check": [
        "test_severity_score",
        "test_has_critical_true",
        "test_has_critical_false",
        "test_unique_levels",
    ],
    "dedup_search_check": [
        "test_deduplicate",
        "test_search_messages",
        "test_search_messages_case_insensitive",
    ],
    "time_merge_check": [
        "test_time_range",
        "test_time_range_single",
        "test_merge_logs_sorted",
        "test_records_after",
    ],
    "lineno_format_check": [
        "test_filter_by_lineno",
        "test_filter_by_lineno_inclusive",
        "test_format_record",
        "test_running_error_count",
        "test_group_by_minute",
        "test_source_line_map",
    ],
}


# ---------------------------------------------------------------------------
# LLM prompt templates for generating fixes
# ---------------------------------------------------------------------------

REAL_ARM_PROMPTS = {
    "g1_direct_fix": (
        "Below is a Python module with a bug. Some tests are failing.\n\n"
        "IMPORTANT RULES:\n"
        "- Return the ENTIRE source code of the module, not just a snippet.\n"
        "- Keep ALL imports, classes, functions, and dataclasses EXACTLY as they are.\n"
        "- Only change the ONE line that has the bug. Do NOT rewrite or restructure.\n"
        "- The fix is likely a single character or operator change.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Failing test output:\n```\n{test_output}\n```\n\n"
        "Return the COMPLETE fixed module in a single ```python ... ``` block. "
        "The output must contain every function from the original."
    ),
    "g2_localized_fix": (
        "A Python module has a bug causing test failures. "
        "Your job: find the ONE broken line and fix it.\n\n"
        "RULES:\n"
        "- Copy the ENTIRE source code below and fix only the broken line.\n"
        "- Do NOT remove, rename, or restructure any functions.\n"
        "- Do NOT change imports or class definitions.\n"
        "- The bug is a wrong operator, wrong variable, or wrong constant.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Test failures:\n```\n{test_output}\n```\n\n"
        "Think step by step:\n"
        "1. Which tests fail and what do they check?\n"
        "2. Which function has the wrong behavior?\n"
        "3. What single-line change fixes it?\n\n"
        "Return the COMPLETE module (all functions) in a ```python ... ``` block."
    ),
    "g3_test_guided_fix": (
        "Fix the bug in this module. The test code shows expected behavior.\n\n"
        "CRITICAL: Return the ENTIRE source code with the fix applied. "
        "Do NOT rewrite the module. Do NOT remove any functions. "
        "Only change the line that causes the test failure.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Test code:\n```python\n{test_code}\n```\n\n"
        "Test output:\n```\n{test_output}\n```\n\n"
        "Return the COMPLETE fixed module in a ```python ... ``` block."
    ),
    "g4_error_focused_fix": (
        "A Python module has a subtle bug — likely a wrong operator or value.\n\n"
        "RULES: Copy the ENTIRE source code and change ONLY the broken line. "
        "Keep all functions, classes, imports exactly as-is.\n\n"
        "Source code:\n```python\n{source_code}\n```\n\n"
        "Errors:\n```\n{test_output}\n```\n\n"
        "Return the COMPLETE fixed module in a ```python ... ``` block."
    ),
}


# ---------------------------------------------------------------------------
# Bug metadata (for reporting)
# ---------------------------------------------------------------------------

BUG_METADATA = {
    "bug_r1":  {"difficulty": "easy",   "description": "filter_by_level: > instead of >="},
    "bug_r2":  {"difficulty": "easy",   "description": "filter_by_source: exact match uses 'in'"},
    "bug_r3":  {"difficulty": "medium", "description": "filter_by_time: inverted end boundary"},
    "bug_r4":  {"difficulty": "medium", "description": "error_windows: off-by-one in context"},
    "bug_r5":  {"difficulty": "medium", "description": "error_rate: missing CRITICAL"},
    "bug_r6":  {"difficulty": "hard",   "description": "latest_by_source: inverted comparison"},
    "bug_r7":  {"difficulty": "hard",   "description": "top_sources: ascending instead of descending"},
    "bug_r8":  {"difficulty": "hard",   "description": "LEVELS: missing CRITICAL entry"},
    "bug_r9":  {"difficulty": "easy",   "description": "filter_by_time: inverted start boundary"},
    "bug_r10": {"difficulty": "easy",   "description": "count_by_level: off-by-one default"},
    "bug_r11": {"difficulty": "easy",   "description": "parse_log: split on whitespace not newlines"},
    "bug_r12": {"difficulty": "easy",   "description": "LEVELS: WARNING same value as ERROR"},
    "bug_r13": {"difficulty": "medium", "description": "parse_line: level/src groups swapped"},
    "bug_r14": {"difficulty": "medium", "description": "error_windows: n_after used for start index"},
    "bug_r15": {"difficulty": "medium", "description": "error_rate: off-by-one denominator"},
    "bug_r16": {"difficulty": "medium", "description": "count_by_source: counts by level instead"},
    "bug_r17": {"difficulty": "hard",   "description": "latest_by_source: keyed by level not src"},
    "bug_r18": {"difficulty": "hard",   "description": "error_windows: ignores CRITICAL errors"},
    "bug_r19": {"difficulty": "hard",   "description": "error_rate: inverted fraction"},
    "bug_r20": {"difficulty": "hard",   "description": "filter_by_level: .lower() breaks lookup"},
    "bug_r21": {"difficulty": "easy",   "description": "running_error_count: initial count 1 not 0"},
    "bug_r22": {"difficulty": "easy",   "description": "parse_line: lineno stays string not int"},
    "bug_r23": {"difficulty": "easy",   "description": "severity_score: WARNING weight wrong (1 not 3)"},
    "bug_r24": {"difficulty": "easy",   "description": "has_critical: checks ERROR not CRITICAL"},
    "bug_r25": {"difficulty": "easy",   "description": "format_record: wrong date separator /"},
    "bug_r26": {"difficulty": "easy",   "description": "filter_by_lineno: > instead of >= lower bound"},
    "bug_r27": {"difficulty": "easy",   "description": "deduplicate: checks level not msg"},
    "bug_r28": {"difficulty": "easy",   "description": "search_messages: searches src not msg"},
    "bug_r29": {"difficulty": "easy",   "description": "group_by_minute: forgets to append record"},
    "bug_r30": {"difficulty": "easy",   "description": "time_range: returns None instead of tuple"},
    "bug_r31": {"difficulty": "medium", "description": "search_messages: case sensitive (no IGNORECASE)"},
    "bug_r32": {"difficulty": "medium", "description": "time_range: returns (latest, earliest) swapped"},
    "bug_r33": {"difficulty": "medium", "description": "running_error_count: ignores CRITICAL"},
    "bug_r34": {"difficulty": "medium", "description": "merge_logs: duplicates first list"},
    "bug_r35": {"difficulty": "medium", "description": "merge_logs: sorts by src not timestamp"},
    "bug_r36": {"difficulty": "medium", "description": "severity_score: CRITICAL weight 5 not 10"},
    "bug_r37": {"difficulty": "medium", "description": "filter_by_lineno: < instead of <= upper bound"},
    "bug_r38": {"difficulty": "medium", "description": "running_error_count: off-by-one (total - 1)"},
    "bug_r39": {"difficulty": "medium", "description": "format_record: level and src swapped"},
    "bug_r40": {"difficulty": "medium", "description": "filter_by_source: non-exact uses == not in"},
    "bug_r41": {"difficulty": "hard",   "description": "group_by_minute: no time bucketing"},
    "bug_r42": {"difficulty": "hard",   "description": "deduplicate: appends msg string not record"},
    "bug_r43": {"difficulty": "hard",   "description": "time_range: extracts lineno not timestamp"},
    "bug_r44": {"difficulty": "hard",   "description": "deduplicate: tracks src not msg in seen set"},
    "bug_r45": {"difficulty": "hard",   "description": "running_error_count: increments by 2"},
    "bug_r46": {"difficulty": "hard",   "description": "unique_levels: returns sources not levels"},
    "bug_r47": {"difficulty": "hard",   "description": "records_after: returns before not after"},
    "bug_r48": {"difficulty": "hard",   "description": "source_line_map: adds level not lineno"},
    "bug_r49": {"difficulty": "hard",   "description": "parse_line: wrong regex group for msg"},
    "bug_r50": {"difficulty": "hard",   "description": "time_range: treats single record as empty"},
}
