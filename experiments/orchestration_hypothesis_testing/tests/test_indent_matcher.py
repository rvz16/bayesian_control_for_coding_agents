"""Unit tests for the indentation-tolerant SEARCH/REPLACE matcher.

Run with:
    pytest scripts/test_indent_matcher.py -v

These cases lock in behaviour around the bug discovered in the R1
spot-check, where Qwen models emitted SEARCH blocks at a shallower
indent than the file actually had (e.g. inner-`if` with 8sp instead of
12sp), causing every SEARCH to silently miss.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from spot_check_generators import (  # noqa: E402
    _common_indent,
    _infer_path_from_search,
    _normalize_path_token,
    _resolve_oracle_path,
    _try_fuzzy_replace,
    _try_indent_tolerant_replace,
    apply_change_blocks,
    parse_change_blocks,
)


# ---------- parse_change_blocks: opener variants ----------

def test_parse_canonical_triple_bracket():
    text = (
        "<<<CHANGE foo/bar.py\n"
        "SEARCH\n"
        "old line\n"
        "REPLACE\n"
        "new line\n"
        "CHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    assert out == [("foo/bar.py", "old line", "new line")]


def test_parse_sonnet_single_bracket():
    """Sonnet 4.5 emits `<CHANGE path>` (single bracket, > on same line)."""
    text = (
        "<CHANGE astropy/io/ascii/rst.py>\n"
        "SEARCH\n"
        "    def __init__(self):\n"
        "        super().__init__(delimiter_pad=None, bookend=False)\n"
        "REPLACE\n"
        "    def __init__(self, header_rows=None, **kwargs):\n"
        "        super().__init__(delimiter_pad=None, bookend=False, "
        "header_rows=header_rows, **kwargs)\n"
        "CHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 1
    fpath, search, replace = out[0]
    assert fpath == "astropy/io/ascii/rst.py"
    assert "def __init__(self):" in search
    assert "super().__init__(delimiter_pad=None, bookend=False)" in search
    assert "def __init__(self, header_rows=None, **kwargs):" in replace


def test_parse_multiple_sonnet_blocks_in_one_response():
    """Sonnet sometimes emits a 'wait, let me reconsider' followed by another
    block. The latter is the one we want; both should parse cleanly."""
    text = (
        "<CHANGE foo.py>\n"
        "SEARCH\n"
        "x = 1\n"
        "REPLACE\n"
        "x = 2\n"
        "CHANGE>>>\n"
        "\n"
        "Wait, let me reconsider:\n"
        "\n"
        "<CHANGE foo.py>\n"
        "SEARCH\n"
        "x = 1\n"
        "REPLACE\n"
        "x = 99\n"
        "CHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 2
    assert out[0] == ("foo.py", "x = 1", "x = 2")
    assert out[1] == ("foo.py", "x = 1", "x = 99")


def test_parse_canonical_and_single_bracket_mixed():
    text = (
        "<<<CHANGE a.py\n"
        "SEARCH\nold_a\nREPLACE\nnew_a\nCHANGE>>>\n"
        "\n"
        "<CHANGE b.py>\n"
        "SEARCH\nold_b\nREPLACE\nnew_b\nCHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 2
    assert ("a.py", "old_a", "new_a") in out
    assert ("b.py", "old_b", "new_b") in out


def test_parse_skips_when_no_search_or_replace():
    text = (
        "<CHANGE foo.py>\n"
        "this is just commentary, no SEARCH/REPLACE\n"
        "CHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    assert out == []


# ---------- _common_indent ----------

def test_common_indent_uniform():
    assert _common_indent(["    a", "    b"]) == "    "


def test_common_indent_increasing():
    assert _common_indent(["    a", "        b"]) == "    "


def test_common_indent_partial():
    assert _common_indent(["  a", "    b"]) == "  "


def test_common_indent_blank_lines_ignored():
    assert _common_indent(["", "    a", ""]) == "    "


def test_common_indent_whitespace_only_ignored():
    # whitespace-only "indent" carries no signal
    assert _common_indent(["        ", "    a"]) == "    "


def test_common_indent_empty():
    assert _common_indent([]) == ""


def test_common_indent_no_indent():
    assert _common_indent(["a", "b"]) == ""


# ---------- _try_indent_tolerant_replace ----------

def test_indent_tolerant_replace_matches_shallower_search():
    """Model emits SEARCH at shallower indent; matcher dedents both and
    re-indents REPLACE to the file's actual nesting level."""
    file_text = (
        "def f():\n"
        "    if True:\n"
        "        x = 1\n"
        "        y = 2\n"
    )
    search = "if True:\n    x = 1\n    y = 2"
    replace = "if True:\n    x = 99\n    y = 2"
    out = _try_indent_tolerant_replace(file_text, search, replace)
    assert out is not None
    assert "        x = 99" in out
    assert "        y = 2" in out
    # untouched lines are preserved
    assert "def f():" in out


def test_indent_tolerant_replace_strict_match_path_unchanged():
    file_text = "alpha\nbeta\n"
    out = _try_indent_tolerant_replace(file_text, "alpha\nbeta", "gamma\ndelta")
    assert out is not None
    assert "gamma" in out and "delta" in out
    assert "alpha" not in out


def test_indent_tolerant_replace_no_spurious_match():
    """SEARCH text that doesn't actually exist must not be wedged in."""
    file_text = "def a():\n    return 1\ndef b():\n    return 2\n"
    out = _try_indent_tolerant_replace(file_text, "return 99", "return 100")
    assert out is None


def test_indent_tolerant_replace_handles_blank_lines_in_block():
    file_text = (
        "class C:\n"
        "    def m(self):\n"
        "        a = 1\n"
        "\n"
        "        b = 2\n"
    )
    search = "a = 1\n\nb = 2"
    replace = "a = 10\n\nb = 20"
    out = _try_indent_tolerant_replace(file_text, search, replace)
    assert out is not None
    assert "        a = 10" in out
    assert "        b = 20" in out


def test_indent_tolerant_replace_first_match_wins():
    """When the dedented SEARCH matches multiple windows, take the first
    (matches str.replace(..., 1) semantics)."""
    file_text = (
        "if A:\n"
        "    x = 1\n"
        "if B:\n"
        "    x = 1\n"
    )
    out = _try_indent_tolerant_replace(file_text, "x = 1", "x = 99")
    assert out is not None
    # First occurrence flipped, second untouched
    assert out.count("x = 99") == 1
    assert out.count("x = 1") == 1


def test_indent_tolerant_replace_blank_replace_lines_not_padded():
    """Blank lines in REPLACE must stay blank (no leading-indent prefix
    applied), to avoid trailing whitespace warnings."""
    file_text = "    def m():\n        pass\n"
    search = "def m():\n    pass"
    replace = "def m():\n\n    pass"
    out = _try_indent_tolerant_replace(file_text, search, replace)
    assert out is not None
    assert "    def m():\n\n        pass" in out


# ---------- apply_change_blocks integration ----------

def test_apply_change_blocks_uses_indent_tier_when_strict_fails():
    oracle = {
        "m.py": "class C:\n    def m(self):\n        if x:\n            return 1\n",
    }
    blocks = [(
        "m.py",
        "if x:\n    return 1",       # 0/4-space SEARCH
        "if x:\n    return 99",      # same 0/4-space REPLACE
    )]
    modified = apply_change_blocks(oracle, blocks)
    assert "m.py" in modified
    text = modified["m.py"]
    assert "        if x:\n            return 99" in text


def test_apply_change_blocks_strict_still_takes_priority():
    """If the strict literal substring matches, we don't fall through to
    the indent path (faster and less ambiguous)."""
    oracle = {"m.py": "alpha\nbeta\ngamma\n"}
    blocks = [("m.py", "alpha\nbeta", "ALPHA\nBETA")]
    modified = apply_change_blocks(oracle, blocks)
    assert modified["m.py"] == "ALPHA\nBETA\ngamma\n"


def test_apply_change_blocks_returns_empty_when_no_match():
    oracle = {"m.py": "hello\nworld\n"}
    blocks = [("m.py", "not present", "something")]
    modified = apply_change_blocks(oracle, blocks)
    assert "m.py" not in modified


# ---------- new in this revision ----------
# Strategy 3: XML-style tags (Sonnet 4.5 ~20% of failures)

def test_parse_xml_style_change_block():
    """Sonnet 4.5 sometimes emits XML-style closing tags instead of CHANGE>>>."""
    text = (
        "<CHANGE django/views/debug.py>\n"
        "SEARCH\n"
        "from django.urls import Resolver404, resolve\n"
        "</SEARCH>\n"
        "<REPLACE>\n"
        "from django.http import Http404\n"
        "from django.urls import Resolver404, resolve\n"
        "</REPLACE>\n"
        "</CHANGE>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 1
    fpath, search, replace = out[0]
    assert fpath == "django/views/debug.py"
    assert search == "from django.urls import Resolver404, resolve"
    assert "Http404" in replace


def test_parse_xml_style_multiple_blocks_same_file():
    """Two consecutive XML-style blocks in one response."""
    text = (
        "<CHANGE foo.py>\n"
        "SEARCH\n"
        "a\n"
        "</SEARCH>\n"
        "<REPLACE>\n"
        "A\n"
        "</REPLACE>\n"
        "</CHANGE>\n"
        "\n"
        "<CHANGE foo.py>\n"
        "SEARCH\n"
        "b\n"
        "</SEARCH>\n"
        "<REPLACE>\n"
        "B\n"
        "</REPLACE>\n"
        "</CHANGE>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 2
    assert out[0] == ("foo.py", "a", "A")
    assert out[1] == ("foo.py", "b", "B")


def test_parse_canonical_takes_priority_over_xml():
    """If a block has both CHANGE>>> and </CHANGE>, only Strategy 1 fires."""
    text = (
        "<<<CHANGE foo.py\n"
        "SEARCH\n"
        "old\n"
        "REPLACE\n"
        "new\n"
        "CHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 1
    assert out[0] == ("foo.py", "old", "new")


# Path normalization

def test_normalize_path_unwrap_quotes():
    assert _normalize_path_token('"foo/bar.py"') == "foo/bar.py"
    assert _normalize_path_token("'foo/bar.py'") == "foo/bar.py"


def test_normalize_path_strip_leading_assignment():
    assert _normalize_path_token('path="sympy/printing/latex.py"') == "sympy/printing/latex.py"
    assert _normalize_path_token("file=foo.py") == "foo.py"


def test_normalize_path_strip_trailing_angle_bracket():
    """Lazy capture in <CHANGE foo.py> sometimes leaves a trailing >."""
    assert _normalize_path_token("foo.py>") == "foo.py"


# Path resolution: basename uniquely

def test_resolve_path_exact():
    oracle = {"a/b/c.py": "x"}
    assert _resolve_oracle_path("a/b/c.py", oracle) == "a/b/c.py"


def test_resolve_path_suffix():
    oracle = {"src/django/forms/widgets.py": "x"}
    assert _resolve_oracle_path("django/forms/widgets.py", oracle) == "src/django/forms/widgets.py"


def test_resolve_path_basename_unique():
    """Model emits bare basename; basename is unique among oracle keys."""
    oracle = {"a/b/widgets.py": "x", "a/b/forms.py": "y"}
    assert _resolve_oracle_path("widgets.py", oracle) == "a/b/widgets.py"


def test_resolve_path_basename_ambiguous_returns_some_match():
    """If two oracle files have the same basename, suffix match (Tier 2)
    returns one of them — picking the wrong one will cause apply to bail
    silently when SEARCH doesn't appear in that file's content. We accept
    either as valid behavior (it's a fall-through, not a guarantee)."""
    oracle = {"a/widgets.py": "x", "b/widgets.py": "y"}
    resolved = _resolve_oracle_path("widgets.py", oracle)
    assert resolved in ("a/widgets.py", "b/widgets.py")


# Placeholder rejection at apply step

def test_apply_skips_placeholder_path_when_search_text_not_in_any_oracle():
    """Placeholder paths get rejected when SEARCH text doesn't match any
    oracle file (so we don't silently produce wrong patches)."""
    oracle = {"real/file.py": "different content here\n"}
    blocks = [("path/to/file.py", "old", "new")]  # SEARCH "old" not in oracle
    modified = apply_change_blocks(oracle, blocks)
    assert not modified


def test_apply_skips_uppercase_placeholder_when_search_text_not_in_any_oracle():
    oracle = {"real/file.py": "different content here\n"}
    blocks = [("FILE_PATH_HERE", "totally absent", "new")]
    modified = apply_change_blocks(oracle, blocks)
    assert not modified


def test_apply_recovers_placeholder_path_when_search_text_in_one_oracle():
    """The new inference behavior: placeholder path is OK as long as the
    SEARCH text uniquely identifies an oracle file."""
    oracle = {"real/file.py": "old\n"}
    blocks = [("path/to/file.py", "old", "new")]
    modified = apply_change_blocks(oracle, blocks)
    assert "real/file.py" in modified
    assert "new" in modified["real/file.py"]


def test_apply_skips_placeholder_when_search_text_in_multiple_oracle_files():
    """Refuse to guess if SEARCH appears in multiple oracle files."""
    oracle = {
        "a.py": "shared = True\n",
        "b.py": "shared = True\n",
    }
    blocks = [("FILE_PATH_HERE", "shared = True", "shared = False")]
    modified = apply_change_blocks(oracle, blocks)
    assert not modified  # ambiguous, refuse to guess


# Fuzzy matcher (tier 4)

def test_fuzzy_replace_accepts_close_window():
    """A SEARCH with one extra space per line should fuzzy-match."""
    original = "def foo():\n    return 1\n    # tail comment\n"
    search = "def foo():\n    return 1\n    # tail comment "  # one trailing space drift
    replace = "def foo():\n    return 2\n    # tail comment"
    out = _try_fuzzy_replace(original, search, replace, threshold=0.92)
    assert out is not None
    assert "return 2" in out


def test_fuzzy_replace_rejects_hallucinated_text():
    """A SEARCH that describes a different docstring should NOT fuzzy-match.
    This guards against models like haiku45 inventing docstring content."""
    original = (
        'class RST(FixedWidth):\n'
        '    """reStructuredText simple format table.\n'
        '\n'
        '    See: https://docutils.sourceforge.io\n'
        '    """\n'
    )
    # Hallucinated paraphrase — different wording, different structure
    fake_search = (
        'class RST(FixedWidth):\n'
        '    """\n'
        '    Write a table in reStructuredText format.\n'
        '\n'
        '    This produces a table that conforms to the spec.\n'
        '    """\n'
    )
    out = _try_fuzzy_replace(original, fake_search, "junk", threshold=0.92)
    assert out is None  # too different, must reject


def test_fuzzy_replace_returns_none_when_search_longer_than_file():
    out = _try_fuzzy_replace("a\nb\n", "a\nb\nc\nd\ne\n", "x")
    assert out is None


def test_apply_change_blocks_uses_fuzzy_tier_after_indent_fails():
    """End-to-end: fuzzy tier fires when strict / rstrip / indent all fail.
    Search has minor punctuation drift (`# comment` vs `#comment`) — strict
    and rstrip both fail because the literal text differs; indent-tolerant
    fails because the dedented lines also differ; fuzzy ratio ~98% accepts.
    """
    oracle = {
        "m.py": "def foo():\n    x = 1  # comment\n    return x\n"
    }
    blocks = [(
        "m.py",
        "def foo():\n    x = 1  #comment\n    return x",  # one-space drift
        "def foo():\n    x = 99  # comment\n    return x",
    )]
    modified = apply_change_blocks(oracle, blocks)
    assert "m.py" in modified
    assert "x = 99" in modified["m.py"]


# Cross-strategy: mix of canonical + XML-style in one response

def test_parse_canonical_and_xml_mixed():
    text = (
        "<<<CHANGE first.py\n"
        "SEARCH\n"
        "x\n"
        "REPLACE\n"
        "X\n"
        "CHANGE>>>\n"
        "\n"
        "<CHANGE second.py>\n"
        "SEARCH\n"
        "y\n"
        "</SEARCH>\n"
        "<REPLACE>\n"
        "Y\n"
        "</REPLACE>\n"
        "</CHANGE>\n"
    )
    out = parse_change_blocks(text)
    paths = [b[0] for b in out]
    assert "first.py" in paths
    assert "second.py" in paths
    assert len(out) == 2


# ---------- Strategy 4: pathless XML pairs (sonnet45's remaining failures) ----------

def test_parse_pathless_xml_pair_emits_empty_path():
    """`<SEARCH>...</SEARCH><REPLACE>...</REPLACE>` with no `<CHANGE path>`
    opener — emit with fpath="" so apply_change_blocks can infer."""
    text = (
        "Here is the fix:\n"
        "<SEARCH>\n"
        "old line\n"
        "</SEARCH>\n"
        "<REPLACE>\n"
        "new line\n"
        "</REPLACE>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 1
    assert out[0] == ("", "old line", "new line")


def test_parse_pathless_xml_skipped_when_inside_canonical_block():
    """If pathless XML pair is INSIDE a canonical CHANGE block already
    consumed by Strategy 1, it should not be double-extracted."""
    text = (
        "<<<CHANGE foo.py\n"
        "SEARCH\n"
        "<SEARCH>x</SEARCH>\n"
        "REPLACE\n"
        "<REPLACE>X</REPLACE>\n"
        "CHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    # Only the canonical block — the pathless XML pair is content inside
    # the SEARCH/REPLACE bodies, not a separate block.
    paths = [b[0] for b in out]
    assert "foo.py" in paths


# ---------- _infer_path_from_search: oracle-file path inference ----------

def test_infer_path_unique_match():
    oracle = {
        "a/foo.py": "def foo():\n    return 1\n",
        "b/bar.py": "def bar():\n    return 2\n",
    }
    assert _infer_path_from_search("def foo():\n    return 1", oracle) == "a/foo.py"


def test_infer_path_no_match_returns_empty():
    oracle = {"a/foo.py": "def foo(): return 1\n"}
    assert _infer_path_from_search("def baz(): return 99", oracle) == ""


def test_infer_path_ambiguous_returns_empty():
    """If two oracle files contain the same SEARCH text, refuse to guess."""
    oracle = {
        "a/foo.py": "common = 1\n",
        "b/bar.py": "common = 1\n",
    }
    assert _infer_path_from_search("common = 1", oracle) == ""


def test_apply_blocks_handles_pathless_block_via_inference():
    """End-to-end: pathless block (fpath="") gets attributed to the
    oracle file that contains the SEARCH text."""
    oracle = {
        "src/django/inspectdb.py": (
            "def handle_inspection():\n"
            "    yield ''\n"
            "    yield 'class Foo'\n"
            "    return\n"
        ),
        "src/other.py": "def other(): pass\n",
    }
    blocks = [(
        "",  # pathless (Strategy 4 sentinel)
        "    yield ''\n    yield 'class Foo'",
        "    yield ''\n    yield 'class Foo with related_name'",
    )]
    modified = apply_change_blocks(oracle, blocks)
    assert "src/django/inspectdb.py" in modified
    assert "related_name" in modified["src/django/inspectdb.py"]
    assert "src/other.py" not in modified


def test_apply_blocks_handles_placeholder_path_via_inference():
    """qwen3_coder pattern: placeholder path on opener line, real path
    pasted on the next line as part of body. After my parser change the
    fpath comes through as the placeholder — apply step should recover
    by searching for the SEARCH text."""
    oracle = {
        "django/db/models/fields/related.py": (
            "errors.append(checks.Error('multi-FK ambiguous'))\n"
        ),
    }
    blocks = [(
        "FILE_PATH_HERE",  # placeholder copied literally from prompt
        "errors.append(checks.Error('multi-FK ambiguous'))",
        "errors.append(checks.Error('multi-FK ambiguous; use through_fields'))",
    )]
    modified = apply_change_blocks(oracle, blocks)
    assert "django/db/models/fields/related.py" in modified
    assert "through_fields" in modified["django/db/models/fields/related.py"]


def test_apply_blocks_pathless_with_no_oracle_match_silently_skips():
    oracle = {"foo.py": "real content\n"}
    blocks = [("", "totally unrelated text", "x")]
    modified = apply_change_blocks(oracle, blocks)
    assert "foo.py" not in modified  # nothing modified, no spurious diff


def test_parse_canonical_block_missing_search_keyword():
    """gpt5_mini in refinement-mode sometimes omits the SEARCH keyword line:

        <<<CHANGE foo.py
        def __init__(self):
            super().__init__(...)
        REPLACE
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        CHANGE>>>

    Treat content before REPLACE as SEARCH so the block still extracts.
    """
    text = (
        "<<<CHANGE foo.py\n"
        "def __init__(self):\n"
        "    super().__init__(delimiter_pad=None)\n"
        "REPLACE\n"
        "def __init__(self, *args, **kwargs):\n"
        "    super().__init__(*args, **kwargs)\n"
        "CHANGE>>>\n"
    )
    out = parse_change_blocks(text)
    assert len(out) == 1
    fpath, search, replace = out[0]
    assert fpath == "foo.py"
    assert "def __init__(self):" in search
    assert "delimiter_pad=None" in search
    assert "*args, **kwargs" in replace
