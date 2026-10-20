"""Task-level features from the public benchmark datasets (local HF cache + cache/lcb_meta.jsonl).

Only information an agent sees BEFORE any paid action is used: the task text and benchmark metadata.
Never used: reference/gold solutions, gold patches, test patches, hidden-test contents, FAIL_TO_PASS lists.
Hidden-test-suite SIZES are exported only as `priv_*` columns (reported in auroc.csv, never fed to a controller).

build() -> {bench: {iid: {"text": str, "num": {name: float}, "cat": {name: str}}}}, match report.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from datetime import datetime

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "cache")
H = os.path.join(os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub"), "")


def _pq(pat, columns=None):
    fs = sorted(glob.glob(H + pat))
    assert fs, pat
    return pq.read_table(fs[0], columns=columns).to_pylist()


def _days(s):
    if not s:
        return None
    try:
        return float((datetime.strptime(str(s)[:10], "%Y-%m-%d") - datetime(2020, 1, 1)).days)
    except ValueError:
        return None


def _ll(x):
    return math.log1p(len(x or ""))


def lcb():
    out = {}
    for l in open(os.path.join(CACHE, "lcb_meta.jsonl")):
        r = json.loads(l)
        q = str(r["question_id"])
        if q in out:
            continue
        txt = (r.get("question_title") or "") + "\n" + (r.get("question_content") or "")
        out[q] = dict(text=txt,
                      num=dict(log_len_statement=_ll(r.get("question_content")),
                               n_public_tests=float(r["n_public_tests"]) if r.get("n_public_tests") is not None else None,
                               contest_days=_days(r.get("contest_date")),
                               release=float(r["_release"]),
                               log_len_starter=_ll(r.get("starter_code")),
                               n_constraints_lines=float(len(re.findall(r"<=", r.get("question_content") or "")))),
                      cat=dict(platform=r.get("platform") or "", difficulty=r.get("difficulty") or ""))
    return out


def mbpp():
    out = {}
    for r in _pq("datasets--evalplus--mbppplus/snapshots/*/data/*.parquet", ["task_id", "prompt", "test_list"]):
        tl = list(r["test_list"]) if r["test_list"] is not None else []
        txt = (r["prompt"] or "") + "\n" + "\n".join(tl)
        out[str(r["task_id"])] = dict(text=txt,
                                      num=dict(log_len_statement=_ll(r["prompt"]), n_public_tests=float(len(tl)),
                                               log_len_tests=_ll("\n".join(tl))),
                                      cat={})
    return out


def _hepack():
    return _pq("datasets--bigcode--humanevalpack/snapshots/*/python/*.parquet")


def humaneval(cal_by=None):
    out = {}
    for r in _hepack():
        k = r["task_id"].replace("Python/", "HumanEval/")
        p = r["prompt"] or ""
        doc = r.get("docstring") or ""
        out[k] = dict(text=p,
                      num=dict(log_len_statement=_ll(p), log_len_docstring=_ll(doc),
                               n_doc_examples=float(p.count(">>>")),
                               n_args=float(max(0, (r.get("signature") or "").count(",") + 1) if "(" in (r.get("signature") or "") and "()" not in (r.get("signature") or "") else 0)),
                      cat={})
    # number of public (base) inputs = the L2 test set: task-level constant recorded in critic_results
    if cal_by:
        for i, recs in cal_by.items():
            if i in out and recs and recs[0].get("n_base_inputs") is not None:
                out[i]["num"]["n_public_tests"] = float(recs[0]["n_base_inputs"])
                out[i]["num"]["priv_n_hidden_tests"] = float(recs[0].get("n_plus_inputs") or 0)
    return out


def humanevalfix():
    out = {}
    for r in _hepack():
        p = r["prompt"] or ""
        bug = r.get("buggy_solution") or ""
        ex = r.get("example_test") or ""
        out[r["task_id"]] = dict(text=(r.get("docstring") or p) + "\n" + bug,
                                 num=dict(log_len_statement=_ll(p), log_len_buggy=_ll(bug),
                                          n_buggy_lines=float(len([x for x in bug.splitlines() if x.strip()])),
                                          n_doc_examples=float(p.count(">>>")),
                                          n_public_tests=float(ex.count("assert"))),
                                 cat=dict(bug_type=r.get("bug_type") or "", failure_symptoms=r.get("failure_symptoms") or ""))
    return out


def codecontests():
    out = {}
    cols = ["name", "description", "public_tests", "private_tests", "generated_tests", "source", "difficulty",
            "cf_rating", "cf_index", "cf_tags", "time_limit", "memory_limit_bytes", "cf_contest_id"]
    for r in _pq("datasets--deepmind--code_contests/snapshots/*/data/test-*.parquet", cols):
        tl = r.get("time_limit") or {}
        idx = (r.get("cf_index") or "")[:1]
        out[r["name"]] = dict(text=r["description"] or "",
                              num=dict(log_len_statement=_ll(r["description"]),
                                       n_public_tests=float(len((r.get("public_tests") or {}).get("input") or [])),
                                       cf_rating=float(r["cf_rating"]) if r.get("cf_rating") else None,
                                       cc_difficulty=float(r["difficulty"]) if r.get("difficulty") is not None else None,
                                       cf_index=float(ord(idx) - ord("A")) if idx.isalpha() else None,
                                       n_tags=float(len(r.get("cf_tags") or [])),
                                       time_limit_s=float((tl.get("seconds") or 0) + (tl.get("nanos") or 0) / 1e9) if tl else None,
                                       log_mem=math.log(r["memory_limit_bytes"]) if r.get("memory_limit_bytes") else None,
                                       cf_contest_id=float(r["cf_contest_id"]) if r.get("cf_contest_id") else None,
                                       priv_n_hidden_tests=float(len((r.get("private_tests") or {}).get("input") or [])
                                                                 + len((r.get("generated_tests") or {}).get("input") or []))),
                              cat=dict(tags=" ".join(sorted(t.replace(" ", "_") for t in (r.get("cf_tags") or [])))))
    return out


SWE_DIFF = {"<15 min fix": 0.0, "15 min - 1 hour": 1.0, "1-4 hours": 2.0, ">4 hours": 3.0}


def swe(kind):
    pat = {"lite": "datasets--princeton-nlp--SWE-bench_Lite/snapshots/*/data/test-*.parquet",
           "verified": "datasets--princeton-nlp--SWE-bench_Verified/snapshots/*/data/test-*.parquet"}[kind]
    cols = ["repo", "instance_id", "problem_statement", "hints_text", "created_at", "version"] + (["difficulty"] if kind == "verified" else [])
    out = {}
    for r in _pq(pat, cols):
        ps = r["problem_statement"] or ""
        num = dict(log_len_statement=_ll(ps), log_len_hints=_ll(r.get("hints_text")),
                   has_hints=float(bool((r.get("hints_text") or "").strip())),
                   created_days=_days(r.get("created_at")),
                   n_code_blocks=float(ps.count("```")),
                   has_traceback=float("Traceback" in ps))
        if kind == "verified":
            num["swe_difficulty"] = SWE_DIFF.get(r.get("difficulty"), None)
        out[r["instance_id"]] = dict(text=ps, num=num, cat=dict(repo=r["repo"]))
    return out


def build(cal_by_bench):
    """cal_by_bench: {bench: {iid: [records]}} (union over generators; used for match rates + n_base_inputs)."""
    src = {"lcb_hard": lcb, "lcb_medium": lcb, "lcb_easy": lcb, "mbpp": mbpp,
           "humaneval": lambda: humaneval(cal_by_bench.get("humaneval")), "humanevalfix": humanevalfix,
           "codecontests": codecontests, "swe_lite": lambda: swe("lite"), "swe_verified": lambda: swe("verified")}
    meta, report, memo = {}, {}, {}
    for b, f in src.items():
        key = "lcb" if b.startswith("lcb") else b
        if key not in memo:
            memo[key] = f()
        M = memo[key]
        ids = list(cal_by_bench[b])
        meta[b] = {i: M[i] for i in ids if i in M}
        report[b] = dict(n_ids=len(ids), n_matched=len(meta[b]), match_rate=len(meta[b]) / len(ids),
                         unmatched_examples=[i for i in ids if i not in M][:5])
    return meta, report
