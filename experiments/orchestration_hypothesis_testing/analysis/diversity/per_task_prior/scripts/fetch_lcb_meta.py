"""Stream livecodebench/code_generation_lite test.jsonl..test6.jsonl and keep only task metadata
(no test cases are stored). test.jsonl is read from the local HF cache when present.
Output: cache/lcb_meta.jsonl (one row per question_id, first occurrence wins, as calibration/lcb.py)."""
import json, os, sys
import requests

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "lcb_meta.jsonl")
LOCAL = os.path.join(os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub"),
                     "datasets--livecodebench--code_generation_lite/snapshots/0fe84c3912ea0c4d4a78037083943e8f0c4dd505/")
URL = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/{}"
KEEP = ("question_id", "platform", "difficulty", "contest_date", "contest_id", "question_title", "question_content", "starter_code")


def rows(fname):
    p = os.path.join(LOCAL, fname)
    if os.path.exists(p):
        with open(p, "rb") as f:
            for l in f:
                yield l
        return
    with requests.get(URL.format(fname), stream=True, timeout=120) as r:
        r.raise_for_status()
        for l in r.iter_lines(chunk_size=1 << 20):
            yield l


done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        done.add(json.loads(l)["_file"])
with open(OUT, "a") as fo:
    for i, fname in enumerate(["test.jsonl"] + [f"test{k}.jsonl" for k in range(2, 7)]):
        if fname in done:
            print("skip", fname, flush=True); continue
        buf = []
        for l in rows(fname):
            if not l.strip():
                continue
            r = json.loads(l)
            o = {k: r.get(k) for k in KEEP}
            try:
                o["n_public_tests"] = len(json.loads(r["public_test_cases"]))
            except Exception:
                o["n_public_tests"] = None
            o["_file"] = fname; o["_release"] = i + 1
            buf.append(o)
        for o in buf:
            fo.write(json.dumps(o) + "\n")
        fo.flush()
        print("done", fname, len(buf), flush=True)
