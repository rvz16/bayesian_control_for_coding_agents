#!/usr/bin/env python3
"""Fetch CodeContests test-split problem statements (name -> description) into cache/cc_test_desc.json.

Used only by make_traces_md.py to show task descriptions. Network; downloads one parquet file
(~165 problems) from the deepmind/code_contests dataset on the HF hub. Not committed (third-party text).
"""
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

OUT = Path(__file__).resolve().parents[1] / "cache" / "cc_test_desc.json"


def main():
    files = [f for f in list_repo_files("deepmind/code_contests", repo_type="dataset") if "/test-" in f]
    path = hf_hub_download("deepmind/code_contests", files[0], repo_type="dataset")
    rows = pq.read_table(path, columns=["name", "description"]).to_pylist()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({r["name"]: r["description"] for r in rows}))
    print(f"wrote {OUT} ({len(rows)} problems)")


if __name__ == "__main__":
    main()
