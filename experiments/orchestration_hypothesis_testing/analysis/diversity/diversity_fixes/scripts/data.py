import json, os, glob, re
# W&B artifacts of nlpresearch.group/orchestration-hypothesis-testing, one <name>__<version>/ dir each
# (diversity/fetch_artifacts.py). Override with BCCA_DATA_DIR; default <repo root>/data/diversity_artifacts.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 6))
DL = os.path.abspath(os.environ.get("BCCA_DATA_DIR") or os.path.join(_REPO_ROOT, "data", "diversity_artifacts"))
BENCHES = ["lcb_hard","lcb_medium","lcb_easy","mbpp","humaneval","swe_lite","swe_verified","humanevalfix","codecontests"]
GENS = ["gpt5_mini","qwen3_coder","haiku45","sonnet45","qwen25_32b","gpt_oss_20b"]
def _jsonl(d):
    return [json.loads(l) for f in sorted(os.listdir(d)) if f.endswith(".jsonl") for l in open(os.path.join(d,f)) if l.strip()]
def versions(bench, gen, kind):
    """kind e.g. 'critic_results', 'selfrefine_records', 'selfrefine_kernel'. Returns list of (artifact_name, version, dir, created)."""
    out = []
    for g in ([gen, gen+"_local"] if gen=="gpt_oss_20b" else [gen]):
        for d in glob.glob(f"{DL}/{bench}_{g}_{kind}__v*"):
            v = d.rsplit("__",1)[1]
            meta = json.load(open(d+"/_artifact_meta.json"))
            out.append((f"{bench}_{g}_{kind}", v, d, meta["created"]))
    return sorted(out, key=lambda x: x[3])
def load_records(d):
    if not d or not os.path.isdir(d):
        raise FileNotFoundError(f"artifact dir missing: {d!r} (BCCA_DATA_DIR={DL}); run diversity/fetch_artifacts.py")
    return _jsonl(d)
def load_json(d):
    fs = [f for f in os.listdir(d) if f.endswith(".json") and not f.startswith("_")]
    return json.load(open(os.path.join(d, fs[0])))
