"""Download the W&B artifacts the diversity analysis replays into BCCA_DATA_DIR.

Layout (what */scripts/data.py expects): one directory per artifact version,
    <BCCA_DATA_DIR>/<collection>__<version>/<artifact files> + _artifact_meta.json
where _artifact_meta.json records spec, created, digest and the logging run.

Default set (180 artifacts, ~100 MB):
  * every calibration / Self-Refine / Reflexion version pinned in action_traces/versions.json, and
  * the max-overlap Self-Refine versions in diversity_fixes/chain_overlap.csv (sr_mx_version),
    which the step1mx_* variants select among.
--diagnostics adds every version of the per-cell critic_results / selfrefine_records /
reflexion_records collections plus the kernel artifacts used by the protocol-search scripts
(action_traces/scripts/{prior_test,kernel_test,repro,diag_colleague}.py).

usage:
  python fetch_artifacts.py [--dest DIR] [--diagnostics] [--dry-run] [--check] [--workers N]
Needs `pip install wandb` and access to the project (wandb login / WANDB_API_KEY).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

PROJECT = "nlpresearch.group/orchestration-hypothesis-testing"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DEFAULT_DEST = os.environ.get("BCCA_DATA_DIR") or os.path.join(REPO_ROOT, "data", "diversity_artifacts")
DIAG_KERNELS = ["lcb_easy_gpt5_mini_selfrefine_kernel:v0", "lcb_easy_gpt5_mini_reflexion_kernel:v0",
                "codecontests_gpt_oss_20b_local_selfrefine_kernel:v0"]


def pinned():
    """{spec: expected digest or None} from versions.json + chain_overlap.csv."""
    cells = json.load(open(os.path.join(HERE, "action_traces", "versions.json")))["cells"]
    out = {}
    for v in cells.values():
        for k in ("calibration", "selfrefine", "reflexion"):
            if v.get(k):
                out[v[k]] = v.get("artifact_meta", {}).get(k, {}).get("digest")
    for r in csv.DictReader(open(os.path.join(HERE, "diversity_fixes", "chain_overlap.csv"))):
        for k in ("sr_at_version", "sr_mx_version", "rx_at_version"):
            if r.get(k):
                out.setdefault(r[k], None)
    return out


def diagnostic_specs(api, base):
    """All versions of the collections behind `base` (critic_results / *_records) + DIAG_KERNELS."""
    colls = sorted({s.split(":")[0] for s in base if s.split(":")[0].endswith(("_critic_results", "_records"))})
    specs = set(DIAG_KERNELS)
    for name in colls:
        try:
            typ = api.artifact(f"{PROJECT}/{name}:v0").type
            specs.update(f"{name}:{a.version}" for a in api.artifacts(typ, f"{PROJECT}/{name}"))
        except Exception as e:  # noqa: BLE001
            print(f"warn: cannot list {name}: {e}", file=sys.stderr)
    return specs


def fetch(api, spec, dest):
    d = os.path.join(dest, spec.replace(":", "__"))
    meta_path = os.path.join(d, "_artifact_meta.json")
    if os.path.exists(meta_path):
        return spec, "skip"
    try:
        a = api.artifact(f"{PROJECT}/{spec}")
        a.download(root=d)
        try:
            run = a.logged_by()
        except Exception:  # noqa: BLE001
            run = None
        meta = {"spec": spec, "created": a.created_at, "digest": a.digest, "metadata": a.metadata,
                "files": [f.name for f in a.files()],
                "logged_by": run.id if run else None, "run_name": run.name if run else None,
                "run_config": dict(run.config) if run else None}
        with open(meta_path, "w") as f:   # written last: an interrupted download is retried
            json.dump(meta, f, indent=1, default=str)
        return spec, "ok"
    except Exception as e:  # noqa: BLE001
        return spec, f"ERROR {e}"


def check(specs, dest):
    bad = 0
    for spec, digest in sorted(specs.items()):
        p = os.path.join(dest, spec.replace(":", "__"), "_artifact_meta.json")
        if not os.path.exists(p):
            print("missing", spec); bad += 1
        elif digest and json.load(open(p)).get("digest") != digest:
            print("digest mismatch", spec); bad += 1
    print(f"{len(specs) - bad}/{len(specs)} artifacts present in {dest}" + ("" if bad else " (digests match versions.json)"))
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=DEFAULT_DEST, help="artifact dir (default: $BCCA_DATA_DIR or <repo>/data/diversity_artifacts)")
    ap.add_argument("--diagnostics", action="store_true", help="also fetch all versions used by the protocol-search scripts")
    ap.add_argument("--dry-run", action="store_true", help="list the pinned specs and exit")
    ap.add_argument("--check", action="store_true", help="verify the pinned artifacts are present (no network)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dest = os.path.abspath(args.dest)

    specs = pinned()
    if args.dry_run:
        print("\n".join(sorted(specs)))
        print(f"{len(specs)} artifacts -> {dest}", file=sys.stderr)
        return
    if args.check:
        sys.exit(0 if check(specs, dest) else 1)

    import wandb
    api = wandb.Api(timeout=300)
    todo = set(specs)
    if args.diagnostics:
        todo |= diagnostic_specs(api, specs)
    os.makedirs(dest, exist_ok=True)
    errors = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for spec, status in ex.map(lambda s: fetch(api, s, dest), sorted(todo)):
            if status.startswith("ERROR"):
                errors += 1
                print(spec, status, file=sys.stderr, flush=True)
    print(f"{len(todo) - errors}/{len(todo)} artifacts in {dest}")
    ok = check(specs, dest)
    sys.exit(0 if ok and not errors else 1)


if __name__ == "__main__":
    main()
