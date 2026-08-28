"""Aggregate spot-check results merging original + indent-fix delta reports.

For each (generator, pid) we union the resolved sets from
`*.{key}_p{pid}.json` and `*.{key}_p{pid}_indentfix.json`. The
`predictions.jsonl` file (now with the post-indent-fix `model_patch`
values) determines which (instance, pid) pairs were considered
"submitted" (non-empty diff).

Writes:
  - <data-dir>/summary.json   : final aggregated payload, replacing the
    pre-fix one.
  - <data-dir>/delta_summary.txt : human-readable before/after table.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import re
import sys
from datetime import datetime, timezone

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from spot_check_generators import (  # noqa: E402
    GeneratorSummary,
    parse_resolved,
    sample_instances,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("agg")


def _load_json(p: pathlib.Path) -> dict:
    return json.loads(p.read_text())


def _resolved_for(eval_dir: pathlib.Path, key: str, pid: int) -> tuple[set[str], list[str]]:
    """Union of resolved IDs from base report + indent-fix delta report.

    Returns (resolved_ids, sources_used).
    """
    resolved: set[str] = set()
    sources: list[str] = []
    for tag in (f"{key}_p{pid}", f"{key}_p{pid}_indentfix"):
        candidates = sorted(eval_dir.glob(f"*.{tag}.json"))
        if not candidates:
            continue
        report = _load_json(candidates[-1])
        resolved |= parse_resolved(report)
        sources.append(candidates[-1].name)
    return resolved, sources


def aggregate(
    key: str,
    model: str,
    instances: list[dict],
    pred_path: pathlib.Path,
    eval_dir: pathlib.Path,
    n_patches: int,
) -> tuple[GeneratorSummary, dict]:
    nonempty: set[tuple[str, int]] = set()
    for line in pred_path.read_text().splitlines():
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
        if rec.get("model_patch"):
            nonempty.add((rec["instance_id"], pid))

    resolved_per_pid: dict[int, set[str]] = {}
    sources_per_pid: dict[int, list[str]] = {}
    for pid in range(n_patches):
        ids, srcs = _resolved_for(eval_dir, key, pid)
        resolved_per_pid[pid] = ids
        sources_per_pid[pid] = srcs

    by_repo_total: dict[str, int] = {}
    by_repo_pass: dict[str, int] = {}
    n_attempted = 0
    n_nonempty = 0
    n_correct = 0
    per_instance_pass_rate: list[float] = []

    for inst in instances:
        inst_id = inst["instance_id"]
        repo = inst["repo"]
        passes = 0
        for pid in range(n_patches):
            n_attempted += 1
            by_repo_total[repo] = by_repo_total.get(repo, 0) + 1
            ok = (
                inst_id in resolved_per_pid[pid]
                and (inst_id, pid) in nonempty
            )
            if (inst_id, pid) in nonempty:
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

    summary = GeneratorSummary(
        generator_key=key,
        generator_model=model,
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
    aux = {
        "sources_per_pid": sources_per_pid,
        "resolved_ids_per_pid": {
            str(p): sorted(resolved_per_pid[p]) for p in resolved_per_pid
        },
    }
    return summary, aux


# Same generator -> model mapping as scripts/spot_check_generators.GENERATORS
GEN_MODELS = {
    "qwen25_7b": "qwen/qwen-2.5-7b-instruct",
    "qwen3_8b": "qwen/qwen3-8b",
    "qwen3_8b_thinking": "qwen/qwen3-8b (thinking=on)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/spot_check")
    ap.add_argument("--n-patches", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-instances", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--generators", default="qwen25_7b,qwen3_8b,qwen3_8b_thinking")
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir).resolve()
    eval_dir = data_dir / "eval"
    if not eval_dir.exists():
        raise SystemExit(f"eval dir not found: {eval_dir}")

    instances = sample_instances(args.seed, args.n_instances)
    log.info("sampled %d instances", len(instances))

    summaries: list[GeneratorSummary] = []
    aux_payloads: dict[str, dict] = {}
    for key in [g.strip() for g in args.generators.split(",") if g.strip()]:
        gen_dir = data_dir / key
        pred_path = gen_dir / "predictions.jsonl"
        if not pred_path.exists():
            log.warning("[%s] predictions.jsonl missing; skipping", key)
            continue
        model = GEN_MODELS.get(key, key)
        summary, aux = aggregate(
            key=key, model=model, instances=instances,
            pred_path=pred_path, eval_dir=eval_dir, n_patches=args.n_patches,
        )
        summaries.append(summary)
        aux_payloads[key] = aux
        log.info(
            "[%s] resolved/pid=%s",
            key, {p: len(aux["resolved_ids_per_pid"][str(p)]) for p in range(args.n_patches)},
        )

    payload = {
        "n_instances": args.n_instances,
        "n_patches": args.n_patches,
        "temperature": args.temperature,
        "seed": args.seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "indent_fix_merged": True,
        "generators": [dataclasses.asdict(s) for s in summaries],
        "report_sources_per_pid": {k: v["sources_per_pid"] for k, v in aux_payloads.items()},
    }
    (data_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    print()
    print("=" * 78)
    print("SPOT-CHECK SUMMARY  (post-indent-fix; deltas merged)")
    print("=" * 78)
    print(f"sample: n_instances={args.n_instances} n_patches={args.n_patches} seed={args.seed}")
    print(f"{'generator':<22} {'attempted':>9} {'nonempty':>8} {'pass':>5} "
          f"{'rate':>7} {'inst-rate':>10}")
    for s in summaries:
        print(
            f"{s.generator_key:<22} {s.n_patches_attempted:>9} {s.n_patches_nonempty:>8} "
            f"{s.n_correct:>5} {s.base_rate:>7.2%} {s.base_rate_per_instance:>10.2%}"
        )
    print("=" * 78)
    print("regime check (PRE_REGISTRATION.md S4):  base_rate ∈ [0.30, 0.70]")
    for s in summaries:
        in_window = 0.30 <= s.base_rate <= 0.70
        verdict = "IN  [0.30,0.70]" if in_window else "OUT [0.30,0.70]"
        print(f"  {s.generator_key:<22} base_rate={s.base_rate:.3f}  -> {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
