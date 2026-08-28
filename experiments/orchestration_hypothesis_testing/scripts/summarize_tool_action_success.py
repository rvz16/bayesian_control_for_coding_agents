#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


CRITIC_ACTIONS = {
    "critic_L0",
    "critic_L1",
    "critic_L2",
    "critic_L3",
    "L0",
    "L1",
    "L2",
    "L3",
    "L0_syntax",
    "L1_lint",
    "L2_public_tests",
    "L3_llm_review",
}
VERIFY_ACTIONS = {"verify", "final_verify", "label_verifier"}
NON_FINAL_VERIFY_ACTIONS = VERIFY_ACTIONS - {"final_verify"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"skip malformed line {line_no}: {exc}")
    return rows


def latest_by_instance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest = {}
    for row in rows:
        iid = str(row.get("instance_id"))
        if iid and iid != "None":
            latest[iid] = row
    return list(latest.values())


def is_success(action: dict[str, Any]) -> bool:
    value = action.get("passed")
    if value is None:
        value = action.get("ok")
    return value is True


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    counts = Counter()
    success = Counter()
    skipped = Counter()
    counts_before_verify_pass = Counter()
    success_before_verify_pass = Counter()
    skipped_before_verify_pass = Counter()
    saw_verify_pass = False

    for action in row.get("trajectory") or []:
        name = str(action.get("action"))
        if name in CRITIC_ACTIONS:
            group = "critic"
        elif name in VERIFY_ACTIONS:
            group = "verifier"
        else:
            continue

        counts[group] += 1
        counts[f"{group}:{name}"] += 1
        if name in NON_FINAL_VERIFY_ACTIONS:
            counts["verifier_no_final"] += 1
        if action.get("skipped") is True:
            skipped[group] += 1
            skipped[f"{group}:{name}"] += 1
            if name in NON_FINAL_VERIFY_ACTIONS:
                skipped["verifier_no_final"] += 1
        if is_success(action):
            success[group] += 1
            success[f"{group}:{name}"] += 1
            if name in NON_FINAL_VERIFY_ACTIONS:
                success["verifier_no_final"] += 1

        if not saw_verify_pass:
            if name in NON_FINAL_VERIFY_ACTIONS and is_success(action):
                saw_verify_pass = True
                continue
            if name in CRITIC_ACTIONS:
                pre_group = "critic"
            elif name in NON_FINAL_VERIFY_ACTIONS:
                pre_group = "verifier_no_final"
            else:
                continue
            counts_before_verify_pass[pre_group] += 1
            if action.get("skipped") is True:
                skipped_before_verify_pass[pre_group] += 1
            if is_success(action):
                success_before_verify_pass[pre_group] += 1

    total = counts["critic"] + counts["verifier"]
    ok_total = success["critic"] + success["verifier"]
    total_no_final = counts["critic"] + counts["verifier_no_final"]
    ok_total_no_final = success["critic"] + success["verifier_no_final"]
    total_no_final_before_verify_pass = (
        counts_before_verify_pass["critic"]
        + counts_before_verify_pass["verifier_no_final"]
    )
    ok_total_no_final_before_verify_pass = (
        success_before_verify_pass["critic"]
        + success_before_verify_pass["verifier_no_final"]
    )
    return {
        "instance_id": row.get("instance_id"),
        "quality": int(bool(row.get("fixed"))),
        "final_action": row.get("final_action", ""),
        "n_steps": int(row.get("n_steps") or len(row.get("trajectory") or [])),
        "critic_n": counts["critic"],
        "critic_ok": success["critic"],
        "critic_success_rate": success["critic"] / counts["critic"] if counts["critic"] else None,
        "verifier_n": counts["verifier"],
        "verifier_ok": success["verifier"],
        "verifier_success_rate": success["verifier"] / counts["verifier"] if counts["verifier"] else None,
        "verifier_no_final_n": counts["verifier_no_final"],
        "verifier_no_final_ok": success["verifier_no_final"],
        "verifier_no_final_success_rate": (
            success["verifier_no_final"] / counts["verifier_no_final"]
            if counts["verifier_no_final"]
            else None
        ),
        "tool_n": total,
        "tool_ok": ok_total,
        "tool_success_rate": ok_total / total if total else None,
        "tool_no_final_n": total_no_final,
        "tool_no_final_ok": ok_total_no_final,
        "tool_no_final_success_rate": (
            ok_total_no_final / total_no_final if total_no_final else None
        ),
        "tool_no_final_before_verify_pass_n": total_no_final_before_verify_pass,
        "tool_no_final_before_verify_pass_ok": ok_total_no_final_before_verify_pass,
        "tool_no_final_before_verify_pass_success_rate": (
            ok_total_no_final_before_verify_pass / total_no_final_before_verify_pass
            if total_no_final_before_verify_pass
            else None
        ),
        "verifier_no_final_before_verify_pass_n": counts_before_verify_pass[
            "verifier_no_final"
        ],
        "verifier_no_final_before_verify_pass_ok": success_before_verify_pass[
            "verifier_no_final"
        ],
        "critic_success_per_all": success["critic"] / total if total else None,
        "verifier_success_per_all": success["verifier"] / total if total else None,
        "verifier_no_final_success_per_all": (
            success["verifier_no_final"] / total_no_final if total_no_final else None
        ),
        "critic_skipped": skipped["critic"],
        "verifier_skipped": skipped["verifier"],
        "verifier_no_final_skipped": skipped["verifier_no_final"],
        "tool_no_final_before_verify_pass_skipped": (
            skipped_before_verify_pass["critic"]
            + skipped_before_verify_pass["verifier_no_final"]
        ),
        "critic_L0_n": counts["critic:critic_L0"] + counts["critic:L0"] + counts["critic:L0_syntax"],
        "critic_L0_ok": success["critic:critic_L0"] + success["critic:L0"] + success["critic:L0_syntax"],
        "critic_L1_n": counts["critic:critic_L1"] + counts["critic:L1"] + counts["critic:L1_lint"],
        "critic_L1_ok": success["critic:critic_L1"] + success["critic:L1"] + success["critic:L1_lint"],
        "critic_L2_n": counts["critic:critic_L2"] + counts["critic:L2"] + counts["critic:L2_public_tests"],
        "critic_L2_ok": success["critic:critic_L2"] + success["critic:L2"] + success["critic:L2_public_tests"],
        "critic_L3_n": counts["critic:critic_L3"] + counts["critic:L3"] + counts["critic:L3_llm_review"],
        "critic_L3_ok": success["critic:critic_L3"] + success["critic:L3"] + success["critic:L3_llm_review"],
        "verify_n": counts["verifier:verify"],
        "verify_ok": success["verifier:verify"],
        "final_verify_n": counts["verifier:final_verify"],
        "final_verify_ok": success["verifier:final_verify"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_jsonl", type=Path)
    p.add_argument("--keep-duplicates", action="store_true")
    p.add_argument("--per-instance-csv", type=Path, default=None)
    args = p.parse_args()

    rows = read_jsonl(args.results_jsonl)
    if not args.keep_duplicates:
        rows = latest_by_instance(rows)

    per_instance = [summarize_row(row) for row in rows]
    if args.per_instance_csv:
        args.per_instance_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.per_instance_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_instance[0].keys()))
            writer.writeheader()
            writer.writerows(per_instance)

    counts = Counter()
    success = Counter()
    skipped = Counter()
    counts_before_verify_pass = Counter()
    success_before_verify_pass = Counter()
    skipped_before_verify_pass = Counter()

    for row in rows:
        saw_verify_pass = False
        for action in row.get("trajectory") or []:
            name = str(action.get("action"))
            if name in CRITIC_ACTIONS:
                group = "critic"
            elif name in VERIFY_ACTIONS:
                group = "verifier"
            else:
                continue

            counts[group] += 1
            counts[f"{group}:{name}"] += 1
            if name in NON_FINAL_VERIFY_ACTIONS:
                counts["verifier_no_final"] += 1
            if action.get("skipped") is True:
                skipped[group] += 1
                skipped[f"{group}:{name}"] += 1
                if name in NON_FINAL_VERIFY_ACTIONS:
                    skipped["verifier_no_final"] += 1
            if is_success(action):
                success[group] += 1
                success[f"{group}:{name}"] += 1
                if name in NON_FINAL_VERIFY_ACTIONS:
                    success["verifier_no_final"] += 1

            if not saw_verify_pass:
                if name in NON_FINAL_VERIFY_ACTIONS and is_success(action):
                    saw_verify_pass = True
                    continue
                if name in CRITIC_ACTIONS:
                    pre_group = "critic"
                elif name in NON_FINAL_VERIFY_ACTIONS:
                    pre_group = "verifier_no_final"
                else:
                    continue
                counts_before_verify_pass[pre_group] += 1
                if action.get("skipped") is True:
                    skipped_before_verify_pass[pre_group] += 1
                if is_success(action):
                    success_before_verify_pass[pre_group] += 1

    total = counts["critic"] + counts["verifier"]
    total_success = success["critic"] + success["verifier"]
    total_no_final = counts["critic"] + counts["verifier_no_final"]
    total_success_no_final = success["critic"] + success["verifier_no_final"]
    total_no_final_before_verify_pass = (
        counts_before_verify_pass["critic"]
        + counts_before_verify_pass["verifier_no_final"]
    )
    total_success_no_final_before_verify_pass = (
        success_before_verify_pass["critic"]
        + success_before_verify_pass["verifier_no_final"]
    )

    print(f"instances: {len(rows)}")
    print()
    for group in ["critic", "verifier"]:
        n = counts[group]
        ok = success[group]
        skip = skipped[group]
        rate = ok / n if n else 0.0
        print(
            f"{group:8s} {ok:4d}/{n:<4d} success_rate={rate:.3f} "
            f"success_per_all={(ok / total if total else 0.0):.3f} "
            f"attempt_share={(n / total if total else 0.0):.3f} skipped={skip}"
        )
    n = counts["verifier_no_final"]
    ok = success["verifier_no_final"]
    skip = skipped["verifier_no_final"]
    print(
        f"{'verifier_no_final':8s} {ok:4d}/{n:<4d} "
        f"success_rate={(ok / n if n else 0.0):.3f} "
        f"success_per_all={(ok / total_no_final if total_no_final else 0.0):.3f} "
        f"attempt_share={(n / total_no_final if total_no_final else 0.0):.3f} "
        f"skipped={skip}"
    )
    print(f"{'overall':8s} {total_success:4d}/{total:<4d} success_rate={(total_success / total if total else 0.0):.3f}")
    print(
        f"{'overall_no_final':8s} {total_success_no_final:4d}/{total_no_final:<4d} "
        f"success_rate={(total_success_no_final / total_no_final if total_no_final else 0.0):.3f}"
    )
    print(
        f"{'overall_no_final_before_verify_pass':8s} "
        f"{total_success_no_final_before_verify_pass:4d}/"
        f"{total_no_final_before_verify_pass:<4d} "
        f"success_rate={(total_success_no_final_before_verify_pass / total_no_final_before_verify_pass if total_no_final_before_verify_pass else 0.0):.3f} "
        f"skipped={skipped_before_verify_pass['critic'] + skipped_before_verify_pass['verifier_no_final']}"
    )

    print("\nby action:")
    for key in sorted(k for k in counts if ":" in k):
        n = counts[key]
        ok = success[key]
        skip = skipped[key]
        print(f"{key:26s} {ok:4d}/{n:<4d} success_rate={(ok / n if n else 0.0):.3f} skipped={skip}")


if __name__ == "__main__":
    main()
