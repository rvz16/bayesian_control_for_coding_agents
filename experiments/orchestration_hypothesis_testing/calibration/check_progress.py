#!/usr/bin/env python3
"""Quick progress checker for calibration runs."""
import json
import sys
from pathlib import Path
from collections import Counter

data_file = Path(__file__).parent / "data" / "raw_results.jsonl"
if not data_file.exists():
    print("No results yet")
    sys.exit(0)

results = [json.loads(l) for l in open(data_file) if l.strip()]
n = len(results)
if n == 0:
    print("No results yet")
    sys.exit(0)

instances = set(r["instance_id"] for r in results)
y1 = sum(1 for r in results if r["ground_truth"] == 1)
l0_pass = sum(1 for r in results if r["critic_results"]["L0_syntax"]["passed"])
l1_pass = sum(1 for r in results if r["critic_results"]["L1_lint"]["passed"])
l2_pass = sum(1 for r in results if r["critic_results"]["L2_fast_test"]["passed"])

print(f"Patches: {n} across {len(instances)} instances")
print(f"Y=1: {y1}/{n} ({100*y1/n:.0f}%)")
print(f"L0 pass: {l0_pass}/{n} ({100*l0_pass/n:.0f}%)")
print(f"L1 pass: {l1_pass}/{n} ({100*l1_pass/n:.0f}%)")
print(f"L2 pass: {l2_pass}/{n} ({100*l2_pass/n:.0f}%)")

# Per-instance summary
print(f"\nPer instance:")
by_instance = {}
for r in results:
    iid = r["instance_id"]
    if iid not in by_instance:
        by_instance[iid] = {"patches": 0, "y1": 0, "l0": 0, "l1": 0, "l2": 0}
    by_instance[iid]["patches"] += 1
    by_instance[iid]["y1"] += r["ground_truth"]
    by_instance[iid]["l0"] += r["critic_results"]["L0_syntax"]["passed"]
    by_instance[iid]["l1"] += r["critic_results"]["L1_lint"]["passed"]
    by_instance[iid]["l2"] += r["critic_results"]["L2_fast_test"]["passed"]

for iid, s in by_instance.items():
    y_str = f"Y1={s['y1']}/{s['patches']}"
    print(f"  {iid:45s} {y_str:10s} L0={s['l0']} L1={s['l1']} L2={s['l2']}")
