#!/usr/bin/env bash
set -euo pipefail
echo "=== Running all benchmarks ==="
python -m abbo.cli run-synthetic-exact
python -m abbo.cli run-synthetic --episodes 100000 --seed 7
python -m abbo.cli run-toy --episodes 10000 --seed 7
echo ""
echo "=== Generating reports ==="
python -m abbo.cli summarize-results --benchmark synthetic
python -m abbo.cli summarize-results --benchmark toy
echo "Done. Check results/ directory."
