#!/usr/bin/env bash
set -euo pipefail
echo "=== Exact synthetic benchmark ==="
python -m abbo.cli run-synthetic-exact
echo ""
echo "=== Monte Carlo synthetic benchmark ==="
python -m abbo.cli run-synthetic --episodes 100000 --seed 7
