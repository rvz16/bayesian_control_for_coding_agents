#!/usr/bin/env bash
set -euo pipefail
python -m abbo.cli run-benchmark --benchmark bugsinpy --policy all
python -m abbo.cli summarize-results --benchmark bugsinpy
