#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[test,dev]"
echo "Environment ready. Activate with: source .venv/bin/activate"
