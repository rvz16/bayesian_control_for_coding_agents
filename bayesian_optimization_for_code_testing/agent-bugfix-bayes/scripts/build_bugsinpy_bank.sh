#!/usr/bin/env bash
set -euo pipefail
python -m abbo.cli build-candidate-bank --benchmark bugsinpy --config configs/bugsinpy_pilot.yaml
