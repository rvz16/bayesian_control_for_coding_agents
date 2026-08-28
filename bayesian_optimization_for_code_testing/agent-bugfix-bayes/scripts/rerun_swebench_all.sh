#!/usr/bin/env bash
# Sweep all (dataset × model) combinations for SWE-Bench.
# Output JSONs go to:
#   sim_results/swebench_full_endtoend__<dataset>__<model-slug>.json
# Resume-safe: re-running skips already-completed (instance, variant) pairs.
#
# Usage:
#   bash scripts/rerun_swebench_all.sh             # all datasets × all models
#   DATASETS="verified" bash scripts/rerun_swebench_all.sh
#   MODELS="openai/gpt-5-mini anthropic/claude-haiku-4.5" bash scripts/rerun_swebench_all.sh
#
# Optional env overrides:
#   DATASETS   space-separated list, e.g. "lite verified"
#   MODELS     space-separated list of OpenRouter model ids
#   STOP_ON_FAIL=1   abort the sweep when any run exits non-zero
#                    (default: continue, log the failure)

set -uo pipefail

DATASETS="${DATASETS:-lite verified}"
MODELS="${MODELS:-openai/gpt-5-mini anthropic/claude-haiku-4.5 anthropic/claude-sonnet-4.5 qwen/qwen3-coder openai/gpt-oss-20b:free}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RUNNER="${SCRIPT_DIR}/run_swebench_full.py"

if [[ ! -f "${RUNNER}" ]]; then
    echo "ERROR: runner not found at ${RUNNER}" >&2
    exit 1
fi

log() {
    printf '[%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$*"
}

fails=()
total=0
done=0

for ds in ${DATASETS}; do
    for model in ${MODELS}; do
        total=$((total + 1))
    done
done

log "Sweep plan: ${total} runs (datasets=${DATASETS}; models=${MODELS})"

for ds in ${DATASETS}; do
    for model in ${MODELS}; do
        done=$((done + 1))
        log "[${done}/${total}] dataset=${ds}  model=${model}"
        if python3 "${RUNNER}" --dataset "${ds}" --model "${model}"; then
            log "  OK: ${ds} × ${model}"
        else
            ec=$?
            log "  FAIL (exit ${ec}): ${ds} × ${model}"
            fails+=("${ds}/${model}")
            if [[ "${STOP_ON_FAIL}" == "1" ]]; then
                log "STOP_ON_FAIL=1 set; aborting sweep."
                break 2
            fi
        fi
    done
done

log "Sweep complete: ${done}/${total} runs attempted."
if (( ${#fails[@]} > 0 )); then
    log "Failed runs (${#fails[@]}):"
    for f in "${fails[@]}"; do
        log "  - ${f}"
    done
    exit 1
fi
