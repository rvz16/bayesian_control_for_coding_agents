#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end launcher:
#   1) set env/cache
#   2) run missing calibrations
#   3) run live GrFt/DPFt
#
# Common use:
#   bash scripts/run_all_fitted_live.sh
#   TASKS=mbpp,humaneval bash scripts/run_all_fitted_live.sh
#   MODE=live bash scripts/run_all_fitted_live.sh
#   CALIB_N=100 bash scripts/run_all_fitted_live.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
cd "${EXP_DIR}"

for env_file in "${REPO_ROOT}/.env" "${EXP_DIR}/.env"; do
  if [[ -s "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
done

# Caches and tokens.
export AWU_CACHE_ROOT="${AWU_CACHE_ROOT:-${HOME}/.cache/agents_with_uncertainty_research}"
export HF_HOME="${HF_HOME:-${AWU_CACHE_ROOT}/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${AWU_CACHE_ROOT}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${AWU_CACHE_ROOT}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${AWU_CACHE_ROOT}/hub}"
export TORCH_HOME="${TORCH_HOME:-${AWU_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${AWU_CACHE_ROOT}/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${AWU_CACHE_ROOT}/pip}"
export TMPDIR="${TMPDIR:-${AWU_CACHE_ROOT}/tmp}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}"

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-${SAGE_OPENROUTER_API_KEY:-${OPEN_ROUTER_API_KEY:-${OPEN_ROUTER:-}}}}"
export SAGE_OPENROUTER_API_KEY="${SAGE_OPENROUTER_API_KEY:-${OPENROUTER_API_KEY:-}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is missing. Put it in ${REPO_ROOT}/.env or export it." >&2
  exit 2
fi

if [[ "${USE_PODMAN:-0}" == "1" ]]; then
  export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/podman/podman.sock}"
  export SWEBENCH_PODMAN_COMPAT=1
fi

PYTHON="${PYTHON:-python3}"
MODE="${MODE:-all}"             # all | calibrate | live
TASKS="${TASKS:-lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval,swebench_lite,swebench_verified}"
POLICIES="${POLICIES:-greedy_fitted,dp_fitted}"
FORCE="${FORCE:-0}"
SKIP_SWE="${SKIP_SWE:-0}"
CALIB_N="${CALIB_N:-100}"

case "${MODE}" in
  all|calibrate|live) ;;
  *) echo "MODE must be one of: all, calibrate, live" >&2; exit 2 ;;
esac

GEN_CAPS="${GEN_CAPS:-qwen3_coder=4.0,haiku45=4.0,sonnet45=15.0,gpt5_mini=4.0}"
LIVE_CAPS="${LIVE_CAPS:-qwen3_coder=20.0,haiku45=20.0,sonnet45=50.0,gpt5_mini=20.0}"
SWE_WORKERS="${SWE_WORKERS:-1}"

normalize_gens() {
  local raw="${1// /}" out="" item key
  IFS=',' read -ra items <<< "${raw}"
  for item in "${items[@]}"; do
    case "${item}" in
      qwen/qwen3-coder) key="qwen3_coder" ;;
      anthropic/claude-haiku-4.5) key="haiku45" ;;
      anthropic/claude-sonnet-4.5) key="sonnet45" ;;
      openai/gpt-5-mini) key="gpt5_mini" ;;
      *) key="${item}" ;;
    esac
    [[ -n "${key}" ]] && out="${out:+${out},}${key}"
  done
  echo "${out}"
}

GENS="$(normalize_gens "${GENS:-qwen3_coder,haiku45,sonnet45,gpt5_mini}")"

run() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

tables_exist() {
  local dir="$1" gen
  IFS=',' read -ra gens <<< "${GENS}"
  for gen in "${gens[@]}"; do
    [[ -f "${dir}/${gen}/likelihood_tables.json" ]] || return 1
  done
}

calibrate() {
  local task="$1"
  case "${task}" in
    lcb_hard)
      tables_exist "data/lcb_calibration_v2_n${CALIB_N}" && [[ "${FORCE}" != "1" ]] && return
      run "${PYTHON}" scripts/lcb_calibrate.py --output-dir "data/lcb_calibration_v2_n${CALIB_N}" \
        --difficulty hard --generators "${GENS}" --n-instances "${CALIB_N}" --n-patches 3 \
        --lcb-version all --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
    lcb_medium|lcb_med)
      tables_exist "data/lcb_calibration_medium_n${CALIB_N}" && [[ "${FORCE}" != "1" ]] && return
      run "${PYTHON}" scripts/lcb_calibrate.py --output-dir "data/lcb_calibration_medium_n${CALIB_N}" \
        --difficulty medium --generators "${GENS}" --n-instances "${CALIB_N}" --n-patches 3 \
        --lcb-version all --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
    lcb_easy)
      tables_exist "data/lcb_calibration_easy_n${CALIB_N}" && [[ "${FORCE}" != "1" ]] && return
      run "${PYTHON}" scripts/lcb_calibrate.py --output-dir "data/lcb_calibration_easy_n${CALIB_N}" \
        --difficulty easy --generators "${GENS}" --n-instances "${CALIB_N}" --n-patches 3 \
        --lcb-version all --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
    mbpp)
      tables_exist "data/mbpp_calibration_n${CALIB_N}" && [[ "${FORCE}" != "1" ]] && return
      run "${PYTHON}" scripts/mbpp_calibrate.py --output-dir "data/mbpp_calibration_n${CALIB_N}" \
        --generators "${GENS}" --n-instances "${CALIB_N}" --n-patches 3 \
        --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
    humaneval)
      tables_exist "data/humaneval_calibration_n${CALIB_N}" && [[ "${FORCE}" != "1" ]] && return
      run "${PYTHON}" scripts/humaneval_calibrate.py --output-dir "data/humaneval_calibration_n${CALIB_N}" \
        --generators "${GENS}" --n-instances "${CALIB_N}" --n-patches 3 --plus-input-cap 200 \
        --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
    swebench_lite)
      [[ "${SKIP_SWE}" == "1" ]] && return
      tables_exist "data/swebench_lite_n${CALIB_N}" && [[ "${FORCE}" != "1" ]] && return
      run "${PYTHON}" scripts/spot_check_generators.py --output-dir "data/swebench_lite_n${CALIB_N}" \
        --dataset princeton-nlp/SWE-bench_Lite --generators "${GENS}" \
        --n-instances "${CALIB_N}" --n-patches 3 --max-workers-gen 1 \
        --max-workers-eval "${SWE_WORKERS}" --max-cost-usd-per-model "${GEN_CAPS}"
      run "${PYTHON}" scripts/calibrate_from_spotcheck.py --output-dir "data/swebench_lite_n${CALIB_N}" \
        --dataset princeton-nlp/SWE-bench_Lite --generators "${GENS}"
      ;;
    swebench_verified)
      [[ "${SKIP_SWE}" == "1" ]] && return
      tables_exist "data/swebench_verified_n${CALIB_N}" && [[ "${FORCE}" != "1" ]] && return
      run "${PYTHON}" scripts/spot_check_generators.py --output-dir "data/swebench_verified_n${CALIB_N}" \
        --dataset princeton-nlp/SWE-bench_Verified --generators "${GENS}" \
        --n-instances "${CALIB_N}" --n-patches 3 --max-workers-gen 1 \
        --max-workers-eval "${SWE_WORKERS}" --max-cost-usd-per-model "${GEN_CAPS}"
      run "${PYTHON}" scripts/calibrate_from_spotcheck.py --output-dir "data/swebench_verified_n${CALIB_N}" \
        --dataset princeton-nlp/SWE-bench_Verified --generators "${GENS}"
      ;;
  esac
}

live() {
  local task="$1" n="" calib_dir="" extra=() kernel=()
  case "${task}" in
    lcb_hard) n=102; calib_dir="data/lcb_calibration_v2_n${CALIB_N}"; extra=(--lcb-version all --lcb-private-test-cap 0)
      [[ -d data/lcb_calibration_v2_iter ]] && kernel=(--kernel-dir data/lcb_calibration_v2_iter) ;;
    lcb_medium|lcb_med) task=lcb_medium; n=207; calib_dir="data/lcb_calibration_medium_n${CALIB_N}"; extra=(--lcb-version all --lcb-private-test-cap 0)
      [[ -d data/lcb_calibration_medium_iter ]] && kernel=(--kernel-dir data/lcb_calibration_medium_iter) ;;
    lcb_easy) n=135; calib_dir="data/lcb_calibration_easy_n${CALIB_N}"; extra=(--lcb-version all --lcb-private-test-cap 0)
      [[ -d data/lcb_calibration_easy_iter ]] && kernel=(--kernel-dir data/lcb_calibration_easy_iter) ;;
    mbpp) n=378; calib_dir="data/mbpp_calibration_n${CALIB_N}" ;;
    humaneval) n=164; calib_dir="data/humaneval_calibration_n${CALIB_N}"; extra=(--plus-input-cap 200) ;;
    swebench_lite) [[ "${SKIP_SWE}" == "1" ]] && return; n=300; calib_dir="data/swebench_lite_n${CALIB_N}"
      [[ -d data/swebench_lite_iter ]] && kernel=(--kernel-dir data/swebench_lite_iter) ;;
    swebench_verified) [[ "${SKIP_SWE}" == "1" ]] && return; n=500; calib_dir="data/swebench_verified_n${CALIB_N}"
      [[ -d data/swebench_verified_iter ]] && kernel=(--kernel-dir data/swebench_verified_iter) ;;
    *) echo "Unknown task: ${task}" >&2; exit 2 ;;
  esac

  run "${PYTHON}" scripts/run_fitted_live.py --benchmark "${task}" \
    --generators "${GENS}" --policies "${POLICIES}" --calibration-dir "${calib_dir}" \
    --n-instances "${n}" \
    --max-generations 3 --max-actions 12 --max-api-cost-usd-per-model "${LIVE_CAPS}" \
    --swe-harness-workers "${SWE_WORKERS}" "${kernel[@]}" "${extra[@]}"
}

echo "Tasks: ${TASKS}"
echo "Generators: ${GENS}"
echo "Mode: ${MODE}"
echo "Calibration N: ${CALIB_N}"
echo "HF_HOME=${HF_HOME}"
echo "TMPDIR=${TMPDIR}"

IFS=',' read -ra task_list <<< "${TASKS}"
for task in "${task_list[@]}"; do
  task="${task// /}"
  [[ -z "${task}" ]] && continue
  [[ "${MODE}" == "all" || "${MODE}" == "calibrate" ]] && calibrate "${task}"
  [[ "${MODE}" == "all" || "${MODE}" == "live" ]] && live "${task}"
done

echo
echo "Done: ${EXP_DIR}/data/fitted_live"
