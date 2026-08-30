#!/usr/bin/env bash
set -euo pipefail

# Run calibration -> transition kernel -> Bayesian DP confidence collection
# for MBPP, HumanEval, and all LCB difficulty slices.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
GENERATORS="${GENERATORS:-deepseek_v4_pro}"
BENCHMARKS="${BENCHMARKS:-mbpp,humaneval,lcb_easy,lcb_medium,lcb_hard}"

N_INSTANCES="${N_INSTANCES:-50}"
N_PATCHES="${N_PATCHES:-3}"
SEED="${SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
N_TRAIN="${N_TRAIN:-}"

MAX_COST_CALIB="${MAX_COST_CALIB:-10.0}"
MAX_COST_COLLECT="${MAX_COST_COLLECT:-10.0}"

OUTPUT_ROOT="${OUTPUT_ROOT:-data/final_confidence_bayes_quality_all_logprobs}"
LOG_DIR="${LOG_DIR:-logs/final_confidence_pipeline}"

LCB_PLATFORM="${LCB_PLATFORM:-leetcode}"
LCB_VERSION="${LCB_VERSION:-all}"
LCB_EXTEND_EXISTING="${LCB_EXTEND_EXISTING:-0}"
HUMANEVAL_PLUS_INPUT_CAP="${HUMANEVAL_PLUS_INPUT_CAP:-200}"

C_GEN="${C_GEN:-5}"
C_L0="${C_L0:-1}"
C_L1="${C_L1:-1}"
C_L2="${C_L2:-2}"
C_L3="${C_L3:-5}"
C_VER="${C_VER:-30}"
REWARD="${REWARD:-100}"

MAX_GENERATIONS="${MAX_GENERATIONS:-5}"
MAX_VERIFICATIONS="${MAX_VERIFICATIONS:-3}"
MAX_ACTIONS="${MAX_ACTIONS:-24}"

OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-deepseek,fireworks}"
OPENROUTER_PROVIDER_ORDER="${OPENROUTER_PROVIDER_ORDER:-deepseek,fireworks}"
LOGPROB_RETRY_ATTEMPTS="${LOGPROB_RETRY_ATTEMPTS:-10}"
LOGPROB_RETRY_SLEEP="${LOGPROB_RETRY_SLEEP:-15}"

export HF_HOME="${HF_HOME:-/users/avazhentsev/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$LOG_DIR"

LOG_FILE="${LOG_DIR}/run_all_confidence_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "ERROR: OPENROUTER_API_KEY is not set."
  exit 1
fi

echo "log_file=${LOG_FILE}"
echo "generators=${GENERATORS}"
echo "benchmarks=${BENCHMARKS}"
echo "n_instances=${N_INSTANCES} n_patches=${N_PATCHES}"
echo "split: n_train=${N_TRAIN:-<unset>} train_fraction=${TRAIN_FRACTION} split_seed=${SPLIT_SEED}"
echo "costs: c_gen=${C_GEN} c_L0=${C_L0} c_L1=${C_L1} c_L2=${C_L2} c_L3=${C_L3} c_ver=${C_VER} reward=${REWARD}"
echo "hf_home=${HF_HOME}"

run() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

src_dir_for_benchmark() {
  case "$1" in
    mbpp)
      echo "data/mbpp_full"
      ;;
    humaneval)
      echo "data/humaneval_full"
      ;;
    lcb_easy|lcb_medium|lcb_hard)
      echo "data/${1}_full"
      ;;
    *)
      echo "unknown benchmark: $1" >&2
      return 1
      ;;
  esac
}

calibrate_benchmark() {
  local benchmark="$1"
  local src_dir="$2"

  case "$benchmark" in
    mbpp)
      run "$PYTHON_BIN" scripts/mbpp_calibrate.py \
        --output-dir "$src_dir" \
        --generators "$GENERATORS" \
        --sample-ids-file "$src_dir/train_sample.json" \
        --n-patches "$N_PATCHES" \
        --seed "$SEED" \
        --max-cost-usd-per-model "$MAX_COST_CALIB"
      ;;
    humaneval)
      run "$PYTHON_BIN" scripts/humaneval_calibrate.py \
        --output-dir "$src_dir" \
        --generators "$GENERATORS" \
        --sample-ids-file "$src_dir/train_sample.json" \
        --n-patches "$N_PATCHES" \
        --plus-input-cap "$HUMANEVAL_PLUS_INPUT_CAP" \
        --seed "$SEED" \
        --max-cost-usd-per-model "$MAX_COST_CALIB"
      ;;
    lcb_easy|lcb_medium|lcb_hard)
      local difficulty="${benchmark#lcb_}"
      local args=(
        "$PYTHON_BIN" scripts/lcb_calibrate.py
        --output-dir "$src_dir"
        --generators "$GENERATORS"
        --sample-ids-file "$src_dir/train_sample.json"
        --n-patches "$N_PATCHES"
        --difficulty "$difficulty"
        --platform "$LCB_PLATFORM"
        --lcb-version "$LCB_VERSION"
        --seed "$SEED"
        --max-cost-usd-per-model "$MAX_COST_CALIB"
      )
      if [[ "$LCB_EXTEND_EXISTING" == "1" ]]; then
        args+=(--extend-existing)
      fi
      run "${args[@]}"
      ;;
  esac
}

prepare_split_benchmark() {
  local benchmark="$1"
  local src_dir="$2"
  local args=(
    "$PYTHON_BIN" scripts/synthesis_prepare_train_test_split.py
    --src-dir "$src_dir"
    --benchmark "$benchmark"
    --generators "$GENERATORS"
    --n-instances "$N_INSTANCES"
    --sample-seed "$SEED"
    --split-seed "$SPLIT_SEED"
  )
  if [[ -n "$N_TRAIN" ]]; then
    args+=(--n-train "$N_TRAIN")
  else
    args+=(--train-fraction "$TRAIN_FRACTION")
  fi
  run "${args[@]}"
}

build_transition_kernel() {
  local benchmark="$1"
  local src_dir="$2"

  run "$PYTHON_BIN" scripts/synthesis_transition_kernel.py \
    --src-dir "$src_dir" \
    --generators "$GENERATORS" \
    --benchmark "$benchmark" \
    --split-scope train
}

collect_confidence_rows() {
  local benchmark="$1"
  local src_dir="$2"

  run "$PYTHON_BIN" scripts/collect_final_confidence_bayes_quality.py \
    --src-dir "$src_dir" \
    --benchmark "$benchmark" \
    --generators "$GENERATORS" \
    --policies bayesian_DP \
    --critic-allowlist L0,L1,L2,L3 \
    --require-split \
    --n-instances "$N_INSTANCES" \
    --c-gen "$C_GEN" \
    --c-l0 "$C_L0" \
    --c-l1 "$C_L1" \
    --c-l2 "$C_L2" \
    --c-l3 "$C_L3" \
    --c-ver "$C_VER" \
    --reward "$REWARD" \
    --max-generations "$MAX_GENERATIONS" \
    --max-verifications "$MAX_VERIFICATIONS" \
    --max-actions "$MAX_ACTIONS" \
    --max-api-cost-usd-per-model "$MAX_COST_COLLECT" \
    --openrouter-provider-only "$OPENROUTER_PROVIDER_ONLY" \
    --openrouter-provider-order "$OPENROUTER_PROVIDER_ORDER" \
    --openrouter-require-parameters \
    --openrouter-no-fallbacks \
    --require-logprobs-in-response \
    --no-logprob-fallback \
    --logprob-retry-attempts "$LOGPROB_RETRY_ATTEMPTS" \
    --logprob-retry-sleep "$LOGPROB_RETRY_SLEEP" \
    --output-dir "$OUTPUT_ROOT"
}

IFS=',' read -r -a benchmark_array <<< "$BENCHMARKS"

for benchmark in "${benchmark_array[@]}"; do
  benchmark="${benchmark//[[:space:]]/}"
  if [[ -z "$benchmark" ]]; then
    continue
  fi

  src_dir="$(src_dir_for_benchmark "$benchmark")"

  echo
  echo
  echo "===== ${benchmark}: prepare held-out split ====="
  prepare_split_benchmark "$benchmark" "$src_dir"

  echo
  echo "===== ${benchmark}: train-only calibration ====="
  calibrate_benchmark "$benchmark" "$src_dir"

  echo
  echo "===== ${benchmark}: train-only transition kernel ====="
  build_transition_kernel "$benchmark" "$src_dir"

  echo
  echo "===== ${benchmark}: Bayesian DP collection ====="
  collect_confidence_rows "$benchmark" "$src_dir"
done

echo
echo "DONE"
echo "outputs=${OUTPUT_ROOT}"
echo "log_file=${LOG_FILE}"
