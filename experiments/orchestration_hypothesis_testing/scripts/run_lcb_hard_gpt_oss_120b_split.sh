#!/usr/bin/env bash
set -euo pipefail

# LCB-hard pipeline with a real held-out split:
# prepare split -> train-only calibration/likelihood/kernel -> test-only live DP.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
GENERATOR="${GENERATOR:-gpt_oss_120b_local}"

export GPT_OSS_120B_BASE_URL="${GPT_OSS_120B_BASE_URL:-http://127.0.0.1:8000/v1}"
export GPT_OSS_120B_MODEL="${GPT_OSS_120B_MODEL:-openai/gpt-oss-120b}"

SRC_DIR="${SRC_DIR:-data/lcb_hard_gpt_oss_120b_split}"
OUTPUT_DIR="${OUTPUT_DIR:-data/final_confidence_bayes_quality_lcb_hard_gpt_oss_120b}"
LOG_DIR="${LOG_DIR:-logs/lcb_hard_gpt_oss_120b_split}"

N_CALIB_INSTANCES="${N_CALIB_INSTANCES:-0}"
N_PATCHES="${N_PATCHES:-3}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.25}"
N_TRAIN="${N_TRAIN:-}"
N_TEST="${N_TEST:-0}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SEED="${SEED:-42}"

LCB_PLATFORM="${LCB_PLATFORM:-leetcode}"
LCB_VERSION="${LCB_VERSION:-all}"
LCB_EXTEND_EXISTING="${LCB_EXTEND_EXISTING:-0}"

MAX_COST_CALIB="${MAX_COST_CALIB:-5.0}"
MAX_COST_COLLECT="${MAX_COST_COLLECT:-5.0}"

C_GEN="${C_GEN:-5}"
C_L0="${C_L0:-1}"
C_L1="${C_L1:-1}"
C_L2="${C_L2:-2}"
C_L3="${C_L3:-5}"
C_VER="${C_VER:-30}"
REWARD="${REWARD:-100}"

MAX_GENERATIONS="${MAX_GENERATIONS:-10}"
MAX_VERIFICATIONS="${MAX_VERIFICATIONS:-5}"
MAX_ACTIONS="${MAX_ACTIONS:-24}"

LOGPROB_RETRY_ATTEMPTS="${LOGPROB_RETRY_ATTEMPTS:-5}"
LOGPROB_RETRY_SLEEP="${LOGPROB_RETRY_SLEEP:-5}"

export ORCH_HF_CACHE_DIR="${ORCH_HF_CACHE_DIR:-/capstor/store/cscs/swissai/a0142/hf_cache}"
export ORCH_FORCE_HF_CACHE="${ORCH_FORCE_HF_CACHE:-1}"
export HF_HOME="${HF_HOME:-${ORCH_HF_CACHE_DIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$LOG_DIR"

LOG_FILE="${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

if [[ -z "${OPENROUTER_API_KEY:-}${OPEN_ROUTER_API_KEY:-}${OPEN_ROUTER:-}" ]]; then
  echo "ERROR: OpenRouter key is required for L3 critic/reviewer calibration."
  echo "Set one of OPENROUTER_API_KEY, OPEN_ROUTER_API_KEY, OPEN_ROUTER."
  exit 1
fi

run() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

echo "log_file=${LOG_FILE}"
echo "generator=${GENERATOR}"
echo "local_model=${GPT_OSS_120B_MODEL}"
echo "local_base_url=${GPT_OSS_120B_BASE_URL}"
echo "src_dir=${SRC_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "n_total_split_instances=${N_CALIB_INSTANCES} (0=all loaded ids) n_patches=${N_PATCHES}"
echo "split: n_train=${N_TRAIN:-<unset>} train_fraction=${TRAIN_FRACTION} n_test_cap=${N_TEST} (0=all held-out)"
echo "costs: c_gen=${C_GEN} c_L0=${C_L0} c_L1=${C_L1} c_L2=${C_L2} c_L3=${C_L3} c_ver=${C_VER} reward=${REWARD}"

prepare_args=(
  "$PYTHON_BIN" scripts/synthesis_prepare_train_test_split.py
  --src-dir "$SRC_DIR"
  --benchmark lcb_hard
  --generators "$GENERATOR"
  --n-instances "$N_CALIB_INSTANCES"
  --sample-seed "$SEED"
  --split-seed "$SPLIT_SEED"
)
if [[ -n "$N_TRAIN" ]]; then
  prepare_args+=(--n-train "$N_TRAIN")
else
  prepare_args+=(--train-fraction "$TRAIN_FRACTION")
fi

echo
echo "===== prepare held-out split before calibration ====="
run "${prepare_args[@]}"

calib_args=(
  "$PYTHON_BIN" scripts/lcb_calibrate.py
  --output-dir "$SRC_DIR"
  --generators "$GENERATOR"
  --sample-ids-file "$SRC_DIR/train_sample.json"
  --n-patches "$N_PATCHES"
  --difficulty hard
  --platform "$LCB_PLATFORM"
  --lcb-version "$LCB_VERSION"
  --seed "$SEED"
  --max-cost-usd-per-model "$MAX_COST_CALIB"
)
if [[ "$LCB_EXTEND_EXISTING" == "1" ]]; then
  calib_args+=(--extend-existing)
fi

echo
echo "===== train-only calibration: LCB hard candidates ====="
run "${calib_args[@]}"

echo
echo "===== train-only transition kernel ====="
run "$PYTHON_BIN" scripts/synthesis_transition_kernel.py \
  --src-dir "$SRC_DIR" \
  --generators "$GENERATOR" \
  --benchmark lcb_hard \
  --split-scope train

echo
echo "===== held-out Bayesian DP live collection ====="
run "$PYTHON_BIN" scripts/collect_final_confidence_bayes_quality.py \
  --src-dir "$SRC_DIR" \
  --benchmark lcb_hard \
  --generators "$GENERATOR" \
  --policies bayesian_DP \
  --critic-allowlist L0,L1,L2,L3 \
  --require-split \
  --n-instances "$N_TEST" \
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
  --require-logprobs-in-response \
  --no-logprob-fallback \
  --logprob-retry-attempts "$LOGPROB_RETRY_ATTEMPTS" \
  --logprob-retry-sleep "$LOGPROB_RETRY_SLEEP" \
  --output-dir "$OUTPUT_DIR"

echo
echo "DONE"
echo "src_dir=${SRC_DIR}"
echo "output_dir=${OUTPUT_DIR}"
echo "actions=${OUTPUT_DIR}/lcb_hard/${GENERATOR}/controller_actions.jsonl"
echo "log_file=${LOG_FILE}"
