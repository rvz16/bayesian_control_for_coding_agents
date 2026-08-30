#!/usr/bin/env bash
set -euo pipefail

# Runs the SAGE uncertainty pipeline against an already running
# OpenAI-compatible model endpoint.
#
# Required for local models:
#   GENERATOR_KEY=gpt_oss_20b_local and GPT_OSS_20B_BASE_URL=http://host:port/v1
#   GENERATOR_KEY=qwen25_32b       and QWEN25_32B_BASE_URL=http://host:port/v1
#
# Required for the L3 critic:
#   OPENROUTER_API_KEY

REPO_DIR="${REPO_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "${REPO_DIR}"

GENERATOR_KEY="${GENERATOR_KEY:-gpt_oss_20b_local}"
BENCHMARKS="${BENCHMARKS:-lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval,humanevalfix,codecontests}"
RUN_ROOT="${RUN_ROOT:-experiments/orchestration_hypothesis_testing/sim_results/sage_uncertainty_${GENERATOR_KEY}}"

N_INSTANCES="${N_INSTANCES:-0}"
N_TRAIN="${N_TRAIN:-}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.25}"
SPLIT_SEED="${SPLIT_SEED:-42}"
PRIOR_PATCHES="${PRIOR_PATCHES:-1}"

LCB_PLATFORM="${LCB_PLATFORM:-leetcode}"
LCB_VERSION="${LCB_VERSION:-all}"
PRIVATE_TEST_CAP="${PRIVATE_TEST_CAP:-0}"
PLUS_INPUT_CAP="${PLUS_INPUT_CAP:-200}"
SWE_HARNESS_WORKERS="${SWE_HARNESS_WORKERS:-1}"

MAX_STEPS="${MAX_STEPS:-20}"
MAX_GENERATIONS="${MAX_GENERATIONS:-5}"
MAX_VERIFICATIONS="${MAX_VERIFICATIONS:-2}"
AGENT_BACKEND="${AGENT_BACKEND:-sage}"
FINAL_VERIFY="${FINAL_VERIFY:-1}"
MAX_TOKENS_DECISION="${MAX_TOKENS_DECISION:-4096}"

TOP_LOGPROBS="${TOP_LOGPROBS:-20}"
SAVE_VERBALIZED_2S="${SAVE_VERBALIZED_2S:-1}"
REQUIRE_VERBALIZED_2S="${REQUIRE_VERBALIZED_2S:-0}"
VERBALIZED_2S_MAX_TOKENS="${VERBALIZED_2S_MAX_TOKENS:-1024}"
VERBALIZED_2S_TEMPERATURE="${VERBALIZED_2S_TEMPERATURE:-0.0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
RESUME="${RESUME:-1}"

mkdir -p "${RUN_ROOT}"

run_one() {
  local bench="$1"
  local stem="${bench}__${GENERATOR_KEY}"
  local out="${RUN_ROOT}/${stem}.jsonl"
  local tool_csv="${RUN_ROOT}/readable/${bench}/tool_success_by_instance.csv"
  local train_args=(--train-fraction "${TRAIN_FRACTION}")
  local resume_args=()
  local final_verify_args=()
  local verbalized_args=()

  if [[ -n "${N_TRAIN}" ]]; then
    train_args=(--n-train "${N_TRAIN}")
  fi
  if [[ "${RESUME}" == "1" ]]; then
    resume_args+=(--resume)
  fi
  if [[ "${FINAL_VERIFY}" == "1" ]]; then
    final_verify_args+=(--final-verify)
  fi
  if [[ "${SAVE_VERBALIZED_2S}" == "1" ]]; then
    verbalized_args+=(
      --save-verbalized-2s
      --verbalized-2s-max-tokens "${VERBALIZED_2S_MAX_TOKENS}"
      --verbalized-2s-temperature "${VERBALIZED_2S_TEMPERATURE}"
      --verbalized-2s-output "${RUN_ROOT}/${stem}.verbalized_2s.jsonl"
    )
  fi
  if [[ "${REQUIRE_VERBALIZED_2S}" == "1" ]]; then
    verbalized_args+=(--require-verbalized-2s)
  fi

  echo
  echo "===== run ${bench} ${GENERATOR_KEY} ====="
  python different_agents/v4/lcb_llm_tool_agent.py \
    --benchmark "${bench}" \
    --generator "${GENERATOR_KEY}" \
    --n-instances "${N_INSTANCES}" \
    --split-seed "${SPLIT_SEED}" \
    "${train_args[@]}" \
    --prior-patches "${PRIOR_PATCHES}" \
    --platform "${LCB_PLATFORM}" \
    --lcb-version "${LCB_VERSION}" \
    --private-test-cap "${PRIVATE_TEST_CAP}" \
    --plus-input-cap "${PLUS_INPUT_CAP}" \
    --swe-harness-workers "${SWE_HARNESS_WORKERS}" \
    --max-steps "${MAX_STEPS}" \
    --max-generations "${MAX_GENERATIONS}" \
    --max-verifications "${MAX_VERIFICATIONS}" \
    --max-tokens-decision "${MAX_TOKENS_DECISION}" \
    --agent-backend "${AGENT_BACKEND}" \
    "${final_verify_args[@]}" \
    --save-generation-logprobs \
    --require-generation-logprobs \
    --top-logprobs "${TOP_LOGPROBS}" \
    --output "${out}" \
    --logprobs-output "${RUN_ROOT}/${stem}.generation_logprobs.jsonl" \
    --split-output "${RUN_ROOT}/${stem}.split.json" \
    --prior-calibration-output "${RUN_ROOT}/${stem}.train_prior_calibration.jsonl" \
    --prior-calibration-logprobs-output "${RUN_ROOT}/${stem}.train_prior_calibration.generation_logprobs.jsonl" \
    --actions-output "${RUN_ROOT}/${stem}.actions.jsonl" \
    "${verbalized_args[@]}" \
    --print-each \
    "${resume_args[@]}"

  if [[ "${RUN_ANALYSIS}" == "1" ]]; then
    echo
    echo "===== analyze ${bench} ${GENERATOR_KEY} ====="
    mkdir -p "${RUN_ROOT}/readable/${bench}"
    python experiments/orchestration_hypothesis_testing/scripts/summarize_tool_action_success.py \
      "${out}" \
      --per-instance-csv "${tool_csv}"
    python experiments/orchestration_hypothesis_testing/scripts/analyze_lcb_llm_tool_agent_logs.py \
      --run-root "${RUN_ROOT}" \
      --benchmark "${bench}" \
      --generator "${GENERATOR_KEY}" \
      --tool-success-csv "${tool_csv}" \
      --output-dir "${RUN_ROOT}/readable/${bench}" \
      --seed "${SPLIT_SEED}" \
      --platform "${LCB_PLATFORM}" \
      --lcb-version "${LCB_VERSION}" \
      --private-test-cap "${PRIVATE_TEST_CAP}" \
      --plus-input-cap "${PLUS_INPUT_CAP}" \
      --swe-harness-workers "${SWE_HARNESS_WORKERS}"
  fi
}

IFS=',' read -ra BENCH_ARRAY <<< "${BENCHMARKS}"
for bench in "${BENCH_ARRAY[@]}"; do
  run_one "${bench}"
done

echo
echo "DONE"
echo "run_root=${RUN_ROOT}"
