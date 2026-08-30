#!/usr/bin/env bash
set -euo pipefail

systemctl --user start podman.socket
export USE_PODMAN=1

# Minimal launcher for fitted GrFt/DPFt experiments.
#
# Defaults:
#   - Synthesis benchmarks (LCB/MBPP+/HumanEval+): 2-fold cross evaluation.
#     Fold 0 trains theta/kernel on the first shuffled half and evaluates live on
#     the second half; fold 1 swaps the halves. Fold outputs are merged.
#   - SWE-Bench Lite/Verified: current live runner path with a calibration
#     subset, because the synthesis 2-fold scripts do not cover SWE.
#
# Common knobs:
#   GENS=qwen3_coder,haiku45,sonnet45,gpt5_mini
#   SYNTH_TASKS=lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval
#   SWE_TASKS=swebench_lite,swebench_verified
#   RUN_SYNTH=1 RUN_SWE=1
#   SYNTH_N_TEST=5       # smoke cap for each fold; 0 = all
#   SWE_CALIB_N=100
#   USE_PODMAN=1         # for SWE on Podman

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
GENS_RAW="${GENS:-qwen3_coder,haiku45,sonnet45,gpt5_mini}"
SYNTH_TASKS="${SYNTH_TASKS:-lcb_hard,lcb_medium,lcb_easy,mbpp,humaneval}"
SWE_TASKS="${SWE_TASKS:-swebench_lite,swebench_verified}"
RUN_SYNTH="${RUN_SYNTH:-1}"
RUN_SWE="${RUN_SWE:-1}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SYNTH_VARIANTS="${SYNTH_VARIANTS:-fitted}"
KERNEL_MODE="${KERNEL_MODE:-measured}"
INITIAL_PRIOR="${INITIAL_PRIOR:-fixed_0.5}"
SYNTH_N_TEST="${SYNTH_N_TEST:-0}"
SWE_CALIB_N="${SWE_CALIB_N:-100}"
SWE_WORKERS="${SWE_WORKERS:-1}"
RESULT_DIR="${RESULT_DIR:-sim_results}"
GEN_CAPS="${GEN_CAPS:-qwen3_coder=4.0,haiku45=4.0,sonnet45=15.0,gpt5_mini=4.0}"
LIVE_CAPS="${LIVE_CAPS:-qwen3_coder=20.0,haiku45=20.0,sonnet45=50.0,gpt5_mini=20.0}"

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

GENS="$(normalize_gens "${GENS_RAW}")"

run() {
  echo
  printf '+'
  printf ' %q' "$@"
  echo
  "$@"
}

model_for_gen() {
  case "$1" in
    qwen3_coder) echo "qwen/qwen3-coder" ;;
    haiku45) echo "anthropic/claude-haiku-4.5" ;;
    sonnet45) echo "anthropic/claude-sonnet-4.5" ;;
    gpt5_mini) echo "openai/gpt-5-mini" ;;
    *) echo "$1" ;;
  esac
}

canonical_task() {
  case "$1" in
    lcb_med) echo "lcb_medium" ;;
    *) echo "$1" ;;
  esac
}

synth_n() {
  case "$(canonical_task "$1")" in
    lcb_hard) echo 102 ;;
    lcb_medium) echo 207 ;;
    lcb_easy) echo 135 ;;
    mbpp) echo 378 ;;
    humaneval) echo 164 ;;
    *) echo "unknown synthesis task: $1" >&2; exit 2 ;;
  esac
}

synth_full_dir() {
  echo "data/$(canonical_task "$1")_full"
}

synth_calibrated() {
  local dir="$1" gen
  IFS=',' read -ra gens <<< "${GENS}"
  for gen in "${gens[@]}"; do
    [[ -f "${dir}/${gen}/critic_results.jsonl" ]] || return 1
  done
}

calibrate_synth() {
  local task n out difficulty
  task="$(canonical_task "$1")"
  n="$(synth_n "${task}")"
  out="$(synth_full_dir "${task}")"
  synth_calibrated "${out}" && return
  case "${task}" in
    lcb_*)
      difficulty="${task#lcb_}"
      run "${PYTHON}" scripts/lcb_calibrate.py --output-dir "${out}" \
        --difficulty "${difficulty}" --generators "${GENS}" --n-instances "${n}" \
        --n-patches 3 --lcb-version all --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
    mbpp)
      run "${PYTHON}" scripts/mbpp_calibrate.py --output-dir "${out}" \
        --generators "${GENS}" --n-instances "${n}" --n-patches 3 \
        --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
    humaneval)
      run "${PYTHON}" scripts/humaneval_calibrate.py --output-dir "${out}" \
        --generators "${GENS}" --n-instances "${n}" --n-patches 3 --plus-input-cap 200 \
        --max-cost-usd-per-model "${GEN_CAPS}"
      ;;
  esac
}

prepare_fold_dir() {
  local task="$1" fold="$2" src dst gen
  src="$(synth_full_dir "${task}")"
  dst="data/$(canonical_task "${task}")_cv/fold${fold}"
  IFS=',' read -ra gens <<< "${GENS}"
  for gen in "${gens[@]}"; do
    mkdir -p "${dst}/${gen}"
    cp -f "${src}/${gen}/critic_results.jsonl" "${dst}/${gen}/critic_results.jsonl"
    if [[ -f "${src}/${gen}/likelihood_tables.json" ]]; then
      cp -f "${src}/${gen}/likelihood_tables.json" "${dst}/${gen}/likelihood_tables.json"
    fi
  done
  echo "${dst}"
}

run_synth_cv() {
  local task="$1" fold fold_dir gen model out ntest_arg=()
  task="$(canonical_task "${task}")"
  calibrate_synth "${task}"
  mkdir -p "${RESULT_DIR}"
  [[ "${SYNTH_N_TEST}" != "0" ]] && ntest_arg=(--n-test "${SYNTH_N_TEST}")

  for fold in 0 1; do
    fold_dir="$(prepare_fold_dir "${task}" "${fold}")"
    run "${PYTHON}" scripts/synthesis_train_test_split.py \
      --src-dir "${fold_dir}" --generators "${GENS}" \
      --train-fraction "${TRAIN_FRACTION}" --fold-index "${fold}" \
      --split-seed "${SPLIT_SEED}"
    run "${PYTHON}" scripts/synthesis_transition_kernel.py \
      --src-dir "${fold_dir}" --generators "${GENS}" \
      --benchmark "${task}" --split-scope train

    IFS=',' read -ra gens <<< "${GENS}"
    for gen in "${gens[@]}"; do
      model="$(model_for_gen "${gen}")"
      out="${RESULT_DIR}/synthesis_live__${task}__${gen}__fold${fold}.json"
      run env ABBO_LLM_MODEL="${model}" "${PYTHON}" scripts/run_synthesis_live.py \
        --src-dir "${fold_dir}" --benchmark "${task}" --generator "${gen}" \
        --output "${out}" --variants "${SYNTH_VARIANTS}" \
        --kernel-mode "${KERNEL_MODE}" --initial-prior "${INITIAL_PRIOR}" \
        "${ntest_arg[@]}"
    done
  done

  IFS=',' read -ra gens <<< "${GENS}"
  for gen in "${gens[@]}"; do
    combine_folds "${task}" "${gen}"
  done
}

combine_folds() {
  local task="$1" gen="$2"
  local out="${RESULT_DIR}/synthesis_live__${task}__${gen}__2fold.json"
  local f0="${RESULT_DIR}/synthesis_live__${task}__${gen}__fold0.json"
  local f1="${RESULT_DIR}/synthesis_live__${task}__${gen}__fold1.json"
  run "${PYTHON}" - "${out}" "${f0}" "${f1}" <<'PY'
import json
import sys
import csv
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
states = []
for p in paths:
    if not p.exists():
        raise SystemExit(f"missing fold output: {p}")
    states.append(json.loads(p.read_text()))

state = {
    "results": {},
    "folds": [{"path": str(p), "n_train": s.get("n_train"), "n_test": s.get("n_test")}
              for p, s in zip(paths, states)],
}
for key in ("benchmark", "generator", "llm_model", "split_seed", "kernel_mode", "costs"):
    state[key] = states[0].get(key)
state["fold_outputs"] = [str(p) for p in paths]

for s in states:
    state["results"].update(s.get("results", {}))

costs = state.get("costs") or {}
reward = float(costs.get("reward", 100))
all_variants = [
    "simple", "best_of_3", "threshold_L0", "threshold_L2", "threshold_L3",
    "fixed_pipeline", "greedy_hand", "greedy_fitted", "dp_hand", "dp_fitted",
]
by_variant = defaultdict(list)
for rec in state["results"].values():
    by_variant[rec["variant"]].append(rec)

summaries = []
aggregate_by_policy = {}
baseline = None
for name in all_variants:
    rows = by_variant.get(name, [])
    if not rows:
        continue
    n = len(rows)
    fix_rate = sum(1 for r in rows if r.get("fixed")) / n
    mean_cost = sum(float(r.get("total_cost", 0.0)) for r in rows) / n
    total_cost = sum(float(r.get("total_cost", 0.0)) for r in rows)
    total_api_cost = sum(float(r.get("api_cost_usd", 0.0)) for r in rows)
    total_llm_calls = sum(int(r.get("n_llm_calls", 0)) for r in rows)
    total_critic_runs = sum(int(r.get("n_critic_runs", 0)) for r in rows)
    total_full_tests = sum(int(r.get("n_full_tests", 0)) for r in rows)
    total_completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in rows)
    final_actions = defaultdict(int)
    for r in rows:
        final_actions[str(r.get("final_action", ""))] += 1
    mean_util = reward * fix_rate - mean_cost
    if name == "simple":
        baseline = mean_util
    entry = {
        "policy": name,
        "n_episodes": n,
        "mean_utility": round(mean_util, 4),
        "pass_rate": round(fix_rate, 4),
        "fix_rate": round(fix_rate, 4),
        "mean_cost": round(mean_cost, 4),
        "total_cost": round(total_cost, 4),
        "mean_api_cost_usd": round(total_api_cost / n, 8),
        "total_api_cost_usd": round(total_api_cost, 8),
        "mean_llm_calls": round(total_llm_calls / n, 4),
        "total_llm_calls": total_llm_calls,
        "mean_critic_runs": round(total_critic_runs / n, 4),
        "total_critic_runs": total_critic_runs,
        "mean_full_tests": round(total_full_tests / n, 4),
        "total_full_tests": total_full_tests,
        "mean_completion_tokens": round(total_completion_tokens / n, 4),
        "total_completion_tokens": total_completion_tokens,
        "final_actions": dict(sorted(final_actions.items())),
    }
    if baseline is not None:
        entry["delta_vs_simple"] = round(mean_util - baseline, 4)
    summaries.append(entry)
    aggregate_by_policy[name] = entry

state["summaries"] = summaries
state["aggregate_by_policy"] = aggregate_by_policy
state["n_test"] = len({k.split("|", 1)[0] for k in state["results"]})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(state, indent=2))
summary_json = out.with_name(out.stem + ".summary.json")
summary_csv = out.with_name(out.stem + ".summary.csv")
summary_json.write_text(json.dumps({
    "benchmark": state.get("benchmark"),
    "generator": state.get("generator"),
    "n_test": state.get("n_test"),
    "folds": state.get("folds"),
    "summaries": summaries,
    "aggregate_by_policy": aggregate_by_policy,
}, indent=2))
csv_fields = [
    "policy", "n_episodes", "pass_rate", "fix_rate", "mean_utility",
    "delta_vs_simple", "mean_cost", "total_cost", "mean_api_cost_usd",
    "total_api_cost_usd", "mean_llm_calls", "total_llm_calls",
    "mean_critic_runs", "total_critic_runs", "mean_full_tests",
    "total_full_tests", "mean_completion_tokens", "total_completion_tokens",
    "final_actions",
]
with summary_csv.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=csv_fields)
    writer.writeheader()
    for entry in summaries:
        row = dict(entry)
        row["final_actions"] = json.dumps(row.get("final_actions", {}), sort_keys=True)
        writer.writerow({field: row.get(field, "") for field in csv_fields})
print(f"combined -> {out}")
print(f"summary  -> {summary_json}")
print(f"csv      -> {summary_csv}")
PY
}

swe_n() {
  case "$1" in
    swebench_lite) echo 300 ;;
    swebench_verified) echo 500 ;;
    *) echo "unknown SWE task: $1" >&2; exit 2 ;;
  esac
}

swe_dataset() {
  case "$1" in
    swebench_lite) echo "princeton-nlp/SWE-bench_Lite" ;;
    swebench_verified) echo "princeton-nlp/SWE-bench_Verified" ;;
  esac
}

swe_calib_dir() {
  echo "data/$1_n${SWE_CALIB_N}"
}

swe_tables_exist() {
  local dir="$1" gen
  IFS=',' read -ra gens <<< "${GENS}"
  for gen in "${gens[@]}"; do
    [[ -f "${dir}/${gen}/likelihood_tables.json" ]] || return 1
  done
}

run_swe() {
  local task="$1" dir dataset n
  dir="$(swe_calib_dir "${task}")"
  dataset="$(swe_dataset "${task}")"
  n="$(swe_n "${task}")"

  if ! swe_tables_exist "${dir}"; then
    run "${PYTHON}" scripts/spot_check_generators.py --output-dir "${dir}" \
      --dataset "${dataset}" --generators "${GENS}" \
      --n-instances "${SWE_CALIB_N}" --n-patches 3 --max-workers-gen 1 \
      --max-workers-eval "${SWE_WORKERS}" --max-cost-usd-per-model "${GEN_CAPS}"
    run "${PYTHON}" scripts/calibrate_from_spotcheck.py --output-dir "${dir}" \
      --dataset "${dataset}" --generators "${GENS}"
  fi

  run "${PYTHON}" scripts/run_fitted_live.py --benchmark "${task}" \
    --generators "${GENS}" --policies greedy_fitted,dp_fitted \
    --calibration-dir "${dir}" --n-instances "${n}" \
    --max-generations 3 --max-actions 12 \
    --max-api-cost-usd-per-model "${LIVE_CAPS}" \
    --swe-harness-workers "${SWE_WORKERS}"
}

echo "Generators: ${GENS}"
echo "Synthesis tasks: ${SYNTH_TASKS}  train_fraction=${TRAIN_FRACTION}"
echo "Synthesis initial prior: ${INITIAL_PRIOR}"
echo "SWE tasks: ${SWE_TASKS}  SWE_CALIB_N=${SWE_CALIB_N}"
echo "Result dir: ${RESULT_DIR}"

if [[ "${RUN_SYNTH}" == "1" ]]; then
  IFS=',' read -ra tasks <<< "${SYNTH_TASKS}"
  for task in "${tasks[@]}"; do
    task="${task// /}"
    [[ -n "${task}" ]] && run_synth_cv "${task}"
  done
fi

if [[ "${RUN_SWE}" == "1" ]]; then
  IFS=',' read -ra tasks <<< "${SWE_TASKS}"
  for task in "${tasks[@]}"; do
    task="${task// /}"
    [[ -n "${task}" ]] && run_swe "${task}"
  done
fi

echo
echo "Done."
