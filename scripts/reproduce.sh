#!/bin/bash
# Usage: bash scripts/reproduce.sh [test|evaluate|all]
#   test      run the Modular and Joint tests from the checkpoints (resumable)
#   evaluate  build the final predictions and every result table from the two test artifacts
#   all       test, then evaluate once both tests are complete (default)
# Exit code 3 means a test stopped before all videos were done: run the same command again.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"
cd "${REPO}"

PIPELINE_ENV="${PIPELINE_ENV:-configs/env/pipeline_test.env}"
MONOLITH_ENV="${MONOLITH_ENV:-configs/env/monolith_test.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO}/outputs}"
[[ "${OUTPUT_ROOT}" = /* ]] || OUTPUT_ROOT="${REPO}/${OUTPUT_ROOT}"
export OUTPUT_ROOT
RESULTS_DIR="${RESULTS_DIR:-${OUTPUT_ROOT}/results}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_BERTSCORE="${SKIP_BERTSCORE:-false}"

env_value() {
  awk -F= -v key="$2" '$1 == key {print substr($0, length(key) + 2); exit}' "${REPO}/$1"
}
PIPELINE_RUN_NAME="${PIPELINE_RUN_NAME:-$(env_value "${PIPELINE_ENV}" RUN_NAME)}"
MONOLITH_RUN_NAME="${MONOLITH_RUN_NAME:-$(env_value "${MONOLITH_ENV}" RUN_NAME)}"
PIPELINE_ARTIFACT="${OUTPUT_ROOT}/online_pipeline_eval/${PIPELINE_RUN_NAME}/pipeline_artifacts.json"
MONOLITH_ARTIFACT="${OUTPUT_ROOT}/abc_monolith/eval/${MONOLITH_RUN_NAME}/monolith_test_artifact.json"

PID=""
on_term() {
  echo "[$(date -Is)] SIGTERM received; forwarding to PID=${PID:-none}"
  if [[ -n "${PID}" ]]; then
    kill -TERM "${PID}" 2>/dev/null || true
    wait "${PID}" || true
  fi
  exit 3
}
trap on_term SIGTERM

run_test() {
  local stage="$1" env_file="$2" run_name="$3" artifact="$4"
  if [[ -f "${artifact}" ]]; then
    echo "[$(date -Is)] ${stage}: complete (${artifact})"
    return 0
  fi
  echo "[$(date -Is)] ${stage}: running ${env_file} -> ${artifact}"
  RUN_NAME="${run_name}" bash scripts/run.sh "${stage}" "${env_file}" &
  PID=$!
  local status=0
  wait "${PID}" || status=$?
  PID=""
  if [[ ! -f "${artifact}" ]]; then
    echo "[$(date -Is)] ${stage}: stopped (exit ${status}) before all videos were done; run the same command again" >&2
    exit 3
  fi
}

run_tests() {
  run_test pipeline_test "${PIPELINE_ENV}" "${PIPELINE_RUN_NAME}" "${PIPELINE_ARTIFACT}"
  run_test monolith_test "${MONOLITH_ENV}" "${MONOLITH_RUN_NAME}" "${MONOLITH_ARTIFACT}"
}

run_evaluation() {
  for artifact in "${PIPELINE_ARTIFACT}" "${MONOLITH_ARTIFACT}"; do
    if [[ ! -f "${artifact}" ]]; then
      echo "Missing ${artifact}. Run: bash scripts/reproduce.sh test" >&2
      exit 1
    fi
  done
  export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
  mkdir -p "${RESULTS_DIR}"
  "${PYTHON_BIN}" evaluate/egoper/final_predictions.py \
    --pipeline-artifact "${PIPELINE_ARTIFACT}" --pipeline-env "${PIPELINE_ENV}" \
    --monolith-artifact "${MONOLITH_ARTIFACT}" --monolith-env "${MONOLITH_ENV}" \
    --output-dir "${RESULTS_DIR}/final_predictions"
  "${PYTHON_BIN}" evaluate/egoper/evaluate_final.py \
    --predictions "${RESULTS_DIR}/final_predictions" --output "${RESULTS_DIR}/metrics.json"
  "${PYTHON_BIN}" evaluate/egoper/component_metrics.py \
    --pipeline-artifact "${PIPELINE_ARTIFACT}" --output "${RESULTS_DIR}/component_metrics.json"
  "${PYTHON_BIN}" evaluate/egoper/timing.py \
    --pipeline-artifact "${PIPELINE_ARTIFACT}" --monolith-artifact "${MONOLITH_ARTIFACT}" \
    --output "${RESULTS_DIR}/timing.json"
  if [[ "${SKIP_BERTSCORE,,}" != "true" ]]; then
    "${PYTHON_BIN}" evaluate/egoper/explanations.py \
      --predictions "${RESULTS_DIR}/final_predictions" --output "${RESULTS_DIR}/explanations.json" \
      ${BERTSCORE_DEVICE:+--device "${BERTSCORE_DEVICE}"}
  fi
  "${PYTHON_BIN}" evaluate/egoper/tables.py --results "${RESULTS_DIR}" --compare "${REPO}/results"
}

case "${STAGE}" in
  test) run_tests ;;
  evaluate) run_evaluation ;;
  all) run_tests; run_evaluation ;;
  *) sed -n '2,6p' "${BASH_SOURCE[0]}" >&2; exit 1 ;;
esac
