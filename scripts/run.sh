#!/bin/bash
# Usage: scripts/run.sh <module_a|module_b|module_c|abc_monolith|pipeline_test|monolith_test> <env file>
# Works locally and inside a Slurm allocation (see scripts/slurm/).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODULE="${1:-}"
ENV_FILE="${2:-}"
if [[ -z "${MODULE}" || -z "${ENV_FILE}" ]]; then
  sed -n '2p' "${BASH_SOURCE[0]}" >&2
  exit 1
fi
ENV_FILE="${ENV_FILE#"${REPO}/"}"
if [[ ! -f "${REPO}/${ENV_FILE}" ]]; then
  echo "Missing env file: ${REPO}/${ENV_FILE}" >&2
  exit 1
fi
export ENV_FILE

case "${MODULE}" in
  module_a|module_b|module_c|abc_monolith) BUILDER="${REPO}/train/${MODULE}/build_train_args.sh" ;;
  pipeline_test) BUILDER="${REPO}/train/pipeline/build_pipeline_args.sh" ;;
  monolith_test) BUILDER="" ;;
  *) echo "Unknown module: ${MODULE}" >&2; exit 1 ;;
esac

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_CACHE="${HF_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}}"
export HF_HOME="${HF_CACHE}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

KERNEL_CACHE_ROOT="${KERNEL_CACHE_ROOT:-/tmp/kernel-cache-${SLURM_JOB_ID:-$$}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${KERNEL_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${KERNEL_CACHE_ROOT}/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${KERNEL_CACHE_ROOT}/nv}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"

if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
  export NUM_GPUS="${SLURM_GPUS_ON_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  export NUM_GPUS="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
fi

cd "${REPO}"
echo "[$(date -Is)] module=${MODULE} env=${ENV_FILE} num_gpus=${NUM_GPUS:-from env} restart=${SLURM_RESTART_COUNT:-0}"

PID=""
on_term() {
  echo "[$(date -Is)] SIGTERM received; forwarding to PID=${PID:-none}"
  if [[ -n "${PID}" ]]; then
    kill -TERM "${PID}" 2>/dev/null || true
    wait "${PID}" || true
  fi
  exit 0
}
trap on_term SIGTERM

if [[ "${MODULE}" == "monolith_test" ]]; then
  bash "${REPO}/evaluate/abc_monolith/run_test.sh" &
else
  (
    # shellcheck disable=SC1090
    source "${BUILDER}"
    export WANDB_DIR="${WANDB_DIR}"
    export WANDB_PROJECT="${WANDB_PROJECT:-online-mistake-detection}"
    export WANDB_NAME="${WANDB_NAME:-${RUN_NAME}}"
    export FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER:-decord}"
    export UNSLOTH_COMPILE_DISABLE="${UNSLOTH_COMPILE_DISABLE:-}"
    echo "[$(date -Is)] launch: ${TRAIN_ARGS[*]}"
    exec "${TRAIN_ARGS[@]}"
  ) &
fi
PID=$!
wait "${PID}"
