#!/usr/bin/env bash
# Launch one independent pipeline process per GPU, each owning a video-id shard.
# Not torchrun / DDP: ranks do not share a process group or a model replica.
set -euo pipefail

WORLD=""
OUTPUT_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHILD_ARGS=()

usage() {
  echo "Usage: $0 --world-size N --output-dir DIR -- python inference/online_pipeline.py ..." >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world-size)
      WORLD="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --)
      shift
      CHILD_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      ;;
  esac
done

if [[ -z "${WORLD}" || -z "${OUTPUT_DIR}" || ${#CHILD_ARGS[@]} -eq 0 ]]; then
  usage
fi
if ! [[ "${WORLD}" =~ ^[1-9][0-9]*$ ]]; then
  echo "world-size must be a positive integer, got: ${WORLD}" >&2
  exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS=()
  for ((index = 0; index < WORLD; index++)); do
    GPU_IDS+=("${index}")
  done
fi
if [[ "${#GPU_IDS[@]}" -ne "${WORLD}" ]]; then
  echo "GPU count ${#GPU_IDS[@]} does not match world-size ${WORLD}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
PIDS=()
STATUS=0

on_term() {
  echo "[$(date -Is)] SIGTERM: asking ${#PIDS[@]} workers to finish the current video" >&2
  for pid in "${PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || true
  done
  echo "[$(date -Is)] workers stopped; writing progress / merging if complete"
  "${PYTHON_BIN}" inference/online_pipeline.py --merge-if-complete "${OUTPUT_DIR}" || true
  exit 0
}
trap on_term TERM INT

for ((rank = 0; rank < WORLD; rank++)); do
  gpu="${GPU_IDS[$rank]}"
  shard_name="$(printf 'rank%02d-of-%02d' "${rank}" "${WORLD}")"
  shard_dir="${OUTPUT_DIR}/shards/${shard_name}"
  mkdir -p "${shard_dir}"
  log="${shard_dir}/rank.log"
  echo "[$(date -Is)] start ${shard_name} gpu=${gpu} log=${log}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    if [[ -n "${TRITON_CACHE_DIR:-}" ]]; then
      export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}-${shard_name}"
      mkdir -p "${TRITON_CACHE_DIR}"
    fi
    if [[ -n "${TORCHINDUCTOR_CACHE_DIR:-}" ]]; then
      export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}-${shard_name}"
      mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
    fi
    exec "${CHILD_ARGS[@]}" \
      --output-dir "${OUTPUT_DIR}" \
      --video-shard-rank "${rank}" \
      --video-shard-world-size "${WORLD}"
  ) >"${log}" 2>&1 &
  PIDS+=("$!")
done

for index in "${!PIDS[@]}"; do
  pid="${PIDS[$index]}"
  shard_name="$(printf 'rank%02d-of-%02d' "${index}" "${WORLD}")"
  if wait "${pid}"; then
    echo "[$(date -Is)] ${shard_name} finished"
  else
    echo "[$(date -Is)] ${shard_name} failed; see ${OUTPUT_DIR}/shards/${shard_name}/rank.log" >&2
    STATUS=1
  fi
done

if [[ "${STATUS}" -ne 0 ]]; then
  echo "One or more workers failed; writing progress for a later resubmit." >&2
  "${PYTHON_BIN}" inference/online_pipeline.py --merge-if-complete "${OUTPUT_DIR}" || true
  exit "${STATUS}"
fi

echo "[$(date -Is)] checking whether all ${WORLD} workers finished the eval split"
exec "${PYTHON_BIN}" inference/online_pipeline.py --merge-if-complete "${OUTPUT_DIR}"
