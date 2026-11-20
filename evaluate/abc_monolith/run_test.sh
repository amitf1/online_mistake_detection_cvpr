#!/bin/bash
# Queue-resumable EgoPER monolith test inference. One process per GPU.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-configs/env/monolith_test.env}"

load_env_defaults() {
  local env_file="$1" line key
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
    key="${line%%=*}"
    if [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ && -z "${!key+x}" ]]; then
      export "${line}"
    fi
  done < "${env_file}"
}

load_env_defaults "${PROJECT_DIR}/${ENV_FILE}"
[[ -f "${PROJECT_DIR}/.env" ]] && load_env_defaults "${PROJECT_DIR}/.env"

EGOPER_ROOT="${EGOPER_ROOT:-${PROJECT_DIR}/data/EgoPER}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/abc_monolith/eval/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR//'${RUN_NAME}'/${RUN_NAME}}"
NUM_GPUS="${NUM_GPUS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${CHECKPOINT:-}" == "auto" ]]; then
  RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/abc_monolith/runs/abc_monolith_egoper_27b}"
  RUN_DIR="${RUN_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
  CHECKPOINT="$("${PYTHON_BIN}" - "${RUN_DIR}/best_checkpoints.json" << 'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
records = list(payload.get("best_checkpoints") or [])
if not records:
    raise SystemExit("best_checkpoints.json has no checkpoints")
greater = bool(payload.get("greater_is_better", True))
def rank(row):
    return (float(row["metric"]), int(row.get("epoch") or 0))
chosen = max(records, key=rank) if greater else min(records, key=rank)
print(f"selected epoch={chosen.get('epoch')} metric={chosen.get('metric')}", file=sys.stderr)
print(chosen["path"])
PY
)"
fi
[[ "${CHECKPOINT}" = /* ]] || CHECKPOINT="${PROJECT_DIR}/${CHECKPOINT}"
export OUTPUT_DIR CHECKPOINT
export FORCE_UNSLOTH_VIDEO_READER="${VIDEO_READER:-decord}"
export UNSLOTH_COMPILE_DISABLE="${UNSLOTH_COMPILE_DISABLE:-1}"

if [[ ! -f "${EGOPER_ROOT}/annotation.json" ]]; then
  echo "Missing EgoPER annotation under EGOPER_ROOT=${EGOPER_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Missing monolith checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/work_queue" "${OUTPUT_DIR}/videos"
cd "${PROJECT_DIR}"

bool_flag() {
  local enabled="$1" name="$2"
  if [[ "${enabled,,}" == "true" ]]; then
    printf -- "--%s" "${name}"
  else
    printf -- "--no-%s" "${name}"
  fi
}

RUN_ARGS=(
  evaluate/abc_monolith/run.py
  --egoper-root "${EGOPER_ROOT}"
  --checkpoint "${CHECKPOINT}"
  --output-json "${OUTPUT_DIR}/shards/generated.json"
  --split-file "${SPLIT_FILE}"
  --split "${EVAL_SPLIT}"
  --model-name "${MODEL_NAME}"
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES}"
  --vision-resize "${VISION_RESIZE}"
  "$(bool_flag "${LOAD_IN_4BIT}" load-in-4bit)"
  "$(bool_flag "${LOAD_IN_8BIT}" load-in-8bit)"
  "$(bool_flag "${LOAD_IN_16BIT}" load-in-16bit)"
  "$(bool_flag "${USE_CACHE}" use-cache)"
  "$(bool_flag "${MONOLITH_CANDIDATE_SCORING}" monolith-candidate-scoring)"
  --gpu-count 1
  --resume-completed-shards
  "$(bool_flag "${DEFER_METRICS}" defer-metrics)"
  --video-queue-dir "${OUTPUT_DIR}/work_queue"
  --video-checkpoint-dir "${OUTPUT_DIR}/videos"
  --queue-session-id "${SLURM_JOB_ID:-manual}"
  --video-shard-world-size "${NUM_GPUS}"
  --history-mode "${HISTORY_MODE:-predicted}"
  --history-max-events "${HISTORY_MAX_EVENTS:-12}"
)

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS=()
  for ((index = 0; index < NUM_GPUS; index++)); do
    GPU_IDS+=("${index}")
  done
fi
if [[ "${#GPU_IDS[@]}" -ne "${NUM_GPUS}" ]]; then
  echo "GPU count ${#GPU_IDS[@]} does not match NUM_GPUS=${NUM_GPUS}" >&2
  exit 1
fi

merge_collected_videos() {
  set +e
  "${PYTHON_BIN}" -m evaluate.abc_monolith.merge \
    --video-dir "${OUTPUT_DIR}/videos" \
    --queue-dir "${OUTPUT_DIR}/work_queue" \
    --output-json "${OUTPUT_DIR}/monolith_test_artifact.json" \
    --egoper-root "${EGOPER_ROOT}" \
    --defer-metrics
  local merge_status=$?
  set -e
  if (( merge_status == 2 )); then
    echo "[$(date -Is)] Incomplete. Run the same command again to continue."
    return 0
  fi
  if (( merge_status != 0 )); then
    return "${merge_status}"
  fi
  echo "[$(date -Is)] Test artifact: ${OUTPUT_DIR}/monolith_test_artifact.json"
}

PIDS=()
on_term() {
  echo "[$(date -Is)] SIGTERM: stopping workers. Finished videos are kept." >&2
  for pid in "${PIDS[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || true
  done
  merge_collected_videos
  exit 0
}
trap on_term SIGTERM SIGINT

for ((rank = 0; rank < NUM_GPUS; rank++)); do
  gpu="${GPU_IDS[$rank]}"
  echo "[$(date -Is)] start rank ${rank} gpu=${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
    exec "${PYTHON_BIN}" "${RUN_ARGS[@]}" --video-shard-rank "${rank}"
  ) &
  PIDS+=("$!")
done

queue_is_complete() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}/work_queue/progress.json" << 'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
progress = json.loads(path.read_text(encoding="utf-8"))
remaining = progress.get("remaining_a") or []
raise SystemExit(0 if progress.get("complete") is True and not remaining else 1)
PY
}

WORKER_STATUS=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    WORKER_STATUS=1
  fi
done
if (( WORKER_STATUS != 0 )); then
  echo "[$(date -Is)] A worker failed. Finished videos stay in ${OUTPUT_DIR}/videos." >&2
  if queue_is_complete; then
    merge_collected_videos
    exit $?
  fi
  exit 1
fi
merge_collected_videos
