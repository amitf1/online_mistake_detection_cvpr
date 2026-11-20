# Builds the online A -> B -> C pipeline command from ENV_FILE. Sourced by scripts/run.sh.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

load_env_defaults() {
  local env_file="$1"
  local line key
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]] && continue
    key="${line%%=*}"
    if [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ && -z "${!key+x}" ]]; then
      export "${line}"
    fi
  done < "${env_file}"
}

ENV_FILE="${ENV_FILE:-configs/env/pipeline_test.env}"
load_env_defaults "${PROJECT_DIR}/${ENV_FILE}"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
  load_env_defaults "${PROJECT_DIR}/.env"
fi

EGOPER_ROOT="${EGOPER_ROOT:-${PROJECT_DIR}/data/EgoPER}"
DATASET="${DATASET:-egoper}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
REPORT_TO="${REPORT_TO:-}"
MAX_VIDEOS="${MAX_VIDEOS:-389}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-35B-A3B}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"
LOAD_IN_16BIT="${LOAD_IN_16BIT:-true}"
SPLIT_FILE="${SPLIT_FILE:-}"
REGENERATE_SPLIT="${REGENERATE_SPLIT:-false}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-}"
TRAIN_VIDEOS_PER_TASK="${TRAIN_VIDEOS_PER_TASK:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES_A="${MAX_FRAMES_A:-52}"
MAX_FRAMES_B="${MAX_FRAMES_B:-54}"
MAX_FRAMES_C="${MAX_FRAMES_C:-40}"
VISION_RESIZE_A="${VISION_RESIZE_A:-0}"
VISION_RESIZE_B="${VISION_RESIZE_B:-0}"
VISION_RESIZE_C="${VISION_RESIZE_C:-0}"
MAX_SEQ_LENGTH_A="${MAX_SEQ_LENGTH_A:-25600}"
MAX_SEQ_LENGTH_B="${MAX_SEQ_LENGTH_B:-25600}"
MAX_SEQ_LENGTH_C="${MAX_SEQ_LENGTH_C:-20480}"
MODULE_A_MAX_NEW_TOKENS="${MODULE_A_MAX_NEW_TOKENS:-8}"
MODULE_B_MAX_NEW_TOKENS="${MODULE_B_MAX_NEW_TOKENS:-64}"
MODULE_C_MAX_NEW_TOKENS="${MODULE_C_MAX_NEW_TOKENS:-128}"
MODULE_C_HISTORY_MAX_EVENTS="${MODULE_C_HISTORY_MAX_EVENTS:-12}"
MODULE_C_CLIP="${MODULE_C_CLIP:-span}"
MODULE_C_CANDIDATE_SCORING="${MODULE_C_CANDIDATE_SCORING:-false}"
MODULE_C_SCORE_ONLY="${MODULE_C_SCORE_ONLY:-false}"
MODULE_A_LABEL_MODE="${MODULE_A_LABEL_MODE:-step_id}"
LABEL_SCORE_EVAL="${LABEL_SCORE_EVAL:-true}"
ONLINE_RESET_FROM_B="${ONLINE_RESET_FROM_B:-true}"
CAUSAL_STREAM="${CAUSAL_STREAM:-false}"
CAUSAL_MAX_VIDEOS="${CAUSAL_MAX_VIDEOS:--1}"
CAUSAL_MODULE_C_IN_LOOP="${CAUSAL_MODULE_C_IN_LOOP:-false}"
STRIDE_SECONDS="${STRIDE_SECONDS:-5.0}"
ZERO_SHOT_EVAL="${ZERO_SHOT_EVAL:-false}"
MODULE_A_CHECKPOINT="${MODULE_A_CHECKPOINT:-}"
MODULE_B_CHECKPOINT="${MODULE_B_CHECKPOINT:-}"
MODULE_C_CHECKPOINT="${MODULE_C_CHECKPOINT:-}"
REUSE_EVENTS_JSON="${REUSE_EVENTS_JSON:-}"
SKIP_INFERENCE="${SKIP_INFERENCE:-false}"
DEFER_METRICS="${DEFER_METRICS:-false}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
VIDEO_READER="${VIDEO_READER:-decord}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-pipeline_${RUN_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/online_pipeline_eval/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR//'${RUN_NAME}'/${RUN_NAME}}"
DRY_RUN="${DRY_RUN:-false}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${HF_CACHE}" "${WANDB_DIR}"

if [[ ! -f "${EGOPER_ROOT}/annotation.json" ]]; then
  echo "Missing EgoPER annotation under EGOPER_ROOT=${EGOPER_ROOT}" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

# One process per GPU, each owning a video-id shard (not torchrun).
PIPELINE_ARGS=(
  "${PYTHON_BIN}" inference/online_pipeline.py
  --dataset "${DATASET}"
  --egoper-root "${EGOPER_ROOT}"
  --model-name "${MODEL_NAME}"
  --max-videos "${MAX_VIDEOS}"
  --val-fraction "${VAL_FRACTION}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames-a "${MAX_FRAMES_A}"
  --max-frames-b "${MAX_FRAMES_B}"
  --max-frames-c "${MAX_FRAMES_C}"
  --vision-resize-a "${VISION_RESIZE_A}"
  --vision-resize-b "${VISION_RESIZE_B}"
  --vision-resize-c "${VISION_RESIZE_C}"
  --max-seq-length-a "${MAX_SEQ_LENGTH_A}"
  --max-seq-length-b "${MAX_SEQ_LENGTH_B}"
  --max-seq-length-c "${MAX_SEQ_LENGTH_C}"
  --module-a-max-new-tokens "${MODULE_A_MAX_NEW_TOKENS}"
  --module-b-max-new-tokens "${MODULE_B_MAX_NEW_TOKENS}"
  --module-c-max-new-tokens "${MODULE_C_MAX_NEW_TOKENS}"
  --module-c-history-max-events "${MODULE_C_HISTORY_MAX_EVENTS}"
  --module-c-clip "${MODULE_C_CLIP}"
  --module-a-label-mode "${MODULE_A_LABEL_MODE}"
  --stride-seconds "${STRIDE_SECONDS}"
  --max-samples "${MAX_SAMPLES}"
  --eval-split "${EVAL_SPLIT}"
  --output-dir "${OUTPUT_DIR}"
)

if [[ -n "${SPLIT_FILE}" ]]; then
  PIPELINE_ARGS+=(--split-file "${SPLIT_FILE}")
fi
if [[ -n "${VAL_VIDEOS_PER_TASK}" ]]; then
  PIPELINE_ARGS+=(--val-videos-per-task "${VAL_VIDEOS_PER_TASK}")
fi
if [[ -n "${TRAIN_VIDEOS_PER_TASK}" ]]; then
  PIPELINE_ARGS+=(--train-videos-per-task "${TRAIN_VIDEOS_PER_TASK}")
fi
expand_output_path() {
  local value="$1"
  value="${value//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
  value="${value//'${RUN_NAME}'/${RUN_NAME}}"
  printf '%s' "${value}"
}

resolve_best_checkpoint() {
  local run_dir="$1"
  local manifest="${run_dir}/best_checkpoints.json"
  if [[ ! -f "${manifest}" ]]; then
    echo "No best_checkpoints.json in ${run_dir}. Finish that module before the pipeline test." >&2
    exit 1
  fi
  "${PYTHON_BIN}" - "${manifest}" << 'PY'
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
path = Path(str(chosen["path"]))
if not path.is_dir():
    raise SystemExit(f"Selected checkpoint is missing: {path}")
print(
    f"selected epoch={chosen.get('epoch')} metric={chosen.get('metric')} path={path}",
    file=sys.stderr,
)
print(path)
PY
}

resolve_if_auto() {
  local checkpoint_var="$1"
  local run_dir_var="$2"
  local checkpoint="${!checkpoint_var:-}"
  if [[ "${checkpoint}" != "auto" ]]; then
    return 0
  fi
  local run_dir selected
  run_dir="$(expand_output_path "${!run_dir_var:-}")"
  selected="$(resolve_best_checkpoint "${run_dir}")"
  printf -v "${checkpoint_var}" '%s' "${selected}"
  export "${checkpoint_var}"
}

resolve_if_auto MODULE_A_CHECKPOINT MODULE_A_RUN_DIR
resolve_if_auto MODULE_B_CHECKPOINT MODULE_B_RUN_DIR
resolve_if_auto MODULE_C_CHECKPOINT MODULE_C_RUN_DIR

if [[ -n "${MODULE_A_CHECKPOINT}" ]]; then
  PIPELINE_ARGS+=(--module-a-checkpoint "${MODULE_A_CHECKPOINT}")
fi
if [[ -n "${MODULE_B_CHECKPOINT}" ]]; then
  PIPELINE_ARGS+=(--module-b-checkpoint "${MODULE_B_CHECKPOINT}")
fi
if [[ -n "${MODULE_C_CHECKPOINT}" ]]; then
  PIPELINE_ARGS+=(--module-c-checkpoint "${MODULE_C_CHECKPOINT}")
fi
if [[ -n "${REUSE_EVENTS_JSON}" ]]; then
  PIPELINE_ARGS+=(--reuse-events-json "${REUSE_EVENTS_JSON}")
fi

case "${LOAD_IN_4BIT}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--load-in-4bit) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-load-in-4bit) ;;
  *) echo "LOAD_IN_4BIT must be true or false, got: ${LOAD_IN_4BIT}" >&2; exit 1 ;;
esac
case "${LOAD_IN_8BIT}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--load-in-8bit) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-load-in-8bit) ;;
  *) echo "LOAD_IN_8BIT must be true or false, got: ${LOAD_IN_8BIT}" >&2; exit 1 ;;
esac
case "${LOAD_IN_16BIT}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--load-in-16bit) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-load-in-16bit) ;;
  *) echo "LOAD_IN_16BIT must be true or false, got: ${LOAD_IN_16BIT}" >&2; exit 1 ;;
esac
case "${REGENERATE_SPLIT}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--regenerate-split) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "REGENERATE_SPLIT must be true or false, got: ${REGENERATE_SPLIT}" >&2; exit 1 ;;
esac
case "${LABEL_SCORE_EVAL}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--label-score-eval) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-label-score-eval) ;;
  *) echo "LABEL_SCORE_EVAL must be true or false, got: ${LABEL_SCORE_EVAL}" >&2; exit 1 ;;
esac
case "${MODULE_C_CANDIDATE_SCORING}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--module-c-candidate-scoring) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-module-c-candidate-scoring) ;;
  *) echo "MODULE_C_CANDIDATE_SCORING must be true or false, got: ${MODULE_C_CANDIDATE_SCORING}" >&2; exit 1 ;;
esac
case "${MODULE_C_SCORE_ONLY}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--module-c-score-only) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-module-c-score-only) ;;
  *) echo "MODULE_C_SCORE_ONLY must be true or false, got: ${MODULE_C_SCORE_ONLY}" >&2; exit 1 ;;
esac
case "${ONLINE_RESET_FROM_B}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--online-reset-from-b) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-online-reset-from-b) ;;
  *) echo "ONLINE_RESET_FROM_B must be true or false, got: ${ONLINE_RESET_FROM_B}" >&2; exit 1 ;;
esac
case "${CAUSAL_STREAM}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--causal-stream --causal-max-videos "${CAUSAL_MAX_VIDEOS}") ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-causal-stream) ;;
  *) echo "CAUSAL_STREAM must be true or false, got: ${CAUSAL_STREAM}" >&2; exit 1 ;;
esac
case "${CAUSAL_MODULE_C_IN_LOOP}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--causal-module-c-in-loop) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-causal-module-c-in-loop) ;;
  *) echo "CAUSAL_MODULE_C_IN_LOOP must be true or false, got: ${CAUSAL_MODULE_C_IN_LOOP}" >&2; exit 1 ;;
esac
case "${ZERO_SHOT_EVAL}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--zero-shot-eval) ;;
  0|false|FALSE|no|NO) PIPELINE_ARGS+=(--no-zero-shot-eval) ;;
  *) echo "ZERO_SHOT_EVAL must be true or false, got: ${ZERO_SHOT_EVAL}" >&2; exit 1 ;;
esac
case "${SKIP_INFERENCE}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--skip-inference) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "SKIP_INFERENCE must be true or false, got: ${SKIP_INFERENCE}" >&2; exit 1 ;;
esac
case "${DEFER_METRICS}" in
  1|true|TRUE|yes|YES) PIPELINE_ARGS+=(--defer-metrics) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "DEFER_METRICS must be true or false, got: ${DEFER_METRICS}" >&2; exit 1 ;;
esac

SHARD_VIDEOS="${SHARD_VIDEOS:-true}"
SHARD_MODE="${SHARD_MODE:-queue}"
WORLD_SIZE="${VIDEO_SHARD_WORLD_SIZE:-${NUM_GPUS:-1}}"
if [[ "${SHARD_MODE}" == "queue" ]]; then
  PIPELINE_ARGS+=(--video-queue-dir "${OUTPUT_DIR}/work_queue")
  PIPELINE_ARGS+=(--video-checkpoint-dir "${OUTPUT_DIR}/videos")
  PIPELINE_ARGS+=(--queue-session-id "${SLURM_JOB_ID:-manual}")
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  WORLD_SIZE="${SLURM_ARRAY_TASK_COUNT:-${WORLD_SIZE}}"
  PIPELINE_ARGS+=(--video-shard-rank "${SLURM_ARRAY_TASK_ID}")
  PIPELINE_ARGS+=(--video-shard-world-size "${WORLD_SIZE}")
fi
case "${SHARD_VIDEOS}" in
  1|true|TRUE|yes|YES)
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      TRAIN_ARGS=("${PIPELINE_ARGS[@]}")
    elif [[ "${WORLD_SIZE}" -gt 1 ]]; then
      TRAIN_ARGS=(
        bash "${PROJECT_DIR}/train/pipeline/run_sharded_pipeline.sh"
        --world-size "${WORLD_SIZE}"
        --output-dir "${OUTPUT_DIR}"
        --
        "${PIPELINE_ARGS[@]}"
      )
    else
      TRAIN_ARGS=("${PIPELINE_ARGS[@]}")
    fi
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS=("${PIPELINE_ARGS[@]}")
    ;;
  *)
    echo "SHARD_VIDEOS must be true or false, got: ${SHARD_VIDEOS}" >&2
    exit 1
    ;;
esac
