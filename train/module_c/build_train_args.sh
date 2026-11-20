# Builds the Module C command from ENV_FILE. Sourced by scripts/run.sh.
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

ENV_FILE="${ENV_FILE:-configs/env/module_c.env}"
load_env_defaults "${PROJECT_DIR}/${ENV_FILE}"
if [[ -f "${PROJECT_DIR}/.env" ]]; then
  load_env_defaults "${PROJECT_DIR}/.env"
fi

EGOPER_ROOT="${EGOPER_ROOT:-${PROJECT_DIR}/data/EgoPER}"
DATASET="${DATASET:-egoper}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
REPORT_TO="${REPORT_TO:-tensorboard}"
MAX_VIDEOS="${MAX_VIDEOS:-50}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"
LOAD_IN_16BIT="${LOAD_IN_16BIT:-true}"
LORA_DTYPE="${LORA_DTYPE:-bfloat16}"
OPTIM="${OPTIM:-adamw_torch_fused}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-auto}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_GPUS="${NUM_GPUS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(( (TARGET_GLOBAL_BATCH_SIZE + NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE - 1) / (NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE) ))}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
TORCH_EMPTY_CACHE_STEPS="${TORCH_EMPTY_CACHE_STEPS:-1}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"
TRAIN_MODE="${TRAIN_MODE:-steps}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
EVAL_EPOCHS="${EVAL_EPOCHS:-1}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
KEEP_BEST_CHECKPOINTS="${KEEP_BEST_CHECKPOINTS:-4}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-2}"
TRAIN_VIDEOS_PER_TASK="${TRAIN_VIDEOS_PER_TASK:-}"
TRAIN_VALIDATION_ERRONEOUS_ONLY="${TRAIN_VALIDATION_ERRONEOUS_ONLY:-false}"
SPLIT_FILE="${SPLIT_FILE:-}"
REGENERATE_SPLIT="${REGENERATE_SPLIT:-false}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_mistake/f1}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-true}"
WANDB_LOG_BEST_CHECKPOINTS="${WANDB_LOG_BEST_CHECKPOINTS:-false}"
WANDB_ARTIFACT_PREFIX="${WANDB_ARTIFACT_PREFIX:-module-c}"
MODULE_C_MIN_DURATION_SECONDS="${MODULE_C_MIN_DURATION_SECONDS:-0.5}"
MODULE_C_JITTER_RATIO_MIN="${MODULE_C_JITTER_RATIO_MIN:-0.05}"
MODULE_C_JITTER_RATIO_MAX="${MODULE_C_JITTER_RATIO_MAX:-0.10}"
INCLUDE_NOT_COMPLETED_EXAMPLES="${INCLUDE_NOT_COMPLETED_EXAMPLES:-true}"
MODULE_C_NOT_COMPLETED_RATIO="${MODULE_C_NOT_COMPLETED_RATIO:-0.10}"
MODULE_C_NON_CORRECT_OVERSAMPLE_FACTOR="${MODULE_C_NON_CORRECT_OVERSAMPLE_FACTOR:-1.0}"
MODULE_C_HISTORY_MAX_EVENTS="${MODULE_C_HISTORY_MAX_EVENTS:-12}"
MODULE_C_HISTORY_DROPOUT_PROBABILITY="${MODULE_C_HISTORY_DROPOUT_PROBABILITY:-0.25}"
EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:--1}"
EVAL_GENERATION_MAX_NEW_TOKENS="${EVAL_GENERATION_MAX_NEW_TOKENS:-128}"
EVAL_GENERATION_BATCH_SIZE="${EVAL_GENERATION_BATCH_SIZE:-1}"
EVAL_GENERATION_USE_CACHE="${EVAL_GENERATION_USE_CACHE:-true}"
GENERATION_ENABLE_THINKING="${GENERATION_ENABLE_THINKING:-false}"
BERTSCORE_MODEL_TYPE="${BERTSCORE_MODEL_TYPE:-microsoft/deberta-xlarge-mnli}"
BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-16}"
BERTSCORE_DEVICE="${BERTSCORE_DEVICE:-auto}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES="${MAX_FRAMES:-32}"
EVAL_MAX_FRAMES="${EVAL_MAX_FRAMES:-${MAX_FRAMES}}"
VISION_RESIZE="${VISION_RESIZE:-512}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-8192}"
FINETUNE_VISION_LAYERS="${FINETUNE_VISION_LAYERS:-false}"
FINETUNE_LANGUAGE_LAYERS="${FINETUNE_LANGUAGE_LAYERS:-true}"
FINETUNE_ATTENTION_MODULES="${FINETUNE_ATTENTION_MODULES:-true}"
FINETUNE_MLP_MODULES="${FINETUNE_MLP_MODULES:-true}"
VIDEO_READER="${VIDEO_READER:-decord}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-module_c_qwen35_lora_reasoning_${RUN_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/module_c_qwen35_lora_reasoning/runs/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR//'${RUN_NAME}'/${RUN_NAME}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/module_c/cluster_checkpoints/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
CKPT_DIR="${CKPT_DIR//'${RUN_NAME}'/${RUN_NAME}}"
RESUME="${RESUME:-}"
CLUSTER_SAVE_INTERVAL_SECONDS="${CLUSTER_SAVE_INTERVAL_SECONDS:-14400}"
CLUSTER_KEEP_CHECKPOINTS="${CLUSTER_KEEP_CHECKPOINTS:-2}"
DRY_RUN="${DRY_RUN:-false}"
PRINT_EXAMPLES="${PRINT_EXAMPLES:-2}"
ZERO_SHOT_EVAL="${ZERO_SHOT_EVAL:-false}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${HF_CACHE}" "${WANDB_DIR}"

if [[ ! -f "${EGOPER_ROOT}/annotation.json" ]]; then
  echo "Missing EgoPER annotation under EGOPER_ROOT=${EGOPER_ROOT}" >&2
  exit 1
fi

if (( NUM_GPUS < 1 || NUM_GPUS > 8 )); then
  echo "NUM_GPUS must be between 1 and 8, got: ${NUM_GPUS}" >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
# Always use torchrun, including nproc=1. Unsloth/Accelerate call
# init_process_group and require WORLD_SIZE even on a single GPU.
LAUNCHER=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}")

TRAIN_ARGS=(
  "${LAUNCHER[@]}" train/module_c.py
  --dataset "${DATASET}"
  --egoper-root "${EGOPER_ROOT}"
  --model-name "${MODEL_NAME}"
  --lora-dtype "${LORA_DTYPE}"
  --optim "${OPTIM}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-target-modules "${LORA_TARGET_MODULES}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-steps "${WARMUP_STEPS}"
  --weight-decay "${WEIGHT_DECAY}"
  --lr-scheduler-type "${LR_SCHEDULER_TYPE}"
  --max-grad-norm "${MAX_GRAD_NORM}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --torch-empty-cache-steps "${TORCH_EMPTY_CACHE_STEPS}"
  --max-videos "${MAX_VIDEOS}"
  --train-mode "${TRAIN_MODE}"
  --max-steps "${MAX_STEPS}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --checkpoint-epochs "${CHECKPOINT_EPOCHS}"
  --eval-epochs "${EVAL_EPOCHS}"
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}"
  --keep-best-checkpoints "${KEEP_BEST_CHECKPOINTS}"
  --val-fraction "${VAL_FRACTION}"
  --val-videos-per-task "${VAL_VIDEOS_PER_TASK}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD}"
  --metric-for-best-model "${METRIC_FOR_BEST_MODEL}"
  --wandb-artifact-prefix "${WANDB_ARTIFACT_PREFIX}"
  --module-c-min-duration-seconds "${MODULE_C_MIN_DURATION_SECONDS}"
  --module-c-jitter-ratio-min "${MODULE_C_JITTER_RATIO_MIN}"
  --module-c-jitter-ratio-max "${MODULE_C_JITTER_RATIO_MAX}"
  --module-c-not-completed-ratio "${MODULE_C_NOT_COMPLETED_RATIO}"
  --module-c-non-correct-oversample-factor "${MODULE_C_NON_CORRECT_OVERSAMPLE_FACTOR}"
  --module-c-history-max-events "${MODULE_C_HISTORY_MAX_EVENTS}"
  --module-c-history-dropout-probability "${MODULE_C_HISTORY_DROPOUT_PROBABILITY}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --eval-generation-max-new-tokens "${EVAL_GENERATION_MAX_NEW_TOKENS}"
  --eval-generation-batch-size "${EVAL_GENERATION_BATCH_SIZE}"
  --bertscore-model-type "${BERTSCORE_MODEL_TYPE}"
  --bertscore-batch-size "${BERTSCORE_BATCH_SIZE}"
  --bertscore-device "${BERTSCORE_DEVICE}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES}"
  --eval-max-frames "${EVAL_MAX_FRAMES}"
  --vision-resize "${VISION_RESIZE}"
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --output-dir "${OUTPUT_DIR}"
  --report-to "${REPORT_TO}"
)

if [[ -n "${DATALOADER_PREFETCH_FACTOR}" ]]; then
  TRAIN_ARGS+=(--dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}")
fi
if [[ -n "${TRAIN_VIDEOS_PER_TASK}" ]]; then
  TRAIN_ARGS+=(--train-videos-per-task "${TRAIN_VIDEOS_PER_TASK}")
fi
case "${GROUP_BY_LENGTH}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--group-by-length) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-group-by-length) ;;
  *) echo "GROUP_BY_LENGTH must be true or false, got: ${GROUP_BY_LENGTH}" >&2; exit 1 ;;
esac
case "${DDP_FIND_UNUSED_PARAMETERS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--ddp-find-unused-parameters) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-ddp-find-unused-parameters) ;;
  *) echo "DDP_FIND_UNUSED_PARAMETERS must be true or false, got: ${DDP_FIND_UNUSED_PARAMETERS}" >&2; exit 1 ;;
esac
case "${TRAIN_VALIDATION_ERRONEOUS_ONLY}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--train-validation-erroneous-only) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "TRAIN_VALIDATION_ERRONEOUS_ONLY must be true or false, got: ${TRAIN_VALIDATION_ERRONEOUS_ONLY}" >&2; exit 1 ;;
esac

case "${LOAD_IN_4BIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--load-in-4bit) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-load-in-4bit) ;;
  *) echo "LOAD_IN_4BIT must be true or false, got: ${LOAD_IN_4BIT}" >&2; exit 1 ;;
esac
case "${LOAD_IN_8BIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--load-in-8bit) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-load-in-8bit) ;;
  *) echo "LOAD_IN_8BIT must be true or false, got: ${LOAD_IN_8BIT}" >&2; exit 1 ;;
esac
case "${LOAD_IN_16BIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--load-in-16bit) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-load-in-16bit) ;;
  *) echo "LOAD_IN_16BIT must be true or false, got: ${LOAD_IN_16BIT}" >&2; exit 1 ;;
esac
case "${WANDB_LOG_BEST_CHECKPOINTS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--wandb-log-best-checkpoints) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-wandb-log-best-checkpoints) ;;
  *) echo "WANDB_LOG_BEST_CHECKPOINTS must be true or false, got: ${WANDB_LOG_BEST_CHECKPOINTS}" >&2; exit 1 ;;
esac
case "${INCLUDE_NOT_COMPLETED_EXAMPLES}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--include-not-completed-examples) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-include-not-completed-examples) ;;
  *) echo "INCLUDE_NOT_COMPLETED_EXAMPLES must be true or false, got: ${INCLUDE_NOT_COMPLETED_EXAMPLES}" >&2; exit 1 ;;
esac
case "${DRY_RUN}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--dry-run --print-examples "${PRINT_EXAMPLES}") ;;
  0|false|FALSE|no|NO) ;;
  *) echo "DRY_RUN must be true or false, got: ${DRY_RUN}" >&2; exit 1 ;;
esac
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
# shellcheck source=../../lib/cluster_checkpoint.sh
source "${PROJECT_DIR}/train/lib/cluster_checkpoint.sh"
if [[ -n "${SPLIT_FILE}" ]]; then
  TRAIN_ARGS+=(--split-file "${SPLIT_FILE}")
fi
case "${GREATER_IS_BETTER}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--greater-is-better) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-greater-is-better) ;;
  "") ;;
  *) echo "GREATER_IS_BETTER must be true, false, or empty, got: ${GREATER_IS_BETTER}" >&2; exit 1 ;;
esac
case "${REGENERATE_SPLIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--regenerate-split) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "REGENERATE_SPLIT must be true or false, got: ${REGENERATE_SPLIT}" >&2; exit 1 ;;
esac
case "${FINETUNE_VISION_LAYERS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-vision-layers) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-vision-layers) ;;
  *) echo "FINETUNE_VISION_LAYERS must be true or false, got: ${FINETUNE_VISION_LAYERS}" >&2; exit 1 ;;
esac
case "${FINETUNE_LANGUAGE_LAYERS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-language-layers) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-language-layers) ;;
  *) echo "FINETUNE_LANGUAGE_LAYERS must be true or false, got: ${FINETUNE_LANGUAGE_LAYERS}" >&2; exit 1 ;;
esac
case "${FINETUNE_ATTENTION_MODULES}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-attention-modules) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-attention-modules) ;;
  *) echo "FINETUNE_ATTENTION_MODULES must be true or false, got: ${FINETUNE_ATTENTION_MODULES}" >&2; exit 1 ;;
esac
case "${FINETUNE_MLP_MODULES}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--finetune-mlp-modules) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-finetune-mlp-modules) ;;
  *) echo "FINETUNE_MLP_MODULES must be true or false, got: ${FINETUNE_MLP_MODULES}" >&2; exit 1 ;;
esac
case "${ZERO_SHOT_EVAL}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--zero-shot-eval)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-zero-shot-eval)
    ;;
  *)
    echo "ZERO_SHOT_EVAL must be true or false, got: ${ZERO_SHOT_EVAL}" >&2
    exit 1
    ;;
esac
case "${EVAL_GENERATION_USE_CACHE}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--eval-generation-use-cache) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-eval-generation-use-cache) ;;
  *) echo "EVAL_GENERATION_USE_CACHE must be true or false, got: ${EVAL_GENERATION_USE_CACHE}" >&2; exit 1 ;;
esac
case "${GENERATION_ENABLE_THINKING}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--generation-enable-thinking) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-generation-enable-thinking) ;;
  *) echo "GENERATION_ENABLE_THINKING must be true or false, got: ${GENERATION_ENABLE_THINKING}" >&2; exit 1 ;;
esac
case "${GENERATION_EVAL_ONLY:-false}" in
  1|true|TRUE|yes|YES)
    if [[ -z "${EVAL_CHECKPOINT:-}" || ! -d "${EVAL_CHECKPOINT}" ]]; then
      echo "GENERATION_EVAL_ONLY=true needs an existing EVAL_CHECKPOINT, got: ${EVAL_CHECKPOINT:-}" >&2
      exit 1
    fi
    TRAIN_ARGS+=(--generation-eval-only --eval-checkpoint "${EVAL_CHECKPOINT}")
    ;;
  0|false|FALSE|no|NO) ;;
  *) echo "GENERATION_EVAL_ONLY must be true or false, got: ${GENERATION_EVAL_ONLY}" >&2; exit 1 ;;
esac
if [[ -n "${GENERATION_EVAL_OUTPUT_JSON:-}" ]]; then
  TRAIN_ARGS+=(--generation-eval-output-json "${GENERATION_EVAL_OUTPUT_JSON}")
fi
