# Builds the Module A command from ENV_FILE. Sourced by scripts/run.sh.
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

ENV_FILE="${ENV_FILE:-configs/env/module_a.env}"
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
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:--1}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:--1}"
VIDEO_IDS="${VIDEO_IDS:-}"
MODEL_NAME="${MODEL_NAME:-unsloth/Qwen3.5-2B}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-false}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-false}"
LOAD_IN_16BIT="${LOAD_IN_16BIT:-true}"
LORA_DTYPE="${LORA_DTYPE:-bfloat16}"
OPTIM="${OPTIM:-adamw_torch_fused}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-auto}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_GPUS="${NUM_GPUS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(( (TARGET_GLOBAL_BATCH_SIZE + NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE - 1) / (NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE) ))}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
TORCH_EMPTY_CACHE_STEPS="${TORCH_EMPTY_CACHE_STEPS:-1}"
GROUP_BY_LENGTH="${GROUP_BY_LENGTH:-false}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-}"
TRAIN_MODE="${TRAIN_MODE:-steps}"
MAX_STEPS="${MAX_STEPS:-100}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-2}"
EVAL_EPOCHS="${EVAL_EPOCHS:-2}"
EVAL_ON_RESUME="${EVAL_ON_RESUME:-false}"
KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
KEEP_BEST_CHECKPOINTS="${KEEP_BEST_CHECKPOINTS:-4}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
VAL_VIDEOS_PER_TASK="${VAL_VIDEOS_PER_TASK:-2}"
TRAIN_VIDEOS_PER_TASK="${TRAIN_VIDEOS_PER_TASK:-}"
TRAIN_VALIDATION_ERRONEOUS_ONLY="${TRAIN_VALIDATION_ERRONEOUS_ONLY:-false}"
SPLIT_FILE="${SPLIT_FILE:-}"
REGENERATE_SPLIT="${REGENERATE_SPLIT:-false}"
POSITIVE_OVERSAMPLE_FACTOR="${POSITIVE_OVERSAMPLE_FACTOR:-1.0}"
POSITIVE_LOSS_WEIGHT="${POSITIVE_LOSS_WEIGHT:-2.0}"
LETTER_ONLY_LOSS="${LETTER_ONLY_LOSS:-false}"
LOAD_ADAPTER_FROM="${LOAD_ADAPTER_FROM:-}"
ZERO_SHOT_EVAL="${ZERO_SHOT_EVAL:-false}"
EVAL_ONLY="${EVAL_ONLY:-false}"
WRITE_CALIBRATION_RECORDS="${WRITE_CALIBRATION_RECORDS:-true}"
CALIBRATION_OUTPUT_DIR="${CALIBRATION_OUTPUT_DIR:-}"
MAX_NEGATIVE_WINDOWS_PER_EPISODE="${MAX_NEGATIVE_WINDOWS_PER_EPISODE:-3}"
RESET_JITTER_SECONDS="${RESET_JITTER_SECONDS:-0.5}"
STRIDE_SECONDS="${STRIDE_SECONDS:-5.0}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-3}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_step_id/total_macro_f1}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-}"
LABEL_SCORE_EVAL="${LABEL_SCORE_EVAL:-true}"
FBETA_BETA="${FBETA_BETA:-2.0}"
WANDB_LOG_BEST_CHECKPOINTS="${WANDB_LOG_BEST_CHECKPOINTS:-false}"
WANDB_ARTIFACT_PREFIX="${WANDB_ARTIFACT_PREFIX:-module-a}"
EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:--1}"
EVAL_GENERATION_MAX_NEW_TOKENS="${EVAL_GENERATION_MAX_NEW_TOKENS:-8}"
GENERATION_EVAL_MODE="${GENERATION_EVAL_MODE:-inline}"
MODULE_A_LABEL_MODE="${MODULE_A_LABEL_MODE:-step_id}"
FPS="${FPS:-1.0}"
MIN_FRAMES="${MIN_FRAMES:-2}"
MAX_FRAMES="${MAX_FRAMES:-32}"
VISION_RESIZE="${VISION_RESIZE:-512}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-6144}"
FINETUNE_VISION_LAYERS="${FINETUNE_VISION_LAYERS:-false}"
FINETUNE_LANGUAGE_LAYERS="${FINETUNE_LANGUAGE_LAYERS:-true}"
FINETUNE_ATTENTION_MODULES="${FINETUNE_ATTENTION_MODULES:-true}"
FINETUNE_MLP_MODULES="${FINETUNE_MLP_MODULES:-true}"
VIDEO_READER="${VIDEO_READER:-decord}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-module_a_${RUN_TIMESTAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/module_a/runs/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR//'${RUN_NAME}'/${RUN_NAME}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/module_a/cluster_checkpoints/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
CKPT_DIR="${CKPT_DIR//'${RUN_NAME}'/${RUN_NAME}}"
RESUME="${RESUME:-}"
CLUSTER_SAVE_INTERVAL_SECONDS="${CLUSTER_SAVE_INTERVAL_SECONDS:-14400}"
CLUSTER_KEEP_CHECKPOINTS="${CLUSTER_KEEP_CHECKPOINTS:-2}"
DRY_RUN="${DRY_RUN:-false}"
PRINT_EXAMPLES="${PRINT_EXAMPLES:-2}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${HF_CACHE}" "${WANDB_DIR}"

if [[ ! -f "${EGOPER_ROOT}/annotation.json" ]]; then
  echo "Missing EgoPER annotation under EGOPER_ROOT=${EGOPER_ROOT}" >&2
  exit 1
fi

if (( MAX_FRAMES >= 32 && MAX_SEQ_LENGTH < 4096 )); then
  echo "MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH} is too small for MAX_FRAMES=${MAX_FRAMES}." >&2
  echo "Use MAX_SEQ_LENGTH=6144, or reduce MAX_FRAMES/VISION_RESIZE." >&2
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
  "${LAUNCHER[@]}" train/module_a.py
  --dataset "${DATASET}"
  --egoper-root "${EGOPER_ROOT}"
  --model-name "${MODEL_NAME}"
  --lora-dtype "${LORA_DTYPE}"
  --optim "${OPTIM}"
  --module-a-label-mode "${MODULE_A_LABEL_MODE}"
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
  --max-train-examples "${MAX_TRAIN_EXAMPLES}"
  --max-eval-examples "${MAX_EVAL_EXAMPLES}"
  --train-mode "${TRAIN_MODE}"
  --max-steps "${MAX_STEPS}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --checkpoint-epochs "${CHECKPOINT_EPOCHS}"
  --eval-epochs "${EVAL_EPOCHS}"
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}"
  --keep-best-checkpoints "${KEEP_BEST_CHECKPOINTS}"
  --val-fraction "${VAL_FRACTION}"
  --val-videos-per-task "${VAL_VIDEOS_PER_TASK}"
  --positive-oversample-factor "${POSITIVE_OVERSAMPLE_FACTOR}"
  --positive-loss-weight "${POSITIVE_LOSS_WEIGHT}"
  --max-negative-windows-per-episode "${MAX_NEGATIVE_WINDOWS_PER_EPISODE}"
  --reset-jitter-seconds "${RESET_JITTER_SECONDS}"
  --stride-seconds "${STRIDE_SECONDS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD}"
  --metric-for-best-model "${METRIC_FOR_BEST_MODEL}"
  --fbeta-beta "${FBETA_BETA}"
  --wandb-artifact-prefix "${WANDB_ARTIFACT_PREFIX}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --eval-generation-max-new-tokens "${EVAL_GENERATION_MAX_NEW_TOKENS}"
  --generation-eval-mode "${GENERATION_EVAL_MODE}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES}"
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
case "${TRAIN_VALIDATION_ERRONEOUS_ONLY}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--train-validation-erroneous-only) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "TRAIN_VALIDATION_ERRONEOUS_ONLY must be true or false, got: ${TRAIN_VALIDATION_ERRONEOUS_ONLY}" >&2; exit 1 ;;
esac

case "${LOAD_IN_4BIT}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--load-in-4bit)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-load-in-4bit)
    ;;
  *)
    echo "LOAD_IN_4BIT must be true or false, got: ${LOAD_IN_4BIT}" >&2
    exit 1
    ;;
esac

case "${LOAD_IN_8BIT}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--load-in-8bit) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-load-in-8bit) ;;
  *) echo "LOAD_IN_8BIT must be true or false, got: ${LOAD_IN_8BIT}" >&2; exit 1 ;;
esac

case "${LOAD_IN_16BIT}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--load-in-16bit)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-load-in-16bit)
    ;;
  *)
    echo "LOAD_IN_16BIT must be true or false, got: ${LOAD_IN_16BIT}" >&2
    exit 1
    ;;
esac

case "${WANDB_LOG_BEST_CHECKPOINTS}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--wandb-log-best-checkpoints)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-wandb-log-best-checkpoints)
    ;;
  *)
    echo "WANDB_LOG_BEST_CHECKPOINTS must be true or false, got: ${WANDB_LOG_BEST_CHECKPOINTS}" >&2
    exit 1
    ;;
esac

case "${DRY_RUN}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--dry-run --print-examples "${PRINT_EXAMPLES}")
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "DRY_RUN must be true or false, got: ${DRY_RUN}" >&2
    exit 1
    ;;
esac

case "${LABEL_SCORE_EVAL}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--label-score-eval)
    UNSLOTH_RETURN_LOGITS="0"
    UNSLOTH_RETURN_HIDDEN_STATES="1"
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-label-score-eval)
    ;;
  *)
    echo "LABEL_SCORE_EVAL must be true or false, got: ${LABEL_SCORE_EVAL}" >&2
    exit 1
    ;;
esac

case "${EVAL_ON_RESUME}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--eval-on-resume)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-eval-on-resume)
    ;;
  *)
    echo "EVAL_ON_RESUME must be true or false, got: ${EVAL_ON_RESUME}" >&2
    exit 1
    ;;
esac

if [[ "${POSITIVE_LOSS_WEIGHT}" != "1.0" && "${POSITIVE_LOSS_WEIGHT}" != "1" ]]; then
  UNSLOTH_RETURN_LOGITS="0"
  UNSLOTH_RETURN_HIDDEN_STATES="1"
fi

case "${LETTER_ONLY_LOSS}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--letter-only-loss)
    UNSLOTH_RETURN_LOGITS="0"
    UNSLOTH_RETURN_HIDDEN_STATES="1"
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-letter-only-loss)
    ;;
  *)
    echo "LETTER_ONLY_LOSS must be true or false, got: ${LETTER_ONLY_LOSS}" >&2
    exit 1
    ;;
esac

case "${ZERO_SHOT_EVAL}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--zero-shot-eval)
    UNSLOTH_RETURN_LOGITS="0"
    UNSLOTH_RETURN_HIDDEN_STATES="1"
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-zero-shot-eval)
    ;;
  *)
    echo "ZERO_SHOT_EVAL must be true or false, got: ${ZERO_SHOT_EVAL}" >&2
    exit 1
    ;;
esac

case "${EVAL_ONLY}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--eval-only)
    UNSLOTH_RETURN_LOGITS="0"
    UNSLOTH_RETURN_HIDDEN_STATES="1"
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-eval-only)
    ;;
  *)
    echo "EVAL_ONLY must be true or false, got: ${EVAL_ONLY}" >&2
    exit 1
    ;;
esac

case "${WRITE_CALIBRATION_RECORDS}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--write-calibration-records) ;;
  0|false|FALSE|no|NO) TRAIN_ARGS+=(--no-write-calibration-records) ;;
  *)
    echo "WRITE_CALIBRATION_RECORDS must be true or false, got: ${WRITE_CALIBRATION_RECORDS}" >&2
    exit 1
    ;;
esac

if [[ -n "${CALIBRATION_OUTPUT_DIR}" ]]; then
  TRAIN_ARGS+=(--calibration-output-dir "${CALIBRATION_OUTPUT_DIR}")
fi

if [[ -n "${LOAD_ADAPTER_FROM}" ]]; then
  TRAIN_ARGS+=(--load-adapter-from "${LOAD_ADAPTER_FROM}")
fi
if [[ -n "${GENERATION_EVAL_OUTPUT_JSON:-}" ]]; then
  TRAIN_ARGS+=(--generation-eval-output-json "${GENERATION_EVAL_OUTPUT_JSON}")
fi

UNSLOTH_ENV_ARGS=()
if [[ "${UNSLOTH_RETURN_LOGITS:-}" == "1" || "${UNSLOTH_RETURN_HIDDEN_STATES:-}" == "1" ]]; then
  UNSLOTH_COMPILE_LOCATION="${UNSLOTH_COMPILE_LOCATION:-/tmp/unsloth_compiled_cache_${RUN_NAME}}"
  echo "Using Unsloth compile cache: ${UNSLOTH_COMPILE_LOCATION}"
  UNSLOTH_ENV_ARGS+=(
    UNSLOTH_RETURN_LOGITS="${UNSLOTH_RETURN_LOGITS:-0}"
    UNSLOTH_RETURN_HIDDEN_STATES="${UNSLOTH_RETURN_HIDDEN_STATES:-1}"
    UNSLOTH_COMPILE_LOCATION="${UNSLOTH_COMPILE_LOCATION}"
  )
fi
case "${UNSLOTH_COMPILE_DISABLE:-0}" in
  1|true|TRUE|yes|YES)
    echo "Unsloth torch.compile disabled (UNSLOTH_COMPILE_DISABLE=1)"
    UNSLOTH_ENV_ARGS+=(UNSLOTH_COMPILE_DISABLE=1)
    ;;
esac
if ((${#UNSLOTH_ENV_ARGS[@]})); then
  TRAIN_ARGS=(
    env
    "${UNSLOTH_ENV_ARGS[@]}"
    "${TRAIN_ARGS[@]}"
  )
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
# shellcheck source=../../lib/cluster_checkpoint.sh
source "${PROJECT_DIR}/train/lib/cluster_checkpoint.sh"

if [[ -n "${SPLIT_FILE}" ]]; then
  TRAIN_ARGS+=(--split-file "${SPLIT_FILE}")
fi
if [[ -n "${VIDEO_IDS}" ]]; then
  # shellcheck disable=SC2206
  TRAIN_ARGS+=(--video-ids ${VIDEO_IDS})
fi

case "${GREATER_IS_BETTER}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--greater-is-better)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-greater-is-better)
    ;;
  "")
    ;;
  *)
    echo "GREATER_IS_BETTER must be true, false, or empty, got: ${GREATER_IS_BETTER}" >&2
    exit 1
    ;;
esac

case "${REGENERATE_SPLIT}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--regenerate-split)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "REGENERATE_SPLIT must be true or false, got: ${REGENERATE_SPLIT}" >&2
    exit 1
    ;;
esac

case "${FINETUNE_VISION_LAYERS}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--finetune-vision-layers)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-finetune-vision-layers)
    ;;
  *)
    echo "FINETUNE_VISION_LAYERS must be true or false, got: ${FINETUNE_VISION_LAYERS}" >&2
    exit 1
    ;;
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

case "${GROUP_BY_LENGTH}" in
  1|true|TRUE|yes|YES)
    TRAIN_ARGS+=(--group-by-length)
    ;;
  0|false|FALSE|no|NO)
    TRAIN_ARGS+=(--no-group-by-length)
    ;;
  *)
    echo "GROUP_BY_LENGTH must be true or false, got: ${GROUP_BY_LENGTH}" >&2
    exit 1
    ;;
esac
