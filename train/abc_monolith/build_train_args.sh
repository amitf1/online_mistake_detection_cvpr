# Builds the ABC monolith command from ENV_FILE. Sourced by scripts/run.sh.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

ENV_FILE="${ENV_FILE:-configs/env/abc_monolith.env}"
load_env_defaults "${PROJECT_DIR}/${ENV_FILE}"
[[ -f "${PROJECT_DIR}/.env" ]] && load_env_defaults "${PROJECT_DIR}/.env"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
EGOPER_ROOT="${EGOPER_ROOT:-${PROJECT_DIR}/data/EgoPER}"
HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/wandb}"
RUN_NAME="${RUN_NAME:-abc_monolith_egoper_27b}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/abc_monolith/runs/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/abc_monolith/cluster_checkpoints/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR//'${RUN_NAME}'/${RUN_NAME}}"
CKPT_DIR="${CKPT_DIR//'${OUTPUT_ROOT}'/${OUTPUT_ROOT}}"
CKPT_DIR="${CKPT_DIR//'${RUN_NAME}'/${RUN_NAME}}"

NUM_GPUS="${NUM_GPUS:-8}"
EXPECTED_WORLD_SIZE="${EXPECTED_WORLD_SIZE:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
TARGET_GLOBAL_BATCH_SIZE="${TARGET_GLOBAL_BATCH_SIZE:-16}"
actual_global_batch=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
if (( actual_global_batch != TARGET_GLOBAL_BATCH_SIZE )); then
  echo "Global batch mismatch: ${NUM_GPUS}x${PER_DEVICE_TRAIN_BATCH_SIZE}x${GRADIENT_ACCUMULATION_STEPS}=${actual_global_batch}, expected ${TARGET_GLOBAL_BATCH_SIZE}" >&2
  exit 1
fi
if [[ "${SPLIT_FILE:-}" != "configs/splits/egoper_video_split_seed3407.json" ]]; then
  echo "ABC monolith requires the fixed EgoPER split, got SPLIT_FILE=${SPLIT_FILE:-}" >&2
  exit 1
fi
if [[ ! -f "${EGOPER_ROOT}/annotation.json" ]]; then
  echo "Missing EgoPER annotation under EGOPER_ROOT=${EGOPER_ROOT}" >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
LAUNCHER=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}")

TRAIN_ARGS=(
  "${LAUNCHER[@]}" train/abc_monolith.py
  --dataset egoper
  --egoper-root "${EGOPER_ROOT}"
  --model-name "${MODEL_NAME}"
  --max-videos "${MAX_VIDEOS}"
  --split-file "${SPLIT_FILE}"
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
  --expected-world-size "${EXPECTED_WORLD_SIZE}"
  --train-mode "${TRAIN_MODE}"
  --max-steps "${MAX_STEPS}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --checkpoint-epochs "${CHECKPOINT_EPOCHS}"
  --eval-epochs "${EVAL_EPOCHS}"
  --keep-last-checkpoints "${KEEP_LAST_CHECKPOINTS}"
  --keep-best-checkpoints "${KEEP_BEST_CHECKPOINTS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-threshold "${EARLY_STOPPING_THRESHOLD}"
  --metric-for-best-model "${METRIC_FOR_BEST_MODEL}"
  --wandb-artifact-prefix "${WANDB_ARTIFACT_PREFIX}"
  --fps "${FPS}"
  --min-frames "${MIN_FRAMES}"
  --max-frames "${MAX_FRAMES}"
  --eval-max-frames "${EVAL_MAX_FRAMES}"
  --vision-resize "${VISION_RESIZE}"
  --max-seq-length "${MAX_SEQ_LENGTH}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --eval-generation-max-new-tokens "${EVAL_GENERATION_MAX_NEW_TOKENS}"
  --eval-generation-batch-size "${EVAL_GENERATION_BATCH_SIZE}"
  --history-max-events "${HISTORY_MAX_EVENTS}"
  --history-dropout-probability "${HISTORY_DROPOUT_PROBABILITY}"
  --late-trigger-ratio "${LATE_TRIGGER_RATIO}"
  --max-natural-not-completed-per-episode "${MAX_NATURAL_NOT_COMPLETED_PER_EPISODE}"
  --non-correct-oversample-factor "${NON_CORRECT_OVERSAMPLE_FACTOR}"
  --dataset-num-proc "${DATASET_NUM_PROC:-1}"
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
  --torch-empty-cache-steps "${TORCH_EMPTY_CACHE_STEPS}"
  --ddp-timeout "${DDP_TIMEOUT:-7200}"
  --token-budget-safety-margin "${TOKEN_BUDGET_SAFETY_MARGIN:-256}"
  --output-dir "${OUTPUT_DIR}"
  --report-to "${REPORT_TO}"
)

bool_arg() {
  local value="$1" enabled="$2" disabled="$3" name="$4"
  case "${value}" in
    1|true|TRUE|yes|YES) TRAIN_ARGS+=("${enabled}") ;;
    0|false|FALSE|no|NO) TRAIN_ARGS+=("${disabled}") ;;
    *) echo "${name} must be true or false, got: ${value}" >&2; exit 1 ;;
  esac
}

bool_arg "${LOAD_IN_4BIT}" --load-in-4bit --no-load-in-4bit LOAD_IN_4BIT
bool_arg "${LOAD_IN_8BIT}" --load-in-8bit --no-load-in-8bit LOAD_IN_8BIT
bool_arg "${LOAD_IN_16BIT}" --load-in-16bit --no-load-in-16bit LOAD_IN_16BIT
bool_arg "${FINETUNE_VISION_LAYERS}" --finetune-vision-layers --no-finetune-vision-layers FINETUNE_VISION_LAYERS
bool_arg "${FINETUNE_LANGUAGE_LAYERS}" --finetune-language-layers --no-finetune-language-layers FINETUNE_LANGUAGE_LAYERS
bool_arg "${FINETUNE_ATTENTION_MODULES}" --finetune-attention-modules --no-finetune-attention-modules FINETUNE_ATTENTION_MODULES
bool_arg "${FINETUNE_MLP_MODULES}" --finetune-mlp-modules --no-finetune-mlp-modules FINETUNE_MLP_MODULES
bool_arg "${GROUP_BY_LENGTH}" --group-by-length --no-group-by-length GROUP_BY_LENGTH
bool_arg "${EVAL_GENERATION_USE_CACHE}" --eval-generation-use-cache --no-eval-generation-use-cache EVAL_GENERATION_USE_CACHE
bool_arg "${GENERATION_ENABLE_THINKING}" --generation-enable-thinking --no-generation-enable-thinking GENERATION_ENABLE_THINKING
bool_arg "${EVAL_ON_RESUME}" --eval-on-resume --no-eval-on-resume EVAL_ON_RESUME
bool_arg "${GREATER_IS_BETTER}" --greater-is-better --no-greater-is-better GREATER_IS_BETTER
bool_arg "${WANDB_LOG_BEST_CHECKPOINTS}" --wandb-log-best-checkpoints --no-wandb-log-best-checkpoints WANDB_LOG_BEST_CHECKPOINTS
bool_arg "${MONOLITH_CANDIDATE_SCORING}" --monolith-candidate-scoring --no-monolith-candidate-scoring MONOLITH_CANDIDATE_SCORING
if [[ -n "${LOAD_ADAPTER_FROM:-}" ]]; then
  TRAIN_ARGS+=(--load-adapter-from "${LOAD_ADAPTER_FROM}")
fi
if [[ -n "${NOT_COMPLETED_RATIO:-}" ]]; then
  TRAIN_ARGS+=(--not-completed-ratio "${NOT_COMPLETED_RATIO}")
fi

if [[ -n "${DATALOADER_PREFETCH_FACTOR:-}" ]]; then
  TRAIN_ARGS+=(--dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}")
fi
case "${DRY_RUN:-false}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--dry-run --print-examples "${PRINT_EXAMPLES:-2}") ;;
  0|false|FALSE|no|NO) ;;
  *) echo "DRY_RUN must be true or false, got: ${DRY_RUN}" >&2; exit 1 ;;
esac
case "${VALIDATE_TOKEN_BUDGET:-false}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--validate-token-budget) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "VALIDATE_TOKEN_BUDGET must be true or false, got: ${VALIDATE_TOKEN_BUDGET}" >&2; exit 1 ;;
esac
case "${MEMORY_PREFLIGHT:-false}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--memory-preflight) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "MEMORY_PREFLIGHT must be true or false, got: ${MEMORY_PREFLIGHT}" >&2; exit 1 ;;
esac
case "${MEMORY_PREFLIGHT_ONLY:-false}" in
  1|true|TRUE|yes|YES) TRAIN_ARGS+=(--memory-preflight-only) ;;
  0|false|FALSE|no|NO) ;;
  *) echo "MEMORY_PREFLIGHT_ONLY must be true or false, got: ${MEMORY_PREFLIGHT_ONLY}" >&2; exit 1 ;;
esac
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  TRAIN_ARGS+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

# shellcheck source=../lib/cluster_checkpoint.sh
source "${PROJECT_DIR}/train/lib/cluster_checkpoint.sh"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_DIR}" "${CKPT_DIR}" "${HF_CACHE}" "${WANDB_DIR}"
