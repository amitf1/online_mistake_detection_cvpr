# Time-based resumable checkpoints. Source after TRAIN_ARGS exists.
# On a Slurm requeue, RESUME defaults to CKPT_DIR/latest.json.

CKPT_DIR="${CKPT_DIR:-}"
RESUME="${RESUME:-}"
CLUSTER_SAVE_INTERVAL_SECONDS="${CLUSTER_SAVE_INTERVAL_SECONDS:-14400}"
CLUSTER_KEEP_CHECKPOINTS="${CLUSTER_KEEP_CHECKPOINTS:-2}"

if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 && -z "${RESUME}" && -n "${CKPT_DIR}" && -f "${CKPT_DIR}/latest.json" ]]; then
  RESUME="${CKPT_DIR}/latest.json"
  echo "[$(date -Is)] SLURM restart #${SLURM_RESTART_COUNT}: resuming from ${RESUME}"
fi

if [[ -n "${CKPT_DIR}" ]]; then
  mkdir -p "${CKPT_DIR}"
  TRAIN_ARGS+=(--ckpt-dir "${CKPT_DIR}")
  TRAIN_ARGS+=(--cluster-save-interval-seconds "${CLUSTER_SAVE_INTERVAL_SECONDS}")
  TRAIN_ARGS+=(--cluster-keep-checkpoints "${CLUSTER_KEEP_CHECKPOINTS}")
fi

if [[ -n "${RESUME}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi
