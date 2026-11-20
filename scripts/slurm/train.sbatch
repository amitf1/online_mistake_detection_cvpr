#!/bin/bash
# Usage: sbatch scripts/slurm/train.sbatch <module_a|module_b|module_c|abc_monolith> <env file>
# Edit the partition/account lines for your cluster. Requeues resume from CKPT_DIR/latest.json.
#SBATCH --job-name=omd-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
exec bash scripts/run.sh "$1" "$2"
