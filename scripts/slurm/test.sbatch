#!/bin/bash
# Usage: sbatch scripts/slurm/test.sbatch <pipeline_test|monolith_test> <env file>
# Resubmit the same command until work_queue/progress.json reports complete=true.
#SBATCH --job-name=omd-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
exec bash scripts/run.sh "$1" "$2"
