#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-ablate
#SBATCH --account=REPLACE_WITH_DELTA_ACCOUNT_CPU
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=results/cluster_logs/02_ablation-%j.log
#SBATCH --error=results/cluster_logs/02_ablation-%j.log
#
# Step 2: 5 configs × 3 seeds = 15 runs via ProcessPoolExecutor (workers=5).
# CPU partition, runs in parallel with Step 1 on the GPU partition.
# Doc: SUPERVISOR_ML_HANDOFF.md "Step 2 — LXCat ablation study".
# Output: results/ml_ablation_lxcat/ablation_results.{json,md}
# Sentinel: results/.cluster_step2.done

set -euo pipefail

mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

# Guard: Step 0 must have completed (NOT Step 1 — ablation is Step 0-dependent only)
if [[ ! -f results/.cluster_step0.done ]]; then
  echo "FAIL: results/.cluster_step0.done missing — Step 0 didn't complete cleanly." >&2
  exit 2
fi

if [[ -z "${PHASE2_ROOT:-}" ]]; then
  echo "FAIL: PHASE2_ROOT not set." >&2
  exit 2
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat

# Critical: doc gotcha #2. Without thread caps, 5 workers × multi-threaded
# BLAS = load average 225 and ~5x slowdown.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ML_DATASET_MODE=lxcat

echo "==> Launching ablation (workers=5, ML_DATASET_MODE=$ML_DATASET_MODE)"
srun python3 scripts/ml/run_ablation_parallel.py --workers 5

if [[ ! -f results/ml_ablation_lxcat/ablation_results.json ]]; then
  echo "FAIL: ablation_results.json not produced." >&2
  exit 3
fi

touch results/.cluster_step2.done
echo "==> Step 2 complete at $(date -Iseconds)"
