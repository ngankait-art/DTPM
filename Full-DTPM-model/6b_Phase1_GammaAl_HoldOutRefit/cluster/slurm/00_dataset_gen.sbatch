#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-data
#SBATCH --account=REPLACE_WITH_DELTA_ACCOUNT_CPU
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=results/cluster_logs/00_dataset_gen-%j.log
#SBATCH --error=results/cluster_logs/00_dataset_gen-%j.log
#
# Step 0: Regenerate the 220-case LXCat dataset (~90 min on 8 cores).
# Doc: SUPERVISOR_ML_HANDOFF.md "Step 0".
# Output: results/ml_dataset/lxcat/<case>/{summary.json, *.npy} (~200 MB total)
# Sentinel: results/.cluster_step0.done

set -euo pipefail

# Submit from 6b_Phase1_GammaAl_HoldOutRefit/.  Slurm cd's there because of
# the relative paths in --output above.
mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"
echo "==> CWD: $(pwd)"
echo "==> PHASE2_ROOT: ${PHASE2_ROOT:-<UNSET>}"

if [[ -z "${PHASE2_ROOT:-}" ]]; then
  echo "FAIL: PHASE2_ROOT not set. Did bootstrap.sh write it to ~/.bashrc?" >&2
  exit 2
fi

# Activate the conda env (Slurm doesn't auto-activate)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat

# Thread caps prevent BLAS oversubscription within the worker pool
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "==> Launching dataset generation"
srun python3 scripts/run_ml_dataset_generation.py --mode lxcat

# Verify the doc's sentinel: 220 summary.json files
COUNT=$(find results/ml_dataset/lxcat -name summary.json 2>/dev/null | wc -l | tr -d ' ')
echo "==> Generated $COUNT / 220 summary.json files"
if [[ "$COUNT" -ne 220 ]]; then
  echo "FAIL: expected 220 cases, got $COUNT" >&2
  exit 3
fi

touch results/.cluster_step0.done
echo "==> Step 0 complete at $(date -Iseconds)"
