#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-ablate-missing
#SBATCH --account=bgip-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/cluster_logs/02b_ablation_missing-%j.log
#SBATCH --error=results/cluster_logs/02b_ablation_missing-%j.log
#
# Recovery: J2 (17891232) timed out after completing none/bias_only/epochs_only.
# This sbatch runs ONLY the two missing configs (reg_only, all_three) by
# monkey-patching EXPERIMENTS in run_ablation_parallel before main() reads it.
# Output goes to results/ml_ablation_lxcat_missing/ to avoid clobbering anything.

set -euo pipefail

mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

if [[ ! -f results/.cluster_step0.done ]]; then
  echo "FAIL: results/.cluster_step0.done missing — Step 0 didn't complete." >&2
  exit 2
fi

export PHASE2_ROOT="/projects/bgip/mae5/DTPM/DTPM_phase1/Plasma Chemistry Module/SF6_surrogate_and_LXCat/phase2_electron_kinetics"

module load miniforge3-python
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ML_DATASET_MODE=lxcat

echo "==> Launching ablation (configs=reg_only,all_three, workers=2)"
srun python3 -c "
import sys, os
sys.path.insert(0, 'scripts/ml')
import ablation_study as abl
abl.OUT = os.path.join(abl.REPO, 'results', 'ml_ablation_lxcat_missing')
os.makedirs(abl.OUT, exist_ok=True)
import run_ablation_parallel as rap
rap.EXPERIMENTS = [e for e in rap.EXPERIMENTS if e['name'] in ('reg_only', 'all_three')]
sys.argv = ['rap', '--workers', '2']
rap.main()
"

OUT_JSON=results/ml_ablation_lxcat_missing/ablation_results.json
if [[ ! -f "$OUT_JSON" ]]; then
  echo "FAIL: $OUT_JSON not produced." >&2
  exit 3
fi

echo "==> Missing-configs ablation complete at $(date -Iseconds)"
