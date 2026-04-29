#!/usr/bin/env bash
#SBATCH --job-name=dtpm-lxcat-ens
#SBATCH --account=bgip-delta-gpu
#SBATCH --partition=gpuA100x4
##SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --output=results/cluster_logs/03_ensemble-%j.log
#SBATCH --error=results/cluster_logs/03_ensemble-%j.log
#
# Step 3: 5-model production ensemble using arch-sweep winner.
# train_ensemble.py reads experiment_table.json from Step 1 and refuses to
# run if missing — the assertion below catches that earlier (saves a node).
# Doc: SUPERVISOR_ML_HANDOFF.md "Step 3 — LXCat production ensemble".
# Output: results/ml_production_ensemble_lxcat/{model_*.pt, ensemble_metrics.json, predictions.npy}
# Sentinel: results/.cluster_step3.done

set -euo pipefail

mkdir -p results/cluster_logs

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"

# Guard: Step 1 must have completed (Step 0 implied — Step 1 already requires it)
if [[ ! -f results/.cluster_step1.done ]]; then
  echo "FAIL: results/.cluster_step1.done missing — Step 1 didn't complete." >&2
  exit 2
fi

TABLE=results/ml_arch_sweep_lxcat/experiment_table.json
if [[ ! -f "$TABLE" ]]; then
  echo "FAIL: $TABLE missing — train_ensemble.py needs the arch-sweep winner table." >&2
  exit 2
fi

export PHASE2_ROOT="/projects/bgip/mae5/DTPM/DTPM_phase1/Plasma Chemistry Module/SF6_surrogate_and_LXCat/phase2_electron_kinetics"

if [[ -z "${PHASE2_ROOT:-}" ]]; then
  echo "FAIL: PHASE2_ROOT not set." >&2
  exit 2
fi

module load miniforge3-python
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat
set -u

export ML_DATASET_MODE=lxcat

echo "==> CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
python -c "import torch; print(f'  cuda available: {torch.cuda.is_available()}')"

# Same defensive cuda watchdog as Step 1
LOG=results/cluster_logs/03_ensemble-${SLURM_JOB_ID}.log
(
  sleep 90
  if ! grep -qE 'Device:\s*cuda|cuda available: True' "$LOG" 2>/dev/null; then
    echo "FAIL: train_ensemble did not pick CUDA after 90 s — cancelling." >> "$LOG"
    scancel "$SLURM_JOB_ID"
  fi
) &

echo "==> Launching ensemble training"
srun python3 scripts/ml/train_ensemble.py

OUT=results/ml_production_ensemble_lxcat
N_MODELS=$(find "$OUT" -name 'model_*.pt' 2>/dev/null | wc -l | tr -d ' ')
echo "==> Models written: $N_MODELS"
if [[ "$N_MODELS" -lt 5 ]]; then
  echo "FAIL: expected >=5 model_*.pt, got $N_MODELS" >&2
  exit 3
fi

touch results/.cluster_step3.done
echo "==> Step 3 complete at $(date -Iseconds)"
echo "==> All steps done. Run cluster/scripts/push_results.sh to sync to feat branch."
