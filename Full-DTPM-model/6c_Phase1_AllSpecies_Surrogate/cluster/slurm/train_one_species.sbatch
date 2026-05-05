#!/usr/bin/env bash
#SBATCH --account=bgip-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/cluster_logs/train_species-%j.log
#SBATCH --error=results/cluster_logs/train_species-%j.log
#
# Per-species ensemble trainer.  Submitter must export SPECIES=<name>
# (e.g. SPECIES=nF, ion_ne, Te) via `sbatch --export=SPECIES=...`.

set -euo pipefail

mkdir -p results/cluster_logs

if [[ -z "${SPECIES:-}" ]]; then
  echo "FAIL: SPECIES env var not set." >&2
  exit 2
fi

echo "==> Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds)"
echo "==> SPECIES=$SPECIES"

if [[ ! -e results/.cluster_step0.done ]]; then
  echo "FAIL: results/.cluster_step0.done missing — dataset not built." >&2
  exit 2
fi

export PHASE2_ROOT="/projects/bgip/mae5/DTPM/DTPM_phase1/Plasma Chemistry Module/SF6_surrogate_and_LXCat/phase2_electron_kinetics"

module load miniforge3-python
set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dtpm-lxcat
set -u

export ML_DATASET_MODE=lxcat
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "==> Launching trainer for ${SPECIES}"
srun python3 scripts/ml/train_species.py \
  --species "${SPECIES}" \
  --out results/ml_production_ensemble_all_species

OUT_SUMMARY="results/ml_production_ensemble_all_species/${SPECIES}/summary.json"
if [[ ! -f "$OUT_SUMMARY" ]]; then
  echo "FAIL: $OUT_SUMMARY not produced." >&2
  exit 3
fi

echo "==> Training for ${SPECIES} complete at $(date -Iseconds)"
