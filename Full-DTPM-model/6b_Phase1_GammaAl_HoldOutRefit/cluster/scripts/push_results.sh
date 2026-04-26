#!/usr/bin/env bash
# Push LXCat training results back to feat branch from the Delta login node.
# Whitelists exactly the three result trees the hand-off doc names; refuses
# to add the regenerable 200 MB ml_dataset/ tree.

set -euo pipefail

if [[ ! -d cluster/slurm ]]; then
  echo "ERROR: run from 6b_Phase1_GammaAl_HoldOutRefit/" >&2
  exit 1
fi

# All four sentinels must exist
for s in 0 1 2 3; do
  if [[ ! -f "results/.cluster_step$s.done" ]]; then
    echo "ERROR: results/.cluster_step$s.done missing — Step $s didn't complete." >&2
    exit 1
  fi
done

PATHS=(
  results/ml_arch_sweep_lxcat
  results/ml_ablation_lxcat
  results/ml_production_ensemble_lxcat
)

for p in "${PATHS[@]}"; do
  if [[ ! -d "$p" ]]; then
    echo "ERROR: $p missing — nothing to push." >&2
    exit 1
  fi
done

cd ../..  # back to repo root for git
echo "==> git status before staging:"
git status -s | head -30 || true

# Defensive: never add the regenerable dataset tree, no matter what
GITROOT="$(git rev-parse --show-toplevel)"
DSET="$GITROOT/Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/results/ml_dataset"
if git status --porcelain "$DSET" 2>/dev/null | grep -q .; then
  echo "WARNING: ml_dataset/ has changes; will NOT stage it (200 MB regenerable)."
fi

for p in "${PATHS[@]}"; do
  git add "Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/$p"
done

echo "==> Staged. Spot-check (first 30 lines):"
git status -s | head -30

WHO="${USER:-cluster}"
git commit -m "$(cat <<EOF
LXCat ML pipeline run on NCSA Delta — $WHO

Arch sweep (21 runs), ablation (15 runs), production ensemble (5 models)
trained on regenerated LXCat dataset (220 cases). Pipeline executed via
cluster/slurm/{00,01,02,03}*.sbatch. Per SUPERVISOR_ML_HANDOFF.md the
ml_dataset/ tree is intentionally excluded from this commit (regenerable
in 90 min on 8 cores).
EOF
)"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "==> Pushing to origin/$CURRENT_BRANCH"
git push origin "$CURRENT_BRANCH"

echo "==> Done. git log -1:"
git log -1 --oneline
