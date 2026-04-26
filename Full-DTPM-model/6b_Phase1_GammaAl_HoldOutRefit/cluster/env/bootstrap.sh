#!/usr/bin/env bash
# One-time bootstrap on an NCSA Delta login node.
# Idempotent: safe to re-run; will skip env creation if it already exists.
#
# Run from the repo root after `git clone` and `git checkout` of the feat
# branch. Example:
#   cd $SCRATCH/$USER/dtpm-lxcat/DTPM
#   bash Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/cluster/env/bootstrap.sh

set -euo pipefail

ENV_NAME="dtpm-lxcat"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIXB_ROOT="$(cd "$HERE/../.." && pwd)"                 # 6b_Phase1_GammaAl_HoldOutRefit
REPO_ROOT="$(cd "$SIXB_ROOT/../.." && pwd)"            # DTPM clone root
PHASE2_PATH="$REPO_ROOT/Plasma Chemistry Module/SF6_surrogate_and_LXCat/phase2_electron_kinetics"

echo "==> Repo root:   $REPO_ROOT"
echo "==> 6b root:     $SIXB_ROOT"
echo "==> PHASE2_ROOT: $PHASE2_PATH"

if [[ ! -d "$PHASE2_PATH" ]]; then
  echo "ERROR: PHASE2_ROOT path not found. Did the git clone include the Plasma Chemistry Module folder?" >&2
  exit 1
fi

# --- Modules ---------------------------------------------------------------
# Delta module names drift after maintenance windows. Echo what we load so
# the user can adjust if a `module load` fails on the day. Run
# `module avail anaconda3 cuda` on a fresh shell to confirm names.
echo "==> Loading modules"
module purge
module load anaconda3_gpu        # Delta's anaconda + CUDA-aware libs
module load cuda/12.4.0          # adjust to whatever `module avail cuda` lists
echo "==> Loaded:"
module list 2>&1 | sed 's/^/    /'

# --- Conda env -------------------------------------------------------------
ENV_FILE="$HERE/environment.yml"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> Conda env '$ENV_NAME' already exists; skipping create"
else
  echo "==> Creating conda env '$ENV_NAME' from $ENV_FILE (~5 min)"
  conda env create -f "$ENV_FILE" -n "$ENV_NAME"
fi

# Activate using the 'source' path because `conda activate` requires conda init
# to have run in the current shell.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
echo "==> Active env: $(conda info --envs | awk '/\*/ {print $1}')"

# --- PHASE2_ROOT export ----------------------------------------------------
# Slurm jobs on Delta source ~/.bashrc by default. Persist the export there
# (or .zshenv if the user runs zsh) so sbatch jobs inherit it.
case "${SHELL:-}" in
  *zsh)  RC_FILE="$HOME/.zshenv" ;;
  *)     RC_FILE="$HOME/.bashrc" ;;
esac
EXPORT_LINE="export PHASE2_ROOT=\"$PHASE2_PATH\""
if grep -qF "$EXPORT_LINE" "$RC_FILE" 2>/dev/null; then
  echo "==> PHASE2_ROOT export already in $RC_FILE"
else
  echo "==> Appending PHASE2_ROOT export to $RC_FILE"
  printf '\n# DTPM LXCat pipeline (added by bootstrap.sh)\n%s\n' "$EXPORT_LINE" >> "$RC_FILE"
fi
export PHASE2_ROOT="$PHASE2_PATH"

# --- Smoke test ------------------------------------------------------------
echo "==> Running smoke test"
bash "$SIXB_ROOT/cluster/scripts/smoke_test.sh"

echo
echo "==> Bootstrap complete."
echo "    Next: cd $SIXB_ROOT && bash cluster/scripts/submit_all.sh"
