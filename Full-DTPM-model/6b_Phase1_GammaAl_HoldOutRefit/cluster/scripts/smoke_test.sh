#!/usr/bin/env bash
# Pre-submission smoke test. Three checks. Exits non-zero on any failure.
#
# Run after bootstrap.sh, with the dtpm-lxcat conda env active and
# PHASE2_ROOT exported. Run on a GPU interactive node if you want to verify
# CUDA detection too:
#   srun --partition=gpuA100x4 --gres=gpu:1 --time=00:10:00 --account=$ACCOUNT --pty bash
#   bash Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/cluster/scripts/smoke_test.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIXB_ROOT="$(cd "$HERE/../.." && pwd)"

if [[ -z "${PHASE2_ROOT:-}" ]]; then
  echo "FAIL: PHASE2_ROOT is not set. Run cluster/env/bootstrap.sh first." >&2
  exit 1
fi
if [[ ! -d "$PHASE2_ROOT" ]]; then
  echo "FAIL: PHASE2_ROOT=$PHASE2_ROOT does not exist." >&2
  exit 1
fi

# --- Check 1: torch importable, CUDA visible (if on a GPU node) -----------
echo "==> [1/3] torch import + CUDA detection"
python - <<'PY'
import sys, torch
print(f"  torch version: {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda device count: {torch.cuda.device_count()}")
    print(f"  cuda device 0: {torch.cuda.get_device_name(0)}")
else:
    print("  (no CUDA visible — expected on login node; required on GPU sbatch jobs)")
PY

# --- Check 2: PHASE2_ROOT package importable ------------------------------
# Reproduces the exact smoke test from SUPERVISOR_ML_HANDOFF.md verbatim.
echo "==> [2/3] tier2_pinn import via PHASE2_ROOT"
python - <<'PY'
import os, sys
sys.path.insert(0, os.environ['PHASE2_ROOT'])
from tier2_pinn.get_rates_pinn import get_rates_pinn
print('  IMPORT OK:', get_rates_pinn)
PY

# --- Check 3: 6b codebase loads -------------------------------------------
echo "==> [3/3] 6b codebase imports"
cd "$SIXB_ROOT"
python - <<'PY'
import sys
sys.path.insert(0, 'src')
from dtpm.chemistry import tier2_interface  # noqa: F401
from dtpm.modules import m11_plasma_chemistry  # noqa: F401
print('  6b imports OK')
PY

echo
echo "==> Smoke test PASSED"
