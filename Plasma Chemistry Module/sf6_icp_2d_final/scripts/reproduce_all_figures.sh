#!/bin/bash
# Reproduce the full figure set including parameter sweeps.
# Run from repository root: bash scripts/reproduce_all_figures.sh

set -e
UNIFIED=code/unified/sf6_icp_unified.py
OUT=outputs/full_reproduction

echo "Full figure reproduction..."

# Essential figures first
bash scripts/reproduce_essential_figures.sh

# Parameter scan
echo "Running parameter scan..."
python3 $UNIFIED --scan --outdir $OUT/scan

echo "Done. All outputs in $OUT/"
