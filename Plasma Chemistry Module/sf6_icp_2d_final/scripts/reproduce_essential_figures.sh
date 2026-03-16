#!/bin/bash
# Reproduce all essential figures for the final report.
# Run from the repository root: bash scripts/reproduce_essential_figures.sh

set -e
UNIFIED=code/unified/sf6_icp_unified.py
OUT=outputs/reproduced

echo "Reproducing essential figures..."
echo "================================"

# Gen-5: Pure SF6, 1500W (primary result)
echo "1/4: Gen-5, pure SF6, 1500W..."
python3 $UNIFIED --gen 5 --power 1500 --ar 0.0 --gamma-wall 0.30 --iter 80 --outdir $OUT/gen5_sf6

# Gen-4b: Same conditions for comparison
echo "2/4: Gen-4b, pure SF6, 1500W..."
python3 $UNIFIED --gen 4 --power 1500 --ar 0.0 --gamma-wall 0.30 --iter 80 --outdir $OUT/gen4b_sf6

# Gen-5: 70/30 mix (Mettler conditions)
echo "3/4: Gen-5, 70/30 mix, 1000W..."
python3 $UNIFIED --gen 5 --power 1000 --ar 0.3 --gamma-wall 0.30 --iter 80 --outdir $OUT/gen5_mix

# Gen-5: Pure Ar
echo "4/4: Gen-5, pure Ar, 1000W..."
python3 $UNIFIED --gen 5 --power 1000 --ar 1.0 --gamma-wall 0.01 --iter 60 --outdir $OUT/gen5_ar

echo ""
echo "Done. All outputs in $OUT/"
ls -la $OUT/*/
