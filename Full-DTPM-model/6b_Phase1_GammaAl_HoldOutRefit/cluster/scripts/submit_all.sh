#!/usr/bin/env bash
# Submits all four Slurm jobs with the right dependency wiring.
# Step 1 (arch_sweep, GPU) and Step 2 (ablation, CPU) run concurrently
# after Step 0 finishes; Step 3 waits for Step 1.
#
# Run from 6b_Phase1_GammaAl_HoldOutRefit/.

set -euo pipefail

if [[ ! -d cluster/slurm ]]; then
  echo "ERROR: run this from the 6b_Phase1_GammaAl_HoldOutRefit/ directory." >&2
  exit 1
fi

# Reject the placeholder accounts before we waste a submission cycle
if grep -q 'REPLACE_WITH_DELTA_ACCOUNT' cluster/slurm/*.sbatch; then
  echo "ERROR: cluster/slurm/*.sbatch still contains REPLACE_WITH_DELTA_ACCOUNT placeholders." >&2
  echo "       Run 'accounts' on Delta to find your account, then sed-replace the placeholders." >&2
  exit 1
fi

J0=$(sbatch --parsable cluster/slurm/00_dataset_gen.sbatch)
echo "Submitted Step 0 (dataset gen):    $J0"

J1=$(sbatch --parsable --dependency=afterok:"$J0" cluster/slurm/01_arch_sweep.sbatch)
echo "Submitted Step 1 (arch sweep):     $J1   (after $J0)"

J2=$(sbatch --parsable --dependency=afterok:"$J0" cluster/slurm/02_ablation.sbatch)
echo "Submitted Step 2 (ablation):       $J2   (after $J0, parallel to $J1)"

J3=$(sbatch --parsable --dependency=afterok:"$J1" cluster/slurm/03_ensemble.sbatch)
echo "Submitted Step 3 (ensemble):       $J3   (after $J1)"

cat <<EOF

Job chain summary:
  J0 dataset → J1 arch sweep → J3 ensemble
            ↘ J2 ablation (parallel with J1)

Monitor:
  squeue -u \$USER
  squeue --jobs=$J0,$J1,$J2,$J3
  tail -f results/cluster_logs/*-\$JOBID.log

Save the IDs:
  echo "$J0 $J1 $J2 $J3" > .last_submission

EOF

echo "$J0 $J1 $J2 $J3" > .last_submission
