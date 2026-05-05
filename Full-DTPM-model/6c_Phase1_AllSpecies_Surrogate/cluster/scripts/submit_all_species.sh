#!/usr/bin/env bash
# Submit one sbatch per species. All independent; no dependency chain.
#
# Run from 6c_Phase1_AllSpecies_Surrogate/.
set -euo pipefail

if [[ ! -d cluster/slurm ]]; then
  echo "ERROR: run from 6c_Phase1_AllSpecies_Surrogate/" >&2
  exit 1
fi

NEUTRALS=(nSF6 nSF5 nSF4 nSF3 nSF2 nSF nF nF2 nS)
CHARGED=(ion_ne ion_n+ ion_n- ion_F+ ion_F- ion_SF3+ ion_SF4+ ion_SF4-
         ion_SF5+ ion_SF5- ion_SF6-)
EXTRA=(Te)

ALL=("${NEUTRALS[@]}" "${CHARGED[@]}" "${EXTRA[@]}")
echo "Submitting ${#ALL[@]} per-species training jobs…"

declare -a IDS
for sp in "${ALL[@]}"; do
  if [[ -f "results/ml_production_ensemble_all_species/${sp}/summary.json" ]]; then
    echo "  [skip] ${sp}: summary.json already exists"
    continue
  fi
  jid=$(sbatch --parsable --export=ALL,SPECIES="$sp" \
    --job-name="dtpm-train-${sp}" \
    cluster/slurm/train_one_species.sbatch)
  echo "  Submitted ${sp}: jobid=${jid}"
  IDS+=("$jid")
done

if (( ${#IDS[@]} > 0 )); then
  echo
  echo "Submitted ${#IDS[@]} jobs total."
  echo "Job IDs: ${IDS[*]}"
  printf '%s\n' "${IDS[@]}" > .last_submission
  echo
  echo "Monitor:"
  echo "  squeue -u \$USER --format='%.10i %.30j %.8T %.10M %.16R'"
  echo "  ls results/ml_production_ensemble_all_species/"
fi
