# ML Pipeline Hand-off — LXCat Half

**Author:** Z. Ngan
**Date:** 2026-04-24
**Branch:** `feat/phase1-global-2d-and-sf6ar-chemistry`
**Folder:** `Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/`

---

## Context

Per your 2026-04-22 message, I'm rerunning the full surrogate ML pipeline against the 6b code (with the BC map fix and γ_Al refit). The pipeline has four stages, each split into a **legacy** Maxwellian-Arrhenius half and an **LXCat** Boltzmann-PINN half:

| Stage | Legacy (running on my Mac) | LXCat (your half — this doc) |
|---|---|---|
| Dataset generation | ✓ done — 220/220, manifest 2026-04-22 | ✓ done by me — 220/220, manifest 2026-04-23 |
| Architecture sweep (E0–E6, 3 seeds each = 21 runs) | ▶ running on MPS, ~32 h ETA | **YOUR JOB ❶** |
| Ablation study (5 configs × 3 seeds = 15 runs) | queued to run after my arch sweep | **YOUR JOB ❷** |
| Production ensemble (5-model) | queued | **YOUR JOB ❸** |

Splitting the LXCat half to your machine cuts wall-clock by roughly half (3 days → ~1.5 days end-to-end). You don't need to redo the LXCat dataset generation — the data is already on the repo, just regenerate it locally if you don't want to transfer the .npy files (it's 220 cases × ~25 fields × 32 KB ≈ 200 MB, faster to regenerate at ~93 min on 8 CPU cores than to download).

---

## Setup

### 1. Clone / pull
```bash
git clone https://github.com/ngankait-art/DTPM.git
cd DTPM
git checkout feat/phase1-global-2d-and-sf6ar-chemistry
git pull
```

### 2. Python environment
Python 3.11+ (3.14 confirmed working). PyTorch (CPU or MPS), NumPy, PyYAML.
```bash
pip install torch numpy pyyaml scipy matplotlib
```

### 3. The PINN cross-section package — set `PHASE2_ROOT`
The LXCat solver uses the Boltzmann-PINN at `Plasma Chemistry Module/SF6_surrogate_and_LXCat/phase2_electron_kinetics/`. The auto-loader has a path bug for fresh clones — please set the env var explicitly:

```bash
# from the DTPM repo root
export PHASE2_ROOT="$(pwd)/Plasma Chemistry Module/SF6_surrogate_and_LXCat/phase2_electron_kinetics"
```

Quick smoke test:
```bash
python -c "
import os, sys
sys.path.insert(0, os.environ['PHASE2_ROOT'])
from tier2_pinn.get_rates_pinn import get_rates_pinn
print('IMPORT OK:', get_rates_pinn)
"
```

### 4. Navigate to 6b
```bash
cd Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit
```
All commands below run from this folder.

---

## Step 0 — Regenerate the LXCat dataset (~90 min, 8 CPU cores)

The dataset is the input to the arch sweep / ablation / ensemble. It's not committed to git (too big), so you'll need to regenerate locally. Same code I ran, deterministic outputs:

```bash
PHASE2_ROOT="$PHASE2_ROOT" nohup python3 scripts/run_ml_dataset_generation.py --mode lxcat \
    > results/ml_dataset_lxcat_run.log 2>&1 &
```

- 220 operating points: 11 P_rf × 5 pressures × 4 x_Ar at fixed γ_Al=0.18, R_coil=0.8, bias ON @ 200 W
- Output: `results/ml_dataset/lxcat/<case_id>/{summary.json, *.npy}` + `dataset_manifest.json`
- Verify when done: `find results/ml_dataset/lxcat -name summary.json | wc -l` should print **220**

---

## Step 1 — LXCat architecture sweep (~32 h on MPS, longer on CPU)

7 architectures × 3 seeds = 21 training runs. Each MLP is small (≤500 K params).

**Best path (Apple Silicon Mac with MPS):**
```bash
ML_DATASET_MODE=lxcat caffeinate -i -s nohup python3 scripts/ml/arch_sweep.py \
    > results/ml_arch_sweep_lxcat/_run.log 2>&1 &
```
- `caffeinate -i -s` keeps the Mac awake while plugged in
- The `Device: mps (MPS)` line should appear in the log within seconds

**Alternative path (CUDA GPU):** same command — `select_device()` auto-prefers CUDA over MPS.

**Alternative path (CPU only, no GPU):** use the parallel wrapper and **cap BLAS threads** to avoid oversubscription:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
ML_DATASET_MODE=lxcat nohup python3 scripts/ml/run_arch_sweep_parallel.py --workers 7 \
    > results/ml_arch_sweep_lxcat/_run.log 2>&1 &
```
Without the thread cap I saw load average of 225+ on an 8-core Mac and ~5× slowdown — please don't skip this.

**Output:** `results/ml_arch_sweep_lxcat/experiment_table.json` (final summary) and `experiment_table.md` (human-readable).
**Per-experiment intermediate:** `results/ml_arch_sweep_lxcat/<exp_name>/run_*.json` (per-seed metrics).

**Expected winner:** `E3_separate_heads` (matched main.pdf v4 result on the prior data).

---

## Step 2 — LXCat ablation study (~5 h on CPU)

Single-factor ablation: bias-init / physics-reg / epochs. 5 configs × 3 seeds.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
ML_DATASET_MODE=lxcat nohup python3 scripts/ml/run_ablation_parallel.py --workers 5 \
    > results/ml_ablation_lxcat/_run.log 2>&1 &
```

**Output:** `results/ml_ablation_lxcat/ablation_results.{json,md}`

**Expected:** bias-init ≈ 80% of the RMSE reduction (matches main.pdf finding).

---

## Step 3 — LXCat production ensemble (~5 h on MPS)

5-model ensemble using the arch-sweep winner. Reads `experiment_table.json` from Step 1, refuses to run if it doesn't exist.

```bash
ML_DATASET_MODE=lxcat caffeinate -i -s nohup python3 scripts/ml/train_ensemble.py \
    > results/ml_production_ensemble_lxcat/_run.log 2>&1 &
```

**Output:** `results/ml_production_ensemble_lxcat/{model_*.pt, ensemble_metrics.json, predictions.npy}`

---

## Sharing results back

Two options, your call:

**(A) Push commits to the same branch** — simplest, also gives me the training output for the report:
```bash
# After all three steps finish
git add results/ml_arch_sweep_lxcat results/ml_ablation_lxcat results/ml_production_ensemble_lxcat
git status                              # spot-check, nothing big should sneak in
# Don't commit results/ml_dataset/ — it's regenerable
git commit -m "LXCat ML pipeline run — JAK"
git push origin feat/phase1-global-2d-and-sf6ar-chemistry
```
Sizes: arch sweep ~5 MB, ablation ~2 MB, ensemble (5 model checkpoints) ~10 MB. Comfortable for git.

**(B) Just send me the JSONs** — `experiment_table.json`, `ablation_results.json`, `ensemble_metrics.json`. ~150 KB total. Email or any IM works.

---

## Timing summary

| Step | Wall time | Hardware |
|---|---|---|
| Dataset gen | 90 min | 8 CPU cores |
| Arch sweep | 32 h | MPS (best) / CUDA / CPU+threadcap |
| Ablation | 5 h | CPU + threadcap |
| Ensemble | 5 h | MPS / CUDA / CPU |
| **Total** | **~43 h** wall (~32 h if you skip dataset regen and grab my .npy files) | |

The arch sweep dominates, and the ablation + ensemble can run sequentially on CPU after MPS finishes. If you can run them concurrently (MPS arch sweep + CPU ablation), shave ~5 h with the thread cap on the CPU side.

---

## Known gotchas (from my runs)

1. **`/tmp` clears on macOS reboot.** Logs go in the project tree (`results/.../_run.log`), not `/tmp/`.
2. **Concurrent MPS + CPU PyTorch jobs without thread caps will thrash.** Always set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on CPU workers when MPS is also running.
3. **`tier2_pinn` import fails without `PHASE2_ROOT`.** The auto-discovered path is wrong for fresh clones.
4. **`save_sweep_point` writes V_peak/V_rms/P_abs as 0.0** — known bug from the D3 sweep, doesn't affect ML targets (nF, nSF6) but worth fixing in m11_plasma_chemistry.py before any future operating-voltage analysis.

Ping me with any errors and I'll debug. The dashboard command for live progress on any of the three steps:

```bash
LOG=results/ml_arch_sweep_lxcat/_run.log    # or whichever step
while true; do
  clear
  echo "═══ $(date '+%H:%M:%S') ═══"
  echo "seeds done: $(grep -cE 'Run [0-9]+: nF=' "$LOG") / 21"
  grep -E "\[OK\]|\[FAIL\]" "$LOG" 2>/dev/null
  tail -3 "$LOG"
  sleep 5
done
```
