# NCSA Delta Runbook — LXCat ML Pipeline

Self-contained cluster bundle for the LXCat half of the surrogate-ML
pipeline (the "your half" Z. Ngan handed off in [`docs/notes/SUPERVISOR_ML_HANDOFF.md`](../docs/notes/SUPERVISOR_ML_HANDOFF.md)).
Migrates the ~43 h Mac workload onto Delta in ~14–18 h wall (Mac stays free).

```
cluster/
  README.md                  ← you are here
  env/
    environment.yml          conda spec (python 3.11, pytorch+cuda 12.1)
    bootstrap.sh             one-time setup on a Delta login node
  scripts/
    smoke_test.sh            three pre-submission sanity checks
    submit_all.sh            sbatch all four jobs with dependency wiring
    push_results.sh          commit + push the 3 result trees back to feat
  slurm/
    00_dataset_gen.sbatch    Step 0  — CPU, ~1.5 h
    01_arch_sweep.sbatch     Step 1  — A100, ~4–8 h (12 h cap)
    02_ablation.sbatch       Step 2  — CPU, ~5 h, parallel with Step 1
    03_ensemble.sbatch       Step 3  — A100, ~5 h, depends on Step 1
```

---

## 1. One-time bootstrap

### 1a. SSH to Delta and create the working directory

```bash
ssh $USER@login.delta.ncsa.illinois.edu
mkdir -p /u/mae5/projects/DTPM
cd /u/mae5/projects/DTPM
```

The conda env created by `bootstrap.sh` is ~3 GB; the clone is ~363 MB;
the regenerated dataset is ~200 MB. Confirm the parent filesystem has
enough quota: `quota -s` (or `du -sh /u/mae5` to spot-check current usage).
If the home filesystem is tight, symlink the conda env elsewhere with
`conda config --append envs_dirs $SCRATCH/conda-envs` before running
`bootstrap.sh`.

### 1b. Find your account name

```bash
accounts                 # Delta utility — lists your bxxx-delta-cpu / -gpu allocations
```

You'll have one CPU account and (probably) one GPU account. Note both.

### 1c. Clone the repo as `DTPM_phase1` and check out the feat branch

```bash
git clone https://github.com/ngankait-art/DTPM.git DTPM_phase1
cd DTPM_phase1
git checkout feat/phase1-global-2d-and-sf6ar-chemistry
git pull
```

Final layout: `/u/mae5/projects/DTPM/DTPM_phase1/` is the repo root.

### 1d. Fill in the account placeholders

Every `*.sbatch` file under `cluster/slurm/` has either
`REPLACE_WITH_DELTA_ACCOUNT_CPU` or `REPLACE_WITH_DELTA_ACCOUNT_GPU` on
the `#SBATCH --account=` line. Replace both:

```bash
cd Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit
sed -i "s/REPLACE_WITH_DELTA_ACCOUNT_CPU/<your-cpu-account>/g" cluster/slurm/*.sbatch
sed -i "s/REPLACE_WITH_DELTA_ACCOUNT_GPU/<your-gpu-account>/g" cluster/slurm/*.sbatch
```

Verify nothing's left:

```bash
grep REPLACE_WITH_DELTA_ACCOUNT cluster/slurm/*.sbatch    # should print nothing
```

### 1e. Run the bootstrap

From the repo root:

```bash
bash Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/cluster/env/bootstrap.sh
```

This creates the `dtpm-lxcat` conda env (~5 min), exports `PHASE2_ROOT` to
your shell rc file, and runs the smoke test. Idempotent — safe to re-run.

If `module load anaconda3_gpu` or `module load cuda/12.4.0` fails, run
`module avail anaconda3 cuda` to see the current names on Delta and edit
`cluster/env/bootstrap.sh` accordingly. Module names drift after
maintenance windows.

### 1f. (Optional) Smoke test on a GPU node

The login-node smoke test confirms torch loads but can't see CUDA. To
verify CUDA detection on a real GPU node before submitting an 8-hour job:

```bash
srun --account=<your-gpu-account> --partition=gpuA100x4 \
     --gres=gpu:1 --time=00:10:00 --pty bash
# inside the srun shell:
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dtpm-lxcat
bash Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/cluster/scripts/smoke_test.sh
exit
```

`cuda available: True` and `cuda device 0: NVIDIA A100-SXM4-40GB` (or
similar) confirms the GPU stack is wired up.

---

## 2. Submit the pipeline

```bash
cd Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit
bash cluster/scripts/submit_all.sh
```

This submits all four jobs and writes the IDs to `.last_submission`. The
dependency graph:

```
J0 dataset (CPU, 1.5 h)
   ├── J1 arch sweep (A100, 4–12 h)  ──► J3 ensemble (A100, 5 h)
   └── J2 ablation   (CPU,  5 h, parallel with J1)
```

Total wall: J0 + max(J1, J2) + J3 ≈ **14–18 h** (vs 43 h on Mac).

---

## 3. Live monitoring

```bash
squeue -u $USER                                                   # all your jobs
squeue --jobs=$(tr ' ' ',' < .last_submission)                    # just this pipeline
tail -f results/cluster_logs/01_arch_sweep-<JOBID>.log            # live log of one step
```

The dashboard from the hand-off doc, adapted for Slurm log paths:

```bash
LOG=results/cluster_logs/01_arch_sweep-<JOBID>.log
while true; do
  clear
  echo "═══ $(date '+%H:%M:%S') ═══"
  echo "seeds done: $(grep -cE 'Run [0-9]+: nF=' "$LOG") / 21"
  grep -E "\[OK\]|\[FAIL\]" "$LOG" 2>/dev/null
  tail -3 "$LOG"
  sleep 5
done
```

Each step writes a sentinel file on success (`results/.cluster_step{0,1,2,3}.done`).
Downstream sbatch jobs check those before running.

---

## 4. Push results back to feat branch

After J3 finishes (sentinel `results/.cluster_step3.done` exists):

```bash
cd Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit
bash cluster/scripts/push_results.sh
```

Stages exactly three trees (`ml_arch_sweep_lxcat`, `ml_ablation_lxcat`,
`ml_production_ensemble_lxcat` — total ~17 MB), commits with a
descriptive message, and pushes to `origin/feat/...`. The 200 MB
regenerable `ml_dataset/` tree is explicitly excluded.

---

## 5. Failure modes & requeue

### Step 1 sits in `PD` for >2 h
The `gpuA100x4` queue is congested. Open `cluster/slurm/01_arch_sweep.sbatch`
and `cluster/slurm/03_ensemble.sbatch`, comment out
`#SBATCH --partition=gpuA100x4`, and uncomment the
`##SBATCH --partition=gpuA40x4` line below it. A40 is ~2× slower than
A100 but the queue is much shorter. Cancel and resubmit:

```bash
scancel $(tr ' ' ',' < .last_submission)
bash cluster/scripts/submit_all.sh
```

### A step fails midway
Find the failure in `results/cluster_logs/<step>-<jobid>.log`. The sentinel
for that step won't exist, so downstream jobs will exit 2 (good — no
wasted node hours). Fix, then submit just the failed step + downstream:

```bash
J1=$(sbatch --parsable cluster/slurm/01_arch_sweep.sbatch)
sbatch --dependency=afterok:$J1 cluster/slurm/03_ensemble.sbatch
```

### CUDA fell through to CPU
Step 1 and Step 3 both have a 90-second watchdog that `scancel`s the job
if `cuda available: True` doesn't appear in the log. If that fires, check
`module list` matches the env's CUDA expectation (12.1 in
`environment.yml`) and the `--gres=gpu:a100:1` line is present in the
sbatch.

### `PHASE2_ROOT not set` in a Slurm log
The bootstrap added the export to `~/.bashrc` (or `~/.zshenv`). Slurm
sources it by default, but if the user changed shells or rebuilt their
home, re-run `bootstrap.sh` (idempotent — safe).

---

## 6. After-action

Z. Ngan expects the JSON outputs (per Doc Option B): `experiment_table.json`,
`ablation_results.json`, `ensemble_metrics.json`. Once `push_results.sh`
has pushed, those are visible at:

```
https://github.com/ngankait-art/DTPM/blob/feat/phase1-global-2d-and-sf6ar-chemistry/Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/results/ml_arch_sweep_lxcat/experiment_table.json
```

(adjust the path for ablation / ensemble).

The expected results from the hand-off doc:
- Arch sweep winner: `E3_separate_heads`
- Ablation: bias-init contributes ~80 % of the RMSE reduction

If your results differ materially, ping Z. Ngan with the JSONs before
overwriting his expected-values text in `main.tex`.
