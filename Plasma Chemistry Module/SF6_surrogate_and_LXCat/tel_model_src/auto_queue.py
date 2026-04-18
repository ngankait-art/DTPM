"""
Auto-queue: watches for prerequisite completion, then launches next steps.

Dependency chain:
  1. Architecture sweep (E0-E6) finishes → select winner → train 5-model ensemble on MPS
  2. All results collected → write final summary

Polls every 30s. Runs until all work is done.
"""
import os, sys, json, time, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
TASK_DIR = "/private/tmp/claude-501/-Users-kaingan8-Downloads-SF6-unified/b3d385a8-93c6-4fd5-9895-7a3651c930d9/tasks"

ARCH_OUTPUT = os.path.join(TASK_DIR, "b6jzyx93d.output")
RESULTS = os.path.join(REPO, 'results')

# Track what we've launched
launched = set()


def check_file_contains(filepath, marker):
    """Check if a file contains a specific string."""
    if not os.path.exists(filepath):
        return False
    with open(filepath) as f:
        return marker in f.read()


def get_arch_winner():
    """Parse architecture sweep results and return the winning experiment name."""
    table_path = os.path.join(RESULTS, 'lxcat_architecture_upgrade', 'experiment_table.json')
    if not os.path.exists(table_path):
        return None
    with open(table_path) as f:
        data = json.load(f)
    experiments = data.get('experiments', [])
    if not experiments:
        return None
    # Find lowest nF RMSE mean
    best = min(experiments, key=lambda e: e['nF_rmse_mean'])
    return best['name']


def launch_ensemble(winner_name):
    """Launch the production ensemble training for the winning architecture."""
    print(f"\n{'='*60}", flush=True)
    print(f"  AUTO-QUEUE: Launching production ensemble", flush=True)
    print(f"  Winner: {winner_name}", flush=True)
    print(f"{'='*60}", flush=True)

    # Write a config file so the ensemble script knows which architecture to use
    config = {
        'winner': winner_name,
        'selected_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'reason': 'Lowest nF RMSE mean across 3 seeds in architecture sweep',
    }
    config_path = os.path.join(RESULTS, 'lxcat_architecture_upgrade', 'winner_selection.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Launch the ensemble training script
    # It reads the winner from experiment_table.json and instantiates the correct architecture
    script = os.path.join(HERE, 'train_lxcat_v4_ensemble.py')
    proc = subprocess.Popen(
        [sys.executable, script],
        cwd=HERE,
        stdout=open(os.path.join(TASK_DIR, 'auto_ensemble.output'), 'w'),
        stderr=subprocess.STDOUT,
    )
    print(f"  Launched PID={proc.pid}", flush=True)
    return proc


def collect_final_summary():
    """Collect all results into one master summary."""
    print(f"\n{'='*60}", flush=True)
    print(f"  AUTO-QUEUE: Collecting final summary", flush=True)
    print(f"{'='*60}", flush=True)

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'completed_experiments': {},
    }

    # Architecture sweep
    table_path = os.path.join(RESULTS, 'lxcat_architecture_upgrade', 'experiment_table.json')
    if os.path.exists(table_path):
        summary['architecture_sweep'] = json.load(open(table_path))

    # Ablation
    ablation_path = os.path.join(RESULTS, 'ablation_study', 'ablation_results.json')
    if os.path.exists(ablation_path):
        summary['ablation'] = json.load(open(ablation_path))

    # Mesh convergence
    mesh_path = os.path.join(RESULTS, 'mesh_convergence', 'mesh_convergence.json')
    if os.path.exists(mesh_path):
        summary['mesh_convergence'] = json.load(open(mesh_path))

    # Spatial errors
    spatial_path = os.path.join(RESULTS, 'spatial_error_analysis', 'spatial_errors.json')
    if os.path.exists(spatial_path):
        summary['spatial_errors'] = json.load(open(spatial_path))

    # Literature validation
    lit_path = os.path.join(RESULTS, 'literature_validation', 'validation_metrics.json')
    if os.path.exists(lit_path):
        summary['literature_validation'] = json.load(open(lit_path))

    # Speedup
    speed_path = os.path.join(RESULTS, 'speedup_measurement', 'speedup.json')
    if os.path.exists(speed_path):
        summary['speedup'] = json.load(open(speed_path))

    # ML baselines
    ml_path = os.path.join(RESULTS, 'ml_baseline_comparison', 'baseline_comparison.json')
    if os.path.exists(ml_path):
        summary['ml_baselines'] = json.load(open(ml_path))

    # Transfer learning
    tl_path = os.path.join(RESULTS, 'transfer_learning', 'transfer_learning.json')
    if os.path.exists(tl_path):
        summary['transfer_learning'] = json.load(open(tl_path))

    # Mixed physics
    mp_path = os.path.join(RESULTS, 'mixed_physics_training', 'mixed_physics.json')
    if os.path.exists(mp_path):
        summary['mixed_physics'] = json.load(open(mp_path))

    # Te auxiliary
    te_path = os.path.join(RESULTS, 'te_auxiliary_head', 'te_auxiliary.json')
    if os.path.exists(te_path):
        summary['te_auxiliary'] = json.load(open(te_path))

    # PINN failure
    pinn_path = os.path.join(RESULTS, 'pinn_failure_analysis', 'pinn_failure_analysis.json')
    if os.path.exists(pinn_path):
        summary['pinn_failure'] = json.load(open(pinn_path))

    # Production ensemble
    ens_path = os.path.join(RESULTS, 'surrogate_lxcat_v4_arch', 'summary.json')
    if os.path.exists(ens_path):
        summary['production_ensemble'] = json.load(open(ens_path))

    out_path = os.path.join(RESULTS, 'all_experiments_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Written to {out_path}", flush=True)
    return summary


def main():
    print(f"{'='*60}", flush=True)
    print(f"  AUTO-QUEUE: Watching for prerequisite completion", flush=True)
    print(f"  Polling every 30s", flush=True)
    print(f"{'='*60}", flush=True)

    ensemble_proc = None

    while True:
        # Step 1: Architecture sweep done → launch ensemble
        if 'ensemble' not in launched:
            if check_file_contains(ARCH_OUTPUT, "ALL EXPERIMENTS COMPLETE"):
                winner = get_arch_winner()
                if winner:
                    print(f"\n  [TRIGGER] Architecture sweep complete. Winner: {winner}", flush=True)
                    ensemble_proc = launch_ensemble(winner)
                    launched.add('ensemble')
                else:
                    print(f"\n  [WARN] Sweep complete but no winner found", flush=True)

        # Step 2: Ensemble done → collect final summary
        if 'ensemble' in launched and 'final_summary' not in launched:
            ens_summary = os.path.join(RESULTS, 'surrogate_lxcat_v4_arch', 'summary.json')
            if os.path.exists(ens_summary):
                if ensemble_proc and ensemble_proc.poll() is not None:
                    print(f"\n  [TRIGGER] Ensemble training complete.", flush=True)
                    collect_final_summary()
                    launched.add('final_summary')

        # Check if everything is done
        all_done = True

        # Check each expected result file
        expected = [
            ('architecture_sweep', os.path.join(RESULTS, 'lxcat_architecture_upgrade', 'experiment_table.json')),
            ('ablation', os.path.join(RESULTS, 'ablation_study', 'ablation_results.json')),
            ('ml_baselines', os.path.join(RESULTS, 'ml_baseline_comparison', 'baseline_comparison.json')),
            ('transfer_learning', os.path.join(RESULTS, 'transfer_learning', 'transfer_learning.json')),
            ('mixed_physics', os.path.join(RESULTS, 'mixed_physics_training', 'mixed_physics.json')),
            ('te_auxiliary', os.path.join(RESULTS, 'te_auxiliary_head', 'te_auxiliary.json')),
            ('ensemble', os.path.join(RESULTS, 'surrogate_lxcat_v4_arch', 'summary.json')),
        ]

        status = []
        for name, path in expected:
            done = os.path.exists(path)
            status.append(f"  {'[DONE]' if done else '[....]'} {name}")
            if not done:
                all_done = False

        # Print status every poll
        print(f"\n  --- Status @ {time.strftime('%H:%M:%S')} ---", flush=True)
        for s in status:
            print(s, flush=True)

        if all_done and 'final_summary' in launched:
            print(f"\n{'='*60}", flush=True)
            print(f"  ALL WORK COMPLETE", flush=True)
            print(f"{'='*60}", flush=True)
            break

        time.sleep(30)


if __name__ == '__main__':
    main()
