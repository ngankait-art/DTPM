#!/usr/bin/env python3
"""Recover experiment_table.json from a partially-completed arch_sweep _run.log.

The serial arch_sweep.py only writes experiment_table.json at the very end
of main(), so a crash mid-sweep loses the JSON even though all per-seed
metrics were printed to the log. This script parses the log and builds the
JSON for whichever experiments completed (those with a `=== <name>:
nF=... ===` summary line).

Used after the 2026-04-25 crash that killed E6 mid seed 0 — E0..E5 had
all finished cleanly and their metrics are recovered byte-for-byte.

Usage:
    python recover_experiment_table.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

MODE = os.environ.get('ML_DATASET_MODE', 'legacy')
OUT_DIR = os.path.join(REPO, 'results', f'ml_arch_sweep_{MODE}')
LOG_PATH = os.path.join(OUT_DIR, '_run.log')


# Static config matching arch_sweep.py main()
EXP_META = {
    'E0_baseline':          {'n_epochs': 1500, 'use_physics_reg': False, 'use_huber': False, 'enhanced': False},
    'E1_v4_recipe':         {'n_epochs': 2000, 'use_physics_reg': True,  'use_huber': False, 'enhanced': False},
    'E2_wider_deeper':      {'n_epochs': 2000, 'use_physics_reg': True,  'use_huber': False, 'enhanced': False},
    'E3_separate_heads':    {'n_epochs': 2000, 'use_physics_reg': True,  'use_huber': False, 'enhanced': False},
    'E4_residual':          {'n_epochs': 2000, 'use_physics_reg': True,  'use_huber': False, 'enhanced': False},
    'E5_enhanced_features': {'n_epochs': 2000, 'use_physics_reg': True,  'use_huber': False, 'enhanced': True},
    'E6_huber_loss':        {'n_epochs': 2000, 'use_physics_reg': True,  'use_huber': True,  'enhanced': False},
}


def parse_log(path):
    """Returns list of completed experiment summaries in log order."""
    with open(path) as f:
        text = f.read()

    summary_pat = re.compile(
        r"=== (?P<name>E\d_\w+): nF=(?P<nF>[\d.]+)\+/-(?P<nF_std>[\d.]+)\s+"
        r"nSF6=(?P<nSF6>[\d.]+)\+/-(?P<nSF6_std>[\d.]+)\s+\((?P<time>\d+)s\) ==="
    )
    block_pat = re.compile(r"=== EXPERIMENT: (?P<name>E\d_\w+) ===")

    # Slice the log into per-experiment blocks
    blocks = {}
    last_name = None
    last_start = None
    for m in block_pat.finditer(text):
        if last_name is not None:
            blocks[last_name] = text[last_start:m.start()]
        last_name = m.group('name')
        last_start = m.start()
    if last_name is not None:
        blocks[last_name] = text[last_start:]

    completed = []
    for sm in summary_pat.finditer(text):
        name = sm.group('name')
        block = blocks.get(name, '')
        run_pat = re.compile(r"Run (\d+): nF=([\d.]+)\s+nSF6=([\d.]+)")
        runs = []
        for rm in run_pat.finditer(block):
            runs.append({
                'seed_idx': int(rm.group(1)),
                'nF': {'rmse': float(rm.group(2))},
                'nSF6': {'rmse': float(rm.group(3))},
            })
        meta = EXP_META.get(name, {})
        completed.append({
            'name': name,
            'nF_rmse_mean': float(sm.group('nF')),
            'nF_rmse_std': float(sm.group('nF_std')),
            'nSF6_rmse_mean': float(sm.group('nSF6')),
            'nSF6_rmse_std': float(sm.group('nSF6_std')),
            'time_s': int(sm.group('time')),
            'n_runs': len(runs),
            'runs': runs,
            **meta,
        })
    return completed


def main():
    if not os.path.exists(LOG_PATH):
        print(f"ERROR: log not found at {LOG_PATH}", file=sys.stderr)
        sys.exit(1)

    experiments = parse_log(LOG_PATH)
    if not experiments:
        print("ERROR: no completed experiments parsed from log", file=sys.stderr)
        sys.exit(1)

    # Same schema as arch_sweep.py main() emits, plus winner_override metadata
    table = {
        'experiments': experiments,
        'reference': {
            'surrogate_lxcat_v3': {'nF_rmse': 0.0112, 'nSF6_rmse': 0.0081},
            'surrogate_v4':       {'nF_rmse': 0.0029, 'nSF6_rmse': 0.0027},
        },
        'recovered_from_log': True,
        'log_path': LOG_PATH,
        'note': 'E6_huber_loss skipped — process died during E6 seed 0 (cause unknown). '
                'E6 was redundant per main.pdf finding that Huber offered no advantage over MSE.',
        'winner_override': 'E3_separate_heads',
        'winner_override_reason': (
            'E4_residual has marginally lower mean (0.00530 vs 0.00542, 2% gap) but 35% '
            'larger seed variance (sigma 0.00094 vs 0.00069). E3 chosen for stability, '
            'matching main.pdf v4 precedent. Mean gap is within both 1-sigma bands.'
        ),
        'mode': MODE,
    }

    out_path = os.path.join(OUT_DIR, 'experiment_table.json')
    with open(out_path, 'w') as f:
        json.dump(table, f, indent=2)

    # Markdown summary mirroring arch_sweep.py main()
    md_lines = [
        f"# Architecture Sweep — Recovered Results ({MODE})",
        "",
        f"Recovered from `{os.path.basename(LOG_PATH)}` after partial-completion crash.",
        "",
        "| Experiment | Epochs | Phys Reg | nF RMSE | nSF6 RMSE | Wall |",
        "|---|---|---|---|---|---|",
    ]
    for e in experiments:
        md_lines.append(
            f"| {e['name']} | {e['n_epochs']} | "
            f"{'Y' if e['use_physics_reg'] else 'N'} | "
            f"{e['nF_rmse_mean']:.5f}+/-{e['nF_rmse_std']:.5f} | "
            f"{e['nSF6_rmse_mean']:.5f}+/-{e['nSF6_rmse_std']:.5f} | "
            f"{e['time_s']/3600:.1f} h |"
        )
    md_lines.append("")
    winner = min(experiments, key=lambda e: e['nF_rmse_mean'])
    md_lines.append(f"**Winner**: `{winner['name']}` (nF RMSE = {winner['nF_rmse_mean']:.5f})")
    md_lines.append("")
    md_lines.append("E6_huber_loss skipped — see note in JSON.")
    md_path = os.path.join(OUT_DIR, 'experiment_table.md')
    with open(md_path, 'w') as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Recovered {len(experiments)} experiments")
    for e in experiments:
        print(f"  {e['name']:<24s} nF={e['nF_rmse_mean']:.5f}+/-{e['nF_rmse_std']:.5f}  ({e['n_runs']} seeds)")
    print(f"\n  Min-mean candidate: {winner['name']} (nF={winner['nF_rmse_mean']:.5f})")
    if 'winner_override' in table:
        ov_name = table['winner_override']
        ov = next(e for e in experiments if e['name'] == ov_name)
        print(f"  Winner override:    {ov_name} (nF={ov['nF_rmse_mean']:.5f}) — see winner_override_reason in JSON")
    print(f"  JSON:   {out_path}")
    print(f"  MD:     {md_path}")


if __name__ == '__main__':
    main()
