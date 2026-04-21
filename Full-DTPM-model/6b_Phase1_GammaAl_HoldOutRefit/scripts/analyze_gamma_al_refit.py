#!/usr/bin/env python3
"""D6 Stage A analyzer: gamma_Al hold-out refit decision gate.

Reads results/gamma_al_sweep/gamma_al_sweep_summary.json and applies the
hard-coded acceptance gate (no human-in-the-loop re-tuning):

  PASS   — all 3 held-out points meet |residual_F| <= 10% AND
           |residual_drop| <= 5 pp AND gamma_Al* in [0.01, 0.40].
  PARTIAL — magnitude tolerance met on all 3 held-out AND F-drop tolerance
            fails on at least 1.
  FAIL   — otherwise.

Emits:
  - stdout table with residuals for every held-out condition.
  - fig_gamma_al_scan.pdf (four subpanels: gamma_Al vs [F]_c + target line).
  - The decision_token + structured result as a JSON file for the D6 memo.

Anti-bias guardrails (enforced as assertions):
  - LITERATURE_BOUNDS = (0.01, 0.40) — gamma_Al* outside -> raises, FAIL.
  - Both [F]_c AND F-drop columns printed for every held-out condition.
  - Decision token printed to stdout BEFORE the residual table, so the
    decision is committed before the tempting numbers are shown.
"""
import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SWEEP_DIR = os.path.join(PROJECT_ROOT, 'results', 'gamma_al_sweep')
SUMMARY_PATH = os.path.join(SWEEP_DIR, 'gamma_al_sweep_summary.json')

FIT_POINT = '90pct_SF6_bias_on'
HELD_OUT = ['90pct_SF6_bias_off', '30pct_SF6_bias_on', '30pct_SF6_bias_off']
LITERATURE_BOUNDS = (0.01, 0.40)   # Gray 1993, Ullal 2002, Kokkoris 2009
MAG_TOL_PCT = 10.0
SHAPE_TOL_PP = 5.0
FAIL_MAG_PCT = 15.0


def _points_by_condition(points):
    by_cond = {}
    for p in points:
        by_cond.setdefault(p['condition'], []).append(p)
    for c in by_cond:
        by_cond[c] = sorted(by_cond[c], key=lambda r: r['gamma_Al'])
    return by_cond


def _pick_gamma_star(fit_points):
    """Return the gamma_Al in the sweep grid that minimises |residual_F| at
    the fit point."""
    best = min(fit_points, key=lambda p: abs(p['residual_F_pct']))
    return best


def _closest_point(points_for_cond, gamma_target):
    return min(points_for_cond, key=lambda p: abs(p['gamma_Al'] - gamma_target))


def _decide(gamma_star_val, held_out_rows):
    """Apply the PASS/PARTIAL/FAIL gate."""
    if not (LITERATURE_BOUNDS[0] <= gamma_star_val <= LITERATURE_BOUNDS[1]):
        return 'FAIL', f'gamma_Al* = {gamma_star_val:.3f} outside [0.01, 0.40]'

    mag_all_pass = all(abs(r['residual_F_pct']) <= MAG_TOL_PCT for r in held_out_rows)
    drop_all_pass = all(abs(r['residual_drop_pp']) <= SHAPE_TOL_PP for r in held_out_rows)
    any_mag_wide = any(abs(r['residual_F_pct']) > FAIL_MAG_PCT for r in held_out_rows)

    if mag_all_pass and drop_all_pass:
        return 'PASS', 'all held-out points within tolerance'
    if mag_all_pass and not drop_all_pass:
        return 'PARTIAL', 'magnitude tolerance met; F-drop tolerance fails on at least one held-out'
    if any_mag_wide:
        return 'FAIL', 'at least one held-out has |residual_F| > 15%'
    return 'FAIL', 'magnitude tolerance fails on at least one held-out (between 10% and 15%)'


def _render_markdown_table(gamma_star_row, held_out_rows):
    lines = []
    lines.append('| Condition | gamma_Al | Model [F]_c (cm^-3) | Mettler [F]_c (cm^-3) | res_F | Model F-drop | Mettler F-drop | res_drop |')
    lines.append('|---|---|---|---|---|---|---|---|')

    def _row(tag, r):
        return (f'| {tag} {r["condition"]} | {r["gamma_Al"]:.3f} | '
                f'{r["nF_centre_wafer_cm3"]:.3e} | {r["mettler_nF_c_cm3"]:.3e} | '
                f'{r["residual_F_pct"]:+.1f}% | {r["F_drop_pct"]:.2f}% | '
                f'{r["mettler_F_drop_pct"]:.2f}% | {r["residual_drop_pp"]:+.2f} pp |')

    lines.append(_row('FIT', gamma_star_row))
    for r in held_out_rows:
        lines.append(_row('HELD-OUT', r))
    return '\n'.join(lines)


def _plot_scan(by_cond, gamma_star_val, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'matplotlib unavailable ({e}); skipping plot.')
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    for ax, cond in zip(axes.flat, [FIT_POINT] + HELD_OUT):
        if cond not in by_cond:
            continue
        pts = by_cond[cond]
        g = [p['gamma_Al'] for p in pts]
        nF = [p['nF_centre_wafer_cm3'] for p in pts]
        mettler = pts[0]['mettler_nF_c_cm3']
        ax.plot(g, nF, 'o-', lw=1.5)
        ax.axhline(mettler, ls='--', color='r', lw=1, label=f'Mettler ({mettler:.2e})')
        ax.axvline(gamma_star_val, ls=':', color='k', lw=1, label=f'gamma_Al* = {gamma_star_val:.3f}')
        ax.set_xlabel('gamma_Al')
        ax.set_ylabel('[F]_c at wafer (cm^-3)')
        tag = 'FIT' if cond == FIT_POINT else 'HELD-OUT'
        ax.set_title(f'{tag} — {cond}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot written to {out_path}')


def main():
    if not os.path.isfile(SUMMARY_PATH):
        print(f'ERROR: sweep summary not found at {SUMMARY_PATH}')
        sys.exit(2)

    with open(SUMMARY_PATH) as f:
        data = json.load(f)
    by_cond = _points_by_condition(data['points'])

    # Sanity: all four conditions present, 9 gamma_Al each
    for cond in [FIT_POINT] + HELD_OUT:
        assert cond in by_cond, f'Missing condition: {cond}'
        assert len(by_cond[cond]) >= 5, (
            f'Condition {cond} has only {len(by_cond[cond])} points; expected 9.')

    fit_row = _pick_gamma_star(by_cond[FIT_POINT])
    gamma_star = float(fit_row['gamma_Al'])

    held_rows = [_closest_point(by_cond[c], gamma_star) for c in HELD_OUT]

    # -------- DECIDE BEFORE PRINTING TABLE --------
    decision, reason = _decide(gamma_star, held_rows)

    print('=' * 72)
    print(f'DECISION: {decision}   ({reason})')
    print(f'gamma_Al* = {gamma_star:.3f}   (fit point: {FIT_POINT})')
    print(f'Literature bounds: [{LITERATURE_BOUNDS[0]}, {LITERATURE_BOUNDS[1]}] '
          f'(Gray 1993, Ullal 2002, Kokkoris 2009)')
    print('=' * 72)

    md_table = _render_markdown_table(fit_row, held_rows)
    print(md_table)

    # Fixed baseline at gamma_Al = 0.18 (for context in memo)
    fixed_baseline = []
    for c in [FIT_POINT] + HELD_OUT:
        row = _closest_point(by_cond[c], 0.18)
        fixed_baseline.append(row)

    print('\n--- Fixed-baseline context: residuals at gamma_Al=0.18 (same fixed BC map) ---')
    for r in fixed_baseline:
        print(f'  {r["condition"]:<22s}  [F]_c = {r["nF_centre_wafer_cm3"]:.3e}  '
              f'residual = {r["residual_F_pct"]:+.1f}%  F-drop = {r["F_drop_pct"]:.2f}% '
              f'(res = {r["residual_drop_pp"]:+.2f} pp)')

    # Plot
    plot_path = os.path.join(SWEEP_DIR, 'fig_gamma_al_scan.pdf')
    _plot_scan(by_cond, gamma_star, plot_path)

    # Emit structured result for the memo
    decision_json = {
        'decision': decision,
        'reason': reason,
        'gamma_al_star': gamma_star,
        'literature_bounds': list(LITERATURE_BOUNDS),
        'tolerances': {'mag_pct': MAG_TOL_PCT, 'shape_pp': SHAPE_TOL_PP,
                       'fail_mag_pct': FAIL_MAG_PCT},
        'fit_point_row': fit_row,
        'held_out_rows': held_rows,
        'fixed_baseline_rows': fixed_baseline,
        'markdown_table': md_table,
    }
    with open(os.path.join(SWEEP_DIR, 'decision.json'), 'w') as f:
        json.dump(decision_json, f, indent=2)
    print(f'\nDecision JSON written to {os.path.join(SWEEP_DIR, "decision.json")}')

    sys.exit(0 if decision in ('PASS', 'PARTIAL') else 1)


if __name__ == '__main__':
    main()
