"""Post-process existing sweep summaries to add missing V_peak, V_rms, P_abs
operating-voltage fields computed from the Lieberman matched-resonance circuit:

    V_peak = I_peak * (R_coil + R_plasma)
    V_rms  = V_peak / sqrt(2)
    P_abs  = eta * P_rf

Rationale: earlier runs predated the m11 fix that now writes V_peak_final and
V_rms_final into the state dict. Existing JSONs on disk do not have those
fields; this script recomputes them without a re-run.
"""
import json
import math
import os
from pathlib import Path

PHASE1 = Path('/Users/kaingan8/Downloads/SF6_unified/active_projects/phase1_self_consistent')


def patch_point(summary_path, R_coil, P_rf):
    """Patch a single summary.json with V_peak, V_rms, P_abs, and aliases."""
    with open(summary_path) as f:
        s = json.load(f)
    eta = s.get('eta_computed') or s.get('eta') or 0.0
    I_peak = s.get('I_peak_final') or s.get('I_peak_A') or 0.0
    R_plasma = s.get('R_plasma_final') or s.get('R_plasma_ohm') or 0.0
    P_abs = eta * P_rf
    V_peak = I_peak * (R_coil + R_plasma)
    V_rms = V_peak / math.sqrt(2.0)
    s.update({
        'P_abs': P_abs,
        'P_abs_final': P_abs,
        'V_peak_final': V_peak,
        'V_rms_final': V_rms,
        'R_coil': R_coil,
    })
    with open(summary_path, 'w') as f:
        json.dump(s, f, indent=2)
    return {
        'eta': eta, 'I_peak': I_peak, 'R_plasma': R_plasma,
        'P_abs': P_abs, 'V_peak': V_peak, 'V_rms': V_rms,
    }


def patch_rcoil_sweep():
    base = PHASE1 / 'results' / 'rcoil_sweep'
    rows = []
    for d in sorted(base.glob('R*ohm')):
        R_coil = float(d.name[1:].rstrip('ohm'))
        s = patch_point(d / 'summary.json', R_coil=R_coil, P_rf=1000)
        rows.append({'R_coil': R_coil, **s})
    # Rewrite aggregate
    with open(base / 'rcoil_summary.json') as f:
        agg = json.load(f)
    for i, row in enumerate(agg):
        m = next(r for r in rows if abs(r['R_coil'] - row['R_coil']) < 1e-6)
        row['V_peak_V'] = m['V_peak']
        row['V_rms_V'] = m['V_rms']
        row['P_abs_W'] = m['P_abs']
    with open(base / 'rcoil_summary.json', 'w') as f:
        json.dump(agg, f, indent=2)
    return rows


def patch_composition_pair():
    base = PHASE1 / 'results' / 'mettler_composition'
    results = []
    for comp in ['90pct_SF6', '30pct_SF6']:
        for bias in ['bias_on', 'bias_off']:
            p = base / comp / bias / 'summary.json'
            if p.exists():
                s = patch_point(p, R_coil=0.8, P_rf=1000)
                results.append((comp, bias, s))
    return results


def patch_baseline_700W():
    p = PHASE1 / 'results' / 'baseline_700W_70SF6_nobias' / 'summary.json'
    if p.exists():
        s = patch_point(p, R_coil=0.8, P_rf=700)
        return s
    return None


if __name__ == '__main__':
    print("=== patching rcoil_sweep ===")
    rows = patch_rcoil_sweep()
    print(f"  {'R_coil':>8s}  {'eta':>6s}  {'Ipk(A)':>8s}  {'Vpk(V)':>8s}  {'Vrms(V)':>8s}  {'Pabs(W)':>8s}")
    for r in rows:
        print(f"  {r['R_coil']:>8.2f}  {r['eta']:>6.3f}  {r['I_peak']:>8.2f}  "
              f"{r['V_peak']:>8.2f}  {r['V_rms']:>8.2f}  {r['P_abs']:>8.1f}")

    print("\n=== patching mettler composition pair ===")
    for comp, bias, s in patch_composition_pair():
        print(f"  {comp:12s} {bias:9s}  eta={s['eta']:.3f}  Ipk={s['I_peak']:.2f}A  "
              f"Vpk={s['V_peak']:.2f}V  Vrms={s['V_rms']:.2f}V  Pabs={s['P_abs']:.1f}W")

    print("\n=== patching baseline_700W_70SF6 ===")
    s = patch_baseline_700W()
    if s:
        print(f"  eta={s['eta']:.3f}  Ipk={s['I_peak']:.2f}A  "
              f"Vpk={s['V_peak']:.2f}V  Vrms={s['V_rms']:.2f}V  Pabs={s['P_abs']:.1f}W")
