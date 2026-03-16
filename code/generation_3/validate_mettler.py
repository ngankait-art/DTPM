#!/usr/bin/env python3
"""
Mettler Validation: Compare Gen-3 radial [F](r) against experimental data.

Mettler Fig 4.14: Normalized F density at 1000W, 10mTorr, 70/30 SF6/Ar
  - Cubic fit: y = 1.01032 - 0.01847*r² + 7.139e-4*r³
  - Center-to-edge drop: ~75%

Key caveat: Mettler uses a TEL etcher at 2 MHz ICP, while our model
simulates a Lallement-type reactor at 13.56 MHz. We compare NORMALIZED
profile shapes, not absolute values.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from validation.mettler_data import (fig414_r_cm, fig414_nF_norm, fig414_fit,
                                      r_417, nF_90off, nF_30off, nF_90on, nF_30on)


def run_validation(P_rf=1000, p_mTorr=10, frac_Ar=0.3, Nr=30, Nz=40):
    """Run Gen-3 and compare [F](r) against Mettler Fig 4.14."""

    from main_v3 import run_v3

    print(f"Running Gen-3: {P_rf}W, {p_mTorr}mT, {(1-frac_Ar)*100:.0f}%SF6/{frac_Ar*100:.0f}%Ar")
    res = run_v3(P_rf=P_rf, p_mTorr=p_mTorr, frac_Ar=frac_Ar,
                 Nr=Nr, Nz=Nz, n_iter=80, verbose=True)

    mesh = res['mesh']
    Rc = mesh.r * 100  # cm
    nF = res['nF']

    # Extract [F] at z ≈ 1 cm above wafer (z=0), so z_index near z=1cm
    z_cm = mesh.z * 100
    j_wafer = 0                                    # z = 0 (wafer plane)
    j_1cm = np.argmin(np.abs(z_cm - 1.0))         # z ≈ 1 cm (probe height)
    j_mid = len(z_cm) // 2                          # midplane

    nF_wafer = nF[:, j_wafer]
    nF_1cm = nF[:, j_1cm]
    nF_mid = nF[:, j_mid]

    # Normalize to center value
    nF_wafer_norm = nF_wafer / max(nF_wafer[0], 1e-10)
    nF_1cm_norm = nF_1cm / max(nF_1cm[0], 1e-10)
    nF_mid_norm = nF_mid / max(nF_mid[0], 1e-10)

    # Mettler cubic fit on fine grid
    r_fit = np.linspace(0, 8, 100)
    nF_mettler = fig414_fit(r_fit)

    # ── Figure 1: Normalized [F](r) comparison ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(r_fit, nF_mettler, 'k-', lw=2.5, label='Mettler Fig 4.14 (cubic fit)')
    ax.plot(fig414_r_cm, fig414_nF_norm, 'ko', ms=10, zorder=5, label='Mettler data points')
    ax.plot(Rc, nF_1cm_norm, 'r--', lw=2, label=f'Gen-3 at z={z_cm[j_1cm]:.1f}cm (probe height)')
    ax.plot(Rc, nF_wafer_norm, 'b:', lw=2, label='Gen-3 at z=0 (wafer)')
    ax.plot(Rc, nF_mid_norm, 'g-.', lw=1.5, alpha=0.6, label=f'Gen-3 at z={z_cm[j_mid]:.1f}cm (midplane)')
    ax.set_xlabel('Radial position r (cm)', fontsize=12)
    ax.set_ylabel('Normalized [F] density', fontsize=12)
    ax.set_title('Normalized radial F profile', fontsize=13)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_xlim(0, max(Rc.max(), 8)); ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='gray', ls=':', alpha=0.4)

    # Annotate drop
    drop_model = (1 - nF_1cm_norm[np.argmin(np.abs(Rc-8))]) * 100
    drop_mettler = (1 - fig414_fit(8)) * 100
    ax.text(0.97, 0.97, f'Center→edge drop:\n  Mettler: {drop_mettler:.0f}%\n  Model: {drop_model:.0f}%',
            transform=ax.transAxes, va='top', ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ── Figure 2: Absolute [F](r) comparison with Fig 4.17 ──
    ax2 = axes[1]
    # Model absolute values
    ax2.plot(Rc, nF_1cm, 'r-', lw=2.5, label=f'Gen-3 ({(1-frac_Ar)*100:.0f}%SF6, no bias)')

    # Mettler Fig 4.17 data (closest conditions)
    if frac_Ar <= 0.15:
        ax2.plot(r_417, nF_90off, 'ks--', ms=7, label='Mettler 90%SF6, no bias')
    elif frac_Ar >= 0.65:
        ax2.plot(r_417, nF_30off, 'k^--', ms=7, label='Mettler 30%SF6, no bias')
    else:
        ax2.plot(r_417, nF_90off, 'ks--', ms=6, alpha=0.5, label='Mettler 90%SF6')
        ax2.plot(r_417, nF_30off, 'k^--', ms=6, alpha=0.5, label='Mettler 30%SF6')

    ax2.set_xlabel('Radial position r (cm)', fontsize=12)
    ax2.set_ylabel('[F] density (m⁻³)', fontsize=12)
    ax2.set_title('Absolute [F] comparison', fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, max(Rc.max(), 9))
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax2.text(0.97, 0.97,
             f'Model: {P_rf}W, {p_mTorr}mT\n'
             f'  {(1-frac_Ar)*100:.0f}%SF6, no bias\n'
             f'Mettler: 1000W (2MHz), 10mT\n'
             f'  TEL etcher (proprietary)\n'
             f'NOTE: different reactors',
             transform=ax2.transAxes, va='top', ha='right', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))

    fig.suptitle(f'Mettler Validation — {P_rf}W, {p_mTorr}mTorr, '
                 f'{(1-frac_Ar)*100:.0f}%SF6/{frac_Ar*100:.0f}%Ar',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.95])
    os.makedirs('outputs_v3', exist_ok=True)
    fig.savefig('outputs_v3/mettler_validation.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved outputs_v3/mettler_validation.png")
    plt.close()

    return res


if __name__ == '__main__':
    # Match Mettler Fig 4.14 conditions: 1000W, 10mT, 70/30 SF6/Ar
    run_validation(P_rf=1000, p_mTorr=10, frac_Ar=0.3, Nr=30, Nz=40)
