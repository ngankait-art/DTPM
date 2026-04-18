#!/usr/bin/env python3
"""
Phase A.1 — V3 anchor points against Mettler Fig 4.9.

Two single-point runs of the Lallement 0D SF6/Ar global model at Mettler's
Fig 4.9 operating conditions (90 sccm SF6 / 10 sccm Ar total = 10% Ar by
volume, 100 sccm total flow):

- Low-p branch: 600 W / 20 mTorr
- High-p branch: 700 W / 40 mTorr

Writes numerical output to 06b/output/v3_anchor_points.csv with Mettler
reference values (from 06b/data/mettler/mettler_fig4p9b_{20,40}mTorr.csv,
last row = 90 sccm).

No figures are produced here (see run_mettler_fig49_overlay.py for the
plot that uses these anchor points).
"""
import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
LALLEMENT = os.path.join(REPO, 'code', 'lallement_sf6ar')
DATA_METTLER = os.path.join(REPO, 'data', 'mettler')
OUTPUT = os.path.join(REPO, 'output')

sys.path.insert(0, LALLEMENT)
from sf6_global_model_final import solve_model  # noqa: E402


def mettler_nF_at_90sccm(csv_path):
    """Last row of Mettler Fig 4.9b CSV (x_sccm, nF_m3); 90 sccm is the rightmost point."""
    with open(csv_path) as f:
        rows = [tuple(float(x) for x in line.strip().split(',')) for line in f if line.strip()]
    rows.sort(key=lambda r: r[0])  # ascending by sccm
    return rows[-1]  # (sccm, nF_m3)


def main():
    os.makedirs(OUTPUT, exist_ok=True)

    # Mettler reference values (90 sccm SF6 endpoints from digitised Fig 4.9b)
    lo_sccm, lo_mettler = mettler_nF_at_90sccm(
        os.path.join(DATA_METTLER, 'mettler_fig4p9b_20mTorr.csv'))
    hi_sccm, hi_mettler = mettler_nF_at_90sccm(
        os.path.join(DATA_METTLER, 'mettler_fig4p9b_40mTorr.csv'))

    print(f"Mettler anchors: 600W/20mTorr @ {lo_sccm:.1f} sccm = {lo_mettler:.2e} m^-3")
    print(f"                 700W/40mTorr @ {hi_sccm:.1f} sccm = {hi_mettler:.2e} m^-3")

    # Two simulation points at Mettler's Fig 4.9 high-flow endpoints.
    # frac_Ar = 0.10 because 90 sccm SF6 / 100 sccm total = 10% Ar.
    print("\nRunning 600W / 20 mTorr / 10% Ar / 100 sccm ...")
    r_lo = solve_model(P_rf=600, p_mTorr=20, frac_Ar=0.10, Q_sccm=100, eta=0.12)
    print(f"  ne={r_lo['ne']:.2e}  Te={r_lo['Te']:.2f} eV  nF={r_lo['n_F']:.3e} m^-3  "
          f"alpha={r_lo['alpha']:.1f}  converged={r_lo['converged']}")

    print("\nRunning 700W / 40 mTorr / 10% Ar / 100 sccm ...")
    r_hi = solve_model(P_rf=700, p_mTorr=40, frac_Ar=0.10, Q_sccm=100, eta=0.12)
    print(f"  ne={r_hi['ne']:.2e}  Te={r_hi['Te']:.2f} eV  nF={r_hi['n_F']:.3e} m^-3  "
          f"alpha={r_hi['alpha']:.1f}  converged={r_hi['converged']}")

    # Write traceability CSV
    csv_out = os.path.join(OUTPUT, 'v3_anchor_points.csv')
    with open(csv_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['label', 'P_rf_W', 'p_mTorr', 'frac_Ar', 'Q_sccm_total',
                    'Q_SF6_sccm', 'nF_model_m3', 'nF_Mettler_m3',
                    'ratio_model_over_Mettler', 'Te_eV', 'ne_m3', 'alpha',
                    'dissoc_frac', 'converged'])
        for label, r, mett in [
            ('low-p  (600W/20mTorr)', r_lo, lo_mettler),
            ('high-p (700W/40mTorr)', r_hi, hi_mettler),
        ]:
            nF = r['n_F']
            w.writerow([label, r.get('P_rf', '-'), r.get('p_mTorr', '-'),
                        r.get('frac_Ar', '-'), 100, 90,
                        f"{nF:.3e}", f"{mett:.3e}", f"{nF/mett:.3f}",
                        f"{r['Te']:.3f}", f"{r['ne']:.3e}",
                        f"{r['alpha']:.2f}", f"{r['dissoc_frac']:.3f}",
                        r['converged']])

    print(f"\n[OK] Wrote {csv_out}")
    print(f"\nSummary:")
    print(f"  Low-p  model/Mettler = {r_lo['n_F']/lo_mettler:.2f}")
    print(f"  High-p model/Mettler = {r_hi['n_F']/hi_mettler:.2f}")


if __name__ == '__main__':
    main()
