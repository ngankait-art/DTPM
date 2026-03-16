"""
Silicon etch rate prediction from fluorine density.

The etch rate is: R_etch = gamma_Si * (1/4) * v_th_F * [F] * (M_Si / rho_Si / N_A)

where:
  gamma_Si = Si etch probability by F atoms (~0.025 from radical probes, Mettler)
  v_th_F = thermal velocity of F at gas temperature
  [F] = fluorine atom density at the wafer surface (m^-3)
  M_Si = 28.09 g/mol
  rho_Si = 2329 kg/m^3
  N_A = 6.022e23 mol^-1

The factor M_Si/(rho_Si*N_A) converts from atoms removed per area per time
to thickness per time (m/s → nm/s with 1e9 factor).
"""

import numpy as np
from scipy.constants import k as kB, pi, N_A

M_F_kg = 19.0 * 1.66054e-27   # F atom mass
M_Si = 28.09e-3                # Si molar mass, kg/mol
rho_Si = 2329.0                # Si density, kg/m3

# Etch probability from Mettler Table 4.18
GAMMA_SI_RADICAL_PROBE = 0.025   # From radical probe measurements
GAMMA_SI_ACTINOMETRY = 0.08      # From actinometry (overestimates)
GAMMA_SI_DEFAULT = 0.025          # Use radical probe value


def etch_rate(nF_at_wafer, Tgas=300.0, gamma_Si=GAMMA_SI_DEFAULT):
    """Compute Si etch rate from F density at the wafer.

    Parameters
    ----------
    nF_at_wafer : float or array
        F atom density at the wafer surface [m^-3]
    Tgas : float
        Gas temperature at the wafer [K] (typically ~25°C = 298 K)
    gamma_Si : float
        Si etch probability (dimensionless)

    Returns
    -------
    R_nm_s : float or array
        Si etch rate in nm/s
    """
    v_th_F = np.sqrt(8 * kB * Tgas / (pi * M_F_kg))   # m/s
    F_flux = 0.25 * v_th_F * nF_at_wafer               # m^-2 s^-1
    atoms_per_m3_Si = rho_Si * N_A / M_Si               # Si atoms per m^3
    R_m_s = gamma_Si * F_flux / atoms_per_m3_Si          # m/s
    R_nm_s = R_m_s * 1e9                                  # nm/s
    return R_nm_s


def etch_rate_profile(nF_2d, mesh, Tgas=300.0, gamma_Si=GAMMA_SI_DEFAULT):
    """Extract the radial etch rate profile at the wafer plane (z=0).

    Parameters
    ----------
    nF_2d : array (Nr, Nz)
        2D fluorine density field [m^-3]
    mesh : Mesh2D object
    Tgas, gamma_Si : floats

    Returns
    -------
    r_cm : array
        Radial positions [cm]
    R_etch : array
        Si etch rate at the wafer [nm/s]
    """
    nF_wafer = nF_2d[:, 0]  # z=0 is the wafer
    R_etch = etch_rate(nF_wafer, Tgas, gamma_Si)
    r_cm = mesh.r * 100
    return r_cm, R_etch


def uniformity(R_etch, r_cm=None, r_max_cm=None):
    """Compute etch rate uniformity metrics.

    Parameters
    ----------
    R_etch : array
        Etch rate profile [nm/s]
    r_cm : array, optional
        Radial positions
    r_max_cm : float, optional
        Max radius for uniformity calculation (e.g., wafer radius)

    Returns
    -------
    dict with: mean, std, nonuniformity_pct, center, edge, range_pct
    """
    if r_cm is not None and r_max_cm is not None:
        mask = r_cm <= r_max_cm
        R = R_etch[mask]
    else:
        R = R_etch

    return {
        'mean': np.mean(R),
        'std': np.std(R),
        'nonuniformity_pct': (np.max(R) - np.min(R)) / (2 * np.mean(R)) * 100,
        'center': R[0],
        'edge': R[-1] if len(R) > 1 else R[0],
        'range_pct': (R[0] - R[-1]) / R[0] * 100 if R[0] > 0 else 0,
    }


if __name__ == '__main__':
    # Test with Mettler conditions
    print("Si Etch Rate from F Density")
    print("=" * 50)

    # Mettler Fig 4.17: 90% SF6, bias OFF, center [F] = 1.7e20 m-3
    nF_test = 1.7e20
    R = etch_rate(nF_test, Tgas=298)
    print(f"\nMettler 90%SF6 center: [F] = {nF_test:.2e} m-3")
    print(f"  Predicted etch rate: {R:.1f} nm/s")
    print(f"  Mettler measured:    22.0 nm/s")
    print(f"  Ratio: {R/22.0:.2f}")

    # Gen-4b center [F]
    nF_model = 3.5e19  # from Gen-4b at 1500W pure SF6
    R_model = etch_rate(nF_model, Tgas=298)
    print(f"\nGen-4b center: [F] = {nF_model:.2e} m-3")
    print(f"  Predicted etch rate: {R_model:.1f} nm/s")
