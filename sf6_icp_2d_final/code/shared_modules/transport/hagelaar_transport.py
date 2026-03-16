"""
Hagelaar-corrected electron transport coefficients.

Based on: G.J.M. Hagelaar and L.C. Pitchford,
"Solving the Boltzmann equation to obtain electron transport coefficients
 and rate coefficients for fluid models,"
Plasma Sources Sci. Technol. 14, 722-733 (2005).

Key corrections over the simple approximations:
  1. Einstein relation D_e = (2/3)μ_e·ε̄ is WRONG by ~2× for Ar (Ramsauer minimum)
  2. The (5/3) factor for energy transport D_ε = (5/3)D_e is wrong by ~2× for Ar
  3. SF6 has NO Ramsauer minimum — corrections are smaller (~15-20%)
  4. For mixtures, use Blanc's law: 1/μ_mix = Σ x_k/μ_k
"""

import numpy as np
from scipy.constants import e as eC, m_e, k as kB, pi

gamma_boltzmann = np.sqrt(2 * eC / m_e)  # ≈ 5.931e5 m/s per eV^{1/2}


def transport_Ar(Te_eV, N):
    """Electron transport coefficients for pure Ar with Ramsauer correction.

    The Ramsauer minimum in Ar at ~0.2 eV causes the momentum-transfer
    frequency to be strongly energy-dependent. This makes the Einstein
    relation underestimate D_e by ~2× at Te = 2-4 eV (typical ICP range).

    Parameters
    ----------
    Te_eV : float or array
        Electron temperature in eV
    N : float
        Gas number density in m⁻³

    Returns
    -------
    dict with keys: mu_e, D_e, mu_eps, D_eps, nu_m (all SI)
    """
    Te = np.asarray(Te_eV, dtype=float)

    # Effective σ_m for Ar (fit to BOLSIG+/Phelps data)
    # Increases from ~1e-20 at 0.3 eV to ~8e-20 at 10 eV
    sigma_eff = 1.0e-20 * (0.5 + 2.5 * Te**0.7)

    v_th = np.sqrt(8 * eC * Te / (pi * m_e))
    nu_m = N * sigma_eff * v_th

    mu_e_base = eC / (m_e * nu_m)
    D_e_einstein = (2.0/3.0) * mu_e_base * 1.5 * Te  # ε̄ = 1.5*Te

    # Ramsauer correction factors (from Hagelaar 2005 Fig 9a)
    ramsauer_D = 1.0 + 1.2 * np.exp(-((Te - 2.0)/2.0)**2)    # D_e/D_einstein ~ 1.5-2.2
    ramsauer_mu = 1.0 + 0.3 * np.exp(-((Te - 2.0)/2.0)**2)   # μ_e correction ~ 1.0-1.3

    D_e = D_e_einstein * ramsauer_D
    mu_e = mu_e_base * ramsauer_mu

    # Energy transport (Hagelaar Eqs 61-62)
    # D_ε/D_e ratio is ~2.5-3.5 for Ar instead of 5/3
    D_eps_ratio = (5.0/3.0) * (1.0 + 0.8 * np.exp(-((Te - 2.5)/2.5)**2))
    mu_eps_ratio = (5.0/3.0) * (1.0 + 0.15 * np.exp(-((Te - 2.5)/2.5)**2))

    D_eps = D_e * D_eps_ratio
    mu_eps = mu_e * mu_eps_ratio

    return {'mu_e': mu_e, 'D_e': D_e, 'mu_eps': mu_eps, 'D_eps': D_eps,
            'nu_m': nu_m, 'eps_bar': 1.5*Te}


def transport_SF6(Te_eV, N):
    """Electron transport for pure SF6 (no Ramsauer minimum).

    SF6 has a very large total scattering cross-section (~20-50 × 10⁻²⁰ m²)
    with no Ramsauer minimum. The Einstein relation is more accurate (~15-20% error).
    """
    Te = np.asarray(Te_eV, dtype=float)

    sigma_eff = 1.0e-20 * (30.0 + 10.0 * Te**0.5)
    v_th = np.sqrt(8 * eC * Te / (pi * m_e))
    nu_m = N * sigma_eff * v_th

    mu_e = eC / (m_e * nu_m)
    D_e = (2.0/3.0) * mu_e * 1.5 * Te * 1.15    # 15% non-Maxwellian correction
    D_eps = (5.0/3.0) * D_e * 1.1                 # 10% correction over (5/3)
    mu_eps = (5.0/3.0) * mu_e * 1.05

    return {'mu_e': mu_e, 'D_e': D_e, 'mu_eps': mu_eps, 'D_eps': D_eps,
            'nu_m': nu_m, 'eps_bar': 1.5*Te}


def transport_mixture(Te_eV, N, x_Ar, x_SF6):
    """Transport coefficients for Ar/SF6 mixture using Blanc's law.

    1/μ_mix = x_Ar/μ_Ar + x_SF6/μ_SF6

    Parameters
    ----------
    Te_eV : float
    N : float — total gas density m⁻³
    x_Ar, x_SF6 : float — mole fractions (should sum to 1)
    """
    ar = transport_Ar(Te_eV, N)
    sf6 = transport_SF6(Te_eV, N)

    if x_Ar > 0 and x_SF6 > 0:
        mu_e = 1.0 / (x_Ar/ar['mu_e'] + x_SF6/sf6['mu_e'])
        D_e = 1.0 / (x_Ar/ar['D_e'] + x_SF6/sf6['D_e'])
        mu_eps = 1.0 / (x_Ar/ar['mu_eps'] + x_SF6/sf6['mu_eps'])
        D_eps = 1.0 / (x_Ar/ar['D_eps'] + x_SF6/sf6['D_eps'])
        nu_m = x_Ar*ar['nu_m'] + x_SF6*sf6['nu_m']
    elif x_Ar >= 1:
        return ar
    else:
        return sf6

    return {'mu_e': mu_e, 'D_e': D_e, 'mu_eps': mu_eps, 'D_eps': D_eps,
            'nu_m': nu_m, 'eps_bar': 1.5*Te_eV}


if __name__ == '__main__':
    p_Pa = 10 * 0.133322
    N = p_Pa / (kB * 300)
    print(f"N = {N:.3e} m⁻³ (10 mTorr, 300 K)\n")

    print("Pure Ar (with Ramsauer correction):")
    print(f"{'Te':>4s} {'μ_e':>10s} {'D_e':>10s} {'D_ε':>10s} {'D_ε/D_e':>8s} {'Rams×':>6s}")
    for Te in [2.0, 3.0, 4.0, 5.0]:
        r = transport_Ar(Te, N)
        r0 = transport_Ar(Te, N)  # base
        print(f"{Te:4.1f} {r['mu_e']:10.4f} {r['D_e']:10.2f} {r['D_eps']:10.1f} "
              f"{r['D_eps']/r['D_e']:8.2f} {r['D_e']/((2/3)*r['mu_e']*1.5*Te):6.2f}")

    print(f"\nPure SF6:")
    for Te in [2.0, 3.0, 4.0, 5.0]:
        r = transport_SF6(Te, N)
        print(f"{Te:4.1f} {r['mu_e']:10.4f} {r['D_e']:10.2f} {r['D_eps']:10.1f} "
              f"{r['D_eps']/r['D_e']:8.2f}")

    print(f"\nMixture at Te=3 eV:")
    for x_Ar in [0.0, 0.3, 0.5, 0.7, 1.0]:
        r = transport_mixture(3.0, N, x_Ar, 1-x_Ar)
        print(f"  {x_Ar*100:3.0f}%Ar: D_e={r['D_e']:.2f} D_ε={r['D_eps']:.1f} "
              f"D_ε/D_e={r['D_eps']/r['D_e']:.2f}")
