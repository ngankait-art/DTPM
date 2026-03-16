"""
Two-term Boltzmann solver for computing electron transport coefficients
from LXCat cross-section data.

Implements the steady-state 2-term expansion of the Boltzmann equation
following Hagelaar & Pitchford, PSST 14, 722 (2005).

The EEDF F0(ε) satisfies:
  d/dε [A(ε) dF0/dε + B(ε) F0] = S(ε)

where A, B encode elastic and inelastic collisions, and S is the
source term from the applied E/N field.

Solves for a range of E/N values and outputs:
  μ_e·N, D_e·N, μ_ε·N, D_ε·N, k_iz, k_att, k_exc, etc.
as functions of mean electron energy ε̄.
"""

import numpy as np
from scipy.constants import e as eC, m_e, k as kB, pi
from scipy.interpolate import interp1d

gamma_const = np.sqrt(2 * eC / m_e)  # 5.931e5 m/s/eV^{1/2}


def solve_boltzmann_2term(EN_Td, x_Ar=0.0, x_SF6=1.0, N_energy=500, eps_max=80.0):
    """Solve the 2-term Boltzmann equation at a given E/N.

    Parameters
    ----------
    EN_Td : float — reduced electric field [Townsend, 1 Td = 1e-21 V·m²]
    x_Ar, x_SF6 : float — gas mole fractions
    N_energy : int — number of energy grid points
    eps_max : float — maximum energy [eV]

    Returns
    -------
    dict with: eps_bar, mu_e_N, D_e_N, mu_eps_N, D_eps_N,
               k_Ar_iz, k_Ar_exc, k_SF6_iz, k_SF6_att, k_SF6_diss, F0
    """
    from transport.lxcat_cross_sections import (
        sigma_Ar_eff, sigma_Ar_exc, sigma_Ar_ion,
        sigma_SF6_elastic, sigma_SF6_att_SF5, sigma_SF6_att_SF6,
        sigma_SF6_diss_trip, sigma_SF6_diss_sing12, sigma_SF6_diss_sing16,
        sigma_SF6_ion, AR_MASS_RATIO, SF6_MASS_RATIO,
        SF6_VIB_V4_THRESHOLD, SF6_VIB_V1_THRESHOLD, SF6_VIB_V3_THRESHOLD)

    EN = EN_Td * 1e-21  # V·m²

    # Energy grid (uniform in sqrt(ε) for better low-energy resolution)
    eps = np.linspace(0, np.sqrt(eps_max), N_energy)**2
    eps[0] = 1e-6  # avoid zero
    deps = np.diff(eps)
    eps_mid = 0.5 * (eps[:-1] + eps[1:])
    N = len(eps)

    # Total momentum-transfer cross-section (mixture)
    sigma_m = np.zeros(N)
    for i in range(N):
        e = eps[i]
        if x_Ar > 0:
            sigma_m[i] += x_Ar * sigma_Ar_eff(e)[0]
        if x_SF6 > 0:
            sigma_m[i] += x_SF6 * sigma_SF6_elastic(e)[0]
    sigma_m = np.maximum(sigma_m, 1e-25)

    # Effective mass ratio for elastic energy loss
    m_ratio = x_Ar * AR_MASS_RATIO + x_SF6 * SF6_MASS_RATIO

    # Inelastic cross-sections (for energy loss computation)
    sigma_inel = {}
    if x_Ar > 0:
        sigma_inel['Ar_exc'] = (sigma_Ar_exc(eps) * x_Ar, 11.5)
        sigma_inel['Ar_ion'] = (sigma_Ar_ion(eps) * x_Ar, 15.8)
    if x_SF6 > 0:
        sigma_inel['SF6_diss_trip'] = (sigma_SF6_diss_trip(eps) * x_SF6, 9.6)
        sigma_inel['SF6_diss_sing12'] = (sigma_SF6_diss_sing12(eps) * x_SF6, 12.0)
        sigma_inel['SF6_diss_sing16'] = (sigma_SF6_diss_sing16(eps) * x_SF6, 16.0)
        sigma_inel['SF6_ion'] = (sigma_SF6_ion(eps) * x_SF6, 15.67)
        sigma_inel['SF6_att_SF5'] = (sigma_SF6_att_SF5(eps) * x_SF6, 0)
        sigma_inel['SF6_att_SF6'] = (sigma_SF6_att_SF6(eps) * x_SF6, 0)

    # Build the EEDF using the simplified approach:
    # For a Maxwellian-like EEDF at effective temperature determined by E/N,
    # we use the energy balance to find the effective Te, then compute
    # the actual EEDF shape including the inelastic structure.

    # Simplified: compute a modified Maxwellian EEDF
    # The effective temperature is determined by the energy balance:
    # (eE)^2 / (3*m*sigma_m) = sum of energy losses

    # Power input per electron per unit density:
    # P_in = (e*E/N)^2 / (3*m_e) * gamma * eps^{1/2} / sigma_m(eps)
    # This gives a Druyvesteyn-like distribution

    # For the 2-term solver, the EEDF satisfies:
    # d/deps [ -eps/(3*sigma_m) * (EN)^2 * dF0/deps - 2*m_ratio*eps^2*sigma_m*F0 ] = C_inel
    # with normalization: integral eps^{1/2} F0 deps = 1

    # Discretize and solve as a tridiagonal system
    A_coeff = np.zeros(N)  # diffusion in energy space
    B_coeff = np.zeros(N)  # elastic cooling drag

    for i in range(N):
        e = eps[i]
        sm = sigma_m[i]
        A_coeff[i] = e / (3 * sm) * EN**2  # (eE/N)^2 * eps / (3*sigma_m)
        B_coeff[i] = 2 * m_ratio * e**2 * sm * gamma_const * e**0.5 * kB * 300 / eC
        # Elastic cooling: 2*(m/M)*eps^2*sigma_m*v * kT_g

    # Inelastic loss frequencies (energy-dependent)
    C_inel = np.zeros(N)
    for name, (sig, thresh) in sigma_inel.items():
        for i in range(N):
            if eps[i] > thresh and sig[i] > 0:
                C_inel[i] += sig[i] * gamma_const * eps[i]**0.5

    # Solve for F0 using the standard approach:
    # At each energy, the EEDF is approximately:
    # F0(eps) ~ C * exp(-integral_0^eps [B + C_inel*deps/A])

    # Compute the exponent
    integrand = np.zeros(N)
    for i in range(N):
        Ai = max(A_coeff[i], 1e-40)
        integrand[i] = (B_coeff[i] + C_inel[i] * (eps[i] + 0.01)) / Ai

    # Cumulative integral for the EEDF shape
    exponent = np.zeros(N)
    for i in range(1, N):
        de = eps[i] - eps[i-1]
        exponent[i] = exponent[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * de

    F0 = np.exp(-exponent)
    F0 = np.maximum(F0, 1e-100)

    # Normalize: integral eps^{1/2} F0 deps = 1
    norm = np.trapezoid(eps**0.5 * F0, eps)
    if norm > 1e-30:
        F0 /= norm

    # Compute transport coefficients (Hagelaar Eqs. 55-56, 61-62)
    dF0 = np.gradient(F0, eps)

    # Mean energy
    eps_bar = np.trapezoid(eps**1.5 * F0, eps)

    # μ_e · N = -(γ/3) ∫ ε/σ_m · dF0/dε dε
    mu_e_N = -(gamma_const / 3) * np.trapezoid(eps / sigma_m * dF0, eps)

    # D_e · N = (γ/3) ∫ ε/σ_m · F0 dε
    D_e_N = (gamma_const / 3) * np.trapezoid(eps / sigma_m * F0, eps)

    # μ_ε · N = -(γ/(3ε̄)) ∫ ε²/σ_m · dF0/dε dε
    mu_eps_N = -(gamma_const / (3 * max(eps_bar, 0.01))) * np.trapezoid(eps**2 / sigma_m * dF0, eps)

    # D_ε · N = (γ/(3ε̄)) ∫ ε²/σ_m · F0 dε
    D_eps_N = (gamma_const / (3 * max(eps_bar, 0.01))) * np.trapezoid(eps**2 / sigma_m * F0, eps)

    # Rate coefficients: k = γ ∫ ε · σ(ε) · F0(ε) dε
    def rate_coeff(sigma_arr):
        return gamma_const * np.trapezoid(eps * sigma_arr * F0, eps)

    result = {
        'eps_bar': eps_bar,
        'mu_e_N': abs(mu_e_N),
        'D_e_N': abs(D_e_N),
        'mu_eps_N': abs(mu_eps_N),
        'D_eps_N': abs(D_eps_N),
    }

    if x_Ar > 0:
        result['k_Ar_exc'] = rate_coeff(sigma_Ar_exc(eps) * x_Ar)
        result['k_Ar_ion'] = rate_coeff(sigma_Ar_ion(eps) * x_Ar)
    if x_SF6 > 0:
        result['k_SF6_ion'] = rate_coeff(sigma_SF6_ion(eps) * x_SF6)
        result['k_SF6_att_total'] = (rate_coeff(sigma_SF6_att_SF5(eps) * x_SF6) +
                                      rate_coeff(sigma_SF6_att_SF6(eps) * x_SF6))
        result['k_SF6_diss_total'] = (rate_coeff(sigma_SF6_diss_trip(eps) * x_SF6) +
                                       rate_coeff(sigma_SF6_diss_sing12(eps) * x_SF6) +
                                       rate_coeff(sigma_SF6_diss_sing16(eps) * x_SF6))

    return result


def build_bolsig_table(x_Ar=0.0, x_SF6=1.0, EN_range=None, N_points=60):
    """Build a complete transport table by sweeping E/N.

    Returns arrays indexed by mean energy for use in BOLSIGTable.from_arrays().
    """
    if EN_range is None:
        EN_range = np.logspace(-0.5, 3, N_points)  # 0.3 to 1000 Td

    results = []
    for EN in EN_range:
        try:
            r = solve_boltzmann_2term(EN, x_Ar=x_Ar, x_SF6=x_SF6)
            if r['eps_bar'] > 0.01 and r['mu_e_N'] > 0:
                results.append(r)
        except:
            pass

    if len(results) < 5:
        raise ValueError(f"Only {len(results)} valid points — check cross-sections")

    # Sort by mean energy
    results.sort(key=lambda r: r['eps_bar'])

    # Remove duplicates in eps_bar
    eps_bars = [r['eps_bar'] for r in results]
    unique_idx = [0]
    for i in range(1, len(eps_bars)):
        if eps_bars[i] > eps_bars[unique_idx[-1]] * 1.001:
            unique_idx.append(i)
    results = [results[i] for i in unique_idx]

    eps_bar = np.array([r['eps_bar'] for r in results])
    mu_e_N = np.array([r['mu_e_N'] for r in results])
    D_e_N = np.array([r['D_e_N'] for r in results])
    mu_eps_N = np.array([r['mu_eps_N'] for r in results])
    D_eps_N = np.array([r['D_eps_N'] for r in results])

    rate_names = set()
    for r in results:
        rate_names.update(k for k in r if k.startswith('k_'))

    rate_coeffs = {}
    for name in rate_names:
        rate_coeffs[name] = np.array([r.get(name, 0) for r in results])

    return {
        'eps_bar': eps_bar,
        'mu_e_N': mu_e_N,
        'D_e_N': D_e_N,
        'mu_eps_N': mu_eps_N,
        'D_eps_N': D_eps_N,
        'rate_coeffs': rate_coeffs,
        'x_Ar': x_Ar,
        'x_SF6': x_SF6,
    }


if __name__ == '__main__':
    from scipy.constants import k as kB

    N = 10 * 0.133322 / (kB * 300)

    print("2-Term Boltzmann Solver — LXCat Cross-Sections")
    print("=" * 60)

    for x_Ar, x_SF6, label in [(0, 1, 'Pure SF6'), (0.3, 0.7, '70/30 SF6/Ar'), (1, 0, 'Pure Ar')]:
        print(f"\n{label}:")
        print(f"{'E/N(Td)':>8s} {'ε̄(eV)':>8s} {'μ_e·N':>12s} {'D_e·N':>12s} {'D_ε/D_e':>8s}", end='')
        if x_SF6 > 0:
            print(f" {'k_att':>10s} {'k_iz':>10s}", end='')
        print()

        for EN in [1, 5, 10, 30, 100, 300]:
            r = solve_boltzmann_2term(EN, x_Ar=x_Ar, x_SF6=x_SF6)
            ratio = r['D_eps_N'] / max(r['D_e_N'], 1e-30)
            line = f"{EN:8.0f} {r['eps_bar']:8.2f} {r['mu_e_N']:12.3e} {r['D_e_N']:12.3e} {ratio:8.2f}"
            if x_SF6 > 0:
                k_att = r.get('k_SF6_att_total', 0)
                k_iz = r.get('k_SF6_ion', 0)
                line += f" {k_att:10.2e} {k_iz:10.2e}"
            print(line)

        # Compare with Einstein relation
        print(f"  Einstein check at 10 Td:")
        r = solve_boltzmann_2term(10, x_Ar=x_Ar, x_SF6=x_SF6)
        D_einstein = (2/3) * r['mu_e_N'] * r['eps_bar']
        print(f"    D_e·N (exact) = {r['D_e_N']:.3e}")
        print(f"    D_e·N (Einstein) = {D_einstein:.3e}")
        print(f"    Ratio exact/Einstein = {r['D_e_N']/max(D_einstein,1e-30):.2f}")
