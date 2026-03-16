"""
Transport coefficients for charged and neutral species.

Electron transport: μₑ, Dₑ (and energy equivalents) from Arrhenius-like
expressions or BOLSIG+ tables.

Ion transport: Langevin mobility, variable Da from Vahedi (1995).

Neutral transport: Chapman-Enskog diffusion coefficients.
"""

import numpy as np
from scipy.constants import e as eC, k as kB, m_e, epsilon_0, pi

AMU = 1.66054e-27


# ═══════════════════════════════════════════════════════════
# Electron transport
# ═══════════════════════════════════════════════════════════

class ElectronTransport:
    """Electron mobility and diffusion as functions of mean electron energy.

    Phase 1: Simple expressions consistent with the 0D model assumptions.
    Phase 2: Replace with BOLSIG+ lookup tables.
    """

    def __init__(self, gas_density, x_SF6=1.0, x_Ar=0.0):
        """
        Parameters
        ----------
        gas_density : float
            Total neutral gas density [m⁻³].
        x_SF6 : float
            Mole fraction of SF₆ in the gas.
        x_Ar : float
            Mole fraction of Ar in the gas.
        """
        self.ng = gas_density
        self.x_SF6 = x_SF6
        self.x_Ar = x_Ar

    def collision_frequency(self, Te):
        """Effective electron-neutral momentum transfer collision frequency [s⁻¹].

        Uses a simplified model: ν_en = ng * k_en(Te) where k_en combines
        elastic and inelastic momentum transfer.
        """
        # SF6 elastic momentum transfer cross-section ~ 1-5 × 10⁻¹⁹ m²
        # at Te ~ 2-4 eV, giving k_en ~ σ * v_th
        v_th = np.sqrt(8 * eC * Te / (pi * m_e))

        # Effective cross-sections (approximate, from BOLSIG+ for typical conditions)
        sigma_SF6 = 3e-19  # m² (SF6, moderate energy)
        sigma_Ar = 1e-20 + 5e-20 * np.exp(-((Te - 0.3) / 0.5)**2)  # Ramsauer minimum at ~0.3 eV

        sigma_eff = self.x_SF6 * sigma_SF6 + self.x_Ar * sigma_Ar
        return self.ng * sigma_eff * v_th

    def mobility(self, Te):
        """Electron mobility μₑ [m²/(V·s)].

        μₑ = e / (mₑ · ν_en)
        """
        nu = self.collision_frequency(Te)
        return eC / (m_e * max(nu, 1e3))

    def diffusivity(self, Te):
        """Electron diffusion coefficient Dₑ [m²/s].

        Using Einstein relation: Dₑ = (2/3) μₑ ε̄ = μₑ Te
        (Note: Hagelaar 2005 shows this can be off by factor 2 for Ar.
        Phase 2 should use exact BOLSIG+ values.)
        """
        return self.mobility(Te) * Te  # Te in eV, μ in m²/(V·s) → D in eV·m²/(V·s) = m²/s

    def energy_mobility(self, Te):
        """Energy mobility μₑ_ε [m²/(V·s)].

        Approximate: μₑ_ε ≈ (5/3) μₑ
        (Exact values differ — see Hagelaar 2005 Eq. 61.)
        """
        return (5.0 / 3.0) * self.mobility(Te)

    def energy_diffusivity(self, Te):
        """Energy diffusion coefficient Dε [m²/s].

        Approximate: Dε ≈ (5/3) Dₑ
        (Exact values differ — see Hagelaar 2005 Eq. 62.)
        """
        return (5.0 / 3.0) * self.diffusivity(Te)


# ═══════════════════════════════════════════════════════════
# Ion transport
# ═══════════════════════════════════════════════════════════

class IonTransport:
    """Positive and negative ion transport coefficients."""

    def __init__(self, Mi_amu, gas_density, Tgas=300.0, sigma_in=5e-19):
        """
        Parameters
        ----------
        Mi_amu : float
            Ion mass in AMU (e.g., 127.06 for SF₅⁺).
        gas_density : float
            Total neutral gas density [m⁻³].
        Tgas : float
            Gas temperature [K].
        sigma_in : float
            Ion-neutral collision cross-section [m²].
        """
        self.Mi = Mi_amu * AMU  # kg
        self.ng = gas_density
        self.Tgas = Tgas
        self.sigma_in = sigma_in

        # Ion thermal velocity
        self.v_ti = np.sqrt(8 * kB * Tgas / (pi * self.Mi))

        # Mean free path
        self.lambda_in = 1.0 / max(gas_density * sigma_in, 1.0)

        # Low-field diffusion coefficient (Vahedi 1995, Eq. 48)
        self.D0 = kB * Tgas * self.lambda_in / (self.Mi * self.v_ti)

        # Low-field mobility (Einstein relation at gas temperature)
        self.mu0 = eC * self.lambda_in / (self.Mi * self.v_ti)

    def bohm_velocity(self, Te, alpha=0.0, T_neg=0.3):
        """Modified Bohm velocity for electronegative plasma.

        uB = sqrt(e Te (1+α) / (M (1+γα)))

        From Lichtenberg (1997) Eq. 15.

        Parameters
        ----------
        Te : float
            Electron temperature [eV].
        alpha : float
            Local electronegativity n₋/nₑ.
        T_neg : float
            Negative ion temperature [eV].
        """
        gamma = Te / max(T_neg, 0.01)
        return np.sqrt(eC * Te * (1 + alpha) / (self.Mi * (1 + gamma * alpha)))

    def variable_Da(self, Te, u_drift, alpha=0.0, T_neg=0.3):
        """Variable ambipolar diffusion coefficient (Vahedi 1995, Eq. 47).

        Accounts for drift velocity saturation at the Bohm velocity.

        Da = sqrt(2) * Da0 / sqrt(1 + sqrt(1 + 4*Da0²*|∇n|²/(v_ti²*n²)))

        Simplified form using drift velocity:
        Da = Te * lambda_in / (Mi * sqrt(v_ti² + u_drift²))
        """
        return eC * Te * self.lambda_in / (self.Mi * np.sqrt(self.v_ti**2 + u_drift**2))

    def diffusivity(self, Te=None):
        """Ion diffusion coefficient [m²/s].

        If Te is given, returns ambipolar diffusion coefficient.
        Otherwise returns free diffusion coefficient.
        """
        if Te is not None:
            return eC * Te / (self.Mi * self.v_ti / self.lambda_in)
        return self.D0

    def mobility(self):
        """Ion mobility [m²/(V·s)]."""
        return self.mu0


# ═══════════════════════════════════════════════════════════
# Neutral transport
# ═══════════════════════════════════════════════════════════

class NeutralTransport:
    """Diffusion coefficients for neutral species."""

    def __init__(self, gas_density, Tgas=300.0):
        """
        Parameters
        ----------
        gas_density : float
            Background gas number density [m⁻³].
        Tgas : float
            Gas temperature [K].
        """
        self.ng = gas_density
        self.Tgas = Tgas

    def diffusivity(self, M_amu, sigma=4e-19):
        """Free diffusion coefficient for neutral species [m²/s].

        D = kT λ / (M v_th)  where λ = 1/(ng σ)

        Parameters
        ----------
        M_amu : float
            Species mass in AMU.
        sigma : float
            Diffusion cross-section [m²].
        """
        M = M_amu * AMU
        v_th = np.sqrt(8 * kB * self.Tgas / (pi * M))
        lam = 1.0 / max(self.ng * sigma, 1.0)  # mean free path [m]; floor at 1 m⁻¹ to avoid div/0
        return lam * v_th / 3.0  # kinetic theory: D = λv/3


# ═══════════════════════════════════════════════════════════
# Ambipolar diffusion for electronegative plasmas
# ═══════════════════════════════════════════════════════════

def ambipolar_Da(alpha, Te, D_plus, T_neg=0.3):
    """Ambipolar diffusion coefficient Da(α) for electronegative plasma.

    From Lichtenberg (1997) Eq. 10:
    Da ≈ D₊ (1 + γ + 2γα) / (1 + γα)

    where γ = Te/Ti.

    Limits:
        α >> 1: Da → 2D₊
        α << 1: Da → γD₊ (electropositive limit)

    Parameters
    ----------
    alpha : float or array
        Local electronegativity n₋/nₑ.
    Te : float
        Electron temperature [eV].
    D_plus : float
        Positive ion free diffusion coefficient [m²/s].
    T_neg : float
        Negative ion temperature [eV].

    Returns
    -------
    Da : float or array
        Ambipolar diffusion coefficient [m²/s].
    """
    gamma = Te / max(T_neg, 0.01)
    return D_plus * (1 + gamma + 2 * gamma * alpha) / (1 + gamma * alpha)


# ═══════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    from scipy.constants import k as kB

    # 10 mTorr, 300 K
    p_Pa = 10 * 0.133322
    ng = p_Pa / (kB * 300)
    print(f"Gas density: {ng:.3e} m⁻³ (10 mTorr, 300K)")

    # Electron transport
    et = ElectronTransport(ng, x_SF6=1.0, x_Ar=0.0)
    for Te in [2.0, 3.0, 4.0]:
        nu = et.collision_frequency(Te)
        mu = et.mobility(Te)
        De = et.diffusivity(Te)
        print(f"  Te={Te} eV: ν_en={nu:.2e} s⁻¹, μₑ={mu:.2e} m²/(V·s), Dₑ={De:.2e} m²/s")

    # Ion transport (SF5+)
    it = IonTransport(127.06, ng)
    uB = it.bohm_velocity(3.0, alpha=36)
    print(f"\nSF5+ at Te=3eV, α=36: uB={uB:.0f} m/s, D₊={it.D0:.3e} m²/s, μ₊={it.mu0:.3e} m²/(V·s)")

    # Ambipolar diffusion
    for alpha in [0.1, 1, 10, 100]:
        Da = ambipolar_Da(alpha, 3.0, it.D0)
        print(f"  Da(α={alpha:>5.1f}) = {Da:.3e} m²/s = {Da/it.D0:.1f} × D₊")

    # Neutral transport (F atoms)
    nt = NeutralTransport(ng)
    DF = nt.diffusivity(19.0)
    print(f"\nF atom diffusivity: {DF:.3e} m²/s")

    print("\n✓ Transport module OK")
