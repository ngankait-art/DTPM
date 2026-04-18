"""
LXCat-backed rate coefficient computation.

Computes Maxwellian-averaged rate coefficients from LXCat cross sections:

    k(Te) = ∫ σ(E) v(E) f_M(E, Te) dE

where f_M is the Maxwellian EEDF and v = sqrt(2E/m_e).

Usage:
    from lxcat_rates import LXCatRateCalculator
    calc = LXCatRateCalculator('data/lxcat/SF6_Biagi_full.txt')
    k_iz = calc.rate_coefficient('IONIZATION', Te_eV=3.0)
    k_att = calc.rate_coefficient('ATTACHMENT', Te_eV=3.0)
"""
from __future__ import annotations
import os
import numpy as np
from scipy.constants import electron_mass as m_e, e as eC
from lxcat_parser import parse_lxcat, LXCatData


class LXCatRateCalculator:
    """Compute rate coefficients from LXCat cross-section data."""

    def __init__(self, filepath: str):
        self.data = parse_lxcat(filepath)
        # Common energy grid for integration (fine, log-spaced)
        self.E_grid = np.logspace(-4, np.log10(500), 2000)  # eV

    def maxwellian_eedf(self, E: np.ndarray, Te: float) -> np.ndarray:
        """Normalised Maxwellian EEDF: f(E) = (2/√π) (1/Te)^(3/2) √E exp(-E/Te).

        Normalised so ∫ f(E) dE = 1.
        """
        if Te <= 0:
            return np.zeros_like(E)
        return (2.0 / np.sqrt(np.pi)) * (1.0 / Te)**1.5 * np.sqrt(E) * np.exp(-E / Te)

    def rate_coefficient(self, reaction_type: str, Te_eV: float) -> float:
        """Compute Maxwellian-averaged rate coefficient k(Te).

        k = ∫ σ(E) v(E) f(E) dE

        where v(E) = sqrt(2*E*eC / m_e).

        Parameters
        ----------
        reaction_type : str
            'IONIZATION', 'ATTACHMENT', 'EXCITATION', 'ELASTIC'
        Te_eV : float
            Electron temperature in eV.

        Returns
        -------
        float
            Rate coefficient in m³/s.
        """
        E = self.E_grid
        sigma_total = self.data.total_cross_section(reaction_type, E)
        f = self.maxwellian_eedf(E, Te_eV)
        v = np.sqrt(2.0 * E * eC / m_e)  # m/s

        # Trapezoidal integration
        integrand = sigma_total * v * f
        k = np.trapezoid(integrand, E)
        return float(k)

    def rate_table(self, reaction_type: str,
                   Te_range: np.ndarray = None) -> dict:
        """Compute rate coefficient table over a Te range.

        Returns dict with 'Te_eV' and 'k_m3s' arrays.
        """
        if Te_range is None:
            Te_range = np.logspace(-1, np.log10(15), 60)

        k_vals = np.array([self.rate_coefficient(reaction_type, Te)
                           for Te in Te_range])
        return {'Te_eV': Te_range, 'k_m3s': k_vals}

    def all_rates(self, Te_eV: float) -> dict:
        """Compute all available rate coefficients at a given Te."""
        result = {}
        for rtype in ['IONIZATION', 'ATTACHMENT', 'EXCITATION', 'ELASTIC']:
            procs = self.data.by_type(rtype)
            if procs:
                result[f'k_{rtype.lower()}'] = self.rate_coefficient(rtype, Te_eV)
        return result


def compare_with_legacy(Te_range: np.ndarray = None):
    """Compare LXCat rates against legacy Arrhenius fits from sf6_rates.py.

    Returns comparison dict.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..'))

    sf6_path = os.path.join(repo, 'data', 'lxcat', 'SF6_Biagi_full.txt')
    if not os.path.exists(sf6_path):
        raise FileNotFoundError(f"LXCat file not found: {sf6_path}")

    calc = LXCatRateCalculator(sf6_path)

    if Te_range is None:
        Te_range = np.linspace(0.5, 10.0, 40)

    # LXCat rates
    k_iz_lxcat = np.array([calc.rate_coefficient('IONIZATION', Te) for Te in Te_range])
    k_att_lxcat = np.array([calc.rate_coefficient('ATTACHMENT', Te) for Te in Te_range])

    # Legacy Arrhenius (from sf6_rates.py)
    from sf6_rates import rates
    cm3 = 1e-6
    k_iz_legacy = np.array([sum(rates(Te)[f'iz{i}'] for i in [18,19,20,21,22,23,24])
                             for Te in Te_range])
    k_att_legacy = np.array([sum(rates(Te)[f'at{i}'] for i in [30,31,32,33,34,35,36])
                              for Te in Te_range])

    return {
        'Te_eV': Te_range,
        'k_iz_lxcat': k_iz_lxcat,
        'k_iz_legacy': k_iz_legacy,
        'k_att_lxcat': k_att_lxcat,
        'k_att_legacy': k_att_legacy,
        'iz_ratio': np.where(k_iz_legacy > 1e-30, k_iz_lxcat / k_iz_legacy, np.nan),
        'att_ratio': np.where(k_att_legacy > 1e-30, k_att_lxcat / k_att_legacy, np.nan),
    }


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..'))

    sf6_path = os.path.join(repo, 'data', 'lxcat', 'SF6_Biagi_full.txt')
    calc = LXCatRateCalculator(sf6_path)

    print("LXCat SF6 rate coefficients:")
    for Te in [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
        rates_all = calc.all_rates(Te)
        print(f"  Te={Te:.1f} eV: " +
              " ".join(f"{k}={v:.2e}" for k, v in rates_all.items()))
