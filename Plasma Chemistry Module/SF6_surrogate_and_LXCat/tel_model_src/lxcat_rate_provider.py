"""
LXCat-backed rate provider — drop-in replacement for sf6_rates.rates().

Precomputes Maxwellian-averaged rate tables from LXCat cross sections at
startup, then provides fast interpolated lookups at any Te.

Usage:
    from lxcat_rate_provider import lxcat_rates
    k = lxcat_rates(Te=3.0)  # same dict interface as sf6_rates.rates()
"""
import os
import numpy as np
from scipy.interpolate import interp1d
from lxcat_parser import parse_lxcat
from lxcat_rates import LXCatRateCalculator

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))
SF6_PATH = os.path.join(REPO, 'data', 'lxcat', 'SF6_Biagi_full.txt')

cm3 = 1e-6  # legacy uses this factor

# ═══════════════════════════════════════════════════════════════
# Precompute tables at import time
# ═══════════════════════════════════════════════════════════════

_Te_grid = np.linspace(0.3, 15.0, 200)
_calc = None
_k_iz_interp = None
_k_att_interp = None
_k_exc_interp = None
_initialized = False

def _initialize():
    global _calc, _k_iz_interp, _k_att_interp, _k_exc_interp, _initialized
    if _initialized:
        return
    _calc = LXCatRateCalculator(SF6_PATH)

    k_iz = np.array([_calc.rate_coefficient('IONIZATION', Te) for Te in _Te_grid])
    k_att = np.array([_calc.rate_coefficient('ATTACHMENT', Te) for Te in _Te_grid])
    k_exc = np.array([_calc.rate_coefficient('EXCITATION', Te) for Te in _Te_grid])

    _k_iz_interp = interp1d(_Te_grid, k_iz, kind='cubic', fill_value='extrapolate')
    _k_att_interp = interp1d(_Te_grid, k_att, kind='cubic', fill_value='extrapolate')
    _k_exc_interp = interp1d(_Te_grid, k_exc, kind='cubic', fill_value='extrapolate')
    _initialized = True


def lxcat_rates(Te):
    """Compute rate coefficients from LXCat cross sections.

    Returns a dict with the SAME keys as sf6_rates.rates() so it
    is a drop-in replacement for the solver. Channels not available
    from LXCat are filled with legacy Arrhenius fallbacks.

    Parameters
    ----------
    Te : float
        Electron temperature in eV.

    Returns
    -------
    dict
        Rate coefficients in m³/s (SI).
    """
    _initialize()
    Te = max(Te, 0.3)
    k = {}

    # ── LXCat-derived total rates ──
    k_iz_total = float(max(_k_iz_interp(Te), 0))
    k_att_total = float(max(_k_att_interp(Te), 0))

    # ── Distribute total ionization across channels proportionally ──
    # LXCat gives us total; we split using legacy channel ratios
    from sf6_rates import rates as legacy_rates
    leg = legacy_rates(Te)

    leg_iz_total = leg['iz_SF6_total']
    leg_att_total = leg['att_SF6_total']

    # Scale each channel by the LXCat/legacy ratio
    iz_scale = k_iz_total / max(leg_iz_total, 1e-30) if leg_iz_total > 1e-30 else 1.0
    att_scale = k_att_total / max(leg_att_total, 1e-30) if leg_att_total > 1e-30 else 1.0

    # Copy ALL legacy keys and scale the electron-impact channels
    for key, val in leg.items():
        k[key] = val

    # Scale ionization channels by LXCat ratio
    for iz_key in ['iz18','iz19','iz20','iz21','iz22','iz23','iz24',
                    'iz25','iz26','iz27','iz28','iz29']:
        if iz_key in k:
            k[iz_key] = leg[iz_key] * iz_scale

    # Scale attachment channels by LXCat ratio
    for att_key in ['at30','at31','at32','at33','at34','at35','at36']:
        if att_key in k:
            k[att_key] = leg[att_key] * att_scale

    # Recompute totals
    k['iz_SF6_total'] = sum(k[f'iz{i}'] for i in [18,19,20,21,22,23,24])
    k['att_SF6_total'] = sum(k[f'at{i}'] for i in [30,31,32,33,34,35,36])

    # Dissociation, excitation, elastic, neutral recombination, Ar:
    # FALLBACK to legacy (LXCat excitation includes all channels but
    # we cannot cleanly separate d1-d11 from the LXCat data without
    # channel-by-channel cross sections for dissociative excitation)

    return k


# Metadata for provenance tracking
LXCAT_FALLBACK_CHANNELS = [
    'd1','d2','d3','d4','d5','d6','d7','d8','d9','d10','d11',
    'vib_SF6','el_SF6','exc_F','el_F','vib_F2','el_F2',
    'nr37','nr38','nr39','nr40','nr41','nr42','nr43','nr44','nr45',
    'rec','Ar_iz','Ar_exc','Ar_iz_m','Ar_q','Ar_el',
    'Penn_SF6','qnch_SF6','qnch_SFx','qnch_F2','qnch_F',
]
LXCAT_REPLACED_CHANNELS = [
    'iz18','iz19','iz20','iz21','iz22','iz23','iz24',
    'iz25','iz26','iz27','iz28','iz29',
    'at30','at31','at32','at33','at34','at35','at36',
    'iz_SF6_total','att_SF6_total',
]
