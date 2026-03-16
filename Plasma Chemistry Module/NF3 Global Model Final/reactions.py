"""
NF3/Ar ICP Global Model - Reaction Database
============================================
Source: Huang et al. 2026, Plasma Sources Sci. Technol. 35, 015019
"Study for Ar/NF3 plasma chemistries in inductively coupled discharges
 with global model and experiment"

All rate coefficients in m^3/s (two-body) or m^6/s (three-body).
Electron temperature Te in eV.
Gas temperature Tg in K, Tng = Tg/300.

Rate coefficient form: k(Te) = A * Te^B * exp(-C / Te^D)
  where D defaults to 1 if not specified.

For heavy-particle reactions, rate coefficients may depend on Tg.
"""

import numpy as np

# =============================================================================
# Species list
# =============================================================================
# Neutrals (10): NF3, NF2, NF, F2, N2, F, N, Ar, Ar_1s5, Ar_4p
# Positive ions (8): NF3+, NF2+, NF+, F2+, N2+, F+, N+, Ar+
# Negative ions (2): F-, F2-
# Electrons: e

NEUTRAL_SPECIES = ['NF3', 'NF2', 'NF', 'F2', 'N2', 'F', 'N', 'Ar', 'Ar_1s5', 'Ar_4p']
POSITIVE_IONS = ['NF3+', 'NF2+', 'NF+', 'F2+', 'N2+', 'F+', 'N+', 'Ar+']
NEGATIVE_IONS = ['F-', 'F2-']
ALL_SPECIES = NEUTRAL_SPECIES + POSITIVE_IONS + NEGATIVE_IONS + ['e']

# Species masses in kg (atomic mass units * 1.6605e-27)
AMU = 1.6605e-27
SPECIES_MASS = {
    'NF3': 71.0 * AMU, 'NF2': 52.0 * AMU, 'NF': 33.0 * AMU,
    'F2': 38.0 * AMU, 'N2': 28.0 * AMU, 'F': 19.0 * AMU,
    'N': 14.0 * AMU, 'Ar': 40.0 * AMU, 'Ar_1s5': 40.0 * AMU,
    'Ar_4p': 40.0 * AMU,
    'NF3+': 71.0 * AMU, 'NF2+': 52.0 * AMU, 'NF+': 33.0 * AMU,
    'F2+': 38.0 * AMU, 'N2+': 28.0 * AMU, 'F+': 19.0 * AMU,
    'N+': 14.0 * AMU, 'Ar+': 40.0 * AMU,
    'F-': 19.0 * AMU, 'F2-': 38.0 * AMU,
}

# =============================================================================
# Reaction database
# =============================================================================
# Each reaction is a dict with:
#   'id': reaction number from Table 1
#   'type': 'electron', 'penning_diss', 'penning_ion', 'neutral', 'ion_neutral',
#           'ion_ion', 'high_temp', 'three_body', 'surface'
#   'reactants': list of species strings (e.g. ['e', 'NF3'])
#   'products': list of species strings
#   'rate_coeff': callable function(Te, Tg, Tng) -> rate coefficient in m^3/s
#   'energy_loss': energy loss in eV (for electron impact reactions in power balance)
#   'description': string

def arrhenius_Te(A, B, C, D=1.0):
    """Rate coefficient k(Te) = A * Te^B * exp(-C / Te^D), Te in eV."""
    def k(Te, Tg=300.0, Tng=1.0):
        if Te <= 0:
            return 0.0
        return A * Te**B * np.exp(-C / Te**D)
    return k

def arrhenius_Tg(A, B_exp=0.0, Ea=0.0):
    """Rate coefficient k(Tg) = A * Tng^B_exp * exp(-Ea/Tg), Tng=Tg/300."""
    def k(Te=3.0, Tg=300.0, Tng=1.0):
        return A * Tng**B_exp * np.exp(-Ea / Tg)
    return k

def constant_rate(val):
    """Constant rate coefficient."""
    def k(Te=3.0, Tg=300.0, Tng=1.0):
        return val
    return k

# Ar elastic scattering has a special log-form rate coefficient
def ar_elastic_rate(Te, Tg=300.0, Tng=1.0):
    """Ar elastic: exp(-31.3879 + 1.6090*log(Te) + 0.0618*log(Te)^2 - 0.1171*log(Te)^3)"""
    if Te <= 0:
        return 0.0
    lt = np.log(Te)
    return np.exp(-31.3879 + 1.6090 * lt + 0.0618 * lt * lt - 0.1171 * lt * lt * lt)


# =============================================================================
# Build reaction list
# =============================================================================
REACTIONS = []

# --- Electron-NFx reactions (R1-R36) ---

# R1: e + NF3 -> NF3 + e (elastic)
REACTIONS.append({
    'id': 1, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['e', 'NF3'],
    'rate_coeff': arrhenius_Te(1.912e-13, 0.4511, 0.1435),
    'energy_loss': 'elastic_NF3',  # 3*Te*me/M_NF3
    'description': 'e + NF3 -> NF3 + e (elastic)'
})

# R2: e + NF3 -> NF3+ + 2e (ionization)
REACTIONS.append({
    'id': 2, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['NF3+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(4.76e-18, 3.734, 0.1615),
    'energy_loss': 15.09,
    'description': 'e + NF3 -> NF3+ + 2e'
})

# R3: e + NF3 -> F + NF2 + e (dissociation)
REACTIONS.append({
    'id': 3, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['F', 'NF2', 'e'],
    'rate_coeff': arrhenius_Te(1.699e-16, 1.901, 0.1049),
    'energy_loss': 2.52,
    'description': 'e + NF3 -> F + NF2 + e'
})

# R4: e + NF3 -> NF2 + F- (dissociative attachment)
REACTIONS.append({
    'id': 4, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['NF2', 'F-'],
    'rate_coeff': arrhenius_Te(6.424e-15, -0.6798, 0.0724),
    'energy_loss': 1.0,
    'description': 'e + NF3 -> NF2 + F-'
})

# R5: e + NF3 -> NF + F2- (dissociative attachment to F2-)
REACTIONS.append({
    'id': 5, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['NF', 'F2-'],
    'rate_coeff': arrhenius_Te(9.668e-18, -0.5818, 0.0787),
    'energy_loss': 2.0,
    'description': 'e + NF3 -> NF + F2-'
})

# R6: e + NF3 -> 2F + NF + e (double dissociation)
REACTIONS.append({
    'id': 6, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['F', 'F', 'NF', 'e'],
    'rate_coeff': arrhenius_Te(2.1884e-17, 3.195, 0.1781),
    'energy_loss': 5.82,
    'description': 'e + NF3 -> 2F + NF + e'
})

# R7: e + NF3 -> 2F + NF+ + 2e
REACTIONS.append({
    'id': 7, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['F', 'F', 'NF+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(8.676e-20, 5.423, 0.2402),
    'energy_loss': 22.92,
    'description': 'e + NF3 -> 2F + NF+ + 2e'
})

# R8: e + NF3 -> F+ + NF2 + 2e
REACTIONS.append({
    'id': 8, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['F+', 'NF2', 'e', 'e'],
    'rate_coeff': arrhenius_Te(7.344e-22, 7.062, 0.3056),
    'energy_loss': 17.52,
    'description': 'e + NF3 -> F+ + NF2 + 2e'
})

# R9: e + NF3 -> NF2+ + F + 2e
REACTIONS.append({
    'id': 9, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['NF2+', 'F', 'e', 'e'],
    'rate_coeff': arrhenius_Te(5.932e-18, 3.818, 0.1756),
    'energy_loss': 12.85,
    'description': 'e + NF3 -> NF2+ + F + 2e'
})

# R10: e + NF3 -> NF+ + F2 + 2e
REACTIONS.append({
    'id': 10, 'type': 'electron',
    'reactants': ['e', 'NF3'], 'products': ['NF+', 'F2', 'e', 'e'],
    'rate_coeff': arrhenius_Te(3.046e-19, 4.787, 0.1941),
    'energy_loss': 19.62,
    'description': 'e + NF3 -> NF+ + F2 + 2e'
})

# R11-R18: Dissociative excitation channels of NF3 (energy loss only, products -> ground state)
for rid, desc, A, B, C, eloss in [
    (11, 'e + NF3 -> NF(b1Sig+) + F2 + e', 3.869e-17, 2.62, 0.2501, 4.84),
    (12, 'e + NF3 -> NF(B3Sig+) + F2 + e', 4.492e-17, 2.33, 0.2417, 4.84),
    (13, 'e + NF3 -> NF2(A2A1) + F + e', 9.760e-18, 2.821, 0.2615, 7.72),
    (14, 'e + NF3 -> NF2(a4A1) + F + e', 7.936e-18, 2.599, 0.2561, 7.72),
    (15, 'e + NF3 -> NF2(B2A1) + F + e', 2.745e-19, 3.03, 0.2698, 9.77),
    (16, 'e + NF3 -> NF2(b4A1) + F + e', 5.584e-19, 3.025, 0.2698, 9.77),
    (17, 'e + NF3 -> NF2(C2B2) + F + e', 4.4448e-19, 2.92, 0.2671, 9.94),
    (18, 'e + NF3 -> NF2(c4B2) + F + e', 9.072e-19, 2.919, 0.2671, 9.94),
]:
    # These excited states are not tracked - treat as dissociation to ground state products
    # R11-R12: NF3 -> NF + F2 (via excited NF)
    # R13-R18: NF3 -> NF2 + F (via excited NF2)
    if rid <= 12:
        prods = ['NF', 'F2', 'e']
    else:
        prods = ['NF2', 'F', 'e']
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'NF3'], 'products': prods,
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# R19: e + NF2 -> NF2 + e (elastic)
REACTIONS.append({
    'id': 19, 'type': 'electron',
    'reactants': ['e', 'NF2'], 'products': ['e', 'NF2'],
    'rate_coeff': arrhenius_Te(2.401e-13, 0.269, 0.09142),
    'energy_loss': 'elastic_NF2',
    'description': 'e + NF2 -> NF2 + e (elastic)'
})

# R20: e + NF2 -> NF2+ + 2e
REACTIONS.append({
    'id': 20, 'type': 'electron',
    'reactants': ['e', 'NF2'], 'products': ['NF2+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(6.836e-18, 3.781, 0.1464),
    'energy_loss': 10.33,
    'description': 'e + NF2 -> NF2+ + 2e'
})

# R21: e + NF2 -> NF + F-
REACTIONS.append({
    'id': 21, 'type': 'electron',
    'reactants': ['e', 'NF2'], 'products': ['NF', 'F-'],
    'rate_coeff': arrhenius_Te(5.804e-16, -0.851, 0.05713),
    'energy_loss': 0.5,
    'description': 'e + NF2 -> NF + F-'
})

# R22: e + NF2 -> F + NF + e
REACTIONS.append({
    'id': 22, 'type': 'electron',
    'reactants': ['e', 'NF2'], 'products': ['F', 'NF', 'e'],
    'rate_coeff': arrhenius_Te(3.2141e-16, 2.028, 0.155),
    'energy_loss': 3.30,
    'description': 'e + NF2 -> F + NF + e'
})

# R23-R25: Dissociative ionization of NF2
for rid, desc, A, B, C, eloss, prods in [
    (23, 'e + NF2 -> NF+ + F + 2e', 1.6916e-18, 3.99, 0.159, 20.4, ['NF+', 'F', 'e', 'e']),
    (24, 'e + NF2 -> N+ + F2 + 2e', 9.648e-21, 5.333, 0.2098, 17.83, ['N+', 'F2', 'e', 'e']),
    (25, 'e + NF2 -> F+ + NF + 2e', 1.0448e-19, 4.505, 0.1907, 18.30, ['F+', 'NF', 'e', 'e']),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'NF2'], 'products': prods,
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# R26-R28: Dissociative excitation of NF2 (excited states -> ground state products)
for rid, desc, A, B, C, eloss in [
    (26, 'e + NF2 -> NF(3Sig-) + F + e', 5.224e-16, 2.062, 0.227, 3.30),
    (27, 'e + NF2 -> NF(1Delta) + F + e', 4.016e-17, 2.296, 0.2386, 4.63),
    (28, 'e + NF2 -> NF(1Sig+) + F + e', 3.434e-17, 2.610, 0.2525, 5.62),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'NF2'], 'products': ['NF', 'F', 'e'],
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# R29: e + NF -> NF + e (elastic)
REACTIONS.append({
    'id': 29, 'type': 'electron',
    'reactants': ['e', 'NF'], 'products': ['e', 'NF'],
    'rate_coeff': arrhenius_Te(4.080e-13, 0.0805, 0.07059),
    'energy_loss': 'elastic_NF',
    'description': 'e + NF -> NF + e (elastic)'
})

# R30: e + NF -> NF+ + 2e
REACTIONS.append({
    'id': 30, 'type': 'electron',
    'reactants': ['e', 'NF'], 'products': ['NF+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(2.383e-17, 3.527, 0.1525),
    'energy_loss': 17.10,
    'description': 'e + NF -> NF+ + 2e'
})

# R31: e + NF -> N + F-
REACTIONS.append({
    'id': 31, 'type': 'electron',
    'reactants': ['e', 'NF'], 'products': ['N', 'F-'],
    'rate_coeff': arrhenius_Te(1.452e-18, -0.7247, 0.06644),
    'energy_loss': 0.6,
    'description': 'e + NF -> N + F-'
})

# R32: e + NF -> F + N + e
REACTIONS.append({
    'id': 32, 'type': 'electron',
    'reactants': ['e', 'NF'], 'products': ['F', 'N', 'e'],
    'rate_coeff': arrhenius_Te(3.462e-16, 1.95, 0.1446),
    'energy_loss': 3.12,
    'description': 'e + NF -> F + N + e'
})

# R33-R34: Dissociative ionization of NF
for rid, desc, A, B, C, eloss, prods in [
    (33, 'e + NF -> F+ + N + 2e', 3.424e-19, 4.37, 0.1891, 18.12, ['F+', 'N', 'e', 'e']),
    (34, 'e + NF -> N+ + F + 2e', 9.604e-19, 4.044, 0.1705, 17.65, ['N+', 'F', 'e', 'e']),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'NF'], 'products': prods,
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# R35-R36: Dissociative excitation of NF
for rid, desc, A, B, C, eloss in [
    (35, 'e + NF -> N(4S) + F + e', 5.272e-17, 2.345, 0.2471, 3.12),
    (36, 'e + NF -> N(2D) + F + e', 1.779e-16, 2.209, 0.2401, 5.50),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'NF'], 'products': ['N', 'F', 'e'],
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# --- Electron-Fx reactions (R37-R49) ---

# R37: e + F2 -> F2 + e (elastic)
REACTIONS.append({
    'id': 37, 'type': 'electron',
    'reactants': ['e', 'F2'], 'products': ['e', 'F2'],
    'rate_coeff': arrhenius_Te(6.002e-13, -0.627, 1.538),
    'energy_loss': 'elastic_F2',
    'description': 'e + F2 -> F2 + e (elastic)'
})

# R38: e + F2 -> F + F-
REACTIONS.append({
    'id': 38, 'type': 'electron',
    'reactants': ['e', 'F2'], 'products': ['F', 'F-'],
    'rate_coeff': arrhenius_Te(1.124e-15, -1.475, 0.535),
    'energy_loss': 1.61,
    'description': 'e + F2 -> F + F-'
})

# R39: e + F2 -> F2+ + 2e
REACTIONS.append({
    'id': 39, 'type': 'electron',
    'reactants': ['e', 'F2'], 'products': ['F2+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(6.71e-16, 1.24, 16.822),
    'energy_loss': 15.7,
    'description': 'e + F2 -> F2+ + 2e'
})

# R40: e + F2 -> F + F + e
REACTIONS.append({
    'id': 40, 'type': 'electron',
    'reactants': ['e', 'F2'], 'products': ['F', 'F', 'e'],
    'rate_coeff': arrhenius_Te(4.983e-15, 0.4, 3.947),
    'energy_loss': 14.59,
    'description': 'e + F2 -> F + F + e'
})

# R41: e + F2 -> F+ + F- + e
REACTIONS.append({
    'id': 41, 'type': 'electron',
    'reactants': ['e', 'F2'], 'products': ['F+', 'F-', 'e'],
    'rate_coeff': arrhenius_Te(9.432e-17, 1.341, 17.322),
    'energy_loss': 39.79,
    'description': 'e + F2 -> F+ + F- + e'
})

# R42-R44: Excitation of F2 (energy loss only)
for rid, desc, A, B, C, eloss in [
    (42, 'e + F2 -> F2(v) + e', 2.586e-13, -1.446, 0.889, 0.25956),
    (43, 'e + F2 -> F2(C1Sig_u) + e', 3.929e-15, 0.665, 11.842, 12.9),
    (44, 'e + F2 -> F2(H1Pi_u) + e', 9.616e-16, 0.344, 17.274, 14.4),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'F2'], 'products': ['F2', 'e'],
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# R45: e + F -> F + e (elastic)
REACTIONS.append({
    'id': 45, 'type': 'electron',
    'reactants': ['e', 'F'], 'products': ['e', 'F'],
    'rate_coeff': arrhenius_Te(3.08e-14, -0.547, 0.412),
    'energy_loss': 'elastic_F',
    'description': 'e + F -> F + e (elastic)'
})

# R46: e + F -> F+ + 2e
REACTIONS.append({
    'id': 46, 'type': 'electron',
    'reactants': ['e', 'F'], 'products': ['F+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(4.711e-16, 1.411, 12.618),
    'energy_loss': 15.0,
    'description': 'e + F -> F+ + 2e'
})

# R47-R49: Excitation of F (energy loss only)
for rid, desc, A, B, C, eloss in [
    (47, 'e + F -> F(3s 4P) + e', 1.667e-15, -0.581, 12.851, 12.818),
    (48, 'e + F -> F(3s 2P) + e', 8.522e-16, 0.386, 12.351, 13.092),
    (49, 'e + F -> F(3s 2D) + e', 5.479e-16, 0.181, 14.558, 15.5),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'F'], 'products': ['F', 'e'],
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# --- Electron-Nx reactions (R50-R65) ---

# R50: e + N2 -> N2 + e (elastic)
REACTIONS.append({
    'id': 50, 'type': 'electron',
    'reactants': ['e', 'N2'], 'products': ['e', 'N2'],
    'rate_coeff': arrhenius_Te(2.615e-24, 0.4028, -0.003184),  # Note: positive exp arg
    'energy_loss': 'elastic_N2',
    'description': 'e + N2 -> N2 + e (elastic)'
})

# R51: e + N2 -> N + N + e
REACTIONS.append({
    'id': 51, 'type': 'electron',
    'reactants': ['e', 'N2'], 'products': ['N', 'N', 'e'],
    'rate_coeff': arrhenius_Te(1.340e-17, 3.572, 0.1666),
    'energy_loss': 18.5,
    'description': 'e + N2 -> N + N + e'
})

# R52-R53: Dissociative ionization of N2
REACTIONS.append({
    'id': 52, 'type': 'electron',
    'reactants': ['e', 'N2'], 'products': ['N', 'N+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(1.7136e-20, 5.762, 0.2391),
    'energy_loss': 33.03,
    'description': 'e + N2 -> N + N+ + 2e'
})

REACTIONS.append({
    'id': 53, 'type': 'electron',
    'reactants': ['e', 'N2'], 'products': ['N+', 'N+', 'e', 'e', 'e'],
    'rate_coeff': arrhenius_Te(8.10e-24, 8.126, 0.3162),
    'energy_loss': 47.56,
    'description': 'e + N2 -> N+ + N+ + 3e'
})

# R54: e + N2 -> N2+ + 2e  (RATE IN KELVIN - marked (k) in paper)
def _k54(Te, Tg=300.0, Tng=1.0):
    Te_K = Te * 11604.5  # convert eV to K
    return 8.58e-18 * Te_K**0.72 * np.exp(-184300.0 / Te_K)
REACTIONS.append({
    'id': 54, 'type': 'electron',
    'reactants': ['e', 'N2'], 'products': ['N2+', 'e', 'e'],
    'rate_coeff': _k54,
    'energy_loss': 15.58,
    'description': 'e + N2 -> N2+ + 2e'
})

# R55-R57: Excitation of N2 (RATES IN KELVIN - marked (k) in paper)
def _k55(Te, Tg=300.0, Tng=1.0):
    Te_K = Te * 11604.5
    return 5.48e-17 * Te_K**0.55 * np.exp(-57700.0 / Te_K)

def _k56(Te, Tg=300.0, Tng=1.0):
    Te_K = Te * 11604.5
    return 1.89e-16 * Te_K**0.50 * np.exp(-75288.0 / Te_K)

def _k57(Te, Tg=300.0, Tng=1.0):
    Te_K = Te * 11604.5
    return 8.86e-10 * Te_K**(-1.13) * np.exp(-124966.0 / Te_K)

for rid, kfunc, eloss, desc in [
    (55, _k55, 4.975, 'e + N2 -> N2(A) + e'),
    (56, _k56, 5.930, 'e + N2 -> N2(B) + e'),
    (57, _k57, 8.897, 'e + N2 -> N2(C) + e'),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'N2'], 'products': ['N2', 'e'],
        'rate_coeff': kfunc,
        'energy_loss': eloss,
        'description': desc
    })

# R58: e + N -> N + e (elastic)
REACTIONS.append({
    'id': 58, 'type': 'electron',
    'reactants': ['e', 'N'], 'products': ['e', 'N'],
    'rate_coeff': arrhenius_Te(5.90e-14, 0.6295, 0.1329),
    'energy_loss': 'elastic_N',
    'description': 'e + N -> N + e (elastic)'
})

# R59: e + N -> N+ + 2e
REACTIONS.append({
    'id': 59, 'type': 'electron',
    'reactants': ['e', 'N'], 'products': ['N+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(2.335e-17, 3.533, 0.1632),
    'energy_loss': 14.53,
    'description': 'e + N -> N+ + 2e'
})

# R60-R65: Excitation of N (energy loss only)
for rid, desc, A, B, C, eloss in [
    (60, 'e + N -> N(2D) + e', 4.708e-15, 0.6004, 0.08892, 2.38),
    (61, 'e + N -> N(2P) + e', 4.368e-16, 1.212, 0.1263, 3.57),
    (62, 'e + N -> N(3s4P) + e', 2.312e-17, 3.069, 0.155, 10.32),
    (63, 'e + N -> N(p4 4P) + e', 2.303e-17, 2.992, 0.1685, 10.92),
    (64, 'e + N -> N(4s4P) + e', 2.065e-18, 3.107, 0.1825, 12.84),
    (65, 'e + N -> N(3d4P) + e', 1.830e-18, 3.288, 0.1707, 12.99),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'N'], 'products': ['N', 'e'],
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# --- Electron-Ar reactions (R66-R77) ---

# R66: e + Ar -> Ar + e (elastic, special form)
REACTIONS.append({
    'id': 66, 'type': 'electron',
    'reactants': ['e', 'Ar'], 'products': ['e', 'Ar'],
    'rate_coeff': ar_elastic_rate,
    'energy_loss': 'elastic_Ar',
    'description': 'e + Ar -> Ar + e (elastic)'
})

# R67: e + Ar -> Ar+ + 2e
REACTIONS.append({
    'id': 67, 'type': 'electron',
    'reactants': ['e', 'Ar'], 'products': ['Ar+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(2.39e-14, 0.57, 17.43),
    'energy_loss': 15.76,
    'description': 'e + Ar -> Ar+ + 2e'
})

# R68-R72: Excitation of Ar
for rid, desc, A, B, C, eloss, prod in [
    (68, 'e + Ar -> Ar(1s5) + e', 9.73e-16, -0.07, 11.69, 11.54, 'Ar_1s5'),
    (69, 'e + Ar -> Ar(3P0) + e', 1.4e-15, 0, 12.42, 11.72, 'Ar'),
    (70, 'e + Ar -> Ar(3P1) + e', 1.9e-15, 0, 12.6, 11.62, 'Ar'),
    (71, 'e + Ar -> Ar(1P1) + e', 2.7e-16, 0, 12.14, 11.82, 'Ar'),
    (72, 'e + Ar -> Ar(4p) + e', 2.1e-14, 0, 13.13, 13.2, 'Ar_4p'),
]:
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', 'Ar'], 'products': [prod, 'e'],
        'rate_coeff': arrhenius_Te(A, B, C),
        'energy_loss': eloss,
        'description': desc
    })

# R73: e + Ar(1s5) -> Ar+ + 2e
REACTIONS.append({
    'id': 73, 'type': 'electron',
    'reactants': ['e', 'Ar_1s5'], 'products': ['Ar+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(2.71e-13, 0.26, 4.59),
    'energy_loss': 4.22,
    'description': 'e + Ar(1s5) -> Ar+ + 2e'
})

# R74: e + Ar(1s5) -> Ar(4p) + e
REACTIONS.append({
    'id': 74, 'type': 'electron',
    'reactants': ['e', 'Ar_1s5'], 'products': ['Ar_4p', 'e'],
    'rate_coeff': arrhenius_Te(8.9e-13, 0.51, 1.59),
    'energy_loss': 1.66,
    'description': 'e + Ar(1s5) -> Ar(4p) + e'
})

# R75: e + Ar(4p) -> Ar+ + 2e
REACTIONS.append({
    'id': 75, 'type': 'electron',
    'reactants': ['e', 'Ar_4p'], 'products': ['Ar+', 'e', 'e'],
    'rate_coeff': arrhenius_Te(1.8e-13, 0.61, 2.61),
    'energy_loss': 2.56,
    'description': 'e + Ar(4p) -> Ar+ + 2e'
})

# R76: e + Ar(4p) -> Ar(1s5) + e (superelastic)
REACTIONS.append({
    'id': 76, 'type': 'electron',
    'reactants': ['e', 'Ar_4p'], 'products': ['Ar_1s5', 'e'],
    'rate_coeff': arrhenius_Te(3.0e-13, 0.51, 0),
    'energy_loss': -1.66,  # Superelastic - gives energy TO electrons
    'description': 'e + Ar(4p) -> Ar(1s5) + e (superelastic)'
})

# R77: e + Ar(4p) -> Ar + e (superelastic)
REACTIONS.append({
    'id': 77, 'type': 'electron',
    'reactants': ['e', 'Ar_4p'], 'products': ['Ar', 'e'],
    'rate_coeff': arrhenius_Te(3.9e-13, 0.51, 0),
    'energy_loss': -13.2,  # Superelastic
    'description': 'e + Ar(4p) -> Ar + e (superelastic)'
})

# --- Electron-ion reactions (R78-R86) ---

# R78-R80: Dissociative recombination of NFx+
for rid, desc, A, B, prods in [
    (78, 'e + NF3+ -> NF2 + F', 1e-13, -0.5, ['NF2', 'F']),
    (79, 'e + NF2+ -> NF + F', 1e-13, -0.5, ['NF', 'F']),
    (80, 'e + NF+ -> N + F', 1e-13, -0.5, ['N', 'F']),
]:
    ion = desc.split('+')[0].split()[-1] + '+'
    REACTIONS.append({
        'id': rid, 'type': 'electron',
        'reactants': ['e', ion], 'products': prods,
        'rate_coeff': arrhenius_Te(A, B, 0),
        'energy_loss': 0,
        'description': desc
    })

# R81: e + F2+ -> F + F
REACTIONS.append({
    'id': 81, 'type': 'electron',
    'reactants': ['e', 'F2+'], 'products': ['F', 'F'],
    'rate_coeff': arrhenius_Te(3.21e-14, -0.5, 0),
    'energy_loss': 0,
    'description': 'e + F2+ -> F + F'
})

# R82: e + N2+ -> N + N
REACTIONS.append({
    'id': 82, 'type': 'electron',
    'reactants': ['e', 'N2+'], 'products': ['N', 'N'],
    'rate_coeff': arrhenius_Te(1.90e-15, -0.30, 0),
    'energy_loss': 0,
    'description': 'e + N2+ -> N + N'
})

# R83: e + F- -> F + 2e (detachment)
REACTIONS.append({
    'id': 83, 'type': 'electron',
    'reactants': ['e', 'F-'], 'products': ['F', 'e', 'e'],
    'rate_coeff': arrhenius_Te(3.047e-14, 0.413, 11.167),
    'energy_loss': 0,
    'description': 'e + F- -> F + 2e'
})

# R84: e + F2- -> F2 + 2e
REACTIONS.append({
    'id': 84, 'type': 'electron',
    'reactants': ['e', 'F2-'], 'products': ['F2', 'e', 'e'],
    'rate_coeff': arrhenius_Te(6.186e-15, -0.535, 11.092),
    'energy_loss': 0,
    'description': 'e + F2- -> F2 + 2e'
})

# R85: e + F2- -> F + F- + e
REACTIONS.append({
    'id': 85, 'type': 'electron',
    'reactants': ['e', 'F2-'], 'products': ['F', 'F-', 'e'],
    'rate_coeff': arrhenius_Te(5.812e-15, 0.062, 11.498),
    'energy_loss': 0,
    'description': 'e + F2- -> F + F- + e'
})

# R86: e + F2- -> F + F + e (dissociative detachment of F2-)
REACTIONS.append({
    'id': 86, 'type': 'electron',
    'reactants': ['e', 'F2-'], 'products': ['F', 'F', 'e'],
    'rate_coeff': arrhenius_Te(1.354e-14, 0.484, 2.178),
    'energy_loss': 0,
    'description': 'e + F2- -> F + F + e'
})

# --- Penning dissociation (R87-R89) ---
for rid, target, products, k_val in [
    (87, 'NF3', ['Ar', 'F', 'NF2'], 1.4e-16),
    (88, 'NF2', ['Ar', 'F', 'NF'], 1.4e-16),
    (89, 'NF', ['Ar', 'F', 'N'], 1.4e-16),
]:
    REACTIONS.append({
        'id': rid, 'type': 'penning_diss',
        'reactants': ['Ar_1s5', target], 'products': products,
        'rate_coeff': constant_rate(k_val),
        'energy_loss': 0,
        'description': f'Ar(1s5) + {target} -> Ar + F + ...'
    })

# --- Penning ionization (R90-R93) ---
REACTIONS.append({
    'id': 90, 'type': 'penning_ion',
    'reactants': ['Ar_4p', 'NF2'], 'products': ['Ar', 'NF2+', 'e'],
    'rate_coeff': arrhenius_Tg(1.2e-15, 0.5, 0),
    'energy_loss': 0,
    'description': 'Ar(4p) + NF2 -> Ar + NF2+ + e'
})

REACTIONS.append({
    'id': 91, 'type': 'penning_ion',
    'reactants': ['Ar_4p', 'NF'], 'products': ['Ar', 'NF+', 'e'],
    'rate_coeff': arrhenius_Tg(1.2e-15, 0.5, 0),
    'energy_loss': 0,
    'description': 'Ar(4p) + NF -> Ar + NF+ + e'
})

REACTIONS.append({
    'id': 92, 'type': 'penning_ion',
    'reactants': ['Ar_4p', 'Ar_4p'], 'products': ['Ar', 'Ar+', 'e'],
    'rate_coeff': constant_rate(5.0e-16),
    'energy_loss': 0,
    'description': 'Ar(4p) + Ar(4p) -> Ar + Ar+ + e'
})

REACTIONS.append({
    'id': 93, 'type': 'penning_ion',
    'reactants': ['Ar_1s5', 'Ar_1s5'], 'products': ['Ar', 'Ar+', 'e'],
    'rate_coeff': constant_rate(5.0e-16),
    'energy_loss': 0,
    'description': 'Ar(1s5) + Ar(1s5) -> Ar + Ar+ + e'
})

# --- Neutral-neutral reactions (R94-R102) ---
REACTIONS.append({
    'id': 94, 'type': 'neutral',
    'reactants': ['NF3', 'NF'], 'products': ['NF2', 'NF2'],
    'rate_coeff': constant_rate(1e-20),
    'energy_loss': 0,
    'description': 'NF3 + NF -> NF2 + NF2'
})

REACTIONS.append({
    'id': 95, 'type': 'neutral',
    'reactants': ['NF3', 'N'], 'products': ['NF2', 'NF'],
    'rate_coeff': arrhenius_Tg(2.13e-18, 1.97, 15120),
    'energy_loss': 0,
    'description': 'NF3 + N -> NF2 + NF'
})

REACTIONS.append({
    'id': 96, 'type': 'neutral',
    'reactants': ['NF2', 'NF2'], 'products': ['NF', 'NF3'],
    'rate_coeff': arrhenius_Tg(1.66e-18, 0, 18600),
    'energy_loss': 0,
    'description': 'NF2 + NF2 -> NF + NF3'
})

REACTIONS.append({
    'id': 97, 'type': 'neutral',
    'reactants': ['NF2', 'F2'], 'products': ['F', 'NF3'],
    'rate_coeff': arrhenius_Tg(3.0e-20, 0, 4860),
    'energy_loss': 0,
    'description': 'NF2 + F2 -> F + NF3'
})

REACTIONS.append({
    'id': 98, 'type': 'neutral',
    'reactants': ['NF2', 'N'], 'products': ['N2', 'F', 'F'],
    'rate_coeff': arrhenius_Tg(1.4e-17, 0, 95),
    'energy_loss': 0,
    'description': 'NF2 + N -> N2 + F + F'
})

REACTIONS.append({
    'id': 99, 'type': 'neutral',
    'reactants': ['NF2', 'N'], 'products': ['NF', 'NF'],
    'rate_coeff': constant_rate(3e-18),
    'energy_loss': 0,
    'description': 'NF2 + N -> NF + NF'
})

REACTIONS.append({
    'id': 100, 'type': 'neutral',
    'reactants': ['NF', 'NF'], 'products': ['N2', 'F', 'F'],
    'rate_coeff': arrhenius_Tg(6.88e-17, 0, 1251),
    'energy_loss': 0,
    'description': 'NF + NF -> N2 + F + F'
})

REACTIONS.append({
    'id': 101, 'type': 'neutral',
    'reactants': ['NF', 'NF'], 'products': ['N2', 'F2'],
    'rate_coeff': constant_rate(4e-18),
    'energy_loss': 0,
    'description': 'NF + NF -> N2 + F2'
})

REACTIONS.append({
    'id': 102, 'type': 'neutral',
    'reactants': ['NF', 'N'], 'products': ['N2', 'F'],
    'rate_coeff': constant_rate(2.5e-16),
    'energy_loss': 0,
    'description': 'NF + N -> N2 + F'
})

# --- Ion-neutral reactions (R103-R136) ---
# These are charge exchange reactions with constant rate coefficients
ION_NEUTRAL_DATA = [
    (103, 'NF3+', 'NF3', 'NF3', 'NF3+', 1.0e-15),
    (104, 'NF3+', 'NF2', 'NF3', 'NF2+', 1.0e-17),
    (105, 'NF3+', 'NF', 'NF3', 'NF+', 1.0e-17),
    (106, 'NF2+', 'NF2', 'NF2', 'NF2+', 1.0e-15),
    (107, 'NF+', 'NF3', 'NF2', 'NF2+', 5.5e-16),
    (108, 'NF+', 'NF2', 'NF', 'NF2+', 1.0e-17),
    (109, 'NF+', 'NF', 'NF', 'NF+', 1.0e-15),
    (110, 'F2+', 'NF3', 'F2', 'NF3+', 1.0e-17),
    (111, 'F2+', 'NF2', 'F2', 'NF2+', 1.0e-17),
    (112, 'F2+', 'NF', 'F2', 'NF+', 1.0e-17),
    (113, 'F2+', 'N2', 'F2', 'N2+', 1.0e-17),
    (114, 'F2+', 'N', 'F2', 'N+', 1.0e-17),
    (115, 'N2+', 'NF3', 'N2', 'NF3+', 1.0e-17),
    (116, 'N2+', 'NF2', 'N2', 'NF2+', 1.0e-17),
    (117, 'N2+', 'NF', 'N2', 'NF+', 1.0e-17),
    (118, 'N2+', 'N2', 'N2', 'N2+', 1.0e-15),
    (119, 'N2+', 'N', 'N2', 'N+', 8.0e-18),
    (120, 'F+', 'NF3', 'F', 'NF3+', 1.0e-17),
    (121, 'F+', 'NF2', 'F', 'NF2+', 1.0e-17),
    (122, 'F+', 'NF', 'F', 'NF+', 1.0e-17),
    (123, 'F+', 'F2', 'F2+', 'F', 7.94e-16),
    (124, 'F+', 'N2', 'F', 'N2+', 9.7e-16),
    (125, 'F+', 'N', 'F', 'N+', 1.0e-17),
    (126, 'N+', 'NF3', 'N', 'NF3+', 1.0e-17),
    (127, 'N+', 'NF2', 'N', 'NF2+', 1.0e-17),
    (128, 'N+', 'NF', 'N', 'NF+', 1.0e-17),
    (129, 'F-', 'F', 'F2', 'e_source', 1.0e-16),  # electron detachment
    (130, 'F-', 'F', 'F', 'F', 1.0e-16),  # + e produced
    (131, 'F2-', 'F', 'F2', 'F', 1.0e-16),  # + e produced
    (132, 'Ar+', 'NF3', 'Ar', 'NF3+', 1.0e-17),
    (133, 'Ar+', 'NF2', 'Ar', 'NF2+', 1.0e-17),
    (134, 'Ar+', 'NF', 'Ar', 'NF+', 1.0e-17),
    (135, 'Ar+', 'N2', 'Ar', 'N2+', 1.0e-17),
    (136, 'Ar+', 'N', 'Ar', 'N+', 1.0e-17),
]

for rid, ion, neutral, prod1, prod2, k_val in ION_NEUTRAL_DATA:
    # Special handling for detachment reactions (129-131) that produce electrons
    if rid == 129:
        REACTIONS.append({
            'id': 129, 'type': 'detachment',
            'reactants': ['F-', 'F'], 'products': ['F2', 'e'],
            'rate_coeff': constant_rate(1.0e-16),
            'energy_loss': 0,
            'description': 'F- + F -> F2 + e'
        })
    elif rid == 130:
        REACTIONS.append({
            'id': 130, 'type': 'detachment',
            'reactants': ['F-', 'F'], 'products': ['F', 'F', 'e'],
            'rate_coeff': constant_rate(1.0e-16),
            'energy_loss': 0,
            'description': 'F- + F -> F + F + e'
        })
    elif rid == 131:
        REACTIONS.append({
            'id': 131, 'type': 'detachment',
            'reactants': ['F2-', 'F'], 'products': ['F2', 'F', 'e'],
            'rate_coeff': constant_rate(1.0e-16),
            'energy_loss': 0,
            'description': 'F2- + F -> F2 + F + e'
        })
    else:
        REACTIONS.append({
            'id': rid, 'type': 'ion_neutral',
            'reactants': [ion, neutral], 'products': [prod1, prod2],
            'rate_coeff': constant_rate(k_val),
            'energy_loss': 0,
            'description': f'{ion} + {neutral} -> {prod1} + {prod2}'
        })

# --- Ion-ion neutralization reactions (R137-R145) ---
ION_ION_DATA = [
    (137, 'F-', 'NF3+', ['F', 'F', 'NF2'], 2e-13),
    (138, 'F-', 'NF2+', ['F', 'F', 'NF'], 2e-13),
    (139, 'F-', 'NF+', ['F', 'F', 'N'], 1e-13),
    (140, 'F-', 'NF3+', ['F', 'NF3'], 4.0e-13),
    (141, 'F-', 'NF2+', ['F', 'NF2'], 4.0e-13),
    (142, 'F-', 'NF+', ['F', 'NF'], 4.0e-13),
    (143, 'F-', 'F2+', ['F', 'F2'], 9.4e-14),
    (144, 'F-', 'F+', ['F', 'F'], 1.1e-13),
    (145, 'F-', 'Ar+', ['F', 'Ar_1s5'], 1.0e-13),
]

for rid, neg, pos, prods, k_val in ION_ION_DATA:
    REACTIONS.append({
        'id': rid, 'type': 'ion_ion',
        'reactants': [neg, pos], 'products': prods,
        'rate_coeff': constant_rate(k_val),
        'energy_loss': 0,
        'description': f'{neg} + {pos} -> {" + ".join(prods)}'
    })

# --- High temperature chemistry / thermal dissociation (R146-R150) ---
REACTIONS.append({
    'id': 146, 'type': 'high_temp',
    'reactants': ['F2', 'M'], 'products': ['F', 'F', 'M'],
    'rate_coeff': arrhenius_Tg(7.6e-18, 0, 14300),
    'energy_loss': 0,
    'description': 'F2 + M -> F + F + M'
})

REACTIONS.append({
    'id': 147, 'type': 'high_temp',
    'reactants': ['N2', 'M'], 'products': ['N', 'N', 'M'],
    'rate_coeff': arrhenius_Tg(9.86e-11, -3.33, 113220),
    'energy_loss': 0,
    'description': 'N2 + M -> N + N + M'
})

REACTIONS.append({
    'id': 148, 'type': 'high_temp',
    'reactants': ['NF3', 'M'], 'products': ['NF2', 'F', 'M'],
    'rate_coeff': arrhenius_Tg(3.98e-16, 0, 18417),
    'energy_loss': 0,
    'description': 'NF3 + M -> NF2 + F + M'
})

REACTIONS.append({
    'id': 149, 'type': 'high_temp',
    'reactants': ['NF2', 'M'], 'products': ['NF', 'F', 'M'],
    'rate_coeff': arrhenius_Tg(1.26e-15, 0, 25700),
    'energy_loss': 0,
    'description': 'NF2 + M -> NF + F + M'
})

REACTIONS.append({
    'id': 150, 'type': 'high_temp',
    'reactants': ['NF', 'M'], 'products': ['N', 'F', 'M'],
    'rate_coeff': arrhenius_Tg(1.31e-16, 0, 52740),
    'energy_loss': 0,
    'description': 'NF + M -> N + F + M'
})

# --- Three-body recombination (R151-R155) ---
# Rate coefficients in m^6/s
THREE_BODY_DATA = [
    (151, ['F', 'F', 'M'], ['F2', 'M'], 2.8e-46),
    (152, ['N', 'N', 'M'], ['N2', 'M'], 1.41e-44),
    (153, ['NF2', 'F', 'M'], ['NF3', 'M'], 1.03e-42),
    (154, ['NF', 'F', 'M'], ['NF2', 'M'], 1.03e-42),
    (155, ['N', 'F', 'M'], ['NF', 'M'], 2.8e-46),
]

for rid, reactants, products, k_val in THREE_BODY_DATA:
    REACTIONS.append({
        'id': rid, 'type': 'three_body',
        'reactants': reactants, 'products': products,
        'rate_coeff': constant_rate(k_val),
        'energy_loss': 0,
        'description': f'{" + ".join(reactants)} -> {" + ".join(products)}'
    })

# --- Surface reactions ---
# Wall recombination coefficients
SURFACE_RECOMBINATION = {
    'F': 0.01,      # F + F -> F2
    'N': 0.07,      # N + N -> N2
    'Ar_1s5': 1.0,  # Ar* -> Ar (de-excitation)
    'Ar_4p': 1.0,   # Ar(4p) -> Ar (de-excitation assumed same)
}

# Positive ions: lost at walls via Bohm velocity
# Negative ions: lost via ion-ion neutralization at sheath edge


# =============================================================================
# Utility: print reaction summary
# =============================================================================
def print_reaction_summary():
    """Print a summary of all reactions."""
    print(f"Total reactions: {len(REACTIONS)}")
    types = {}
    for r in REACTIONS:
        t = r['type']
        types[t] = types.get(t, 0) + 1
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")
    print(f"\nSpecies: {len(ALL_SPECIES)}")
    print(f"  Neutrals: {NEUTRAL_SPECIES}")
    print(f"  Positive ions: {POSITIVE_IONS}")
    print(f"  Negative ions: {NEGATIVE_IONS}")


if __name__ == '__main__':
    print_reaction_summary()
    
    # Test a few rate coefficients at Te=3 eV, Tg=600 K
    Te = 3.0
    Tg = 600.0
    Tng = Tg / 300.0
    print(f"\nRate coefficients at Te={Te} eV, Tg={Tg} K:")
    for r in REACTIONS[:10]:
        if callable(r['rate_coeff']):
            k = r['rate_coeff'](Te, Tg, Tng)
        else:
            k = r['rate_coeff']
        print(f"  R{r['id']:3d}: k = {k:.3e} m^3/s  [{r['description']}]")
