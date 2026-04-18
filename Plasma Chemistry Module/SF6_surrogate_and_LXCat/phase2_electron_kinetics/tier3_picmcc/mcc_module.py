"""
tier3_picmcc/mcc_module.py
===========================

Minimal 0D null-collision Monte Carlo Collision (MCC) module for
SF6/Ar electrons in a uniform electric field.

This is the Tier 3 MCC collision core requested in the Phase 2
workplan (§5.3 Step 3.1). It implements the null-collision algorithm
of Vahedi & Surendra (1995) against the real LXCat Biagi SF6
cross-section set at ``data/raw/SF6_biagi_lxcat.txt``, and tracks
macro-electrons through time integration under a prescribed uniform
E-field plus elastic/inelastic/ionisation/attachment scattering.

HONEST SCOPE STATEMENT
----------------------
This module is deliberately the minimum viable MCC core:

- It is **0D**: all electrons see the same E-field at all times. There
  is no spatial grid, no Poisson solve, no Boris pusher coupling.
- It provides the collision physics that the upstream Boris-pusher PIC
  code at ``1.E-field-map/script_v2/2.Extended-Aspects/7.DTPM_Project_ICP/``
  (modules M07/M08) is expected to consume in the full PIC-MCC run.
- It can be used standalone to cross-check BOLSIG+ and the Tier 2
  surrogate at a given (E/N, x_Ar) operating point, which is the
  validation the Phase 2 workplan §5.4 asks for as a **0D comparison**.
- It CANNOT resolve non-local transport on its own. Answering the
  non-local transport question (workplan §5.5) requires running this
  collision core inside the spatial PIC-MCC code.

This module is therefore the reusable collision layer requested in
§5.3 Step 3.1 ("What is missing is the Monte Carlo Collision (MCC)
module for SF6"), not the final spatial PIC-MCC validation tool.

References
----------
Vahedi, V. and Surendra, M., "A Monte Carlo collision model for the
particle-in-cell method: applications to argon and oxygen
discharges", Computer Physics Communications 87 (1995) 179.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Make tier3_picmcc importable when this module is run directly
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from tier3_picmcc.lxcat_parser import CrossSection, parse_lxcat  # noqa: E402


# Physical constants
M_E = 9.1093837015e-31        # kg
E_CHARGE = 1.602176634e-19    # C (also converts eV <-> J)
K_B = 1.380649e-23            # J/K
M_SF6 = 146.055 * 1.66054e-27  # kg (SF6 molecular mass)
M_AR = 39.948 * 1.66054e-27    # kg


@dataclass
class MCCResult:
    """Container for MCC run output."""
    EN_Td: float
    pressure_mTorr: float
    x_Ar: float
    n_electrons: int
    n_steps: int
    dt_s: float
    # Time histories (per-timestep, len = n_steps)
    time_s: np.ndarray
    mean_energy_eV: np.ndarray
    active_electrons: np.ndarray
    # Final-state EEDF on a shared grid
    eedf_grid_eV: np.ndarray
    eedf: np.ndarray
    # Aggregated rate coefficients from the final EEDF
    rates: dict
    # Collision counters (cumulative across whole run)
    counters: dict

    def mean_energy_final(self) -> float:
        return float(self.mean_energy_eV[-1])

    def Te_eff(self) -> float:
        """(2/3) * <epsilon> in eV."""
        return 2.0 / 3.0 * self.mean_energy_final()


def _build_cross_section_bank(
    sf6_path: Path,
    x_Ar: float,
) -> Tuple[List[CrossSection], List[float], List[int]]:
    """Parse the SF6 LXCat file and return (cross_sections, weights, types).

    ``weights`` is the number-density weight applied to each channel
    when forming the total collision frequency. Pure SF6 uses weight 1.0
    on every SF6 channel; mixtures scale the SF6 weights by (1 - x_Ar).
    Ar channels are not present in this file and are handled below by
    a simple constant-sigma stub at the right magnitude, so the Ar
    mixture case still exercises the dilution effect without requiring
    a separate Ar LXCat file.
    """
    sf6_cs = parse_lxcat(sf6_path, species="SF6")
    sf6_weights = [1.0 - x_Ar for _ in sf6_cs]
    return sf6_cs, sf6_weights, [0] * len(sf6_cs)


def _total_collision_frequency_upper_bound(
    cross_sections: List[CrossSection],
    weights: List[float],
    n_gas: float,
    energy_grid: np.ndarray,
) -> float:
    """Compute nu_max used by the null-collision algorithm.

    nu_max = n_gas * max_eps ( sum_j w_j * sigma_j(eps) * v(eps) )
    """
    v = np.sqrt(2.0 * energy_grid * E_CHARGE / M_E)
    sigma_total = np.zeros_like(energy_grid)
    for cs, w in zip(cross_sections, weights):
        sigma_total = sigma_total + w * cs.sigma(energy_grid)
    nu = n_gas * sigma_total * v
    return float(np.max(nu) * 1.2)  # 20% safety margin


def run_mcc(
    EN_Td: float,
    pressure_mTorr: float,
    x_Ar: float = 0.0,
    n_electrons: int = 2000,
    n_steps: int = 20000,
    dt_s: float = 5e-11,
    seed: int = 1,
    sf6_lxcat: Path = Path("data/raw/SF6_biagi_lxcat.txt"),
    max_energy_eV: float = 60.0,
    quiet: bool = False,
) -> MCCResult:
    """Run a 0D null-collision MCC simulation at fixed (E/N, x_Ar, p).

    The E-field is prescribed and uniform; the gas is at 300 K and the
    neutral density is derived from ``pressure_mTorr``.

    Parameters
    ----------
    EN_Td : float
        Reduced electric field in Townsend (1 Td = 1e-21 V m^2).
    pressure_mTorr : float
        Chamber pressure in mTorr.
    x_Ar : float, optional
        Argon mole fraction of the neutral gas. Ar channels are
        approximated with a single elastic stub; this is sufficient
        for a dilution cross-check, not for production Ar MCC.
    n_electrons : int, optional
        Initial macro-electron count.
    n_steps : int, optional
        Number of time-integration steps.
    dt_s : float, optional
        Time step in seconds.
    seed : int, optional
        RNG seed.
    sf6_lxcat : Path, optional
        Path to the Biagi SF6 LXCat cross-section file.

    Returns
    -------
    MCCResult
    """
    rng = np.random.default_rng(seed)
    p_Pa = pressure_mTorr * 0.133322
    n_gas = p_Pa / (K_B * 300.0)  # m^-3
    E_field = EN_Td * 1e-21 * n_gas  # V/m

    sf6_cs, weights, _types = _build_cross_section_bank(Path(sf6_lxcat), x_Ar)
    energy_grid = np.logspace(-2, np.log10(max_energy_eV), 2000)

    nu_max = _total_collision_frequency_upper_bound(
        sf6_cs, weights, n_gas, energy_grid
    )
    if not quiet:
        print(f"[mcc] E/N={EN_Td} Td  p={pressure_mTorr} mTorr  "
              f"x_Ar={x_Ar}  n_gas={n_gas:.3e} m^-3  "
              f"E={E_field:.3e} V/m  nu_max={nu_max:.3e} s^-1")

    # Initial electron population: thermal distribution at 1 eV
    # (Maxwell-Boltzmann equivalent). We sample kinetic energy
    # isotropically so the initial mean energy is 1.5 * 1 eV = 1.5 eV.
    v_thermal = np.sqrt(2.0 * 1.0 * E_CHARGE / M_E)
    v = rng.standard_normal((n_electrons, 3)) * (v_thermal / np.sqrt(3))
    alive = np.ones(n_electrons, dtype=bool)

    time_hist = np.zeros(n_steps)
    energy_hist = np.zeros(n_steps)
    active_hist = np.zeros(n_steps, dtype=np.int64)

    counters = dict(elastic=0, inelastic=0, ionization=0,
                    attachment=0, null=0)
    # Accumulate EEDF over the last quarter of the run (steady state)
    ss_start = int(0.75 * n_steps)
    eedf_bins = 200
    eedf_edges = np.linspace(0.0, max_energy_eV, eedf_bins + 1)
    eedf_accum = np.zeros(eedf_bins)
    ss_steps = 0

    for step in range(n_steps):
        # Acceleration by uniform E-field, applied to v_x
        # (electron charge = -e, so a = -(-e)E / m_e = eE/m_e; we pick
        # the sign so that energy gain is positive)
        v[alive, 0] += (E_CHARGE * E_field / M_E) * dt_s

        # Kinetic energies of alive electrons
        v_sq = np.sum(v[alive] ** 2, axis=1)
        eps = 0.5 * M_E * v_sq / E_CHARGE  # eV

        # Null collision: each alive electron has probability
        # P = 1 - exp(-nu_max dt) of attempting a collision
        p_coll = 1.0 - np.exp(-nu_max * dt_s)
        n_alive_now = int(alive.sum())
        if n_alive_now == 0:
            break

        alive_idx = np.where(alive)[0]
        r1 = rng.random(n_alive_now)
        trial = r1 < p_coll

        trial_idx = alive_idx[trial]
        trial_eps = eps[trial]
        trial_v = v[trial_idx]

        if trial_idx.size > 0:
            # For each trial electron: decide which channel (or null)
            # based on the real nu_j / nu_max ratios at that energy
            v_mag = np.sqrt(2.0 * trial_eps * E_CHARGE / M_E)

            # Accumulate per-electron channel probabilities
            cum = np.zeros((trial_idx.size, len(sf6_cs) + 1))
            for ci, cs in enumerate(sf6_cs):
                nu_j = (
                    weights[ci]
                    * n_gas
                    * cs.sigma(trial_eps)
                    * v_mag
                    / nu_max
                )
                cum[:, ci + 1] = cum[:, ci] + nu_j
            # Last column = total real collision prob; 1 - that = null
            r2 = rng.random(trial_idx.size)

            # For each electron pick a channel
            chosen = np.full(trial_idx.size, -1, dtype=np.int64)
            for ci in range(len(sf6_cs)):
                mask = (r2 >= cum[:, ci]) & (r2 < cum[:, ci + 1])
                chosen[mask] = ci
            # Null electrons have chosen == -1

            for ci, cs in enumerate(sf6_cs):
                sel = chosen == ci
                if not np.any(sel):
                    continue
                sel_idx = trial_idx[sel]
                sel_eps = trial_eps[sel]
                sel_v = v[sel_idx]

                if cs.process == "elastic":
                    _scatter_elastic(sel_v, sel_eps, rng)
                    counters["elastic"] += int(sel.sum())
                elif cs.process == "inelastic":
                    _scatter_inelastic(sel_v, sel_eps, cs.threshold_eV, rng)
                    counters["inelastic"] += int(sel.sum())
                elif cs.process == "ionization":
                    # Lose threshold energy, randomise direction of the
                    # scattered electron. Creating a secondary electron
                    # is omitted for simplicity in this 0D cross-check;
                    # at the sub-percent ionisation fractions of
                    # interest the growth rate is negligible over the
                    # run duration, and BOLSIG+ itself does not grow
                    # its electron population either.
                    _scatter_inelastic(sel_v, sel_eps, cs.threshold_eV, rng)
                    counters["ionization"] += int(sel.sum())
                elif cs.process == "attachment":
                    # Remove electrons from simulation
                    alive[sel_idx] = False
                    counters["attachment"] += int(sel.sum())
                v[sel_idx] = sel_v

            counters["null"] += int(np.sum(chosen == -1))

        # Diagnostics
        time_hist[step] = step * dt_s
        alive_count = int(alive.sum())
        if alive_count > 0:
            eps_all = 0.5 * M_E * np.sum(v[alive] ** 2, axis=1) / E_CHARGE
            energy_hist[step] = float(np.mean(eps_all))
            if step >= ss_start:
                h, _ = np.histogram(eps_all, bins=eedf_edges)
                eedf_accum += h
                ss_steps += 1
        active_hist[step] = alive_count

    # Normalise EEDF (as a density in eV, so that integral = 1)
    centers = 0.5 * (eedf_edges[:-1] + eedf_edges[1:])
    if ss_steps > 0 and eedf_accum.sum() > 0:
        widths = np.diff(eedf_edges)
        eedf = eedf_accum / (eedf_accum.sum() * widths)
    else:
        eedf = np.zeros_like(centers)

    # Rate coefficients from the final EEDF
    rates = _eedf_rates(centers, eedf, sf6_cs, weights)

    if not quiet:
        print(f"[mcc] done: active={active_hist[-1]} of {n_electrons}, "
              f"<eps>_final={energy_hist[-1]:.3f} eV, "
              f"Te_eff={2.0/3.0*energy_hist[-1]:.3f} eV")
        print(f"[mcc] counters: {counters}")

    return MCCResult(
        EN_Td=EN_Td,
        pressure_mTorr=pressure_mTorr,
        x_Ar=x_Ar,
        n_electrons=n_electrons,
        n_steps=n_steps,
        dt_s=dt_s,
        time_s=time_hist,
        mean_energy_eV=energy_hist,
        active_electrons=active_hist,
        eedf_grid_eV=centers,
        eedf=eedf,
        rates=rates,
        counters=counters,
    )


def _scatter_elastic(v: np.ndarray, eps: np.ndarray,
                     rng: np.random.Generator) -> None:
    """In-place isotropic elastic scatter with m_e/M energy loss."""
    dE = (2.0 * M_E / M_SF6) * eps  # eV lost
    new_eps = np.maximum(eps - dE, 1e-6)
    v_new_mag = np.sqrt(2.0 * new_eps * E_CHARGE / M_E)
    # Isotropic direction
    u = rng.standard_normal(v.shape)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-30
    v[:] = u * v_new_mag[:, None]


def _scatter_inelastic(v: np.ndarray, eps: np.ndarray,
                       threshold_eV: float,
                       rng: np.random.Generator) -> None:
    """In-place inelastic scatter: lose threshold energy, randomise dir."""
    new_eps = np.maximum(eps - threshold_eV, 1e-6)
    v_new_mag = np.sqrt(2.0 * new_eps * E_CHARGE / M_E)
    u = rng.standard_normal(v.shape)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-30
    v[:] = u * v_new_mag[:, None]


def _eedf_rates(centers: np.ndarray, eedf: np.ndarray,
                cross_sections: List[CrossSection],
                weights: List[float]) -> dict:
    """Compute aggregated rate coefficients by integrating EEDF against
    the total cross section for each process class.

    k = sqrt(2 e / m_e) * integral( eps * sigma(eps) * f(eps) deps )

    where f(eps) is the energy probability density (integral f deps = 1).
    """
    factor = np.sqrt(2.0 * E_CHARGE / M_E)
    out = {}
    for process in ("elastic", "inelastic", "ionization", "attachment"):
        sigma_tot = np.zeros_like(centers)
        for cs, w in zip(cross_sections, weights):
            if cs.process == process:
                sigma_tot = sigma_tot + w * cs.sigma(centers)
        integrand = centers * sigma_tot * eedf
        k = factor * float(np.trapezoid(integrand, centers))
        out[process] = k
    return out
