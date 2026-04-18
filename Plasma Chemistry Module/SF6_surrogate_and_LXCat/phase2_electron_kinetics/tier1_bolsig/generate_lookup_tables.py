#!/usr/bin/env python3
"""
tier1_bolsig/generate_lookup_tables.py
=======================================

Tier 1 of the Phase 2 electron-kinetics workplan.

Generates and exposes the BOLSIG+ lookup tables for the SF6/Ar grid used
by the downstream DTPM model. The raw two-term Boltzmann solution for the
full (E/N, x_Ar) grid has already been produced by an upstream BOLSIG+
batch run and is stored in ``data/raw/bolsig_data.h5``. This script
validates the HDF5 file, writes human-readable CSV projections of the
key aggregated rates, and emits a short summary of what the lookup
contains so that the downstream Tier 2 training code and Tier 3 MCC
validation code can consume it unambiguously.

Usage
-----

    python tier1_bolsig/generate_lookup_tables.py

Outputs
-------

- ``tier1_bolsig/outputs/lookup_summary.txt``
- ``tier1_bolsig/outputs/rates_boltzmann_pure_SF6.csv``
- ``tier1_bolsig/outputs/rates_maxwell_pure_SF6.csv``
- ``tier1_bolsig/outputs/transport_pure_SF6.csv``

References
----------
Hagelaar, G. J. M. and Pitchford, L. C., "Solving the Boltzmann equation
to obtain electron transport coefficients and rate coefficients for
fluid models", PSST 14 (2005) 722.
Biagi SF6 cross-section compilation, LXCat (www.lxcat.net).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parent.parent
H5 = REPO / "data" / "raw" / "bolsig_data.h5"
OUT = REPO / "tier1_bolsig" / "outputs"


# Canonical names for the aggregated rates that the DTPM reduced model
# needs. The downstream Tier 2 surrogate is trained on exactly these
# five scalar outputs.
AGG_RATES = [
    ("SF6_total_ionization", "k_iz"),
    ("SF6_total_attachment", "k_att"),
    ("SF6_total_excitation", "k_diss_like"),
    ("Ar_total_ionization",  "k_iz_Ar"),
    ("SF6_elastic_momentum", "k_el_SF6"),
]


def main() -> int:
    if not H5.exists():
        print(f"ERROR: BOLSIG+ lookup not found at {H5}", file=sys.stderr)
        print("Run the upstream BOLSIG+ batch or restore from backup.",
              file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5, "r") as f:
        EN = f["grid/EN_Td"][()]
        xAr = f["grid/x_Ar"][()]
        eps_grid = f["grid/epsilon_eV"][()]
        Te_eff = f["transport/Te_eff_eV"][()]
        mean_eps = f["transport/mean_energy_eV"][()]

        # Collect Boltzmann rates + Maxwellian reference rates
        rates_bolt = {
            out_name: f[f"rates_boltzmann/{h5_name}"][()]
            for h5_name, out_name in AGG_RATES
        }
        rates_max = {
            out_name: f[f"maxwell_ref/rates_maxwell/{h5_name}"][()]
            for h5_name, out_name in AGG_RATES
        }

        # Pure-SF6 slice (x_Ar = 0)
        ar0 = int(np.argmin(np.abs(xAr)))
        assert abs(xAr[ar0]) < 1e-9, f"x_Ar=0 not on grid (got {xAr[ar0]})"

        # Summary
        summary_lines = [
            "Tier 1 BOLSIG+ lookup — summary",
            "================================",
            f"Source HDF5         : {H5.relative_to(REPO)}",
            f"E/N grid (Td)       : {len(EN)} points, {EN.min():g} to {EN.max():g}",
            f"x_Ar grid           : {len(xAr)} points, {list(xAr)}",
            f"Energy grid (eV)    : {len(eps_grid)} points, "
            f"{eps_grid.min():g} to {eps_grid.max():g}",
            f"Aggregated rates    : {len(AGG_RATES)}",
            "",
            "Pure-SF6 reference row (x_Ar = 0):",
            f"  E/N (Td)        {' '.join(f'{v:10g}' for v in EN[::4])}",
            f"  Te_eff (eV)     {' '.join(f'{v:10.3f}' for v in Te_eff[::4, ar0])}",
            f"  k_iz Bolt       {' '.join(f'{v:10.2e}' for v in rates_bolt['k_iz'][::4, ar0])}",
            f"  k_iz Maxwell    {' '.join(f'{v:10.2e}' for v in rates_max['k_iz'][::4, ar0])}",
            f"  k_att Bolt      {' '.join(f'{v:10.2e}' for v in rates_bolt['k_att'][::4, ar0])}",
            f"  k_att Maxwell   {' '.join(f'{v:10.2e}' for v in rates_max['k_att'][::4, ar0])}",
            "",
            "Interpretation:",
            "- k_iz ratio (Maxwell/Bolt) is the EEDF-tail sensitivity metric.",
            "- k_att ratio is insensitive because attachment is below 1 eV.",
            "- Ratios driven well above 1 signify Maxwellian bias in",
            "  subsequent fluid-model rate tables.",
        ]
        (OUT / "lookup_summary.txt").write_text("\n".join(summary_lines))

        # CSV: pure-SF6 Boltzmann rates
        _write_csv(
            OUT / "rates_boltzmann_pure_SF6.csv",
            EN,
            {name: rates_bolt[name][:, ar0] for _, name in AGG_RATES},
        )
        # CSV: pure-SF6 Maxwell rates
        _write_csv(
            OUT / "rates_maxwell_pure_SF6.csv",
            EN,
            {name: rates_max[name][:, ar0] for _, name in AGG_RATES},
        )
        # CSV: transport (Te_eff, mean energy) for pure-SF6
        _write_csv(
            OUT / "transport_pure_SF6.csv",
            EN,
            {"Te_eff_eV": Te_eff[:, ar0], "mean_energy_eV": mean_eps[:, ar0]},
        )

    print("[tier1] lookup summary written to", OUT / "lookup_summary.txt")
    print("[tier1] CSV projections in", OUT)
    return 0


def _write_csv(path: Path, EN, cols: dict) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["EN_Td"] + list(cols.keys()))
        for i, en in enumerate(EN):
            w.writerow([f"{en:g}"] + [f"{v[i]:.6e}" for v in cols.values()])


if __name__ == "__main__":
    sys.exit(main())
