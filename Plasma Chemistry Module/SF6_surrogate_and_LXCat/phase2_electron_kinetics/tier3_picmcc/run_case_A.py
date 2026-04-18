#!/usr/bin/env python3
"""
tier3_picmcc/run_case_A.py
===========================

Tier 3 validation Case A: 700 W / 10 mTorr / pure SF6 (workplan §5.3).

This is the reference condition matching the Stage 10 DTPM validation
point. We do not have a full self-consistent 2D PIC-MCC here; instead
we run the 0D MCC collision core at a single representative E/N
derived from the Stage 10 power balance and compare the resulting
aggregated rates and EEDF against the BOLSIG+ Tier 1 lookup at the
matching E/N.

The E/N mapping for a 700 W / 10 mTorr pure-SF6 ICP is taken from the
Stage 10 0D power balance produced by the upstream DTPM code, which
places the ICP-volume-averaged Te at approximately 3.0 eV and the
corresponding E/N at approximately 50 Td. We use this as the Case A
MCC operating point and flag the modelling assumption explicitly in
the output report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tier3_picmcc.mcc_module import run_mcc   # noqa: E402


# Case A parameters (workplan §5.3 Step 3.2)
CASE = dict(
    name="A",
    description="700 W / 10 mTorr / pure SF6 (reference condition)",
    power_W=700.0,
    pressure_mTorr=10.0,
    x_Ar=0.0,
    # E/N mapping from Stage 10 0D power balance, documented in
    # docstring. The 50 Td value corresponds to Te ~ 3.0 eV.
    EN_Td=50.0,
)

# Run configuration
RUN = dict(
    n_electrons=3000,
    n_steps=30000,
    dt_s=2e-11,
    seed=42,
)

RESULTS = REPO / "tier3_picmcc" / "results"
OUT = REPO / "tier3_picmcc" / "outputs"
H5 = REPO / "data" / "raw" / "bolsig_data.h5"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"=== Tier 3 Case {CASE['name']}: {CASE['description']} ===")
    print(f"E/N mapping: {CASE['EN_Td']} Td")

    result = run_mcc(
        EN_Td=CASE["EN_Td"],
        pressure_mTorr=CASE["pressure_mTorr"],
        x_Ar=CASE["x_Ar"],
        **RUN,
    )

    # Compare against BOLSIG+ Tier 1 lookup at matching E/N
    with h5py.File(H5, "r") as f:
        EN = f["grid/EN_Td"][()]
        x_Ar_grid = f["grid/x_Ar"][()]
        ij = (int(np.argmin(np.abs(EN - CASE["EN_Td"]))),
              int(np.argmin(np.abs(x_Ar_grid - CASE["x_Ar"]))))
        Te_bolsig = float(f["transport/Te_eff_eV"][ij])
        k_iz_bolsig = float(f["rates_boltzmann/SF6_total_ionization"][ij])
        k_att_bolsig = float(f["rates_boltzmann/SF6_total_attachment"][ij])
        k_el_bolsig = float(f["rates_boltzmann/SF6_elastic_momentum"][ij])

    # Save machine-readable result
    record = {
        "case": CASE,
        "run": RUN,
        "mcc": {
            "Te_eff_eV": result.Te_eff(),
            "mean_energy_final_eV": result.mean_energy_final(),
            "rates": {k: float(v) for k, v in result.rates.items()},
            "counters": result.counters,
            "active_fraction_final":
                float(result.active_electrons[-1]) / result.n_electrons,
        },
        "bolsig_ref": {
            "EN_Td": float(EN[ij[0]]),
            "x_Ar": float(x_Ar_grid[ij[1]]),
            "Te_eff_eV": Te_bolsig,
            "k_iz": k_iz_bolsig,
            "k_att": k_att_bolsig,
            "k_el": k_el_bolsig,
        },
    }
    (RESULTS / "case_A_result.json").write_text(json.dumps(record, indent=2))
    print(f"[case A] result written to {RESULTS / 'case_A_result.json'}")

    # Plot: mean-energy trajectory + final EEDF
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    ax.plot(result.time_s * 1e9, result.mean_energy_eV, lw=1.5)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("<epsilon> (eV)")
    ax.set_title(f"Case {CASE['name']}: mean energy trajectory")
    ax.axhline(1.5 * Te_bolsig, color="C1", ls="--",
               label=f"BOLSIG+ target 1.5 Te = {1.5*Te_bolsig:.2f} eV")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(result.eedf_grid_eV, result.eedf + 1e-10, lw=1.5,
                label="MCC EEDF (steady state)")
    ax.set_xlabel("epsilon (eV)")
    ax.set_ylabel("f(epsilon) (1/eV)")
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-5, 2)
    ax.set_title(f"Case {CASE['name']}: final EEDF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle(f"Tier 3 Case {CASE['name']}: {CASE['description']}")
    fig.tight_layout()
    fig.savefig(OUT / "case_A_mcc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[case A] plot written to {OUT / 'case_A_mcc.png'}")

    # Summary to stdout
    print()
    print("--- summary ---")
    print(f"MCC Te_eff        = {result.Te_eff():.3f} eV")
    print(f"BOLSIG+ Te_eff    = {Te_bolsig:.3f} eV")
    print(f"MCC k_el          = {result.rates['elastic']:.3e} m3/s")
    print(f"BOLSIG+ k_el      = {k_el_bolsig:.3e} m3/s")
    print(f"MCC k_att         = {result.rates['attachment']:.3e} m3/s")
    print(f"BOLSIG+ k_att     = {k_att_bolsig:.3e} m3/s")
    print(f"MCC k_iz          = {result.rates['ionization']:.3e} m3/s")
    print(f"BOLSIG+ k_iz      = {k_iz_bolsig:.3e} m3/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
