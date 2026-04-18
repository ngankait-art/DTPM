#!/usr/bin/env python3
"""
tier3_picmcc/run_case_B.py
===========================

Tier 3 validation Case B: 700 W / 5 mTorr / pure SF6 (workplan §5.3).

Low-pressure variant. At 5 mTorr the reduced field is approximately
twice the Case A value (because E is unchanged while N halves), so we
map this to E/N = 100 Td for the MCC run. The Phase 2 workplan
explicitly flags this case as the one where non-local effects are
expected to be larger, though the 0D MCC cross-check here cannot
resolve non-locality on its own.
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


CASE = dict(
    name="B",
    description="700 W / 5 mTorr / pure SF6 (low-pressure variant)",
    power_W=700.0,
    pressure_mTorr=5.0,
    x_Ar=0.0,
    EN_Td=100.0,
)

RUN = dict(n_electrons=3000, n_steps=30000, dt_s=2e-11, seed=43)

RESULTS = REPO / "tier3_picmcc" / "results"
OUT = REPO / "tier3_picmcc" / "outputs"
H5 = REPO / "data" / "raw" / "bolsig_data.h5"


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== Tier 3 Case {CASE['name']}: {CASE['description']} ===")
    print(f"E/N mapping: {CASE['EN_Td']} Td")
    result = run_mcc(EN_Td=CASE["EN_Td"], pressure_mTorr=CASE["pressure_mTorr"],
                     x_Ar=CASE["x_Ar"], **RUN)

    with h5py.File(H5, "r") as f:
        EN = f["grid/EN_Td"][()]
        x_Ar_grid = f["grid/x_Ar"][()]
        ij = (int(np.argmin(np.abs(EN - CASE["EN_Td"]))),
              int(np.argmin(np.abs(x_Ar_grid - CASE["x_Ar"]))))
        Te_bolsig = float(f["transport/Te_eff_eV"][ij])
        k_iz_bolsig = float(f["rates_boltzmann/SF6_total_ionization"][ij])
        k_att_bolsig = float(f["rates_boltzmann/SF6_total_attachment"][ij])
        k_el_bolsig = float(f["rates_boltzmann/SF6_elastic_momentum"][ij])

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
    (RESULTS / "case_B_result.json").write_text(json.dumps(record, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    ax.plot(result.time_s * 1e9, result.mean_energy_eV, lw=1.5, color="C2")
    ax.axhline(1.5 * Te_bolsig, color="C1", ls="--",
               label=f"BOLSIG+ 1.5 Te = {1.5*Te_bolsig:.2f} eV")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("<epsilon> (eV)")
    ax.set_title(f"Case {CASE['name']}: mean energy")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(result.eedf_grid_eV, result.eedf + 1e-10, lw=1.5, color="C2")
    ax.set_xlabel("epsilon (eV)")
    ax.set_ylabel("f(epsilon) (1/eV)")
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-5, 2)
    ax.set_title(f"Case {CASE['name']}: final EEDF")
    ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Tier 3 Case {CASE['name']}: {CASE['description']}")
    fig.tight_layout()
    fig.savefig(OUT / "case_B_mcc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print("--- summary ---")
    for k, v in [("Te_eff", (result.Te_eff(), Te_bolsig)),
                 ("k_el",   (result.rates['elastic'], k_el_bolsig)),
                 ("k_att",  (result.rates['attachment'], k_att_bolsig)),
                 ("k_iz",   (result.rates['ionization'], k_iz_bolsig))]:
        print(f"{k:8s}  MCC={v[0]:.3e}  BOLSIG+={v[1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
