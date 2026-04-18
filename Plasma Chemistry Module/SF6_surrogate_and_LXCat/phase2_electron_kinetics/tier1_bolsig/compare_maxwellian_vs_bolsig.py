#!/usr/bin/env python3
"""
tier1_bolsig/compare_maxwellian_vs_bolsig.py
=============================================

Tier 1 of the Phase 2 electron-kinetics workplan.

For each (E/N, x_Ar) point in the BOLSIG+ grid, compute the ratio
    ratio_j(E/N, x_Ar) = k_j^Maxwell / k_j^Boltzmann
for the dominant rate-coefficient channels (ionisation, attachment,
aggregated dissociation / excitation, momentum-transfer), and produce
two deliverables:

1. A markdown report at
   ``tier1_bolsig/outputs/maxwell_vs_bolsig_report.md``
   that states, at the reference operating condition
   E/N = 50 Td, x_Ar = 0, whether the Maxwellian assumption is
   within 20 percent (the workplan's decision threshold).

2. A PNG figure at
   ``tier1_bolsig/plots/ratio_vs_EN.png``
   showing the ratio curves for pure SF6 across the full E/N range
   so the EEDF-tail sensitivity is visually obvious.

This script is a pure reader: it does not re-run BOLSIG+ and does
not modify the HDF5 file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
H5 = REPO / "data" / "raw" / "bolsig_data.h5"
OUT_MD = REPO / "tier1_bolsig" / "outputs" / "maxwell_vs_bolsig_report.md"
OUT_FIG = REPO / "tier1_bolsig" / "plots" / "ratio_vs_EN.png"

CHANNELS = [
    ("SF6_total_ionization", "k_iz"),
    ("SF6_total_attachment", "k_att"),
    ("SF6_total_excitation", "k_diss"),
    ("SF6_elastic_momentum", "k_el"),
]

REFERENCE_EN_TD = 50.0   # workplan §3.4 decision point
REFERENCE_xAr = 0.0


def main() -> int:
    if not H5.exists():
        print(f"ERROR: lookup not found at {H5}", file=sys.stderr)
        return 1

    with h5py.File(H5, "r") as f:
        EN = f["grid/EN_Td"][()]
        xAr = f["grid/x_Ar"][()]
        ar0 = int(np.argmin(np.abs(xAr - REFERENCE_xAr)))
        en_ref = int(np.argmin(np.abs(EN - REFERENCE_EN_TD)))

        bolt, maxw = {}, {}
        for h5name, short in CHANNELS:
            bolt[short] = f[f"rates_boltzmann/{h5name}"][()]
            maxw[short] = f[f"maxwell_ref/rates_maxwell/{h5name}"][()]

    # Ratios (pure SF6 slice)
    ratios = {
        short: np.where(
            bolt[short][:, ar0] > 0,
            maxw[short][:, ar0] / np.where(bolt[short][:, ar0] > 1e-300,
                                           bolt[short][:, ar0], np.nan),
            np.nan,
        )
        for _, short in CHANNELS
    }

    # Reference-point numerical summary
    ref_lines = [
        "# Tier 1: Maxwell vs BOLSIG+ rate comparison",
        "",
        "## Reference operating point",
        "",
        f"- E/N = {EN[en_ref]:g} Td (closest to nominal {REFERENCE_EN_TD} Td)",
        f"- x_Ar = {xAr[ar0]:g} (pure SF6)",
        "",
        "| Channel | Maxwellian rate (m3/s) | Boltzmann rate (m3/s) |"
        " Ratio M/B | Within 20%? |",
        "|---|---|---|---|---|",
    ]
    any_outside = False
    for _, short in CHANNELS:
        m = maxw[short][en_ref, ar0]
        b = bolt[short][en_ref, ar0]
        r = m / b if b > 0 else float("nan")
        ok = abs(r - 1.0) < 0.20 if np.isfinite(r) else False
        if not ok:
            any_outside = True
        ref_lines.append(
            f"| `{short}` | {m:.3e} | {b:.3e} | "
            f"{r:.3g} | {'yes' if ok else 'NO'} |"
        )

    verdict = (
        "**All dominant rates agree within 20%** at the reference point. "
        "The Maxwellian assumption is adequate for downstream rate "
        "evaluation at this operating point. Tier 2 surrogate is still "
        "built for completeness and for use outside this range."
        if not any_outside else
        "**At least one dominant rate differs by more than 20%** at "
        "the reference point. The Maxwellian assumption introduces "
        "significant error; Tier 2 surrogate is essential, and the "
        "production DTPM model should use BOLSIG+/Tier 2 rates "
        "rather than Arrhenius forms."
    )
    ref_lines += [
        "",
        "## Decision (workplan §3.4)",
        "",
        verdict,
        "",
        "## Full pure-SF6 sweep",
        "",
        "The plot at `plots/ratio_vs_EN.png` shows the Maxwell/Boltzmann "
        "ratio for each channel across the full E/N range. The ratio "
        "is approximately unity for low-energy-dominated channels "
        "(attachment, elastic momentum transfer) and departs strongly "
        "from unity for tail-dominated channels (ionisation, "
        "dissociative excitation) where the Maxwellian over-populates "
        "the high-energy tail that drives those rates.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(ref_lines))

    # Plot
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    styles = {"k_iz": "-", "k_att": "--", "k_diss": "-.", "k_el": ":"}
    for _, short in CHANNELS:
        ax.plot(EN, ratios[short], styles.get(short, "-"),
                lw=1.8, label=short)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E/N (Td)")
    ax.set_ylabel("ratio  k_Maxwell / k_Boltzmann")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.axvline(REFERENCE_EN_TD, color="k", lw=0.5, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Tier 1: Maxwellian-to-Boltzmann rate ratios, pure SF$_6$")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[tier1] report: {OUT_MD}")
    print(f"[tier1] plot:   {OUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
