# Mettler TEL Experimental Data — Hand-Digitised Reference

**Source**: J.D. Mettler, UIUC PhD dissertation (2025), Chapter 4.

These values are **visually digitised from the published figures** (not from raw data tables). They are accurate to ~±5% and sufficient for benchmarking. If higher precision is needed, ask Mettler for the underlying data file.

---

## Fig 4.14 — Normalised radial [F] profile (p. 70)

**Conditions**: P_ICP = 1000 W, 70 sccm SF6 / 30 sccm Ar (90% SF6 referred to SF6 alone after Ar subtraction), p_tot = 10 mTorr, 200 W rf wafer bias.

Cubic fit (from the paper's inset): `y(r) = 1.01032 − 0.01847·r² + 7.139×10⁻⁴·r³`, r in cm.

| r (cm) | y_normalised (data) | y_normalised (cubic fit) |
|---|---|---|
| 0 | 1.00 | 1.010 |
| 2 | 0.95 | 0.943 |
| 4 | 0.81 | 0.766 |
| 6 | 0.53 | 0.498 |
| 8 | 0.25 | 0.159 |

**Centre-to-edge (0 → 8 cm) drop**: 75%.

---

## Fig 4.17 — Absolute radial [F] and Si etch rate (p. 75)

Conditions: P_ICP = 1000 W, 100 sccm total flow, p_tot = 10 mTorr.

### Top panel — Wafer Bias OFF

**90% SF6 (90 sccm SF6 / 10 sccm Ar)** — red circles:

| r (cm) | n_F (×10²⁰ /m³) | n_F (×10¹⁴ cm⁻³) |
|---|---|---|
| 0 | 2.5 | 2.5 |
| 2 | 2.3 | 2.3 |
| 4 | 1.6 | 1.6 |
| 6 | 1.0 | 1.0 |
| 8 | 0.6 | 0.6 |

*Centre-to-edge drop: 76%.*

**30% SF6 (30 sccm SF6 / 70 sccm Ar)** — red triangles:

| r (cm) | n_F (×10²⁰ /m³) | n_F (×10¹⁴ cm⁻³) |
|---|---|---|
| 0 | 0.60 | 0.60 |
| 2 | 0.55 | 0.55 |
| 4 | 0.42 | 0.42 |
| 6 | 0.28 | 0.28 |
| 8 | 0.20 | 0.20 |

*Centre-to-edge drop: 67%.*

**Si etch rate (90% SF6, bias off)** — black squares, nm/s:

| r (cm) | Si etch rate (nm/s) |
|---|---|
| 0 | 25 |
| 2 | 23 |
| 4 | 16 |
| 6 | 9 |
| 8 | 5 |

### Bottom panel — Wafer Bias ON (200 W rf)

**90% SF6** — red circles:

| r (cm) | n_F (×10²⁰ /m³) |
|---|---|
| 0 | 4.0 |
| 2 | 3.6 |
| 4 | 2.6 |
| 6 | 1.7 |
| 8 | 1.0 |

*Bias enhancement factor (vs bias-off, 90% SF6): ~1.6 uniformly.*

**30% SF6** — red triangles:

| r (cm) | n_F (×10²⁰ /m³) |
|---|---|
| 0 | 1.3 |
| 2 | 1.2 |
| 4 | 0.75 |
| 6 | 0.45 |
| 8 | 0.32 |

*Bias enhancement factor (vs bias-off, 30% SF6): ~2.15 at centre, ~1.6 at edge.*

---

## Fig 4.9 — ICP-region [F] vs SF6 flow (p. 61)

**Conditions**: total flow 100 sccm (SF6 + balance Ar), two operating points.

### Branch 1: 20 mTorr, 600 W

| SF6 flow (sccm) | Etch rate (×10²¹ /m²·s) | n_F (×10²⁰ /m³) |
|---|---|---|
| 10 | 1.0 | 1.0 |
| 30 | 3.3 | 3.2 |
| 50 | 5.0 | 5.3 |
| 70 | 5.5 | 7.5 |
| 90 | 6.4 | 9.6 |

### Branch 2: 40 mTorr, 700 W

| SF6 flow (sccm) | Etch rate (×10²¹ /m²·s) | n_F (×10²¹ /m³) |
|---|---|---|
| 10 | 1.0 | 0.25 |
| 30 | 4.7 | 0.77 |
| 50 | 7.2 | 1.15 |
| 70 | 7.6 | 1.62 |
| 90 | 8.6 | 2.0 |

---

## Fig 4.18 — Si etch probability vs F flux (p. 77)

Not tabulated here (scatter-plot, values read approximately):

- **Radical-probe-derived** ε_Si: 0.025–0.04, roughly independent of F flux (1×10¹⁷ – 5×10¹⁸ /cm²·s).
- **Actinometry-derived** ε_Si: 0.055–0.10, with apparent inverse flux dependence — Mettler attributes this scatter to the spatial-averaging bias of actinometry.

---

## Derived Quantities

### Actinometry vs local-centre ratio (Mettler Eq 4.2, Fig 4.14)

From Mettler p. 70: `n_0 = 2.5×10²⁰ /m³` (local centre) vs `n_act = 1.54×10²⁰ /m³` (actinometry volume-average).

```
n_centre / n_actinometry = 2.5 / 1.54 ≈ 1.62
```

Equivalent cross-chamber integral formula:
```
n_act = (n_0 / R) × ∫₀^R f(r) dr
```
where `f(r)` is the normalised radial profile and `R` is the chamber radius.

### Kinetic ↔ diffusion-limited threshold (Table 4.4)

| Pressure (mTorr) | D_F (m²/s) | δ_diff (cm) | r_etch,diff (×10²¹ /m²·s) |
|---|---|---|---|
| 20 | 1.63 | 4.0 | 6.4 |
| 40 | 1.03 | 2.3 | 8.5 |

Our Phase-1 simulations yield centre [F] of a few × 10²⁰ /m³ = a few × 10¹⁴ cm⁻³, well below the ~1.1×10²¹ /m³ saturation density. Kinetic-regime comparison is valid.

---

## How to Use These Tables in `generate_stage10_figures.py`

```python
# Mettler Fig 4.14 cubic fit
import numpy as np
def mettler_fig414_fit(r_cm):
    return 1.01032 - 0.01847*r_cm**2 + 7.139e-4*r_cm**3

# Mettler Fig 4.17 bias-off 90% SF6
mettler_fig417_90pct_r_cm = np.array([0, 2, 4, 6, 8])
mettler_fig417_90pct_nF_per_m3 = np.array([2.5, 2.3, 1.6, 1.0, 0.6]) * 1e20
# convert to cm^-3 for plotting: * 1e-6

# Mettler Fig 4.9 at 700W / 40 mTorr
mettler_fig49_sccm = np.array([10, 30, 50, 70, 90])
mettler_fig49_nF_per_m3 = np.array([0.25, 0.77, 1.15, 1.62, 2.0]) * 1e21
```

These blocks can be pasted directly into the overlay functions described in
`~/.claude/plans/mutable-swimming-chipmunk.md` (Stage B of the Mettler-accuracy
pass).
