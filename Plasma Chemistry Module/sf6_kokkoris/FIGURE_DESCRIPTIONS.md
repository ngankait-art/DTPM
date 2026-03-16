# SF6 Global Plasma Model — Figure Descriptions and Explanations

This document describes each figure produced by the model, explains what physics is being shown, and highlights the key trends a reader should look for.

---

## Figure 1: `pressure_rise_and_F_density_vs_power.png`

**Conditions:** p_OFF = 0.921 Pa (fixed), power swept from 50 to 3500 W.

**Left panel — Pressure rise vs Power:**
Shows the difference between the pressure after plasma ignition and the pre-discharge pressure (Δp = p_plasma − p_OFF). This is the most direct experimental observable for validating the model because it requires no calibration — it is simply read from a pressure gauge.

The pressure rise increases monotonically with power. The physical reason is that higher power means more electrons with higher energy, which dissociate more SF6 molecules. Each SF6 that breaks into SF5 + F creates two molecules from one, increasing the total number of gas-phase particles and therefore the pressure. At 3500 W the pressure rise is about 0.52 Pa, meaning the total pressure has increased by roughly 56% over p_OFF.

The curve is sublinear (concave down) because at higher power, the parent gas SF6 becomes increasingly depleted — there is less SF6 left to dissociate, so each additional watt of power produces diminishing returns in pressure rise.

**Right panel — F atom density vs Power:**
Shows the density of fluorine atoms, which are the primary reactive species for silicon etching. F density increases nearly linearly with power, reaching about 1.1×10²⁰ m⁻³ at 3500 W. The near-linearity arises because F is produced by electron-impact dissociation (rate proportional to ne × n_SF6) and consumed mainly by surface recombination and gas-phase recombination with SF5 — both of which scale roughly linearly with F density itself, leading to F ∝ ne ∝ Power.

---

## Figure 2: `pressure_rise_and_F_density_vs_pOFF.png`

**Conditions:** Power = 2000 W (fixed), p_OFF swept from 0.3 to 4.5 Pa.

**Left panel — Pressure rise vs p_OFF:**
Shows how the degree of dissociation changes as the initial amount of SF6 in the reactor is varied. The pressure rise initially increases from 0.3 to about 2.5 Pa as more SF6 becomes available for dissociation. Above ~2.5 Pa, the pressure rise begins to decrease. This decrease is the signature of SFx deposition on the reactor walls — the fluoro-sulfur film growth becomes more significant at higher pressures because there are more SFx radicals available for deposition. The paper identifies this turnover as evidence that a loss mechanism for SFx radicals (wall deposition) is needed to explain the experimental data.

**Right panel — F atom density vs p_OFF:**
F density decreases monotonically from ~8×10¹⁹ m⁻³ at 0.3 Pa to ~6.4×10¹⁹ m⁻³ at 4.5 Pa. At low pressure, the discharge is more strongly dissociated (higher fractional conversion of SF6), so F density per unit of SF6 is higher. At higher pressure, the electron temperature decreases slightly and the recombination of F with SFx fragments becomes more effective, reducing the F density. The surface loss of F atoms also contributes to the decline.

---

## Figure 3: `species_densities_vs_power.png`

**Conditions:** p_OFF = 0.921 Pa (fixed), power swept from 50 to 3500 W.

**Left panel — Neutral species densities vs Power (log scale):**
Shows how the gas composition evolves as power increases. This is the central result of the model.

- **SF6** (blue) is the dominant species at all powers, starting at ~1.7×10²⁰ m⁻³ and decreasing to ~1.0×10²⁰ m⁻³ at 3500 W. The decrease reflects progressive dissociation of the parent gas.
- **F** (brown) rises steeply at low power and becomes the second most abundant species above ~300 W, reaching ~1.1×10²⁰ m⁻³ at 3500 W.
- **SF4** (red) rises from ~10¹⁸ to ~5×10¹⁹ m⁻³ and is the third most abundant neutral. It is produced both by direct dissociation of SF6 (G2) and by multi-step processes (G5, recombinations).
- **F2** (purple) rises to ~3–4×10¹⁹ m⁻³. It is produced primarily by surface recombination of F atoms with adsorbed F (S8).
- **SF5** (green) plateaus at ~10¹⁹ m⁻³. Although it is the primary dissociation product of SF6, it is rapidly consumed by recombination with F (G35: SF5+F→SF6), which keeps its density relatively low and flat.
- **SF3** (orange) is the least abundant SFx radical, rising slowly to ~10¹⁹ m⁻³ at 3500 W. Its low density confirms that products of SF3 (such as SF2, SF, S) can be neglected.

**Right panel — Charged species densities vs Power (log scale):**
Shows the ion and electron populations.

- **F⁻** (purple dashed) is the dominant negative ion at all powers, reaching ~10¹⁷ m⁻³. It is produced by dissociative attachment of SF6 (G17).
- **SF5+** (blue solid) is the dominant positive ion, consistent with SF5 being the primary ionization product of SF6 (G8). Its cross section for dissociative ionization is the largest among all ions.
- **SF6⁻** (cyan dashed) is significant at low power but decreases relative to F⁻ at higher power as SF6 is depleted.
- **SF4+**, **SF3+** (green, red) are secondary positive ions, about 3–5× lower than SF5+.
- **F2+** (orange) is the least abundant positive ion.
- **e⁻** (black dash-dot) rises from ~10¹⁴ at 50 W to ~4×10¹⁶ at 3500 W. The electron density is much lower than the total positive ion density because the negative ions carry most of the negative charge — this is the hallmark of an electronegative plasma.

---

## Figure 4: `species_densities_vs_Te_power_sweep.png`

**Conditions:** Same data as Figure 3, but plotted against electron temperature Te (which ranges from ~5.0 eV at 50 W down to ~4.5 eV at 3500 W). Note that Te *decreases* as power increases, so the x-axis runs right-to-left relative to the power sweep.

**Left panel — Neutral species vs Te:**
This representation reveals how the gas composition depends on the electron energy distribution. At high Te (low power, right side), the plasma is weakly dissociated: SF6 dominates and all fragment densities are low. As Te decreases (higher power), more power is available to sustain higher ne, which drives more dissociation even though the per-electron rate is slightly lower.

The key physics is that Te is set primarily by the ionization-attachment balance, not by power directly. Higher power creates more electrons, which increases the total ionization rate. To maintain balance with attachment (which scales the same way with ne), Te must decrease slightly. The neutral composition then responds to the combination of ne and Te.

**Right panel — Charged species vs Te:**
All charged species increase as Te decreases (i.e., as power increases), reflecting the growth of ne. The negative ions F⁻ and SF6⁻ track each other, with F⁻ always dominant. The spread between species narrows at low Te because the plasma becomes less electronegative (α decreases from ~60 to ~4).

---

## Figure 5: `species_densities_vs_Te_pressure_sweep.png`

**Conditions:** Same data as the pressure sweep (P = 2000 W, p_OFF = 0.3–4.5 Pa), plotted against Te. Here Te *increases* with p_OFF (from ~4.4 eV at 0.3 Pa to ~4.85 eV at 4.5 Pa), so the x-axis runs left-to-right in the same direction as increasing pressure.

**Left panel — Neutral species vs Te:**
This figure shows a qualitatively different relationship from the power sweep. At low Te (low pressure), the discharge is highly dissociated: F, F2, SF4, SF5 have high densities and SF6 is relatively depleted. As Te increases (higher pressure), SF6 density rises steeply — there is simply more parent gas in the reactor, and the fractional dissociation decreases. The dissociation product densities (F, SF4, F2) remain relatively flat because although there is more SF6 to dissociate, the electrons are shared over more SF6 molecules.

**Right panel — Charged species vs Te:**
The electron density drops dramatically as Te increases (higher pressure), from ~4.5×10¹⁶ m⁻³ at 4.4 eV to ~3×10¹⁵ m⁻³ at 4.85 eV. This is because at higher pressure the discharge becomes more electronegative — more SF6 is available for attachment, producing more F⁻ and SF6⁻, which replace electrons as the negative charge carriers. SF5+ remains the dominant positive ion throughout, but its density also decreases at higher Te because the electron density (which drives ionization) is lower.

---

## Figure 6: `all_species_and_Te_vs_power.png`

**Conditions:** p_OFF = 0.921 Pa, power swept from 50 to 3500 W. Three-panel overview.

This is the comprehensive power sweep figure that corresponds to Figure 7 in the published paper. It combines all information into three panels:

**Left panel — Neutral species (same as Figure 3 left).**

**Center panel — Charged species (same as Figure 3 right).**

**Right panel — Electronegativity (α = n⁻/ne) and electron temperature (Te) vs Power:**
Two quantities on dual y-axes. The electronegativity ratio α starts extremely high (~62 at 50 W) and decreases hyperbolically with power, reaching ~4 at 3500 W. This reflects the transition from a strongly electronegative regime (where negative ions vastly outnumber electrons) to a mildly electronegative regime. The physical reason is that at higher power, ne increases and the attachment rate cannot keep up — the negative ion production is limited by the SF6 density, while ionization benefits from both SF6 and its fragments.

Te decreases from ~5.0 eV at 50 W to ~4.5 eV at 3500 W. The decrease happens because at higher power, more fragments (SF5, SF4, F2, F) are present, and these provide additional ionization channels (G11–G16) with lower thresholds than SF6 ionization. The system can sustain the required ionization rate at a lower Te.

---

## Figure 7: `all_species_and_Te_vs_pOFF.png`

**Conditions:** Power = 2000 W, p_OFF swept from 0.3 to 4.5 Pa. Three-panel overview.

This is the comprehensive pressure sweep figure that corresponds to Figure 8 in the published paper.

**Left panel — Neutral species:**
SF6 increases steeply with pressure (more gas in the reactor), while dissociation products remain relatively flat or increase slowly. At the highest pressures, SF6 dominates by more than an order of magnitude over all other species.

**Center panel — Charged species:**
At the lowest pressure (0.3 Pa), the discharge is nearly electropositive: ne is high (~4.5×10¹⁶ m⁻³), and the electron density is comparable to the positive ion density. As pressure increases, ne drops by more than an order of magnitude while the negative ion densities (F⁻, SF6⁻) remain high or increase, creating a strongly electronegative plasma.

**Right panel — Electronegativity and Te vs p_OFF:**
α increases nearly linearly from ~2 at 0.5 Pa to ~45 at 4.5 Pa. This linear increase with pressure is consistent with experimental observations by Chabert et al. (1999). Te increases from ~4.4 eV to ~4.85 eV with pressure. This is the opposite trend from the power sweep and arises because at higher pressure the discharge is more electronegative — electrons are depleted by attachment, so Te must rise to maintain the ionization rate needed to sustain the plasma.

---

## Summary of Key Physical Trends

| Trend | Physics |
|-------|---------|
| Te decreases with power | More fragment species provide low-threshold ionization pathways |
| Te increases with pressure | More SF6 enhances attachment, depleting electrons; Te must rise to sustain ionization |
| α decreases with power | ne grows faster than n⁻ because ionization benefits from multiple species |
| α increases with pressure | More SF6 feeds attachment, producing more F⁻ |
| SF6 decreases with power | Progressive dissociation by electrons |
| Pressure rise increases with power | More dissociation = more gas-phase particles |
| Pressure rise decreases at high p_OFF | SFx deposition on walls removes gas-phase particles |
| F⁻ dominant negative ion | Dissociative attachment of SF6 (G17) has the largest attachment cross section |
| SF5+ dominant positive ion | Dissociative ionization of SF6 (G8) has the largest ionization cross section |
| SF5 density is flat vs power | Rapid recombination with F (G35) acts as a buffer |
