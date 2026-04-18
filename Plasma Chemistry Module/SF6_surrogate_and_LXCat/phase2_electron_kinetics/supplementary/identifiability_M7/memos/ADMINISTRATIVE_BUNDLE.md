# Administrative bundle

All fields the PSST (or equivalent) submission portal will ask for.
Fields marked `[FILL]` require author input. Fields marked `[DRAFT]`
contain proposed text the author should review and approve before
submitting.

---

## 1. Title

**What two etch-rate measurements constrain: identifiability and EEDF
sensitivity of a reduced SF₆ ICP model**

(16 words. Question-forward. Leads with the scientific contribution
— what can be identified from two measurements — rather than the
method.)

---

## 2. Authors and affiliations

### Corresponding author

- **Name:** `[FILL]`
- **Affiliation:** `[FILL]`
- **ORCID:** `[FILL]`
- **Email:** `[FILL]`
- **Postal address:** `[FILL]`

### Co-authors (in order)

1. Name: `[FILL]` — Affiliation: `[FILL]` — ORCID: `[FILL]`
2. Name: `[FILL]` — Affiliation: `[FILL]` — ORCID: `[FILL]`
3. ...

### Author contributions (CRediT taxonomy) `[DRAFT]`

> **`[Lead author]`**: Conceptualization, Methodology, Software,
> Formal analysis, Investigation, Data curation, Writing — original
> draft, Writing — review & editing, Visualization.
>
> **`[Co-author 2]`**: `[FILL: e.g., Conceptualization, Methodology,
> Supervision, Writing — review & editing, Funding acquisition]`.
>
> **`[Co-author 3]`**: `[FILL]`.

Note to author: CRediT uses 14 contributor roles. The draft above
assigns all technical roles to the lead author because that matches
the structure of the work (single-person model development and
calibration). Adjust if supervision, funding acquisition, or domain
expertise were provided by co-authors.

---

## 3. Significance statement (≤100 words) `[DRAFT]`

> Reduced-order models of inductively coupled plasmas are routinely
> calibrated against a handful of experimental etch-rate measurements,
> and the fitted parameters are then used to predict behavior at
> unexplored operating points. We show that for a two-anchor
> calibration of an SF₆ ICP model, only a single scalar combination
> of the depletion, dissociation, and electron-density parameters is
> identifiable; individual parameters are not. We demonstrate that
> assuming a Maxwellian electron energy distribution produces a
> fitted residence-time parameter that is physically impossible by
> more than a factor of two, and that correcting to a Boltzmann EEDF
> restores order-of-magnitude consistency. We show which single
> additional experiment distinguishes the two cases.

(99 words.)

---

## 4. Keywords (PSST allows 4–6) `[DRAFT]`

- inductively coupled plasma
- SF₆ plasma
- parameter identifiability
- electron energy distribution function
- reduced-order modeling
- silicon etching

---

## 5. Conflict of interest declaration `[DRAFT]`

> The authors declare no financial, personal, or professional
> relationships that could be perceived as influencing the work
> reported in this manuscript.

**If there is a conflict to declare**, replace with the specific
disclosure in the form the journal requires (for PSST: name the
institution, funding source, or commercial relationship and describe
the potential conflict in one or two sentences).

---

## 6. Funding acknowledgment `[FILL]`

> This work was supported by `[FILL: grant agency, grant number,
> project name if applicable]`. The authors thank `[FILL: any
> computational resources, data providers, or individuals to
> acknowledge]`.

Specific entities that should probably be acknowledged based on the
methods used in the paper:
- **LXCat / Biagi database** for the SF₆ and Ar cross sections used
  to build the Boltzmann rate tables. PSST's standard acknowledgment
  is a one-line mention of the database and a retrieval date in
  both the acknowledgments section and the reference list.
- **BOLSIG+** / Hagelaar & Pitchford 2005 — cited in the reference
  list, no separate acknowledgment needed.
- Anyone who provided computing time or code review.

---

## 7. Data and code availability statement `[DRAFT]`

> The reduced-model simulation code (Python), the Boltzmann rate
> tables (HDF5), the M7g calibration and rate-source-swap experiment
> script, and all figure-generation scripts used in this work are
> available at `[FILL: Zenodo DOI or GitHub URL]` under the
> `[FILL: MIT / BSD / Apache 2.0]` license. The CSV files containing
> the power-sweep comparison data plotted in Figures 3–7 are
> included in the repository at the paths listed in the
> supplementary reproducibility script. No new experimental data
> were generated in this work; the two literature etch-rate anchor
> values used for calibration are reproduced from Mansano et al.
> (1999) and Panduranga et al. (2019), both of which are cited in
> the reference list.

---

## 8. Suggested referees (PSST asks for 3–5) `[DRAFT]`

The paper sits at the intersection of low-temperature plasma
modeling, Boltzmann solver methodology, and reduced-order modeling
for semiconductor processing. Three natural referee communities:

### A. Reduced-model and equipment-modeling community
1. **Mark J. Kushner** (University of Michigan)
   — Affiliation: ECE, University of Michigan, Ann Arbor, MI, USA
   — Expertise: HPEM and reduced-order ICP modeling, SF₆ plasma
     chemistry
   — Why: foundational author on reduced-model calibration against
     experimental anchors; will scrutinize the identifiability claim

2. **Miles M. Turner** (Dublin City University)
   — Affiliation: National Centre for Plasma Science and Technology,
     DCU, Dublin, Ireland
   — Expertise: verification and validation of plasma fluid models,
     uncertainty quantification
   — Why: published extensively on the reproducibility and
     sensitivity of low-temperature plasma simulations

### B. BOLSIG+ / cross-section community
3. **Gerjan J. M. Hagelaar** (LAPLACE, Toulouse)
   — Affiliation: LAPLACE, Université Paul Sabatier, Toulouse, France
   — Expertise: author of BOLSIG+, two-term Boltzmann approximation,
     rate-coefficient calculation
   — Why: the paper's central EEDF-sensitivity claim rests on the
     Maxwell vs 2-term Boltzmann contrast. Natural reviewer for the
     §5 rate-source-swap argument. Note: the paper cites his 2005
     paper, so this is not a conflict — he may actually decline for
     that reason, in which case suggest Leanne Pitchford instead.

4. **Leanne C. Pitchford** (LAPLACE / LXCat)
   — Affiliation: LAPLACE, Toulouse, France
   — Expertise: co-author of BOLSIG+, LXCat database curation
   — Why: alternative to Hagelaar; same community

### C. SF₆ plasma / silicon etch community
5. **Annemie Bogaerts** (University of Antwerp)
   — Affiliation: PLASMANT, Department of Chemistry, University of
     Antwerp, Belgium
   — Expertise: hybrid Monte-Carlo / fluid modeling of SF₆ ICP
     silicon etching (with Tinck, Dussart)
   — Why: published the closest-neighbor SF₆ ICP modeling work;
     will be quick to read the methodology section

### Referees to suggest as "do not include"
- `[FILL IF NECESSARY]`: if there are reviewers the author has had
  direct conflict or collaboration with in the past 3 years, list
  them here. PSST allows up to 3 exclusions.

---

## 9. Suggested topical category / section

- Primary: **Modeling** (PSST has explicit modeling section)
- Secondary: **Methods and instrumentation** (for the identifiability
  analysis and rate-source-swap experiment design)

---

## 10. Cover letter `[DRAFT]`

> Dear Editor,
>
> Please consider the attached manuscript,
> "What two etch-rate measurements constrain: identifiability and EEDF
> sensitivity of a reduced SF₆ ICP model", for publication in
> *Plasma Sources Science and Technology*.
>
> This work addresses a question that arises routinely in reduced-order
> plasma modeling but is rarely made quantitative: given a handful of
> literature etch-rate measurements, which parameters of a reduced
> ICP model are actually identified, and which are not? We derive a
> closed-form identifiability theorem for a two-anchor calibration of
> an SF₆ / silicon ICP model and show that the calibration constrains
> only a single scalar combination of the depletion, dissociation,
> and electron-density parameters — not each parameter separately.
>
> We then use this result to demonstrate a concrete methodological
> consequence. Replacing the Maxwellian EEDF assumption (a common
> default in analytic plasma modeling) with a two-term Boltzmann EEDF
> built from standard LXCat cross sections shifts the fitted depletion
> timescale by a factor of ~36, and we show that the direction and
> magnitude of the shift are predicted exactly by our identifiability
> theorem to a residual of less than one percent. The Maxwellian fit
> produces a depletion timescale that is physically impossible as a
> gas residence time for any reasonable flow rate; the Boltzmann fit
> produces an order-of-magnitude-consistent value. We further compute
> which single additional measurement would distinguish the two
> cases, and identify Ar dilution at x_Ar ≈ 0.3 and 200 W as the
> optimal discriminating experiment.
>
> We believe this work will be of interest to PSST's modeling and
> methodology readership because it (1) provides a worked example
> of how identifiability analysis can be applied to plasma calibration
> workflows rather than left implicit, (2) shows a quantitative,
> reproducible consequence of the Maxwellian-EEDF default in a
> practically relevant gas, and (3) formulates a concrete experimental
> follow-up designed to lift the parameter degeneracy. The full
> simulation code, Boltzmann rate tables, and figure-generation
> scripts are provided at `[FILL]`.
>
> The manuscript has not been submitted to any other journal and is
> not under consideration elsewhere. All authors have read and
> approved the submission.
>
> Thank you for considering our work.
>
> Sincerely,
>
> `[FILL: Corresponding author name]`
> on behalf of all authors
