# ChatGPT / Claude-Web Prompt — Mettler ↔ 06a/Phase-1 Cross-Check

Paste this entire block into a new ChatGPT Project or Claude.ai conversation
after attaching both PDFs:

1. `Mettler Dissertation.pdf`
2. Either the Phase-1 `main.pdf` (from `Steps/5.Phase1_EM_Chemistry_Merged/docs/report/main.pdf`) or the 06a consolidated report `main.pdf` — state which in your first message.

---

## Prompt

```
I have attached two PDFs:
  1. "Mettler Dissertation.pdf" — Jeremy Mettler's 2025 UIUC PhD thesis on
     spatially resolved fluorine-radical probes in a TEL etcher.
  2. "main.pdf" — my consolidated SF6/Ar plasma-model report that benchmarks
     against Mettler.

My text-only analysis of main.tex has already flagged that the report may
contain the following errors. Please treat these as hypotheses and VISUALLY
CONFIRM or REFUTE each using the attached figures. Do not just restate my
claims — open the referenced figures in the Mettler PDF and match them
against what my report says.

CLAIMS TO VERIFY:

  (a) My report describes Mettler's diagnostic as "BBAS + OES". Mettler's
      actual primary diagnostic is W/Al non-equilibrium radical probes +
      Ar/SF6 actinometry. Please confirm by looking at Mettler's Ch. 3 and
      §4.3.

  (b) My report cites "Mettler Fig 4.5" in the context of absolute [F] vs
      ICP power in the TEL etcher. Please open Mettler's Fig 4.5 (page ~54)
      and confirm it is actually HELICON / PMIC data, not TEL. Also confirm
      that Mettler's true TEL absolute-[F] data lives in Fig 4.9 (p. 61) as
      an SF6-FLOW sweep, not a power sweep.

  (c) My report repeatedly says "74% centre-to-edge [F] drop" as if it's a
      single universal Mettler number. Please open Fig 4.14 (p. 70) and Fig
      4.17 (p. 75) and confirm that:
        - The drop ranges from 67% to 75% depending on SF6/Ar composition.
        - The 74% specifically applies to the 90% SF6 condition, bias OFF.
        - Fig 4.14's cubic fit has coefficients A=1.01032, B=0, C=-0.01847,
          D=7.139e-4 (r in cm).

  (d) My report's baseline is 700 W / 10 mTorr / ~pure SF6 / no bias.
      Mettler's radial benchmark (Fig 4.14/4.17) is 1000 W / 10 mTorr /
      70% SF6 / 30% Ar / 200 W rf bias. Please confirm the operating-
      condition mismatch and tell me qualitatively how large an error this
      might introduce — specifically the effect of bias-on (×1.6 at 90%
      SF6, up to ×2.15 at 30% SF6 per Mettler's own Fig 4.17
      side-by-side comparison).

ADDITIONAL TASKS:

  (e) Look at every figure in my main.pdf that mentions Mettler. For each,
      tell me:
        - Does the caption's quoted conditions match the Mettler figure
          being compared?
        - Is the axis/unit conversion correct (Mettler uses #/m^3; I use
          cm^-3; factor of 1e6)?
        - If I have overlaid digitised Mettler points, do they look right
          against the original figure?

  (f) Digitise for me: from Mettler Fig 4.17 TOP PANEL (wafer bias OFF),
      the 90% SF6 (red circles) and 30% SF6 (red triangles) data points at
      r = 0, 2, 4, 6, 8 cm. Return as Python lists of (r_cm, density_per_m3)
      tuples.

  (g) Flag any Mettler TEL figure (4.6 onward) that contains benchmarkable
      data I am NOT currently using in my report.

OUTPUT STRUCTURE:
  Section A — Factual citation errors (confirm/refute each of a, b, c)
  Section B — Operating-condition mismatch (d)
  Section C — Per-figure audit (e) — one paragraph per figure in my report
  Section D — Digitised data (f)
  Section E — Missing benchmarks (g)

Please be specific. If you cannot read a panel clearly, say so explicitly
rather than guessing.
```

---

## Iterative follow-ups

**If the response is too abstract**:
```
Go back to Mettler Fig 4.14 specifically. Read the inset text box with the
cubic-fit coefficients. Confirm A = 1.01032, C = -0.01847, D = 7.139e-4,
R-squared = 0.997. Then evaluate the fit at r = 0, 2, 4, 6, 8 cm and tell
me the 5 normalised densities. These should be approximately 1.00, 0.95,
0.81, 0.53, 0.25.
```

**If visual matching is wrong**:
```
In my main.pdf, find the figure whose caption mentions "Mettler" and
"radial" or "centre-to-edge". Tell me (a) the figure number, (b) what power
and pressure the simulated curve is at, (c) what power and pressure the
Mettler overlaid points are at. If those don't match, that's the core
issue.
```

**Post-digitisation**:
```
Now create a pandas-style table with columns [r_cm, n_F_per_m3, condition]
containing every Mettler TEL radial [F] data point from Fig 4.14, Fig 4.17
top panel (90% and 30% SF6 bias-off), and Fig 4.17 bottom panel (bias-on
versions). I want about 20 rows total.
```
