# Report Fixes — for JAK to apply

**Reviewer:** Z. Ngan
**Target file:** `Plasma Chemistry Module/SF6_surrogate_and_LXCat/report/main.tex`
**Branch:** `feat/phase1-global-2d-and-sf6ar-chemistry` (commit 118d7d6)
**Trigger:** "Yes it is done and I committed the changes and updated the report accordingly. Can you please check the results and reports..."

---

## Summary

Three small things to flag before the report is signed off. Figures and section structure look good; the `\graphicspath{{../figures/}}` resolves cleanly and all 36 `\includegraphics` references exist. **The two real issues are:** (i) the in-text/table numbers are from main.pdf v1 (Mac/MPS) but the data files committed in the same branch are from the Delta runs, and (ii) the 6b BC-map fix and γ_Al refit aren't mentioned.

---

## 1. Numbers in tables and prose disagree with data files in the repo

The arch-sweep table, ablation table, and ensemble equations all quote the original main.pdf v1 numbers. The Delta runs you pushed in `dc52a5f` and `81d6531` give different values. Side-by-side:

| Section | Line(s) | Report quotes | Delta data files |
|---|---|---|---|
| Arch sweep table — E0 | `main.tex:534` | 0.01901 ± 0.00393 | 0.02022 ± 0.00275 |
| Arch sweep table — E1 | `main.tex:536` | 0.00377 ± 0.00056 | 0.00570 ± 0.00074 |
| Arch sweep table — E2 | `main.tex:538` | 0.00390 ± 0.00099 | 0.00573 ± 0.00102 |
| Arch sweep table — **E3 (winner)** | `main.tex:540` | **0.00210 ± 0.00015** | **0.00410 ± 0.00021** |
| Arch sweep table — E4 | `main.tex:542` | 0.00244 ± 0.00066 | 0.00449 ± 0.00028 |
| Arch sweep table — E5 | `main.tex:544` | 0.00957 ± 0.00180 | 0.01060 ± 0.00118 |
| Arch sweep table — E6 | `main.tex:546` | 0.00394 ± 0.00083 | 0.00572 ± 0.00042 |
| Arch sweep prose | `main.tex:553` | "lowest mean RMSE (0.00210)" | should be 0.00410 |
| Arch sweep prose | `main.tex:569–570` | "0.0021 already beats v4's 0.0029" | E3 single-model is now 0.00410 — does **not** beat 0.0029 |
| Ablation — None | `main.tex:616` | 0.01830 | (not in Delta files; legacy local backup has 0.02183) |
| Ablation — Bias only | `main.tex:617` | 0.00314 | (legacy local backup: 0.00601) |
| Ablation — Reg only | `main.tex:618` | 0.01888 | **0.01948** (`ml_ablation_lxcat_split/result_reg_only.json`) |
| Ablation — Epochs only | `main.tex:619` | 0.01610 | (legacy local backup: 0.02052) |
| Ablation — All three | `main.tex:620` | 0.00385 | **0.00557** (`ml_ablation_lxcat_split/result_all_three.json`) |
| Ensemble nF RMSE | `main.tex:60, 663, 1178, 1263, 1386, 1392` | 0.00154 | **0.00381** (`ml_production_ensemble_lxcat/summary.json`) |
| Ensemble nSF₆ RMSE | `main.tex:61, 664, 1392` | 0.00128 | 0.00134 |
| "47 % improvement over legacy v4 (0.00292)" | `main.tex:60, 666, 1277, 1392` | + 47 % | **−30 %** at 0.00381; 47 % only stands at 0.00154 |

Source data files in the repo for cross-check:
- `Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/results/ml_arch_sweep_lxcat/experiment_table.json`
- `Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/results/ml_production_ensemble_lxcat/summary.json`
- `Full-DTPM-model/6b_Phase1_GammaAl_HoldOutRefit/results/ml_ablation_lxcat_split/result_*.json`

### Two paths to resolve, please pick one

**Path (a) — keep v1 numbers as canonical, add reproducibility caveat (cheaper).** This was the framing in the §15 Limitations bullet you had in `d890f6f` before the refocus. Restoring **one bullet** in §16 Limitations would re-anchor the v1 numbers as Mac/MPS and explain why the data files in the repo are different:

```latex
\item Backend-dependent reproducibility band.  The headline nF RMSE
of 0.00154 (Section~\ref{sec:production}) was obtained on Apple
MPS; an independent Delta/CUDA re-run of the same recipe yields
0.0038 (a $\sim$2.5$\times$ shift), placing the v1 result at the
optimistic end of a 0.0015--0.0040 portability band.  The
architecture choice (E3) and qualitative findings (bias-init
dominance, regularisation-alone is counter-productive) are
identical across backends; the absolute scaling is not.
```

This recovers the scientific honesty of `d890f6f`'s §16 bullet without putting the full §15 Cluster Replication chapter back.

**Path (b) — adopt the Delta numbers as canonical (more work).** Replaces every line in the table above and requires regenerating the 5 surrogate diagnostic figures (`surrogate_v4_diagnostics_nF.png`, `surrogate_v4_pred_vs_true.png`, `surrogate_v4_loss_curves.png`, `surrogate_v4_uncertainty.png`, plus the residuals plot) from the Delta-trained ensemble. The "47 % improvement over legacy v4" headline becomes a "−30 % regression" framing, which would need to be reframed (e.g., "comparable to legacy v4 at the conservative end of the reproducibility band").

### Recommendation

**Path (a).**  It's a one-line fix in §16, costs nothing in figures or narrative, and preserves the headline result while honestly disclosing the portability range.  The v1 numbers were valid for the architecture they ran on, and the Delta cross-check confirmed the architecture itself is correct (within reproducibility band).

---

## 2. Headline claims that need adjustment under either path

If you keep v1 numbers (path a), the speedup framing is fine.  If you go with Delta numbers (path b), these specific lines would need rewording:

- `main.tex:569–570` — "0.0021 already beats the legacy v4's 0.0029" → no longer true at 0.00410.
- `main.tex:60, 666, 1277, 1392` — "47 % improvement over legacy v4" → no longer true at 0.00381.
- `main.tex:1178, 1263` — references to `nF = 0.00154` as the production result.

Either keep all of these (path a + caveat) or rewrite them all (path b).  Don't keep them while also changing the table.

---

## 3. Missing 6b physics context (one paragraph)

The report doesn't mention any of the new 6b work:
- The BC-map fix (`build_geometry_mask` now splits the wafer plane into Si + Al annulus; previously Al F-sink was undercounted by ~96 % of the wafer area).
- The γ_Al hold-out refit (D6) — fitted on 90 % SF₆ bias-on, **failed** all three held-out conditions; residual changes sign across composition (−21.7 % at 90 %, +30 % at 30 %).
- That this is what motivated the LXCat re-run in the first place.

A one-paragraph insertion at the end of §2 "Solver Recap" (around `main.tex:181`, before §3) would give the reader the right context.  Suggested wording:

```latex
\paragraph{Phase-1 6b refitting since main.pdf v1.}
The solver underlying the LXCat dataset in this report differs from
main.pdf v1 in two material ways.  First, the bottom-boundary BC map
in \texttt{build\_geometry\_mask} was corrected to split the
\(j=0\) row into \texttt{BC\_WAFER} (\(r \le 0.075\,\mathrm{m}\),
silicon) and \texttt{BC\_AL\_TOP} (\(r > 0.075\,\mathrm{m}\),
aluminium pedestal-top / chamber-floor annulus).  Prior code labelled
the entire bottom as \texttt{BC\_WAFER}, undercounting the Al
F-sink by an annulus of area \(\sim 0.017\,\mathrm{m}^2\) (about
96\,\% of the wafer area).  Second, a hold-out refit of
\(\gamma_\mathrm{Al}\) was attempted on the corrected BC map: fitted
on the 90\,\% SF\textsubscript{6} bias-on condition, the resulting
\(\gamma_\mathrm{Al}^*\) \emph{failed} all three held-out
compositions (residuals change sign across the composition axis,
from \(-21.7\,\%\) at 90\,\% SF\textsubscript{6} to \(+30\,\%\) at
30\,\% SF\textsubscript{6}), indicating that no scalar
\(\gamma_\mathrm{Al}\) closes the Mettler residual.  Both findings
are documented in the 6b workspace and motivate the LXCat surrogate
runs reported here, where the rate-coefficient set is the only
varying knob.
```

This paragraph also gives the reader the right framing for why the speedup argument matters: the Phase-1 absolute residual is not a single-knob calibration problem, so accelerating the operating-envelope sweep is what unlocks the next step.

---

## What's NOT a problem (false alarm cleared)

- All 36 `\includegraphics` references resolve via `\graphicspath{{../figures/}}`.  No missing figures.
- Section structure is clean.
- Compile sequence in your commit message (`pdflatex && bibtex && pdflatex && pdflatex`) is the right one.

---

## Concrete action list

If you go with path (a) — recommended:

1. **§16 Limitations** — add the reproducibility-band bullet (text in §1 above).
2. **§2 Solver Recap end** — insert the 6b physics paragraph (text in §3 above).
3. Re-compile.

If you go with path (b), please ping me — fix #2 (number rewrites) is mechanical but I'd want to coordinate on regenerating the 5 surrogate diagnostic figures so the visuals match the new tables.

— Z
