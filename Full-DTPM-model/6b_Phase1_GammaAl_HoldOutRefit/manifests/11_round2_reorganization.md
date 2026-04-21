# Round-2 Reorganization Manifest

Date executed: 2026-04-18

This round reclassified the 52 folders that had accumulated at the root of `SF6_unified/` after the first cleanup (manifests 01–10). All moves were directory renames inside the same volume; no content was rewritten or deleted.

The taxonomy from manifest `02_cleanup_manifest.md` was extended with three additional families that surfaced in this batch:

- `sf6_2d_icp` — unified 6-generation 2D ICP simulator (was scattered across 6 snapshot folders)
- `sf6_ar_wallchem` — wall-surface chemistry extension built on the Lallement / Kokkoris bases
- `kokkoris_global_model` / `lallement_global_model` / `nf3_global_model` — standalone reproductions of three published global plasma models (distinct from the SF6 global model lineage already covered)
- `benchmark_data` — digitised reference data extracted from Kokkoris, Lallement, and Mettler papers (kept under `active_projects/` since it is referenced directly by the live projects)

---

## New canonical promotions (→ `active_projects/`)

| New canonical path | Source folder | mtime | Why this one |
|---|---|---|---|
| `active_projects/sf6_2d_icp/` | `sf6_icp_2d_final/` | 2026-04-17 | Newest mtime in entire repo. Unified 6-generation framework with `archive/`, `generation_1`–`generation_5`, git repo and venv. Strict superset of the five other "publication / publishable / for presentation" snapshots. |
| `active_projects/sf6_ar_wallchem/` | `sf6ar_wallchem_project_final/` | 2026-03-21 | Self-titled "final"; byte-identical to "Kokkoris and Lallement sf6_wallchem_project" (which was the duplicate); strictly larger code tree than `publication:wallchem_extension`. |
| `active_projects/kokkoris_global_model/` | `sf6_kokkoris/` | 2026-03-15 | Full reproduction of Kokkoris 2009 with outputs + figures. Near-identical to `Kokkoris_v1and3_SF6_Global_Plasma_Model` but slightly larger and earlier-named. |
| `active_projects/lallement_global_model/` | `sf6_lallement/` | 2026-03-21 | Sole full reproduction of Lallement 2009; no competing variant. |
| `active_projects/nf3_global_model/` | `NF3 Global Model Final/` | 2026-03-13 | Distinct chemistry family (NF₃ vs SF₆); only one folder in this family. |
| `active_projects/benchmark_data/{kokkoris,lallement,mettler}/` | 6 benchmark folders | 2026-03-15…21 | New bucket for digitised reference data referenced by the live projects. |

---

## TEL/M7 conflict resolution

Five `TEL_Simulation_Full_Package*` variants were present at root, with the newest (`TEL_Simulation_Full_Package 2`, mtime 2026-04-12) being newer than the canonical `active_projects/tel_model/` (mtime 2026-04-16, but that mtime reflects the cleanup move, not the code).

`diff -rq active_projects/tel_model/ "TEL_Simulation_Full_Package 2"/` showed the canonical `tel_model` is a **strict superset**:

- `tel_model` exclusively contains: 6 milestone markdown docs (`HYBRID_SPATIAL_ARCHITECTURE.md`, `PINN_STRATEGY_DECISION.md`, `PURE_PINN_NEGATIVE_RESULT.md`, `SPATIAL_LEARNING_TARGETS.md`, `TECHNICAL_MILESTONE_V2.md`), the entire `results/` family (`ablation_study/`, all `pinn_dataset_*` and `surrogate_*` variants, `literature_validation/`, `mesh_convergence/`, `transfer_learning/`, …), the `scripts/` directory, `data/lxcat/`, plus several lxcat-related src files.
- `TEL_Simulation_Full_Package 2` exclusively contains: nothing — only re-rendered copies of files that already exist in `tel_model` (animations, docs/figures, main.pdf, slides.pdf, results.tex).

Decision: **keep `tel_model` untouched, archive all five TEL_Simulation_Full_Package variants** under `archive_versions/tel_model/`. The re-rendered outputs in #2 are preserved there if any need to be cherry-picked later.

---

## Archive layout

```
archive_versions/
├── duplicates/                                          # byte-identical or near-identical copies
│   ├── sf6_plasma_model_copy{1,2,3}/                    # 3 byte-identical copies; copy1 retained as canonical of the group
│   ├── Kokkoris_v1and3_near_dup_of_sf6_kokkoris/
│   ├── SF6_simulation_final_near_dup_of_SF6_global_model_final/
│   └── Kokkoris_and_Lallement_wallchem_dup/
├── tel_model/                                           # 7 superseded TEL snapshots
│   ├── v_apr01_0d_results/
│   ├── v_apr01_complete_code/
│   ├── v_apr01_minimal_src/
│   ├── v_apr04_626pm/
│   ├── v_apr04_626pm_2/
│   ├── v_apr04_pkg5/
│   └── v_apr12_rerendered_outputs/                      # newest of the 5 TEL_Simulation_Full_Package variants
├── sf6_2d_icp/                                          # 5 older 2D-ICP publication snapshots
├── sf6_ar_wallchem/                                     # 2 older wallchem snapshots
├── kokkoris_global_model/                               # 2 older intermediate snapshots
├── sf6_global_legacy/                                   # 7 legacy SF6 0D global model variants
├── sf6_2d_legacy/                                       # 6 pre-unified 2D variants
├── misc/                                                # 2 standalone reference artifacts
│   ├── Ar67Kinkanalysis/                                # kink stability TeX+PDF report
│   └── sarf_website/                                    # static brand site
├── intermediate_merges/                                 # (round-1, untouched)
├── phase2_electron_kinetics/                            # (round-1, untouched)
├── plasma_dtpm/                                         # (round-1, untouched)
└── sf6_tier3/                                           # (round-1, untouched)
```

---

## Deliverables added

| `deliverables/` path | Source | Contents |
|---|---|---|
| `deliverables/dtpm/` | `DTPM/` | DTPM_Complete_Delivery.zip, presentation PPTX, speaker script |
| `deliverables/pub2d/` | `Pub2D/` | Publication / presentation materials for the 2D project |
| `deliverables/final_draft_for_publication/` | `final draft for publication/` | merged_paper.pdf + .tex |

---

## Verdict legend (used across the table)

- **LATEST** — newer than what was previously canonical; promoted to `active_projects/`
- **DUPLICATE** — byte-identical or strict subset of an existing organized version; archived under `archive_versions/duplicates/`
- **OUTDATED_VERSION** — older snapshot of an already-canonical family; archived under `archive_versions/<family>/`
- **DISTINCT** — a project family not previously represented; promoted as new `active_projects/<family>/`
- **DELIVERABLE** — packaged PDF/zip/slides; routed to `deliverables/`
- **BENCHMARK_DATA** — digitised experimental reference data; routed to `active_projects/benchmark_data/`

---

## Per-folder verdict table

| Source folder | Family | Verdict | Destination |
|---|---|---|---|
| sf6_icp_2d_final | sf6_2d_icp | LATEST/DISTINCT | active_projects/sf6_2d_icp |
| sf6ar_wallchem_project_final | sf6_ar_wallchem | LATEST/DISTINCT | active_projects/sf6_ar_wallchem |
| sf6_kokkoris | kokkoris_global_model | DISTINCT | active_projects/kokkoris_global_model |
| sf6_lallement | lallement_global_model | DISTINCT | active_projects/lallement_global_model |
| NF3 Global Model Final | nf3_global_model | DISTINCT | active_projects/nf3_global_model |
| kokkoris_benchmark_data_publication | benchmark_data | BENCHMARK_DATA | active_projects/benchmark_data/kokkoris/publication_csv |
| kokkoris-extracted-data | benchmark_data | BENCHMARK_DATA | active_projects/benchmark_data/kokkoris/extracted_data |
| kokkoris-overlay_plots_and_csv | benchmark_data | BENCHMARK_DATA | active_projects/benchmark_data/kokkoris/overlay_plots |
| Lallement Figures | benchmark_data | BENCHMARK_DATA | active_projects/benchmark_data/lallement/figures |
| Mettler Figures | benchmark_data | BENCHMARK_DATA | active_projects/benchmark_data/mettler/figures |
| mettler benchmarking | benchmark_data | BENCHMARK_DATA | active_projects/benchmark_data/mettler/benchmarking |
| DTPM | dtpm | DELIVERABLE | deliverables/dtpm |
| Pub2D | sf6_2d_icp | DELIVERABLE | deliverables/pub2d |
| final draft for publication | sf6_2d_icp | DELIVERABLE | deliverables/final_draft_for_publication |
| TEL_Simulation_Full_Package 2 | tel_model | OUTDATED_VERSION (newer outputs only) | archive_versions/tel_model/v_apr12_rerendered_outputs |
| TEL_Simulation_Full_Package_5 | tel_model | OUTDATED_VERSION | archive_versions/tel_model/v_apr04_pkg5 |
| TEL_Simulation_Full_Package | tel_model | OUTDATED_VERSION | archive_versions/tel_model/v_apr01_minimal_src |
| TEL_Simulation_Full_Package 6.26.01 pm | tel_model | OUTDATED_VERSION | archive_versions/tel_model/v_apr04_626pm |
| TEL_Simulation_Full_Package 2 6.26.01 pm | tel_model | OUTDATED_VERSION | archive_versions/tel_model/v_apr04_626pm_2 |
| TEL_Complete_Code | tel_model | OUTDATED_VERSION | archive_versions/tel_model/v_apr01_complete_code |
| tel_0d_results | tel_model | OUTDATED_VERSION | archive_versions/tel_model/v_apr01_0d_results |
| sf6_plasma_model | sf6_global_legacy | DUPLICATE (3 copies) | archive_versions/duplicates/sf6_plasma_model_copy1 |
| sf6_plasma_model 2 | sf6_global_legacy | DUPLICATE | archive_versions/duplicates/sf6_plasma_model_copy2 |
| sf6_plasma_model 3 | sf6_global_legacy | DUPLICATE | archive_versions/duplicates/sf6_plasma_model_copy3 |
| Kokkoris_v1and3_SF6_Global_Plasma_Model | kokkoris_global_model | DUPLICATE | archive_versions/duplicates/Kokkoris_v1and3_near_dup_of_sf6_kokkoris |
| SF6_simulation_final | sf6_global_legacy | DUPLICATE | archive_versions/duplicates/SF6_simulation_final_near_dup_of_SF6_global_model_final |
| Kokkoris and Lallement sf6_wallchem_project | sf6_ar_wallchem | DUPLICATE | archive_versions/duplicates/Kokkoris_and_Lallement_wallchem_dup |
| publication-final:sf6_icp_2d_project | sf6_2d_icp | OUTDATED_VERSION | archive_versions/sf6_2d_icp/v_mar26_publication_final |
| for presentation:sf6_icp_2d_final | sf6_2d_icp | OUTDATED_VERSION | archive_versions/sf6_2d_icp/v_mar24_presentation |
| publishable\sf6_2d_full | sf6_2d_icp | OUTDATED_VERSION | archive_versions/sf6_2d_icp/v_mar24_publishable_no_anchors |
| documentation\sf6_icp_2d_project_self_consistent | sf6_2d_icp | OUTDATED_VERSION | archive_versions/sf6_2d_icp/v_mar24_self_consistent_docs |
| final for pub\sf6_icp_2d_project_self_consistent_latex | sf6_2d_icp | OUTDATED_VERSION | archive_versions/sf6_2d_icp/v_mar25_self_consistent_latex |
| publication:wallchem_extension | sf6_ar_wallchem | OUTDATED_VERSION | archive_versions/sf6_ar_wallchem/v_mar22_publication |
| extension:sf6ar_plasma_model_lallement | sf6_ar_wallchem | OUTDATED_VERSION | archive_versions/sf6_ar_wallchem/v_mar31_lallement_extension |
| v3_kokkoris | kokkoris_global_model | OUTDATED_VERSION | archive_versions/kokkoris_global_model/v_mar17_v3_subset |
| for publication: kokkoris SF6 global model | kokkoris_global_model | OUTDATED_VERSION | archive_versions/kokkoris_global_model/v_mar22_publication_intermediate |
| SF6 global model final | sf6_global_legacy | OUTDATED_VERSION | archive_versions/sf6_global_legacy/SF6_global_model_final_mar12 |
| SF6_Global_Plasma_Model6 | sf6_global_legacy | OUTDATED_VERSION | archive_versions/sf6_global_legacy/Global_Plasma_Model6_mar19 |
| sf6_plasma_model_package | sf6_global_legacy | OUTDATED_VERSION | archive_versions/sf6_global_legacy/plasma_model_package_mar13 |
| sf6_final_project | sf6_global_legacy | OUTDATED_VERSION | archive_versions/sf6_global_legacy/sf6_final_project_mar22 |
| sf6_ar_project | sf6_global_legacy | OUTDATED_VERSION | archive_versions/sf6_global_legacy/sf6_ar_project_mar22 |
| sf6arfinal | sf6_global_legacy | OUTDATED_VERSION | archive_versions/sf6_global_legacy/sf6arfinal_mar13 |
| updates SF6 | sf6_global_legacy | OUTDATED_VERSION | archive_versions/sf6_global_legacy/updates_SF6_mar12 |
| sf6_2d_DTPM_complete | sf6_2d_legacy | OUTDATED_VERSION | archive_versions/sf6_2d_legacy/sf6_2d_DTPM_complete_mar16 |
| sf6_2d_complete | sf6_2d_legacy | OUTDATED_VERSION | archive_versions/sf6_2d_legacy/sf6_2d_complete_mar16 |
| sf6_2d_bolsig_final | sf6_2d_legacy | OUTDATED_VERSION | archive_versions/sf6_2d_legacy/sf6_2d_bolsig_final_mar16 |
| sf6_gen5_complete | sf6_2d_legacy | OUTDATED_VERSION | archive_versions/sf6_2d_legacy/sf6_gen5_complete_mar24 |
| sf6_gen5_complete_1 | sf6_2d_legacy | OUTDATED_VERSION | archive_versions/sf6_2d_legacy/sf6_gen5_complete_1_mar24 |
| sf6_dist | sf6_2d_legacy | OUTDATED_VERSION | archive_versions/sf6_2d_legacy/sf6_dist_mar16 |
| Ar67Kinkanalysis | misc | DISTINCT (small) | archive_versions/misc/Ar67Kinkanalysis |
| sarf-website | misc | DISTINCT (infrastructure) | archive_versions/misc/sarf_website |

---

## Verification

After execution the repo root contains only the five organized buckets and the README:

```
active_projects/  archive_versions/  deliverables/  final_package/  manifests/  README.md
```

No file content was modified. All moves were `mv` within the same volume; revert is possible by inverting any row of the table above.
