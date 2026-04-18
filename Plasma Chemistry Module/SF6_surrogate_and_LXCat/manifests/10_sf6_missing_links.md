# MISSING_LINKS

Inventory of expected Tier-3 artifacts that are missing or incomplete in `SF6_tier3_repo_merged/`.

---

## Science completeness scorecard

### A. Multicase training

| Artifact | Expected | Status |
|---|---|---|
| Final dataset (`phaseB_teacher_dataset_multicase_v4.npz`) | present | ✔ |
| All intermediate datasets (v1, v2, v3) | present | ✔ |
| Multicase training log (`tier3_pinn_multicase_log.md`) | present | ✔ |
| Multicase prediction plots (loss, train_pred, val_pred) | present | ✔ |
| Multicase checkpoint (`tier3_pinn_multicase.pt`) | present | ✔ |

### B. Rotation generalization

| Artifact | Expected | Status |
|---|---|---|
| 8-ch rotation: 5 holdouts × ≥2 seeds (checkpoints + JSONs) | 15 pt + 15 json | ✔ (15+15) |
| 9-ch PRF rotation: 5 holdouts × 2 seeds | 10 pt + 10 json | ✔ (10+10) |
| v3 rotation: 2 holdouts × 2 seeds | 4 pt + 4 json | ✔ (4+4) |
| v4 rotation: 4 holdouts × 2 seeds | 8 pt + 8 json | ✔ (8+8) |
| Rotation summary tables (md + json) | rotation_table, rotation_prf_table, rotation_v3_table, rotation_v4_table | ✔ (all present) |

### C. Seed variance

| Artifact | Expected | Status |
|---|---|---|
| `seed_variance_summary.json` (v1, hold-out C n=3) | present | ✔ |
| `seed_variance_v2_summary.json` (all holdouts, 8-ch + PRF) | present | ✔ |
| `fig10_seed_variance.png` | present | ✔ |
| `fig13_seed_variance_all_holdouts.png` | present | ✔ |

### D. Trained model weights

| Checkpoint | Expected | Status |
|---|---|---|
| `tier3_pinn_m1.pt` | present | ✔ |
| `tier3_pinn_m2.pt` | present | ✔ |
| `tier3_pinn_m3.pt` | present | ✔ |
| `tier3_pinn_m3b.pt` | present | ✔ |
| `tier3_pinn_m3c.pt` | present | ✔ |
| `tier3_pinn_multicase.pt` | present | ✔ |
| Paired training logs (`*_log.md`) | present for all 6 | ✔ |
| Paired loss/pred PNGs | present for all 6 | ✔ |

### E. Phase A / Phase B

| Artifact | Expected | Status |
|---|---|---|
| `phaseA_local_field_map.py` (generator) | present | ✔ |
| `outputs/phaseA_local_field.npz` | present | ✔ |
| `outputs/phaseA_local_field.png` | present | ✔ |
| `outputs/phaseA_log.md` | present | ✔ |
| `phaseB_teacher_generator.py` (generator) | present | ✔ |
| `outputs/phaseB_teacher_dataset.npz` (single-case) | present | ✔ |
| `outputs/phaseB_teacher_dataset_m2.npz` (single-case, M2 variant) | present | ✔ |
| `outputs/phaseB_log.md` | present | ✔ |
| `outputs/phaseB_sweep.png` (lambda sweep comparison) | present | ✔ |
| Phase-B lambda-comparison PNGs (4 lambda values) | present | ✔ |

### F. Integration scaffold

| Artifact | Expected | Status |
|---|---|---|
| `tier3_pinn/integration_scaffold.py` | present | ✔ |
| `tier3_pinn/fluid_loop_mock.py` | present | ✔ |
| `scripts/integration_smoke_test.py` | present | ✔ |
| `scripts/run_integration_selftest.sh` | present | ✔ |

### G. PIC-MCC comparison harness

| Artifact | Expected | Status |
|---|---|---|
| `scripts/compare_picmcc.py` | present | ✔ |
| `scripts/make_picmcc_test_fixture.py` | present | ✔ |
| `outputs/picmcc_reference_TEST_FIXTURE.npz` | present | ✔ |
| `outputs/picmcc_comparison_A.{md,json}` | present | ✔ |
| `docs/picmcc_comparison_interface.md` | present | ✔ |

---

## Missing links

**NONE.** Every expected Tier-3 artifact is present and accounted for.

## Items that are by-design absent (not missing — externally blocked)

| Item | Why absent | Unblocker |
|---|---|---|
| Real PIC-MCC reference NPZ | External Boris-pusher repo | Provide NPZ matching `docs/picmcc_comparison_interface.md` §6 schema |
| Real DTPM fluid solver callbacks | External DTPM codebase | Replace mock callbacks in `fluid_loop_mock.py` |
| Third training power level | Experiment not yet run | Add case at P_rf ∉ {500, 700} W to `regenerate_teacher_v4.py` CASES |
| Experimental validation against Mettler/Lallement | Requires downstream fluid coupling | Wire PINN into TEL or external solver |

These are documented in `docs/project_status.md` under "Externally blocked" and do not represent missing artifacts — they represent the scope boundary of the Tier-3 PINN project.
