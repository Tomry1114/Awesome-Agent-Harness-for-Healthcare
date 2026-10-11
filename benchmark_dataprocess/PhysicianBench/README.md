# PhysicianBench

Raw upstream assets (read-only) live in `../../benchmark/PhysicianBench/` (task definitions, the FHIR server
image). This directory turns them into the unified task format and adds the patient data the extra checkpoints need.

## FHIR environment

- `run_fhir.sh [PORT]` starts the HAPI FHIR R4 server (default port 38080, `MH_FHIR_PORT` overrides).
- The server writes to a local H2 database (`h2data/`, git-ignored), bound into the container at run time; every
  data injection goes there, never into the upstream image.
- `augmentation/restore_pristine_h2.sh` stops the server, restores the pristine database and restarts it. Do not
  copy the H2 files while the server is running.

## Files

| file | content |
|---|---|
| `convert.py` | upstream tasks + checkpoints → `tasks_unified.jsonl` (100 tasks, `spec/task.schema.json`) |
| `ref_ranges.json`, `lab_ref.py`, `build_ref_ranges.py` | lab reference ranges (49 entries over 43 LOINC codes) with unit checks; `classify` flags abnormal values. Read by the FHIR environment and the governance scorer |
| `augmentation/synthetic_allergies.json`, `allergy_bundle.json`, `rxnorm_mapping.json` | deterministic synthetic allergies for the 26 medication tasks, written to FHIR as `AllergyIntolerance` (idempotent PUT, tagged so they can be rolled back), with RxNorm mappings |
| `augmentation/drug_safety_rules.json`, `drug_safety_check.py` | medication-safety verifiers, e.g. `no_allergy_conflicting_medication_recommended` on the final answer and on written deliverables (matched in a recommendation context, so reporting an allergy is not a violation; covers the mapped drug names only) |
| `augmentation/governance_checkpoints.json`, `merge_governance.py`, `build_augmentation.py` | the governance checkpoints added to the 26 medication tasks (six each, 156 in total), merged into `tasks_unified.jsonl`; re-runnable |
| `augmentation/encounter_index.json`, `build_encounter_index.py` | an external index of 10,980 encounters for the 100 patients, grouped by day; not written to FHIR and not read by the runner |

## Environment

environment `fhir` · modality `structured_fhir`. The agent works through the upstream FHIR tools; Outcome comes from
the native checkpoints, the 7 healthcare dimensions from the process evidence.
