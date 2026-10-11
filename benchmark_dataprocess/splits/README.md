# Fixed dataset split — train / val / test over all 342 tasks

| file | tasks | use |
|---|---|---|
| `held_in_train.txt` | 168 | development: failures may be inspected freely |
| `held_in_val.txt` | 68 | model / harness selection |
| `held_out_test.txt` | 106 | held out for the final reported numbers only |

Deterministic and stratified 5:2:3 — PhysicianBench by clinical phase, HealthAdminBench by category × difficulty,
MedCTA by modality. Regenerate identically with `python benchmark_dataprocess/splits/make_split.py`.
