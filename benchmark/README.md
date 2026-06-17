# `benchmark/` — vendored raw benchmarks (not tracked)

This directory holds the **upstream benchmark repositories and their large raw
assets** (FHIR container images, H2 databases, OCI layers, parquet, images).
They are multi-GB and third-party, so the contents are **git-ignored**; only this
README and empty `.gitkeep` placeholders are committed to preserve the structure.

The harness code (`runner/`, `benchmark_dataprocess/`) expects these populated at
runtime. Restore them from the pinned upstream revisions below (also in
`/TASK_MANIFEST.json`).

## Expected layout

```
benchmark/
├── PhysicianBench/      # HealthRex/PhysicianBench upstream + fhir-full.sif + OCI image
├── HealthAdminBench/    # som-shahlab/health-admin-bench upstream (v3 tasks)
└── MedCTA/              # IVUL-KAUST/MedCTA (parquet + images)
```

## Restore

| Benchmark | Upstream | Pinned revision |
|---|---|---|
| **PhysicianBench** | `github.com/HealthRex/PhysicianBench` (tasks/v1) | `48c135b2a64177a07bcd08d67d0cc28b9d7ed946` |
| **HealthAdminBench** | `github.com/som-shahlab/health-admin-bench` (v3 tasks) | `e71a8f4d6923037805b7f51fbbf608d12ea56cf5` |
| **MedCTA** | HuggingFace `IVUL-KAUST/MedCTA` (via hf-mirror.com) | parquet sha256 `2dfe190e…0e779b`, image manifest sha256 `687af5f2…3281f` |

The PhysicianBench FHIR environment image (`fhir-full:v1`, HAPI FHIR 8.8.0) is
built/run separately; see `docs/` and `benchmark_dataprocess/PhysicianBench/`.
Checksums for verifying a restored copy are in `/TASK_MANIFEST.json`.
