# Benchmark

Place upstream benchmark code and runtime assets here. These assets are git-ignored; the repository includes directory placeholders.

## Layout

```text
benchmark/
  PhysicianBench/
    PhysicianBench/  # upstream Python repository
    fhir-full.sif    # FHIR server image
  HealthAdminBench/
    harness/         # upstream portals and tasks
  MedCTA/
    opencompass/     # dataset and extracted images
```

## Sources

| Benchmark | Upstream | Pinned revision |
|---|---|---|
| **PhysicianBench** | `github.com/HealthRex/PhysicianBench` (tasks/v1) | `48c135b2a64177a07bcd08d67d0cc28b9d7ed946` |
| **HealthAdminBench** | `github.com/som-shahlab/health-admin-bench` (v3 tasks) | `e71a8f4d6923037805b7f51fbbf608d12ea56cf5` |
| **MedCTA** | HuggingFace `IVUL-KAUST/MedCTA` | parquet sha256 `2dfe190e…0e779b`, image manifest sha256 `687af5f2…3281f` |

Revisions and checksums are recorded in [TASK_MANIFEST.json](../TASK_MANIFEST.json). See [PhysicianBench setup](../benchmark_dataprocess/PhysicianBench/README.md) for the FHIR server.
