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
user_study/
  MedR-Bench/       # pinned source repository
  Meissa-SFT/      # training trajectories needed for overlap screening
  Meissa-Evaluation/ # PathVQA source parquet and extracted images
  MedCUA-BenchMark/ # pinned source repository and native checker
  runtime/medcua_env/ # isolated browser runtime
```

## Sources

| Benchmark | Upstream | Pinned revision |
|---|---|---|
| **PhysicianBench** | `github.com/HealthRex/PhysicianBench` (tasks/v1) | `48c135b2a64177a07bcd08d67d0cc28b9d7ed946` |
| **HealthAdminBench** | `github.com/som-shahlab/health-admin-bench` (v3 tasks) | `e71a8f4d6923037805b7f51fbbf608d12ea56cf5` |
| **MedCTA** | HuggingFace `IVUL-KAUST/MedCTA` | parquet sha256 `2dfe190e…0e779b`, image manifest sha256 `687af5f2…3281f` |
| **MedRBench** | [MAGIC-AI4Med/MedRBench](https://github.com/MAGIC-AI4Med/MedRBench) | `ff60ab440afd2f2bc0c603b3a65d715ec83138a7` |
| **MeissaEval** | [Schuture/Meissa](https://github.com/Schuture/Meissa), [PathVQA](https://huggingface.co/datasets/flaviagiammarino/path-vqa) | Meissa `eb119115d2f5dfcfae772ba5f2466b4c603fac90`; PathVQA revision in `user_study/Meissa-Evaluation/source/download_manifest.json` |
| **MedCUABench** | [Jia7878/MedCUA-BenchMark](https://github.com/Jia7878/MedCUA-BenchMark) | `159c7681cbc84d446ceafa8046bd498036809b70`; tracked-file hashes in `user_study/config/medcua_source.json` |

The original three benchmarks are recorded in [TASK_MANIFEST.json](../TASK_MANIFEST.json). The user-study sources have separate pins and conversion reports; see [user-study setup](../user_study/README.md). See [PhysicianBench setup](../benchmark_dataprocess/PhysicianBench/README.md) for the FHIR server.
