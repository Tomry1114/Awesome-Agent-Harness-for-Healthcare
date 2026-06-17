# Awesome Agent Harness for Healthcare

A **unified, multimodal evaluation harness** for medical AI agents. Instead of
scoring whether a model *answers* a medical question, it puts an agent inside a
real working environment (an EHR API, an admin web portal, a multimodal tool
sandbox), lets it complete multi-step tasks, and scores the run across **7
ETCLOVG dimensions**: **E**xecution, **T**ooling, **C**ontext, **L**ifecycle,
**O**bservability, **V**erification, **G**overnance.

## Portfolio — 3 non-overlapping agentic benchmarks → one schema

| Benchmark | Environment | Modality | Stresses |
|---|---|---|---|
| **PhysicianBench** | FHIR EHR (HAPI R4, real de-identified data) via API tools | structured clinical | Execution / Tooling / Context / Verification / Governance |
| **HealthAdminBench** | Web portal (browser actions) | GUI admin (prior-auth / appeals) | Lifecycle / Governance(compliance) / Verification |
| **MedCTA** | Multimodal tool sandbox (OCR, image, search, calc) | medical images (CT/X-ray/path/…) | Tooling / Context(image) / Observability / Verification |

All three are converted into one **unified task schema** (342 tasks total) and run
through one runner that emits one **result schema** with the 7-dimension scores.

## Layout

```
runner/                 # unified harness: load task → env adapter → agent loop → trajectory → scorer → result
  run.py                #   single-task CLI
  run_batch.py          #   batch + filters + per-task result bundles + summary.json
  environments.py       #   FhirEnv (real) / GuiEnv (mock portal) / ToolSandboxEnv (MedCTA replay)
  agents.py             #   StubAgent / StubGuiAgent / ReplayAgent (regression agents, NOT baselines)
  scoring.py            #   checkpoint dispatch → weighted 7-module aggregation → result
  native_pytest.py      #   runs PhysicianBench upstream pytest checkpoints
spec/                   # 6 frozen JSON schemas: task / checkpoint / tool / trajectory / governance / result
benchmark_dataprocess/  # per-benchmark converters, augmentations, validators, and unified outputs
  <Bench>/tasks_unified.jsonl   # the converted benchmark assets
  PhysicianBench/augmentation/  # allergy/RxNorm/drug-safety + encounter index (Governance/Lifecycle)
docs/                   # STATUS.md (single source of progress), task spec, validation reports
TASK_MANIFEST.json      # 342 tasks + pinned upstream revisions + checksums
```

## Scoring tiers

- **strict verifier** → counts toward `success` + `dimension_scores` (formal benchmark score)
- **proxy verifier** (e.g. offline whitelist for replay/smoke-test) → `score_eligible=false`,
  reported only in `proxy_dimension_scores` / `proxy_evaluated_checkpoints`, **never** in formal score
- **skipped** (missing judge/verifier backend) → excluded from both

## Not in this repo (re-fetch separately)

`benchmark/` holds vendored upstream repos and multi-GB container images / databases
(FHIR `.sif`, H2 DB, OCI layers, parquet, images). They are git-ignored; restore them
from the upstream revisions pinned in `TASK_MANIFEST.json` and the deployment notes in
`docs/`.

## Status

See `docs/STATUS.md`. The harness runs end-to-end with deterministic regression
agents (no API key) across all three environments; plugging in a real LLM agent +
LLM judge is the remaining step to produce formal benchmark scores.
