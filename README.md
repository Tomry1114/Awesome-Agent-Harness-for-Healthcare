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
| **HealthAdminBench** | Real Next.js admin portal driven by Playwright (real DOM) | GUI admin (prior-auth / appeals) | Lifecycle / Governance(compliance) / Verification |
| **MedCTA** | Multimodal tool sandbox (image description / region / OCR / search / calc) | medical images (CT/X-ray/path/…) | Tooling / Context(image) / Observability / Verification |

All three are converted into one **unified task schema** (342 tasks total) and run
through one runner that emits one **result schema** with the 7-dimension scores.

## Design: one loop, two convergence points, three roles

**One runner main loop**, agnostic to benchmark. Differences converge at exactly two places:

- **Execution differences → `EnvironmentAdapter`** (one class per substrate). The loop only ever
  does `obs = env.step(action)`; how an action becomes a FHIR HTTP call, a Playwright DOM event,
  or a local VLM tool call is the adapter's concern.
- **Evaluation differences → `checkpoint.method` scorer dispatch** (`native_pytest` / `jmespath` /
  `llm_judge` / `policy` / `deterministic`). The loop never hard-codes how a checkpoint is judged.

**Three roles, recorded separately in `provenance`** (never collapsed, even when one model fills two):

| Role | Who | Example |
|---|---|---|
| **brain** (`agent_model`) | the agent — emits *action intent* only (`<tool_call>{…}</tool_call>` / `<answer>…</answer>`) | text-only Qwen3-VL |
| **hands** (`tool_backend_model`) | the environment / tool backend — performs real execution | FHIR HTTP · Playwright · Qwen3-VL *inside* the image tools |
| **judge** (`judge_model`) | the verifier that scores outcome | `llm_judge` / offline proxy / `none` |

> In MedCTA the brain **and** the ImageDescription tool are both Qwen3-VL. They are still recorded as
> distinct roles — the agent never "sees" the image; it only sees what the tool backend returns.

## Layout

```
runner/                 # unified harness: load task → env adapter → agent loop → trajectory → scorer → result
  run.py                #   single-task CLI (also builds provenance + qualification flags)
  run_batch.py          #   batch + filters + per-task result bundles + summary.json
  environments.py       #   FhirEnv (real HAPI) · GuiEnvReal (real Playwright portal) + GuiEnvMock · ToolSandboxEnv
  qwen_agent.py         #   QwenToolAgent — the real, environment-aware BRAIN (text-only, local, no API key)
  agents.py             #   StubAgent / ScriptedAgent / ReplayAgent (regression agents, NOT baselines)
  vlm_backend.py        #   pluggable local Qwen3-VL (singleton) — image_description / region (real crop) / ocr / chat
  tools_medcta.py       #   MedCTA tool backend: ImageDescription / RegionAttributeDescription / OCR / Calculator / GoogleSearch(frozen)
  scoring.py            #   checkpoint dispatch → weighted 7-module aggregation → result
  native_pytest.py      #   runs PhysicianBench upstream pytest checkpoints
  medcta_profile.py     #   small-batch failure-mode profiler (multi-label classifier over a real-agent run)
  run_*.sbatch          #   Slurm launchers (debug partition, gpu:1) for each substrate + the profiler
spec/                   # 6 frozen JSON schemas: task / checkpoint / tool / trajectory / governance / result
benchmark_dataprocess/  # per-benchmark converters, augmentations, validators, and unified outputs
  <Bench>/tasks_unified.jsonl   # the converted benchmark assets
  PhysicianBench/augmentation/  # allergy/RxNorm/drug-safety + encounter index (Governance/Lifecycle)
docs/                   # STATUS.md (single source of progress) · CHANGELOG.md (per-round changes) · task spec · reports
TASK_MANIFEST.json      # 342 tasks + pinned upstream revisions + checksums
```

## Scoring tiers & honesty mechanisms

- **strict verifier** → counts toward `success` + `dimension_scores` (formal benchmark score)
- **proxy verifier** (e.g. MedCTA offline-whitelist outcome) → `score_eligible=false`,
  reported only in `proxy_dimension_scores` / `proxy_evaluated_checkpoints`, **never** in formal score
- **skipped** (missing judge/verifier backend) → excluded from both
- **`evaluation_status`** — 6 states: `complete` / `partial` / `proxy_partial` / `proxy_only` /
  `not_evaluated` / `error`, so a run is never silently counted as fully scored.
- **`_qualification`** — a result is downgraded ONLY for `mock_env` / `replay_tool_backend` /
  `outcome_proxy` / `uses_hidden_reference` / `scorer_validation_only` / `proxy_scored_checkpoints`
  — **not** by which substrate it ran on. A real Playwright GUI run and a real VLM tool run are
  first-class, not "stub".

## Agents

- **`QwenToolAgent`** (`qwen`) — the real brain. Same `<tool_call>`/`<answer>` protocol on all three
  substrates; only the system prompt differs. Fully local Qwen3-VL, **no API key**. Produces real
  `success` + ToolAcc / ArgAcc (and, for MedCTA, a *proxy* outcome until a real outcome judge is wired).
- **`ReplayAgent` / `ScriptedAgent` / `StubAgent`** — regression agents that verify env wiring + scorer
  paths. Their success is `scorer_validation_only` and never enters a real-agent baseline.

## Running

```bash
# single task (login node is fine for FHIR / MedCTA; GUI defaults to mock unless MH_GUI_MODE=real)
python runner/run.py --bench MedCTA --task MCTA-0 --agent qwen

# GUI against the real portal — needs a launchable chromium (GPU node), so opt in explicitly:
MH_GUI_MODE=real MH_PORTAL_BASE=http://<host>:3002 python runner/run.py --bench HealthAdminBench --task HAB-... --agent qwen

# MedCTA failure-mode profiling (real Qwen3-VL, multi-label classifier)
sbatch runner/run_medcta_profile.sbatch
```

Key env vars: `MH_VLM_PATH` (model dir, default `~/hf_models/Qwen3-VL-2B-Instruct`),
`MH_GUI_MODE` (`real`|`mock`, default mock — login nodes lack a launchable browser),
`MH_TOOL_MODE` (`real`|`replay`), `MH_PORTAL_BASE` (portal URL for GUI).

## Not in this repo (re-fetch separately)

`benchmark/` holds vendored upstream repos and multi-GB container images / databases
(FHIR `.sif`, H2 DB, OCI layers, parquet, images). They are git-ignored; restore them
from the upstream revisions pinned in `TASK_MANIFEST.json` and the deployment notes in
`docs/`.

## Status

See `docs/STATUS.md` (progress) and `docs/CHANGELOG.md` (per-round changes). The real
Qwen3-VL agent now runs **end-to-end across all three substrates**; the current bottleneck is
model capacity (the 2B under-uses tools and misreads images) rather than the harness. Promoting
MedCTA's outcome from *proxy* to *formal* score needs a real outcome (Gacc) judge; that is the
remaining step for formal benchmark numbers.
