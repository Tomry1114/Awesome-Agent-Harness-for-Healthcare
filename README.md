# Awesome Agent Harness for Healthcare

A **unified, multimodal evaluation harness** for medical AI agents. Instead of scoring whether a
model *answers* a medical question, it puts an agent inside a real working environment (an EHR API,
an admin web portal, a multimodal tool sandbox), lets it complete multi-step tasks, and scores the
run on two parallel lines:

- **Outcome** — did the agent get the dataset's *native* task right (a single, separate line).
- **7 ETCLOVG harness dimensions** — **E**xecution, **T**ooling, **C**ontext, **L**ifecycle,
  **O**bservability, **V**erification, **G**overnance — *how* it worked, scored from process evidence.

Outcome never occupies any of the 7; the 7 never absorb clinical correctness. Both are always reported.

## Portfolio — 3 non-overlapping agentic benchmarks → one schema

| Benchmark | Environment | Modality | Stresses |
|---|---|---|---|
| **PhysicianBench** | FHIR EHR (HAPI R4, real de-identified data) via API tools | structured clinical | Execution / Tooling / Context / Verification / Governance |
| **HealthAdminBench** | **Real Next.js admin portal** driven by Playwright (real DOM, real localStorage state) | GUI admin (prior-auth / appeals) | Lifecycle / Governance(compliance) / Verification |
| **MedCTA** | Multimodal tool sandbox (image description / region / OCR / search / calc) | medical images (CT/X-ray/path/…) | Tooling / Context(image) / Observability / Verification |

All three convert into one **unified task schema** and run through one runner that emits one **result
schema** carrying the 7-dimension scores + the Outcome line. `TASK_MANIFEST.json` pins the task
universe and upstream revisions.

## Design: one loop, two convergence points, three roles

**One runner main loop**, agnostic to benchmark. Differences converge at exactly two places:

- **Execution differences → `EnvironmentAdapter`** (one class per substrate). The loop only ever does
  `obs = env.step(action)`; how an action becomes a FHIR HTTP call, a Playwright DOM event, or a VLM
  tool call is the adapter's concern.
- **Evaluation differences → `checkpoint.method` scorer dispatch** (`native_pytest` / `jmespath` /
  `llm_judge` / `policy` / `deterministic`). The loop never hard-codes how a checkpoint is judged.

**Three roles, recorded separately in `provenance`** (never collapsed, even when one model fills two):

| Role | Field | Who |
|---|---|---|
| **brain** | `agent_model` | the agent — emits *action intent* only (`<tool_call>{…}</tool_call>` / `<answer>…</answer>`) |
| **hands** | `tool_backend_model` | the environment / tool backend — performs real execution (FHIR HTTP · Playwright DOM · the VLM *inside* the image tools) |
| **judge** | `judge_model` | the verifier that scores process / outcome |

Judge **independence** is enforced against BOTH the agent brain and the tool backend: a judge sharing
either model is `judge_not_independent` (fail-closed, never a silent pass).

## Models & multi-key support

The default brain is an **API model via the micuapi gateway** (`gpt-5.5` strong / `gpt-5.4-mini` weak,
for sensitivity studies); a fully local **Qwen3-VL** brain is still available (`--agent qwen`, no API
key). The agent layer is vendor-neutral: the shared text tool-call scaffolding lives in
`ToolProtocolAgent` (`runner/tool_agent.py`); the API brain `ApiToolAgent` (`runner/api_agent.py`)
subclasses it and only swaps the transport.

A run can use **different API keys per role** (e.g. a gemini/deepseek-only key for the agent brain,
a gpt key for the VLM perception and the judge), all in one process:

| Role | Key env | Model env |
|---|---|---|
| agent brain | `OPENAI_API_KEY` (global) | `MH_API_MODEL` |
| VLM perception (MedCTA image tools) | `MH_VLM_API_KEY` | `MH_VLM_API_MODEL` |
| judge (gacc / mm / governance / context / verification) | `MH_JUDGE_KEY` | `MH_JUDGE_MODEL` etc. |

`runner/gateway.py` resolves the key per call: `override > MH_JUDGE_KEY (judge) > MH_OPENAI_KEY /
OPENAI_API_KEY > ~/.xbai_key`.

## Phased scoring — judge calls are isolated, the report is pure-read

| Stage | File | Model calls? |
|---|---|---|
| 1. Run the agent | `run.py` / `run_batch.py` | agent + tool backend (+ in-run judges) → `result.json` + `trajectory.jsonl` |
| 2. Post-hoc judge | `rescore_judges.py` | **the ONLY judge caller for Governance.** Caches per `(task, output, judge, prompt)`; writes `result.rescored.json` (top-level `Governance` + audit) |
| 3. Aggregate | `aggregate_report.py` | **PURE-READ — zero model calls.** Reads the persisted blocks → `report.json` |

`governance_contract.py` is the single source of truth for the governance blend / critical veto /
`scoring_config`. The aggregate holds no scoring math — there is exactly one blend implementation.

## Evidence tiers & honesty mechanisms

- **`substrate_universal`** (deterministic: Execution / Tooling / Lifecycle / Observability) →
  `formal_analysis_eligible` and enters formal stats **only when adapter admission is `ok`** (a
  LOW_COVERAGE dimension is `strict_by_definition` but NOT formally admitted).
- **`experimental_hybrid`** (judge-backed: Context / Verification / Governance) → reportable but
  `formal_analysis_eligible=false`; never counted as strict.
- **Outcome** = dataset-native task success via the single function `native_task_outcome()`
  (PB/HAB: *all* Outcome-dimension checkpoints pass; MedCTA: mean GAcc≥0.5). It **never** reads
  `r["success"]` (the harness all-checkpoints gate, reported separately as `harness_gate`). A task
  with an unresolved/missing Outcome is excluded from the denominator → `native_evaluation_coverage`.
- **`evaluation_status`** — `complete` / `partial` / `proxy_partial` / `proxy_only` / `not_evaluated`
  / `error`, so a run is never silently counted as fully scored. Unparseable judge verdict →
  `verifier_error`, never a silent pass.
- **`_qualification`** downgrades only for `mock_env` / `replay_tool_backend` / `outcome_proxy` /
  `uses_hidden_reference` / `scorer_validation_only` — not by which substrate ran. A real Playwright
  GUI run and a real VLM tool run are first-class.

## Metrics — the 7 ETCLOVG dimensions, in two panels

Every qualified run is scored on the same 7 dimensions in [0,1], grouped into two panels:
**Efficiency** (did it do the task well) and **Safety** (can it be trusted). Reported per benchmark.

| Panel | Dimension | Meaning |
|---|---|---|
| **Efficiency** | **E**xecution | completed the task steps and reached the goal state |
| | **T**ooling | right tool choice, valid arguments, successful calls |
| | **C**ontext | grounded in the real patient / image evidence (no fabricated facts) |
| | **L**ifecycle | followed the required multi-step workflow through to completion |
| **Safety** | **O**bservability | left a complete, inspectable trace (actions + observations + state) |
| | **V**erification | checked its own work / confirmed results before committing |
| | **G**overnance | policy & safety compliance (patient scope, prechecks, no forbidden / unsafe actions) |

> **Outcome** — the dataset-native task correctness — is a SEPARATE line, not one of the 7.

## Provenance integrity — `current` means *verified*, not *unverified*

Every rescored bundle records hashes so the report can prove it is consistent with the code AND the
inputs that produced it. `aggregate_report` recomputes them live and emits **three orthogonal axes +
an overall rollup**:

| Axis | Verifies |
|---|---|
| `scoring_code_status` | `git rev-parse HEAD:runner` (scoring-code TREE hash) == recorded; uniform judge/prompt; clean worktree |
| `task_asset_status` | `tasks_unified.jsonl` (the dimension/weight map `aggregate` re-reads) unchanged |
| `source_bundle_status` | `result.json` / `trajectory.jsonl` / `task.json` / the rescored checkpoint set (incl status+score) / the judged **deliverable files** unchanged |
| `overall_artifact_status` | `current` **only if all three are explicitly current** (absence of provenance = `incomplete_provenance`, never `current`) |

The TREE hash (not full-repo HEAD) is deliberate: committing artifacts/docs advances HEAD but leaves
`runner/` unchanged, so an artifact stays `current` — no self-reference loop. Tamper tests
(editing a trajectory / tasks_unified / a deliverable / a rescored checkpoint) flip the matching axis
to stale; `runner/test_conformance.py` (109 checks) guards all of this.

## Layout

```
runner/                 # unified harness
  run.py / run_batch.py #   single-task + batch CLI (env adapter → agent loop → trajectory → scorer → result)
  environments.py       #   FhirEnv (real HAPI) · GuiEnvReal (real Playwright portal) + GuiEnvMock · ToolSandboxEnv
  tool_agent.py         #   ToolProtocolAgent — shared text <tool_call>/<answer> protocol (model-agnostic)
  api_agent.py          #   ApiToolAgent — API brain over the gateway (subclasses ToolProtocolAgent)
  agents.py             #   make_agent() registry: gpt5/openai → ApiToolAgent · qwen → local Qwen3-VL · stub/replay/scripted
  gateway.py            #   unified OpenAI-compatible client; per-role key resolution; bounded retry/deadline
  vlm_backend.py        #   VLM perception (api gpt-5.x default · local Qwen3-VL); own key via MH_VLM_API_KEY
  tools_medcta.py       #   MedCTA tool backend: ImageDescription / RegionAttributeDescription / OCR / Calculator / GoogleSearch(frozen)
  scoring.py            #   checkpoint dispatch · subject-scope state machine · weighted 7-module aggregation
  governance_contract.py#   SINGLE source: governance blend · critical veto · scoring_config · checkpoint_set hash
  rescore_judges.py     #   the ONLY Governance judge caller; writes result.rescored.json (+ input_provenance)
  aggregate_report.py   #   PURE-READ report builder; native_task_outcome; provenance audit; model comparison
  test_conformance.py   #   109 conformance checks (metric integrity, provenance, tamper detection)
scripts/                # run launchers (run_*.sh / run2_*.sh) for each model × dataset
spec/                   # frozen JSON schemas: task / checkpoint / tool / trajectory / governance / result
benchmark_dataprocess/  # per-benchmark converters, augmentations, validators, and unified outputs
  <Bench>/tasks_unified.jsonl   # the converted, scored benchmark assets
benchmark_metric/       # action-level safety + efficiency + integrity-meta reporting panels
docs/                   # ARCHITECTURE · CANONICAL_CONTRACT · CAPABILITY_MATRIX · task spec · processing notes
TASK_MANIFEST.json      # task universe + pinned upstream revisions + checksums
```

> Per-run experiment outputs (`res_*` / `res2_*` / `res3_*` / `results_*`) are **regenerable** and
> git-ignored — rebuild them with the `scripts/` launchers + `rescore_judges.py`.

## Running

```bash
# single task (API brain via gateway; needs OPENAI_API_KEY for the gateway)
python runner/run.py --bench MedCTA --task MCTA-0 --agent gpt5

# fully local Qwen3-VL brain (no API key)
python runner/run.py --bench MedCTA --task MCTA-0 --agent qwen

# batch (per-task result bundles + summary.json), then post-hoc judge + pure-read report
python runner/run_batch.py --bench PhysicianBench --agent gpt5 --limit 10 --fhir-base $FHIR --out results/
python runner/rescore_judges.py results/gpt5 --judge-model gpt-5.4   # writes result.rescored.json
python runner/aggregate_report.py results/gpt5                       # pure-read report.json
```

Key env vars: `MH_API_MODEL` (agent model), `OPENAI_API_KEY` + `MH_OPENAI_BASE` (gateway),
`MH_JUDGE_MODEL` / `MH_JUDGE_KEY` (judge), `MH_VLM_API_MODEL` / `MH_VLM_API_KEY` (MedCTA perception),
`MH_GUI_MODE` (`real`|`mock` for HealthAdminBench), `MH_VLM_PATH` (local Qwen3-VL dir),
`MH_GATEWAY_TIMEOUT` / `MH_GATEWAY_RETRIES`.

## Not in this repo (re-fetch separately)

`benchmark/` holds vendored upstream repos and multi-GB container images / databases (FHIR `.sif`,
H2 DB, OCI layers, parquet, images). They are git-ignored; restore them from the upstream revisions
pinned in `TASK_MANIFEST.json` and the deployment notes in `docs/`.

## Status

All scorer methods are wired; the three substrates are real (live HAPI FHIR · real Playwright portal ·
real VLM tool sandbox); the phased judge + pure-read report + provenance integrity are in place;
conformance is 109/109.
