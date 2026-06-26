# Awesome Agent Harness for Healthcare

A **unified, multimodal evaluation harness** for medical AI agents. Instead of scoring whether a
model *answers* a medical question, it puts an agent inside a real working environment (an EHR API,
an admin web portal, a multimodal tool sandbox), lets it complete multi-step tasks, and scores the
run on two parallel lines:

- **Outcome** — did the agent get the dataset's *native* task right (a single, separate line).
- **7 ETCLOVG harness dimensions** — **E**xecution, **T**ooling, **C**ontext, **L**ifecycle,
  **O**bservability, **V**erification, **G**overnance — *how* it worked, scored from process evidence.

Outcome never occupies any of the 7; the 7 never absorb clinical correctness. Both are always reported.

## Benchmarks

| Benchmark | Modality | Scenario |
|---|---|---|
| **PhysicianBench** | structured clinical | real FHIR EHR (HAPI R4, de-identified) via API tools |
| **HealthAdminBench** | GUI admin | real Next.js portal (prior-auth / appeals) driven by Playwright |
| **MedCTA** | medical images (CT / X-ray / path / …) | multimodal tool sandbox (image description / region / OCR / search / calc) |

## Architecture — one loop, two convergence points, three roles

```
                 unified task   (one schema · PhysicianBench · HealthAdminBench · MedCTA)
                                       │
         ┌─────────────────────────────▼─────────────────────────────┐
         │                  RUNNER — one main loop                    │
         │               obs = env.step(action)   (×N steps)          │
         └──────┬──────────────────────────────────────────▲─────────┘
                │ action intent                  observation │
   BRAIN  agent_model  (emits <tool_call>/<answer> only)     │
                ▼                                             │
   ┌─── ①  EnvironmentAdapter  · one class per substrate ─────┴───────┐
   │    FhirEnv             GuiEnvReal             ToolSandboxEnv      │
   │    real HAPI FHIR      real Playwright DOM    VLM image tools     │
   └──────────────── HANDS  tool_backend_model  · real execution ─────┘
                                       │
                  trajectory.jsonl   (action + observation + state, per step)
                                       │
   ┌─── ②  checkpoint.method  · scorer dispatch ──────────────────────┐
   │    native_pytest · jmespath · policy · deterministic · llm_judge  │
   │                       JUDGE  judge_model                          │
   └───────────────────────────────────┬──────────────────────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                              ▼
          Outcome                                    7 ETCLOVG dimensions
          native task correctness                    Efficiency  ·  Safety
          (separate line)                            E·T·C·L     ·  O·V·G
```

**Two convergence points** keep the loop benchmark-agnostic — execution differences live only in the
`EnvironmentAdapter` (`obs = env.step(action)`), evaluation differences only in the `checkpoint.method`
dispatch. **Three roles** are recorded separately in `provenance`, never collapsed even when one model
fills two: **brain** `agent_model` (emits intent only) · **hands** `tool_backend_model` (performs the
real execution: FHIR HTTP / Playwright DOM / the VLM *inside* the image tools) · **judge** `judge_model`
(scores process + outcome). Judge **independence** is enforced against BOTH the brain and the tool
backend — a judge sharing either is `judge_not_independent` (fail-closed, never a silent pass).

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

## Models & multi-key support

The brain is any **OpenAI-compatible chat model** behind a gateway. The agent layer is vendor-neutral:
the shared text tool-call scaffolding lives in `ToolProtocolAgent` (`runner/tool_agent.py`); the API
brain `ApiToolAgent` (`runner/api_agent.py`) subclasses it and only swaps the transport.

Each of the three model **roles** takes its own model + key env, so one run can put the agent, the
image-perception VLM, and the judge on **different providers/keys** — useful when one key serves the
agent model and another serves the judge:

| Role | Model env | Key env |
|---|---|---|
| agent brain | `MH_API_MODEL` | `OPENAI_API_KEY` (global default) |
| VLM perception (image tools) | `MH_VLM_API_MODEL` | `MH_VLM_API_KEY` |
| judge (gacc / mm / governance / context / verification) | `MH_JUDGE_MODEL` | `MH_JUDGE_KEY` |

`runner/gateway.py` resolves the key per call: explicit `override` > `MH_JUDGE_KEY` (judge calls) >
`MH_OPENAI_KEY` / `OPENAI_API_KEY`. Set just `OPENAI_API_KEY` for a single-key run, or add
`MH_JUDGE_KEY` / `MH_VLM_API_KEY` to split roles across keys.

## Layout

```
runner/                # the harness — env adapters · agents · gateway · scoring · judge · report · tests
scripts/               # run launchers (one per model × dataset)
spec/                  # frozen JSON schemas (task · checkpoint · tool · trajectory · governance · result)
benchmark_dataprocess/ # per-benchmark converters + the unified tasks_unified.jsonl assets
benchmark_metric/      # efficiency · safety · integrity-meta reporting panels
docs/                  # architecture · contract · processing notes
TASK_MANIFEST.json     # task universe + pinned upstream revisions
```

> Per-run outputs (`res_*` / `results_*`) are regenerable and git-ignored.

## Running

```bash
# single task (API brain via the gateway; needs OPENAI_API_KEY)
python runner/run.py --bench MedCTA --task MCTA-0 --agent gpt5

# batch (per-task result bundles + summary.json), then post-hoc judge + pure-read report
python runner/run_batch.py --bench PhysicianBench --agent gpt5 --limit 10 --fhir-base $FHIR --out results/
python runner/rescore_judges.py results/gpt5 --judge-model <judge>   # writes result.rescored.json
python runner/aggregate_report.py results/gpt5                       # pure-read report.json
```

Key env vars: `MH_API_MODEL` (agent model) + `OPENAI_API_KEY` + `MH_OPENAI_BASE` (gateway),
`MH_JUDGE_MODEL` / `MH_JUDGE_KEY` (judge), `MH_VLM_API_MODEL` / `MH_VLM_API_KEY` (image perception),
`MH_GUI_MODE` (`real`|`mock` for HealthAdminBench), `MH_GATEWAY_TIMEOUT` / `MH_GATEWAY_RETRIES`.