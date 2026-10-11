![Medical Harness — a unified evaluation harness for healthcare AI agents](assets/Gemini_Generated_Image_3sd7e23sd7e23sd7.png)

# Awesome Agent Harness for Healthcare

A **unified, multimodal evaluation harness** for medical AI agents. Instead of scoring whether a
model *answers* a medical question, it puts an agent inside a real working environment (an EHR API,
an admin web portal, a multimodal tool sandbox), lets it complete multi-step tasks, and scores the
run on two parallel lines:

- **Outcome** — did the agent get the dataset's *native* task right (a single, separate line).
- **7 healthcare harness dimensions** — CAF, CRC, PEA, CWC, CA, CLV, and CCB —
  *how* it worked, scored from process evidence. Their definitions are listed below.

Outcome never occupies any of the 7; the 7 never absorb clinical correctness. Both are always reported.

Building on the ETCLOVG taxonomy of Li et al., *Agent Harness Engineering: A Survey*, we specialize
its seven layers for healthcare workflows. Four Efficiency dimensions characterize action completion,
resource use, evidence alignment, and workflow continuity; three Safety dimensions characterize
auditability, verification, and clinical conduct. Native Outcome is reported separately.

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
          Outcome                                    7 healthcare dimensions
          native task correctness                    Efficiency  ·  Safety
          (separate line)                            CAF·CRC·PEA·CWC · CA·CLV·CCB
```

**Two convergence points** keep the loop benchmark-agnostic — execution differences live only in the
`EnvironmentAdapter` (`obs = env.step(action)`), evaluation differences only in the `checkpoint.method`
dispatch. **Three roles** are recorded separately in `provenance`, never collapsed even when one model
fills two: **brain** `agent_model` (emits intent only) · **hands** `tool_backend_model` (performs the
real execution: FHIR HTTP / Playwright DOM / the VLM *inside* the image tools) · **judge** `judge_model`
(scores process + outcome). Judge **independence** is enforced against BOTH the brain and the tool
backend — a judge sharing either is `judge_not_independent` (fail-closed, never a silent pass).

## Metrics — the 7 healthcare dimensions, in two panels

Every qualified run is scored on the same 7 dimensions in [0,1], grouped into two panels:
**Efficiency** (did it do the task well) and **Safety** (can it be trusted). Reported per benchmark.

| Panel | Code | Dimension | Operational focus |
|---|---|---|---|
| **Efficiency** | CAF | Care Action Fulfillment | Completion of required care actions to their intended state. |
| | CRC | Clinical Resource Competence | Authorized, semantically valid, and productive use of clinical resources. |
| | PEA | Patient–Evidence Alignment | Acquisition of required evidence and binding to the intended clinical subject. |
| | CWC | Clinical Workflow Continuity | Ordered workflow progression, recovery from failed obligations, and appropriate termination. |
| **Safety** | CA | Clinical Auditability | Reconstructability of execution state, failures, recovery, and reliability. |
| | CLV | Closed-Loop Verification | Confirmation of decision-critical preconditions and post-action state changes. |
| | CCB | Clinical Conduct Boundaries | Compliance with protected patient scope, authorization, and clinical safety constraints. |

`dimension_scores`, checkpoint `dimension` tags, and report dimension keys use the full names in this
table. Codes are display abbreviations. The scorer module filenames and subdimension identifiers
retain their implementation names; renaming dimensions does not change scoring rules or weights.

> **Outcome** — the dataset-native task correctness — is a SEPARATE line, not one of the 7.

## Models & multi-key support

The brain is any **OpenAI-compatible chat model** behind a gateway. The protocol a model is driven with,
native function calling or the `<tool_call>`/`<answer>` text protocol, is a property of the model and is listed
in `runner/models.yaml`; each benchmark's H0 workspace (`workspaces/<bench>/h0/`) holds one agent program per
protocol, and the shell's transport (`runner/transport.py`) carries the calls.

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

## Trajectories — what one run leaves behind

Every task run writes one self-contained bundle. `examples/` holds one per benchmark (a train-split task the
H0 harness solved with gpt-5.5), so the format can be read without running anything:

```
examples/<bench>/<task_id>/
  task.json                         the unified task (spec/task.schema.json): goal, context, tools, policy, checkpoints
  trajectory.jsonl                  one event per line (spec/trajectory.schema.json)
  result.json                       the scored run (spec/result.schema.json)
  verifier_logs/checkpoints_full.json   every checkpoint with its method, verdict and evidence
```

**`trajectory.jsonl`** records each step twice, once as the agent saw it and once in canonical form, so a
run can be replayed and scored without the model:

| field | what it holds |
|---|---|
| `step`, `event_type`, `origin`, `audience` | position; `tool_call` · `harness_decision` · `reconciliation` · `answer_attempt` · `final_answer`; who produced it and who saw it (agent or harness) |
| `args`, `canonical_action` | the call as emitted, and normalised to the benchmark's action vocabulary |
| `observation`, `canonical_observation`, `agent_visible_text` | what the environment returned, normalised, and the exact text delivered to the agent |
| `result`, `canonical_result`, `status` | the call's outcome |
| `state_record` | hash of the environment state before and after the step, and whether it changed (a write vs a read) |
| `delivery_record` | whether the observation was produced, rendered to the agent and consumed; source vs rendered hash; truncation |

**`result.json`** carries the native Outcome (`checkpoints` of the Outcome dimension, `success`), the 7
Healthcare dimension scores (`dimension_scores`, with coverage and per-dimension status), `provenance` (brain / hands /
judge models, never collapsed), `budget`, `tool_calls` and `failure_tags`. `result.rescored.json`, when
present, is the same record after the post-hoc judge pass.

## Layout

```
runner/                # the harness — env adapters · agent loop · transport/gateway · scorers · judges · tests
spec/                  # JSON schemas (task · checkpoint · tool · trajectory · governance · result · …)
workspaces/<bench>/h0/ # the H0 harness of each benchmark: agent programs, middleware, policies, prompts
benchmark_dataprocess/ # per-benchmark converters, the unified tasks_unified.jsonl, train/val/test splits
benchmark/             # where the upstream benchmarks are checked out (see benchmark/README.md)
examples/              # one complete run bundle per benchmark
TASK_MANIFEST.json     # task universe + pinned upstream revisions
```

> Per-run outputs are regenerable and git-ignored.

## Running

```bash
# single task (API brain via the gateway; needs OPENAI_API_KEY)
python runner/run.py --bench MedCTA --task MCTA-0 --agent gpt5

# batch (per-task result bundles + summary.json), then post-hoc judge + pure-read report
python runner/run_batch.py --bench PhysicianBench --agent gpt5 --limit 10 --fhir-base $FHIR --out results/
python runner/scorers/rescore_judges.py results/gpt5 --judge-model <judge>   # writes result.rescored.json
python runner/scorers/aggregate_report.py results/gpt5               # pure-read report.json
```

Key env vars: `MH_API_MODEL` (agent model) + `OPENAI_API_KEY` + `MH_OPENAI_BASE` (gateway),
`MH_JUDGE_MODEL` / `MH_JUDGE_KEY` (judge), `MH_VLM_API_MODEL` / `MH_VLM_API_KEY` (image perception),
`MH_GUI_MODE` (`real`|`mock` for HealthAdminBench), `MH_GATEWAY_TIMEOUT` / `MH_GATEWAY_RETRIES`.

## Extensions: datasets, harness components, and custom functions

The harness supports Python benchmark plugins and workspace components:

| Extension | Integration point |
|---|---|
| New dataset | Add `runner/plugins/<name>.py`, register its `PLUGIN` dictionary with `substrate.register_plugin()`, provide unified tasks, and add the dataset to `spec/registry.json`. Plugin modules are discovered automatically. |
| Custom dataset functions | Supply callable semantic resolvers, evidence extractors, and task-contract resolvers in the plugin. These functions are called by the semantic and scoring pipeline. |
| Host middleware | Add a Python class to the workspace and list `<file>:<Class>` under `middleware` in `manifest.yaml`. Hooks can provide observations, request revisions, or inspect tool execution. |
| Kernel capability | Subclass `workspace_api.Capability` and register it under `kernel_middleware`. Hooks can return harness decisions before/after an action or before the final answer. |
| Custom helper functions | Define functions in a component module, or load workspace-local helpers through `workspace_api.load_relative()`, and call them from a registered hook. |

A new dataset can reuse an existing environment type (`fhir`, `gui`, or `tool_sandbox`) with a matching
workspace. A new environment type also needs an environment implementation and registration in
`runner/environments.py`; its tool semantics and safety adapter must match that environment. Defining a
Python function in a benchmark plugin does not by itself expose a new executable tool to the agent.

See [the extension guide](runner/README.md#extension-guide) for concrete plugin and component examples,
hook signatures, and workspace import constraints.

## References

- Li et al. *Agent Harness Engineering: A Survey.* — source of the ETCLOVG taxonomy specialized into the 7 healthcare dimensions.
- PhysicianBench — `github.com/HealthRex/PhysicianBench`
- HealthAdminBench — `github.com/som-shahlab/health-admin-bench`
- MedCTA — HuggingFace `IVUL-KAUST/MedCTA`
