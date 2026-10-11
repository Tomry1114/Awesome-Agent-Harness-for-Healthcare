# Unified Harness Runner

One main loop turns any unified task into a scored result:
**load task → environment adapter → agent loop → unified trajectory → checkpoint scorer → result**
(validated against `spec/result.schema.json`). Three substrates, two parallel score lines
(Outcome + the 7 healthcare dimensions, specialized from the ETCLOVG taxonomy of Li et al.,
*Agent Harness Engineering: A Survey*),
three separately-recorded roles (brain / hands / judge).

## Quick start

```bash
# API brain via an OpenAI-compatible gateway (needs OPENAI_API_KEY + MH_OPENAI_BASE; MH_API_MODEL picks the model)
python3 runner/run.py --bench PhysicianBench --task PB-chronic_urticaria_allergist --agent gpt5 --out /tmp/r.json
```
PhysicianBench needs a live FHIR server first: `benchmark_dataprocess/PhysicianBench/run_fhir.sh`.

## Pipeline (3 phases — judge calls are isolated, the report is pure-read)

```
run.py / run_batch.py              →  result.json + trajectory.jsonl   (agent + tool backend + in-run judges)
scorers/rescore_judges.py          →  result.rescored.json             (the ONLY Clinical Conduct Boundaries judge caller)
scorers/aggregate_report.py        →  report.json                      (PURE-READ — zero model calls)
```

## Files

| File | Role |
|---|---|
| `run.py` | single-task CLI: load task → agent loop → score → build + schema-validate result; records role-separated `provenance` (`agent_model` / `tool_backend_model` / `judge_model`) |
| `run_batch.py` | batch + filters → per-task bundles + `summary.json`; reuses a finished task only when its recorded resume key matches |
| `environments.py` | env adapters by `environment.type`: **PbNativeEnv** (live HAPI FHIR, the upstream PhysicianBench tool registry) · **GuiEnvReal** (real Playwright portal, state from browser localStorage) · **ToolSandboxEnv** (MedCTA) |
| `agents.py` | `make_agent()` registry: `gpt5`/`openai` → `workspace.WorkspaceAgent` (the substrate's default workspace, running its agent program for the model's protocol) · `stub`/`replay`/`scripted` (regression agents, NOT baselines) |
| `models.py` / `models.yaml` | the agent model table: model id → protocol (`function_calling` or `text`); a model not listed does not run |
| `workspace.py` / `transport.py` | the workspace loader and agent object; the model transport (`chat_fc`, `chat_text`). The agent programs themselves live in `workspaces/<bench>/h0/` (`agent.py` function calling, `agent_text.py` text) |
| `middleware_host.py` | calls a workspace's host middleware (`before_model`, `before_tool`, `before_final`, `after_episode`, …) in manifest order |
| `harness/` | the harness kernel: capabilities (`scope_evidence`, `obligation_lifecycle`, `verify_commit`, `reference_integrity`), policy and semantic engines, ledger, repair primitives |
| `gateway.py` | unified OpenAI-compatible HTTP client; **per-role key** resolution (`override > MH_JUDGE_KEY > MH_OPENAI_KEY > OPENAI_API_KEY > ~/.mh_openai_key`); bounded retry under a hard wall-clock deadline |
| `adapters/medcta/vlm_backend.py` | MedCTA image perception: `ApiVLM` (a multimodal model behind the gateway, `MH_VLM_API_MODEL`, own `MH_VLM_API_KEY`) |
| `adapters/medcta/tools_medcta.py` | MedCTA tool backend: ImageDescription / RegionAttributeDescription (VLM) / OCR (EasyOCR) / Calculator / GoogleSearch (Serper, behind `search_guard.py`) |
| `adapters/medcta/gacc_judge.py` | MedCTA Goal-Accuracy (GAcc) judge, prompts copied from upstream |
| `adapters/pb/grader.py` | runs PhysicianBench upstream pytest checkpoints |
| `scorers/scoring.py` | checkpoint dispatch · subject-scope time-ordered state machine (cross-patient veto) · weighted 7-module aggregation |
| `scorers/governance_contract.py` | **single source**: governance blend · critical predicate/veto · `scoring_config` (tree hash, g14 weight) · `checkpoint_set_sha` |
| `scorers/governance.py` / `scorers/dim_*.py` | per-dimension evaluators (Care Action Fulfillment/Clinical Resource Competence/Patient–Evidence Alignment/Clinical Workflow Continuity/Clinical Auditability/Closed-Loop Verification + Clinical Conduct Boundaries) |
| `scorers/rescore_judges.py` | post-hoc Clinical Conduct Boundaries judge; writes `result.rescored.json` top-level `Clinical Conduct Boundaries` + `input_provenance` hashes; records judge independence (vs agent AND tool backend) |
| `scorers/aggregate_report.py` | pure-read report: `native_task_outcome` (single source) · evidence-tier coverage · provenance audit · paired model comparison (contract-gated) |
| `tests/test_conformance.py` | conformance checks (metric integrity, governance contract, provenance, tamper detection) |

## Scoring semantics

- **Outcome** (`native_task_outcome`) = dataset-native task success, a SEPARATE line from the 7 dims:
  PB = all `dimension=="Outcome"` checkpoints terminal-and-passed; HAB = every native eval passed;
  MedCTA = mean GAcc ≥ 0.5. Non-terminal/missing Outcome → unresolved (excluded from the rate; see
  `native_evaluation_coverage`). It NEVER reads `result["success"]` (the harness all-checkpoints gate, kept
  separate as `harness_gate`).
- **7 dimensions** = weighted aggregation of `dimension`-tagged checkpoints. `substrate_universal`
  (deterministic) dims are `formal_analysis_eligible` only when adapter admission is `ok`;
  `experimental_hybrid` (judge-backed) dims are reportable but never strict.
- **`success`** = has an evaluated checkpoint, no error, all passed (skipped ≠ success).
- **Clinical Conduct Boundaries** = the unified G1–G4 judge blended with the deterministic subject-scope critical veto
  (the one blend lives in `governance_contract.blend_governance`); judge failure → N/A, never a
  scope-only fallback.

## Provenance & integrity

`rescore_judges` stamps each `Clinical Conduct Boundaries` block with `scoring_config` (`scoring_code_tree_hash =
git rev-parse HEAD:runner`, judge/prompt ids, `tasks_unified_sha256`) and `input_provenance`
(`raw_result` / `trajectory` / `task` / `checkpoint_set` / `deliverable_files` sha256).
`aggregate_report` recomputes them live and reports `scoring_code_status` / `task_asset_status` /
`source_bundle_status` / `overall_artifact_status` (current only if all three verify). Run
`python3 -m pytest runner/tests/test_conformance.py -q` after any scoring-path change.

## FHIR state isolation

`run_task` cleans agent-created stub resources (`_tag=stub-run` MedicationRequest) per task by default
(`--unsafe-no-reset` skips it); `--reset-mode restore_pristine|per_task` re-derives a pristine H2 database
from the OCI layer (`benchmark_dataprocess/PhysicianBench/augmentation/restore_pristine_h2.sh`; a reset failure
is a hard error, never a silent dirty run).

## Extension guide

### A new dataset plugin

Create `runner/plugins/my_dataset.py`. The package discovers non-underscore Python modules automatically;
each module registers its own plugin. `benchmark` and a nonempty `tool_semantics` dictionary are required.

```python
import substrate as S

def resolve_read(event, previous_state):
    found = bool((event.get("result") or {}).get("found"))
    return {
        "role": "acquire",
        "status": "success" if found else "partial",
        "milestones_added": ["record_read"] if found else [],
        "progress_token": "record_found" if found else None,
    }

PLUGIN = {
    "benchmark": "MyDataset",
    "tool_semantics": {
        "read_record": {"role": "acquire", "success_milestones": ["record_read"]},
    },
    "resolvers": {"read_record": resolve_read},
}
S.register_plugin(PLUGIN)
```

Semantic roles are `acquire`, `act`, `verify`, `commit`, `final`, `escalate`, and `other`.
The resolver reads the actual tool result; declare milestones only when that result proves completion.
The plugin may also provide:

| Function entry | Signature | Consumer |
|---|---|---|
| `resolvers[tool_name]` | `(event, previous_state) -> dict` | `substrate.map_trace()` |
| `evidence_extractor` | `(trace) -> list[dict]` | `substrate.evidence_view()` |
| `task_contract_resolver` | `(task) -> dict` | `substrate.dimension_policy()` |

Store tasks in `benchmark_dataprocess/MyDataset/tasks_unified.jsonl`, conforming to
`spec/task.schema.json`, and register the dataset's prefix, domain, modalities, and environment in
`spec/registry.json`. Provide a workspace whose manifest declares `benchmark: MyDataset` and the task's
`env_type`; pass it with `--workspace`. The existing `fhir`, `gui`, and `tool_sandbox` environments can be
reused when their tool interfaces fit the dataset. Implement and register an environment and safety
adapter for a different execution interface. A semantic plugin describes tool behavior; the environment
must implement the actual tool dispatch. Define the dataset's native Outcome checks separately from the
seven process dimensions.

### Harness components and custom functions

A host middleware class is constructed with the public task. A kernel capability is constructed without
arguments. Register the classes in the workspace's `manifest.yaml`:

```yaml
middleware:
  - middleware/my_component.py:MyComponent
kernel_middleware:
  - middleware/my_capability.py:MyCapability
```

For example, a kernel component can call an ordinary Python helper to decide when to intervene:

```python
from workspace_api import Capability

def needs_review(action):
    return action.get("tool") == "read_record" and action.get("args", {}).get("needs_review")

class MyCapability(Capability):
    name = "my_review_gate"

    def before_action(self, action, ctx):
        if needs_review(action):
            return self._decide(
                "REVISE", rule_id="review_required",
                reason="Review this request before execution.",
                feedback="Review this request before execution.",
            )
        return None
```

Host hooks include `before_model(ctx) -> Observation | None`,
`before_tool(ctx, action) -> Revision | Nudge | None`,
`after_tool(ctx, action, result) -> PostRevision | None`, and
`before_final(ctx, answer) -> Revision | Nudge | None`. See `middleware_host.py` for all hooks and result
types. Host middleware runs in manifest order; the first non-`None` return wins at each hook.

Kernel hooks include `before_action(action, ctx)`,
`after_action(action, result, before_state, after_state, ctx)`, and `before_final(answer, ctx)`.
They return a harness decision or `None`; shell arbitration, safety checks, and budgets still apply.

Workspace modules import shell interfaces only through `workspace_api`. Standard-library imports must
be in `workspace.STDLIB_IMPORTS`; arbitrary third-party imports are not accepted by the workspace loader.
Use `workspace_api.load_relative(__file__, "helpers.py")` for a workspace-local helper module.
Registration loads components; a helper function runs only when a registered hook calls it.
