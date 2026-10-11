# Runner

The runner loads a unified task, executes the agent in its benchmark environment, records the trajectory, and evaluates checkpoints. Run commands from the repository root after configuring the environment and API settings in the [main README](../README.md#running).

## Commands

| Command | Output |
|---|---|
| `run.py` | Single-task result and trajectory |
| `run_batch.py` | Per-task bundles and `summary.json`; resumes matching completed runs |
| `scorers/rescore_judges.py` | `result.rescored.json`, including the Clinical Conduct Boundaries judge score |
| `scorers/aggregate_report.py` | `report.json`, without model calls |

Use `--help` for task filters, execution budgets, and workspace options.

## Code

| Path | Purpose |
|---|---|
| `environments.py` | FHIR, GUI, and tool-sandbox adapters |
| `agents.py` | API agent and stub, replay, and scripted test agents |
| `models.yaml` | Agent model IDs and tool-call protocols |
| `workspace.py`, `transport.py` | Workspace loading and model calls |
| `middleware_host.py` | Host middleware hooks |
| `harness/` | Kernel capabilities, policies, semantic analysis, and evidence ledger |
| `gateway.py` | API client, role-specific keys, timeouts, and retries |
| `adapters/` | Benchmark tools and native graders |
| `scorers/scoring.py` | Checkpoint evaluation |
| `scorers/governance_contract.py` | Clinical Conduct Boundaries blending, critical veto, and scoring configuration |
| `scorers/dim_*.py`, `scorers/governance.py` | Dimension evaluators |
| `tests/` | Runner and scoring tests |

Default agent programs and components live in `workspaces/<benchmark>/h0/`. Model provenance records the agent, tool backend, and judge separately. A judge matching the agent or tool backend is rejected as non-independent.

## Scoring

`native_task_outcome` uses benchmark-native checks: all Outcome checkpoints must pass for PhysicianBench and HealthAdminBench; MedCTA's mean GAcc must be at least 0.5. Missing or non-terminal checks remain unresolved and are reflected in `native_evaluation_coverage`.

`result.json`'s `success` field requires evaluated checkpoints, no error, and all checks passed. The report exposes it as `harness_gate` separately from native outcome.

The aggregate report includes the seven healthcare dimensions, evidence coverage, and scoring status. Deterministic `substrate_universal` scores are eligible for formal analysis only when adapter admission is `ok`. Judge-backed `experimental_hybrid` scores are reported with that status.

Clinical Conduct Boundaries combines the G1–G4 judge with a deterministic subject-scope veto. A failed judge leaves the score unavailable.

Post-hoc scores include hashes of scoring code, tasks, trajectories, checkpoint sets, and deliverables. Report generation verifies these against current inputs and records artifact status. After changing scoring code, run:

```bash
python3 -m pytest runner/tests/test_conformance.py -q
```

## FHIR resets

The runner removes agent-created `stub-run` MedicationRequest resources before each task by default. `--unsafe-no-reset` skips this cleanup.

`--reset-mode restore_pristine` or `--reset-mode per_task` restores the H2 database from the container's OCI layer using `benchmark_dataprocess/PhysicianBench/augmentation/restore_pristine_h2.sh`. Reset failures stop the run.

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
