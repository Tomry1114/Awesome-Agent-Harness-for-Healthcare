# Unified Harness Runner

One main loop turns any unified task into a scored result:
**load task → environment adapter → agent loop → unified trajectory → checkpoint scorer → result**
(validated against `spec/result.schema.json`). Three substrates, two parallel score lines
(Outcome + 7 ETCLOVG dimensions), three separately-recorded roles (brain / hands / judge).

## Quick start

```bash
# API brain via the micuapi gateway (needs OPENAI_API_KEY + MH_OPENAI_BASE)
python3 runner/run.py --bench PhysicianBench --task PB-chronic_urticaria_allergist --agent gpt5 --out /tmp/r.json
# fully local Qwen3-VL brain (no API key)
python3 runner/run.py --bench MedCTA --task MCTA-0 --agent qwen
```
PhysicianBench needs a live FHIR first: `benchmark_dataprocess/PhysicianBench/run_fhir.sh`.

## Pipeline (3 phases — judge calls are isolated, the report is pure-read)

```
run.py / run_batch.py   →  result.json + trajectory.jsonl   (agent + tool backend + in-run judges)
rescore_judges.py       →  result.rescored.json             (the ONLY Governance judge caller; caches)
aggregate_report.py     →  report.json                      (PURE-READ — zero model calls)
```

## Files

| File | Role |
|---|---|
| `run.py` | single-task CLI: load task → agent loop → score → build + schema-validate result; records role-separated `provenance` (`agent_model` / `tool_backend_model` / `judge_model`) + `_qualification` flags |
| `run_batch.py` | batch + filters → per-task bundles + `summary.json` |
| `environments.py` | env adapters by `environment.type`: **FhirEnv** (real HAPI: search/read/create/lab_ref/write) · **GuiEnvReal** (real Playwright portal, state from browser localStorage) + **GuiEnvMock** · **ToolSandboxEnv** (MedCTA) |
| `tool_agent.py` | **ToolProtocolAgent** — the shared, model-agnostic `<tool_call>`/`<answer>` protocol + MedCTA multimodal helpers + prompt build |
| `api_agent.py` | **ApiToolAgent** — API brain over the gateway; subclasses ToolProtocolAgent, swaps only transport + FC schema |
| `agents.py` | `make_agent()` registry: `gpt5`/`openai`→ApiToolAgent · `qwen`→local Qwen3-VL brain · `stub`/`replay`/`scripted` (regression agents, NOT baselines) |
| `gateway.py` | unified OpenAI-compatible HTTP client; **per-role key** resolution (`override > MH_JUDGE_KEY > MH_OPENAI_KEY/OPENAI_API_KEY > ~/.xbai_key`); bounded retry under a hard wall-clock deadline |
| `vlm_backend.py` | MedCTA image perception: `ApiVLM` (gateway, default gpt-5.x, own `MH_VLM_API_KEY`/`MH_VLM_API_MODEL`) or local `Qwen3-VL` |
| `tools_medcta.py` | MedCTA tool backend: ImageDescription / RegionAttributeDescription (real crop) / OCR / Calculator / GoogleSearch(frozen) |
| `scoring.py` | checkpoint dispatch · subject-scope time-ordered state machine (cross-patient veto) · weighted 7-module aggregation |
| `governance_contract.py` | **single source**: governance blend · critical predicate/veto · `scoring_config` (tree hash, g14 weight) · `checkpoint_set_sha` |
| `governance.py` / `dim_*.py` | per-dimension evaluators (Execution/Tooling/Context/Lifecycle/Observability/Verification + Governance) |
| `rescore_judges.py` | post-hoc Governance judge; writes `result.rescored.json` top-level `Governance` + `input_provenance` hashes; fail-closed judge independence (vs agent AND tool backend) |
| `aggregate_report.py` | pure-read report: `native_task_outcome` (single source) · evidence-tier coverage · provenance audit (3 axes + overall) · paired model comparison (contract-gated) |
| `native_pytest.py` | runs PhysicianBench upstream pytest checkpoints |
| `test_conformance.py` | 109 conformance checks (metric integrity, governance contract, provenance, tamper detection) |

## Scoring semantics

- **Outcome** (`native_task_outcome`) = dataset-native task success, a SEPARATE line from the 7 dims:
  PB/HAB = *all* `dimension=="Outcome"` checkpoints terminal-and-passed; MedCTA = mean GAcc≥0.5.
  Non-terminal/missing Outcome → unresolved (excluded from the rate; see `native_evaluation_coverage`).
  It NEVER reads `result["success"]` (the harness all-checkpoints gate, kept separate as `harness_gate`).
- **7 dimensions** = weighted aggregation of `dimension`-tagged checkpoints. `substrate_universal`
  (deterministic) dims are `formal_analysis_eligible` only when adapter admission is `ok`;
  `experimental_hybrid` (judge-backed) dims are reportable but never strict.
- **`success`** = has an evaluated checkpoint, no error, all passed (skipped ≠ success).
- **Governance** = the unified G1–G4 judge blended with the deterministic subject-scope critical veto
  (the one blend lives in `governance_contract.blend_governance`); judge failure → N/A, never a
  scope-only fallback.

## Provenance & integrity

`rescore_judges` stamps each `Governance` block with `scoring_config` (`scoring_code_tree_hash =
git rev-parse HEAD:runner`, judge/prompt ids, `tasks_unified_sha256`) and `input_provenance`
(`raw_result` / `trajectory` / `task` / `checkpoint_set` / `deliverable_files` sha256).
`aggregate_report` recomputes them live and reports `scoring_code_status` / `task_asset_status` /
`source_bundle_status` / `overall_artifact_status` (current only if all three verify). Run
`python3 runner/test_conformance.py` (expects **109/109**) after any scoring-path change.

## FHIR state isolation

`run_task` cleans agent-created stub resources (`_tag=stub-run` MedicationRequest) per task by default;
`--reset-mode restore_pristine|per_task` re-derives a pristine H2 from the OCI layer (reset failure is
a hard error, never a silent dirty run).
