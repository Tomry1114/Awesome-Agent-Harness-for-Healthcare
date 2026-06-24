# Capability / Expressiveness-Preservation Matrix (Codex A)

Does the unified canonical representation PRESERVE each benchmark's native action/observation
expressiveness, or does it silently flatten it? One row per benchmark. "Preserved" = the canonical
form carries the same information a native consumer would need; gaps are registered (not hidden).

| Benchmark | Native actions | Native observation | → CanonicalAction | → CanonicalObservation | Preserved | Registered deviation |
|---|---|---|---|---|---|---|
| **PhysicianBench** (FHIR) | FHIR tool calls: `search` / `read` / `create` (granular FHIR ops) | FHIR `Bundle` (resources) | `tool_call` w/ canonical tool id + args | `modalities.structured` = flattened `entries` + `current_url`=base | ✅ full | tool-name 3-hop → single `_canon_fhir_tool` (env); Bundle `entry`→`entries` flatten recorded |
| **MedCTA** (tool sandbox) | Lagent ReAct: `ImageDescription` / `RegionAttributeDescription` / `OCR` / `GoogleSearch` / `Calculator` + `final_answer` | tool TEXT outputs (perception of a hidden image) | `tool_call` + `final_answer` | `modalities.text` (+ `localization{requested,mode,resolved}` for region tools) | ✅ full (tool-mediated) | **agent is BLIND by design** — image NOT in agent-visible obs; perception only via tools. `region_query` (semantic) accepted alongside `bbox` (pixel); no silent full-image fallback |
| **HealthAdminBench** (GUI) | bracket ops: `click([id])` / `fill([id],"…")` / `download([id])` / `done()` | page text + `data-mh-ref` interactable list | `tool_call` (GUI_OPS) | `modalities.text` + `current_url` + `artifacts` (downloads) | ✅ full (GUI ops) | hidden `full_state` (localStorage EMR) is **scorer-only**, never in agent obs (by design); `surface_changed` is a url+obs hash diff, NOT semantic progress (renamed, not over-claimed) |

## Cross-cutting expressiveness notes

- **Three observation layers are saved per step** (not collapsed): `result` (raw env dict) · `observation`
  (agent-visible serialization) · `canonical_observation` (audit-grade: `modalities` / `current_url` /
  `artifacts` / `previous_action_result`). The canonical layer is now **consumed** (Observability proxy reads
  `canonical_observation.modalities`), not write-only.
- **Hidden-state boundary** is explicit and identical in shape across benchmarks: anything the agent must
  NOT see (FHIR gold reference, GUI `full_state`, MedCTA image bytes) goes to the scorer ctx only, never
  into `observation`/`canonical_observation`.
- **What is NOT preserved (honest)**: native PROTOCOL syntax is deliberately dropped — we run ONE unified
  action protocol, not each benchmark's raw ReAct/bracket grammar. The "equivalent to upstream action
  syntax" claim is NOT made; `native_parsers.py` (the ReAct/bracket↔canonical bridge) was dead and removed.
  This is a scope decision (no native track), recorded here and in the deviation registry, with
  `gate_status: blocked` on the prompt-fidelity passport until human-checked.

## What this matrix is NOT

It asserts INFORMATION preservation (the scorer/agent can recover what they need), not BYTE/GRAMMAR
equivalence with the upstream harness. Byte-equivalence would require the native track, which is out of
scope. Coverage is reported as **strict X/7 + proxy (score_eligible=False)**, never an unqualified "7/7"
(see `report.json.coverage_summary`).
