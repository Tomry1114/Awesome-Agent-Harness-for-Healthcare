# HealthAdminBench — data processing

Raw upstream assets (read-only) live in `../../benchmark/HealthAdminBench/` (v3 task JSON and the Next.js portals).

## Files
| file | content |
|---|---|
| `convert.py` | upstream v3 tasks → `tasks_unified.jsonl` (135 tasks, `spec/task.schema.json`) |

`convert.py` maps each task's `evals` to unified checkpoints: JMESPath checks become `deterministic` checkpoints,
rubric checks become `llm_judge` checkpoints, both with `provenance=native`. Each eval category is mapped to one of
the 7 healthcare dimensions and a sub-dimension. Most native evals check the FINAL portal state, so they feed
Closed-Loop Verification and Patient–Evidence Alignment; the workflow stage used by Clinical Workflow Continuity comes from the task type and the observed trace, not
from the order of the eval array. Clinical Conduct Boundaries uses an administrative-compliance policy overlay.

## Place in the unified format
environment `gui` (Playwright driving the Next.js portals) · modality `gui_web`.

## Dependencies
Python ≥ 3.10, Node ≥ 18 and Playwright Chromium for the portals, and an API key for the agent and judge models.
