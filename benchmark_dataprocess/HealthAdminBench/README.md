# HealthAdminBench

Raw upstream assets (read-only) live in `../../benchmark/HealthAdminBench/` (v3 task JSON and the Next.js portals).

## Files

| file | content |
|---|---|
| `convert.py` | upstream v3 tasks → `tasks_unified.jsonl` (135 tasks, `spec/task.schema.json`) |

`convert.py` maps each task's `evals` to unified checkpoints: JMESPath checks become `deterministic` checkpoints,
rubric checks become `llm_judge` checkpoints, both with `provenance=native`. Evaluation categories determine checkpoint dimensions and subdimensions. Workflow progression is derived from the task type and observed trace. Administrative-compliance checks are added by the harness.

## Environment

environment `gui` (Playwright driving the Next.js portals) · modality `gui_web`.

## Dependencies

Python ≥ 3.10, Node ≥ 18 and Playwright Chromium for the portals, and an API key for the agent and judge models.
