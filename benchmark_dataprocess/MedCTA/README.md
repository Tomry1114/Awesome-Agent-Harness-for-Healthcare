# MedCTA

Raw upstream assets (read-only) live in `../../benchmark/MedCTA/`: the upstream repository (agentlego tools,
OpenCompass evaluation) and the HuggingFace dataset `IVUL-KAUST/MedCTA` (`train.parquet`, 107 tasks with embedded
images, extracted under `opencompass/data/medcta_dataset/`).

## Files

| file | content |
|---|---|
| `convert.py` | parquet rows → `tasks_unified.jsonl` (107 tasks, validated against `spec/task.schema.json`) |

Fields: `question` → goal, `image_path` → the task image, `tool_names` / `tool_chain` / `trajectory` /
`gt_answer_json` → `reference`.

The agent receives all five tools; the native tool subset, tool chain,
reference trajectory and gold answer stay in `reference`, which the agent never sees.

## Environment

environment `tool_sandbox` (five tools: `ImageDescription`, `RegionAttributeDescription`, `OCR`, `GoogleSearch`, `Calculator`) · modality
`image_text`. Outcome is the native goal accuracy (Gacc),
graded by a judge model.

## Dependencies

The perception tools call a vision-language model through an OpenAI-compatible API; web search needs a Serper key
(`MH_SERPER_KEY` / `SERPER_API_KEY`) and outbound network access.
