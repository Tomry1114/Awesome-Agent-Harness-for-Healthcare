# Validation Report — Unified Tasks (A-line)

> 日期:2026-06-17 · 规范:Task Spec v2 (`docs/00_task_specification.md`) · schema:`spec/*.schema.json`
> 复现:`benchmark_dataprocess/<bench>/convert.py` → `validate_tasks.py`(JSON-schema)+ `semantic_validate_tasks.py`(语义)

## 结果

| Bench | tasks | JSON-schema | semantic errors | expected_warnings | unexpected_warnings |
|---|---|---|---|---|---|
| MedCTA | 107/107 | ✅ valid | 0 | 2 | 0 |
| HealthAdminBench | 135/135 | ✅ valid | 0 | 35 | 0 |
| PhysicianBench | 100/100 | ✅ valid | 0 | 0 | 0 |
| **Total** | **342/342** | **✅** | **0** | **37** | **0** |

## Known expected warnings(详见 `KNOWN_WARNINGS.md`)

- **MedCTA forced-choice options: 2**(`MCTA-98`, `MCTA-101`)— gold 答案是题目里明示的二选一选项;dataset-native,不泄露 `U`/`π`/tool-path。
- **HAB instruction-contained form values: 35** — 目标表单值写在指令里;考 GUI 执行/流程合规,而非数值推断。

> validator 已将这两类显式归为 `expected_warnings`(非 `unexpected_warnings`),不计入失败。
> 退出码:仅当出现 `errors` 或 `unexpected_warnings` 时非零。

## 复现命令(HPC ce483)

```bash
cd ~/Medical_harness/benchmark_dataprocess
DS=~/Medical_harness/benchmark/MedCTA/opencompass/data/medcta_dataset
PB=~/Medical_harness/benchmark/PhysicianBench/PhysicianBench
# convert
python3 MedCTA/convert.py --parquet $DS/train.parquet --img-out $DS/image --out MedCTA/tasks_unified.jsonl
python3 HealthAdminBench/convert.py --tasks-dir ~/Medical_harness/benchmark/HealthAdminBench/benchmark/v3/tasks --out HealthAdminBench/tasks_unified.jsonl
python3 PhysicianBench/convert.py --tasks-dir $PB/tasks/v1 --out PhysicianBench/tasks_unified.jsonl
# validate
for b in MedCTA HealthAdminBench PhysicianBench; do python3 validate_tasks.py $b/tasks_unified.jsonl; done
python3 semantic_validate_tasks.py MedCTA/tasks_unified.jsonl --medcta-root $DS
python3 semantic_validate_tasks.py HealthAdminBench/tasks_unified.jsonl
python3 semantic_validate_tasks.py PhysicianBench/tasks_unified.jsonl --pb-root $PB
```

数据/版本/校验和固化于 `TASK_MANIFEST.json`(项目根)。
