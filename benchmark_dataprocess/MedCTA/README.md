# MedCTA — 数据处理

原始数据(只读):`../../benchmark/MedCTA/`(upstream repo:`agentlego/` 工具库、`opencompass/` 评测、`vlm_models/`、`clinical_accuracy.py`、`goal_accuracy.py`)+ HF 数据集 `IVUL-KAUST/MedCTA`(待下载到本目录)。
统一任务规范:`../../docs/00_task_specification.md`。

## 处理任务清单
- [x] 下载 HF 数据集(经 `hf-mirror.com`;`train.parquet` 107 任务 + 内嵌图像)到 `../../benchmark/MedCTA/opencompass/data/medcta_dataset/`
- [x] 解析任务字段:`question`(Q)/`image_path`(X)/`tool_names`+`tool_chain`(U)/`trajectory`(π)/`gt_answer_json`(A)
- [x] `convert.py` → `tasks_unified.jsonl`(**107/107 通过 `spec/task.schema.json` 校验**)
- [x] **防泄露**:`available_tools`=固定 5 工具全集(agent 可见);`U`/`π`/`A`→`reference`(隐藏)。107 张图已从 parquet 抽出
- [ ] 工具后端落地:5 工具走 API 还是本地 VLM(GPU);**GoogleSearch frozen 缓存**
- [ ] 接 judge(Gacc,gpt judge)跑实际评测

## 产物
- `tasks_unified.jsonl` — 107 条统一任务
- `convert.py` — 转换脚本;`../validate_tasks.py` — schema 校验器

## 在统一规范中的定位
environment=`tool_sandbox`(agentlego 5 工具)· modality=`image_text` · 主压维度:Tool Use / Context(影像)/ Lifecycle / Observability。**本 bench 承载「图像」模态。**

## 部署依赖
OpenCompass + agentlego;感知工具可能需 VLM/GPU 或模型 API;GoogleSearch 需外网。
