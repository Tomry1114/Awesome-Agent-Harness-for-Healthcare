# HealthAdminBench — 数据处理

原始数据(只读):`../../benchmark/HealthAdminBench/`(upstream repo:`benchmark/v2/tasks/` 任务 JSON、`portals/` 门户、`harness/` 现成 CUA harness)。
统一任务规范:`../../docs/00_task_specification.md`。

## 处理任务清单
- [ ] 研读 upstream `harness/`(现成 computer-use agent harness)——抽取轨迹/接口设计供统一 harness 参考
- [ ] 解析 `benchmark/v2/tasks/*.json`:`evals` = JMESPath(deterministic) + rubric(llm_judge)
- [ ] 映射:每个 subtask → 统一 `checkpoint`(JMESPath→deterministic;rubric→llm_judge)
- [ ] 门户部署确认:vercel(`emrportal.vercel.app`)或本地 `npm`(需 Node≥18 + Playwright);确认 HPC 计算节点可达性
- [ ] `convert.py` → `tasks_unified.jsonl`

## 在统一规范中的定位
environment=`gui`(Playwright + NextJS 门户)· modality=`gui_web` · 主压维度:Workflow Compliance / Governance(行政合规)/ Verification。

## 部署依赖
Python≥3.10 + uv + Node≥18 + Playwright Chromium + 模型 API key(OpenAI/Anthropic/Gemini/OpenRouter)。
