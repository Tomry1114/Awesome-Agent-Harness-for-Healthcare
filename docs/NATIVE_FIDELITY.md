# Native 保真度偏差登记册(Medical Harness)

> 统一轨道允许 registered deviation,但必须**登记**(诚信门)。本册列出我们相对各源 benchmark 原生设置的偏差、严重度、修复状态。配套 `PROMPT_PROVENANCE.md`(prompt 层)、`DATASET_PROCESSING.md`(维度/指标)。

## 偏差表

| # | 维度 | 原生 | 我们 | 严重度 | 状态 |
|---|---|---|---|---|---|
| 1 | **脑模态(MedCTA)** | headline=多模态直接看图(VQA);tool-path 才是文本脑 | text-only 脑 + 感知工具 + **新增 VQA-direct** | 🔴→✅ | **已修**:`runner/vqa_direct.py`(gpt-5.5 vision 直接看图,GAcc 评分)→ report.native_metrics.vqa_direct headline,与 tool-path 并列 |
| 2 | **clinical_accuracy 4 子指标** | F_acc 临床忠实 / C_s 上下文整合 / F_p 事实精确 / S_comp 语义完整 | 原只有 GAcc | 🔴 缺 | **已实现**:`runner/clinical_accuracy_judge.py` 忠实复用原生 `clinical_accuracy.py` rubric,经 gateway 重评,写入 report.native_metrics.clinical_accuracy |
| 3 | **max_steps** | PB=30 / MedCTA max_turn=10 | 40 / 50 | 🟡 偏 | **已修**:run_batch `NATIVE_MAX_STEPS`(PB 30 / MedCTA 10 / HAB 30) |
| 4 | **判官采样** | HAB num_runs 多次取均 | **MH_JUDGE_SAMPLES 多次平均** | 🟡→✅ | **已修**:tool_use_judge 多采样取均,降 ~0.12 噪声 |

## 已登记的协议层偏差(见 PROMPT_PROVENANCE.md)
- 文本 `<tool_call>` 统一协议 vs 原生 function-calling(PB)/ ReAct(MedCTA)/ `click([id])`(HAB)——unified track 的 canonical 接口,by design。
- HAB 文本观测(可访问性树)vs 原生 screenshot——stage-2 待办。

## 修复优先级
1. #2 clinical_accuracy(已做)→ 验证数值
2. #3 max_steps(已做)
3. #4 判官采样(便宜,且降噪)
4. #1 VQA-direct 模式(最大,新 agent 路径)


## 公平性前提(unified track 要"公平"而非仅"统一")

| # | 要求 | 为何是前提 | 状态 |
|---|---|---|---|
| F2 | prompt/obs 中性不污染(公平暴露工具、不藏不教、不截断、无 fail-by-construction)| 否则测的是我们的偏置(MedCTA 藏 GoogleSearch、obs 截断 200;PB 教策略+暗示 demographics 被评分)| ✅ obs→10k;**MedCTA**(91b5301)与 **PB**(本次)均中性化——PB 直接用上游 `agent/prompts.py` SYSTEM_PROMPT 原文(无机制教学/无 scored 暗示/无 obs-bug 脚手架)|
| F3 | 协议选公平标准(native function-calling),别用 text `<tool_call>` 怪协议 | text 统一但偏袒——惩罚更擅长原生 function-calling 的前沿模型 | ✅ `MH_PROTOCOL=function_calling`(openai_agent),gateway 实测支持,MedCTA replay 闭环跑通 |


## prompt 一致性审计(三家 vs 上游仓库)

| Bench | 上游源 | 一致性 | 偏差 |
|---|---|---|---|
| PB | `PhysicianBench/agent/prompts.py` SYSTEM_PROMPT | **guidelines 逐字一致** | + {tools} 暴露 + 文本协议(unified track) |
| MedCTA | Lagent ReAct(`lagent/agents/react.py`) | **语义一致**(中性、全工具) | 非 Lagent 精确模板(协议偏差) |
| HAB | `harness/prompts.py` GENERAL("autonomous web agent" + click([id])/fill/select/scroll/back/download/upload/done) | **不一致** | 协议:我们 ref=N+JSON vs 上游 click([id]) bracket;**动作缺口**:已补 back/scroll/done,**仍缺 download/upload**(stage-2 文件处理)→ 需要下载/上传的任务暂 out-of-scope |

**HAB 行动项**:① download/upload(Playwright 文件处理)② 可选:上游 click([id]) bracket 协议 + screenshot 观测(完整 stage-2)。在补齐前,涉及文件下载/上传的 HAB 任务应排除,避免 fail-by-construction。
