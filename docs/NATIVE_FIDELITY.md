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
