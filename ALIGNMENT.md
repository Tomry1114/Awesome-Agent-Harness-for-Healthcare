# 对齐门 — Medical Harness(prompt / agent 协议 / 答案评分维度)

_更新于 2026-06-21 · 主机 ce483 `~/Medical_harness/`_
_覆盖范围:MedCTA + HAB 的 agent prompt、MedCTA 答案评分(Gacc)口径。**不含 PB;不是全论文 claim 清单**,只登记本轮核对到的交互协议/裁判/口径偏离。_

## 门禁判定:🚧 未完全通过(有已登记 gap)

本维度 8 项:**aligned 3 / gap 4 / extra 1**。存在 gap → 不宣称"已完全对齐";以下 gap 为**已知、经决策接受**的偏离(成本折中 / 统一协议设计选择),登记在案,进入与原文并表前需消解或显式声明。

## claim 表

| id | 原文 claim | 原文位置 | 我们的实现 | 状态 | 说明 |
|---|---|---|---|---|---|
| medcta-gacc-method | 答案由 LLM 裁判**语义**判分(非字符串匹配) | `benchmark/MedCTA/goal_accuracy.py:19,199` | cp_outcome llm_judge → `runner/gacc_judge.py` | ✅ aligned | 方法一致;白名单是喂 LLM 的 gold 文本 |
| medcta-gacc-prompt | `GOAL_ACCURACY_SYSTEM/USER_PROMPT` | `goal_accuracy.py:33-66` | `gacc_judge.py` **逐字复用** | ✅ aligned | — |
| medcta-gacc-granularity | 0.0–1.0 连续分 + `safe_mean` 聚合 | `goal_accuracy.py:37,231,138` | `gacc_judge` 出 0–1 + `report.py gacc_mean` | ✅ aligned | 二值 pass/fail 仅作阈值(MH_GACC_THRESHOLD)派生视图 |
| medcta-gacc-model | `EVAL_MODEL = "gpt-5.4"` | `goal_accuracy.py:19` | deepseek-v3.2(`MH_GACC_MODEL` 默认) | ⚠️ gap | 成本折中(账户额度);**影响与原文 leaderboard 可比性** |
| medcta-agent-prompt | "You are an expert medical imaging assistant…" | `benchmark/MedCTA/vlm_models/{gemini,claude,biomedx}.py` | `runner/qwen_agent.py` `SYS_BY_ENV["tool_sandbox"]`(自写) | ⚠️ gap | 措辞/约束不同 |
| pb-content-judge-model | PB 内容检查点用 LLM 判官(eval_helpers.llm_judge,默认 GPT-5) | `eval_helpers.py:58-72,623` | gemini-2.5-flash via xbai(`LLM_JUDGE_MODEL`;gpt-5.x 全 503) | ⚠️ gap | 方法忠实(LLM chat 判官)、模型偏离;影响并表 |
| hab-agent-prompt | DOM 动作空间 click/fill/select/scroll/back/download/upload/done + `ACTION:/KEY_INFO:` + 滚动记忆 | `benchmark/HealthAdminBench/sft/zero_shot_system_prompt_request.json` | `qwen_agent.py` `SYS_BY_ENV["gui"]`: click/type/select/submit/navigate + `<tool_call>/<answer>` | ⚠️ gap | **交互协议不同**;影响可比性 |
| medcta-cp_grounding | 原文**无** grounding 评分(goal_accuracy 只评答案) | `goal_accuracy.py`(缺) | 新增 cp_grounding(多模态裁判) | 🔵 extra | 已标 `provenance=augmented` + `multimodal_judge`(非冒充 native) |

## 7 类复刻失败自检(命中项)

- **命名口径漂移**:medcta-gacc-granularity 曾把原文 0–1 压成二值 → 本轮(CHANGELOG 续4)已修为连续分。
- **baseline 不实 / 可比性**:medcta-gacc-model + 两处 agent-prompt → 与原文数字**不可直接并表**,已登记。
- **bug 当创新**:cp_grounding 曾以 `native` 名义进正式分(续3 已 relabel augmented + 上真多模态裁判)。

## 阻断项(进入与原文并表/"已复刻"声明前需消解或显式接受)

1. `medcta-gacc-model`:要与原文 leaderboard 并表需切 gpt-5.4;否则 gacc_mean 仅供自洽横比。
2. `medcta-agent-prompt` / `hab-agent-prompt`:要复刻原文 agent 行为需用原文 prompt;当前统一 `<tool_call>` 协议是 harness 设计选择,非论文复刻。
