# Prompt Provenance (native reproduction spec)

权威来源:用户 2026-06-23 给定。三个 benchmark 主实验各自的 native agent prompt + 评测 prompt 出处,
以及我们 harness 接入时**只允许运行时变量替换、不得增删临床/工具提示**。

## PhysicianBench
- **System**:`agent/prompts.py` 的全局极简 SYSTEM_PROMPT(clinical AI assistant;检索后决策;create 下单;write_file 交付;thorough/accurate;完成全部)
- **User/task**:`tasks/v1/<task>/instruction.md`(逐题)——**我们的 `goal` 已逐字等于 instruction.md** ✅
- **Eval**:每题 `task.toml` + `tests/`(部分代码验证、部分 rubric/LLM)。无统一 judge prompt。
- **允许改**:FHIR URL / workspace / 输出路径等运行时变量
- **禁止加**:"必须先查过敏""一定要 write_file""别逐个读"等额外提示(会改变它要测的自主临床工作流)

## MedCTA
- **Agent**:论文 Figure 9 的 ReAct execution prompt(Thought/Action/Action Input/Response/Finish);官方 LagentAgent(ReAct,max_turn=10)
- **User**:数据集原始 `{questions}` + 原图/文件;**不告诉 agent 工具序列**(step-implicit)
- **Judge**:`goal_accuracy.py`(GAcc 0-1,evaluator=GPT-5.4)+ `clinical_accuracy.py`(Faithfulness/Context/Precision/Completeness 各自 prompt)
- **禁用**:Figure 10 的 trajectory-generation prompt(会泄漏工具数/轨迹结构/gt answer)

## HealthAdminBench
- **Prompt mode**:**general**(Task Description + Portal Guidance)——**不是** zero_shot、**不是** task_specific
- **System**:`harness/prompts.py`(PromptMode/DOM action-space/坐标 action-space/响应格式/identifier 规则/done 规则)
- **Portal Guidance**:`harness/healthcare_hints.py`(general 模式追加)
- **Task goal**:`benchmark/v2/tasks/<task>.json`
- **Observation**:**screenshot-only**(论文主设置;每步注入 OBJECTIVE/date/URL/step/screenshot/recent actions/observations/page elements,要求只给下一步 action)
- **不用于主结果**:task_specific

## 我们 harness 的接入(prompt_track 标注)
```
physicianbench: {prompt_track: native, system: agent/prompts.py, task: instruction.md, mod: runtime_vars_only}
medcta:         {prompt_track: native, agent: paper_fig9_react, user: dataset.questions,
                 judge: [goal_accuracy.py, clinical_accuracy.py], excluded: paper_fig10_traj_gen}
healthadminbench:{prompt_track: native, prompt_mode: general, system: harness/prompts.py,
                 portal_guidance: harness/healthcare_hints.py, task: benchmark/v2/tasks/<task>.json,
                 observation: screenshot_only}
```

## 当前实现状态(REGISTERED DEVIATIONS)
- `MH_PROMPT_TRACK=native` 开关:已实现 PB native system prompt(`NATIVE_SYS_BY_ENV["fhir"]`,上游极简原文)。
- **偏差(stage-2 待消)**:① 工具调用机制——我们用统一文本 `<tool_call>` 协议,非各自 native(PB function-calling / MedCTA ReAct / HAB bracket-action);② HAB 观测——我们用 DOM 文本+ref,非 screenshot;③ MedCTA/HAB 的 native system prompt 未接(仍用 harness 版)。
- 结论:当前 native track 仅 **prompt 文本对齐 PB**;协议/观测未对齐 → **数字暂不可直接对标 published leaderboard**,需 stage-2。
