# Medical Harness — 项目设计锚(CLAUDE.md)

> **一句话定义(权威):**
> Medical Harness 是一套**统一的端到端医疗 agent 测评系统**:agent 获得任务许可的**完整多模态观测和工具能力**,**自主选择执行路径**;系统**保留源 benchmark 的结果指标**,同时以统一 canonical trace 评价**执行、工具、上下文、生命周期、可观测性、验证和治理(ETCLOVG 七维)**。

这是**一个统一测评系统,支持异构任务**——**不是**多条 benchmark 轨道(不要再发明"主轨/辅轨/native track/unified track")。

## 统一 vs 不统一

**统一的(系统的骨架,所有任务共享):**
- CanonicalTask / CanonicalObservation / CanonicalAction / CanonicalTrace
- 工具与能力声明(capability manifest)
- **required / optional / alternative** 工具语义
- 错误与 qualification 语义
- 七维 ETCLOVG 评分规则
- integrity(诚信门)报告
- 成本与可靠性统计

**不统一的(任务的内容,异构):**
- PB / MedCTA / HAB 的任务内容
- 各环境拥有的工具
- 图像 / FHIR / GUI 等观测模态
- 各自的**原生 outcome 指标**(如 Pass@1 / GAcc / task-subtask),作为 provenance 一并保留报告

## 默认配置 = 正常测评(配置不是轨道)

默认:**给 agent 任务许可的全部观测 + 全部工具,agent 自主决定看什么、调什么、何时结束。**
- `image_visible = true`,`tools_enabled = true` —— 脑能直接看图(多模态),工具**可选**调用。
- 消融只是同一系统里改 flag,放在 ablation section 解释失败来源,**不是独立 benchmark**:
  - 消融 A:`tools_enabled=false`(禁工具,看工具增益/干扰)
  - 消融 B:`image_visible=false`(禁直接看图,看感知是否瓶颈)
  - fixed-hand:所有 brain 用**同一个固定 VLM** 当感知工具 → 编排归因(注:测编排只要求 hand 在被比较 agent 间**恒定**,**不要求** hand≠brain)

## 工具语义与 Tooling 指标(关键纠偏)

工具调用**本身不是目标**。绝不能"没调 reference 工具就扣 Tooling"。工具按任务分三类:
- **Required**:任务/安全规则明确要求(如高风险给药前必须查 AllergyIntolerance/MedicationRequest)。缺 → 扣 `required_check_completion`。
- **Optional**:有帮助但不调也能正确完成(脑已看清图,OCR 只是辅助)→ **不调不扣**。
- **Alternative**:多条合法路径任选其一(直接视觉识别 OR ImageDescription+RegionAttribute)。

Tooling 问的是:**"需要工具时调了吗、不需要时避免了乱调吗、调了之后正确利用结果了吗"**,而非"有没有调参考工具"。统一为四子项:Required compliance / Selection appropriateness / Necessity calibration(对 Optional 不扣) / Evidence utilization。工具**执行层**(success / arg validity / redundant / latency)单独报告,不混进编排分。

## 评测层:ETCLOVG 七维 + native 指标

执行类(Execution / Tooling / Context / Lifecycle)+ 可信治理类(Observability / Verification / Governance)。**有原生检查点处 strict,其余 proxy(`score_eligible=false` 明标);结构性 n/a 保持 n/a,补=造假。** 同一份报告里并列:源 benchmark 原生指标 + ETCLOVG 七维 + integrity/cost/failure taxonomy。

## 诚信门(贯穿全程)

- **偏差登记**:所有相对原生的偏差进 `docs/NATIVE_FIDELITY.md`(维度/原生/我们/严重度/状态);native fidelity 同时作为**每次 run 的 provenance 字段**(source_benchmark / prompt_fidelity / protocol_fidelity / environment_fidelity / metric_definition)。
- **判官**:固定外部 **gpt-5.4**(= 上游 EVAL_MODEL,忠实)+ 多采样(降方差)+ cross-judge 方差;判 gpt agent 的同族重叠 = 上游同款局限,已登记。
- **provenance**:脑 / 手 / 判三角色如实记录(含各自真实模型名);track/config 标注;raw/native 证据不丢。
- **canonical 正确性靠 conformance tests 证明,不靠文档宣称。**

## 待证(论文成立的关键,非"是否能跑")

- A 表达力保留(每个原任务都有 canonical 轨迹能完成)
- B 契约最小字段集 + schema 校验落地
- C Conformance tests(adapter 对错的硬证据)
- D **必要性 / 解离**:统一画像揭示单 benchmark 看不到的能力分歧,**且对 scaffold(prompt/协议/感知后端)选择稳健**(robustness ablation)。这是把"统一"洗成"合理"的最终证据。

## 运行纪律(操作约束)

- **代码只在 ce483(`~/Medical_harness/`),不留本地副本。** 连接:`ssh ce483@hpc2login.hpc.hkust-gz.edu.cn`。
- 每轮改动记入 `docs/STATUS.md`(当前快照)+ `docs/CHANGELOG.md`(追加历史)。
- 绝不 `git add -A`;`.gitignore` 已挡 `/benchmark/`、`/results_*/`、`*.log`、`*.bak_*`;密钥 `~/.xbai_key`(chmod 600)绝不提交。
- `benchmark/` 是 git-ignored 的上游 vendored 资产(只读参照,复现 prompt/指标从这里取原文)。
