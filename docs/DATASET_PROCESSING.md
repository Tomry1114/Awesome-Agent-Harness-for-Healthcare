# Medical Harness — 数据集处理与 7 维覆盖说明

> 本文说明三个源 benchmark 如何被纳入统一 harness、7 个 ETCLOVG 维度如何在**每个数据集**上补全、以及统一协议(canonical action interface)如何工作。配套:`ARCHITECTURE.md`(架构契约)、`IMPLEMENTATION_STATUS.md`(逐件状态)、`METRICS.md`(指标口径)。

## 1. 总览

三个 agentic、互不重叠的源 benchmark,合起来 + 各自补全后,**每个数据集独立覆盖全部 7 个 ETCLOVG 维度**:

| 源 benchmark | 模态 | 真实环境 |
|---|---|---|
| PhysicianBench (PB) | 结构化 FHIR 临床 | 真 HAPI-FHIR(H2)+ 粒度 FHIR 工具 |
| HealthAdminBench (HAB) | GUI 行政流程 | 真 Next.js 门户 + Playwright |
| MedCTA | 多模态影像工具使用 | 真工具沙箱(Qwen3-VL 像素 grounding / 离线检索 / 计算器) |

7 维分两大类:

| 大类 | 维度 |
|---|---|
| 任务执行能力 | Execution / Tooling / Context / Lifecycle |
| 可信治理 | Observability / Verification / Governance |

## 2. 统一协议(canonical action interface)

源 benchmark 的原生协议各不相同(PB=function-calling、MedCTA=ReAct、HAB=`click([id])`)。统一轨道**规定一套 canonical 动作接口**,底层各环境执行自己的动作:

```
异构原生任务与协议
      ↓ adapters(协议解析 + 观测渲染 + 错误映射)
统一任务 / 轨迹表示(CanonicalTask / CanonicalTrace,保留 raw/native 证据)
      ↓
7 维 ETCLOVG 评测(两大类)
      ↓
保留 native 指标(Pass@1 / GAcc / task-subtask) + 新增 harness 指标
```

- canonical 动作:`tool_call`(FHIR / MedCTA 工具)、`gui_action`(click/type/select/submit)。
- 统一 trace event:`{event_type, tool, args, observation, result, status, step, thought}`。
- 判据不是"逐字复现原论文动作语法",而是"**canonical 接口完整保留原任务的表达能力**,并支持统一审计"(详见 ARCHITECTURE §1–2)。
- Native replication 轨道单独保留原 prompt / 观测 / 协议 / 终止规则 / native 指标(实验语义等价,不可避免偏差进 alignment passport)。

## 3. 每个数据集如何补全 7 维

补全分三种合法来源,**严格分级、互不污染**:

| 来源 | 进主分? | 说明 |
|---|---|---|
| **strict** | ✅ | 源 benchmark 自带的正式 checkpoint(deterministic / judge / policy) |
| **retag** | ✅(strict) | 口径 B 下把 `category=reasoning` 的 cp 从 Execution 重标为 Verification(正确性) |
| **post-hoc judge** | ✅(strict) | 对已存产出(答案 / 轨迹中的输入内容)补判被关掉的 judge/policy cp,不重跑 agent |
| **proxy** | ❌ soft | 从轨迹推的启发式软信号(`score_eligible=False`),只填 strict 未覆盖的格、不进成功率 |

### 口径 B(Execution vs Verification 边界)
- **Execution = 完成度**:动作发生 / 产出存在。
- **Verification = 正确性**:产出与 ground-truth 比对(推理对不对、答案对不对、终态合规)。
- 落地规则:`category=reasoning → dimension=Verification`(三家一致,`runner/retag_verification.py`,可回滚,保留 `original_dimension`)。

### proxy 软信号(`runner/proxy_verifiers.py`,modality-agnostic)
- **Tooling** = 工具使用质量 = `1 − 0.5·错误率 − 0.5·冗余率`
- **Lifecycle** = 顺序合理性 = 目标动作(create / final_answer)前是否有过信息获取
- **Observability** = 工具调用产生可观测输出的比例(留痕)
- **Execution**(仅当无 strict 时)= 到达终答且 ≥1 次成功工具调用
- 仅对**没有 strict 覆盖**的维度发出,绝不覆盖/冲突 strict 分。

### post-hoc judge(`runner/rescore_judges.py`,gateway gpt-5.5,无 GPU、不重跑 agent)
- MedCTA Governance(`cp_no_fabrication`):判最终答案 vs 工具观测有无编造。
- HAB Verification(`llm_judge` + rubric):agent 输入的 triage note 从轨迹 type 动作恢复 → 跑 rubric。
- HAB Governance(`policy` forbidden_actions):判轨迹动作有无违规。
- 关键:GUI 的 full_state 未持久化,但 **agent 自己的输入/动作在轨迹里**,故可 post-hoc 恢复评分。

## 4. 逐数据集 7 维覆盖(当前,10 题基线)

> 数值为 10 题基线;native=源自带、retag=口径B重标、proxy=软信号、post-hoc=补判。

| 维度 | PB | MedCTA | HAB |
|---|---|---|---|
| Execution | strict 0.30 | proxy 1.0 | proxy 0.70 |
| Tooling | proxy 0.99 | strict 0.20 | proxy 0.50 |
| Context | strict 0.40 | strict 0.90 | strict 0.75 |
| Lifecycle | proxy 0.90 | proxy 1.0 | strict 0.60 |
| Observability | strict 0.60 | proxy 1.0 | strict 0.27 (含 post-hoc) |
| Verification | strict 0.66 (retag) | strict 0.70 (retag) | strict 0.0 (retag + post-hoc) |
| Governance | strict 1.0 | post-hoc 0.70 | post-hoc 0.90 |
| **小计** | **7/7** | **7/7** | **7/7** |

> 注:HAB Verification=0.0 是真实严苛结果(从轨迹恢复的 triage note 经 rubric 判官全 fail);proxy 值见 §3 软信号说明(MedCTA 的 1.0 多为饱和、信息量低)。

## 5. 诚信门记录(over-attribution 更正 + 边界)
- **更正**:旧 STATUS 把"Verification ◎ = native_pytest"当成维度覆盖,实为过度归因——native_pytest 是跑所有确定性 cp 的*引擎*,不等于 Verification 维度有题。经 retag 后才有真实 strict 覆盖。
- **覆盖来源透明**:report.json 中每维标注 strict / proxy(`score_eligible`)/ post-hoc,proxy 永不进成功率。
- **proxy 饱和提示**:MedCTA 的 Execution/Lifecycle/Observability proxy 多为 1.0——任务结构(QA:观察→工具→答)保证了这些,软信号信息量低,不应解读为强能力。

## 6. 指标 roll-up(`runner/aggregate_report.py` → 各 results 目录 `report.json`)
- `native_metrics`:PB Pass@1、MedCTA GAcc、HAB task/subtask
- `harness_dimensions`:7 维两大类 + 诚实覆盖(covered / not_exercised)
- `proxy_dimensions`:gap-fill 软信号
- `integrity`:judge 独立性 / 工具后端 / qualification 汇总
- `failure_taxonomy`:checkpoint failure_mode × 维度 + task failure_tags
- 自动 remap 旧 run 到当前 tasks_unified 标签(不重跑)。
