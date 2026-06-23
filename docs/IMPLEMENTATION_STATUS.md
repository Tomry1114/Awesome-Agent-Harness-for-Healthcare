# Medical Harness 实现状态(草稿)

> 状态:草稿,集群恢复后提交到 ce483 `docs/IMPLEMENTATION_STATUS.md`。
> 图例:✅ 已实现 · 🟡 部分/非正式 · 🔴 缺。置信度=这轮亲见代码 / 推断。

## 段① 异构原生任务与协议(env 层)

| 件 | 状态 | 说明 |
|---|---|---|
| FHIR env(PB) | ✅ | 真 HAPI-FHIR H2 + 13 粒度工具 + entries 修复(亲见) |
| 工具沙箱(MedCTA) | ✅ | 真 Qwen3-VL 像素 grounding + 离线检索 + 计算器(亲见) |
| 门户(HAB) | ✅ | 真 Playwright + Next.js,GuiEnvReal(亲见) |
| 三套任务→tasks_unified.jsonl | ✅ | 已转,字段契约未文档化 |

## 段② Adapter(隐式散落,未形成正式接口)

| Adapter 组件 | 状态 | 说明 |
|---|---|---|
| 环境执行 adapter | 🟡 | 已存在(各 env `act()`),但分散、未形成正式接口 |
| 任务转换 adapter | 🟡 | tasks_unified.jsonl 已存在,字段契约未文档化 |
| 观测 adapter | 🟡 | 三环境都能输出,但输出类型与语义未规范 |
| 动作 adapter | 🟡 | 有统一执行入口,但原生协议转换能力不足 |
| Native protocol adapter | 🔴 | 特别是 HAB 尚缺(screenshot+bracket) |
| Adapter conformance tests | 🔴 | 未见 |

> 技术核心**不是从零开始**,而是把现有隐式结构抽成 `BenchmarkAdapter`(见 ARCHITECTURE §2)。

## 协议保真(两 track 分件,不可笼统说"原生协议全没有")

| 项目 | 状态 |
|---|---|
| Unified canonical protocol | 🟡 已运行,契约未正式化(动作词汇表+表达能力保留未写成合同) |
| PB native protocol fidelity | 🟡 未完成实测 |
| MedCTA native protocol fidelity | 🟡 文本层已对齐,ReAct/runtime 是否完全一致待确认 |
| HAB native protocol fidelity | 🔴 screenshot+bracket/general 尚未实现 |

## 段③ 统一任务/轨迹表示

| 件 | 状态 | 说明 |
|---|---|---|
| trace 落盘 | 🟡 | trajectory.log 有,schema 未正式定 |
| 统一 task schema | 🟡 | tasks_unified.jsonl 在,字段未文档化 |
| raw/native 证据保留 | 🟡 | 部分保留,未规范为 raw↔canonical provenance |

## 段④ 7 维 ETCLOVG 评测(二大类)

| 件 | 状态 | 说明 |
|---|---|---|
| 7 维评分框架 | ✅ | summary.dimension_means 已出(亲见 pbC/mctaC) |
| 维度覆盖 | 🔴 | pbC 里 Tooling/Lifecycle/Verification=null(coverage 0 task)——框架在,半数维度当前无 checkpoint |
| 二大类分组 | 🔴 | 维度平铺,未分"任务执行能力/可信治理" |
| safety 维 | 🟡 | risk_annotator 动作安全 judge 已写,门控 MH_ACTION_SAFETY_JUDGE,默认关 |

## 段⑤ native + harness 双指标

| 件 | 状态 | 说明 |
|---|---|---|
| harness 指标 | 🟡 | 7 维(覆盖不全) |
| native 指标并排 | 🔴 | Pass@1 / GAcc / task-subtask 未与 harness 维度并排报 |
| failure taxonomy(显式标签) | 🔴 | 只有分数,无失败类型标签 |
| integrity report(汇总) | 🟡 | qualifications + context_grounding guard + judge provenance 有,未汇总成报告 |

## 需要证明的三件事(均 🔴 待做)

| | 内容 | 状态 |
|---|---|---|
| A | 表达能力保留(逐 bench 能力矩阵) | 🔴 |
| B | 契约正式化(action/obs/result/error schema + 终止 + version + provenance) | 🔴 |
| C | Conformance tests(每 env 6 项) | 🔴 |

## 干净基线(已有,供后续对照)

| | pbC(PB-10) | mctaC(MedCTA-10) |
|---|---|---|
| task_success | 0/10(9 fail,1 error) | 2/10 partial |
| checkpoint | 45p/32f | 20p/10skip |
| Execution / Context | 0.587 / 0.40 | 0.70 / 0.90 |
| Tooling / Observability | n/a / 0.60 | 0.20 / n/a |

## 集群恢复后待办(顺序)

1. 确认 mctaC 的 tool_backend(本地 GPU vs api gateway),保证 native 对照同后端
2. 删旧 pbN/mctaN 残留(被否决launch的产物)
3. 定 CanonicalTrace schema(B)→ 抽 PB BenchmarkAdapter 样板 → MedCTA/HAB
4. A 能力矩阵逐 bench 填 + C conformance tests
5. native 指标并排 + failure taxonomy + integrity report
6. 干净数据上重评"指标解离"(主线 task #9)
