# Codex review — 11 issues, fix tracking

Codex 架构审计(11 条)逐条核实属实(抽查 5 条全中)。修复按"是否腐蚀分数"分级。

| # | 严重 | 问题 | 状态 |
|---|---|---|---|
| 6 | 🔴 P0 | `score_eligible` 默认 True(fail-open)→ 新 cp 忘设 False 静默进主分 | ✅ **已修**:`is_score_eligible` 默认翻 **False(fail-closed)**;native_pytest/toolset/jmespath/policy 显式 `score_eligible=True`。验证:缺 flag → 不计 |
| 3 | 🔴 P0 | null 维度无状态静默传播(mctaD 六维空之根因) | ✅ **已修**:`build_result` 加 `dimension_status`(valid_score / proxy_only / evaluation_error / not_exercised / not_applicable) |
| — | 🔴 P0 | RegionAttributeDescription 协议自相矛盾(盲 agent 被逼给 bbox + 静默整图 fallback + 扣分) | ✅ **已修**:bbox OR region_query 都合法;工具返显式 localization 状态(无静默);arg_match bbox/region 等价 + attribute 可选 |
| 2 | 🔴→🟠 | gateway HTTP 客户端复制 7 份、retry/timeout 不一致 | 🟡 **统一客户端已建**(`runner/gateway.py`:单一 key/retry/timeout/billing/结构化 error);判官迁移到它 = 机械跟进(P1) |
| 4 | 🟠 | `native_parsers.py` 死代码(0 import) | ✅ **已删除**(本轮:全仓 0 import 复核确认 = 死代码,`rm runner/native_parsers.py`;git 历史保留) |
| 5 | 🟠 | `canonical_observation` 定义未接线 | 🟡 **已诚实标注**(CLAUDE.md「已定义未接线」);接线 = P1 |
| 11 | 🟠 | `"error" in obs` 子串判错(误判 error_rate) | 🔜 待修:proxy_verifiers 已优先读 status 字段;子串 fallback 收紧 = P1(批次跑完动,proxy_verifiers 在用) |
| 7 | 🟠 | 判官独立性只事后记,不在 init 拦 | 🟡 部分:provenance + `non_independent_judge` 已记录;init 拦截(strict 拒/exploratory 标 score_eligible=False)= P1 |
| 8 | 🟠 | skip/error/environment_error 语义不一致 | 🔜 待修:分层 error taxonomy(not_evaluated / evaluation_failure / environment_failure)= P1 |
| 1 | 🟠 | `run_task()` 442 行 god-function + PB 脚手架塞进通用 runner | 🔴 **未做(P2)**:风险高,需把 deliverable nudge/budget/forced-write 抽到 PBPolicy、MedCTA/HAB hook;通用 runner 只留循环。dedicated 重构 pass |
| 9 | 🟡 | FHIR 工具名三跳映射 | 🔴 未做(P2):单一 canonical tool ID + adapter 一次映射 + trajectory 存 {canonical, native} |
| 10 | 🟡 | `_as_entries` 挂基类、`--source-benchmark` 死参、旧 `dissociation.py` | 🔜 卫生(P3):基类抽象、删死参/旧脚本 |

## 核判断(与 IMPLEMENTATION_STATUS 自评一致)

"结果聚合层统一了,但交互/环境适配层还没真正模块化":系统能跑,benchmark 特殊逻辑/观测格式/error 语义/协议转换仍散落。**最急的不是为漂亮重构,而是 P0 分数可信度——已修。**

## 修复顺序

- **P0(分数可信度)**:#6 score_eligible ✅ / #3 null ✅ / region 协议 ✅ → **已完成**
- **P1(兑现 canonical + 可靠性)**:gateway 迁移、observation 接线、error taxonomy、判官 init 拦截 → 进行中(gateway 客户端已建)
- **P2(清 runner)**:god-function 拆分、PB 逻辑抽离、adapter 正式接口、工具名单映射、conformance tests
- **P3(卫生)**:native_parsers、死参、旧脚本、子串判错
