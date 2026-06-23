# Medical Harness 架构(设计锁定稿 · 草稿)

> 状态:草稿,集群恢复后提交到 ce483 `docs/ARCHITECTURE.md`。
> 本文是设计纲领,所有跑批与改动以此为准。

## 0. 总管线

```
异构原生任务与协议
      ↓ adapters(双向:协议解析 + 观测渲染 + 错误映射)
统一任务 / 轨迹表示(CanonicalTask / CanonicalTrace,保留 raw/native 证据)
      ↓
7 维 ETCLOVG 评测(二大类:任务执行能力 / 可信治理)
      ↓
保留 native metrics(Pass@1 / GAcc / task-subtask) + 新增 harness metrics
```

## 2. Adapter 是双向契约(不只是 canonical→env)

现 `FhirEnv / ToolSandboxEnv / GuiEnvReal.act()` 已构成隐式 adapter,但完整 adapter 必须双向,否则只能统一"发动作",不能保证三类环境的 observation / 错误 / 状态变化具有统一可审计语义。

正式定义的 6 个转换(**同时保留 raw/native 内容,避免转换后丢证据**):

```
Raw Task          → CanonicalTask
Raw Observation   → CanonicalObservation
CanonicalAction   → Native Environment Action
Native Result     → CanonicalResult
Native Error      → CanonicalError
Native Evaluator  → CanonicalVerifierResult
```

### 统一接口

```python
class BenchmarkAdapter:
    def load_task(self, raw_task) -> CanonicalTask: ...
    def render_observation(self, env_state) -> CanonicalObservation: ...
    def parse_action(self, model_output) -> CanonicalAction: ...
    def execute_action(self, action) -> CanonicalToolResult: ...
    def native_metrics(self, trajectory) -> dict: ...
```

是否需要"每个 benchmark 的原文语法解析器":
- **Unified track**:不一定(canonical 接口即可)
- **Native replication track**:需要(原协议解析 / 渲染)

## 3. 现在真正需要证明的三件事

### A. 表达能力保留(expressiveness preservation)

逐 benchmark 建能力矩阵,检查每个原任务要求**是否至少有一条 canonical 轨迹能完成**:

| 原任务能力 | Canonical 表达 |
|---|---|
| PB 查询 FHIR | `tool_call(fhir_search, ...)` |
| PB 创建资源 | `tool_call(fhir_create, ...)` |
| PB 写交付物 | `tool_call(write_file, ...)` |
| MedCTA 区域分析 | `tool_call(region_description, ...)` |
| MedCTA OCR / 搜索 | 对应 canonical tool call |
| HAB 点击 / 输入 / 提交 | `gui_action(click/type/submit, ...)` |
| 任务完成 | `final` 或明确终止动作 |
| 工具失败 | canonical error |
| 无法完成 / 升级 | `abstain` / `escalate` 状态 |

### B. 契约正式化(最小字段集)

- action 类型与参数 schema
- observation 类型
- result / error 类型
- action-result 配对
- 终止状态:final / abort / timeout
- schema version
- N/A、unknown、skipped 的语义
- raw 与 canonical provenance

### C. Conformance tests(每个环境至少)

- 合法动作能执行
- 非法参数能被拒绝
- 返回值能稳定转成 canonical result
- 原生错误能映射到正确 canonical error
- canonical action 不丢关键参数
- 最小成功轨迹能完成一个代表任务

> 原则:adapter 正确性**不靠文档宣称,靠 conformance tests 证明**。

## 4. 评测层(锁定:7 维 ETCLOVG,二大类)

| 大类 | ETCLOVG 维度 | 问的问题 |
|---|---|---|
| 任务执行能力 | Execution / Tooling / Context / Lifecycle | 事做没做对 |
| 可信治理 | Observability / Verification / Governance | 能不能信任它 |

> 不收敛成 4 维。outcome/execution/safety/integrity 只是讲清每维在三个 bench 对应的物理量(failure 映射),不替换评分维度。

维度↔物理量映射(failure 映射,示意):

| 维度面 | PB | MedCTA | HAB |
|---|---|---|---|
| Outcome | checkpoint/task | GAcc/task | task/subtask |
| Execution | FHIR 调用 | 感知工具 | GUI action |
| Safety | 前置检查 | 不安全结论 | patient/scope/action |
| Workflow | 交付物 | 工具轨迹 | 页面流程 |
| Integrity | coverage | judge provenance | environment qualification |
