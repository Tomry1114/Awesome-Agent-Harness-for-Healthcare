# Canonical Contract(统一协议契约 · 正式化)

> 统一不是让 PB/MedCTA/HAB 长得一样,而是**统一"动作的语义结构"**,由 adapter 翻译成各环境真正执行的动作。canonical protocol 是**能力型接口**,不是扁平 tool_call。本契约是 ARCHITECTURE.md §2 的 B(契约正式化)落地。

```
模型 ── canonical action ──▶ Benchmark Adapter ── native action ──▶ FHIR / MedCTA sandbox / Playwright
模型 ◀─ canonical observation ── Adapter ◀── native result/error/state ──┘
```
Adapter 职责:参数验证 → 转换环境调用 → 执行 → 把原生结果/错误/状态变化转回 canonical observation。**兼容的是动作语义,不是字符串格式**。

## 1. CanonicalAction(5 类,能力型)

```
CanonicalAction ├── ToolCall ├── GUIAction ├── FileAction ├── ControlAction └── FinalAnswer
```

**ToolCall**(PB / MedCTA):`{action_type:"tool_call", name, arguments:{}}`
**GUIAction**(浏览器):`{action_type:"gui_action", operation, target}`;operation ∈ navigate/click/type/select/check/scroll/back/submit
**FileAction**(上传/下载/交付,**不再塞进 gui_action**):
```
{action_type:"file_action", operation:"upload", file_ref:"workspace/appeal.pdf", target:{element_id}}
{action_type:"file_action", operation:"download", target:{element_id}, destination:"workspace/"}
{action_type:"file_action", operation:"write", path:"output/report.md", content:"..."}   # 兼容 PB 交付物
```
PB 仍可把 write_file 当工具,但 canonical schema 中它属于 **artifact/file capability**。
**ControlAction**(生命周期,跨 benchmark 共用):`{action_type:"control_action", operation}`;operation ∈ done/abort/retry/wait/escalate
**FinalAnswer**(与 done **分开**):`{action_type:"final_answer", content}`
> "输出了一段答案" ≠ "完成了任务"。有的任务只需交付物、有的先写文件再完成、GUI 提交成功只需 done、MedCTA 需最终医学回答。

## 2. CanonicalTarget(可扩展对象,不固定成整数)

```
target: {element_id, role, name, text, selector, coordinates:[x,y]}
```
adapter 按优先级解析:`element_id → role/name → text → selector → coordinates`。
- 当前 `ref=N` → element_id;上游 `click([id])` → element_id;screenshot agent → coordinates;DOM agent → role/name。
- **统一协议不因 observation 模态变化而重写**。

## 3. CanonicalObservation(多模态容器)

```
{observation_type:"environment_state",
 modalities:{text, structured:{elements:[]}, image_ref},
 current_url, artifacts:[], previous_action_result:{}}
```
| Benchmark | observation 填充 |
|---|---|
| PB | FHIR JSON / Bundle / resource(structured) |
| MedCTA | image_ref + OCR/工具输出(text) |
| HAB unified-text | elements + visible text(structured+text) |
| HAB native screenshot | image_ref + marked element ids |
> 不要求同模态,要求通过**同一 envelope 明确声明"这一步模型究竟看到了什么"**。

## 4. CanonicalResult / CanonicalError

```
CanonicalResult ├── Success ├── Failure ├── StateChange ├── ArtifactProduced └── NoProgress
CanonicalError  ├── InvalidAction ├── ToolError ├── EnvironmentError ├── MissingCapability ├── Timeout └── InfrastructureError
```
**NoProgress 关键**:API success ≠ semantic progress。
```
{status:"success", state_changed:false, semantic_progress:false}   # 工具调用成功但页面无进展
```
解决 HAB"90 次调用 85 次没页面进展"被误记为成功的问题。

## 5. Capability manifest(解决功能差异)

每个环境声明能力;每个任务声明 required_capabilities;运行前检查 `task.required ⊆ env.capabilities`。
```
physicianbench: {tools:[fhir_search,fhir_read,fhir_create,fhir_update], file_operations:[write]}
healthadminbench: {gui_operations:[navigate,click,type,select,scroll,back,submit],
                   file_operations:[upload,download], observation_modalities:[text,structured_elements]}
```
缺能力时(如 HAB 要 upload 但 env 无):
```
qualification = environment_capability_missing
missing_capability = file.upload
```
该任务:**排除能力成绩 + 不算 agent fail + 进 integrity 报告**(`not_exercised_due_to_missing_capability`)。
> 不能让模型在不可能完成的环境里得 0。

## 6. Native 与 Unified 双轨共存

```
Native model output ─ native parser ─┐
                                     ├─ CanonicalAction → env → CanonicalTrace(同一套审计指标)
Unified model output ─ canonical parser┘
```
同一语义动作存两版:`{canonical_action:{...}, native_representation:"click([12])", adapter:"hab_native_v1"}`。
- **Unified**:canonical JSON → adapter → Playwright
- **Native**:`click([12])` → native parser → canonical event → Playwright
- **MedCTA ReAct**:`Thought/Action: OCR/Action Input` → native parser → `{action_type:tool_call, name:OCR}`,保留 `raw_model_output`。Unified track 不需精确 Lagent 模板;Native track 才需 ReAct parser + 运行循环。

## 7. Track / provenance 标注(诚信门)

PB 用原 SYSTEM_PROMPT + 额外 {tools} + canonical 协议 = **合理的 Unified**,但必须标:
```
prompt_semantics = native_clinical_guidance
protocol_layer  = medical_harness_canonical
track           = unified         # 不能标 exact native
```
Native PB 才需:官方 prompt + 官方工具暴露 + 官方 function-calling/runtime + 官方终止规则。

## 8. 当前实现差距(对照本契约)

| 件 | 契约要求 | 现状 |
|---|---|---|
| 5 动作类型 | ToolCall/GUIAction/FileAction/ControlAction/FinalAnswer | 🔴 现仅 tool_call + gui_action(扁平);final 用 <answer> |
| GUIAction operations | +check/scroll/back | 🟡 已补 back/scroll/done,缺 check |
| FileAction | upload/download/write 正式列出 | 🔴 write 有(write_file);**upload/download 缺**(HAB download 41%) |
| ControlAction | done/abort/retry/wait/escalate | 🔴 仅隐式 |
| CanonicalTarget 可扩展 | element_id/role/name/text/coords | 🔴 现仅 ref=N |
| CanonicalObservation envelope | 多模态容器 | 🟡 各 env 各自渲染,未统一 envelope |
| NoProgress / state_changed | API success≠progress | 🔴 未区分 |
| Capability manifest + precheck | required ⊆ env | 🔴 未实现(本轮起步) |

## 9. HAB download/upload 行动项(§9)

HAB 任务 capability 统计(regex 粗估):download ~41%、fax ~13%。属 canonical 表达能力缺口,补齐前不能声称 HAB 覆盖完整。顺序:
1. 精确统计 HAB 各任务 required capability(去 "submit" 误匹配)
2. 实现 Playwright file chooser/upload + download event + workspace 路径 + artifact provenance
3. conformance test(upload:canonical→file chooser→页面显示文件→trace 记 hash/path;download:canonical→等 download event→存 workspace→trace 记 hash/path)
4. 恢复这些任务进有效集合
暂不补则标 `not_exercised_due_to_missing_capability`(非模型失败)。
