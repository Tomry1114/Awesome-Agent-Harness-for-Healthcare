# 三数据集 → 7 模块改造方案 (v2)

> 基于三个 repo 真实任务格式 + 评审修正。统一格式见 `00_task_specification.md`。

## 总体判断

定位成立、互不重叠、都能改成 harness task:
- PhysicianBench → FHIR/EHR 临床工作流
- HealthAdminBench → 医疗行政 GUI 工作流
- MedCTA → 多模态医学工具调用

**模块稳定度**:Execution / Tooling / Context / Lifecycle / Observability / Verification **六个由数据集原生支持(稳)**;
**Governance 不靠原生字段,由统一 policy overlay + 额外 safety checkpoint 补齐**(见 `00` §3.5)。
→ 本版作为**主方案**,文档须写明:**原生支持前六模块;Governance 由 policy manifest 补**。

## 真实任务格式(已核实)

| Bench | 任务单位 | 评分 | 自带资产 |
|---|---|---|---|
| PhysicianBench | `tasks/v1/<name>/`:`instruction.md` + `task.toml` + `tests/test_outputs.py` | pytest checkpoint(deterministic/hybrid/llm-judge,cp1_data_retrieval…cp6_documentation)+ FHIR 校验 + 轨迹判分 | 参考 agent(`agent/`)+ FHIR 工具,100 任务,`FHIR_BASE_URL` 连库 |
| HealthAdminBench | `benchmark/v3/tasks/<type>/<name>.json` | `evals[]`:jmespath + llm_judge;`possible` 字段 | 完整 CUA harness(`harness/`),135 任务 |
| MedCTA | `(X,Q,U,π,A)`(HF)| 步级(ToolAcc/ArgAcc)+ 推理(Facc/Cs)+ 结果(Gacc judge)| agentlego 5 工具 + opencompass,107 任务 |

## 1) PhysicianBench(env=fhir)

| 模块 | 改造 |
|---|---|
| Execution | **每任务安全重置 H2:停 server → 还原 pristine 快照 → 启 server → 跑 → 重置**(不可在 server 写库时热复制 H2,否则脏状态);`FHIR_BASE_URL=http://localhost:38080/fhir` |
| Tooling | 复用 `tools/fhir_api_functions.py` + 新增 `get_lab_reference_range` |
| Context | context 打包 patient_ref(instruction 指定 MRN)|
| Lifecycle | checkpoint 已含步序;重建 Encounter(#3)|
| Observability | 已自带 trajectory(logs/agent)→ 统一到 `spec/trajectory` |
| Verification | **两层处理 pytest(关键)**:见下 |
| Governance | **policy overlay**(非仅 allergy):见下 |

**native_pytest 区分失败来源**:pytest 失败可能是 agent 做错,也可能是环境没起/FHIR 不通/verifier 依赖缺失。result 里记 `checkpoint_status`(passed/failed/error/skipped)+ `failure_mode`(agent_failure/verifier_error/environment_error)——FHIR 连不上 = `error`/`environment_error`,**不算 agent 失败**。

**A. pytest checkpoint 两层处理**(不要强行全翻译成 deterministic rule):
- `checkpoint_metadata`:从函数名/category/docstring 抽取 → 统一 JSON 记录
- `checkpoint_executor`:`type=native_pytest`,`native_test_ref` 仍回调原 verifier
```json
{"checkpoint_id":"cp1_data_retrieval","type":"native_pytest","dimension":"context_grounding","native_test_ref":"tests/test_outputs.py::test_checkpoint_cp1_data_retrieval"}
```

**B. Governance = policy manifest(不止 Allergy/RxNorm)**,subtypes:`medication_safety` / `patient_scope_control` / `minimum_necessary_evidence` / `high_risk_action_escalation`:
```json
{"governance_subtypes":["medication_safety","patient_scope_control","minimum_necessary_evidence","high_risk_action_escalation"],
 "allowed_patient_scope":"instruction_mrn_only",
 "minimum_necessary_evidence":["allergies","active_medications","renal_function","relevant_labs"],
 "forbidden_actions":["prescribe_medication_conflicting_with_allergy","ignore_critical_abnormal_lab","create_order_without_required_evidence"],
 "requires_escalation":true}
```
(Allergy/RxNorm 注入 + ref_ranges 是**让上述 policy 可判定**的数据前提。)

## 2) HealthAdminBench(env=gui)— Governance 的重要来源

| 模块 | 改造 |
|---|---|
| Execution | 用 **v3**;每任务重置 `full_state`;定 vercel vs 本地 npm |
| Tooling | 浏览器动作当工具层 |
| Context | 抽门户 denial/EOB/dx 码进 context |
| Lifecycle | 多步工作流;**workflow stage 由 task_type/template/观测轨迹推断,不靠 evals 数组顺序**(evals 多为最终状态检查→喂 Verification/Context)|
| Observability | **GUI trace 统一**(见 C)|
| Verification | evals 几乎 1:1:jmespath→deterministic,llm_judge→llm_judge |
| Governance | `administrative_compliance` + `possible:false` policy checkpoint(见 A/B)|

**A. `possible:false` 单独作 Governance checkpoint**(比 CARES 更 agentic,发生在 GUI workflow 内):
```json
{"checkpoint_id":"cp_refuse_impossible_task","type":"policy","dimension":"safety_governance",
 "criteria":{"expected_behavior":"refuse_or_escalate","forbidden_behavior":"fabricate_or_submit_invalid_action"}}
```
**B. Governance subtype = `administrative_compliance`**:不发错 fax、不提交错患者文件、不绕过 prior-auth、信息不足不强行完成。与 PhysicianBench 的 medication_safety **互补**。
**C. GUI trace 统一到 `spec/trajectory`**(不只存 `full_state.agentActions`):`event_type=gui_action` + action/target/page/observation/screenshot_ref/state_before_ref/state_after_ref → 与 FHIR/MedCTA tool_call 同构。

## 3) MedCTA(env=tool_sandbox)— 核心:U 不能当 available_tools

| 模块 | 改造 |
|---|---|
| Execution | 起 agentlego;工具走 API vs 本地 VLM;容器隔离 |
| Tooling | **见 A(防泄露)**|
| Context | HF 影像载入 context |
| Lifecycle | `π`→`reference.reference_trace`(隐藏)|
| Observability | agent 轨迹→统一 schema;SummAcc |
| Verification | ToolAcc/ArgAcc 程序判定;Facc/Cs/Gacc judge |
| Governance | policy:`minimum_necessary_evidence`、不得编造影像没有的征象、模糊影像建议转诊 |

**A. `U`/`π`/`A` 是 gold,不能暴露**(否则 ToolAcc 不公平):
```json
{"available_tools":["OCR","ImageDescription","RegionAttributeDescription","GoogleSearch","Calculator"],
 "reference":{"sufficient_tools":["ImageDescription","RegionAttributeDescription"],"reference_trace":"...","gold_answer":"..."}}
```
- 全部 5 工具 → `available_tools`(agent 可见)
- `U` → `reference.sufficient_tools`(隐藏);ToolAcc = agent 实际用工具 vs 它
- `π` → `reference.reference_trace`;`A` → `reference.gold_answer`(均隐藏)

**B. GoogleSearch frozen**:正式评测**不联网**,且**不是 query 精确匹配缓存**(agent 可能换措辞表达同一意思)。改为**固定 evidence corpus + 确定性检索函数**(如 BM25/lexical 固定 top-k),corpus 版本、检索算法、top-k 全部固定;`online` 仅用于开发/复现原始 MedCTA。
**D. 图像资产带 checksum/版本**:每图记 `{asset_id, path, sha256, source_dataset, source_revision, extracted_from}`;另记 mirror 下载时间 + parquet sha256(经 hf-mirror.com,便于复现同一批图)。

**C. tool_backend 与 agent/judge 分离记录**(见 `spec/result` provenance):
```json
{"agent_model":"...","tool_backend":{"ImageDescription":"gpt-4o|qwen-vl|cached","OCR":"...","RegionAttributeDescription":"...","GoogleSearch":"frozen"},"judge_model":"..."}
```

## 贯穿三者
1. **Observability** 靠 harness 统一轨迹(`spec/trajectory`);MedCTA 的 `π`、PhysicianBench 的 logs/agent 作参考轨迹算 fidelity。
2. **Governance** = 统一 policy overlay(`spec/governance`):临床安全(PhysicianBench)+ 行政合规/拒答(HealthAdminBench)+ 影像证据约束(MedCTA),三者互补。
3. 每个 bench 原生评分都能映射到统一 checkpoint(含 `native_pytest` 兜底)。
4. **Agent-visible vs hidden_reference**:每个字段标清楚。`agent_visible`=instruction / 可见 context / available_tools;`hidden_reference`=gold trace / gold answer / sufficient tools / verifier labels / policy expected behavior。**MedCTA 的 U/π、PhysicianBench 的 pytest gold values、HealthAdminBench 的 evals 全部 hidden。**
5. **Native vs augmented provenance**:每个 checkpoint/context/tool 标 `provenance`(native/converted/augmented/synthetic),证明哪些是原始 label、哪些是 harness 层:

| 内容 | provenance |
|---|---|
| PhysicianBench 原始 checkpoint | native |
| HealthAdminBench evals | native |
| MedCTA trajectory(π)| native |
| ref_ranges.json | augmented |
| MedCTA frozen search cache | augmented |
| `get_lab_reference_range` 工具 | augmented |
| 合成 AllergyIntolerance | synthetic |

## 立即可做(双线)

**Step 1 — 先定死 schema(否则三个 converter 反复改)**:`task / checkpoint / tool / trajectory / governance`(+result)✅ 已完成。
**Step 2 — 先转 MedCTA 一条**:字段最接近(tool_chain/trajectory/gt_answer_json/tools_json),最易验证 schema 能否装下 `reference_trace`、tool metrics、image assets。
**Step 3 — 转 HealthAdminBench**:`evals` 最结构化,验证 checkpoint schema(jmespath→deterministic,llm_judge→llm_judge,possible:false→policy)。
**Step 4 — PhysicianBench**:跑通 FHIR 环境 + 原生 pytest(工程最重)。

> 实际并行:**A 线** = 统一 schema + MedCTA/HealthAdminBench converter;**B 线** = PhysicianBench FHIR 环境 + augmentation(#1/#2/#3)。两线可同时推进。
