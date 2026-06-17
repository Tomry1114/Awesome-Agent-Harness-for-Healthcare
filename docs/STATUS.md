# Medical Harness — 项目状态总览 (STATUS)

> **单一进度入口**。最后更新:2026-06-17 · 主机:HPC ce483 `~/Medical_harness/` · 规范:Task Spec v2
> 三数据集组合:**PhysicianBench(FHIR 临床)+ HealthAdminBench(GUI 行政)+ MedCTA(多模态影像)**,全 agentic、不重叠、合起来覆盖 7 个 ETCLOVG 模块。

## 0. 文档地图

| 主题 | 文件 |
|---|---|
| 本总览 | `docs/STATUS.md` |
| 统一任务规范(格式/checkpoint/policy/轨迹/结果)| `docs/00_task_specification.md` |
| Lab 参考范围分析 + 建表 | `docs/01_lab_reference_range_analysis.md` |
| 三数据集 → 7 模块改造方案 | `docs/02_dataset_transformation_plan.md` |
| 校验结果固化 | `docs/VALIDATION_REPORT.md` |
| 已知(预期)warning | `docs/KNOWN_WARNINGS.md` |
| 机器可读 schema(6)| `spec/{task,checkpoint,tool,trajectory,governance,result}.schema.json` |
| 任务清单/版本/校验和 | `TASK_MANIFEST.json`(根)|
| 各 bench 处理 | `benchmark_dataprocess/<bench>/README.md` |

## 1. 进度看板

### ✅ 已完成
- **目录规范**:`benchmark/`(只读原始)/ `benchmark_dataprocess/`(处理)/ `spec/` / `docs/`
- **环境**:FHIR server 跑通(HAPI 8.8.0,`:38080`,108 患者)— 启动 `benchmark_dataprocess/PhysicianBench/run_fhir.sh`
- **6 个 schema 定死**(task/checkpoint/tool/trajectory/governance/result),含 dimension=7 模块 + subdimension、可见性、provenance、policy_source
- **3 个 converter + 342 统一任务**:MedCTA 107 · HealthAdminBench 135 · PhysicianBench 100 → `benchmark_dataprocess/<bench>/tasks_unified.jsonl`
- **双校验**:JSON-schema(`validate_tasks.py`)+ 语义(`semantic_validate_tasks.py`)→ **0 error / 0 unexpected**(37 expected_warnings 已归类)
- **固化文件**:`VALIDATION_REPORT.md` · `KNOWN_WARNINGS.md` · `TASK_MANIFEST.json`
- **增强 #1**:`ref_ranges.json`(49 条/43 LOINC)+ `lab_ref.py`(`get_lab_reference_range`+`classify`+单位硬检查),活 FHIR 实测覆盖 64%、异常检出正确
- **增强 #2(全量 26 medication 任务,已合并进 unified)**:`augmentation/` = build_augmentation(自动选 26 任务+确定性分配过敏原)、synthetic_allergies(26)、rxnorm_mapping(RxNav 19/19)、drug_safety_rules、allergy_bundle(幂等 PUT,确定性 id)、5 个 verifier(drug_safety_check,RxNorm **frozen**)、merge_governance(可重入)、restore_pristine_h2(恢复脚本)。FHIR 已注入 **26 条 AllergyIntolerance**(tag=synthetic-augmentation,幂等可回滚);**104 个 governance checkpoint 已并入 `PhysicianBench/tasks_unified.jsonl`**(26×4:allergy_exists/patient_scope/checked_before_med/no_conflict)。semantic +8 项 governance 审计全过(0 error)。

- **Unified harness runner**(`runner/`):load task→env adapter→agent loop→统一 trajectory→scorer→result JSON(schema-valid)。`environments.py`(FhirEnv 真实,Gui/ToolSandbox stub)+ `agents.py`(StubAgent,无 key)+ `scoring.py`(7 模块聚合)。
- **PhysicianBench 线端到端(skeleton,已加固)**:`native_pytest.py` 执行器(subprocess pytest `<node>`;分类 passed/agent_failure/environment_error/verifier_error)+ JOB_DIR(workspace/output + logs/agent/trajectory.log)+ skip_reason(unsupported_in_skeleton/missing_judge_backend/missing_native_verifier/disabled_by_config)。result 加固:success 不含 skipped + `evaluation_status` + `dimension_coverage` + **加权** dimension_scores + governance **具体 failure_tag**(unsafe_action/cross_patient_access/…)。StubAgent 反应式选安全药 + 写 deliverable + tag,**每任务自动清理 stub 资源(FHIR 不污染)**。`run_batch.py`:`--governance-only`/`--has-dimension` 过滤 + **per-task bundle**(task/trajectory/result/workspace)+ `summary.json`(success_buckets:complete/partial/failed/error)。验证:governance-only batch 8→Governance mean 0.975、schema 8/8、stub 资源清理为 0。
  - **audit 结论**:PB native_pytest 两类——无-LLM(cp1 trajectory / cp5 FHIR order)现可评;llm_judge 类(cp2/3/4/6 读 deliverable + judge)**需 LLM API key**。
- **Medication safety 扩展(医嘱→推荐→文书)**:`no_allergy_conflicting_medication_{created,recommended,documented}` 三个 verifier;recommended/documented 用**推荐动词上下文**匹配(纯报告过敏不算冲突,假阳性已修);merge 后每 action 任务 6 个 governance cp(100/100 schema、0 semantic error)。验证:recommend 过敏药→fail、报告过敏→pass、创建过敏药医嘱→fail。
- **增强 #3 Encounter 外部索引**:`augmentation/build_encounter_index.py` → `encounter_index.json`(100 患者 / **10,980 encounters**,按日聚类,linked_resources,`provenance=augmented`,**未注入 FHIR**)。强化 Lifecycle/Context。
- **MedCTA ToolSandboxEnv v0(第三条 execution substrate)**:`ToolSandboxEnv`(replay 缓存输出)+ `ReplayAgent`(重放 π 参考轨迹,gold-replay 验证 agent,非真实 agent)+ scorer:`toolset_match`(ToolAcc)/`arg_match`(ArgAcc,新增 cp_arg_accuracy)/ llm_judge **离线 whitelist**(Gacc 代理)。验证:gold replay → ToolAcc/ArgAcc/Gacc 全 pass;判别力 OK(错工具/错答案→fail);batch 5 → Tooling 1.0/Execution 0.8(1 个 Gacc 离线未命中=措辞差异,真实 Gacc 需 judge)。grounding/no_fabrication skipped(judge/verifier 待接)。**三条 substrate(FHIR/GUI/tool_sandbox)skeleton 端到端齐了。**
- **HealthAdminBench GuiEnv v0(第二条 execution substrate)**:`GuiEnv`(mock portal,browser 动作 navigate/click/type/select/upload/submit/snapshot 改写内存 `full_state`)+ `StubGuiAgent` + **JMESPath deterministic scorer**(`jmespath.search` 对 full_state)。验收 7/7:HAB 单任务/batch5 jmespath 真实执行(failed,pass 已演示)、llm_judge+criteria-policy skipped、schema 5/5、bundle 齐全、跨环境跑通。**证明 unified runner 不只 FHIR 特化**(FHIR substrate + GUI substrate 同一 runner/scoring/schema)。v1=真 Playwright 驱动门户。

### 🟡 进行中 / 下一步
**推荐下一项**:**真实 LLM agent**(替换 StubAgent,用 upstream 语义 FHIR 工具 → native_pytest 可 pass,**需模型 API key**)+ `llm_judge`。或先做 **GuiEnv/ToolSandboxEnv 真实化** 或 **增强 #3 Encounter**(均无需 key)。详见 `runner/README.md` TODO。
> 目标:让 medication_safety governance policy 从"有数据+已并入"走到"端到端可判分"。
> ⚠️ 运维教训:本节点 `pkill -f` 不可靠 → 必须按 PID 杀(run_fhir.sh 已修);**严禁热复制 H2**(server 运行时 cp 会致 DB closed),重置用 `augmentation/restore_pristine_h2.sh`。

### ⬜ 待办
| 项 | 解锁 | 需要 |
|---|---|---|
| ~~增强 #2~~ ✅ 已完成(全量 26 任务,104 governance cp 已并入 unified,注入+审计通过)| Governance(临床安全/禁忌药)| 完成 |
| **B 线**:PhysicianBench 接 `FHIR_BASE_URL` + 参考 agent 跑通 native_pytest | 端到端闭环(任务→agent→checkpoint→判分)| **模型 API key** + FHIR 安全重置(停→还原 pristine→启)|
| 增强 #3(可选):从时间戳重建 Encounter | Lifecycle(就诊级流程)| 无 |
| MedCTA 工具后端 + frozen GoogleSearch corpus + judge | MedCTA 端到端 | 模型/VLM 或 API |
| 各 bench `tasks_unified` 接入统一 harness 运行器 | 全量跑分 | harness runner(待建)|
| policy 临床/行政专家复核(`review_status: pending`→reviewed)| Governance 可信度 | 医生/行政专家 |

## 2. 7 模块覆盖(当前)

| 模块 | 主来源 | 状态 |
|---|---|---|
| Execution | 三个环境(FHIR/GUI/tool-sandbox)| FHIR 真实 + GUI(mock portal v0)跑通;tool-sandbox 待做 |
| Tooling | FHIR 工具 / GUI 动作 / MedCTA 5 工具 | FHIR 真实;GUI 动作;MedCTA ToolAcc+ArgAcc(replay)跑通 |
| Context | EHR / portal / 影像 | 规范就绪 |
| Lifecycle | 多步工作流(由 task_type/轨迹推断,非 eval 顺序)| 规范就绪;+ encounter_index(就诊级)增强 |
| Observability | 统一 trajectory schema(tool_call+gui_action 同构)| ✅ runner 已记录统一轨迹(FhirEnv) |
| Verification | deterministic/native_pytest + **ref_ranges 异常 lab(✅ 增强#1)** | ✅ PB native_pytest 执行器跑通(区分 agent/verifier/environment 失败);llm_judge 待 judge 模型 |
| Governance | 统一 policy overlay(临床安全=PB+#2 / 行政合规=HAB / 影像证据=MedCTA)| ✅ #2 全量:26 任务注入 allergy + 104 governance cp 并入 unified + 5 verifier(frozen)+ 8 项审计过;余:B 线端到端跑分 + 专家复核(review_status=pending)|

## 3. 复现 / 校验

```bash
# 启 FHIR
~/Medical_harness/benchmark_dataprocess/PhysicianBench/run_fhir.sh
# 转换 + 双校验(详见 VALIDATION_REPORT.md)
cd ~/Medical_harness/benchmark_dataprocess && ...   # 见 docs/VALIDATION_REPORT.md 复现命令
# lab 参考范围自测
cd PhysicianBench && python3 lab_ref.py
```

## 4. 关键约束(详见 docs/00)
- gold 不泄露:`reference`(U/π/A、gold)对 agent 隐藏;`available_tools` 是全集
- dimension=7 模块 enum;subdimension=细分;两者不混
- Governance=统一 policy overlay(可由 native/converted/augmented/synthetic instantiate)
- native_pytest 区分 agent/verifier/environment 失败
- agent_model ≠ tool_backend ≠ judge_model(分开记录)
