# Medical Harness — 项目状态总览 (STATUS)

> **单一进度入口**。最后更新:2026-06-18 · 主机:HPC ce483 `~/Medical_harness/` · 规范:Task Spec v2
> **运行环境:`medicalharness`**(conda,Python 3.10;`~/.conda/envs/medicalharness/bin/python`)——与 AgentOCR 解耦,代码一律用此环境。GPU 经 `sbatch -p debug --gres=gpu:1`(A40)。
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
| 指标体系设计 + 指标代码 | `benchmark_metric/README.md` |
| Action-level 安全规范(正式)| `benchmark_metric/SAFETY_SPEC_v1.md` |
| Action-level 安全实现 | `benchmark_metric/{risk_annotator,fhir_scope,safety_metrics}.py` + `test_safety.py`(14 单测)|
| llm_judge 后端(本地 Qwen 判官)| `runner/judge_backend.py`(MH_JUDGE=qwen 开启)|

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

### ✅ v1 真实环境(2026-06-18 新增)
- **专用 env `medicalharness`**(8.9G):torch 2.8.0+cu128 / transformers 4.57.3 / playwright 1.48 / jmespath / jsonschema / pytest / pandas / pyarrow。imports 全 OK。
- **MedCTA ToolSandboxEnv v1(真实工具后端)**:`runner/vlm_backend.py`(可插拔 Qwen3-VL 单例,默认本地 `~/hf_models/Qwen3-VL-2B-Instruct`,`MH_VLM_BACKEND=local`)+ `runner/tools_medcta.py`(5 真实工具:ImageDescription/RegionAttributeDescription/OCR→VLM、Calculator 安全 AST、GoogleSearch 离线 frozen 语料)+ `ToolSandboxEnv` real/replay 双模(`MH_TOOL_MODE=real`)。**验证**:Qwen3-VL-2B 在 A40 ~6s 加载/~8s 每图出真实医学描述;集成测试(sbatch,replay agent + 真实工具)success / Tooling=1.0 / schema OK,工具真在 GPU 上执行。
- **HealthAdminBench GuiEnv v1(真实门户)**:Node 22 + v2 Next.js 16 门户(`benchmark/HealthAdminBench/benchmark/v2/portals`,`:3002`)+ Playwright 1.48 + chromium 130(npmmirror 镜像;azureedge 被墙)+ `GuiEnvReal`(`environments.py`):真 chromium 驱动真门户,`full_state` 从 `localStorage.portals_state.emr` 读出(即 jmespath checkpoint 的评分对象)。`MH_GUI_MODE=mock` 回退 v0。**验证**:HAB 任务端到端,真 DOM 操作→真 full_state→jmespath 正确评分(stub 正确判 fail,schema OK)。
- **关键设计发现**:原始 MedCTA agent 是**纯文本 LLM**——用户问题**不含图**,agent 必须**调 ImageDescription/RegionAttributeDescription 工具才能"看到"图**(reference_trace 实证:user 仅文本→assistant 调 ImageDescription→tool 返回描述→调 RegionAttributeDescription→final)。决定真实 agent 的 I/O:不直接喂图,逼其用工具,工具增益才可测。

### 🟡 进行中 / 下一步
**真实 Qwen3-VL tool-calling agent 已建成且环境感知**(`runner/qwen_agent.py`)——**三数据集全部跑通**:MedCTA(3) + PhysicianBench(1) + HealthAdminBench(2,GUI)。
> GUI(HAB-denial-easy-1,真实 agent,2B):观测管线已建(`data-mh-ref` 枚举 77 个可交互元素 + 页面文本注入),chromium 在 GPU 节点正常、门户经 login IP(10.120.31.247:3002)可达、`full_state` 经 jmespath 评分。但 2B **未按 `<tool_call>` 协议输出**(吐裸文本)→ step0 即 final,所有 cp failed(status=partial)。**[更正]** 上一行此前误判为数据不对齐——实为观测截断假象:门户本就有 DEN-001/Martinez（denials-worklist-row-DEN-001 存在,共 40 行）。已用 gold 路径证明该 GUI 任务有可达 ground truth:navigate /emr/denied/DEN-001→点 remittance tab→选 disposition 'Route to Clinical Appeals'→填 triage note→submit,经真实 GuiEnvReal+真门户+真 scorer → success=True,4 个确定性 cp(viewedDenialDetails/viewedRemittanceImage/selectedDisposition/documentedAppealInEpic)全 PASS(Context/Lifecycle/Observability=1.0;llm_judge/policy 因无后端 skip→status=partial)。对齐审计:门户 50 个 DEN,57 个 HAB 任务引用 DEN id,0 缺失 → denial 类任务全对齐。**结论:三条 substrate 真实 agent 端到端全通,2B 太弱是共同瓶颈,应上 4B/7B。**
runner 已加健壮性(工具错误→observation 不崩溃)——纯文本 brain,看到问题后用 Qwen3-VL function-calling 自行决定调哪个工具(ImageDescription/RegionAttribute/OCR/Search/Calculator),工具真实执行后综合作答 → 产出**真实 acc + ToolAcc/ArgAcc**。同思路复用到 GUI(读 axtree→发 DOM 动作)与 PB(语义 FHIR 工具)。**全本地 Qwen3-VL,不用 API key。**
> 注:replay agent 的 cp_outcome 通过 = 回放 gold 答案,**非模型真实正确性**;真实 acc 必须由真实 agent 自答产生。已实测(真实 agent,MCTA-0,2B):自主调 ImageDescription/RegionAttribute/Search,但误读为"胰腺病灶/无血栓" → cp_outcome failed(real acc=0)、ToolAcc passed。2B 偏弱+循环 → 应上 4B/7B。
> **MedCTA 失败画像 + B 协议修复(2026-06-18,MCTA-0..9,真实 2B)**:确定性多标签分类把失败分为「可工程修」vs「模型能力」。B 修复(qwen_agent.py:花括号配平解析器替脆正则 + 感知工具剥 `image` 参数 + tool_sandbox 提示词「勿传 image/一次一 action/多步 RegionAttribute grounding」)→ **协议噪声定向清零:tool_argument_error 4→0、final_answer_format_error 1→0**。残余为纯模型能力:image_misread 6→8(MCTA-0 由「解析截断空答」正确重归类为「真感知后读错」)、underuse_vs_ref 8→9、loop 1(MCTA-7)、proxy 命中 2→1。tool_selection_error / search_misuse 始终 0。**结论:协议层已干净,瓶颈=感知+grounding 纪律 → 转 option A(4B/7B)。** 产物:medcta_profile.py(可复用)、medcta_profile.json(后)、medcta_profile.preB.json(前)、qwen_agent.py.bak_preB。
> **评审复修(2026-06-18,4 文件 + Δ 重跑 job 9887269)**:B 修复后评审实测发现 1 回归+2 旧未修,全收。**新-1/新-2(解析器回归)**:`<answer>{...JSON...}</answer>` 被误当 tool_call 吞答案——改为仅当出现 `<tool_call>` 标签且 name 非空才走工具,否则走 answer(6 用例实测)。**#1(GUI 默认崩)**:`make_env` 默认改 `GuiEnvMock`(login 节点无可启动 chromium),真实门户改显式 `MH_GUI_MODE=real`(HAB sbatch 已设)。**#3(RegionAttribute 假接地)**:bbox/attribute 分开传 + 数值 bbox **真裁剪 PIL 图**再喂 VLM(像素级接地),自由文本区域退化为 focus 并注明未裁剪。**重跑验证无回归**:final_answer_format_error 仍 0/10、tool_selection 0/10;#3 生效证据=MCTA-5 从单步答变为切上/下半真区域接地(rep 2→4、10 次调用),但 2B 循环+占位符 attribute→loop 1→2、tool_argument_error 1/10(均为模型能力,非 harness bug)→ 印证 option A。待办(评审#2/5 低):ArgAcc 全等过脆、dtype/torch_dtype。
> 目标:让 medication_safety governance policy 从"有数据+已并入"走到"端到端可判分"。
> ⚠️ 运维教训:本节点 `pkill -f` 不可靠 → 必须按 PID 杀(run_fhir.sh 已修);**严禁热复制 H2**(server 运行时 cp 会致 DB closed),重置用 `augmentation/restore_pristine_h2.sh`。

### ⬜ 待办
| 项 | 解锁 | 需要 |
|---|---|---|
| ~~增强 #2~~ ✅ 已完成(全量 26 任务,104 governance cp 已并入 unified,注入+审计通过)| Governance(临床安全/禁忌药)| 完成 |
| **B 线**:三 bench 接真实 agent 跑通(PB native_pytest / GUI / MedCTA)| 端到端闭环 + 真实 acc | **真实 agent(可用本地 Qwen3-VL,无需 API)** + FHIR 安全重置 |
| 增强 #3(可选):从时间戳重建 Encounter | Lifecycle(就诊级流程)| 无 |
| ~~MedCTA 工具后端 + frozen GoogleSearch~~ ✅ **v1 真实后端**(本地 Qwen3-VL,无 key)| MedCTA 端到端 | 余:真实 tool-calling agent |
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


## 设计不变量(口径)

> 评审校准后的精确表述,后续描述以此为准。

- **角色分离(脑/手/裁判)**:`Agent` 只输出 action intent（act→tool_call|final）。真实工具执行属 **tool-backend/环境层**(FHIR HTTP · Playwright DOM · `tools_medcta.py` · `vlm_backend.py`)。provenance 必分记:`agent_model`(脑) · `tool_backend_model`(手内部模型,如图像工具的 VLM) · `judge_model`。**同为 Qwen3-VL 也要按角色分开**,否则会被误读为「agent 直接看图」。
- **两类差异、两处收敛**:执行差异 → `EnvironmentAdapter`;评价差异 → `checkpoint.method` 的 scorer dispatch(native_pytest/jmespath/llm_judge/policy)。runner 主循环对两者无感知。
- **qualification 规则**:降级只看 mock_env / replay_tool_backend / outcome_proxy / uses_hidden_reference / scorer_validation_only / proxy_scored_checkpoints,**不按 substrate**。真实 GUI / 真实 ToolSandbox 不天然降级。
- **gold/replay success 的边界**:只算 `scorer_validation_success`(env 接线 OK + scorer 通路 OK + reference 可复现),标 `uses_hidden_reference / scorer_validation_only`,**不进真实 agent baseline aggregate**。
- **隐藏状态边界(GUI)**:`full_state`(`localStorage.portals_state.emr`)是 **scorer-only hidden state**,仅进 scorer ctx;agent 只能看页面文本 + `data-mh-ref` 可交互元素列表。`data-mh-ref` 是观测层临时定位辅助,不改业务状态,每步重生成 ref map 防 stale。
- **MedCTA acc 口径**:真实 agent 轨迹 + ToolAcc/ArgAcc(真) + **proxy outcome**;接真实 Gacc judge/人工评审前不称正式 acc/Gacc。
- **benchmark vs harness**:当前 = evaluation harness / benchmark runner(含 policy overlay);下一步 = active runtime intervention harness,用同一 runner 做 with/without A/B Δ。
