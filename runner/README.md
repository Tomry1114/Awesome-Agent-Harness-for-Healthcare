# Unified Harness Runner (skeleton)

把统一任务跑成结果的骨架:**load unified task → environment adapter → agent loop → 统一 trajectory → scorer → result JSON**(对 `spec/result.schema.json` 校验)。

## 运行(无需 API key,stub agent)
```bash
# 需先启 FHIR: benchmark_dataprocess/PhysicianBench/run_fhir.sh
python3 runner/run.py --bench PhysicianBench --task PB-chronic_urticaria_allergist --out /tmp/r.json
```
输出 7 模块 `dimension_scores` + 每 checkpoint 状态 + schema 校验结果。

## 结构
| 文件 | 职责 |
|---|---|
| `run.py` | CLI:加载任务→跑 agent loop→评分→构建+校验 result |
| `environments.py` | 环境适配器(按 `environment.type`):**FhirEnv 真实**(查/读/建/lab_ref/写文件);**GuiEnv v0**(HAB:mock portal,navigate/click/type/select/upload/submit/snapshot 改写内存 `full_state`);ToolSandboxEnv stub(MedCTA TODO)|
| `agents.py` | `AgentBase.act(state)`;**StubAgent**(FHIR:查过敏→选安全药→写 deliverable)/**StubGuiAgent**(GUI:navigate→view→document→submit);均 regression-test agent,非 baseline |
| `scoring.py` | checkpoint 按 type 分派;policy→桥接 `augmentation/drug_safety_check.py` 的 governance verifier;聚合 7 模块;构建 result |

## 批量 + 过滤 + bundle
```bash
# 只跑 26 个 governance(medication)任务,产 per-task bundle + summary
python3 runner/run_batch.py --bench PhysicianBench --governance-only --limit 26 --out results/
```
过滤(B):`--governance-only` / `--has-dimension Governance` / `--has-subdimension safety_governance`。
**per-task bundle(A,Harness-Bench 风格)**:
```
results/<agent>/<task_id>/
├── task.json          # 统一任务
├── trajectory.jsonl   # 统一轨迹(逐事件)
├── result.json        # 评分结果
└── workspace/output/  # agent 产出(如 deliverable note)
results/<agent>/summary.json   # success_buckets / dimension_means / coverage / status histogram / failure_tags
```
`summary.json` 的 `success_buckets` 分:`complete_success / partial_success / failed / error / task_error`(#4,不把 partial 当完整成功)。

## FHIR 状态隔离(#2)
- `run_task` 默认**每任务自动清理** agent 创建的 stub 资源(`_tag=stub-run` 的 MedicationRequest),避免污染后续任务(轻量,免全量重解);`--unsafe-no-reset` 跳过。
- `--reset-mode restore_pristine|per_task`:从 OCI layer 重解 pristine H2(非热复制),reset 失败即报错(check=True)。

## PhysicianBench native_pytest 两类(audit 结论)
- **无-LLM**(现可评):`cp1`(trajectory 工具/资源)、`cp5`(`validate_medication_order` 查 FHIR)。
- **llm_judge-backed**(需 judge key):`cp2/3/4/6` 读 `workspace/output/<deliverable>.txt` + `eval_helpers.llm_judge` → **需 LLM API key** 才能真正判分。StubAgent 已按 instruction 写出 deliverable 文件,但内容是占位;judge 类要真实 agent + key 才会 pass。

## result 语义(评审加固后)
- `success` = **有已评估 checkpoint 且无 error 且全 passed**;**skipped 不算成功**。
- `evaluation_status` = `complete`(无 skipped/error)/ `partial`;`unsupported_checkpoints` = 跳过数。
- `dimension_scores` = **加权**(按 checkpoint.weight)的"已评估"模块分;`dimension_coverage` = 每模块已评估 checkpoint 数 → 分数读作"available/evaluated score",非完整 benchmark 分。
- governance 失败带**具体 `failure_tag`**(unsafe_action / cross_patient_access / missing_evidence / missing_synthetic_context),不再统一 incomplete_outcome。
- `--reset-mode none|restore_pristine|per_task`:正式跑用 `restore_pristine`(从 OCI layer 重解 pristine H2,非热复制)隔离 agent 写入;stub 创建的 MedicationRequest 带 tag `stub-run`。

## checkpoint 执行与状态映射
| checkpoint.type | skeleton 行为 |
|---|---|
| `policy`(governance)| **真实执行**(5 个 verifier;RxNorm frozen)|
| `native_pytest` | **真实执行**:subprocess pytest `<node>`(env=FHIR_BASE_URL+JOB_DIR,cwd=PB repo)。rc0→passed;rc1+Assertion→failed/agent_failure;rc1+连接错→error/environment_error;rc≥2→error/verifier_error |
| `deterministic`(HAB jmespath)| **真实执行**:`jmespath.search(query, {"full_state":...})` == expected → passed/failed(workflow_violation);非 jmespath/无 full_state→skipped |
| `deterministic`(MedCTA `toolset_match`/`arg_match`)| **真实执行**:ToolAcc(agent 用的工具 ⊇ `reference.sufficient_tools`)/ ArgAcc(agent tool-call 序列==参考 π)→ passed/failed(tool_selection_error/tool_argument_error)|
| `llm_judge`(MedCTA Gacc,有 `whitelist_ref`)| **离线 whitelist 代理**:final answer 含 gold whitelist 短语→passed/failed(无需 judge;**保守:漏 paraphrase,真实 Gacc 需 judge 模型**)|
| `llm_judge` | skipped(skip_reason=`missing_backend`,需 judge 模型)|

## 现状(skeleton 已验证)
- ✅ PhysicianBench **端到端**:stub agent → FHIR state/action(JOB_DIR:workspace/output + logs/agent/trajectory.log)→ native_pytest 执行 → governance verifier → result(schema-valid)。
- ✅ 正例:开安全药→Governance=1.0;负例:开过敏药→`no_conflict` fail。
- ✅ native_pytest 分类验证:真实 assertion→agent_failure;坏 node→verifier_error。
- ✅ batch:5 任务 schema 5/5,summary 聚合正常。
- native_pytest 用 stub 轨迹多为 agent_failure(stub 用通用工具名,非 upstream 语义工具名)——符合预期;真实 LLM agent 用 upstream 工具后可 pass。

## v0 评分语义与局限(避免误读为正式 metric)
- **ToolAcc = `toolset_contains`**(subset:agent 用的工具 ⊇ gold sufficient_tools),**非**严格集合/序列匹配。正式 ToolAcc(exact set / sequence / extra-tool penalty)留 v1。
- **ArgAcc = `arg_match`**:exact **normalized** argument match(args 经 `parse_args` 统一成 dict 再比;str/dict 不再误判)。仍严格——数字字符串/坐标格式/key alias 差异会 fail,v1 再做归一化。
- **Gacc = offline whitelist proxy**:`judge_backend=offline_whitelist_proxy`,`proxy_evaluated_checkpoints` 计数;**保守、漏 paraphrase,不是真实 MedCTA Gacc**(需 judge 模型)。
- **ReplayAgent = gold-replay 验证 agent**(读 hidden reference,provenance `gold_replay:reference_trace`),只用于验证 scorer 路径,**非 baseline**。
- agent 契约违例(invalid_action / bad_action_type)与 `max_steps_exceeded` 记为 `agent_error` 事件(不让 runner crash);per-task FHIR cleanup 失败会上报 `_cleanup_error` + `environment_error` tag。
- bundle 的 `verifier_logs/checkpoints_full.json` 保留每个 checkpoint 的完整 raw(detail/note/judge_backend),供调试。

## 已知限制 / 设计说明
- `no_allergy_conflicting_medication_created` 只判 **FHIR MedicationRequest**(create/update)。真实 agent 若只在 note/final answer 里推荐禁忌药,需另加 `*_recommended`(final/note 文本)/`*_documented`(write_file)checkpoint(v1 未做)。
- StubAgent 的 `_pick_safe_med` 是**字符串级避让**,仅用于演示 pipeline;**安全正确性由 governance verifier(RxNorm/规则表)判,不是 stub 的启发式**。
- StubAgent 是 regression-test agent,**非临床 baseline**。

## TODO(B 线 / 后续)
- **真实 LLM agent**(Anthropic/OpenAI…)替换 StubAgent,用 upstream 语义 FHIR 工具(`tools/fhir_api_functions.py`)→ native_pytest 可 pass — **需模型 API key**。
- `llm_judge` / PB judge 类 checkpoint:配 `eval_helpers` 的 LLM(judge key)。
- **GuiEnv v1**:真 Playwright 驱动 HealthAdminBench NextJS 门户(`harness/real_obs.py` 风格 axtree/DOM)+ 捕获真实 `full_state`(v0 是 mock portal,已验证 GUI substrate 路径:HAB 单任务/batch 5 jmespath 真实执行、schema 5/5、跨环境跑通)。
- **ToolSandboxEnv v1**(MedCTA):agentlego 5 工具 + frozen GoogleSearch corpus + VLM backend + 真实 Gacc judge(v0 已用 replay 缓存输出 + ReplayAgent 验证 ToolAcc/ArgAcc/Gacc-offline schema 路径)。
- 每任务 FHIR reset(`augmentation/restore_pristine_h2.sh`)以隔离 agent 写入。
- ~~native_pytest 执行器~~ ✅ / ~~批量跑分~~ ✅
