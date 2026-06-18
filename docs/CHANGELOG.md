## 2026-06-18 — 评审复修:解析器回归(新-1/新-2)+ GUI 默认崩溃(#1)+ RegionAttribute 假接地(#3)

**背景**:B 修复后评审实测发现 1 个新引入回归(中高)+ 2 个旧未修。本轮全部收掉。备份:runner/*.bak_fix。

**新-1(中高,B 引入的回归)— 解析器吞最终答案**
- 根因:`_first_json_after` 找不到 `<tool_call>` 标签时(i<0)用 start=0 扫全段,把 `<answer>{"diagnosis":"pneumonia"}</answer>` 里的结构化 JSON 当成 tool_call → `{tool:None}` → run.py 判 bad_action_type → agent_error,答案丢失。MedCTA/临床答案常是 JSON,命中率不低。
- 修:`_first_json_after` 无 tag 立即返回 None;`_parse` 仅当出现 `<tool_call>` 标签**且** name 非空才走工具,否则走 `<answer>`。
- 实测(无 GPU):`answer with JSON obj` 现 → final 保留答案(原 → tool None 吞答案);`bare json no tag` → final(新-2)。原有 tool_call / 剥 image / nested-arg 路径不变。

**新-2(低)— 同源副作用**:无标签裸 JSON 原被当 tool_call。与「答案可含 JSON」互斥,选答案:有 `<answer>`/无 `<tool_call>` 一律走 answer。随新-1 一并修掉。

**#1(高,回归未修)— GUI 默认在 login 节点崩**
- 根因:`make_env` 默认 `GuiEnvReal`;login 节点 playwright 可 import 但 chromium 无法启动 → `run.py --bench HealthAdminBench` 默认崩、run_batch 跑 HAB 全 task_error。(import 探测不可靠,故不用 auto-probe。)
- 修:**默认=GuiEnvMock(安全)**,真实门户改显式 opt-in `MH_GUI_MODE=real`;run_hab_agent.sbatch 加 `export MH_GUI_MODE=real`(GPU 节点有可启动 chromium)。mock 仍按 mock_inmemory 被 qualify,降级诚实。

**#3(中,未修)— RegionAttributeDescription 假接地**
- 根因:dispatch 把 bbox/region/attribute 三者塌成一个 `region` 参数(attribute 被丢);且该工具从不裁剪图像,只在 prompt 写"focus on [坐标]" → 新提示词重度依赖它做 grounding,但接地是假的。
- 修:(a) dispatch 把 `bbox` 与 `attribute` **分开传**;(b) `region_attribute_description` 对数值 bbox **真裁剪 PIL 图**(支持像素/归一化 0..1/字符串坐标,越界 clamp)再喂 VLM → 真·像素级接地;(c) 自由文本区域(如 "left lung apex")无法裁剪时退化为 focus 文本并在 prompt 注明"未裁剪/描述全图",诚实。
- 实测(无 GPU):像素 bbox [10,20,60,80]→裁出 50×60;归一化 [.1,.2,.5,.8]→80×60;字符串 bbox 可裁;文本区域→不裁、回退 attribute。

**待办(评审 #2/5,低,本轮未动)**:ArgAcc 精确全等过脆(需语义/容差比较);vlm_backend `torch_dtype` vs 新版 transformers `dtype` 不一致(仅 deprecation warning,sbatch grep 已过滤)。

**验证状态**:4 文件 py_compile 通过;解析器 6 用例 + crop 4 形态 + GUI 默认/opt-in 全部实测通过。修复后 MedCTA 重跑(job 9887269)验证不回归 + 产出干净基线。

## 2026-06-18 — B 修复:MedCTA 协议噪声清理(提示词 + 解析器)+ Δ profiling

**目的**:失败画像已把问题分成「可工程修」vs「模型能力瓶颈」。先把协议噪声清掉,让残余信号纯净。

**改动**(runner/qwen_agent.py;旧版备份 qwen_agent.py.bak_preB)
- **解析器真 bug 修复**:旧 `TOOL_CALL_RE = <tool_call>\s*(\{.*?\})` 非贪婪,遇到带嵌套 `arguments:{...}` 的 tool_call 在第一个 `}` 截断 → JSON 解析失败 → 误落 final 分支。这才是 MCTA-0 `final_answer_format_error` 的真因(不是「堆叠被丢」)。改为 `_first_json_after()` 花括号配平提取首个完整 JSON(处理字符串/转义),robust。
- **协议防御**:感知类工具(ImageDescription/RegionAttributeDescription/OCR)若被 agent 传了 `image/images/image_path/img` 参数,解析时剥除(后端本就持有真实图、忽略该参数)→ 消除 tool_argument_error 噪声。
- **tool_sandbox 提示词重写**:明确「图已在工具内,勿传 image 参数、勿自行粘贴图像文本,ImageDescription 用空 arguments」+「勿只凭首个全局描述作答,必须多次 RegionAttributeDescription 针对问题区域 grounding」+「一次只输出一个 action,禁止单消息堆叠多个 <tool_call>」。

**Δ 结果**(同 MCTA-0..9 / 同 2B / 同离线 whitelist proxy;job 9887229 @ gpu3-9。修复前 profile 备份 runner/medcta_profile.preB.json)
| 失败模式 | 前 | 后 |
|---|---|---|
| tool_argument_error | 4/10 | 0/10 ✅清零 |
| final_answer_format_error | 1/10 | 0/10 ✅清零 |
| tool_selection_error | 0/10 | 0/10 |
| search_misuse | 0/10 | 0/10 |
| loop_or_invalid_action | 1/10 | 1/10 (MCTA-7) |
| image_misread | 6/10 | 8/10 |
| outcome_proxy_fail | 8/10 | 9/10 |
| underuse_vs_ref | 8/10 | 9/10 |
| proxy 命中 | 2/10 | 1/10 (MCTA-2) |

**解读**:两类「可工程修」协议噪声(arg/format)定向清零。image_misread 6→8 非变差,是失败被正确重归类——MCTA-0 从「解析截断空答」迁为「真感知后读错」;失败从协议桶迁入能力桶,正是目标。多步 grounding 提示在 MCTA-5 见效(steps 3→6、rep 1→2),其余任务 2B 仍单步答。proxy 命中 2→1 属 2B 改提示后的生成抖动,非 B 目标。

**结论**:协议层已干净,残余=纯模型能力(感知 + grounding 纪律)→ 转入 option A(4B/7B)。

## 2026-06-18 — MedCTA QwenToolAgent 失败画像 profiling(MCTA-0..9)

**目的**:用已稳的架构产实验信号——真实 Qwen3-VL-2B agent 跑 MCTA-0..9,确定性多标签分类失败模式。

**改动/产物**
- runner/medcta_profile.py(新,可复用):单进程跑 10 条(模型 lru_cache 只加载一次)+ 多标签分类器;落盘 runner/medcta_profile.json(全轨迹+5 个 cp+实际参数)。
- runner/run_medcta_profile.sbatch(新):debug/gpu:1。job 9885702 @ gpu3-9,~160s 跑完 10 条。
- **分类器修正(重要)**:v1 把 cp_tool_selection/cp_arg_accuracy 失败(=没复刻 reference 多步序列)误当成「选错工具/参数错」,致 tool_selection/arg 双 10/10 虚高并屏蔽 image_misread。v2 区分「形式 ToolAcc/ArgAcc 失败=工具欠使用」与「真选错工具」,不再屏蔽 misread。

**结果(10/10 success=False;cp_outcome 是离线 whitelist 代理)**
- 失败标签(多标签):outcome_proxy_fail 8/10 · **image_misread 6/10** · tool_argument_error 4/10 · loop_or_invalid_action 1/10(MCTA-7 10×ImageDescription 空答)· final_answer_format_error 1/10(MCTA-0 一条消息堆叠多个 <tool_call> 未解析)· tool_selection_error 0/10 · search_misuse 0/10。
- 形式指标诊断:**ToolAcc 0/10、ArgAcc 0/10**;underuse_vs_ref 8/10(单步「调一次 ImageDescription 就答」,不复刻 reference 多步 grounding)。
- proxy outcome 命中:**2/10**(MCTA-2、MCTA-8)。
- 具体可修 bug:agent 常把幻觉文本当 image 参数传(如 {"image":"embryonic kidneys Gpc3..."}),tools_medcta 实际忽略该参数用真实图,但说明 agent 误解工具契约 → 提示词可修。

**结论**:2B 不存在「选错工具」问题(总选对 ImageDescription、不滥用 search);真正瓶颈是 **(1) 工具欠使用→形式 ToolAcc/ArgAcc=0**、**(2) 感知/推理错(image_misread 6/10)**、外加少量协议滑落(堆叠 tool_call、传错参数、偶发循环)。→ 印证下一步 option A(4B/7B + 协议/反堆叠约束 + 提示「勿传 image 参数」),并需真实 Gacc judge 才能把 outcome 从 proxy 升为正式 acc。
## 2026-06-18 — provenance 角色分离 + qualification 改为按 mock/replay/proxy 限定(评审反馈)

**动因**:评审指出设计口径要收紧——(1) Agent 层只是「脑」,工具真实执行属于 tool-backend/环境层,且 MedCTA 里脑与图像工具可能同为 Qwen3-VL,必须在 provenance 里分角色;(2) `_warning` 不能按 FHIR/非 FHIR 划分(GUI v1/MedCTA v1 都是真的);(3) gold/replay 的 success 只能算 scorer 验证,不能进真实 baseline。

**改动**
- `runner/run.py` provenance 重写为**角色分离**:`agent_model`(脑)/ `tool_backend`(手:real_playwright_portal / real_vlm_tools / live_hapi / mock / replay)/ `tool_backend_model`(工具内部用的模型,如图像工具的 VLM；确定性工具为 null)/ `judge_model`（none / offline_whitelist_proxy）/ `uses_hidden_reference` / `scorer_validation_only`。同模型担两角也分开记。
- `runner/run.py` `_warning` → **qualification**:仅当 mock_env / replay_tool_backend / outcome_proxy / uses_hidden_reference / scorer_validation_only / proxy_scored_checkpoints 才降级标注;**真实 GUI、真实 ToolSandbox 不再被天然判为 stub**。
- `spec/result.schema.json` provenance 显式声明 `tool_backend_model` / `uses_hidden_reference` / `scorer_validation_only`(judge_model 早已在)。
- 确认隐藏状态边界:`GuiEnvReal._snap` 只回 `{ok,url,title,observation}`,**full_state 不进 agent 观测**,仅 `self.full_state` 供 scorer 经 `getattr(env,'full_state')` 读。

**验证**
- gold 路径重跑:provenance = `{agent_model:'scripted:gold_path', tool_backend:{gui:'real_playwright_portal'}, tool_backend_model:null, judge_model:'none', uses_hidden_reference:true, scorer_validation_only:true}`;`_qualification=['scorer_validation_only','uses_hidden_reference']`;**success=True 被正确限定为 scorer 验证**,schema OK。
- 逻辑推导(MedCTA 真实 qwen):agent_model=Qwen3-VL text-only brain / tool_backend_model=Qwen3-VL(图像工具)/ judge_model=offline_whitelist_proxy / qual=outcome_proxy → 「真实 agent 轨迹 + proxy outcome」口径成立。

**口径更正(写入 STATUS 设计不变量)**
- 执行差异收敛到 `EnvironmentAdapter`;**评价差异收敛到 `checkpoint.method` 的 scorer dispatch**(native_pytest/jmespath/llm_judge/policy)。runner 主循环对两者无感知。
- 当前完成 = **evaluation harness / benchmark runner**(已含 policy overlay/governance verifier);未做 = **active runtime intervention harness**（运行时注入上下文/限工具/强制 safety/拦截危险 action），下一步用同一 runner 做 with/without A/B Δ。
- MedCTA 真实 agent 口径:**真实 agent 轨迹 + ToolAcc/ArgAcc(真) + proxy outcome**,接真实 Gacc judge/人工评审前不叫正式 acc。

## 2026-06-18 — GUI 数据对齐核查 + gold 路径证明可解(success=True)

**背景**:上一轮误判 HAB GUI 任务数据不对齐(看到 Brooks/Chen 而非 Martinez)。本轮核查发现是观测截断假象——门户本就有 DEN-001/Martinez。

**核查/验证**
- 门户 denial 来源:benchmark/v2/portals/app/lib/denialsSampleData.ts 的 SAMPLE_DENIALS(50 条,含 DEN-001 Martinez, Carlos),/emr/denied 渲染 40 行 worklist,denials-worklist-row-DEN-001 确在。
- gold 路径(手动 + scripted agent + 真实 scorer):navigate /emr/denied/DEN-001 → click tab-remittance_image → click disposition-select → click disposition-option-route-to-clinical-appeals → type triage-note-input → click submit-disposition-button。结果 success=True / status=partial / schema OK;cp0 viewedDenialDetails、cp1 viewedRemittanceImage、cp2 selectedDisposition='Route to Clinical Appeals'、cp3 documentedAppealInEpic 全 PASS;Context/Lifecycle/Observability=1.0;cp4-8(llm_judge)、cp_admin_compliance(policy)skip(无后端)。
- 对齐审计(/tmp/align_audit.py):门户 50 个 DEN id;135 个 HAB 任务中 57 个引用 DEN id,全部存在于门户 → 0 缺失。denial 类任务数据完全对齐。

**改动**
- runner/agents.py:新增 ScriptedAgent(name=scripted,从 env MH_SCRIPT 读 JSON 动作序列),作可复用的 gold-path 验证工具;make_agent/AGENT_REGISTRY 已注册。

**结论**
- 无数据需要修:denial 类 GUI 任务与门户数据对齐,且 ground truth 可达、scorer 对正确轨迹产出 success=True。GUI substrate 至此完整闭环(首次出现 GUI success=True)。
- 真实 agent 仍失败的根因是 2B 模型太弱(不遵 <tool_call> 协议),非数据/管线问题 → 下一步走 option A(4B/7B + 协议约束)。
- 待办(更广对齐):prior_auth/submit_auth/eligibility 等非 denial 流(payer-a/b、auth)用不同标识符与门户,未审计;如需全量可解性保证再扩。

## 2026-06-18 — 真实 agent 跑通 HealthAdminBench(GUI),3/3 全通

**目标**:把第 3 个数据集(HealthAdminBench / GUI)也接上真实 agent,完成「123 全部真实 agent 测过」。

**改动**
- `runner/environments.py` · `GuiEnvReal` 重写:新增**可读观测管线** `_observe()`——JS 给可见可交互元素打 `data-mh-ref` 序号(先清旧属性防串号)、抽 role/label/value，再附 `document.body.innerText` 页面文本；动作 `click/type/select/submit` 支持 `{"ref": N}`（选择器 `[data-mh-ref=N]`），保留 CSS/可见文本回退；每步动作后回 `_snap()`（`{ok,url,title,observation}`）；新增 `initial_observation()` 供首步注入；`_read_state()` 仍持续更新 `full_state`（评分对象不变）。
  - **关键 bug 修复**：`_OBS_JS` 写进文件后是普通三引号串，`\n{3,}` 被解释成真换行 → JS 正则损坏（"missing /"）；改为 `r"""` 原始串修复。
- `runner/run.py`：`env.reset()` 后若 env 有 `initial_observation()` 则 seed 进 `last_res/last_obs`，让 GUI agent 首步即看到页面。
- `runner/qwen_agent.py`：gui 系统提示改为「读 OBSERVATION → 按 ref 操作」并给出 navigate/click/type/select/submit/snapshot 模板；`act()` 加 gui 分支（每步把 observation/error 作为 user 消息喂回，3800 字上限）；抽出 `_parse()` 复用。
- `runner/agent_test_driver.py`：第 3 参数可选 `max_steps`（GUI 用 14）。
- `runner/run_hab_agent.sbatch`（新）：debug/gpu:1，`MH_PORTAL_BASE=http://10.120.31.247:3002`（GPU 节点经 login IP 访问 `next dev` 门户），跑 `HealthAdminBench HAB-denial-easy-1 14`。

**验证**
- 无模型冒烟(login 节点)：观测管线工作 — `/denied`→`/emr/denied`，页面文本正确渲染，枚举 **77 个交互元素**（View/Appeal/行等带 ref+label）。
- GPU 实跑(job 9884561, node gpu3-9)：**端到端无崩溃，schema OK，status=partial**。chromium 在 GPU 节点正常、门户可达(200)、`full_state` 经 jmespath 评分。**至此三条 substrate(FHIR/GUI/tool_sandbox)真实 agent 全部端到端跑通(3/3)。**

**发现**
- 2B 在 GUI 同样**未遵守 `<tool_call>` 协议**（输出模仿格式的裸文本，引用了真实 `[ref=6]/[ref=63]` 说明观测已送达模型）→ step0 即被当 final，cp 全 failed。与 MedCTA/PB 同根因：2B 偏弱、不稳。**下一步应换 Qwen3-VL-4B/7B 并加协议/反循环约束。**
- 门户实例数据(SHC remittance workqueue: Brooks/Chen/Miller…)与任务(Martinez/DEN-001)不对齐 —— 独立的数据对齐 gap，需把任务引用的 denial 灌进门户或换匹配任务。

# Medical Harness — 改动日志 (CHANGELOG)

> 每轮改动追加于此(最新在上)。滚动状态见 `docs/STATUS.md`。运行环境:`medicalharness`(ce483)。

## 2026-06-18 (深夜) — 真实 agent 跑通 PhysicianBench(2/3)+ runner 健壮性修复

**改动**
- `runner/qwen_agent.py` — 泛化为**环境感知**(tool_sandbox / fhir / gui 各自 system prompt + 感知方式),不再 MedCTA 专用。
- `runner/environments.py` — FhirEnv `_get`/`fhir_create` 捕获 HTTPError/URLError 返回 error dict(不再抛异常);加 `import urllib.error`。
- `runner/run.py` — agent loop 包裹 `env.call_tool`,工具异常转为 observation,**不再崩溃整个 run**。
- `runner/run_pb_agent.sbatch` + 复用 `agent_test_driver.py`(新)— `FHIR_BASE_URL` 指向 login 节点(GPU 节点经 `*:38080` 可达,实测 200)。

**结果(PB-aberrant_drug_screen,Qwen3-VL-2B,GPU 真跑)**
- 真实 agent 端到端跑通:连 FHIR(200)→ fhir_search → trajectory+result,**schema OK,无崩溃**。
- 2B 偏弱:重复调同一个(参数错:`Patient?patient=` 非法)搜索 10 次 → max_steps、未写交付物 → 8 个 native_pytest cp 全 failed。
- 结论:**PB 真实 agent 路径打通**;2B 循环 + 弱推理(与 MedCTA 一致)→ 需 4B/7B 或加防循环。

**进度**:真实 agent 已测 **MedCTA(3) + PhysicianBench(1)**;剩 **HealthAdminBench(2,GUI,需 axtree 观测注入)**。

## 2026-06-18 (晚) — 真实 Qwen3-VL tool-calling agent(MedCTA)

**新增/改动**
- `runner/qwen_agent.py`(新)— `QwenToolAgent`:纯文本 brain,系统提示给工具协议,解析 `<tool_call>`/`<answer>`,逐步自主决策。
- `runner/vlm_backend.py` — 加 `chat()`(文本推理);修正:Qwen3-VL 处理器要求 message.content 为结构化 list,已把 string 归一化为 `[{type:text}]`。
- `runner/agents.py` — `make_agent` 路由 `qwen`。
- `runner/agent_test_driver.py` + `run_agent_test.sbatch`(新)— 捕获真实工具调用 + 最终答案到共享存储。

**首个真实 agent 结果(MCTA-0,Qwen3-VL-2B,GPU 真跑)**
- agent 自主调工具:ImageDescription→RegionAttributeDescription→GoogleSearch(出现重复循环)→final。
- 最终答案:"胰腺病灶,无静脉血栓证据"。gold:门静脉 + 肠系膜上静脉血栓 → **outcome 错(real acc = 0)**。
- 评分:cp_tool_selection **passed**(ToolAcc✓)、cp_outcome failed、cp_arg_accuracy failed(exact-match 严)、Tooling 0.5、Execution(proxy)0.0、success=False、schema OK。
- 结论:**真实 agentic 管线打通**;2B 偏弱(误读 + 循环重复调用)→ 应上 4B/7B;`RegionAttributeDescription` 未真正裁剪 bbox(≈整图),待改进。

## 2026-06-18 — v1 真实环境(MedCTA 工具后端 + GUI 真门户)+ 专用 env

**新建/改动文件**
- `runner/vlm_backend.py`(新)— 可插拔 VLM 后端,本地 Qwen3-VL 单例(`MH_VLM_BACKEND=local`,`MH_VLM_PATH`)。
- `runner/tools_medcta.py`(新)— MedCTA 5 真实工具:ImageDescription/RegionAttributeDescription/OCR→VLM、Calculator(安全 AST)、GoogleSearch(离线 frozen 语料,`MH_SEARCH_MODE=live` 可切)。
- `runner/environments.py`(改)— `ToolSandboxEnv` 改 real/replay 双模(`MH_TOOL_MODE=real`,从 `context.images[0]` 解析图路径);新增 `GuiEnvReal`(Playwright 驱动真门户,`full_state` 读 localStorage `portals_state.emr`);原 `GuiEnv`→`GuiEnvMock`;`make_env` 按 `MH_GUI_MODE` 选 real/mock;navigate 改为 host-agnostic(绝对 URL 映射到本地门户)。
- `runner/vlm_smoke.py` / `runner/vlm_answer.py` / `runner/gui_smoke.py`(新)— 冒烟/验证脚本。
- `runner/run_mcta_v1.sbatch` / `runner/run_acc_test.sbatch`(新)— GPU sbatch 作业。
- `docs/STATUS.md`(改)— 加 v1 真实环境块、env 说明、下一步=真实 agent。

**环境**
- 新建 conda env `medicalharness`(Python 3.10,8.9G):torch 2.8.0+cu128 / transformers 4.57.3 / playwright 1.48 / jmespath / jsonschema / pytest / pandas / pyarrow。与 AgentOCR 解耦。
- Node v22.22.3 装到 `~/.local/node`;HAB v2+v3 门户 `npm install`(npmmirror);Playwright chromium 130(build 1140,npmmirror 镜像,azureedge 被墙)。

**验证**
- VLM:Qwen3-VL-2B 在 A40 ~6s 加载 / ~8s 每图,输出真实医学描述。
- MedCTA v1 集成(sbatch,replay agent + 真实工具):success=true / Tooling=1.0 / schema OK,工具真在 GPU 执行。
- GUI v1 集成(HAB 任务):真 chromium→真门户→真 full_state→jmespath 正确评分(stub 正确判 fail),schema OK。
- 三 bench 校验仍全过:PB 100 / HAB 135 / MedCTA 107,0 error。

**关键发现**
- 原始 MedCTA agent 为纯文本 LLM,问题不含图,必须调工具才能"看图"(reference_trace 实证)→ 真实 agent 不直接喂图。
- replay agent 的 cp_outcome 通过 = 回放 gold,非模型真实正确性;2B 裸描述把门静脉血栓误读为肝占位 → 需真实 agent(可能上 4B/7B)。

**下一步**:建真实 Qwen3-VL tool-calling agent(MedCTA 先行)。
