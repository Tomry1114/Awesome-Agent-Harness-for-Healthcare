# Lab 异常判定补全 — LOINC 分布分析与 benchmark 固定参考范围建表方案

> Medical Harness 项目 / PhysicianBench FHIR 数据集
> 数据源：HAPI FHIR 8.8.0 (R4)，`http://localhost:38080/fhir`（HPC ce483，`~/Medical_harness/fhir-full.sif`）
> 分析脚本：`~/Medical_harness/loinc_stats.py`、`loinc_stats2.py`；原始输出：`~/Medical_harness/loinc_stats.txt`
> 日期：2026-06-16

> ⚠️ **This table is used for reproducible benchmark grading, not for clinical decision-making.**
> 本文所有范围均为 **benchmark 固定参考范围（benchmark-fixed reference range）**，仅用于评测可复现判分，
> 不代表真实临床正常值（后者受年龄、性别、实验室、测量方法、样本类型影响）。

## 1. 背景与目的

PhysicianBench 的 FHIR 数据集里，**lab Observation 没有 `referenceRange`、也没有 `interpretation`(H/L) 标记**。
要解锁"识别异常 lab"类任务的可判定 ground-truth，需补一张 **按 LOINC 码的 benchmark 固定参考范围表**，
供评分器（和可选的 lab-range 查询工具）使用。本文档回答：覆盖率口径、要覆盖多少码、范围怎么取、怎么落地。

## 2. 覆盖率（三种口径，分清分母）

参考范围只对**数值型 lab** 有意义，所以覆盖率必须分母分清：

| 口径 | 分母 | 用途 |
|---|---|---|
| 全部 lab Observation | **71,025** | 描述整体数据分布 |
| 数值型 lab（有 valueQuantity）| **57,518** | ✅ reference range 建表真正关心这个 |
| 标准 LOINC 数值型 | 数值型里去掉本地码/定性码（共 428 个数值型 LOINC 码）| 最适合估算建表工作量 |

非数值型 13,507 条（尿试纸 presence、Specimen type、Product Code Name、ABO+Rh 等）**不需要数值范围**。

### 覆盖率里程碑

| 累计覆盖 | 全口径需码数（含定性码）| **数值型口径需码数** |
|---|---|---|
| 50% | 27 | **21** |
| 80% | 91 | **45** ✅ 建表目标 |
| 90% | 175 | **88** |
| 95% | 309 | — |

> **结论：只需约 45 个数值型 LOINC 码即覆盖 80% 的数值型 lab。** 工作量很小，可逐项录入。

### 本文第 4 节列出的 43 个种子码覆盖

| 指标 | 值 |
|---|---|
| 43 码累计条数 | 41,952 |
| 占全部 lab | **59.1%** |
| 占数值型 lab | **72.9%** |

> 注意区分：上面"数值型口径 80% 需 45 码"与"43 种子码 ≈ 73% 数值型"是**自洽**的——再补 ~2 个高频数值码即达 80% 数值型。
> 之前草稿里"43 码覆盖 77%"是口径混淆，已作废。

## 3. 数据质量提示

1. **经验分位数 ≠ 参考范围。** 第 4 节 p2.5/p97.5 是本数据集（患病队列）经验区间，明显偏向异常
   （葡萄糖 p97.5=263、血小板 p2.5=13.5），**仅用于交叉核对单位/量级**，不能当正常范围。
2. **少量经验值是噪声**（Holter HR p97.5≈2e7、Eosinophils# p97.5=83.8）→ 建表只取标准定量分析物，
   跳过本地码（`base_name` 系统）与明显噪声。
3. **范围依性别/年龄而异** → 性别相关项（Hgb/Hct/Cr/RBC/Ferritin/HDL）分性别给值。

## 4. range_type：把 lab 按判定规则分类（核心设计）

不能把所有 lab 都当"双侧正常区间"判 H/L。scorer 里每个 LOINC 配如下字段：

```json
{
  "range_type": "interval | upper_threshold | lower_threshold | clinical_threshold | risk_target",
  "abnormal_direction": "low | high | both",
  "clinical_context_required": true
}
```

### A. interval（真正的双侧参考区间）— 直接判 normal/low/high

> Na, K, Cl, CO₂, BUN, Cr, Ca, Mg, Phos；Hgb, Hct, WBC, Platelets, RBC, MCV, MCH, MCHC, RDW；
> Albumin, Total protein, AST, ALT, ALP, Total/Direct bilirubin, Globulin；TSH；differential %、LDH、Anion gap

### B. 阈值型 / 风险目标（不要混成普通 H/L）

| 项 | range_type | 规则 | 说明 |
|---|---|---|---|
| LDL | risk_target | ≥100 非理想 | 是治疗目标，非统一"正常范围" |
| Total cholesterol | risk_target | ≥200 非理想 | |
| Triglyceride | upper_threshold | ≥150 高 | |
| HDL | lower_threshold | M<40 / F<50 低 | **低了不好，高通常不算异常** |
| HbA1c | clinical_threshold | ≥6.5 糖尿病 | 诊断阈值，非 lab 上限 |
| eGFR | clinical_threshold | <60 CKD 风险；≥90 正常 | 风险分层 |
| INR | **v1 跳过 / context_required** | 是否异常取决于是否抗凝 | `"context_note": "Assumes patient is not on anticoagulation."` |

### C. Glucose（2345-7）特殊处理

LOINC 2345-7 是 **serum/plasma glucose，不一定是空腹**；FHIR 里没有 fasting 状态，直接用 70–99 会把大量随机血糖误判异常。
**保守做法（推荐 v1）**：
```
glucose: lower_threshold 70 (low) ; upper_threshold 200 (clinically high)
optional fasting_reference: 70–99  # 仅在文档声明"按 fasting-like 固定范围判分、fasting 状态未编码"时启用
```

## 5. 种子表（top 定量分析物，benchmark 固定值）

> "范围/阈值"为常规成人 conventional US 单位种子值，正式使用前对照可引用化验来源核定并标注 `source`。

| LOINC | 分析物 | 单位 | range_type | 范围/阈值 | 经验 p2.5–p97.5 | N |
|---|---|---|---|---|---|---|
| 2345-7 | Glucose | mg/dL | clinical_threshold | <70 低 / ≥200 高（fasting 70–99 可选）| 78–263 | 2004 |
| 33914-3 | eGFR | mL/min/1.73m² | clinical_threshold | <60 风险 / ≥90 正常 | 26–131 | 1683 |
| 2823-3 | Potassium | mmol/L | interval | 3.5–5.0 | 3.05–5.18 | 1447 |
| 2160-0 | Creatinine | mg/dL | interval | M 0.7–1.3 / F 0.6–1.1 | 0.35–2.74 | 1443 |
| 4544-3 | Hematocrit | % | interval | M 41–50 / F 36–44 | 19.9–47 | 1406 |
| 3094-0 | BUN | mg/dL | interval | 7–20 | 7–51 | 1406 |
| 17861-6 | Calcium | mg/dL | interval | 8.5–10.2 | 7.7–10.3 | 1404 |
| 2951-2 | Sodium | mmol/L | interval | 135–145 | 125–145 | 1392 |
| 718-7 | Hemoglobin | g/dL | interval | M 13.5–17.5 / F 12.0–15.5 | 6.6–15.6 | 1382 |
| 2075-0 | Chloride | mmol/L | interval | 98–107 | 91–112 | 1381 |
| 777-3 | Platelets | K/uL | interval | 150–450 | 13.5–443 | 1367 |
| 6690-2 | Leukocytes (WBC) | K/uL | interval | 4.5–11.0 | 0.21–28.3 | 1363 |
| 2028-9 | CO₂ (Bicarb) | mmol/L | interval | 22–29 | 20–31 | 1357 |
| 787-2 | MCV | fL | interval | 80–100 | 76–114 | 1314 |
| 785-6 | MCH | pg | interval | 27–33 | 24–39 | 1311 |
| 786-4 | MCHC | g/dL | interval | 32–36 | 29–36 | 1307 |
| 789-8 | RBC | MIL/uL | interval | M 4.7–6.1 / F 4.2–5.4 | 1.89–5.27 | 1307 |
| 788-0 | RDW | % | interval | 11.5–14.5 | 11.9–28 | 1306 |
| 33037-3 | Anion gap | mmol/L | interval | 8–12 | 4–15 | 1266 |
| 1975-2 | Total bilirubin | mg/dL | interval | 0.1–1.2 | 0.2–4.5 | 1193 |
| 1742-6 | ALT | U/L | interval | 7–56 | 9–177 | 1182 |
| 1920-8 | AST | U/L | interval | 10–40 | 12–139 | 1174 |
| 1751-7 | Albumin | g/dL | interval | 3.5–5.0 | 2.27–4.87 | 1166 |
| 2885-2 | Total protein | g/dL | interval | 6.0–8.3 | 4.73–8.78 | 1151 |
| 6768-6 | Alkaline phosphatase | U/L | interval | 44–147 | 45–338 | 1150 |
| 10834-0 | Globulin | g/dL | interval | 2.0–3.5 | 1.73–4.87 | 1022 |
| 5905-5 | Monocytes [%] | % | interval | 2–8 | 0.87–22.6 | 911 |
| 770-8 | Neutrophils [%] | % | interval | 40–70 | 11.6–91.3 | 910 |
| 736-9 | Lymphocytes [%] | % | interval | 20–40 | 1.62–48.9 | 878 |
| 714-6 | Eosinophils [%] | % | interval | 1–4 | 0–6.9 | 799 |
| 706-2 | Basophils [%] | % | interval | 0.5–1 | 0.1–3 | 754 |
| 2777-1 | Phosphorus | mg/dL | interval | 2.5–4.5 | 2.09–5.12 | 333 |
| 3016-3 | TSH | uIU/mL | interval | 0.4–4.0 | 0.09–12.7 | 316 |
| 2089-1 | LDL cholesterol | mg/dL | risk_target | ≥100 非理想 | 41–182 | 257 |
| 19123-9 | Magnesium | mg/dL | interval | 1.7–2.2 | 1.52–2.74 | 250 |
| 2571-8 | Triglyceride | mg/dL | upper_threshold | ≥150 高 | 40–319 | 246 |
| 4548-4 | Hemoglobin A1c | % | clinical_threshold | ≥6.5 糖尿病 | 4.8–10.6 | 246 |
| 2085-9 | HDL cholesterol | mg/dL | lower_threshold | M<40 / F<50 低 | 32–111 | 232 |
| 2093-3 | Cholesterol (total) | mg/dL | risk_target | ≥200 非理想 | 118–268 | 228 |
| 2532-0 | LDH | U/L | interval | 140–280 | 94–1400 | 195 |
| 1968-7 | Direct bilirubin | mg/dL | interval | 0.0–0.3 | 0.1–3.97 | 187 |
| 2276-4 | Ferritin | ng/mL | interval | M 24–300 / F 12–150 | 13–1710 | 165 |
| 6301-6 | INR | — | context_required | v1 跳过（假设未抗凝）| 0.91–1.83 | 161 |

## 6. ref_ranges.json schema（含单位硬检查）

```json
{
  "loinc": "2823-3",
  "display": "Potassium",
  "canonical_unit": "mmol/L",
  "accepted_units": ["mmol/L", "mEq/L"],
  "range_type": "interval",
  "abnormal_direction": "both",
  "low": 3.5,
  "high": 5.0,
  "sex": "any",
  "age_min": 18,
  "age_max": null,
  "clinical_context_required": false,
  "source": "<cited reference>",
  "benchmark_note": "Adult fixed benchmark range."
}
```

### 单位处理规则（必须做，不能只看 LOINC）

| 情况 | 处理 |
|---|---|
| unit 与 `canonical_unit` 完全一致 | 直接判定 |
| unit 在 `accepted_units`（已知别名，如 mmol/L↔mEq/L、10*3/uL↔K/uL）| 归一化后判定 |
| unit 需换算且已实现 conversion rule | 换算后判定 |
| unit 未知 / 冲突 | **skip，并记录 `unit_mismatch`** |

## 7. 建表优先级（不必一口气补 90 个）

| 级别 | 内容 | 目标 |
|---|---|---|
| **Tier 1** | BMP + CBC + LFT + renal + electrolytes | 覆盖大多数 abnormal lab 任务 |
| **Tier 2** | lipids + HbA1c + TSH + ferritin + INR + Mg/Phos | 慢病、内分泌、凝血任务 |
| Tier 3 | urine、blood gas、special endocrine、local codes | 之后再说 |

> 第一版做 **Tier 1 + 少量 Tier 2 ≈ 40–60 码**即可（≈80% 数值型 lab）。第 5 节种子表已接近 Tier 1/2 核心。

## 8. 落地决定：不注入 FHIR，改做工具

**v1 不写入 FHIR**，评分器用外部 `ref_ranges.json`。理由：① 不污染原始 PhysicianBench 数据；② 便于复现；
③ 便于切换 range 版本；④ 避免 agent 直接看到 H/L 标签使任务变简单。

更好的做法是加一个工具，既测 Tooling 又不改原数据：
```
get_lab_reference_range(loinc, sex, age, unit) -> {range_type, low, high, threshold, unit, source}
```
（可选）日后若需 agent 能在 FHIR 内看到范围，再批量 `PUT` 补 `referenceRange`/`interpretation`（transaction Bundle，每包 200–500 条；先备份 H2）。

## 9. 这一步能解锁的 benchmark 任务与维度映射

可构造的任务：
- 识别最近一次异常 potassium / creatinine / hemoglobin
- 判断是否存在肾功能异常（Cr/eGFR）
- 某药物是否因 eGFR/Cr 需谨慎/调量
- 判断贫血、血小板减少、白细胞异常
- 判断肝功能异常（AST/ALT/ALP/bilirubin）
- 判断糖尿病控制不佳 / 高血糖风险（HbA1c/glucose）

维度映射：

| 维度 | 本步贡献 |
|---|---|
| Context | 患者 lab context |
| Tooling | lab query + `get_lab_reference_range` |
| **Verification** | **abnormal lab 自动判分（本步主要补这个）** |
| Governance | 仅在与 Allergy/RxNorm 合起来做用药安全时才真正补 |

> 单独的参考范围主要补 **Verification**；**Governance 要等增强项 #2（Allergy + RxNorm）才解锁**。

## 10. 关联：整体数据增强路线（本文档是第 1 项）

| # | 增强项 | 状态 | 解锁维度 |
|---|---|---|---|
| **1** | **Lab 固定参考范围表（本文档）** | ✅ **已建**：`benchmark_dataprocess/PhysicianBench/ref_ranges.json`（49 条/43 LOINC，Tier 1/2）+ `lab_ref.py`（get_lab_reference_range + classify + 单位硬检查）；活 FHIR 实测覆盖 64% 抽样、异常检出正确 | Verification / Clinical Task Success |
| 2 | 合成 AllergyIntolerance + 药品 RxNorm 映射 | 待做 | Safety & Governance |
| 3 | 从时间戳重建 Encounter | 可选 | Workflow Compliance |

> 数据集整体画像与 7 维度可测性评估见记忆 `physicianbench-fhir-hpc`；模块/维度框架见 `Medical harness.pptx`。
