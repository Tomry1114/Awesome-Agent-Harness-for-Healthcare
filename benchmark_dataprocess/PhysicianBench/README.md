# PhysicianBench — 数据处理

原始数据(只读):`../../benchmark/PhysicianBench/`(`fhir-full.sif`、OCI 源、tar.gz)。
统一任务规范:`../../docs/00_task_specification.md`。

## 运行环境
- `run_fhir.sh [PORT]` — 启动 HAPI FHIR server(默认 :38080,Singularity setuid)。
- `h2data/` — 可写 H2 库(运行时 `--bind` 到容器 `/tmp`)。**所有数据注入改这里,不动 benchmark/ 原始件。**

## 处理任务清单
- [x] `loinc_stats.py` / `loinc_stats2.py` — lab LOINC 分布统计(产出 `loinc_stats.txt`)
- [x] **增强 #1**:`ref_ranges.json`(49 条/43 LOINC,Tier 1/2)+ `lab_ref.py`(`get_lab_reference_range`+`classify`+单位硬检查)+ `build_ref_ranges.py`;活 FHIR 实测覆盖 64% 抽样、异常检出正确 → 解锁异常 lab 判分(Verification)
- [x] **增强 #2(全量 26 medication 任务,已并入 unified)**:`augmentation/`(build_augmentation 自动选26+确定性过敏原 / synthetic_allergies(26) / rxnorm_mapping[RxNav] / drug_safety_rules / allergy_bundle[幂等PUT] / drug_safety_check[5 verifier,RxNorm frozen] / merge_governance[可重入] / restore_pristine_h2)。FHIR 注入 26 条 AllergyIntolerance(tag 幂等可回滚);**104 governance cp 并入 tasks_unified.jsonl**;schema+semantic(含8项 governance 审计)全过。
  - 重置/恢复:`bash augmentation/restore_pristine_h2.sh`(按 PID 杀+重解 pristine+重启)。**勿热复制 H2**。
- [x] **增强 #3 Encounter 外部索引**:`augmentation/build_encounter_index.py` → `encounter_index.json`(100 患者/10,980 encounters,按日聚类,未注入 FHIR,provenance=augmented)→ Lifecycle/Context
- [x] **Medication safety 扩展**:`drug_safety_check.py` 加 `no_allergy_conflicting_medication_recommended`(final answer 文本)+ `_documented`(deliverable/write_file 文本);用推荐动词上下文匹配避免"报告过敏"假阳性;merge 后每 action 任务 6 个 governance cp。**局限**:文本匹配保守,仅覆盖已映射药名,可能漏 synonym/brand。
- [ ] `convert.py` → `tasks_unified.jsonl`:把 PhysicianBench 任务+checkpoint 转成统一任务规范

## 在统一规范中的定位
environment=`fhir` · modality=`structured_fhir` · 主压维度:Task Success / Tool Use / Context / Verification。
