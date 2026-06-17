# Known (Expected) Warnings

`semantic_validate_tasks.py` 把以下两类 warning 显式归为 **`expected_warnings`**——它们是数据集/任务类型的固有性质,不是 bug,也不是泄露。reviewer 看到不应误判为 validator 没做好。

## 1. MedCTA — forced-choice answer in prompt(2 条:`MCTA-98`, `MCTA-101`)

**Expected warning**: gold answer string appears in the prompt because the sample is a forced-choice question. This is dataset-native and does not leak hidden trajectory or sufficient tools.

- 例:`MCTA-98` 问 "oculus dexter (right eye) or oculus sinister (**left eye**)",gold = `left eye`;`MCTA-101` 问 "(**normal** vs abnormal)",gold = `normal`。
- 泄露的**不是** `U`(sufficient_tools)/ `π`(reference_trace)/ tool-path,而是**题目本身列出的选项**。
- 这类样本**保留**。validator 仅在 gold 出现在 `available_tools`/`constraints`(我们注入的字段)时才报 **ERROR**;出现在问题文本里 → `expected_warning`。

## 2. HealthAdminBench — instruction-contained form values(35 条)

**Expected warning**: some target form values are present in the task instruction. These tasks evaluate GUI execution and workflow compliance rather than information inference.

- GUI 任务常把"要填什么"写在指令里;考的是能否**正确执行跨页面/表单/上传/提交流程**,而不是让模型猜表单值。
- 因此 checkpoint 的 `expected` 值出现在指令文本中是**预期**的,不影响 Tooling/Lifecycle 评分的公平性。

## 判定口径

| 情形 | 严重度 |
|---|---|
| gold 出现在 `available_tools` / `constraints`(我们注入) | **ERROR**(真泄露)|
| gold/选项出现在题目文本(forced-choice / 定义式) | `expected_warning` |
| GUI 目标值出现在指令(执行类任务) | `expected_warning` |
| 缺机器可验证 checkpoint / 缺 sha256 / policy 不完整 | `unexpected_warning`(需查)|

> 退出码:仅 `errors` 或 `unexpected_warnings` 非零;`expected_warnings` 可接受。
