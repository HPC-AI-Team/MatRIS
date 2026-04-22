# Task 10 Open Questions

本文件用于记录任务 10 过程中当前仍不确定、需要进一步确认或后续补充标准的问题。

## 1. 扰动响应指标的量化标准仍不明确

当前 `force_change_mae_eVA_mean`（扰动响应平均强度指标）和 `force_change_max_eVA_max`（扰动响应最坏样本指标）主要作为相对 `FP32 baseline` 的横向比较指标使用，但尚未建立明确的工程验收标准。

当前疑问包括：

- 这两个指标是否应设置固定阈值，还是仅要求相对 `FP32 baseline` 不出现显著恶化。
- “显著恶化”的量化标准应如何定义，例如采用倍率阈值、百分比阈值，还是按结构类型分组设置不同标准。
- 是否需要针对 `equilibrium`、`perturbed_small`、`strained_high`、`md_initial` 分别制定不同容忍范围，避免总体平均掩盖敏感结构的退化。
- 是否需要结合 `fd_abs_error_eVA_mean`、`fd_abs_error_eVA_max` 与 `smoke test` 结果联合判定，而不是单独依据扰动响应指标做结论。

建议后续补充：

- 一套明确的任务 10 工程验收表述，例如“相对 `FP32 baseline` 的增幅不超过某个比例，且不得出现数量级恶化”。
- 一套更细化的分组检查标准，用于识别少数高应变或 MD 初始结构上的异常放大问题。
