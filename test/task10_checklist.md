# Task 10 Checklist

任务 10 的目标是从低风险运行时优化里筛出一个新的工程 baseline。当前建议按下面顺序执行：

## 0. 确认默认配置

先检查以下启动脚本当前启用的是不是 `fp32 baseline` 配置块：

- `test/eval/sh/run_static_eval.sh`
- `test/eval/sh/run_force_consistency.sh`
- `test/eval/sh/run_smoke_relax_md.sh`

## 1. 跑 FP32 baseline 静态评估

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_static_eval.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/static_eval/fp32_baseline`

## 2. 跑 FP32 baseline 轻量一致性检查

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_force_consistency.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/consistency/fp32_baseline`

## 3. 跑 FP32 baseline smoke test

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_smoke_relax_md.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/smoke/fp32_baseline`

## 4. 切到 TF32 配置

修改以下脚本，把 `tf32` 块取消注释，并把 `fp32` 块注释掉：

- `test/eval/sh/run_static_eval.sh`
- `test/eval/sh/run_force_consistency.sh`
- 如需要 smoke，也改：
  - `test/eval/sh/run_smoke_relax_md.sh`

## 5. 跑 TF32 静态评估

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_static_eval.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/static_eval/tf32`

## 6. 比较 FP32 vs TF32

在 `test/eval/sh/run_compare_eval.sh` 中启用：

- `fp32 vs tf32`

然后运行：

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_compare_eval.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/comparisons/fp32_vs_tf32`

## 7. 跑 TF32 一致性检查

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_force_consistency.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/consistency/tf32`

## 8. 可选：跑 TF32 smoke test

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_smoke_relax_md.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/smoke/tf32`

如果 `TF32` 没明显问题，可以保留为候选 baseline。

## 9. 切到 BF16 autocast 配置

修改以下脚本，启用 `bf16` 块：

- `test/eval/sh/run_static_eval.sh`
- `test/eval/sh/run_force_consistency.sh`
- `test/eval/sh/run_smoke_relax_md.sh`

## 10. 跑 BF16 静态评估

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_static_eval.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/static_eval/bf16_autocast`

## 11. 比较 FP32 vs BF16

在 `test/eval/sh/run_compare_eval.sh` 中启用：

- `fp32 vs bf16`

然后运行：

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_compare_eval.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/comparisons/fp32_vs_bf16`

## 12. 跑 BF16 一致性检查

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_force_consistency.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/consistency/bf16_autocast`

## 13. 可选：跑 BF16 smoke test

```bash
/home/lht/lab/MatRIS/test/eval/sh/run_smoke_relax_md.sh
```

默认输出目录：

- `/home/lht/lab/MatRIS/results/smoke/bf16_autocast`

如果 `BF16` 也稳定，再考虑 `FP16` 或 `compile`。

## 14. 按同样方式试 FP16 / compile

建议顺序：

1. `fp16`
2. `fp32 + compile`
3. `bf16 + compile`

每个版本都重复：

1. `run_static_eval.sh`
2. `run_compare_eval.sh`
3. `run_force_consistency.sh`

必要时再加：

4. `run_smoke_relax_md.sh`

## 15. 每轮优先检查的结果

至少看下面三类输出：

- `results/static_eval/.../summary_overall.json`
- `results/comparisons/.../comparison_summary.json`
- `results/consistency/.../force_consistency_summary.json`

必要时再看：

- `results/smoke/.../smoke_relax_md_summary.json`

## 16. 新 baseline 的筛选标准

优先保留满足以下条件的版本：

- 相对 `FP32 baseline` 有明确速度收益
- `energy / force / stress` 增量在可接受范围内
- `fd_abs_error` 没明显恶化
- smoke test 没有明显异常

## 备注

- `FP32 baseline` 的结果要一直保留，不要覆盖
- 每个候选版本都应单独保存到自己的结果目录
- 最终汇报时，建议仍然以“最终版本 vs 原始 FP32 baseline”作为主比较方式
