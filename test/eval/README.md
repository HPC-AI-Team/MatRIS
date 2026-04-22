# Test Eval

本目录用于功能正确性与可用性评估，包括静态预测、环境确认、relaxation/MD smoke test。

目录结构分为两层：

- `*.py`
  具体评估逻辑实现
- `sh/`
  固定配置启动脚本，方便直接运行

## 文件说明

`evaluate_static_metrics.py`
- 用途：对给定测试集逐结构运行 MatRIS 预测，保存静态预测结果和基础 timing。
- 主要输入：
  - `metadata csv`
  - 模型名、task、precision mode、device
- 主要输出：
  - `run_config.json`
  - `per_structure_predictions.jsonl`
  - `summary_overall.json`
- 适用场景：
  - 建立 `FP32 baseline`
  - 跑 `TF32/BF16/FP16/compile` 候选版本
  - 为后续版本比较提供统一结果格式

`benchmark_relax_md.py`
- 用途：运行一个简短的 relaxation 和一个短程 MD，做应用级 smoke test。
- 主要输出：
  - `eval/logs/benchmark_relax_md.log`
- 关注点：
  - 收敛是否成功
  - 最终 `fmax`
  - MD 是否能稳定跑完
  - steps/s 等基本性能

`collect_env_info.py`
- 用途：打印当前实验环境信息。
- 主要内容：
  - Python / PyTorch / CUDA
  - GPU 型号
  - ASE / NumPy / pymatgen / matris 版本
  - GraphConverter 算法
- 适用场景：
  - 建 baseline 时记录环境
- 后续复现实验时核对环境差异

`compare_eval_runs.py`
- 用途：对比 `baseline` 和 `candidate` 两个静态评估结果目录。
- 主要输出：
  - `comparison_summary.json`
  - `comparison_by_group.json`
  - `comparison_rows.jsonl`
  - `comparison_report.md`
- 适用场景：
  - 计算 speedup
  - 计算 energy/force/stress 增量
  - 生成总体表和分组表

`check_force_consistency.py`
- 用途：做轻量平滑性/物理一致性检查。
- 当前包含：
  - 随机扰动测试
  - 有限差分 `F ≈ -dE/dR`
- 主要输出：
  - `force_consistency_records.jsonl`
  - `force_consistency_summary.json`

`run_smoke_relax_md.py`
- 用途：做轻量 `relaxation + MD` smoke test。
- 主要输出：
  - `smoke_relax_md_summary.json`
- 适用场景：
  - 候选版本的快速应用级检查
  - 看是否存在明显异常、崩溃或不可收敛

## 启动脚本

`sh/run_static_eval.sh`
- 固定配置启动 `evaluate_static_metrics.py`

`sh/run_compare_eval.sh`
- 固定配置启动 `compare_eval_runs.py`

`sh/run_force_consistency.sh`
- 固定配置启动 `check_force_consistency.py`

`sh/run_smoke_relax_md.sh`
- 固定配置启动 `run_smoke_relax_md.py`

## 日志说明

当前 `eval/logs/` 目录预留给后续评估日志使用。

## 结果存储建议

静态评估结果建议统一写到项目外部结果目录，例如：

- `MatRIS/results/static_eval/fp32_baseline/`
- `MatRIS/results/static_eval/tf32/`
- `MatRIS/results/static_eval/bf16_autocast/`

这样可以避免与源码目录混在一起。
