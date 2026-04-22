# Test Perf

本目录用于性能测试与 pipeline profile。

## 文件说明

`benchmark_single_e2e.py`
- 用途：从 ASE 端到端调用 MatRISCalculator，测单结构 latency。
- 测试任务：
  - `e`
  - `ef`
  - `efs`
- 适用场景：
  - 获取最贴近用户调用方式的端到端单结构耗时

`benchmark_batch.py`
- 用途：测试不同 batch size 下的吞吐与显存。
- 测试任务：
  - `e`
  - `ef`
  - `efs`
- 适用场景：
  - 对比不同精度模式下的 batch throughput
  - 统计 peak memory

`benchmark_model_core.py`
- 用途：绕过 ASE 包装，更聚焦模型核心路径的单结构性能。
- 适用场景：
  - 分离端到端 overhead 和模型核心开销
  - 更适合看模型本体加速效果

`profile_pipeline.py`
- 用途：拆分 pipeline 各阶段耗时。
- 主要模块：
  - atoms -> structure
  - graph converter
  - process_graphs
  - embedding
  - interaction blocks
  - energy head
  - force autograd
  - stress autograd
- 适用场景：
  - 定位瓶颈
  - 判断优化收益主要来自哪里

## 日志说明

`logs/benchmark_single_e2e.log`
- 来源：`benchmark_single_e2e.py`
- 内容：ASE 端到端单结构 latency 和显存统计

`logs/benchmark_batch.log`
- 来源：`benchmark_batch.py`
- 内容：不同 batch size 的 throughput、per-structure latency、peak memory

`logs/benchmark_model_core.log`
- 来源：`benchmark_model_core.py`
- 内容：模型核心单结构 latency 和显存统计

`logs/profile_pipeline.log`
- 来源：`profile_pipeline.py`
- 内容：pipeline 各阶段耗时拆分和瓶颈排名

## 使用建议

如果任务重点是：

- 端到端用户体验：优先看 `benchmark_single_e2e.py`
- batch 推理能力：优先看 `benchmark_batch.py`
- 模型核心是否真的加速：优先看 `benchmark_model_core.py`
- 瓶颈定位：优先看 `profile_pipeline.py`
