# Test Data

本目录用于测试数据集的构建与增强。

## 文件说明

`download_mp_testset.py`
- 用途：从 Materials Project 下载一批母结构，作为测试集的平衡结构来源。
- 主要输出：
  - `mp_testset/raw_structures/`
  - `mp_testset/metadata/mp_testset_metadata.csv`
- 关注点：
  - 控制材料体系、晶系、磁性等覆盖范围
  - 生成后续增强脚本的输入元数据

`generate_perturbed_structures.py`
- 用途：基于母结构生成测试集增强版本。
- 主要输出：
  - `mp_testset/equilibrium/`
  - `mp_testset/perturbed_small/`
  - `mp_testset/strained_high/`
  - `mp_testset/md_initial/`
  - `mp_testset/metadata/mp_testset_augmented_metadata.csv`
- 关注点：
  - 生成任务 7 需要的四类结构
  - 为后续静态评估、relaxation、MD 测试提供统一输入

## 结果存储

本目录下的脚本本身不在 `test/data/` 内生成日志文件。
数据结果统一写到：

- `/home/lht/lab/mp_testset/`

其中最关键的文件是：

- `metadata/mp_testset_metadata.csv`
- `metadata/mp_testset_augmented_metadata.csv`

这两个元数据文件会被后续 `test/eval/` 中的脚本读取。
