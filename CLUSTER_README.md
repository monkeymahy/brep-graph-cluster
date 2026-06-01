# AAG 图聚簇使用指南

针对 60k+ 样本优化的图聚簇工具。

## 版本说明

| 版本 | 文件 | 适用场景 |
|------|------|----------|
| 基础版 | `graph_cluster.py` | 小规模数据 (< 1k)，学习和调试 |
| 优化版 | `graph_cluster_fast.py` | 中等规模 (1k-10k) |
| 大规模版 | `graph_cluster_large.py` | **大规模 (10k+) - 推荐** |
| 高召回版 | `graph_cluster_large_relaxed.py` | **分桶更宽松**，适合“看起来一样但没被去掉”的场景 |

## 快速开始

### 1. 先提取 AAG（如果还没有）

```bash
# 从 STEP 文件夹提取 AAG
python aag_extractor.py --step_path ./steps --output ./output --num_workers 8

# 说明：AAG 提取器默认现在使用旋转不变特征（例如 `FaceCentroidRadiusAttribute`）。
```

### 2. 运行聚簇（推荐使用大规模版本）

```bash
# 基本用法
python graph_cluster_large.py --graphs-json ./output/graphs.json --output ./cluster_result --step-dir ./steps

# 多进程加速
python graph_cluster_large.py --graphs-json ./output/graphs.json --output ./cluster_result --step-dir ./steps --num-workers 16

# 跳过文件复制（先看结果）
python graph_cluster_large.py --graphs-json ./output/graphs.json --output ./cluster_result --skip-copy
```

## 输出目录结构

```
cluster_result/
├── clusters.json      # 完整的聚簇信息
├── file_mapping.json  # 每个文件对应的簇
├── stats.json         # 统计信息
├── cluster_0000/      # 第1个簇（最大）
│   ├── file1.step
│   └── file2.step
├── cluster_0001/      # 第2个簇
├── ...
└── result/            # 每个簇的代表文件
    ├── cluster_0000_file1.step
    └── cluster_0001_file3.step
```

## 性能优化说明

### 为什么这么快？

1. **多级分桶策略** - 只比较可能同构的图
   - 层级1: 节点数 + 边数
   - 层级2: 属性统计
   - 层级3: 属性签名

2. **延迟加载** - 只在需要时转换图格式

3. **内存优化** - 避免同时加载所有图

4. **并行处理** - 多进程处理不同的桶

### 什么时候用高召回版？

如果你确认 AAG 在业务上应该视为重复，但大规模版仍然留下少量未合并的样本，通常是因为分桶条件过严，把同一个 AAG 的不同版本拆散了。此时优先使用 `graph_cluster_large_relaxed.py`。

它会：

1. 只按节点数和边数分桶，减少误拆。
2. 把节点和边属性的比较容差适度放宽，降低浮点抖动带来的漏检。
3. 代价是桶会更大，运行时间可能增加。

### 性能预估（仅供参考）

| 数据规模 | 预估时间 (16核) | 内存使用 |
|----------|-----------------|----------|
| 1k       | < 1 分钟        | < 2GB    |
| 10k      | 5-15 分钟       | < 8GB    |
| 60k      | 1-2 小时        | ~32GB    |

## 完整工作流示例

```bash
# 步骤1: 提取 AAG
python aag_extractor.py --step_path /data/steps --output /data/aag_output --num_workers 16

# 步骤2: 先试运行聚簇（不复制文件）
python graph_cluster_large.py \
    --graphs-json /data/aag_output/graphs.json \
    --output /data/cluster_result \
    --num-workers 32 \
    --skip-copy

# 步骤3: 查看统计结果确认后，再复制文件（可选）
# 重新运行不带 --skip-copy，或手动根据 file_mapping.json 处理
```

## 结果解读

### stats.json

```json
{
  "total_graphs": 60000,
  "num_clusters": 12345,
  "largest_cluster": 256,
  "singleton_clusters": 8000
}
```

### clusters.json

```json
[
  {
    "cluster_id": 0,
    "size": 256,
    "representative": "file_0001",
    "files": ["file_0001", "file_0002", ...]
  }
]
```
