# AAG 图聚簇

使用 networkx 的 VF2++ 算法对 AAG JSON 数据进行精确图匹配和聚簇。

## 安装

```bash
pip install numpy tqdm networkx
```

## 快速开始

```bash
# 使用大规模版本（推荐）- 目录模式会走低内存路径
python graph_cluster_large.py --input E:\mhy\aagnet\v2\data\SFCAD_4\aag --output .\cluster_result

# 或从单个 graphs.json 文件
python graph_cluster_large.py --input .\graphs.json --output .\cluster_result

# 带 STEP 文件复制
python graph_cluster_large.py --input .\aag --output .\cluster_result --step-dir .\steps

# 移动 STEP 文件到簇目录（替代复制）
python graph_cluster_large.py --input .\aag --output .\cluster_result --step-dir .\steps --move-step

# 多进程加速
python graph_cluster_large.py --input .\aag --output .\cluster_result --num-workers 16

# 跳过文件复制，只生成 JSON 结果（先看结果）
python graph_cluster_large.py --input .\aag --output .\cluster_result --skip-copy
```

## 版本说明

| 版本 | 文件 | 适用场景 |
|------|------|----------|
| 基础版 | `graph_cluster.py` | 小规模数据 (< 1k) |
| 优化版 | `graph_cluster_fast.py` | 中等规模 (1k-10k) |
| 大规模版 | `graph_cluster_large.py` | **大规模 (10k+) - 推荐** |

## 输入数据格式

### 方式1：目录模式（推荐）
每个文件是一个单独的JSON：
```
aag/
  3211C01001G70.json
  3211C01001G70__rot_any230p2_00.json
  ...
```

### 方式2：单个graphs.json
```json
[
  [
    "filename_001",
    {
      "graph": {"edges": [[0, 0, 1], [1, 2, 2]], "num_nodes": 3},
      "graph_face_attr": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
      "graph_face_grid": [],
      "graph_edge_attr": [[0.0], [1.0], [0.5]],
      "graph_edge_grid": []
    }
  ]
]
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

## 性能优化

- 多级分桶策略，避免O(n²)全量比较
- 延迟加载，内存优化
- 多进程并行处理

| 数据规模 | 预估时间 (16核) | 内存使用 |
|----------|-----------------|----------|
| 1k       | < 1 分钟        | < 2GB    |
| 10k      | 5-15 分钟       | < 8GB    |
| 60k      | 1-2 小时        | ~32GB    |

## STEP 移动/复制说明

- 默认复制 STEP 到簇目录。
- 需要节省磁盘空间时用 `--move-step` 直接移动。
- 若希望聚类后再移动（更安全），可用后处理脚本。

```bash
# 根据 clusters.json 移动 STEP 到簇目录（默认移动）
python move_step_by_cluster.py --clusters .\cluster_result\clusters.json --step-dir .\steps --output .\cluster_result

# 只复制，不移动
python move_step_by_cluster.py --clusters .\cluster_result\clusters.json --step-dir .\steps --output .\cluster_result --copy-only
```
