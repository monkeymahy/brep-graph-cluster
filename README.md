# AAG 图聚簇

按几何等价性对 B-rep CAD 零件（STEP 文件）聚类：每个 STEP 先抽取一个属性邻接图（AAG），再把 AAG 同构的文件归为同一簇。同一零件的刚体旋转会落在同一簇，因为 AAG 特征按构造是旋转不变的。

本仓库只做**聚簇**与**文件搬运**，AAG 抽取在别处完成（输入是已抽好的 AAG JSON）。

## 安装

```bash
pip install numpy tqdm networkx
```

## 快速开始

```bash
# 聚簇（目录模式，低内存路径）。--skip-copy 只生成 JSON，不碰 STEP 文件
python graph_cluster.py --input ./aag --output ./cluster_result --skip-copy

# 带 STEP 文件复制到簇目录
python graph_cluster.py --input ./aag --output ./cluster_result --step-dir ./steps

# 多进程
python graph_cluster.py --input ./aag --output ./cluster_result --num-workers 16
```

### 容差

AAG 属性是浮点向量，节点/边匹配用 `np.allclose`：

```bash
# 召回更高（漂移更大的“同一零件”也能合并），但更慢、更易过合并
python graph_cluster.py --input ./aag --output ./cluster_result --atol 1e-2 --rtol 1e-2
```

### 卡死对（VF2 指数回溯）

大图 + 大容差时，少数近似但不同构的对会让 VF2 回溯几天。用步数预算切断：

```bash
# 每对最多 1e6 步可行性检查，超限按不同构处理并记到 timed_out_pairs.json
python graph_cluster.py --input ./aag --output ./cluster_result \
    --atol 1e-2 --rtol 1e-2 --vf2-step-budget 1000000

# 若宁可过合并也不要漏合并，把超时对按同构处理（仍记 timed_out_pairs.json）
python graph_cluster.py --input ./aag --output ./cluster_result \
    --atol 1e-2 --rtol 1e-2 --vf2-step-budget 1000000 --timeout-isomorphic
```

## 预估计算量与时间

正式跑前先看每个桶预过滤后的候选对数，判断大桶是“多样”（预过滤有效）还是“同族”（无损剪枝到头），并可抽样校准每对 VF2 耗时外推总时间：

```bash
python estimate_cluster_cost.py --input ./aag --atol 1e-2 --rtol 1e-2
# 带时间外推（抽样 50 对 VF2 校准）
python estimate_cluster_cost.py --input ./aag --atol 1e-2 --rtol 1e-2 --calibrate 50
```

候选对数是可靠指标；VF2 耗时方差大，外推仅作量级参考。

## STEP 移动/复制（NAS）

STEP 常在 NAS 上、本机写不了。`move_step_by_cluster.py` 不碰文件——读 `clusters.json`，生成纯 POSIX shell 脚本拷到 NAS 执行：

```bash
python move_step_by_cluster.py --clusters ./cluster_result/clusters.json \
    --step-dir /volume1/steps --output /volume1/cluster_result [--copy-only]

# 拷到 NAS 后
sh move_step_by_cluster.sh               # 执行
DRY_RUN=1 sh move_step_by_cluster.sh     # 仅预览
```

默认移动，`--copy-only` 改为复制。代表文件复制到 `result/`，**不带** cluster 前缀（用原文件名）。

## 输入数据格式

### 方式1：目录模式（推荐）

每个文件是一个 AAG JSON：

```
aag/
  3211C01001G70.json
  3211C01001G70__rot_any230p2_00.json
  ...
```

### 方式2：单个 graphs.json

```json
[
  [
    "filename_001",
    {
      "graph": {"edges": [[0, 0, 1], [1, 2, 2]], "num_nodes": 3},
      "graph_face_attr": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
      "graph_edge_attr": [[0.0], [1.0], [0.5]]
    }
  ]
]
```

每文件 AAG JSON 即数据字典本身；`graphs.json` 是 `[filename, data]` 对的列表。两种格式同一个聚簇器都收。

## 输出目录结构

```
cluster_result/
├── clusters.json         # id/size/representative/files
├── file_mapping.json     # 每个文件对应的簇
├── stats.json            # 统计
├── timed_out_pairs.json  # VF2 步数预算超时的对（仅设置 --vf2-step-budget 时）
├── cluster_0000/         # 簇目录（带 STEP 文件时）
├── cluster_0001/
├── ...
└── result/               # 每簇代表文件
```

## 旋转不变性

无需旋转标志位。AAG 特征相对质心表达，而非绝对坐标：
- 面属性含面心到体心的距离；
- 面网格存（点到面心距离、法向·（体心→面心单位向量）、内掩码）；
- 边网格存（点到边心距离、切向/左右法向在点→心单位向量上的投影）。

因此同一零件的不同刚体旋转，AAG 相同，落入同一簇。

## 算法要点

1. **分桶**：按 `(num_nodes, num_edges)`，只比同桶。
2. **预过滤**：度序列精确匹配 + 节点/边属性逐维 mean/min/max/max-abs，置换不变且可证无损（不会漏掉真同构对）。`--no-prefilter` 可关闭做 A/B 验证。
3. **VF2**：候选对用 `nx.isomorphism.GraphMatcher` + `np.allclose` 容差匹配，Union-Find 合并。
4. **进度**：`imap_unordered`，每完成一个桶就刷新进度条（不会因某个慢桶冻住）。
