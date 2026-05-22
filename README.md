# B-Rep Graph Match - AAG Extractor

从STEP文件中提取几何属性邻接图(AAG, Attributed Adjacency Graph)的独立模块。

## 安装

### 使用conda创建环境

```bash
# 使用提供的environment.yml
conda env create -f environment.yml

# 激活环境
conda activate brep-graph
```

### 或者手动安装

```bash
# 创建环境
conda create -n brep-graph python=3.10
conda activate brep-graph

# 安装依赖
conda install -c conda-forge pythonocc-core=7.5.1 occt=7.5.1 numpy scipy h5py tqdm networkx
pip install torch dgl scikit-learn
pip install git+https://github.com/AutodeskAILab/occwl.git
```

说明：`numba` 在 Windows 下会和 `pythonocc-core=7.5.1` 的 `tbb` 约束冲突，这个项目代码本身也没有直接使用 `numba`，因此不建议放进同一个环境。

## 使用方法

### 命令行使用

```bash
# 单文件处理
python -c "
from aag_extractor import AAGExtractor
from pathlib import Path
extractor = AAGExtractor(Path('path/to/your/file.step'))
result = extractor.process()
print(result)
"

# 批量处理
python aag_extractor.py --step_path ./steps --output ./output --num_workers 4
```

### Python API使用

```python
from pathlib import Path
from aag_extractor import AAGExtractor, extract_aag_from_step
from graph_loader import AAGDataset

# 单个文件提取
extractor = AAGExtractor(Path('example.step'))
aag = extractor.process()
print(f"Nodes: {aag['graph']['num_nodes']}")
print(f"Edges: {len(aag['graph']['edges'][0])}")

# 批量提取
extract_aag_from_step(
    step_path='./steps',
    output_path='./output',
    num_workers=4
)

# 加载提取后的图
dataset = AAGDataset('./output/graphs.json', './output/attr_stat.json')
sample = dataset[0]
print(sample['graph'])
```

## 输出数据结构

```python
{
    'graph': {
        'edges': (src_list, dst_list),  # 边连接关系
        'num_nodes': N                   # 节点数量（面数）
    },
    'graph_face_attr': [...],           # 每个面的属性特征
    'graph_face_grid': [...],           # 每个面的UV网格（可选）
    'graph_edge_attr': [...],           # 每条边的属性特征
    'graph_edge_grid': [...]            # 每条边的UV网格（可选）
}
```

### 面属性 (Face Attributes)

- Plane: 是否为平面
- Cylinder: 是否为圆柱面
- Cone: 是否为圆锥面
- SphereFaceAttribute: 是否为球面
- TorusFaceAttribute: 是否为圆环面
- FaceAreaAttribute: 面积
- RationalNurbsFaceAttribute: 是否为有理NURBS曲面
- FaceCentroidAttribute: 质心坐标(x, y, z)

### 边属性 (Edge Attributes)

- Concave edge: 是否为凹边
- Convex edge: 是否为凸边
- Smooth: 是否为光滑边
- EdgeLengthAttribute: 边长
- CircularEdgeAttribute: 是否为圆弧
- ClosedEdgeAttribute: 是否为封闭边
- EllipticalEdgeAttribute: 是否为椭圆边
- StraightEdgeAttribute: 是否为直线
- 其他曲线类型属性

## 文件结构

```
.
├── aag_extractor.py      # AAG提取核心模块
├── graph_loader.py       # 图数据加载模块
├── schema.json           # 默认属性定义
├── environment.yml       # conda环境配置
└── README.md            # 说明文档
```
