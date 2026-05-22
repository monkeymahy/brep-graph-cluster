"""
AAG Graph Loader
加载提取好的AAG图数据，支持转换为DGL图
"""
import json
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np
import torch
import dgl


def load_json(json_path):
    with open(json_path, "r") as fp:
        return json.load(fp)


def load_one_graph(fn: str, data: Dict) -> Dict:
    """
    将单个AAG数据转换为DGL图

    Args:
        fn: 文件名
        data: AAG数据字典

    Returns:
        包含DGL图的字典
    """
    edges = tuple(data["graph"]["edges"])
    num_nodes = data["graph"]["num_nodes"]
    dgl_graph = dgl.graph(data=edges, num_nodes=num_nodes)

    node_attributes = np.array(data["graph_face_attr"])
    node_attributes = torch.from_numpy(node_attributes).type(torch.float32)
    dgl_graph.ndata["x"] = node_attributes

    node_grid_attributes = data["graph_face_grid"]
    if len(node_grid_attributes) > 0:
        node_grid_attributes = np.array(node_grid_attributes)
        node_grid_attributes = torch.from_numpy(node_grid_attributes).type(torch.float32)
        dgl_graph.ndata["grid"] = node_grid_attributes

    edge_attributes = np.array(data["graph_edge_attr"])
    edge_attributes = torch.from_numpy(edge_attributes).type(torch.float32)
    dgl_graph.edata["x"] = edge_attributes

    edge_grid_attributes = data["graph_edge_grid"]
    if len(edge_grid_attributes) > 0:
        edge_grid_attributes = np.array(edge_grid_attributes)
        edge_grid_attributes = torch.from_numpy(edge_grid_attributes).type(torch.float32)
        dgl_graph.edata["grid"] = edge_grid_attributes

    return {"graph": dgl_graph, "filename": fn}


def load_statistics(stat_path: Union[str, Path]) -> Dict:
    """
    加载归一化统计数据

    Args:
        stat_path: attr_stat.json路径

    Returns:
        包含均值和标准差的字典
    """
    stat_path = Path(stat_path)
    stat = load_json(stat_path)

    mean_face_attr = torch.tensor(stat["mean_face_attr"], dtype=torch.float32)
    std_face_attr = torch.tensor(stat["std_face_attr"], dtype=torch.float32)
    mean_edge_attr = torch.tensor(stat["mean_edge_attr"], dtype=torch.float32)
    std_edge_attr = torch.tensor(stat["std_edge_attr"], dtype=torch.float32)

    eps = 1e-8
    std_face_attr[std_face_attr < eps] = 1.0
    std_edge_attr[std_edge_attr < eps] = 1.0

    return {
        "mean_face_attr": mean_face_attr,
        "std_face_attr": std_face_attr,
        "mean_edge_attr": mean_edge_attr,
        "std_edge_attr": std_edge_attr
    }


def standardize_graph(sample: Dict, stat: Dict) -> Dict:
    """
    对图数据进行归一化

    Args:
        sample: 包含图的样本
        stat: 统计数据

    Returns:
        归一化后的样本
    """
    sample["graph"].ndata["x"] -= stat["mean_face_attr"]
    sample["graph"].ndata["x"] /= stat["std_face_attr"]
    sample["graph"].edata["x"] -= stat["mean_edge_attr"]
    sample["graph"].edata["x"] /= stat["std_edge_attr"]
    return sample


class AAGDataset:
    """
    AAG数据集类
    """
    def __init__(self, graphs_path: Union[str, Path], stat_path: Optional[Union[str, Path]] = None):
        """
        Args:
            graphs_path: graphs.json路径
            stat_path: 可选的attr_stat.json路径
        """
        self.graphs_path = Path(graphs_path)
        self.stat_path = Path(stat_path) if stat_path else None

        self._load_data()

    def _load_data(self):
        self.graphs_data = load_json(self.graphs_path)
        self.stat = None
        if self.stat_path and self.stat_path.exists():
            self.stat = load_statistics(self.stat_path)

    def __len__(self):
        return len(self.graphs_data)

    def __getitem__(self, idx):
        fn, data = self.graphs_data[idx]
        sample = load_one_graph(fn, data)
        if self.stat:
            sample = standardize_graph(sample, self.stat)
        return sample

    def get_by_filename(self, filename: str):
        """通过文件名获取样本"""
        for fn, data in self.graphs_data:
            if fn == filename:
                sample = load_one_graph(fn, data)
                if self.stat:
                    sample = standardize_graph(sample, self.stat)
                return sample
        return None


def collate_fn(batch: List[Dict]) -> Dict:
    """
    批量数据整理函数，用于DataLoader

    Args:
        batch: 样本列表

    Returns:
        批处理后的字典
    """
    batched_graph = dgl.batch([sample["graph"] for sample in batch])
    filenames = [sample["filename"] for sample in batch]
    return {"graph": batched_graph, "filename": filenames}
