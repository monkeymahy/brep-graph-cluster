"""
AAG Graph Clustering - Optimized Version
针对大规模数据优化的图聚簇（60k+样本）
"""
import argparse
import json
import shutil
import struct
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing.pool import Pool
from multiprocessing import Manager, cpu_count
from tqdm import tqdm

import networkx as nx
import numpy as np


def load_json(json_path):
    with open(json_path, "r") as fp:
        return json.load(fp)


def save_json_data(pathname, data):
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def compute_graph_bucket_key(aag_data: Dict) -> Tuple:
    """
    计算图的桶键 - 用于快速过滤不可能同构的图

    桶键包含：
    - 节点数（面数）
    - 边数
    - 面属性的统计哈希（排序后的）
    - 边属性的统计哈希（排序后的）
    """
    num_nodes = aag_data['graph']['num_nodes']
    src, dst = aag_data['graph']['edges']
    num_edges = len(src)

    # 计算面属性的摘要
    face_attrs = aag_data['graph_face_attr']
    if face_attrs:
        # 对每个面属性计算 hash，排序后合并
        face_hashes = []
        for attr in face_attrs:
            # 将属性转换为 bytes 计算 hash
            attr_bytes = struct.pack(f'{len(attr)}d', *attr)
            face_hashes.append(md5(attr_bytes).digest())
        face_hashes.sort()
        face_summary = md5(b''.join(face_hashes)).digest()
    else:
        face_summary = b''

    # 计算边属性的摘要
    edge_attrs = aag_data['graph_edge_attr']
    if edge_attrs:
        edge_hashes = []
        for attr in edge_attrs:
            attr_bytes = struct.pack(f'{len(attr)}d', *attr)
            edge_hashes.append(md5(attr_bytes).digest())
        edge_hashes.sort()
        edge_summary = md5(b''.join(edge_hashes)).digest()
    else:
        edge_summary = b''

    return (num_nodes, num_edges, face_summary, edge_summary)


def aag_to_networkx_light(aag_data: Dict) -> nx.Graph:
    """
    轻量级转换 - 仅在需要比较时转换
    """
    G = nx.Graph()

    # 添加节点
    num_nodes = aag_data['graph']['num_nodes']
    face_attrs = aag_data['graph_face_attr']

    for i in range(num_nodes):
        G.add_node(i, attr=tuple(face_attrs[i]))

    # 添加边
    src, dst = aag_data['graph']['edges']
    edge_attrs = aag_data['graph_edge_attr']

    for idx, (u, v) in enumerate(zip(src, dst)):
        G.add_edge(u, v, attr=tuple(edge_attrs[idx]))

    return G


def node_match(node1: Dict, node2: Dict) -> bool:
    """节点匹配 - 使用 np.allclose 比较浮点属性"""
    return np.allclose(np.array(node1['attr']), np.array(node2['attr']), atol=1e-6)


def edge_match(edge1: Dict, edge2: Dict) -> bool:
    """边匹配 - 使用 np.allclose 比较浮点属性"""
    return np.allclose(np.array(edge1['attr']), np.array(edge2['attr']), atol=1e-6)


def are_graphs_isomorphic_light(data1: Dict, data2: Dict) -> bool:
    """直接从 AAG 数据比较，避免预转换所有图"""
    try:
        G1 = aag_to_networkx_light(data1)
        G2 = aag_to_networkx_light(data2)
        matcher = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
        return matcher.is_isomorphic()
    except Exception as e:
        print(f"图比较失败: {e}")
        return False


def process_bucket(args: Tuple) -> List[Tuple[str, str]]:
    """处理单个桶内的所有比较 - 多进程任务"""
    bucket_items, graphs_dict = args
    matches = []

    # 桶内使用增量聚簇
    bucket_filenames = [fn for fn, _ in bucket_items]
    if not bucket_filenames:
        return []

    # 每个文件初始化为自己的代表
    representatives = {fn: fn for fn in bucket_filenames}

    def find_rep(fn):
        while representatives[fn] != fn:
            representatives[fn] = representatives[representatives[fn]]
            fn = representatives[fn]
        return fn

    # 对桶内文件进行两两比较
    n = len(bucket_items)
    for i in range(n):
        fn1, data1 = bucket_items[i]
        rep1 = find_rep(fn1)

        for j in range(i + 1, n):
            fn2, data2 = bucket_items[j]
            rep2 = find_rep(fn2)

            if rep1 == rep2:
                continue  # 已经在同一簇

            # 比较图
            if are_graphs_isomorphic_light(data1, data2):
                # 合并
                representatives[rep2] = rep1

    # 收集匹配对
    for fn in bucket_filenames:
        rep = find_rep(fn)
        if fn != rep:
            matches.append((fn, rep))

    return matches


def cluster_graphs_buckets(graphs_data: List[Tuple[str, Dict]], num_workers: int = 1) -> List[List[str]]:
    """
    基于分桶策略的图聚簇

    流程：
    1. 计算每个图的桶键，分桶
    2. 每个桶内进行图比较
    3. 使用 Union-Find 合并簇
    """
    total_graphs = len(graphs_data)
    print(f"总共有 {total_graphs} 个图")

    # 步骤1: 分桶
    print("步骤1: 计算桶键并分桶...")
    buckets = defaultdict(list)
    graphs_dict = {}

    for fn, data in tqdm(graphs_data):
        bucket_key = compute_graph_bucket_key(data)
        buckets[bucket_key].append((fn, data))
        graphs_dict[fn] = data

    print(f"分到 {len(buckets)} 个桶")
    bucket_sizes = [len(items) for items in buckets.values()]
    print(f"桶大小统计: 最大={max(bucket_sizes)}, 最小={min(bucket_sizes)}, 平均={np.mean(bucket_sizes):.1f}")

    # 过滤空桶和单元素桶
    valid_buckets = [items for items in buckets.values() if len(items) > 1]
    print(f"有效桶数（需要比较）: {len(valid_buckets)}")

    # 步骤2: 多进程处理每个桶
    print("步骤2: 桶内图比较...")
    all_matches = []

    if num_workers <= 1:
        # 单进程版本
        for bucket_items in tqdm(valid_buckets):
            matches = process_bucket((bucket_items, graphs_dict))
            all_matches.extend(matches)
    else:
        # 多进程版本
        tasks = [(items, graphs_dict) for items in valid_buckets]
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(process_bucket, tasks), total=len(tasks)))
        for matches in results:
            all_matches.extend(matches)

    # 步骤3: 构建最终簇
    print("步骤3: 构建最终簇...")
    parent = {fn: fn for fn, _ in graphs_data}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root

    # 应用匹配关系
    for fn, rep in all_matches:
        union(fn, rep)

    # 收集簇
    clusters_dict = defaultdict(list)
    for fn, _ in graphs_data:
        root = find(fn)
        clusters_dict[root].append(fn)

    clusters = list(clusters_dict.values())
    # 按大小排序
    clusters.sort(key=lambda x: len(x), reverse=True)

    return clusters


def save_clustering_result(
    clusters: List[List[str]],
    output_dir: Path,
    step_source_dir: Optional[Path] = None,
    copy_files: bool = True
):
    """
    保存聚簇结果 - 优化版本支持大文件列表
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存聚簇信息
    cluster_info = []
    for idx, cluster in enumerate(clusters):
        cluster_info.append({
            "cluster_id": idx,
            "size": len(cluster),
            "files": cluster,
            "representative": cluster[0]
        })

    save_json_data(output_dir / "clusters.json", cluster_info)

    # 保存简化版本（仅簇ID和代表文件映射）
    file_to_cluster = {}
    for idx, cluster in enumerate(clusters):
        rep = cluster[0]
        for fn in cluster:
            file_to_cluster[fn] = {"cluster_id": idx, "representative": rep}
    save_json_data(output_dir / "file_mapping.json", file_to_cluster)

    # 创建簇文件夹
    if copy_files and step_source_dir and step_source_dir.exists():
        print("正在复制文件到簇文件夹...")

        # 创建 result 文件夹
        result_dir = output_dir / "result"
        result_dir.mkdir(exist_ok=True)

        # 分批处理，避免一次性操作太多文件
        for idx, cluster in enumerate(tqdm(clusters)):
            cluster_dir = output_dir / f"cluster_{idx:04d}"
            cluster_dir.mkdir(exist_ok=True)

            # 复制文件
            for fn in cluster:
                for ext in ['.step', '.stp', '.STEP', '.STP']:
                    src_file = step_source_dir / f"{fn}{ext}"
                    if src_file.exists():
                        shutil.copy2(src_file, cluster_dir / src_file.name)
                        break

            # 复制代表文件到 result
            rep_fn = cluster[0]
            for ext in ['.step', '.stp', '.STEP', '.STP']:
                src_file = step_source_dir / f"{rep_fn}{ext}"
                if src_file.exists():
                    dst_name = f"cluster_{idx:04d}_{src_file.name}"
                    shutil.copy2(src_file, result_dir / dst_name)
                    break

    print(f"\n聚簇完成!")
    print(f"  - 总簇数: {len(clusters)}")
    print(f"  - 最大簇大小: {len(clusters[0]) if clusters else 0}")
    print(f"  - 单元素簇: {sum(1 for c in clusters if len(c) == 1)}")


def load_graphs_stream(graphs_json_path: str, max_count: Optional[int] = None) -> List[Tuple[str, Dict]]:
    """流式加载图数据 - 更省内存"""
    print(f"加载图数据: {graphs_json_path}")
    data = load_json(graphs_json_path)
    if max_count:
        data = data[:max_count]
    print(f"已加载 {len(data)} 个图")
    return data


def cluster_from_graphs_json(
    graphs_json_path: str,
    output_dir: str,
    step_source_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    max_count: Optional[int] = None
):
    """
    从 graphs.json 聚簇 - 优化版本
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"使用 {num_workers} 个进程")

    # 加载图数据
    graphs_data = load_graphs_stream(graphs_json_path, max_count)

    # 聚簇
    clusters = cluster_graphs_buckets(graphs_data, num_workers)

    # 保存结果
    step_dir = Path(step_source_dir) if step_source_dir else None
    save_clustering_result(clusters, Path(output_dir), step_dir)

    return clusters


def main():
    parser = argparse.ArgumentParser(description='AAG Graph Clustering - Optimized for large datasets')
    parser.add_argument("--graphs-json", type=str, required=True, help="graphs.json 文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--step-dir", type=str, default=None, help="STEP 文件源目录")
    parser.add_argument("--num-workers", type=int, default=None, help="并行进程数 (默认: CPU数-1)")
    parser.add_argument("--max-count", type=int, default=None, help="最大处理文件数 (用于测试)")

    args = parser.parse_args()

    cluster_from_graphs_json(
        args.graphs_json,
        args.output,
        args.step_dir,
        args.num_workers,
        args.max_count
    )


if __name__ == '__main__':
    main()
