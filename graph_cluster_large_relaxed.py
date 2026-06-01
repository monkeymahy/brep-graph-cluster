"""
AAG Graph Clustering - High Recall Version
针对“看起来相同但数值有轻微漂移”的 AAG 提供更宽松的去重版本
"""
import argparse
import json
import shutil
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
from tqdm import tqdm

from graph_cluster_large import (
    load_json,
    save_json_data,
    merge_mappings,
    mapping_to_clusters,
    save_clustering_result,
    load_graphs_from_dir_or_json,
)


def compute_relaxed_bucket_key(aag_data: Dict) -> Tuple:
    """只按节点数和边数分桶，避免把轻微数值波动拆散到不同桶里。"""
    num_nodes = aag_data['graph']['num_nodes']
    src, dst = aag_data['graph']['edges']
    num_edges = len(src)
    return num_nodes, num_edges


def _normalize_graph_item(json_path: Path, data: object) -> Tuple[str, Dict]:
    """兼容 [fn, data] 和纯 data 两种格式。"""
    if (
        isinstance(data, list)
        and len(data) == 2
        and isinstance(data[0], str)
        and isinstance(data[1], dict)
    ):
        return data[0], data[1]
    return json_path.stem, data  # type: ignore[return-value]


def aag_to_networkx_relaxed(aag_data: Dict) -> nx.Graph:
    """构建 NetworkX 图，供宽松同构检测使用。"""
    G = nx.Graph()

    num_nodes = aag_data['graph']['num_nodes']
    face_attrs = aag_data['graph_face_attr']
    for i in range(num_nodes):
        G.add_node(i, attr=tuple(face_attrs[i]))

    src, dst = aag_data['graph']['edges']
    edge_attrs = aag_data['graph_edge_attr']
    for idx, (u, v) in enumerate(zip(src, dst)):
        G.add_edge(u, v, attr=tuple(edge_attrs[idx]))

    return G


def node_match_relaxed(node1: Dict, node2: Dict) -> bool:
    """更宽松的节点属性比较，降低浮点抖动带来的漏检。"""
    a1 = np.array(node1['attr'])
    a2 = np.array(node2['attr'])
    return np.allclose(a1, a2, atol=1e-4, rtol=1e-5)


def edge_match_relaxed(edge1: Dict, edge2: Dict) -> bool:
    """更宽松的边属性比较，降低浮点抖动带来的漏检。"""
    a1 = np.array(edge1['attr'])
    a2 = np.array(edge2['attr'])
    return np.allclose(a1, a2, atol=1e-4, rtol=1e-5)


def check_isomorphism_pair_relaxed(data1: Dict, data2: Dict) -> bool:
    """检查一对图是否同构。"""
    try:
        if (
            data1['graph']['num_nodes'] != data2['graph']['num_nodes']
            or len(data1['graph']['edges'][0]) != len(data2['graph']['edges'][0])
        ):
            return False

        G1 = aag_to_networkx_relaxed(data1)
        G2 = aag_to_networkx_relaxed(data2)
        matcher = nx.isomorphism.GraphMatcher(
            G1,
            G2,
            node_match=node_match_relaxed,
            edge_match=edge_match_relaxed,
        )
        return matcher.is_isomorphic()
    except Exception:
        return False


def split_into_buckets(graphs_data: List[Tuple[str, Dict]]) -> Dict[Tuple, List[Tuple[str, Dict]]]:
    """将图分配到多个桶中（高召回版）。"""
    buckets = defaultdict(list)
    for fn, data in tqdm(graphs_data, desc="分桶中"):
        bucket_key = compute_relaxed_bucket_key(data)
        buckets[bucket_key].append((fn, data))
    return buckets


def split_dir_into_buckets(input_dir: Path, max_count: Optional[int] = None) -> Tuple[Dict[Tuple, List[Tuple[str, Path]]], List[str]]:
    """目录模式分桶：只保留文件路径，避免加载所有图到内存。"""
    buckets: Dict[Tuple, List[Tuple[str, Path]]] = defaultdict(list)
    failed = []

    json_files = sorted(input_dir.glob("*.json"))
    if max_count:
        json_files = json_files[:max_count]

    for json_file in tqdm(json_files, desc="分桶中"):
        try:
            data = load_json(json_file)
            fn, graph_data = _normalize_graph_item(json_file, data)
            bucket_key = compute_relaxed_bucket_key(graph_data)
            buckets[bucket_key].append((fn, json_file))
        except Exception as e:
            print(f"读取 {json_file} 失败: {e}")
            failed.append(json_file.stem)

    return buckets, failed


def process_single_bucket_uf_relaxed(task):
    """处理单个桶，返回簇映射关系（内存模式）。"""
    bucket_items = task

    if len(bucket_items) <= 1:
        return {fn: fn for fn, _ in bucket_items}

    filenames = [fn for fn, _ in bucket_items]
    data_list = [data for _, data in bucket_items]
    n = len(filenames)

    parent = list(range(n))

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

    for i in range(n):
        root_i = find(i)
        data_i = data_list[i]
        for j in range(i + 1, n):
            root_j = find(j)
            if root_i == root_j:
                continue
            if check_isomorphism_pair_relaxed(data_i, data_list[j]):
                union(root_i, root_j)

    mapping = {}
    for i, fn in enumerate(filenames):
        root = find(i)
        mapping[fn] = filenames[root]

    return mapping


def process_single_bucket_uf_files_relaxed(task):
    """处理单个桶（文件路径模式），桶内加载，避免常驻内存。"""
    bucket_items = task
    if len(bucket_items) <= 1:
        return {fn: fn for fn, _ in bucket_items}

    loaded_items: List[Tuple[str, Dict]] = []
    failed = []
    for fn, json_path in bucket_items:
        try:
            data = load_json(json_path)
            _, graph_data = _normalize_graph_item(json_path, data)
            loaded_items.append((fn, graph_data))
        except Exception as e:
            print(f"读取 {json_path} 失败: {e}")
            failed.append(fn)

    if len(loaded_items) <= 1:
        mapping = {fn: fn for fn, _ in loaded_items}
        mapping.update({fn: fn for fn in failed})
        return mapping

    filenames = [fn for fn, _ in loaded_items]
    data_list = [data for _, data in loaded_items]
    n = len(filenames)

    parent = list(range(n))

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

    for i in range(n):
        root_i = find(i)
        data_i = data_list[i]
        for j in range(i + 1, n):
            root_j = find(j)
            if root_i == root_j:
                continue
            if check_isomorphism_pair_relaxed(data_i, data_list[j]):
                union(root_i, root_j)

    mapping = {}
    for i, fn in enumerate(filenames):
        root = find(i)
        mapping[fn] = filenames[root]

    for fn in failed:
        mapping[fn] = fn

    return mapping


def cluster_graphs_large_scale(graphs_data: List[Tuple[str, Dict]], num_workers: int = None) -> List[List[str]]:
    """高召回版大规模图聚簇主流程。"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = len(graphs_data)
    print(f"开始处理 {total} 个图，使用 {num_workers} 个进程")

    print("\n[1/4] 分桶（高召回模式）...")
    buckets = split_into_buckets(graphs_data)
    print(f"  分到 {len(buckets)} 个桶")

    bucket_sizes = [len(items) for items in buckets.values()]
    print(f"  桶大小: 最大={max(bucket_sizes)}, 最小={min(bucket_sizes)}, 平均={np.mean(bucket_sizes):.1f}")

    work_buckets = [items for items in buckets.values() if len(items) > 1]
    single_item_buckets = [items for items in buckets.values() if len(items) == 1]
    print(f"  需要处理的桶: {len(work_buckets)}")
    print(f"  单图桶: {len(single_item_buckets)} (直接作为独立簇)")

    print("\n[2/4] 桶内图比较...")
    all_mappings = []

    for items in single_item_buckets:
        fn = items[0][0]
        all_mappings.append({fn: fn})

    if work_buckets:
        if num_workers == 1:
            for bucket_items in tqdm(work_buckets, desc="处理桶"):
                mapping = process_single_bucket_uf_relaxed(bucket_items)
                all_mappings.append(mapping)
        else:
            with Pool(processes=num_workers) as pool:
                for mapping in tqdm(pool.imap(process_single_bucket_uf_relaxed, work_buckets), total=len(work_buckets), desc="处理桶"):
                    all_mappings.append(mapping)

    print("\n[3/4] 合并簇...")
    final_mapping = merge_mappings(all_mappings)

    print("\n[4/4] 构建最终簇...")
    clusters = mapping_to_clusters(final_mapping)
    return clusters


def cluster_graphs_large_scale_from_dir(input_dir: Path, num_workers: int = None, max_count: Optional[int] = None) -> List[List[str]]:
    """目录模式聚簇：分桶阶段只保留路径，桶内再加载。"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    json_files = sorted(input_dir.glob("*.json"))
    if max_count:
        json_files = json_files[:max_count]
    total = len(json_files)
    print(f"开始处理 {total} 个图（目录模式，高召回），使用 {num_workers} 个进程")

    print("\n[1/4] 分桶（高召回模式）...")
    buckets, failed = split_dir_into_buckets(input_dir, max_count=max_count)
    print(f"  分到 {len(buckets)} 个桶")

    if buckets:
        bucket_sizes = [len(items) for items in buckets.values()]
        print(f"  桶大小: 最大={max(bucket_sizes)}, 最小={min(bucket_sizes)}, 平均={np.mean(bucket_sizes):.1f}")
    else:
        print("  桶大小: 无有效桶")

    work_buckets = [items for items in buckets.values() if len(items) > 1]
    single_item_buckets = [items for items in buckets.values() if len(items) == 1]
    print(f"  需要处理的桶: {len(work_buckets)}")
    print(f"  单图桶: {len(single_item_buckets)} (直接作为独立簇)")
    if failed:
        print(f"  读取失败: {len(failed)} 个文件（将作为单图簇）")

    print("\n[2/4] 桶内图比较...")
    all_mappings = []

    for items in single_item_buckets:
        fn = items[0][0]
        all_mappings.append({fn: fn})

    for fn in failed:
        all_mappings.append({fn: fn})

    if work_buckets:
        if num_workers == 1:
            for bucket_items in tqdm(work_buckets, desc="处理桶"):
                mapping = process_single_bucket_uf_files_relaxed(bucket_items)
                all_mappings.append(mapping)
        else:
            with Pool(processes=num_workers) as pool:
                for mapping in tqdm(pool.imap(process_single_bucket_uf_files_relaxed, work_buckets), total=len(work_buckets), desc="处理桶"):
                    all_mappings.append(mapping)

    print("\n[3/4] 合并簇...")
    final_mapping = merge_mappings(all_mappings)

    print("\n[4/4] 构建最终簇...")
    clusters = mapping_to_clusters(final_mapping)
    return clusters


def cluster_from_graphs_json(
    graphs_json_path: str,
    output_dir: str,
    step_source_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    max_count: Optional[int] = None,
    skip_copy: bool = False,
    move_step: bool = False,
):
    """从 graphs_json 或目录聚簇。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(graphs_json_path)
    if input_path.is_dir():
        clusters = cluster_graphs_large_scale_from_dir(input_path, num_workers, max_count)
    else:
        graphs_data = load_graphs_from_dir_or_json(graphs_json_path, max_count=max_count)
        clusters = cluster_graphs_large_scale(graphs_data, num_workers)

    step_dir = Path(step_source_dir) if step_source_dir else None
    save_clustering_result(
        clusters,
        output_dir,
        step_dir,
        copy_files=not skip_copy,
        move_files=move_step,
    )

    return clusters


def main():
    parser = argparse.ArgumentParser(description='AAG Graph Clustering - High Recall Version')
    parser.add_argument("--input", type=str, required=True, help="AAG目录或单个graphs.json文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--step-dir", type=str, default=None, help="STEP 文件源目录 (可选)")
    parser.add_argument("--num-workers", type=int, default=None, help="并行进程数 (默认: CPU数-1)")
    parser.add_argument("--max-count", type=int, default=None, help="最大处理文件数 (用于测试)")
    parser.add_argument("--skip-copy", action="store_true", help="跳过文件复制，只生成 json 结果")
    parser.add_argument("--move-step", action="store_true", help="将 STEP 文件移动到簇目录")

    args = parser.parse_args()

    cluster_from_graphs_json(
        args.input,
        args.output,
        args.step_dir,
        args.num_workers,
        args.max_count,
        args.skip_copy,
        args.move_step,
    )


if __name__ == '__main__':
    main()