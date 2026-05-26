"""
AAG Graph Clustering - Ultra Large Scale Version
专为 60k+ 样本优化的超大规模图聚簇
"""
import argparse
import json
import shutil
import struct
import pickle
from collections import defaultdict
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from multiprocessing.managers import SyncManager
from tqdm import tqdm

import networkx as nx
import numpy as np


def load_json(json_path):
    with open(json_path, "r") as fp:
        return json.load(fp)


def save_json_data(pathname, data):
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


# ============================================================================
# 阶段1: 超快速预过滤 - 使用多级分桶策略
# ============================================================================

def compute_sketch_signature(attrs: List[List[float]]) -> bytes:
    """计算属性集合的 sketch 签名 - 用于快速去重"""
    if not attrs:
        return b''

    # 对每个属性计算指纹并排序，消除顺序影响
    finger_prints = []
    for attr in attrs:
        # 四舍五入到3位小数，减少浮点误差影响
        rounded = np.round(attr, 3)
        attr_bytes = struct.pack(f'{len(rounded)}d', *rounded)
        finger_prints.append(md5(attr_bytes).digest())

    finger_prints.sort()
    return md5(b''.join(finger_prints)).digest()


def compute_multi_level_bucket_key(aag_data: Dict) -> Tuple:
    """
    计算多级分桶键

    层级1: 节点数 + 边数
    层级2: 面属性维度统计
    层级3: 边属性维度统计
    层级4: 面属性 sketch
    层级5: 边属性 sketch
    """
    num_nodes = aag_data['graph']['num_nodes']
    src, dst = aag_data['graph']['edges']
    num_edges = len(src)

    face_attrs = aag_data['graph_face_attr']
    edge_attrs = aag_data['graph_edge_attr']

    # 层级2: 面属性统计 (每个维度的均值)
    if face_attrs:
        face_array = np.array(face_attrs)
        face_stats = tuple(np.round(face_array.mean(axis=0), 3))
    else:
        face_stats = ()

    # 层级3: 边属性统计
    if edge_attrs:
        edge_array = np.array(edge_attrs)
        edge_stats = tuple(np.round(edge_array.mean(axis=0), 3))
    else:
        edge_stats = ()

    # 层级4-5: sketch 签名
    face_sketch = compute_sketch_signature(face_attrs)
    edge_sketch = compute_sketch_signature(edge_attrs)

    # 返回完整的桶键
    return (
        num_nodes,
        num_edges,
        face_stats,
        edge_stats,
        face_sketch,
        edge_sketch
    )


def _normalize_graph_item(json_path: Path, data: object) -> Tuple[str, Dict]:
    """规范化单个图数据，兼容 [fn, data] 和纯 data 两种格式"""
    if (
        isinstance(data, list)
        and len(data) == 2
        and isinstance(data[0], str)
        and isinstance(data[1], dict)
    ):
        return data[0], data[1]
    return json_path.stem, data  # type: ignore[return-value]


def split_into_buckets(graphs_data: List[Tuple[str, Dict]]) -> Dict[Tuple, List[Tuple[str, Dict]]]:
    """将图分配到多个桶中（内存模式）"""
    buckets = defaultdict(list)

    for fn, data in tqdm(graphs_data, desc="分桶中"):
        bucket_key = compute_multi_level_bucket_key(data)
        buckets[bucket_key].append((fn, data))

    return buckets


def split_dir_into_buckets(input_dir: Path, max_count: Optional[int] = None) -> Tuple[Dict[Tuple, List[Tuple[str, Path]]], List[str]]:
    """目录模式分桶：只保留文件路径，避免加载所有图到内存"""
    buckets: Dict[Tuple, List[Tuple[str, Path]]] = defaultdict(list)
    failed = []

    json_files = sorted(input_dir.glob("*.json"))
    if max_count:
        json_files = json_files[:max_count]

    for json_file in tqdm(json_files, desc="分桶中"):
        try:
            data = load_json(json_file)
            fn, graph_data = _normalize_graph_item(json_file, data)
            bucket_key = compute_multi_level_bucket_key(graph_data)
            buckets[bucket_key].append((fn, json_file))
        except Exception as e:
            print(f"读取 {json_file} 失败: {e}")
            failed.append(json_file.stem)

    return buckets, failed


# ============================================================================
# 阶段2: 图比较 - 优化的同构检测
# ============================================================================

def aag_to_networkx_cached(aag_data: Dict, cache: Dict = None) -> nx.Graph:
    """带缓存的图转换"""
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


def node_match_fast(node1: Dict, node2: Dict) -> bool:
    """快速节点匹配"""
    a1 = np.array(node1['attr'])
    a2 = np.array(node2['attr'])
    return np.allclose(a1, a2, atol=1e-5)


def edge_match_fast(edge1: Dict, edge2: Dict) -> bool:
    """快速边匹配"""
    a1 = np.array(edge1['attr'])
    a2 = np.array(edge2['attr'])
    return np.allclose(a1, a2, atol=1e-5)


def check_isomorphism_pair(data1: Dict, data2: Dict) -> bool:
    """检查一对图是否同构"""
    try:
        # 快速预检
        if (data1['graph']['num_nodes'] != data2['graph']['num_nodes'] or
            len(data1['graph']['edges'][0]) != len(data2['graph']['edges'][0])):
            return False

        # 转换并比较
        G1 = aag_to_networkx_cached(data1)
        G2 = aag_to_networkx_cached(data2)

        matcher = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match_fast, edge_match=edge_match_fast)
        return matcher.is_isomorphic()
    except Exception as e:
        return False


def process_single_bucket_uf(task):
    """处理单个桶，返回簇映射关系（内存模式）"""
    bucket_items = task

    if len(bucket_items) <= 1:
        return {fn: fn for fn, _ in bucket_items}

    filenames = [fn for fn, _ in bucket_items]
    data_list = [data for _, data in bucket_items]
    n = len(filenames)

    # Union-Find 结构
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

    # 两两比较
    for i in range(n):
        root_i = find(i)
        data_i = data_list[i]
        for j in range(i + 1, n):
            root_j = find(j)
            if root_i == root_j:
                continue
            if check_isomorphism_pair(data_i, data_list[j]):
                union(root_i, root_j)

    # 构建映射
    mapping = {}
    for i, fn in enumerate(filenames):
        root = find(i)
        mapping[fn] = filenames[root]

    return mapping


def process_single_bucket_uf_files(task):
    """处理单个桶（文件路径模式），桶内加载，避免常驻内存"""
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
            if check_isomorphism_pair(data_i, data_list[j]):
                union(root_i, root_j)

    mapping = {}
    for i, fn in enumerate(filenames):
        root = find(i)
        mapping[fn] = filenames[root]

    for fn in failed:
        mapping[fn] = fn

    return mapping


def merge_mappings(mappings: List[Dict[str, str]]) -> Dict[str, str]:
    """合并多个桶的映射结果"""
    all_files = set()
    for m in mappings:
        all_files.update(m.keys())

    parent = {fn: fn for fn in all_files}

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

    for m in mappings:
        for fn, rep in m.items():
            if fn != rep:
                union(fn, rep)

    final_mapping = {fn: find(fn) for fn in all_files}
    return final_mapping


def mapping_to_clusters(mapping: Dict[str, str]) -> List[List[str]]:
    """将映射转换为簇列表"""
    clusters_dict = defaultdict(list)
    for fn, rep in mapping.items():
        clusters_dict[rep].append(fn)

    clusters = list(clusters_dict.values())
    clusters.sort(key=lambda x: len(x), reverse=True)
    return clusters


# ============================================================================
# 阶段3: 主流程
# ============================================================================

def cluster_graphs_large_scale(graphs_data: List[Tuple[str, Dict]], num_workers: int = None) -> List[List[str]]:
    """
    大规模图聚簇主流程

    策略:
    1. 多级分桶 - 快速过滤不可能同构的图
    2. 桶内比较 - 只在桶内进行 VF2++ 比较
    3. 并行处理 - 多进程加速
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = len(graphs_data)
    print(f"开始处理 {total} 个图，使用 {num_workers} 个进程")

    # 步骤1: 分桶
    print("\n[1/4] 多级分桶...")
    buckets = split_into_buckets(graphs_data)
    print(f"  分到 {len(buckets)} 个桶")

    # 统计桶大小
    bucket_sizes = [len(items) for items in buckets.values()]
    print(f"  桶大小: 最大={max(bucket_sizes)}, 最小={min(bucket_sizes)}, 平均={np.mean(bucket_sizes):.1f}")

    # 筛选需要处理的桶
    work_buckets = [items for items in buckets.values() if len(items) > 1]
    single_item_buckets = [items for items in buckets.values() if len(items) == 1]
    print(f"  需要处理的桶: {len(work_buckets)}")
    print(f"  单图桶: {len(single_item_buckets)} (直接作为独立簇)")

    # 步骤2: 处理桶
    print("\n[2/4] 桶内图比较...")
    all_mappings = []

    # 添加单图桶的映射
    for items in single_item_buckets:
        fn = items[0][0]
        all_mappings.append({fn: fn})

    # 处理工作桶
    if work_buckets:
        if num_workers == 1:
            for bucket_items in tqdm(work_buckets, desc="处理桶"):
                mapping = process_single_bucket_uf(bucket_items)
                all_mappings.append(mapping)
        else:
            with Pool(processes=num_workers) as pool:
                for mapping in tqdm(pool.imap(process_single_bucket_uf, work_buckets),
                                   total=len(work_buckets), desc="处理桶"):
                    all_mappings.append(mapping)

    # 步骤3: 合并结果
    print("\n[3/4] 合并簇...")
    final_mapping = merge_mappings(all_mappings)

    # 步骤4: 构建最终簇
    print("\n[4/4] 构建最终簇...")
    clusters = mapping_to_clusters(final_mapping)

    return clusters


def cluster_graphs_large_scale_from_dir(input_dir: Path, num_workers: int = None, max_count: Optional[int] = None) -> List[List[str]]:
    """目录模式聚簇：分桶阶段只保留路径，桶内再加载"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    json_files = sorted(input_dir.glob("*.json"))
    if max_count:
        json_files = json_files[:max_count]
    total = len(json_files)
    print(f"开始处理 {total} 个图（目录模式，低内存），使用 {num_workers} 个进程")

    print("\n[1/4] 多级分桶...")
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
                mapping = process_single_bucket_uf_files(bucket_items)
                all_mappings.append(mapping)
        else:
            with Pool(processes=num_workers) as pool:
                for mapping in tqdm(pool.imap(process_single_bucket_uf_files, work_buckets),
                                   total=len(work_buckets), desc="处理桶"):
                    all_mappings.append(mapping)

    print("\n[3/4] 合并簇...")
    final_mapping = merge_mappings(all_mappings)

    print("\n[4/4] 构建最终簇...")
    clusters = mapping_to_clusters(final_mapping)

    return clusters


# ============================================================================
# 结果保存
# ============================================================================

def save_clustering_result(
    clusters: List[List[str]],
    output_dir: Path,
    step_source_dir: Optional[Path] = None,
    copy_files: bool = True,
    move_files: bool = False
):
    """保存聚簇结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存完整簇信息
    print("保存簇信息...")
    cluster_info = []
    for idx, cluster in enumerate(clusters):
        cluster_info.append({
            "cluster_id": idx,
            "size": len(cluster),
            "representative": cluster[0],
            "files": cluster
        })
    save_json_data(output_dir / "clusters.json", cluster_info)

    # 保存文件到簇的映射
    print("保存文件映射...")
    file_mapping = {}
    for idx, cluster in enumerate(clusters):
        rep = cluster[0]
        for fn in cluster:
            file_mapping[fn] = {
                "cluster_id": idx,
                "representative": rep,
                "is_representative": (fn == rep)
            }
    save_json_data(output_dir / "file_mapping.json", file_mapping)

    # 保存统计信息
    stats = {
        "total_graphs": sum(len(c) for c in clusters),
        "num_clusters": len(clusters),
        "largest_cluster": len(clusters[0]) if clusters else 0,
        "cluster_sizes": [len(c) for c in clusters],
        "singleton_clusters": sum(1 for c in clusters if len(c) == 1)
    }
    save_json_data(output_dir / "stats.json", stats)

    # 打印统计
    print(f"\n{'='*60}")
    print(f"聚簇统计")
    print(f"{'='*60}")
    print(f"总图数: {stats['total_graphs']}")
    print(f"簇数量: {stats['num_clusters']}")
    print(f"最大簇: {stats['largest_cluster']}")
    print(f"单图簇: {stats['singleton_clusters']}")
    print(f"\n簇大小分布 (前20):")
    for i, size in enumerate(stats['cluster_sizes'][:20]):
        print(f"  簇 {i}: {size} 个图")

    # 复制文件
    missing_steps = []
    if copy_files and step_source_dir and step_source_dir.exists():
        print(f"\n{'='*60}")
        print("复制文件...")
        print(f"{'='*60}")

        result_dir = output_dir / "result"
        result_dir.mkdir(exist_ok=True)

        for idx, cluster in enumerate(tqdm(clusters, desc="处理簇")):
            # 创建簇文件夹
            cluster_dir = output_dir / f"cluster_{idx:04d}"
            cluster_dir.mkdir(exist_ok=True)

            # 复制/移动该簇的文件
            for fn in cluster:
                src_file = None
                for ext in ['.step', '.stp', '.STEP', '.STP']:
                    candidate = step_source_dir / f"{fn}{ext}"
                    if candidate.exists():
                        src_file = candidate
                        break

                if src_file:
                    dst_file = cluster_dir / src_file.name
                    if move_files:
                        shutil.move(str(src_file), str(dst_file))
                    else:
                        shutil.copy2(src_file, dst_file)
                else:
                    missing_steps.append(fn)

            # 复制代表文件到 result
            rep_fn = cluster[0]
            for ext in ['.step', '.stp', '.STEP', '.STP']:
                src_file = cluster_dir / f"{rep_fn}{ext}"
                if not src_file.exists():
                    src_file = step_source_dir / f"{rep_fn}{ext}"
                if src_file.exists():
                    dst_name = f"cluster_{idx:04d}_{src_file.name}"
                    shutil.copy2(src_file, result_dir / dst_name)
                    break

        if missing_steps:
            print(f"\n缺失 STEP 文件: {len(missing_steps)}")
            save_json_data(output_dir / "missing_steps.json", missing_steps)


def load_graphs_from_dir_or_json(input_path: str, max_count: int = None):
    """从目录或单个JSON加载图数据"""
    input_path = Path(input_path)

    if input_path.is_dir():
        # 目录模式：读取所有JSON文件（注意：此模式会加载所有图到内存）
        print(f"从目录加载图数据: {input_path}")
        data = []
        json_files = sorted(input_path.glob("*.json"))
        print(f"找到 {len(json_files)} 个JSON文件")

        for json_file in tqdm(json_files[:max_count] if max_count else json_files):
            try:
                item = load_json(json_file)
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], dict)
                ):
                    data.append((item[0], item[1]))
                else:
                    data.append((json_file.stem, item))
            except Exception as e:
                print(f"读取 {json_file} 失败: {e}")

        print(f"已加载 {len(data)} 个图")
        return data
    else:
        # 单个JSON文件模式
        print(f"从文件加载图数据: {input_path}")
        data = load_json(input_path)
        if max_count:
            data = data[:max_count]
        print(f"已加载 {len(data)} 个图")
        return data


def cluster_from_graphs_json(
    graphs_json_path: str,
    output_dir: str,
    step_source_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    max_count: Optional[int] = None,
    skip_copy: bool = False,
    move_step: bool = False
):
    """从 graphs_json 或目录聚簇"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(graphs_json_path)
    if input_path.is_dir():
        # 目录模式：低内存路径
        clusters = cluster_graphs_large_scale_from_dir(input_path, num_workers, max_count)
    else:
        # 文件模式：会占用更多内存
        graphs_data = load_graphs_from_dir_or_json(graphs_json_path, max_count=max_count)
        clusters = cluster_graphs_large_scale(graphs_data, num_workers)

    # 保存结果
    step_dir = Path(step_source_dir) if step_source_dir else None
    save_clustering_result(
        clusters,
        output_dir,
        step_dir,
        copy_files=not skip_copy,
        move_files=move_step
    )

    return clusters


def main():
    parser = argparse.ArgumentParser(description='AAG Graph Clustering - Ultra Large Scale Version')
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
        args.move_step
    )


if __name__ == '__main__':
    main()
