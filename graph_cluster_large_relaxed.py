"""
AAG Graph Clustering - High Recall Version
针对“看起来相同但数值有轻微漂移”的 AAG 提供更宽松的去重版本
"""
import argparse
import json
import shutil
from collections import defaultdict
from functools import partial
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


DEFAULT_ATOL = 1e-3
DEFAULT_RTOL = 1e-4


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
    """构建 NetworkX 图，供宽松同构检测使用。

    属性以 ndarray 缓存，避免 VF2 每次匹配都重复 np.array 转换。
    """
    G = nx.Graph()

    num_nodes = aag_data['graph']['num_nodes']
    face_attrs = aag_data['graph_face_attr']
    for i in range(num_nodes):
        G.add_node(i, attr=np.asarray(face_attrs[i], dtype=np.float64))

    src, dst = aag_data['graph']['edges']
    edge_attrs = aag_data['graph_edge_attr']
    for idx, (u, v) in enumerate(zip(src, dst)):
        G.add_edge(u, v, attr=np.asarray(edge_attrs[idx], dtype=np.float64))

    return G


def node_match_relaxed(node1: Dict, node2: Dict, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    """更宽松的节点属性比较，降低浮点抖动带来的漏检。attr 已是 ndarray，直接比较。"""
    return np.allclose(node1['attr'], node2['attr'], atol=atol, rtol=rtol)


def edge_match_relaxed(edge1: Dict, edge2: Dict, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    """更宽松的边属性比较，降低浮点抖动带来的漏检。attr 已是 ndarray，直接比较。"""
    return np.allclose(edge1['attr'], edge2['attr'], atol=atol, rtol=rtol)


class _VF2BudgetExceeded(Exception):
    """单对 VF2 比较步数预算超限。"""


class _BudgetedGraphMatcher(nx.isomorphism.GraphMatcher):
    """GraphMatcher 子类：步数预算超限抛异常，协作式中断 VF2。

    Windows 下纯 Python 的 VF2 死循环线程杀不掉、整桶进程杀掉代价太大，
    所以走协作式——在 VF2 的可行性检查钩子里计数，超限即抛异常 unwind 整个搜索，
    只放弃这一对（不影响桶内其它对的比较）。
    """

    def __init__(self, G1, G2, node_match, edge_match, step_budget: int):
        super().__init__(G1, G2, node_match=node_match, edge_match=edge_match)
        self._step_budget = step_budget
        self._steps = 0

    def syntactic_feasibility(self, G1_node, G2_node):
        self._steps += 1
        if self._steps > self._step_budget:
            raise _VF2BudgetExceeded()
        return super().syntactic_feasibility(G1_node, G2_node)


def _vf2_isomorphic(G1: nx.Graph, G2: nx.Graph, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL, step_budget: Optional[int] = None):
    """对两个已构建的图跑 VF2 同构检测（带属性容差）。

    返回三态：True=同构, False=不同构, None=步数预算超时（无法判定，保守按不同构处理）。
    """
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False
    node_match = partial(node_match_relaxed, atol=atol, rtol=rtol)
    edge_match = partial(edge_match_relaxed, atol=atol, rtol=rtol)
    if step_budget:
        matcher = _BudgetedGraphMatcher(G1, G2, node_match, edge_match, step_budget)
    else:
        matcher = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
    try:
        return matcher.is_isomorphic()
    except _VF2BudgetExceeded:
        return None


def check_isomorphism_pair_relaxed(data1: Dict, data2: Dict, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL) -> bool:
    """检查一对图是否同构（向后兼容包装：内部构建图后调 _vf2_isomorphic）。"""
    try:
        if (
            data1['graph']['num_nodes'] != data2['graph']['num_nodes']
            or len(data1['graph']['edges'][0]) != len(data2['graph']['edges'][0])
        ):
            return False
        G1 = aag_to_networkx_relaxed(data1)
        G2 = aag_to_networkx_relaxed(data2)
        return _vf2_isomorphic(G1, G2, atol=atol, rtol=rtol)
    except Exception:
        return False


# ============================================================================
# 预过滤：置换不变统计量 + 候选矩阵（保正确性，零漏判）
# ============================================================================

PREFILTER_SAFETY = 2.0


def _stats_block(arr) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从属性矩阵算 [mean,min,max] (3,D) 与 max-abs (D,)。空或非法返回 (None,None)。"""
    if arr is None:
        return None, None
    try:
        a = np.asarray(arr, dtype=np.float64)
    except Exception:
        return None, None
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.size == 0 or a.ndim != 2:
        return None, None
    stats = np.stack([a.mean(axis=0), a.min(axis=0), a.max(axis=0)])  # (3, D)
    maxabs = np.max(np.abs(a), axis=0)  # (D,)
    return stats, maxabs


def compute_invariant_stats(aag_data: Dict) -> Optional[Dict]:
    """计算单个图的置换不变统计量。

    返回 dict（degree_seq, node_stats(3,D), node_maxabs(D,), edge_*）；
    空属性/空边的块为 None。整体失败返回 None（调用方按保守处理）。

    这些量都是同构的必要条件：同构图必有相同的排序度序列与相同的属性统计量
    （在 atol/rtol 容差内），因此预过滤不会漏掉任何真正的同构对。
    """
    try:
        num_nodes = aag_data['graph']['num_nodes']
        src, dst = aag_data['graph']['edges']

        deg = np.zeros(num_nodes, dtype=np.int64)
        for u, v in zip(src, dst):
            deg[u] += 1
            deg[v] += 1
        degree_seq = np.sort(deg)

        node_stats, node_maxabs = _stats_block(aag_data.get('graph_face_attr'))
        edge_stats, edge_maxabs = _stats_block(aag_data.get('graph_edge_attr'))

        return {
            'degree_seq': degree_seq,
            'node_stats': node_stats,
            'node_maxabs': node_maxabs,
            'edge_stats': edge_stats,
            'edge_maxabs': edge_maxabs,
        }
    except Exception:
        return None


def _group_match_matrix(stats_list: List[Optional[Dict]], stats_key: str, maxabs_key: str,
                        atol: float, rtol: float, safety: float = PREFILTER_SAFETY) -> Optional[np.ndarray]:
    """构建单个统计块的 (n,n) bool 矩阵：True 表示该块不排除同构。

    全块缺失返回 None（跳过）。缺某个图的该块 ⇒ 与其他图保守判 True。
    阈值 |s1-s2| <= safety*(atol + rtol*max(m1,m2))，rtol 参考用 max-abs（属性可负时安全）。
    """
    n = len(stats_list)
    valid = []  # (idx, S(3,D), M(D,))
    for idx, s in enumerate(stats_list):
        if s is None:
            continue
        st = s.get(stats_key)
        ma = s.get(maxabs_key)
        if st is None or ma is None:
            continue
        valid.append((idx, np.asarray(st, dtype=np.float64), np.asarray(ma, dtype=np.float64)))
    if not valid:
        return None

    ok = np.ones((n, n), dtype=bool)
    # 按 D 分组：维度不同的对必不同构（属性维度不同 ⇒ VF2 本会抛 False）
    groups: Dict[int, List[Tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for idx, st, ma in valid:
        groups[int(st.shape[1])].append((idx, st, ma))
    keys = list(groups.keys())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ia_idx = [it[0] for it in groups[keys[a]]]
            ib_idx = [it[0] for it in groups[keys[b]]]
            ok[np.ix_(ia_idx, ib_idx)] = False
            ok[np.ix_(ib_idx, ia_idx)] = False

    # 组内向量化比较（逐维循环，内存安全）
    for D, items in groups.items():
        k = len(items)
        idxs = np.array([it[0] for it in items])
        S = np.stack([it[1] for it in items])  # (k,3,D)
        M = np.stack([it[2] for it in items])  # (k,D)
        block_ok = np.ones((k, k), dtype=bool)
        for d in range(D):
            col = S[:, :, d]  # (k,3)
            md = np.maximum(M[:, d][:, None], M[:, d][None, :])  # (k,k)
            thr = safety * (atol + rtol * md)  # (k,k)
            diff = np.abs(col[:, None, :] - col[None, :, :])  # (k,k,3)
            block_ok &= np.all(diff <= thr[:, :, None], axis=2)
        not_ok = np.tril(~block_ok, -1)  # 下三角中“不通过”的对
        a_idx, b_idx = np.where(not_ok)
        if len(a_idx):
            ok[idxs[a_idx], idxs[b_idx]] = False
            ok[idxs[b_idx], idxs[a_idx]] = False
    return ok


def _degree_match_matrix(stats_list: List[Optional[Dict]]) -> Optional[np.ndarray]:
    """度序列精确匹配的 (n,n) bool 矩阵。度是整数结构量，同构图必相等。"""
    n = len(stats_list)
    valid = [(idx, s['degree_seq']) for idx, s in enumerate(stats_list)
             if s is not None and s.get('degree_seq') is not None]
    if not valid:
        return None
    ok = np.ones((n, n), dtype=bool)
    groups: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for idx, deg in valid:
        groups[int(len(deg))].append((idx, np.asarray(deg)))
    keys = list(groups.keys())
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ia_idx = [it[0] for it in groups[keys[a]]]
            ib_idx = [it[0] for it in groups[keys[b]]]
            ok[np.ix_(ia_idx, ib_idx)] = False
            ok[np.ix_(ib_idx, ia_idx)] = False
    for L, items in groups.items():
        k = len(items)
        idxs = np.array([it[0] for it in items])
        D = np.stack([it[1] for it in items])  # (k, L)
        match = np.all(D[:, None, :] == D[None, :, :], axis=2)  # (k,k)
        not_ok = np.tril(~match, -1)
        a_idx, b_idx = np.where(not_ok)
        if len(a_idx):
            ok[idxs[a_idx], idxs[b_idx]] = False
            ok[idxs[b_idx], idxs[a_idx]] = False
    return ok


def build_candidate_matrix(stats_list: List[Optional[Dict]], atol: float, rtol: float) -> np.ndarray:
    """构建桶内候选对矩阵。cand[i,j]=True ⇒ (i,j) 可能同构，需走 VF2。

    保守原则：任何统计块缺失（空属性/计算失败）都不参与否决（视为候选），
    让 VF2 做最终裁决。因此预过滤只会跳过“必不同构”的对，结果与全 VF2 一致。
    """
    n = len(stats_list)
    cand = np.ones((n, n), dtype=bool)
    np.fill_diagonal(cand, False)
    blocks = []
    for b in (
        _degree_match_matrix(stats_list),
        _group_match_matrix(stats_list, 'node_stats', 'node_maxabs', atol, rtol),
        _group_match_matrix(stats_list, 'edge_stats', 'edge_maxabs', atol, rtol),
    ):
        if b is not None:
            blocks.append(b)
    if blocks:
        cand &= np.logical_and.reduce(blocks)
    np.fill_diagonal(cand, False)
    return cand


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


def process_single_bucket_uf_relaxed(task, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL, use_prefilter: bool = True, step_budget: Optional[int] = None, timeout_isomorphic: bool = False):
    """处理单个桶，返回 (簇映射关系, 步数预算超时对列表)（内存模式）。"""
    bucket_items = task

    if len(bucket_items) <= 1:
        return {fn: fn for fn, _ in bucket_items}, []

    filenames = [fn for fn, _ in bucket_items]
    data_list = [data for _, data in bucket_items]
    n = len(filenames)

    # 每个图只构建一次 nx.Graph（缓存 ndarray 属性），配对时复用
    G_list = [aag_to_networkx_relaxed(d) for d in data_list]

    # 置换不变预过滤：跳过“必不同构”的对，只对候选对跑 VF2（零漏判，结果不变）
    if use_prefilter:
        stats_list = [compute_invariant_stats(d) for d in data_list]
        cand = build_candidate_matrix(stats_list, atol, rtol)
    else:
        cand = None

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

    timed_out: List[Tuple[str, str]] = []
    for i in range(n):
        root_i = find(i)
        Gi = G_list[i]
        for j in range(i + 1, n):
            if cand is not None and not cand[i, j]:
                continue
            root_j = find(j)
            if root_i == root_j:
                continue
            res = _vf2_isomorphic(Gi, G_list[j], atol=atol, rtol=rtol, step_budget=step_budget)
            if res is True:
                union(root_i, root_j)
            elif res is None:
                timed_out.append((filenames[i], filenames[j]))
                if timeout_isomorphic:
                    union(root_i, root_j)

    mapping = {}
    for i, fn in enumerate(filenames):
        root = find(i)
        mapping[fn] = filenames[root]

    return mapping, timed_out


def process_single_bucket_uf_files_relaxed(task, atol: float = DEFAULT_ATOL, rtol: float = DEFAULT_RTOL, use_prefilter: bool = True, step_budget: Optional[int] = None, timeout_isomorphic: bool = False):
    """处理单个桶（文件路径模式），桶内加载，避免常驻内存。返回 (映射, 超时对列表)。"""
    bucket_items = task
    if len(bucket_items) <= 1:
        return {fn: fn for fn, _ in bucket_items}, []

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
        return mapping, []

    filenames = [fn for fn, _ in loaded_items]
    data_list = [data for _, data in loaded_items]
    n = len(filenames)

    # 每个图只构建一次 nx.Graph（缓存 ndarray 属性），配对时复用
    G_list = [aag_to_networkx_relaxed(d) for d in data_list]

    # 置换不变预过滤：跳过“必不同构”的对，只对候选对跑 VF2（零漏判，结果不变）
    if use_prefilter:
        stats_list = [compute_invariant_stats(d) for d in data_list]
        cand = build_candidate_matrix(stats_list, atol, rtol)
    else:
        cand = None

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

    timed_out: List[Tuple[str, str]] = []
    for i in range(n):
        root_i = find(i)
        Gi = G_list[i]
        for j in range(i + 1, n):
            if cand is not None and not cand[i, j]:
                continue
            root_j = find(j)
            if root_i == root_j:
                continue
            res = _vf2_isomorphic(Gi, G_list[j], atol=atol, rtol=rtol, step_budget=step_budget)
            if res is True:
                union(root_i, root_j)
            elif res is None:
                timed_out.append((filenames[i], filenames[j]))
                if timeout_isomorphic:
                    union(root_i, root_j)

    mapping = {}
    for i, fn in enumerate(filenames):
        root = find(i)
        mapping[fn] = filenames[root]

    for fn in failed:
        mapping[fn] = fn

    return mapping, timed_out


def cluster_graphs_large_scale(
    graphs_data: List[Tuple[str, Dict]],
    num_workers: int = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    use_prefilter: bool = True,
    step_budget: Optional[int] = None,
    timeout_isomorphic: bool = False,
) -> Tuple[List[List[str]], List[Tuple[str, str]]]:
    """高召回版大规模图聚簇主流程。返回 (簇列表, 步数预算超时对列表)。"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    total = len(graphs_data)
    print(f"开始处理 {total} 个图，使用 {num_workers} 个进程，预过滤: {'开启' if use_prefilter else '关闭'}，步数预算: {step_budget if step_budget else '无限'}，超时按: {'同构' if timeout_isomorphic else '不同构'}")

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
    all_timed_out: List[Tuple[str, str]] = []

    for items in single_item_buckets:
        fn = items[0][0]
        all_mappings.append({fn: fn})

    if work_buckets:
        if num_workers == 1:
            for bucket_items in tqdm(work_buckets, desc="处理桶"):
                mapping, timed_out = process_single_bucket_uf_relaxed(bucket_items, atol=atol, rtol=rtol, use_prefilter=use_prefilter, step_budget=step_budget, timeout_isomorphic=timeout_isomorphic)
                all_mappings.append(mapping)
                all_timed_out.extend(timed_out)
        else:
            worker = partial(process_single_bucket_uf_relaxed, atol=atol, rtol=rtol, use_prefilter=use_prefilter, step_budget=step_budget, timeout_isomorphic=timeout_isomorphic)
            with Pool(processes=num_workers) as pool:
                for mapping, timed_out in tqdm(pool.imap_unordered(worker, work_buckets), total=len(work_buckets), desc="处理桶"):
                    all_mappings.append(mapping)
                    all_timed_out.extend(timed_out)

    print("\n[3/4] 合并簇...")
    final_mapping = merge_mappings(all_mappings)

    print("\n[4/4] 构建最终簇...")
    clusters = mapping_to_clusters(final_mapping)
    return clusters, all_timed_out


def cluster_graphs_large_scale_from_dir(
    input_dir: Path,
    num_workers: int = None,
    max_count: Optional[int] = None,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    use_prefilter: bool = True,
    step_budget: Optional[int] = None,
    timeout_isomorphic: bool = False,
) -> Tuple[List[List[str]], List[Tuple[str, str]]]:
    """目录模式聚簇：分桶阶段只保留路径，桶内再加载。返回 (簇列表, 超时对列表)。"""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    json_files = sorted(input_dir.glob("*.json"))
    if max_count:
        json_files = json_files[:max_count]
    total = len(json_files)
    print(f"开始处理 {total} 个图（目录模式，高召回），使用 {num_workers} 个进程，预过滤: {'开启' if use_prefilter else '关闭'}，步数预算: {step_budget if step_budget else '无限'}，超时按: {'同构' if timeout_isomorphic else '不同构'}")

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
    all_timed_out: List[Tuple[str, str]] = []

    for items in single_item_buckets:
        fn = items[0][0]
        all_mappings.append({fn: fn})

    for fn in failed:
        all_mappings.append({fn: fn})

    if work_buckets:
        if num_workers == 1:
            for bucket_items in tqdm(work_buckets, desc="处理桶"):
                mapping, timed_out = process_single_bucket_uf_files_relaxed(bucket_items, atol=atol, rtol=rtol, use_prefilter=use_prefilter, step_budget=step_budget, timeout_isomorphic=timeout_isomorphic)
                all_mappings.append(mapping)
                all_timed_out.extend(timed_out)
        else:
            worker = partial(process_single_bucket_uf_files_relaxed, atol=atol, rtol=rtol, use_prefilter=use_prefilter, step_budget=step_budget, timeout_isomorphic=timeout_isomorphic)
            with Pool(processes=num_workers) as pool:
                for mapping, timed_out in tqdm(pool.imap_unordered(worker, work_buckets), total=len(work_buckets), desc="处理桶"):
                    all_mappings.append(mapping)
                    all_timed_out.extend(timed_out)

    print("\n[3/4] 合并簇...")
    final_mapping = merge_mappings(all_mappings)

    print("\n[4/4] 构建最终簇...")
    clusters = mapping_to_clusters(final_mapping)
    return clusters, all_timed_out


def cluster_from_graphs_json(
    graphs_json_path: str,
    output_dir: str,
    step_source_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    max_count: Optional[int] = None,
    skip_copy: bool = False,
    move_step: bool = False,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    use_prefilter: bool = True,
    step_budget: Optional[int] = None,
    timeout_isomorphic: bool = False,
):
    """从 graphs_json 或目录聚簇。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(graphs_json_path)
    if input_path.is_dir():
        clusters, timed_out = cluster_graphs_large_scale_from_dir(input_path, num_workers, max_count, atol=atol, rtol=rtol, use_prefilter=use_prefilter, step_budget=step_budget, timeout_isomorphic=timeout_isomorphic)
    else:
        graphs_data = load_graphs_from_dir_or_json(graphs_json_path, max_count=max_count)
        clusters, timed_out = cluster_graphs_large_scale(graphs_data, num_workers, atol=atol, rtol=rtol, use_prefilter=use_prefilter, step_budget=step_budget, timeout_isomorphic=timeout_isomorphic)

    if timed_out:
        save_json_data(output_dir / "timed_out_pairs.json",
                       [{"a": a, "b": b} for a, b in timed_out])
        if timeout_isomorphic:
            print(f"\n[!] VF2 步数预算超时的对: {len(timed_out)}（已按同构合并，见 timed_out_pairs.json）")
        else:
            print(f"\n[!] VF2 步数预算超时的对: {len(timed_out)}（已按不同构处理，见 timed_out_pairs.json，可调大 --vf2-step-budget 重跑这些对）")

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
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL, help=f"节点/边属性的绝对容差 (默认: {DEFAULT_ATOL})")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help=f"节点/边属性的相对容差 (默认: {DEFAULT_RTOL})")
    parser.add_argument("--no-prefilter", action="store_true", help="关闭置换不变预过滤，桶内全对跑 VF2（用于 A/B 验证结果一致）")
    parser.add_argument("--vf2-step-budget", type=int, default=None,
                        help="单对 VF2 比较的最大可行性检查步数；超限则按不同构处理并记到 timed_out_pairs.json。"
                             "用于跳过指数级回溯的卡死对（如某桶卡死不前时设置，建议从 1e6 起调）")
    parser.add_argument("--timeout-isomorphic", action="store_true",
                        help="VF2 步数预算超时的对按同构处理（合并）。默认按不同构（保守不合并）。"
                             "开启会过合并（牺牲精度换速度），仅在能接受时使用")

    args = parser.parse_args()

    cluster_from_graphs_json(
        args.input,
        args.output,
        args.step_dir,
        args.num_workers,
        args.max_count,
        args.skip_copy,
        args.move_step,
        args.atol,
        args.rtol,
        use_prefilter=not args.no_prefilter,
        step_budget=args.vf2_step_budget,
        timeout_isomorphic=args.timeout_isomorphic,
    )


if __name__ == '__main__':
    main()