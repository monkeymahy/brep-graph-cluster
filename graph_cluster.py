"""
AAG Graph Clustering
使用 networkx 的 VF2++ 算法对 AAG 图进行精确匹配和聚簇
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing.pool import Pool
from itertools import combinations
from tqdm import tqdm

import networkx as nx
import numpy as np


def load_json(json_path):
    with open(json_path, "r") as fp:
        return json.load(fp)


def save_json_data(pathname, data):
    with open(pathname, 'w', encoding='utf8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False, sort_keys=False)


def aag_to_networkx(aag_data: Dict) -> nx.Graph:
    """
    将 AAG 数据转换为 networkx 图

    Args:
        aag_data: AAG 数据字典

    Returns:
        networkx Graph 对象，节点和边都带有属性
    """
    G = nx.Graph()

    # 添加节点
    num_nodes = aag_data['graph']['num_nodes']
    face_attrs = aag_data['graph_face_attr']

    for i in range(num_nodes):
        # 将面属性转换为元组以便比较
        node_attr = tuple(face_attrs[i]) if i < len(face_attrs) else ()
        G.add_node(i, attr=node_attr)

    # 添加边
    src, dst = aag_data['graph']['edges']
    edge_attrs = aag_data['graph_edge_attr']

    for idx, (u, v) in enumerate(zip(src, dst)):
        # 将边属性转换为元组以便比较
        edge_attr = tuple(edge_attrs[idx]) if idx < len(edge_attrs) else ()
        G.add_edge(u, v, attr=edge_attr)

    return G


def node_match(node1: Dict, node2: Dict) -> bool:
    """
    节点匹配函数：比较节点属性是否完全相同

    Args:
        node1, node2: 两个节点的属性字典

    Returns:
        是否匹配
    """
    # 检查节点属性是否存在
    if 'attr' not in node1 or 'attr' not in node2:
        return False
    # 比较属性元组
    return np.allclose(np.array(node1['attr']), np.array(node2['attr']), atol=1e-6)


def edge_match(edge1: Dict, edge2: Dict) -> bool:
    """
    边匹配函数：比较边属性是否完全相同

    Args:
        edge1, edge2: 两条边的属性字典

    Returns:
        是否匹配
    """
    # 检查边属性是否存在
    if 'attr' not in edge1 or 'attr' not in edge2:
        return False
    # 比较属性元组
    return np.allclose(np.array(edge1['attr']), np.array(edge2['attr']), atol=1e-6)


def are_graphs_isomorphic(G1: nx.Graph, G2: nx.Graph) -> bool:
    """
    使用 VF2++ 算法检查两个图是否同构

    Args:
        G1, G2: 两个 networkx 图

    Returns:
        是否同构
    """
    # 快速检查：节点数或边数不同直接返回 False
    if G1.number_of_nodes() != G2.number_of_nodes():
        return False
    if G1.number_of_edges() != G2.number_of_edges():
        return False

    # 使用 VF2++ 算法
    matcher = nx.isomorphism.GraphMatcher(G1, G2, node_match=node_match, edge_match=edge_match)
    return matcher.is_isomorphic()


def cluster_graphs(graphs_data: List[Tuple[str, Dict]]) -> List[List[str]]:
    """
    对图进行聚簇

    Args:
        graphs_data: 列表，每个元素为 (filename, aag_data)

    Returns:
        聚簇结果，每个簇是一个文件名列表
    """
    # 预转换所有图为 networkx 格式
    print("正在转换图格式...")
    nx_graphs = {}
    for fn, data in tqdm(graphs_data):
        try:
            nx_graphs[fn] = aag_to_networkx(data)
        except Exception as e:
            print(f"转换图 {fn} 失败: {e}")

    # 初始化：每个图自己为一个簇
    clusters = [[fn] for fn in nx_graphs.keys()]
    visited = set()

    print("正在进行图匹配和聚簇...")
    # 遍历所有图对进行比较
    filenames = list(nx_graphs.keys())
    for i, fn1 in tqdm(enumerate(filenames), total=len(filenames)):
        if fn1 in visited:
            continue

        # 找到包含 fn1 的簇
        cluster_idx = None
        for idx, cluster in enumerate(clusters):
            if fn1 in cluster:
                cluster_idx = idx
                break

        if cluster_idx is None:
            continue

        # 与后面的图比较
        for j in range(i + 1, len(filenames)):
            fn2 = filenames[j]
            if fn2 in visited:
                continue

            # 检查是否同构
            try:
                if are_graphs_isomorphic(nx_graphs[fn1], nx_graphs[fn2]):
                    # 将 fn2 加入到当前簇
                    clusters[cluster_idx].append(fn2)
                    visited.add(fn2)
            except Exception as e:
                print(f"比较 {fn1} 和 {fn2} 时出错: {e}")

        visited.add(fn1)

    # 移除空簇
    clusters = [c for c in clusters if len(c) > 0]

    # 按簇大小降序排序
    clusters.sort(key=lambda x: len(x), reverse=True)

    return clusters


def process_one_comparison(args):
    """多进程比较函数"""
    (fn1, data1), (fn2, data2) = args
    try:
        G1 = aag_to_networkx(data1)
        G2 = aag_to_networkx(data2)
        iso = are_graphs_isomorphic(G1, G2)
        return (fn1, fn2, iso)
    except Exception as e:
        print(f"比较 {fn1} 和 {fn2} 时出错: {e}")
        return (fn1, fn2, False)


def cluster_graphs_parallel(graphs_data: List[Tuple[str, Dict]], num_workers: int = 1) -> List[List[str]]:
    """
    并行版本的图聚簇

    Args:
        graphs_data: 列表，每个元素为 (filename, aag_data)
        num_workers: 并行进程数

    Returns:
        聚簇结果，每个簇是一个文件名列表
    """
    filenames = [fn for fn, _ in graphs_data]
    graph_dict = dict(graphs_data)

    if num_workers <= 1 or len(filenames) <= 1:
        return cluster_graphs(graphs_data)

    # 使用 Union-Find 数据结构
    parent = {fn: fn for fn in filenames}

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

    # 生成所有需要比较的图对
    pairs = list(combinations(graphs_data, 2))
    print(f"共 {len(pairs)} 对图需要比较，使用 {num_workers} 个进程...")

    pool = Pool(processes=num_workers)
    try:
        results = list(tqdm(
            pool.imap(process_one_comparison, pairs),
            total=len(pairs)))
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        return []

    # 根据结果合并簇
    for fn1, fn2, iso in results:
        if iso:
            union(fn1, fn2)

    # 收集簇
    clusters_dict = {}
    for fn in filenames:
        root = find(fn)
        if root not in clusters_dict:
            clusters_dict[root] = []
        clusters_dict[root].append(fn)

    clusters = list(clusters_dict.values())
    clusters.sort(key=lambda x: len(x), reverse=True)

    return clusters


def save_clustering_result(
    clusters: List[List[str]],
    output_dir: Path,
    step_source_dir: Optional[Path] = None,
    copy_files: bool = True
):
    """
    保存聚簇结果

    Args:
        clusters: 聚簇结果
        output_dir: 输出目录
        step_source_dir: STEP 文件源目录（用于复制原始文件）
        copy_files: 是否复制文件到簇文件夹
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

    # 创建簇文件夹
    if copy_files and step_source_dir and step_source_dir.exists():
        # 创建各个簇文件夹
        for idx, cluster in enumerate(clusters):
            cluster_dir = output_dir / f"cluster_{idx:04d}"
            cluster_dir.mkdir(exist_ok=True)

            # 复制该簇的 STEP 文件
            for fn in cluster:
                # 查找对应的 STEP 文件
                step_files = list(step_source_dir.glob(f"{fn}.st*p"))
                if step_files:
                    src_file = step_files[0]
                    shutil.copy2(src_file, cluster_dir / src_file.name)

        # 创建 result 文件夹保存代表文件
        result_dir = output_dir / "result"
        result_dir.mkdir(exist_ok=True)

        for idx, cluster in enumerate(clusters):
            rep_fn = cluster[0]
            step_files = list(step_source_dir.glob(f"{rep_fn}.st*p"))
            if step_files:
                src_file = step_files[0]
                # 复制并重命名，加上簇编号
                dst_name = f"cluster_{idx:04d}_{src_file.name}"
                shutil.copy2(src_file, result_dir / dst_name)

    print(f"聚簇结果已保存到 {output_dir}")
    print(f"共 {len(clusters)} 个簇")
    print(f"簇大小分布: {[len(c) for c in clusters]}")


def load_from_dir_or_json(input_path: str):
    """从目录或JSON文件加载数据"""
    input_path = Path(input_path)

    if input_path.is_dir():
        print(f"从目录加载图数据: {input_path}")
        data = []
        json_files = sorted(input_path.glob("*.json"))
        print(f"找到 {len(json_files)} 个JSON文件")

        for json_file in tqdm(json_files):
            try:
                item = load_json(json_file)
                data.append(item)
            except Exception as e:
                print(f"读取 {json_file} 失败: {e}")

        print(f"已加载 {len(data)} 个图")
        return data
    else:
        print(f"加载图数据: {input_path}")
        return load_json(input_path)


def cluster_from_graphs_json(
    graphs_json_path: str,
    output_dir: str,
    step_source_dir: Optional[str] = None,
    num_workers: int = 1
):
    """
    从 graphs.json 文件或目录加载数据并进行聚簇

    Args:
        graphs_json_path: graphs.json 文件路径或目录
        output_dir: 输出目录
        step_source_dir: STEP 文件源目录
        num_workers: 并行进程数
    """
    graphs_data = load_from_dir_or_json(graphs_json_path)

    print(f"开始聚簇 {len(graphs_data)} 个图...")
    if num_workers > 1:
        clusters = cluster_graphs_parallel(graphs_data, num_workers)
    else:
        clusters = cluster_graphs(graphs_data)

    # 保存结果
    step_dir = Path(step_source_dir) if step_source_dir else None
    save_clustering_result(clusters, Path(output_dir), step_dir)

    return clusters


def main():
    parser = argparse.ArgumentParser(description='AAG Graph Clustering using VF2++')

    parser.add_argument("--input", type=str, required=True, help="AAG目录或单个graphs.json文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--step-dir", type=str, default=None, help="STEP 文件源目录（用于复制文件）")
    parser.add_argument("--num-workers", type=int, default=1, help="并行进程数")

    args = parser.parse_args()

    cluster_from_graphs_json(
        args.input,
        args.output,
        args.step_dir,
        args.num_workers
    )


if __name__ == '__main__':
    main()
