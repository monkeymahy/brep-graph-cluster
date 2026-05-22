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


def cluster_from_graphs_json(
    graphs_json_path: str,
    output_dir: str,
    step_source_dir: Optional[str] = None,
    num_workers: int = 1
):
    """
    从 graphs.json 文件加载数据并进行聚簇

    Args:
        graphs_json_path: graphs.json 文件路径
        output_dir: 输出目录
        step_source_dir: STEP 文件源目录
        num_workers: 并行进程数
    """
    print("加载图数据...")
    graphs_data = load_json(graphs_json_path)

    print(f"开始聚簇 {len(graphs_data)} 个图...")
    if num_workers > 1:
        clusters = cluster_graphs_parallel(graphs_data, num_workers)
    else:
        clusters = cluster_graphs(graphs_data)

    # 保存结果
    step_dir = Path(step_source_dir) if step_source_dir else None
    save_clustering_result(clusters, Path(output_dir), step_dir)

    return clusters


def cluster_step_folder(
    step_dir: str,
    output_dir: str,
    num_workers: int = 1
):
    """
    直接从 STEP 文件夹进行聚簇（先提取 AAG）

    Args:
        step_dir: STEP 文件目录
        output_dir: 输出目录
        num_workers: 并行进程数
    """
    from aag_extractor import extract_aag_from_step

    step_dir = Path(step_dir)
    output_dir = Path(output_dir)

    # 临时目录用于存储提取的 AAG
    temp_aag_dir = output_dir / "_temp_aag"
    temp_aag_dir.mkdir(parents=True, exist_ok=True)

    # 1. 提取 AAG
    print("步骤 1: 提取 AAG...")
    extract_aag_from_step(
        step_path=str(step_dir),
        output_path=str(temp_aag_dir),
        num_workers=num_workers
    )

    # 2. 聚簇
    print("\n步骤 2: 图聚簇...")
    graphs_json = temp_aag_dir / "graphs.json"
    if not graphs_json.exists():
        print("未找到 graphs.json，提取可能失败。")
        return

    cluster_output = output_dir
    clusters = cluster_from_graphs_json(
        str(graphs_json),
        str(cluster_output),
        step_source_dir=str(step_dir),
        num_workers=num_workers
    )

    # 清理临时文件
    try:
        shutil.rmtree(temp_aag_dir)
    except:
        pass

    return clusters


def main():
    parser = argparse.ArgumentParser(description='AAG Graph Clustering using VF2++')

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 从 graphs.json 聚簇
    from_json_parser = subparsers.add_parser('from-json', help='从 graphs.json 聚簇')
    from_json_parser.add_argument("--graphs-json", type=str, required=True, help="graphs.json 文件路径")
    from_json_parser.add_argument("--output", type=str, required=True, help="输出目录")
    from_json_parser.add_argument("--step-dir", type=str, default=None, help="STEP 文件源目录（用于复制文件）")
    from_json_parser.add_argument("--num-workers", type=int, default=1, help="并行进程数")

    # 从 STEP 文件夹直接聚簇
    from_step_parser = subparsers.add_parser('from-step', help='从 STEP 文件夹直接聚簇')
    from_step_parser.add_argument("--step-dir", type=str, required=True, help="STEP 文件目录")
    from_step_parser.add_argument("--output", type=str, required=True, help="输出目录")
    from_step_parser.add_argument("--num-workers", type=int, default=1, help="并行进程数")

    args = parser.parse_args()

    if args.command == 'from-json':
        cluster_from_graphs_json(
            args.graphs_json,
            args.output,
            args.step_dir,
            args.num_workers
        )
    elif args.command == 'from-step':
        cluster_step_folder(
            args.step_dir,
            args.output,
            args.num_workers
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
