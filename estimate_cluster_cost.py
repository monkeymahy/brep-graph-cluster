"""
预估 relaxed 聚簇的计算量与时间。

只统计每个桶预过滤后的候选对数（不跑完整 VF2），可直接判断大桶是“多样”
（候选对远小于全量对，预过滤有效）还是“同族”（候选对≈全量对，无损剪枝到头）。
可选地抽样跑 VF2 校准每对耗时，外推总时间。校准带步数上限，不会被卡死对拖住。

示例:
  python estimate_cluster_cost.py --input ./aag --atol 1e-2 --rtol 1e-2
  python estimate_cluster_cost.py --input ./aag --atol 1e-2 --rtol 1e-2 --calibrate 50
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from graph_cluster_large_relaxed import (
    load_json,
    load_graphs_from_dir_or_json,
    _normalize_graph_item,
    compute_relaxed_bucket_key,
    compute_invariant_stats,
    build_candidate_matrix,
    aag_to_networkx_relaxed,
    _vf2_isomorphic,
)
from multiprocessing import cpu_count


def main():
    parser = argparse.ArgumentParser(description="预估 relaxed 聚簇的计算量与时间")
    parser.add_argument("--input", type=str, required=True, help="AAG 目录或单个 graphs.json")
    parser.add_argument("--max-count", type=int, default=None, help="只处理前 N 个图（测试用）")
    parser.add_argument("--atol", type=float, default=1e-2, help="绝对容差（应与实际聚簇一致）")
    parser.add_argument("--rtol", type=float, default=1e-2, help="相对容差（应与实际聚簇一致）")
    parser.add_argument("--top-buckets", type=int, default=20, help="显示候选对数前 N 的桶")
    parser.add_argument("--no-prefilter", action="store_true", help="不计预过滤，统计全量对数")
    parser.add_argument("--calibrate", type=int, default=0, help="抽样跑多少对 VF2 校准每对耗时")
    parser.add_argument("--vf2-step-budget", type=int, default=1000000, help="校准时单对 VF2 步数上限（防卡死）")
    parser.add_argument("--num-workers", type=int, default=None, help="外推时间用的并行进程数（默认 CPU-1）")
    args = parser.parse_args()

    input_path = Path(args.input)

    # ---- 1. 流式加载：每图只保留 (fn, 桶键, 不变量统计, 源) ----
    per_graph: List[Tuple] = []  # (fn, bucket_key, stats, src)  src=Path(目录模式) 或 data(json模式)
    if input_path.is_dir():
        json_files = sorted(input_path.glob("*.json"))
        if args.max_count:
            json_files = json_files[:args.max_count]
        print(f"目录模式：{len(json_files)} 个文件")
        for jf in tqdm(json_files, desc="加载并计算不变量"):
            try:
                d = load_json(jf)
                fn, gd = _normalize_graph_item(jf, d)
                key = compute_relaxed_bucket_key(gd)
                stats = compute_invariant_stats(gd)
                per_graph.append((fn, key, stats, jf))
            except Exception as e:
                print(f"读取 {jf} 失败: {e}")
    else:
        graphs = load_graphs_from_dir_or_json(args.input, max_count=args.max_count)
        for fn, gd in tqdm(graphs, desc="计算不变量"):
            try:
                key = compute_relaxed_bucket_key(gd)
                stats = compute_invariant_stats(gd)
                per_graph.append((fn, key, stats, gd))
            except Exception:
                pass

    total = len(per_graph)
    print(f"\n总图数: {total}")
    if total == 0:
        return

    # ---- 2. 分桶 ----
    buckets: Dict[Tuple, List] = defaultdict(list)
    for fn, key, stats, src in per_graph:
        buckets[key].append((fn, stats, src))

    sizes = sorted((len(v) for v in buckets.values()), reverse=True)
    total_pairs = sum(s * (s - 1) // 2 for s in sizes)
    print(f"桶数: {len(buckets)}")
    print(f"桶大小: 最大={sizes[0]}, 平均={total / len(buckets):.1f}")
    print(f"全量对数（无预过滤）: {total_pairs:,}")

    # ---- 3. 每个桶算候选对数 ----
    rows: List[Tuple] = []  # (key, size, cand_pairs)
    total_candidates = 0
    for key, items in tqdm(buckets.items(), desc="统计候选对"):
        s = len(items)
        if s <= 1:
            rows.append((key, s, 0))
            continue
        if args.no_prefilter:
            cp = s * (s - 1) // 2
        else:
            stats_list = [it[1] for it in items]
            cand = build_candidate_matrix(stats_list, args.atol, args.rtol)
            cp = int(np.count_nonzero(np.triu(cand, 1)))
        rows.append((key, s, cp))
        total_candidates += cp

    rows.sort(key=lambda r: r[2], reverse=True)

    print(f"\n预过滤后候选对总数: {total_candidates:,}")
    if total_pairs:
        prune = (1 - total_candidates / total_pairs) * 100
        print(f"剪枝率: {prune:.1f}%  （越高说明预过滤越有效；接近 0% 说明大桶是同族，无损剪枝到头）")

    print(f"\n候选对数 Top {args.top_buckets} 桶:")
    print(f"  {'num_nodes':>9} {'num_edges':>9} {'size':>7} {'cand_pairs':>13} {'cand/size^2':>13}")
    for (nn, ne), sz, cp in rows[:args.top_buckets]:
        ratio = (cp / (sz * (sz - 1) / 2)) if sz > 1 else 0.0
        print(f"  {nn:>9} {ne:>9} {sz:>7} {cp:>13,} {ratio:>12.1%}")

    # ---- 4. 可选：抽样校准每对耗时并外推 ----
    if args.calibrate > 0:
        import random
        random.seed(0)
        times: List[float] = []
        timeouts = 0
        sampled = 0
        rows_sorted = [r for r in rows if r[2] > 0]
        print(f"\n校准：抽样 {args.calibrate} 对 VF2（步数上限 {args.vf2_step_budget}）...")
        for (key, sz, cp) in rows_sorted:
            if sampled >= args.calibrate:
                break
            take = min(args.calibrate - sampled,
                       max(1, int(np.ceil(args.calibrate * cp / max(1, total_candidates)))))
            items = buckets[key]
            stats_list = [it[1] for it in items]
            cand = build_candidate_matrix(stats_list, args.atol, args.rtol)
            cand_pairs = list(zip(*np.where(np.triu(cand, 1))))
            if not cand_pairs:
                continue
            take = min(take, len(cand_pairs))
            sampled_pairs = random.sample(cand_pairs, take)
            # 每个桶只构建一次 G_list
            G_list = []
            for (fn, stats, src) in items:
                if isinstance(src, Path):
                    gd = _normalize_graph_item(src, load_json(src))[1]
                else:
                    gd = src
                G_list.append(aag_to_networkx_relaxed(gd))
            for (i, j) in sampled_pairs:
                t0 = time.perf_counter()
                res = _vf2_isomorphic(G_list[i], G_list[j], args.atol, args.rtol,
                                      step_budget=args.vf2_step_budget)
                dt = time.perf_counter() - t0
                times.append(dt)
                if res is None:
                    timeouts += 1
                sampled += 1
            print(f"  桶(n={key[0]},e={key[1]}): 采样 {take}/{cp:,} 对")

        if times:
            ts = np.array(times)
            nw = args.num_workers or max(1, cpu_count() - 1)
            med = float(np.median(ts))
            mean = float(ts.mean())
            p90 = float(np.percentile(ts, 90))
            mx = float(ts.max())
            print(f"\n校准结果（{len(ts)} 对，其中超时 {timeouts}）:")
            print(f"  每对耗时(s): min={ts.min():.4f} median={med:.4f} mean={mean:.4f} p90={p90:.4f} max={mx:.4f}")
            est_med = total_candidates * med / nw
            est_p90 = total_candidates * p90 / nw
            print(f"  外推总时间（{nw} 进程）: 中位口径≈{est_med/3600:.1f}h, p90口径≈{est_p90/3600:.1f}h")
            if timeouts > 0:
                print(f"  [!] 校准中有 {timeouts}/{len(ts)} 对超时——说明存在卡死对，实际时间可能被长尾主导；")
                print(f"    建议正式跑时加 --vf2-step-budget {args.vf2_step_budget} 保证能跑完。")
            else:
                print(f"  校准未见超时；若仍卡死，调高 --vf2-step-budget 再校准。")
            print(f"  注：VF2 耗时长尾严重，外推仅为量级参考；候选对数才是可靠指标。")


if __name__ == "__main__":
    main()
