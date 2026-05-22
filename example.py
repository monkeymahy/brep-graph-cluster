"""
使用示例 - 演示如何使用AAG提取器
"""
from pathlib import Path
import sys


def test_single_file():
    """测试单个STEP文件的AAG提取"""
    print("=" * 60)
    print("测试单个STEP文件提取")
    print("=" * 60)

    from aag_extractor import AAGExtractor

    # 检查是否有示例文件
    example_steps = list(Path(".").glob("*.step")) + list(Path(".").glob("*.stp"))

    if example_steps:
        step_file = example_steps[0]
        print(f"处理文件: {step_file}")

        try:
            extractor = AAGExtractor(step_file)
            result = extractor.process()

            print(f"\n提取成功!")
            print(f"节点数(面数): {result['graph']['num_nodes']}")
            print(f"边数: {len(result['graph']['edges'][0])}")
            print(f"面属性维度: {len(result['graph_face_attr'][0]) if result['graph_face_attr'] else 0}")
            print(f"边属性维度: {len(result['graph_edge_attr'][0]) if result['graph_edge_attr'] else 0}")

            return result
        except Exception as e:
            print(f"提取失败: {e}")
            return None
    else:
        print("当前目录下没有找到STEP文件。")
        print("请将STEP文件放在当前目录，或修改代码指定文件路径。")
        return None


def test_batch_extraction():
    """测试批量提取（需要有steps目录）"""
    print("\n" + "=" * 60)
    print("测试批量提取")
    print("=" * 60)

    steps_dir = Path("./steps")
    output_dir = Path("./output")

    if steps_dir.exists():
        from aag_extractor import extract_aag_from_step

        print(f"从 {steps_dir} 批量提取...")
        extract_aag_from_step(
            step_path=str(steps_dir),
            output_path=str(output_dir),
            num_workers=1
        )

        # 测试加载
        if (output_dir / "graphs.json").exists():
            print("\n测试加载提取的数据...")
            from graph_loader import AAGDataset
            dataset = AAGDataset(output_dir / "graphs.json", output_dir / "attr_stat.json")
            print(f"加载了 {len(dataset)} 个图")

            if len(dataset) > 0:
                sample = dataset[0]
                print(f"第一个图: {sample['graph']}")
                print(f"节点特征形状: {sample['graph'].ndata['x'].shape}")
                print(f"边特征形状: {sample['graph'].edata['x'].shape}")
    else:
        print(f"目录 {steps_dir} 不存在，跳过批量测试。")
        print("如需测试批量提取，请创建steps目录并放入STEP文件。")


def test_graph_loading():
    """测试图加载功能"""
    print("\n" + "=" * 60)
    print("测试图加载功能")
    print("=" * 60)

    output_dir = Path("./output")
    if (output_dir / "graphs.json").exists():
        from graph_loader import AAGDataset, load_statistics

        print("加载数据集...")
        dataset = AAGDataset(output_dir / "graphs.json", output_dir / "attr_stat.json")

        print(f"\n数据集大小: {len(dataset)}")

        if len(dataset) > 0:
            # 获取第一个样本
            sample = dataset[0]
            graph = sample['graph']
            print(f"\n样本 '{sample['filename']}':")
            print(f"  - 图: {graph}")
            print(f"  - 节点数: {graph.num_nodes()}")
            print(f"  - 边数: {graph.num_edges()}")
            print(f"  - 节点特征: {graph.ndata['x'].shape}")
            print(f"  - 边特征: {graph.edata['x'].shape}")

            # 演示批处理
            if len(dataset) >= 2:
                from graph_loader import collate_fn
                import torch.utils.data as data

                loader = data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
                batch = next(iter(loader))
                print(f"\n批处理示例:")
                print(f"  - 批量图: {batch['graph']}")
                print(f"  - 文件名: {batch['filename']}")


if __name__ == "__main__":
    print("AAG Extractor 使用示例\n")

    # 检查是否可以导入模块
    try:
        import aag_extractor
        import graph_loader
    except ImportError as e:
        print(f"导入模块失败: {e}")
        print("请确保所有依赖已安装！")
        sys.exit(1)

    # 运行测试
    result = test_single_file()

    # 如果有output目录，测试加载功能
    if Path("./output").exists():
        test_graph_loading()

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
