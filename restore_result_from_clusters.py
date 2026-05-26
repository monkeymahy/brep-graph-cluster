"""
Restore result/ representative STEP files from cluster_* folders.
"""
import argparse
import shutil
from pathlib import Path


STEP_EXTS = [".step", ".stp", ".STEP", ".STP"]


def find_step_in_cluster(cluster_dir: Path) -> Path | None:
    for ext in STEP_EXTS:
        candidates = list(cluster_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def restore_results(output_dir: Path, overwrite: bool = False) -> None:
    result_dir = output_dir / "result"
    result_dir.mkdir(exist_ok=True)

    cluster_dirs = sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("cluster_")])
    restored = 0
    skipped = 0
    missing = 0

    for cluster_dir in cluster_dirs:
        rep_file = find_step_in_cluster(cluster_dir)
        if rep_file is None:
            missing += 1
            continue

        dst_path = result_dir / rep_file.name

        if dst_path.exists() and not overwrite:
            skipped += 1
            continue

        shutil.copy2(rep_file, dst_path)
        restored += 1

    print(f"Restored: {restored}, Skipped: {skipped}, Missing: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore result folder from cluster_* directories")
    parser.add_argument("--output", type=str, required=True, help="聚类输出目录 (包含 cluster_* 目录)")
    parser.add_argument("--overwrite", action="store_true", help="覆盖 result 下已存在的文件")
    args = parser.parse_args()

    restore_results(Path(args.output), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
