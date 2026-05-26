"""
Move or copy STEP files into cluster folders based on clusters.json.
"""
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict


STEP_EXTS = [".step", ".stp", ".STEP", ".STP"]


def load_json(path: Path):
    with open(path, "r", encoding="utf8") as fp:
        return json.load(fp)


def find_step_file(step_dir: Path, stem: str) -> Path | None:
    for ext in STEP_EXTS:
        candidate = step_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def move_steps_by_clusters(clusters: List[Dict], step_dir: Path, output_dir: Path, move_files: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir = output_dir / "result"
    result_dir.mkdir(exist_ok=True)
    missing_steps = []

    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        cluster_files = cluster["files"]
        rep_fn = cluster["representative"]

        cluster_dir = output_dir / f"cluster_{cluster_id:04d}"
        cluster_dir.mkdir(exist_ok=True)

        for fn in cluster_files:
            src_file = find_step_file(step_dir, fn)
            if src_file is None:
                missing_steps.append(fn)
                continue
            dst_file = cluster_dir / src_file.name
            if move_files:
                shutil.move(str(src_file), str(dst_file))
            else:
                shutil.copy2(src_file, dst_file)

        rep_path = find_step_file(cluster_dir, rep_fn)
        if rep_path is None:
            rep_path = find_step_file(step_dir, rep_fn)
        if rep_path is not None:
            dst_name = f"cluster_{cluster_id:04d}_{rep_path.name}"
            dst_path = result_dir / dst_name
            if dst_path.exists():
                continue
            shutil.copy2(rep_path, dst_path)
        else:
            missing_steps.append(rep_fn)

    if missing_steps:
        missing_path = output_dir / "missing_steps.json"
        with open(missing_path, "w", encoding="utf8") as fp:
            json.dump(sorted(set(missing_steps)), fp, indent=4, ensure_ascii=False)
        print(f"Missing STEP files: {len(set(missing_steps))} (see {missing_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move/copy STEP files according to clusters.json")
    parser.add_argument("--clusters", type=str, required=True, help="clusters.json 路径")
    parser.add_argument("--step-dir", type=str, required=True, help="STEP 文件目录")
    parser.add_argument("--output", type=str, required=True, help="输出目录")
    parser.add_argument("--copy-only", action="store_true", help="只复制，不移动")
    args = parser.parse_args()

    clusters_path = Path(args.clusters)
    step_dir = Path(args.step_dir)
    output_dir = Path(args.output)

    clusters = load_json(clusters_path)
    move_steps_by_clusters(clusters, step_dir, output_dir, move_files=not args.copy_only)


if __name__ == "__main__":
    main()
