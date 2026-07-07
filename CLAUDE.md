# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Clusters B-rep CAD parts (STEP files) by geometric equivalence. Pipeline: extract an Attributed Adjacency Graph (AAG) from each STEP file, then group files whose AAGs are isomorphic. The same part at a different rigid-body rotation lands in the same cluster because AAG features are rotation-invariant by construction.

## Two dependency tiers

The repo runs in two distinct environments — do not assume one covers everything:

- **Clustering tier** (lightweight): `numpy`, `networkx`, `tqdm`. All `graph_cluster*.py` scripts run here. This is the common path.
- **Extraction tier** (heavy): `pythonocc-core`/`occt`, `occwl` (AutodeskAILab), plus `torch`/`dgl` only for `graph_loader.py`. Needed for `aag_extractor.py`, `graph_loader.py`, and `example.py`.

See `INSTALL.md` for heavy-env setup, including two gotchas: **do not install `numba`** (conflicts with `tbb` pinned by `pythonocc-core=7.5.1`), and a known `vtk` metadata-corruption fix.

## Common commands

```bash
# 1. Extract AAGs from STEP files (heavy env) → graphs.json + attr_stat.json
python aag_extractor.py --step_path ./steps --output ./aag_output --num_workers 16
#   --schema ./my_schema.json overrides the default attribute set

# 2. Cluster (lightweight env). --input accepts a directory of per-file *.json
#    OR a single graphs.json. Directory input triggers the low-memory path.
python graph_cluster_large.py --input ./aag --output ./cluster_result \
    [--step-dir ./steps] [--num-workers 16] [--skip-copy] [--move-step] [--max-count N]

# 3. High-recall variant when the large version leaves near-duplicates unmerged.
#    Looser bucketing (num_nodes+num_edges only) + tunable tolerances.
python graph_cluster_large_relaxed.py --input ./aag --output ./cluster_result \
    --atol 1e-3 --rtol 1e-4

# Smoke-test extraction + loader round-trip (heavy env)
python example.py
```

**STEP files on a NAS**: STEP files often live on a NAS this machine can't write to directly. `move_step_by_cluster.py` does NOT touch files — it reads `clusters.json` and emits a pure POSIX shell script to run on the NAS:

```bash
python move_step_by_cluster.py --clusters ./cluster_result/clusters.json \
    --step-dir /volume1/steps --output /volume1/cluster_result [--copy-only]
# copy the generated move_step_by_cluster.sh to the NAS, then:
sh move_step_by_cluster.sh               # apply
DRY_RUN=1 sh move_step_by_cluster.sh     # preview only

# Rebuild result/ representatives from existing cluster_* dirs (local)
python restore_result_from_clusters.py --output ./cluster_result [--overwrite]
```

There are no test suites, linters, or build steps.

## Architecture

### AAG extraction (`aag_extractor.py`)
`AAGExtractor.process()` reads a STEP body via OpenCASCADE, validates it is a closed manifold `TopoDS_Solid` (`TopologyChecker`), scales it to a unit box, computes the body centroid, then builds a face-adjacency graph via `occwl.graph.face_adjacency`. Faces → nodes, face-face edges → graph edges. Per-face/per-edge attribute vectors come from `schema.json` (geometry-type flags, area/length, convexity, BSpline rationality, etc.).

**Rotation invariance is the core design choice** — features are expressed relative to centroids, not absolute coordinates:
- `FaceCentroidRadiusAttribute` = distance from face centroid to body centroid.
- UV face grid stores `(point-to-face-centroid distance, normal·(body_centroid→face_centroid unit), inside-mask)` — 3 channels, shape `(3, U, V)`.
- Edge grid stores `(point-to-edge-centroid distance, tangent/left-normal/right-normal projected onto point-to-centroid unit)` — 4 channels.

This is why no rotation flag is needed at cluster time. `find_standardization` emits per-attr-dim mean/std (`attr_stat.json`) for optional normalization via `graph_loader.standardize_graph`.

### Clustering (`graph_cluster*.py`)
Four versions share the same shape (argparse CLI, `cluster_from_graphs_json` entry, `save_clustering_result` writer) but differ in scale strategy:

| File | Scale | Strategy |
|------|-------|----------|
| `graph_cluster.py` | <1k | O(n²) pairwise VF2++ |
| `graph_cluster_fast.py` | 1k–10k | md5-hash bucketed VF2++ |
| `graph_cluster_large.py` | 10k+ (default) | multi-level bucket + Union-Find + VF2++, low-memory dir mode |
| `graph_cluster_large_relaxed.py` | high recall | imports large's helpers, overrides bucketing + tolerances |

Large version's core algorithm (`graph_cluster_large.py`):
1. **Bucket** by a multi-level key `(num_nodes, num_edges, face_attr_mean, edge_attr_mean, face_sketch_md5, edge_sketch_md5)` so only plausibly-isomorphic graphs are compared. A sketch = sorted md5 fingerprints of rounded attribute vectors (order-invariant).
2. **Compare within each bucket** with VF2++ (`nx.isomorphism.GraphMatcher` + `node_match`/`edge_match` using `np.allclose`). Union-Find merges matches (`process_single_bucket_uf*`).
3. **Merge** per-bucket mappings into a global Union-Find (`merge_mappings`), then emit clusters sorted by size descending (`mapping_to_clusters`). Each cluster's representative is `files[0]`.

Two execution paths, chosen by `--input` type:
- **Memory mode** (single `graphs.json`): loads everything once (`load_graphs_from_dir_or_json`).
- **Dir mode** (directory of per-file JSONs, recommended for large data): bucketing keeps only file paths; each bucket re-reads its JSONs inside the worker (`process_single_bucket_uf_files`). Bounds resident memory.

The **relaxed** version reuses `merge_mappings`, `mapping_to_clusters`, `save_clustering_result`, `load_graphs_from_dir_or_json` from `graph_cluster_large` and overrides only: bucket key (just `num_nodes, num_edges`), and match tolerances (`DEFAULT_ATOL=1e-3`, `DEFAULT_RTOL=1e-4`, tunable via `--atol`/`--rtol`). Bigger buckets → slower but fewer false splits from float drift. Strict versions use `atol` 1e-6 (basic) / 1e-5 (large).

### Outputs (`save_clustering_result`)
Writes to `--output`: `clusters.json` (id/size/representative/files), `file_mapping.json` (per-file cluster), `stats.json`, `cluster_NNNN/` dirs (copied or moved STEP files when `--step-dir` given), `result/` (cluster-prefixed representatives), and `missing_steps.json` when STEP files aren't found. `--skip-copy` writes only the JSONs (inspect results before touching files); `--move-step` moves instead of copies.

### Input format
A per-file AAG JSON (dir mode) is the data dict alone; a `graphs.json` is a list of `[filename, data]` pairs. `graph_face_attr`/`graph_edge_attr` are lists of attribute vectors; `graph_face_grid`/`graph_edge_grid` hold the rotation-invariant grids above. `_normalize_graph_item` (in both large and relaxed) accepts either form, so the same clusterer handles extractor output and hand-authored files.
