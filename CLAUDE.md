# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Clusters B-rep CAD parts (STEP files) by geometric equivalence. Each STEP has already been converted to an Attributed Adjacency Graph (AAG) JSON (extraction happens elsewhere); this repo groups AAGs that are isomorphic. The same part at a different rigid-body rotation lands in the same cluster because AAG features are rotation-invariant by construction.

## Dependencies

Single lightweight tier: `numpy`, `networkx`, `tqdm`. No heavy env, no torch/dgl/pythonocc.

## Common commands

```bash
# Cluster (the only clusterer). --input accepts a directory of per-file *.json
# OR a single graphs.json. Directory input triggers the low-memory path.
python graph_cluster.py --input ./aag --output ./cluster_result \
    [--step-dir ./steps] [--num-workers 16] [--skip-copy] [--move-step] [--max-count N] \
    [--atol 1e-2] [--rtol 1e-2] [--vf2-step-budget 1000000] [--timeout-isomorphic]

# Estimate candidate-pair count and (optionally) extrapolate runtime before a full run
python estimate_cluster_cost.py --input ./aag --atol 1e-2 --rtol 1e-2 [--calibrate 50]
```

`--vf2-step-budget` caps per-pair VF2 feasibility checks; pairs exceeding it are treated as non-isomorphic and logged to `timed_out_pairs.json`. `--timeout-isomorphic` instead merges them (sacrifice precision for recall). `--no-prefilter` disables the lossless invariant prefilter for A/B verification.

**STEP files on a NAS**: STEP files often live on a NAS this machine can't write to directly. `move_step_by_cluster.py` does NOT touch files — it reads `clusters.json` and emits a pure POSIX shell script to run on the NAS:

```bash
python move_step_by_cluster.py --clusters ./cluster_result/clusters.json \
    --step-dir /volume1/steps --output /volume1/cluster_result [--copy-only]
# copy the generated move_step_by_cluster.sh to the NAS, then:
sh move_step_by_cluster.sh               # apply
DRY_RUN=1 sh move_step_by_cluster.sh     # preview only
```

There are no test suites, linters, or build steps.

## Architecture

### Clustering (`graph_cluster.py`)
Self-contained — the I/O and Union-Find helpers (`load_json`, `save_json_data`, `merge_mappings`, `mapping_to_clusters`, `save_clustering_result`, `load_graphs_from_dir_or_json`) are all inlined here (no sibling module). Argparse CLI → `cluster_from_graphs_json` → `save_clustering_result`.

Core algorithm:
1. **Bucket** by `(num_nodes, num_edges)` (`compute_bucket_key`) so only plausibly-isomorphic graphs are compared. Loose key on purpose — avoids splitting same-part AAGs whose node/edge counts drift.
2. **Prefilter** within each bucket with permutation-invariant, provably-lossless features: sorted degree sequence (exact) + per-dim mean/min/max/max-abs of node and edge attrs (`compute_invariant_stats` / `build_candidate_matrix`). Non-candidates are skipped without a VF2 call and never false-prune a true isomorphism. The rtol reference MUST be max-abs (attrs can be negative — CAD curvature/normals); `PREFILTER_SAFETY=2.0`.
3. **VF2** on candidate pairs (`nx.isomorphism.GraphMatcher` + `node_match`/`edge_match` using `np.allclose` at `atol`/`rtol`). Union-Find merges matches (`process_single_bucket_uf` / `process_single_bucket_uf_files`). `_vf2_isomorphic` returns three states: `True` / `False` / `None` (None = step-budget exceeded); `--timeout-isomorphic` makes None a merge.
4. **Merge** per-bucket mappings into a global Union-Find (`merge_mappings`), then emit clusters sorted by size descending (`mapping_to_clusters`). Each cluster's representative is `files[0]`.

Two execution paths, chosen by `--input` type:
- **Memory mode** (single `graphs.json`): loads everything once (`load_graphs_from_dir_or_json`).
- **Dir mode** (directory of per-file JSONs, recommended for large data): bucketing keeps only file paths; each bucket re-reads its JSONs inside the worker. Bounds resident memory.

Multiprocessing uses `pool.imap_unordered` so the progress bar advances per completed bucket (Union-Find merge is order-independent). `merge_mappings` iterates a `set()`, so representative/cluster-id ordering is not deterministic across runs (the grouping itself is deterministic).

Defaults: `DEFAULT_ATOL=1e-3`, `DEFAULT_RTOL=1e-4`. The user typically runs at `1e-2`/`1e-2` for recall.

### Cost estimator (`estimate_cluster_cost.py`)
Imports `compute_bucket_key`, `compute_invariant_stats`, `build_candidate_matrix`, `_vf2_isomorphic`, etc. from `graph_cluster`. Counts candidate pairs per bucket, reports prune rate and top buckets; with `--calibrate N` samples N candidate pairs (stratified by bucket candidate count) timing VF2 under a step budget and extrapolates total time. Candidate-pair count is the reliable metric; VF2 timing has a heavy tail, so the extrapolation is only order-of-magnitude.

### NAS move script (`move_step_by_cluster.py`)
Reads `clusters.json`, emits a pure POSIX shell script. `place_cluster_file` moves/copies each STEP into `cluster_NNNN/` with its original name; `copy_representative` copies each cluster's representative to `result/` with its original name (no cluster prefix). `find_step_file` tries `.step/.stp/.STEP/.STP`. Missing files recorded to `missing_steps.txt`.

### Outputs (`save_clustering_result`)
Writes to `--output`: `clusters.json` (id/size/representative/files), `file_mapping.json` (per-file cluster), `stats.json`, `cluster_NNNN/` dirs (copied or moved STEP files when `--step-dir` given), `result/` (representatives, cluster-prefixed in the local writer), `timed_out_pairs.json` (when `--vf2-step-budget` set), and `missing_steps.json` when STEP files aren't found. `--skip-copy` writes only the JSONs; `--move-step` moves instead of copies.

### Input format
A per-file AAG JSON (dir mode) is the data dict alone; a `graphs.json` is a list of `[filename, data]` pairs. `graph_face_attr`/`graph_edge_attr` are lists of attribute vectors. `_normalize_graph_item` accepts either form. AAG features are rotation-invariant (expressed relative to centroids, not absolute coordinates), which is why no rotation flag is needed at cluster time.
