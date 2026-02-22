#!/usr/bin/env python3
"""
cogaps_prep_cache.py

One-time preprocessing + cache creation for CoGAPS sweeps on HPC.

This script runs the frozen preprocessing pipeline once and writes two cache files:

  1) preprocessed_cells.h5ad
     - cells × genes
     - normalized + log1p
     - HVG subset applied
     - used for plots (UMAP, boxplots, etc.)

  2) cogaps_input_genesxcells.h5ad
     - genes × cells
     - dense float64 matrix in .X (what CoGAPS expects)
     - cell metadata stored in .var (includes condition, cell_type)
     - used as *input* for all CoGAPS runs

Recommended workflow on Slurm
-----------------------------
(1) sbatch prep job (this script) to build caches
(2) sbatch job-array where each task runs ONE (K, seed, n_iter) using the cached CoGAPS input
(3) sbatch aggregate job after the array finishes

No downloads
------------
This script does NOT download data. Provide a real path on the cluster.

Example:
  python cogaps_prep_cache.py \
    --input-h5ad /path/to/kang_counts_25k.h5ad \
    --outdir results_cogaps_singleprocess_hpc \
    --n-top-genes 3000 \
    --blas-threads 1
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import scanpy as sc


def set_thread_env(n_threads: int) -> None:
    n = str(int(n_threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n  # macOS Accelerate


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cogaps_prep_cache")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"[LOG] {log_path}")
    return logger


def atomic_write_h5ad(adata: sc.AnnData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    adata.write_h5ad(tmp_path)
    if tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Atomic write failed (0 bytes): {tmp_path}")
    tmp_path.replace(out_path)


def require_columns(adata: sc.AnnData) -> None:
    if "condition" not in adata.obs and "label" in adata.obs:
        adata.obs["condition"] = adata.obs["label"]
    required = ["condition", "cell_type"]
    missing = [c for c in required if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")


def preprocess_freeze(
    adata_raw: sc.AnnData,
    min_cells: int,
    target_sum: float,
    n_top_genes: int,
    hvg_flavor: str,
    logger: logging.Logger,
) -> sc.AnnData:
    """Frozen preprocessing pipeline used for every run."""
    adata = adata_raw.copy()
    logger.info(f"[PRE] Start: cells={adata.n_obs:,}, genes={adata.n_vars:,}")

    # Store raw counts BEFORE any normalization/log
    adata.layers["counts"] = adata.X.copy()

    before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    after = adata.n_vars
    logger.info(f"[PRE] filter_genes(min_cells={min_cells}): {before:,} -> {after:,}")

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=hvg_flavor,
        layer="counts",
    )
    hvgs = int(adata.var["highly_variable"].sum())
    logger.info(f"[PRE] HVGs selected: {hvgs:,} / {adata.n_vars:,}")

    adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info(f"[PRE] After HVG subset: cells={adata.n_obs:,}, genes={adata.n_vars:,}")
    return adata


def to_cogaps_input(adata_cells: sc.AnnData, logger: logging.Logger) -> sc.AnnData:
    """Convert cells×genes -> genes×cells and ensure dense float64."""
    adata_cg = adata_cells.T.copy()  # genes × cells
    X = adata_cg.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    adata_cg.X = np.asarray(X, dtype=np.float64)
    logger.info(f"[COGAPS] Input matrix: {adata_cg.shape} (genes×cells), dtype={adata_cg.X.dtype}")
    return adata_cg


def main() -> None:
    ap = argparse.ArgumentParser(description="Create cached preprocessed AnnData + CoGAPS input for HPC sweeps.")
    ap.add_argument("--input-h5ad", required=True, help="Path to kang_counts_25k.h5ad (no download).")
    ap.add_argument("--outdir", default="results_cogaps_singleprocess_hpc", help="Output directory.")
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument("--n-top-genes", type=int, default=3000)
    ap.add_argument("--hvg-flavor", default="seurat_v3")
    ap.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads (recommend 1 on HPC).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    args = ap.parse_args()

    set_thread_env(args.blas_threads)

    outdir = Path(args.outdir)
    cache_dir = outdir / "cache"
    logs_dir = outdir / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"prep_cache_{now_stamp()}.log"
    logger = setup_logger(log_path)

    input_path = Path(args.input_h5ad)
    if not input_path.exists():
        raise SystemExit(f"ERROR: input file not found: {input_path}")

    preprocessed_path = cache_dir / f"preprocessed_cells_hvg{args.n_top_genes}.h5ad"
    cogaps_input_path = cache_dir / f"cogaps_input_genesxcells_hvg{args.n_top_genes}_float64.h5ad"
    cfg_path = cache_dir / f"preprocess_config_hvg{args.n_top_genes}.json"

    if (not args.overwrite) and preprocessed_path.exists() and cogaps_input_path.exists() and cfg_path.exists():
        logger.info("[CACHE] Cache already exists. Use --overwrite to rebuild.")
        logger.info(f"[CACHE] {preprocessed_path}")
        logger.info(f"[CACHE] {cogaps_input_path}")
        logger.info(f"[CACHE] {cfg_path}")
        return

    cfg = {
        "input_h5ad": str(input_path),
        "min_cells": args.min_cells,
        "target_sum": args.target_sum,
        "n_top_genes": args.n_top_genes,
        "hvg_flavor": args.hvg_flavor,
        "blas_threads": args.blas_threads,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info(f"[CFG] Wrote {cfg_path}")

    t0 = time.time()
    logger.info("[DATA] Loading raw AnnData...")
    adata_raw = sc.read_h5ad(str(input_path))
    require_columns(adata_raw)

    logger.info("[PRE] Running frozen preprocessing...")
    adata_cells = preprocess_freeze(
        adata_raw,
        min_cells=args.min_cells,
        target_sum=args.target_sum,
        n_top_genes=args.n_top_genes,
        hvg_flavor=args.hvg_flavor,
        logger=logger,
    )
    require_columns(adata_cells)

    logger.info(f"[OUT] Writing {preprocessed_path} ...")
    atomic_write_h5ad(adata_cells, preprocessed_path)

    logger.info("[COGAPS] Creating CoGAPS input (genes×cells, dense float64) ...")
    adata_cg = to_cogaps_input(adata_cells, logger=logger)

    logger.info(f"[OUT] Writing {cogaps_input_path} ...")
    atomic_write_h5ad(adata_cg, cogaps_input_path)

    dt = time.time() - t0
    logger.info(f"✅ Done in {dt/60:.2f} minutes")
    logger.info(f"- preprocessed: {preprocessed_path}")
    logger.info(f"- cogaps input: {cogaps_input_path}")


if __name__ == "__main__":
    main()
