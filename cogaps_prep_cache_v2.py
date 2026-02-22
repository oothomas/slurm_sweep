#!/usr/bin/env python3
"""
cogaps_prep_cache.py

Prepare a *frozen* preprocessed dataset + a CoGAPS input matrix for HPC sweeps.

Why this exists
---------------
Running preprocessing inside every Slurm task wastes time and creates subtle inconsistencies.
This script runs preprocessing once, then writes two cached files:

1) preprocessed_cells_*.h5ad
   - shape: cells × genes (Scanpy orientation)
   - contains .obs metadata (including condition, cell_type, etc.)

2) cogaps_input_genesxcells_*.h5ad
   - shape: genes × cells (CoGAPS orientation)
   - dense float64 matrix (what PyCoGAPS expects in practice)
   - IMPORTANT: cell metadata is stored in .var (because cells are variables here)

This script *explicitly* copies cell metadata into cogaps_input.var so downstream code can always
access `cogaps_input.var['condition']`, even if AnnData transpose behavior changes across versions.

Usage
-----
python cogaps_prep_cache.py \
  --raw-h5ad kang_counts_25k.h5ad \
  --outdir results_cogaps_singleprocess_hpc \
  --n-top-genes 3000

Outputs
-------
$outdir/cache/preprocessed_cells_hvg{N}.h5ad
$outdir/cache/cogaps_input_genesxcells_hvg{N}_float64.h5ad
$outdir/cache/preprocess_config_hvg{N}.json
"""

from __future__ import annotations

import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import scanpy as sc


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"prep_cache_{now_stamp()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"[LOG] {log_path}")
    return logger


def require_columns(adata: sc.AnnData) -> None:
    # Ensure "condition" exists (common cases: label or stim)
    if "condition" not in adata.obs:
        if "label" in adata.obs:
            adata.obs["condition"] = adata.obs["label"]
        elif "stim" in adata.obs:
            # sometimes stored as 'stim' with values like 'stim'/'ctrl'
            adata.obs["condition"] = adata.obs["stim"]
    if "condition" not in adata.obs:
        raise ValueError(
            "Missing required obs column 'condition' (and no 'label'/'stim' fallback found)."
        )

    # cell_type isn't strictly required for running CoGAPS, but the reporting expects it.
    if "cell_type" not in adata.obs:
        # Keep going, but make it explicit.
        adata.obs["cell_type"] = "unknown"


def preprocess_freeze(
    adata_raw: sc.AnnData,
    *,
    min_cells: int,
    target_sum: float,
    n_top_genes: int,
    hvg_flavor: str,
    logger: logging.Logger,
) -> sc.AnnData:
    """
    Frozen preprocessing:
      1) counts layer
      2) filter_genes(min_cells)
      3) normalize_total(target_sum) + log1p
      4) HVGs using Seurat v3 on raw counts layer
    """
    adata = adata_raw.copy()
    logger.info(f"[PRE] Start: cells={adata.n_obs:,}, genes={adata.n_vars:,}")

    adata.layers["counts"] = adata.X.copy()

    before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info(f"[PRE] filter_genes(min_cells={min_cells}): {before:,} -> {adata.n_vars:,}")

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


def to_cogaps_input(preprocessed_cells: sc.AnnData, logger: logging.Logger) -> sc.AnnData:
    """
    Build genes×cells dense float64 AnnData for CoGAPS.
    Ensures cell metadata is present in .var.
    """
    # Start with transpose
    cg = preprocessed_cells.T.copy()  # genes × cells

    # Force dense float64
    X = cg.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    cg.X = np.asarray(X, dtype=np.float64)

    # Explicitly carry metadata:
    # - cells are VARIABLES now -> put original .obs into .var
    cg.var = preprocessed_cells.obs.copy()
    cg.var_names = preprocessed_cells.obs_names.copy()

    # - genes are OBSERVATIONS now -> put original .var into .obs
    cg.obs = preprocessed_cells.var.copy()
    cg.obs_names = preprocessed_cells.var_names.copy()

    if "condition" not in cg.var:
        raise ValueError("Internal error: expected 'condition' in cg.var after copying metadata.")

    logger.info(f"[COGAPS] Input matrix: {cg.shape} (genes×cells), dtype={cg.X.dtype}")
    logger.info(f"[COGAPS] cg.var columns (first 8): {list(cg.var.columns)[:8]}")
    return cg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-h5ad", required=True, help="Raw kang_counts_25k.h5ad path")
    ap.add_argument("--outdir", default="results_cogaps_singleprocess_hpc", help="Output directory")
    ap.add_argument("--min-cells", type=int, default=3)
    ap.add_argument("--target-sum", type=float, default=1e4)
    ap.add_argument("--hvg-flavor", default="seurat_v3")
    ap.add_argument("--n-top-genes", type=int, default=3000)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    cache_dir = outdir / "cache"
    logs_dir = outdir / "logs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"prep_cache_{now_stamp()}.log"
    logger = setup_logger(log_path)

    raw_path = Path(args.raw_h5ad)
    if not raw_path.exists():
        raise SystemExit(f"ERROR: raw file not found: {raw_path}")

    tag = f"hvg{args.n_top_genes}"
    preprocessed_path = cache_dir / f"preprocessed_cells_{tag}.h5ad"
    cg_path = cache_dir / f"cogaps_input_genesxcells_{tag}_float64.h5ad"
    cfg_path = cache_dir / f"preprocess_config_{tag}.json"

    if not args.overwrite and preprocessed_path.exists() and cg_path.exists() and cfg_path.exists():
        logger.info("[SKIP] Cache files already exist (use --overwrite to rebuild).")
        logger.info(f"- {preprocessed_path}")
        logger.info(f"- {cg_path}")
        logger.info(f"- {cfg_path}")
        return

    t0 = time.time()
    logger.info("[DATA] Loading raw AnnData...")
    adata_raw = sc.read_h5ad(str(raw_path))
    require_columns(adata_raw)

    cfg = {
        "raw_h5ad": str(raw_path),
        "min_cells": args.min_cells,
        "target_sum": args.target_sum,
        "hvg_flavor": args.hvg_flavor,
        "n_top_genes": args.n_top_genes,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    logger.info(f"[CFG] Wrote {cfg_path}")

    logger.info("[PRE] Running frozen preprocessing...")
    pre = preprocess_freeze(
        adata_raw,
        min_cells=args.min_cells,
        target_sum=args.target_sum,
        n_top_genes=args.n_top_genes,
        hvg_flavor=args.hvg_flavor,
        logger=logger,
    )
    require_columns(pre)

    logger.info(f"[OUT] Writing {preprocessed_path} ...")
    pre.write_h5ad(preprocessed_path)

    logger.info("[COGAPS] Creating CoGAPS input (genes×cells, dense float64) ...")
    cg = to_cogaps_input(pre, logger=logger)

    logger.info(f"[OUT] Writing {cg_path} ...")
    cg.write_h5ad(cg_path)

    dt = time.time() - t0
    logger.info(f"✅ Done in {dt/60:.2f} minutes")
    logger.info(f"- preprocessed: {preprocessed_path}")
    logger.info(f"- cogaps input: {cg_path}")


if __name__ == "__main__":
    main()
