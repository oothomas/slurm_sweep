#!/usr/bin/env python3
"""
cogaps_run_one_singleprocess.py

Run ONE CoGAPS job (single-process) for a given (K, seed, n_iter) using a cached
genes×cells input matrix.

Key fixes vs earlier version
----------------------------
1) The cache-skip logic now only skips when metrics.status == "ok".
   If a previous run failed, re-running the same job will try again automatically.

2) IFN/condition correlation no longer *requires* 'condition' in cogaps_input.var.
   If missing, it falls back to loading the preprocessed cells file from:
       outdir/cache/preprocessed_cells_*.h5ad
   and aligns by cell IDs.

Usage (typical)
---------------
python cogaps_run_one_singleprocess.py \
  --cogaps-input-h5ad results_cogaps_singleprocess_hpc/cache/cogaps_input_genesxcells_hvg3000_float64.h5ad \
  --outdir results_cogaps_singleprocess_hpc \
  --k 7 --seed 1 --n-iter 2000 \
  --use-sparse-opt \
  --cogaps-threads 1

Notes
-----
- Keep this "single-process" by NOT setting the 'distributed' parameter at all.
- Control BLAS threads via --blas-threads (important on HPC).
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc

from PyCoGAPS.parameters import CoParams, setParams
from PyCoGAPS.pycogaps_main import CoGAPS


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"run_one_{now_stamp()}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_blas_threads(n: int) -> None:
    n = str(int(n))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n  # macOS Accelerate


def atomic_write_h5ad(adata: sc.AnnData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    adata.write_h5ad(tmp_path)
    if tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Atomic write failed: temp file is 0 bytes: {tmp_path}")
    tmp_path.replace(out_path)


def find_preprocessed_cells(outdir: Path) -> Optional[Path]:
    """
    Best-effort: locate outdir/cache/preprocessed_cells_*.h5ad.
    If multiple exist, prefer one containing 'hvg3000' in the name.
    """
    cache = outdir / "cache"
    if not cache.exists():
        return None
    candidates = sorted(cache.glob("preprocessed_cells_*.h5ad"))
    if not candidates:
        return None
    # prefer common naming
    for p in candidates:
        if "hvg3000" in p.name:
            return p
    return candidates[0]


def ensure_condition_column(adata_cells: sc.AnnData) -> None:
    if "condition" not in adata_cells.obs:
        if "label" in adata_cells.obs:
            adata_cells.obs["condition"] = adata_cells.obs["label"]
        elif "stim" in adata_cells.obs:
            adata_cells.obs["condition"] = adata_cells.obs["stim"]
    if "condition" not in adata_cells.obs:
        raise ValueError("Could not find/create 'condition' in preprocessed cells .obs")


def compute_ifn_pattern(
    result: sc.AnnData,
    cogaps_input: sc.AnnData,
    *,
    outdir: Path,
    top_n: int,
    stim_label: str = "stim",
) -> Tuple[str, float, List[str]]:
    """
    Identify IFN-associated pattern = pattern whose per-cell activity most correlates with stim.

    - CoGAPS result.var: pattern activities per cell (cells are variables)
    - Condition vector:
        1) try cogaps_input.var['condition'] (cells are vars in cogaps_input)
        2) else load preprocessed cells from outdir/cache and align by cell IDs
    - Gene loadings come from result.obs (genes are obs)
    """
    # Pattern columns (robust ordering Pattern1..K)
    pats = [c for c in result.var.columns if str(c).lower().startswith("pattern")]
    pats = sorted(pats, key=lambda s: int(str(s).replace("Pattern", "")) if str(s).replace("Pattern", "").isdigit() else 10**9)
    if not pats:
        raise ValueError("No Pattern* columns found in result.var")

    # Build condition vector aligned to result.var_names (cells)
    cells = result.var_names.astype(str)

    cond_series = None
    if "condition" in cogaps_input.var.columns:
        # Align using cell IDs
        if cogaps_input.var_names.astype(str).equals(cells):
            cond_series = cogaps_input.var["condition"]
        else:
            # reindex by cell name
            cond_series = cogaps_input.var.reindex(cells)["condition"]
    else:
        # Fallback: load preprocessed cells file
        pre_path = find_preprocessed_cells(outdir)
        if pre_path is None:
            raise ValueError(
                "cogaps_input.var missing 'condition' and could not find preprocessed_cells_*.h5ad in outdir/cache"
            )
        ad_cells = sc.read_h5ad(str(pre_path))
        ensure_condition_column(ad_cells)
        # align to cell IDs
        cond_series = ad_cells.obs.reindex(cells)["condition"]

    if cond_series is None or cond_series.isna().any():
        raise ValueError("Could not align condition labels to CoGAPS result cells (NaNs after reindex).")

    cond = (cond_series.astype(str) == stim_label).astype(int)

    # Correlation per pattern
    P = result.var[pats]
    corrs = {pat: float(P[pat].corr(cond)) for pat in pats}
    ifn_pattern = max(corrs, key=lambda k: corrs[k])
    ifn_corr = corrs[ifn_pattern]

    # Top genes from result.obs
    pats_obs = [c for c in result.obs.columns if str(c).lower().startswith("pattern")]
    pats_obs = sorted(pats_obs, key=lambda s: int(str(s).replace("Pattern", "")) if str(s).replace("Pattern", "").isdigit() else 10**9)
    if ifn_pattern not in pats_obs:
        raise ValueError(f"Expected {ifn_pattern} in result.obs Pattern columns; found {pats_obs[:5]}...")

    top_genes = (
        result.obs[ifn_pattern]
        .sort_values(ascending=False)
        .head(top_n)
        .index.astype(str)
        .tolist()
    )
    return ifn_pattern, ifn_corr, top_genes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cogaps-input-h5ad", required=True, help="Cached genes×cells AnnData (dense float64)")
    ap.add_argument("--outdir", default="results_cogaps_singleprocess_hpc", help="Output dir containing runs/ logs/ cache/")
    ap.add_argument("--k", type=int, required=True, help="Number of patterns (K)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-iter", type=int, required=True)
    ap.add_argument("--top-genes", type=int, default=50)
    ap.add_argument("--stim-label", type=str, default="stim", help="Which condition value counts as 'stim' (default: stim)")
    ap.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads (important on HPC)")
    ap.add_argument("--cogaps-threads", type=int, default=1, help="Threads passed to CoGAPS(nThreads=...)")
    ap.add_argument("--use-sparse-opt", action="store_true", help="Use sparseOptimization / useSparseOptimization")
    ap.add_argument("--no-sparse-opt", action="store_true", help="Disable sparse optimization")
    ap.add_argument("--force-rerun", action="store_true", help="Re-run even if metrics exist and status==ok")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    runs_dir = outdir / "runs"
    logs_dir = outdir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag = f"K{args.k}_seed{args.seed}_iter{args.n_iter}"
    run_log = logs_dir / f"run_{tag}.log"
    logger = setup_logger(run_log)

    set_blas_threads(args.blas_threads)

    result_path = runs_dir / f"cogaps_{tag}.h5ad"
    metrics_path = runs_dir / f"cogaps_{tag}.metrics.json"

    # Cache skip ONLY if status==ok
    if result_path.exists() and metrics_path.exists() and (not args.force_rerun):
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if m.get("status") == "ok":
                logger.info(f"[CACHE] status=ok; skipping {tag}")
                logger.info(f"[CACHE] result:  {result_path}")
                logger.info(f"[CACHE] metrics: {metrics_path}")
                return
            else:
                logger.info(f"[CACHE] Found previous status={m.get('status')}; re-running {tag}")
        except Exception:
            logger.info("[CACHE] Could not parse metrics; re-running.")

    # If previous partial result exists, keep it for debugging but avoid mixing with new writes
    # (Atomic write will overwrite cleanly at the end.)
    t0 = time.time()
    status = "ok"
    err_txt = None
    ifn_pattern = None
    ifn_corr = None
    top_genes = None

    try:
        cogaps_input = sc.read_h5ad(args.cogaps_input_h5ad)

        params = CoParams(adata=cogaps_input)

        # IMPORTANT: do NOT set 'distributed' at all (keep single-process).
        sparse_opt = True
        if args.no_sparse_opt:
            sparse_opt = False
        elif args.use_sparse_opt:
            sparse_opt = True

        setParams(
            params,
            {
                "nPatterns": int(args.k),
                "nIterations": int(args.n_iter),
                "seed": int(args.seed),
                "useSparseOptimization": bool(sparse_opt),
            },
        )

        logger.info(f"[RUN] Starting {tag} | sparse_opt={sparse_opt} | blas_threads={args.blas_threads} | cogaps_threads={args.cogaps_threads}")
        result = CoGAPS(cogaps_input, params, nThreads=int(args.cogaps_threads))

        logger.info(f"[INFO] Writing result: {result_path}")
        atomic_write_h5ad(result, result_path)

        ifn_pattern, ifn_corr, top_genes = compute_ifn_pattern(
            result=result,
            cogaps_input=cogaps_input,
            outdir=outdir,
            top_n=int(args.top_genes),
            stim_label=args.stim_label,
        )
        logger.info(f"[METRICS] IFN pattern: {ifn_pattern} corr={ifn_corr:.3f}")
        logger.info(f"[METRICS] Top {args.top_genes} genes (first 10): {top_genes[:10]}")

    except Exception as e:
        status = "failed"
        err_txt = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error("[ERROR] Run failed:\n" + err_txt)

    runtime = time.time() - t0
    payload = {
        "K": int(args.k),
        "seed": int(args.seed),
        "n_iter": int(args.n_iter),
        "status": status,
        "runtime_sec": float(runtime),
        "result_path": str(result_path),
        "metrics_path": str(metrics_path),
        "ifn_pattern": ifn_pattern,
        "ifn_corr": ifn_corr,
        "top_genes": top_genes,
        "error": err_txt,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = metrics_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(metrics_path)

    logger.info(f"[DONE] {tag} | {status.upper()} | {runtime/60:.2f} min | metrics: {metrics_path}")


if __name__ == "__main__":
    main()
