#!/usr/bin/env python3
"""
cogaps_run_one_singleprocess.py

Run ONE (K, seed, n_iter) CoGAPS job in **single-process** mode.

Designed for Slurm job arrays: each array task runs exactly one configuration and writes:
  - runs/cogaps_K{K}_seed{seed}_iter{n_iter}.h5ad
  - runs/cogaps_K{K}_seed{seed}_iter{n_iter}.metrics.json
  - logs/run_K{K}_seed{seed}_iter{n_iter}.log  (captured stdout/stderr)

Key point
---------
We do NOT set params['distributed'] at all. That keeps CoGAPS single-process.

Example:
  python cogaps_run_one_singleprocess.py \
    --cogaps-input-h5ad results_cogaps_singleprocess_hpc/cache/cogaps_input_genesxcells_hvg3000_float64.h5ad \
    --outdir results_cogaps_singleprocess_hpc \
    --k 7 --seed 3 --n-iter 20000 \
    --blas-threads 1 --cogaps-threads 1
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from PyCoGAPS.parameters import CoParams, setParams
from PyCoGAPS.pycogaps_main import CoGAPS


class TeeTextIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> None:
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


@contextmanager
def tee_stdio(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = TeeTextIO(old_out, f)
    sys.stderr = TeeTextIO(old_err, f)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout, sys.stderr = old_out, old_err
        f.close()


def atomic_write_h5ad(adata: sc.AnnData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    adata.write_h5ad(tmp_path)
    if tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Atomic write failed (0 bytes): {tmp_path}")
    tmp_path.replace(out_path)


def atomic_write_json(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)


def set_thread_env(n_threads: int) -> None:
    n = str(int(n_threads))
    os.environ["OMP_NUM_THREADS"] = n
    os.environ["OPENBLAS_NUM_THREADS"] = n
    os.environ["MKL_NUM_THREADS"] = n
    os.environ["NUMEXPR_NUM_THREADS"] = n
    os.environ["VECLIB_MAXIMUM_THREADS"] = n


def pattern_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).lower().startswith("pattern")]
    def key(c: str) -> int:
        s = str(c).replace("Pattern", "")
        return int(s) if s.isdigit() else 10**9
    return sorted(cols, key=key)


def compute_ifn_pattern_from_cogaps_input(
    cogaps_input: sc.AnnData,
    result: sc.AnnData,
    top_n_genes: int,
) -> Tuple[str, float, List[str]]:
    if "condition" not in cogaps_input.var.columns:
        raise ValueError("cogaps_input.var missing 'condition' (build cache with cogaps_prep_cache.py)")
    cond = (cogaps_input.var["condition"].astype(str) == "stim").astype(int)

    pats_var = pattern_columns(result.var)
    P = result.var[pats_var].copy()

    if not np.all(P.index.values == cogaps_input.var_names.values):
        P = P.reindex(cogaps_input.var_names)
        if P.isna().any().any():
            raise ValueError("Could not align result.var (cells) to cogaps_input.var_names")

    corrs = {pat: float(P[pat].corr(cond)) for pat in P.columns}
    ifn_pattern = max(corrs, key=lambda k: corrs[k])
    ifn_corr = float(corrs[ifn_pattern])

    pats_obs = pattern_columns(result.obs)
    A = result.obs[pats_obs].copy()
    top_genes = (
        A[ifn_pattern]
        .sort_values(ascending=False)
        .head(top_n_genes)
        .index.astype(str)
        .tolist()
    )
    return ifn_pattern, ifn_corr, top_genes


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one single-process CoGAPS job (Slurm-array friendly).")

    ap.add_argument("--cogaps-input-h5ad", required=True, help="Cache file from cogaps_prep_cache.py (genes×cells, dense float64).")

    ap.add_argument("--outdir", default="results_cogaps_singleprocess_hpc", help="Output directory.")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-iter", type=int, required=True)

    ap.add_argument("--top-genes", type=int, default=50, help="Top genes for IFN stability metric.")
    ap.add_argument("--blas-threads", type=int, default=1, help="BLAS/OpenMP threads (recommend 1 on HPC).")
    ap.add_argument("--cogaps-threads", type=int, default=1, help="Threads passed to CoGAPS (nThreads). Keep 1.")

    ap.add_argument("--use-sparse-opt", action="store_true", default=True)
    ap.add_argument("--no-sparse-opt", action="store_true")

    ap.add_argument("--force-rerun", action="store_true", help="Rerun even if outputs already exist.")
    args = ap.parse_args()

    use_sparse_opt = bool(args.use_sparse_opt) and (not args.no_sparse_opt)
    set_thread_env(args.blas_threads)

    outdir = Path(args.outdir)
    runs_dir = outdir / "runs"
    logs_dir = outdir / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag = f"K{args.k}_seed{args.seed}_iter{args.n_iter}"
    result_path = runs_dir / f"cogaps_{tag}.h5ad"
    metrics_path = runs_dir / f"cogaps_{tag}.metrics.json"
    run_log = logs_dir / f"run_{tag}.log"

    if (not args.force_rerun) and result_path.exists() and metrics_path.exists():
        print(f"[CACHE] Exists: {result_path}")
        return

    cogaps_input_path = Path(args.cogaps_input_h5ad)
    if not cogaps_input_path.exists():
        raise SystemExit(f"ERROR: cogaps-input file not found: {cogaps_input_path}")

    start = time.time()
    status = "ok"
    err: Optional[str] = None
    ifn_pattern: Optional[str] = None
    ifn_corr: Optional[float] = None
    top_genes: Optional[List[str]] = None

    with tee_stdio(run_log):
        try:
            print(f"[INFO] Loading CoGAPS input: {cogaps_input_path}")
            adata_cg = sc.read_h5ad(str(cogaps_input_path))
            print(f"[INFO] Input shape: {adata_cg.shape} (genes×cells)")

            params = CoParams(adata=adata_cg)

            # IMPORTANT: single-process => DO NOT set 'distributed'
            setParams(
                params,
                {
                    "nPatterns": int(args.k),
                    "nIterations": int(args.n_iter),
                    "seed": int(args.seed),
                    "useSparseOptimization": bool(use_sparse_opt),
                },
            )

            print("\n[PyCoGAPS] Parameters:")
            params.printParams()
            print("")

            result = CoGAPS(adata_cg, params, nThreads=int(args.cogaps_threads))

            print(f"[INFO] Writing result: {result_path}")
            atomic_write_h5ad(result, result_path)

            ifn_pattern, ifn_corr, top_genes = compute_ifn_pattern_from_cogaps_input(
                cogaps_input=adata_cg,
                result=result,
                top_n_genes=int(args.top_genes),
            )
            print(f"\n[METRICS] IFN-associated pattern: {ifn_pattern} (corr={ifn_corr:.6f})")
            print(f"[METRICS] Top {args.top_genes} genes (first 10): {top_genes[:10]}")
        except Exception as e:
            status = "failed"
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            print("\n[ERROR] Run failed:")
            print(err)

    runtime = time.time() - start

    payload = {
        "K": int(args.k),
        "seed": int(args.seed),
        "n_iter": int(args.n_iter),
        "distributed": None,
        "status": status,
        "runtime_sec": float(runtime),
        "result_path": str(result_path),
        "metrics_path": str(metrics_path),
        "ifn_pattern": ifn_pattern,
        "ifn_corr": ifn_corr,
        "top_genes": top_genes,
        "error": err,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "blas_threads": int(args.blas_threads),
        "cogaps_threads": int(args.cogaps_threads),
    }
    atomic_write_json(payload, metrics_path)

    if status == "ok":
        print(f"[DONE] {tag} | {runtime/60:.2f} min | IFN={ifn_pattern} corr={ifn_corr:.3f}")
    else:
        print(f"[DONE] {tag} | FAILED | {runtime/60:.2f} min | see: {run_log}")


if __name__ == "__main__":
    main()
