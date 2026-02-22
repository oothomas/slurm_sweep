#!/usr/bin/env python3
"""
cogaps_aggregate_results.py

Aggregate many single-process CoGAPS runs produced by a Slurm job array.

Reads:
  outdir/runs/*.metrics.json

Writes:
  outdir/per_run_metrics.csv
  outdir/summary_by_K.csv
  outdir/report.md
  outdir/figures/*
  outdir/chosen_model/*

Optional:
  --preprocessed-h5ad lets the script generate UMAP/boxplots/heatmaps for the chosen run.
"""

from __future__ import annotations

import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scanpy as sc


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cogaps_aggregate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path = log_dir / f"aggregate_{now_stamp()}.log"
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


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0


def save_heatmap(df: pd.DataFrame, title: str, outpath: Path) -> None:
    plt.figure(figsize=(0.6 * len(df.columns) + 4, 0.6 * len(df.index) + 3))
    plt.imshow(df.values, aspect="auto")
    plt.colorbar(label="value")
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def pattern_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).lower().startswith("pattern")]
    def key(c: str) -> int:
        s = str(c).replace("Pattern", "")
        return int(s) if s.isdigit() else 10**9
    return sorted(cols, key=key)


def summarize_stability(group: pd.DataFrame) -> Tuple[float, float, float, float, pd.DataFrame]:
    ok = group[(group["status"] == "ok") & group["ifn_corr"].notna() & group["top_genes"].notna()].copy()
    if len(ok) < 2:
        return np.nan, np.nan, np.nan, np.nan, pd.DataFrame()

    corrs = ok["ifn_corr"].astype(float).to_numpy()
    gene_sets = {f"seed{int(r.seed)}({r.ifn_pattern})": set(r.top_genes) for r in ok.itertuples()}
    keys = list(gene_sets.keys())

    mat = np.zeros((len(keys), len(keys)), dtype=float)
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            mat[i, j] = jaccard(gene_sets[k1], gene_sets[k2])
    jac = pd.DataFrame(mat, index=keys, columns=keys)

    vals = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            vals.append(jac.iloc[i, j])
    vals = np.array(vals, dtype=float)

    return float(np.mean(corrs)), float(np.std(corrs, ddof=1)), float(np.mean(vals)), float(np.min(vals)), jac


def choose_best_K(best_rows: List[Dict[str, object]], jaccard_target: float, max_corr_std: float) -> Dict[str, object]:
    df = pd.DataFrame(best_rows).copy()
    df["meets_stability"] = (df["jaccard_mean"] >= jaccard_target) & (df["ifn_corr_std"] <= max_corr_std)

    candidates = df[df["meets_stability"]].sort_values(by=["K", "ifn_corr_mean"], ascending=[True, False])
    if len(candidates) > 0:
        chosen = candidates.iloc[0].to_dict()
        chosen["selection_reason"] = "Meets stability thresholds; chose smallest K with strong IFN correlation."
        return chosen

    df_sorted = df.sort_values(by=["jaccard_mean", "ifn_corr_mean", "K"], ascending=[False, False, True])
    chosen = df_sorted.iloc[0].to_dict()
    chosen["selection_reason"] = "No K met thresholds; chose K with best stability (Jaccard), then best correlation."
    return chosen


def attach_patterns_to_adata(adata_cells: sc.AnnData, result: sc.AnnData) -> Tuple[sc.AnnData, List[str]]:
    ad = adata_cells.copy()
    pats = pattern_columns(result.var)
    P = result.var[pats].copy().reindex(ad.obs_names)
    if P.isna().any().any():
        raise ValueError("Could not align result.var (cells) to preprocessed adata.obs_names")
    for pat in pats:
        ad.obs[pat] = P[pat].values
    return ad, pats


def plot_mean_by_condition(adata: sc.AnnData, pattern_names: List[str], outpath: Path) -> None:
    df = adata.obs.groupby("condition")[pattern_names].mean()
    if "ctrl" in df.index and "stim" in df.index:
        df = df.loc[["ctrl", "stim"]]
    plt.figure(figsize=(0.6 * len(pattern_names) + 4, 3.5))
    plt.imshow(df.values, aspect="auto")
    plt.colorbar(label="mean pattern activity")
    plt.xticks(range(len(pattern_names)), pattern_names, rotation=45, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    plt.title("Mean CoGAPS pattern activity by condition")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_isg_boxplots(adata: sc.AnnData, isg_pattern: str, out_all: Path, out_split: Path) -> None:
    df = adata.obs[["cell_type", "condition", isg_pattern]].copy()

    order = df.groupby("cell_type")[isg_pattern].mean().sort_values(ascending=False).index.tolist()
    groups = [df[df["cell_type"] == ct][isg_pattern].values for ct in order]

    plt.figure(figsize=(12, 4))
    plt.boxplot(groups, labels=order, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"{isg_pattern} activity")
    plt.title("IFN/ISG pattern activity across immune cell types (all cells)")
    plt.tight_layout()
    out_all.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_all, dpi=200)
    plt.close()

    plt.figure(figsize=(12, 5))
    positions, data, labels = [], [], []
    pos = 1
    for ct in order:
        for cond in ["ctrl", "stim"]:
            vals = df[(df["cell_type"] == ct) & (df["condition"] == cond)][isg_pattern].values
            data.append(vals)
            positions.append(pos)
            labels.append(f"{ct}\n{cond}")
            pos += 1
        pos += 0.5
    plt.boxplot(data, positions=positions, showfliers=False)
    plt.xticks(positions, labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel(f"{isg_pattern} activity")
    plt.title("IFN/ISG pattern activity by cell type and condition")
    plt.tight_layout()
    out_split.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_split, dpi=200)
    plt.close()


def compute_umap_if_missing(adata: sc.AnnData, n_neighbors: int, n_pcs: int) -> sc.AnnData:
    ad = adata.copy()
    if "X_umap" in ad.obsm:
        return ad
    sc.pp.pca(ad, n_comps=n_pcs)
    sc.pp.neighbors(ad, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(ad)
    return ad


def plot_umap(adata: sc.AnnData, color: List[str], outpath: Path) -> None:
    sc.pl.umap(adata, color=color, show=False)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()


def write_report(outdir: Path, summary_df: pd.DataFrame, chosen: Dict[str, object], chosen_ifn_genes: List[str], report_path: Path) -> None:
    lines = []
    lines.append("# CoGAPS Stability Report (Single-process on HPC)\n\n")
    lines.append(f"- Summary rows: {len(summary_df)}\n\n")
    lines.append("## Selected model\n")
    lines.append(f"- **Chosen K**: {int(chosen['K'])}\n")
    lines.append(f"- **Chosen iterations**: {int(chosen['n_iter'])}\n")
    lines.append(f"- **Reason**: {chosen.get('selection_reason','')}\n")
    lines.append(f"- IFN corr mean/std: {chosen.get('ifn_corr_mean', float('nan')):.3f} / {chosen.get('ifn_corr_std', float('nan')):.3f}\n")
    lines.append(f"- IFN top-gene Jaccard mean/min: {chosen.get('jaccard_mean', float('nan')):.3f} / {chosen.get('jaccard_min', float('nan')):.3f}\n\n")
    lines.append("## IFN/ISG evidence (chosen run)\n")
    for g in chosen_ifn_genes:
        lines.append(f"- {g}\n")
    report_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate sweep outputs from Slurm array (single-process CoGAPS runs).")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--preprocessed-h5ad", default=None, help="Optional: preprocessed_cells.h5ad for plots.")
    ap.add_argument("--jaccard-target", type=float, default=0.55)
    ap.add_argument("--max-corr-std", type=float, default=0.10)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-pcs", type=int, default=50)
    ap.add_argument("--no-umap", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    runs_dir = outdir / "runs"
    figs_dir = outdir / "figures"
    logs_dir = outdir / "logs"
    chosen_dir = outdir / "chosen_model"
    figs_dir.mkdir(parents=True, exist_ok=True)
    chosen_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(logs_dir)

    metric_files = sorted(runs_dir.glob("*.metrics.json"))
    if not metric_files:
        raise SystemExit(f"ERROR: no metrics found in {runs_dir}")

    logger.info(f"[IN] Found {len(metric_files)} metrics files")

    rows = []
    for p in metric_files:
        m = json.loads(p.read_text(encoding="utf-8"))
        rows.append({
            "K": m.get("K"),
            "seed": m.get("seed"),
            "n_iter": m.get("n_iter"),
            "status": m.get("status"),
            "runtime_sec": m.get("runtime_sec"),
            "ifn_pattern": m.get("ifn_pattern"),
            "ifn_corr": m.get("ifn_corr"),
            "top_genes": m.get("top_genes"),
            "result_path": m.get("result_path"),
            "metrics_path": m.get("metrics_path"),
        })

    per_run_df = pd.DataFrame(rows).sort_values(["K", "n_iter", "seed"])
    per_run_csv = outdir / "per_run_metrics.csv"
    per_run_df.to_csv(per_run_csv, index=False)
    logger.info(f"[OUT] Wrote {per_run_csv}")

    summary_rows = []
    for (K, n_iter), g in per_run_df.groupby(["K", "n_iter"], dropna=True):
        corr_mean, corr_std, jac_mean, jac_min, jac = summarize_stability(g)
        summary_rows.append({
            "K": int(K),
            "n_iter": int(n_iter),
            "n_ok": int((g["status"] == "ok").sum()),
            "ifn_corr_mean": corr_mean,
            "ifn_corr_std": corr_std,
            "jaccard_mean": jac_mean,
            "jaccard_min": jac_min,
        })
        if isinstance(jac, pd.DataFrame) and jac.shape[0] > 0:
            save_heatmap(jac, f"IFN top-gene Jaccard (K={K}, iter={n_iter})", figs_dir / f"jaccard_ifn_topgenes_K{K}_iter{n_iter}.png")

    summary_df = pd.DataFrame(summary_rows).sort_values(["K", "n_iter"])
    summary_csv = outdir / "summary_by_K.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"[OUT] Wrote {summary_csv}")

    best_rows = []
    for K, dfk in summary_df.groupby("K", dropna=True):
        meets = dfk[(dfk["jaccard_mean"] >= args.jaccard_target) & (dfk["ifn_corr_std"] <= args.max_corr_std)]
        if len(meets) > 0:
            best_rows.append(meets.iloc[0].to_dict())
        else:
            best_rows.append(dfk.iloc[-1].to_dict())

    chosen = choose_best_K(best_rows, jaccard_target=args.jaccard_target, max_corr_std=args.max_corr_std)
    chosen_K = int(chosen["K"])
    chosen_iter = int(chosen["n_iter"])
    logger.info(f"[SELECT] Chosen K={chosen_K}, iter={chosen_iter} | {chosen.get('selection_reason','')}")

    cand = per_run_df[(per_run_df["K"] == chosen_K) & (per_run_df["n_iter"] == chosen_iter) & (per_run_df["status"] == "ok")].copy()
    if len(cand) == 0:
        raise SystemExit("ERROR: No successful runs for chosen K/iter.")

    cand = cand.sort_values("ifn_corr", ascending=False)
    best = cand.iloc[0]
    best_result_path = Path(best["result_path"])
    best_metrics_path = Path(best["metrics_path"])

    if chosen_dir.exists():
        shutil.rmtree(chosen_dir)
    chosen_dir.mkdir(parents=True, exist_ok=True)
    if best_result_path.exists():
        shutil.copy2(best_result_path, chosen_dir / best_result_path.name)
    if best_metrics_path.exists():
        shutil.copy2(best_metrics_path, chosen_dir / best_metrics_path.name)

    chosen_ifn_genes: List[str] = []
    if args.preprocessed_h5ad is not None:
        pre_path = Path(args.preprocessed_h5ad)
        if not pre_path.exists():
            raise SystemExit(f"ERROR: preprocessed-h5ad not found: {pre_path}")

        logger.info("[PLOT] Loading preprocessed AnnData...")
        adata_cells = sc.read_h5ad(str(pre_path))
        if "condition" not in adata_cells.obs and "label" in adata_cells.obs:
            adata_cells.obs["condition"] = adata_cells.obs["label"]

        logger.info("[PLOT] Loading chosen CoGAPS result...")
        result = sc.read_h5ad(str(best_result_path))

        adata_with_pats, pat_names = attach_patterns_to_adata(adata_cells, result)

        ifn_pat = best["ifn_pattern"]
        if ifn_pat is None:
            raise SystemExit("ERROR: chosen run missing ifn_pattern in metrics")

        A = result.obs[pattern_columns(result.obs)]
        chosen_ifn_genes = (
            A[ifn_pat]
            .sort_values(ascending=False)
            .head(30)
            .index.astype(str)
            .tolist()
        )

        plot_mean_by_condition(adata_with_pats, pat_names, figs_dir / "mean_pattern_by_condition.png")
        plot_isg_boxplots(adata_with_pats, ifn_pat, figs_dir / "ifn_pattern_by_celltype_all.png", figs_dir / "ifn_pattern_by_celltype_split_condition.png")

        if not args.no_umap:
            logger.info("[PLOT] Computing UMAP (if missing)...")
            ad_umap = compute_umap_if_missing(adata_with_pats, n_neighbors=args.umap_neighbors, n_pcs=args.umap_pcs)
            plot_umap(ad_umap, ["condition", ifn_pat], figs_dir / "umap_condition_and_ifn.png")

    report_path = outdir / "report.md"
    write_report(outdir, summary_df, chosen, chosen_ifn_genes, report_path)
    logger.info(f"[OUT] Wrote {report_path}")

    logger.info("✅ Done.")
    logger.info(f"- summary: {summary_csv}")
    logger.info(f"- per-run:  {per_run_csv}")
    logger.info(f"- figures:  {figs_dir}/")
    logger.info(f"- chosen:   {chosen_dir}/")


if __name__ == "__main__":
    main()
