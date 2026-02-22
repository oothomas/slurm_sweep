#!/usr/bin/env python3
"""
make_cogaps_jobs_tsv.py

Create a jobs file (TSV) for Slurm job arrays.

Each line is:
  K<TAB>seed<TAB>n_iter

Ordering is deterministic:
  for K in K_grid:
    for n_iter in iters:
      for seed in seeds:

Example:
  python make_cogaps_jobs_tsv.py \
    --k-grid 7,9,11,13 \
    --seeds 1,2,3,4,5 \
    --iters 2000,10000,20000 \
    --out results_cogaps_singleprocess_hpc/jobs.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Create jobs.tsv for Slurm arrays (no header).")

    ap.add_argument("--k-grid", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--iters", required=True)
    ap.add_argument("--out", default="jobs.tsv")

    args = ap.parse_args()

    K_grid = parse_int_list(args.k_grid)
    seeds = parse_int_list(args.seeds)
    iters = parse_int_list(args.iters)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for K in K_grid:
        for n_iter in iters:
            for seed in seeds:
                lines.append(f"{K}\t{seed}\t{n_iter}\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} with {len(lines)} jobs.")


if __name__ == "__main__":
    main()
