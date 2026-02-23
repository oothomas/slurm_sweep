#!/bin/bash
#SBATCH --job-name=cogaps_prep_cache
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "================"

# -----------------------------------------------------------------------------
# Environment (matches your known-good template)
# -----------------------------------------------------------------------------
# Fix for: 'PYTHONPATH: unbound variable' under `set -u`
export PYTHONPATH="${PYTHONPATH:-}"

source /app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate oshane-jlab

# Make PyCoGAPS importable (your source checkout lives here)
export PYTHONPATH="$HOME/src/pycogaps:${PYTHONPATH:-}"

# Prep uses Scanpy/NumPy heavily; allow BLAS to use the allocated cores.
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export NUMEXPR_NUM_THREADS="$NTHREADS"

# Run from the directory you submit sbatch from (recommended: repo root)
cd "${SLURM_SUBMIT_DIR:-$HOME/CS4/slurm_sweep}"

# -----------------------------------------------------------------------------
# Paths + grid settings
# -----------------------------------------------------------------------------
H5AD="kang_counts_25k.h5ad"
OUTDIR="results_cogaps_singleprocess_hpc"

# Must match sweep + aggregation scripts
K_GRID="7,9,11,13"
SEEDS="1,2,3,4,5"
ITERS="2000,10000,20000"

# -----------------------------------------------------------------------------
# 1) Preprocess once -> write cache files
# -----------------------------------------------------------------------------
python3 cogaps_prep_cache.py \
  --input-h5ad "${H5AD}" \
  --outdir "${OUTDIR}" \
  --min-cells 3 \
  --target-sum 10000 \
  --n-top-genes 3000 \
  --hvg-flavor seurat_v3

# -----------------------------------------------------------------------------
# 2) Build jobs.tsv (NO header). Each line = one (K, seed, n_iter) run.
#    NOTE: make_cogaps_jobs_tsv.py expects --out (path), not --outdir.
# -----------------------------------------------------------------------------
python3 make_cogaps_jobs_tsv.py \
  --k-grid "${K_GRID}" \
  --seeds "${SEEDS}" \
  --iters "${ITERS}" \
  --out "${OUTDIR}/jobs.tsv"

echo "\n=== Cache + jobs created ==="
ls -lh \
  "${OUTDIR}/cache/preprocessed_cells_hvg3000.h5ad" \
  "${OUTDIR}/cache/cogaps_input_genesxcells_hvg3000_float64.h5ad" \
  "${OUTDIR}/cache/preprocess_config_hvg3000.json" \
  "${OUTDIR}/jobs.tsv"

echo "Num jobs: $(wc -l < "${OUTDIR}/jobs.tsv")"
echo "============================="
