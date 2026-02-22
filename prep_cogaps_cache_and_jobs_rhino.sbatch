#!/bin/bash
#SBATCH --job-name=cogaps_prep_cache
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# One-time cache build for the CoGAPS sweep (HPC, single-process CoGAPS runs later).
# - Builds frozen-preprocessing caches in $OUTDIR/cache/
# - Writes a jobs file $OUTDIR/jobs.tsv for the Slurm array

set -euo pipefail

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "================"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
# IMPORTANT: with 'set -u', conda activate.d scripts that reference $PYTHONPATH can crash
# if PYTHONPATH is not defined. Define it *before* conda activation.
export PYTHONPATH="${PYTHONPATH-}"

# Conda
source /app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate oshane-jlab

# Make PyCoGAPS importable (repo checkout lives here)
export PYTHONPATH="$HOME/src/pycogaps:${PYTHONPATH:-}"

# Threads (for preprocessing only)
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export NUMEXPR_NUM_THREADS="$NTHREADS"

# ------------------------------------------------------------------------------
# Paths (edit these if you want)
# ------------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$HOME/CS4}"

H5AD_PATH="kang_counts_25k.h5ad"
OUTDIR="results_cogaps_singleprocess_hpc"

# Grid definition for jobs.tsv (edit as needed)
K_GRID="7,9,11,13"
SEEDS="1,2,3,4,5"
ITERS="2000,10000,20000"

echo "[RUN] Building cache into: ${OUTDIR}"
echo "[RUN] Using input: ${H5AD_PATH}"
echo "[RUN] Job grid: K=${K_GRID} | seeds=${SEEDS} | iters=${ITERS}"

# ------------------------------------------------------------------------------
# 1) Build caches (frozen preprocessing + genes×cells CoGAPS input)
# ------------------------------------------------------------------------------
python cogaps_prep_cache.py \
  --input-h5ad "${H5AD_PATH}" \
  --outdir "${OUTDIR}" \
  --n-top-genes 3000 \
  --blas-threads "${NTHREADS}"

# ------------------------------------------------------------------------------
# 2) Create jobs.tsv for the array
# ------------------------------------------------------------------------------
python make_cogaps_jobs_tsv.py \
  --outdir "${OUTDIR}" \
  --k-grid "${K_GRID}" \
  --seeds "${SEEDS}" \
  --iters "${ITERS}"

echo "✅ Prep done."
echo "jobs.tsv:"
wc -l "${OUTDIR}/jobs.tsv" || true
