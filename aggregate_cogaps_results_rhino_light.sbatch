#!/bin/bash
#SBATCH --job-name=cogaps_aggregate
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Typical usage after the array job is submitted:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> aggregate_cogaps_results_rhino_light.sbatch

set -euo pipefail

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "================"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}"

source /app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate oshane-jlab

export PYTHONPATH="$HOME/src/pycogaps:${PYTHONPATH:-}"

# Aggregation is light; keep threads modest.
NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export NUMEXPR_NUM_THREADS="$NTHREADS"

cd "${SLURM_SUBMIT_DIR:-$HOME/CS4/slurm_sweep}"

OUTDIR="results_cogaps_singleprocess_hpc"
PREPROCESSED="${OUTDIR}/cache/preprocessed_cells_hvg3000.h5ad"

python cogaps_aggregate_results.py \
  --outdir "${OUTDIR}" \
  --preprocessed-h5ad "${PREPROCESSED}" \
  --no-umap

echo "✅ Aggregation complete."
ls -lh "${OUTDIR}/summary_by_K.csv" "${OUTDIR}/per_run_metrics.csv" "${OUTDIR}/report.md" 2>/dev/null || true
