#!/bin/bash
#SBATCH --job-name=cogaps_aggregate
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Aggregate results after the sweep finishes:
# - Reads per-run metrics in $OUTDIR/runs/*.metrics.json
# - Writes per_run_metrics.csv + summary_by_K.csv + report.md + figures/
#
# Typical usage:
#   OUTDIR=results_cogaps_singleprocess_hpc
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> aggregate_cogaps_results.sbatch

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
export PYTHONPATH="${PYTHONPATH-}"

source /app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate oshane-jlab

export PYTHONPATH="$HOME/src/pycogaps:${PYTHONPATH:-}"

NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export NUMEXPR_NUM_THREADS="$NTHREADS"

# ------------------------------------------------------------------------------
# Paths + grid (must match what you used to generate jobs.tsv)
# ------------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$HOME/CS4}"

OUTDIR="results_cogaps_singleprocess_hpc"
K_GRID="7,9,11,13"
SEEDS="1,2,3,4,5"
ITERS="2000,10000,20000"

python cogaps_aggregate_results.py \
  --outdir "${OUTDIR}" \
  --k-grid "${K_GRID}" \
  --seeds "${SEEDS}" \
  --iters "${ITERS}" \
  --top-genes 50 \
  --no-umap

echo "✅ Aggregation complete."
echo "Outputs:"
ls -lh "${OUTDIR}/summary_by_K.csv" "${OUTDIR}/per_run_metrics.csv" "${OUTDIR}/report.md" 2>/dev/null || true
