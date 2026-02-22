#!/bin/bash
#SBATCH --job-name=cogaps_sweep
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# Slurm array job: ONE task = ONE (K, seed, n_iter) CoGAPS run (single-process).
#
# Submit like this (after running the prep job):
#   OUTDIR=results_cogaps_singleprocess_hpc
#   N=$(wc -l < "${OUTDIR}/jobs.tsv")
#   sbatch --array=0-$((N-1))%20 cogaps_sweep_array_singleprocess.sbatch
#
# Notes:
# - cpus-per-task=1 keeps each CoGAPS run single-process and avoids oversubscription.
# - Increase cpus-per-task if you *know* CoGAPS/BLAS will use threads effectively.

set -euo pipefail

echo "=== Job info ==="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "================"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
# Prevent nounset failures during conda activation if activate.d scripts reference $PYTHONPATH
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
# Paths
# ------------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$HOME/CS4}"

OUTDIR="results_cogaps_singleprocess_hpc"
JOBFILE="${OUTDIR}/jobs.tsv"
PREP_DIR="${OUTDIR}/cache"

if [[ ! -f "${JOBFILE}" ]]; then
  echo "ERROR: jobs.tsv not found at ${JOBFILE}"
  echo "Run the prep job first."
  exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID}"
LINE="$(sed -n "$((TASK_ID + 1))p" "${JOBFILE}")"

if [[ -z "${LINE}" ]]; then
  echo "ERROR: No line for task ${TASK_ID} in ${JOBFILE}"
  exit 1
fi

K="$(echo "${LINE}" | awk '{print $1}')"
SEED="$(echo "${LINE}" | awk '{print $2}')"
NITER="$(echo "${LINE}" | awk '{print $3}')"

echo "[TASK] K=${K} seed=${SEED} n_iter=${NITER}"

python cogaps_run_one_singleprocess.py \
  --prep-dir "${PREP_DIR}" \
  --outdir "${OUTDIR}" \
  --k "${K}" \
  --seed "${SEED}" \
  --n-iter "${NITER}" \
  --top-genes 50 \
  --use-sparse-opt \
  --blas-threads "${NTHREADS}" \
  --cogaps-threads "${NTHREADS}"

echo "✅ Task complete: K=${K} seed=${SEED} n_iter=${NITER}"
