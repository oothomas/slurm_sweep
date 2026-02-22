#!/bin/bash
#SBATCH --job-name=cogaps_sweep
#SBATCH --partition=campus-new
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err

# ------------------------------------------------------------------------------
# IMPORTANT:
# Submit this as an ARRAY job with a concurrency cap, e.g.:
#   N=$(( $(wc -l < results_cogaps_singleprocess_hpc/jobs.tsv) - 1 ))
#   sbatch --array=0-$((N-1))%5 cogaps_sweep_array_singleprocess_rhino_light.sbatch
#
# This keeps CoGAPS single-process but parallelizes the *grid* across the cluster.
# ------------------------------------------------------------------------------

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
# Fix for: 'PYTHONPATH: unbound variable' under `set -u`
export PYTHONPATH="${PYTHONPATH:-}"

source /app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh
conda activate oshane-jlab

export PYTHONPATH="$HOME/src/pycogaps:${PYTHONPATH:-}"

NTHREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$NTHREADS"
export OPENBLAS_NUM_THREADS="$NTHREADS"
export MKL_NUM_THREADS="$NTHREADS"
export NUMEXPR_NUM_THREADS="$NTHREADS"

cd "${SLURM_SUBMIT_DIR:-$HOME/CS4}"

OUTDIR="results_cogaps_singleprocess_hpc"
JOBS_TSV="${OUTDIR}/jobs.tsv"
COGAPS_INPUT="${OUTDIR}/cache/cogaps_input.h5ad"

if [[ ! -f "${JOBS_TSV}" ]]; then
  echo "ERROR: missing jobs file: ${JOBS_TSV}"
  echo "Run the prep job first."
  exit 1
fi
if [[ ! -f "${COGAPS_INPUT}" ]]; then
  echo "ERROR: missing cache file: ${COGAPS_INPUT}"
  echo "Run the prep job first."
  exit 1
fi

# jobs.tsv has a header row, so the first job is line 2
LINE_NO=$((SLURM_ARRAY_TASK_ID + 2))
LINE=$(sed -n "${LINE_NO}p" "${JOBS_TSV}" || true)

if [[ -z "${LINE}" ]]; then
  echo "No job found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} (line ${LINE_NO}). Exiting."
  exit 0
fi

IFS=$'\t' read -r K SEED NITER <<< "${LINE}"

echo "Running job: K=${K} seed=${SEED} n_iter=${NITER}"
echo "Cache: ${COGAPS_INPUT}"
echo "Outdir: ${OUTDIR}"

python cogaps_run_one_singleprocess.py \
  --cogaps-input "${COGAPS_INPUT}" \
  --outdir "${OUTDIR}" \
  --K "${K}" \
  --seed "${SEED}" \
  --n-iter "${NITER}" \
  --top-genes 50 \
  --use-sparse-opt \
  --cogaps-threads "${NTHREADS}"

echo "✅ Task complete: K=${K} seed=${SEED} n_iter=${NITER}"
