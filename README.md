# CoGAPS Slurm Sweep (Single-Process, Array-Based)

This repository runs a **CoGAPS parameter sweep on an HPC cluster** using Slurm arrays:

1. **Prep stage** (one job): preprocess once and cache stable inputs.
2. **Sweep stage** (array): run one `(K, seed, n_iter)` per task.
3. **Aggregate stage** (one job): collect run metrics, choose best `K`, and build summary outputs.

The workflow is optimized for cluster throughput while keeping each CoGAPS run single-process (no distributed CoGAPS mode).

---

## Repository layout

- `cogaps_prep_cache.py`  
  Builds cached preprocessing outputs and a CoGAPS-ready genes×cells matrix.
- `make_cogaps_jobs_tsv.py`  
  Generates headerless `jobs.tsv` used by Slurm array tasks.
- `cogaps_run_one_singleprocess.py`  
  Runs one CoGAPS configuration and writes result + metrics.
- `cogaps_aggregate_results.py`  
  Aggregates all runs, scores stability, and creates report artifacts.
- `prep_cogaps_cache_and_jobs_rhino_light.sbatch`  
  Slurm script for prep + jobs file generation.
- `cogaps_sweep_array_singleprocess_rhino_light.sbatch`  
  Slurm array script for the parameter sweep.
- `aggregate_cogaps_results_rhino_light.sbatch`  
  Slurm script for post-array aggregation.

---

## Prerequisites on cluster

## 1) Software environment

The provided `.sbatch` files assume:

- Miniforge conda init at:  
  `/app/software/Miniforge3/24.1.2-0/etc/profile.d/conda.sh`
- Conda env named: `oshane-jlab`
- Local PyCoGAPS source checkout available at:  
  `$HOME/src/pycogaps`

If your cluster differs, edit the environment block in each `.sbatch`.

## 2) Python packages

Required runtime packages include:

- `scanpy`
- `anndata`
- `numpy`
- `pandas`
- `matplotlib`
- `PyCoGAPS`

## 3) Input data

Provide an `.h5ad` input file (currently defaulted in script to `kang_counts_25k.h5ad`).

Your input AnnData should include:

- `obs['condition']` (or fallback `obs['label']` / `obs['stim']`)
- ideally `obs['cell_type']` (if missing, prep fills as `"unknown"`)

---

## End-to-end cluster run

Run from repo root (`/workspace/slurm_sweep` or your cluster checkout path).

## Step A — Prep cache and jobs table

Submit:

```bash
sbatch prep_cogaps_cache_and_jobs_rhino_light.sbatch
```

This does two things:

1. Runs `cogaps_prep_cache.py` to generate:
   - `results_cogaps_singleprocess_hpc/cache/preprocessed_cells_hvg3000.h5ad`
   - `results_cogaps_singleprocess_hpc/cache/cogaps_input_genesxcells_hvg3000_float64.h5ad`
   - `results_cogaps_singleprocess_hpc/cache/preprocess_config_hvg3000.json`
2. Runs `make_cogaps_jobs_tsv.py` to generate:
   - `results_cogaps_singleprocess_hpc/jobs.tsv`

### Important options (prep)

- `--raw-h5ad` (preferred) or legacy `--input-h5ad`
- `--n-top-genes 3000` controls HVG count and cache filename tag
- `--min-cells`, `--target-sum`, `--hvg-flavor`

If you rerun prep with different HVG size, ensure sweep/aggregate paths match the new cache filename.

---

## Step B — Submit sweep array

After prep finishes:

```bash
OUTDIR=results_cogaps_singleprocess_hpc
N=$(wc -l < ${OUTDIR}/jobs.tsv)
sbatch --array=0-$((N-1))%15 cogaps_sweep_array_singleprocess_rhino_light.sbatch
```

- `%15` is a concurrency cap (tune for your cluster allocation).
- Each task reads one line of `jobs.tsv` with tab-separated:
  - `K`
  - `seed`
  - `n_iter`

Each task writes:

- `runs/cogaps_K{K}_seed{seed}_iter{n_iter}.h5ad`
- `runs/cogaps_K{K}_seed{seed}_iter{n_iter}.metrics.json`
- `logs/run_K{K}_seed{seed}_iter{n_iter}.log`

### Sweep behavior details

- Single-process mode is preserved by **not setting CoGAPS distributed mode**.
- "Single-process" here means **one Python/CoGAPS process per Slurm array task**. It does **not** mean single-threaded execution inside that process.
- `--cogaps-threads` maps to `CoGAPS(..., nThreads=...)`; in the provided Slurm script this is set to `${SLURM_CPUS_PER_TASK}` (currently `4`). So each run is one process using up to 4 threads.
- The sweep fan-out (e.g., 60 runs) comes from the number of rows in `jobs.tsv` and array submission (`--array=0-$((N-1))%...`), so many single processes run concurrently across the cluster.
- BLAS/OpenMP threads are set via Slurm `cpus-per-task` and passed through env vars.
- Cache skip logic only skips when previous metrics have `status == "ok"`; failed runs are retried on rerun.
- IFN pattern scoring can use `cogaps_input.var['condition']`, with fallback to cached preprocessed cells if needed.

---

## Step C — Aggregate results

Once array job ID is known:

```bash
sbatch --dependency=afterok:<ARRAY_JOB_ID> aggregate_cogaps_results_rhino_light.sbatch
```

This runs `cogaps_aggregate_results.py` and generates:

- `results_cogaps_singleprocess_hpc/per_run_metrics.csv`
- `results_cogaps_singleprocess_hpc/summary_by_K.csv`
- `results_cogaps_singleprocess_hpc/report.md`
- figures under `results_cogaps_singleprocess_hpc/figures/`
- selected model under `results_cogaps_singleprocess_hpc/chosen_model/`

Aggregation arguments `--k-grid`, `--seeds`, and `--iters` are used as **coverage checks** (warnings if expected combinations are missing from metrics), while scoring/selection always uses discovered run metrics. `--top-genes` controls how many genes are written into the final report for the chosen run.

---

## Typical customization points

Edit these in sbatch scripts as needed:

- Partition, memory, walltime, CPUs.
- Grid values:
  - `K_GRID="7,9,11,13"`
  - `SEEDS="1,2,3,4,5"`
  - `ITERS="2000,10000,20000"`
- Input file path (`H5AD`) and output directory (`OUTDIR`).

If you change `OUTDIR`, keep it consistent across all three sbatch files.

---

## Failure handling and reruns

### Rerun failed array tasks only

Inspect metric statuses:

```bash
python - <<'PY'
from pathlib import Path
import json
p = Path('results_cogaps_singleprocess_hpc/runs')
for m in sorted(p.glob('*.metrics.json')):
    d = json.loads(m.read_text())
    if d.get('status') != 'ok':
        print(m.name, d.get('status'))
PY
```

Then resubmit targeted array indices (or full array; successful runs skip).

### Common problems

- Missing cache file in sweep script:
  - Re-run prep stage first.
- `condition` not found:
  - Ensure input `obs` has `condition`, `label`, or `stim`.
- PyCoGAPS import failure:
  - Check `PYTHONPATH="$HOME/src/pycogaps:${PYTHONPATH:-}"` in sbatch scripts.

---

## Minimal manual (non-sbatch) run examples

Prep:

```bash
python cogaps_prep_cache.py \
  --raw-h5ad kang_counts_25k.h5ad \
  --outdir results_cogaps_singleprocess_hpc \
  --n-top-genes 3000
```

Make jobs:

```bash
python make_cogaps_jobs_tsv.py \
  --k-grid 7,9,11,13 \
  --seeds 1,2,3,4,5 \
  --iters 2000,10000,20000 \
  --out results_cogaps_singleprocess_hpc/jobs.tsv
```

Run one job manually:

```bash
python cogaps_run_one_singleprocess.py \
  --cogaps-input-h5ad results_cogaps_singleprocess_hpc/cache/cogaps_input_genesxcells_hvg3000_float64.h5ad \
  --outdir results_cogaps_singleprocess_hpc \
  --k 7 --seed 1 --n-iter 2000 \
  --use-sparse-opt --blas-threads 1 --cogaps-threads 1
```

Aggregate:

```bash
python cogaps_aggregate_results.py \
  --outdir results_cogaps_singleprocess_hpc \
  --k-grid 7,9,11,13 \
  --seeds 1,2,3,4,5 \
  --iters 2000,10000,20000 \
  --top-genes 50 \
  --preprocessed-h5ad results_cogaps_singleprocess_hpc/cache/preprocessed_cells_hvg3000.h5ad \
  --no-umap
```

---

## Notes on version cleanup

This repo now keeps **one canonical version** of each pipeline stage script and sbatch launcher, with version-suffixed historical duplicates removed.


## Copy results from HPC to local (while sweep is still running)

Use the helper script from your **local machine** to pull `results_cogaps_singleprocess_hpc` from Rhino:

```bash
./sync_cogaps_results_from_hpc.sh \
  --host rhino03 \
  --user othomas \
  --remote-dir /home/othomas/CS4/slurm_sweep/results_cogaps_singleprocess_hpc \
  --local-dir ./results_cogaps_singleprocess_hpc
```

For an in-progress sweep (e.g., 55/60 complete), just run it repeatedly; it uses resumable `rsync` flags.

Useful options:

```bash
# Preview only
./sync_cogaps_results_from_hpc.sh --dry-run --user othomas

# If SSH needs custom options (port/key)
./sync_cogaps_results_from_hpc.sh --user othomas --ssh-opts "-p 22 -i ~/.ssh/id_ed25519"
```

---

