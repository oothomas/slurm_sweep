#!/usr/bin/env bash
# Pull CoGAPS sweep results from HPC to local machine.
#
# Designed for in-progress sweeps (e.g., 55/60 done): run repeatedly to sync new files.
# Uses rsync with partial/resume flags so large files can continue on retries.
#
# Examples:
#   ./sync_cogaps_results_from_hpc.sh
#   ./sync_cogaps_results_from_hpc.sh --host rhino03 --user othomas \
#     --remote-dir /home/othomas/CS4/slurm_sweep/results_cogaps_singleprocess_hpc \
#     --local-dir ./results_cogaps_singleprocess_hpc
#   ./sync_cogaps_results_from_hpc.sh --dry-run
#   ./sync_cogaps_results_from_hpc.sh --delete

set -euo pipefail

HOST="rhino03"
USER_NAME="${USER:-}"
REMOTE_DIR=""
LOCAL_DIR="./results_cogaps_singleprocess_hpc"
SSH_OPTS=""
DRY_RUN=0
DELETE=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --host <hostname>         HPC host (default: rhino03)
  --user <username>         HPC username (default: local \$USER)
  --remote-dir <path>       Remote results directory
                            (default: /home/<user>/CS4/slurm_sweep/results_cogaps_singleprocess_hpc)
  --local-dir <path>        Local destination directory (default: ./results_cogaps_singleprocess_hpc)
  --ssh-opts <opts>         Extra SSH options (quoted), e.g. "-p 2222 -i ~/.ssh/id_ed25519"
  --dry-run                 Show what would copy, but do not transfer
  --delete                  Delete local files missing on remote (dangerous; off by default)
  -h, --help                Show help

Notes:
  - Run this from your local machine (not on rhino03).
  - Safe to re-run while jobs are still completing; rsync resumes partial files.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?missing value for --host}"; shift 2 ;;
    --user)
      USER_NAME="${2:?missing value for --user}"; shift 2 ;;
    --remote-dir)
      REMOTE_DIR="${2:?missing value for --remote-dir}"; shift 2 ;;
    --local-dir)
      LOCAL_DIR="${2:?missing value for --local-dir}"; shift 2 ;;
    --ssh-opts)
      SSH_OPTS="${2:?missing value for --ssh-opts}"; shift 2 ;;
    --dry-run)
      DRY_RUN=1; shift ;;
    --delete)
      DELETE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ -z "${USER_NAME}" ]]; then
  echo "ERROR: username is empty. Pass --user <username> or set USER." >&2
  exit 2
fi

# Resolve default remote path *after* parsing args so --user updates it.
if [[ -z "${REMOTE_DIR}" ]]; then
  REMOTE_DIR="/home/${USER_NAME}/CS4/slurm_sweep/results_cogaps_singleprocess_hpc"
fi

mkdir -p "${LOCAL_DIR}"

RSYNC_ARGS=(
  -avh
  --progress
  --partial
  --prune-empty-dirs
)

# macOS/BSD rsync (2.6.9) does not support --append-verify.
# Prefer --append-verify when available; otherwise fall back to --append.
if rsync --help 2>&1 | grep -q -- '--append-verify'; then
  RSYNC_ARGS+=(--append-verify)
else
  RSYNC_ARGS+=(--append)
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  RSYNC_ARGS+=(--dry-run)
fi
if [[ ${DELETE} -eq 1 ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ -n "${SSH_OPTS}" ]]; then
  RSYNC_ARGS+=(-e "ssh ${SSH_OPTS}")
fi

SRC="${USER_NAME}@${HOST}:${REMOTE_DIR%/}/"
DST="${LOCAL_DIR%/}/"

echo "Syncing from: ${SRC}"
echo "Syncing to:   ${DST}"

rsync "${RSYNC_ARGS[@]}" "${SRC}" "${DST}"

echo "✅ Sync complete."
