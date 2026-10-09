#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${STARVLA_DIR}/playground/Datasets/LEROBOT_LIBERO_DATA}"
# Number of files downloaded concurrently inside each repository.  The four
# repositories are also started in parallel below, so the default is 32
# network workers in total.
MAX_WORKERS="${MAX_WORKERS:-8}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/.download_logs}"
DATASET_REPOS=(
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
)
DATASET_REVISIONS=(
  bf14d6258218d12c2e3c1a3b9922e163cdf6455d
  15657dac2ad1c01b4e94bf54ab0493b46a8d63f9
  222cf888ed360fad0a5f983748c1cc40743d43e7
  e1a223d30b896c1613f270a2bfc63d382b3de7e1
)

if [[ ! "${MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_WORKERS must be a positive integer." >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI 'hf' is required in the active Python environment." >&2
  exit 1
fi

mkdir -p "${DATA_ROOT}"
mkdir -p "${LOG_ROOT}"

# Prevent two copies of this accelerated script from writing the same local
# cache at once. Existing files and .incomplete files remain resumable.
exec 9>"${DATA_ROOT}/.download.lock"
if ! flock -n 9; then
  echo "Another LIBERO download is already running under ${DATA_ROOT}." >&2
  exit 2
fi

download_one() {
  local repo="$1"
  local revision="$2"
  local target="${DATA_ROOT}/${repo##*/}"
  echo "[$(date '+%F %T')] Downloading ${repo} to ${target} (workers=${MAX_WORKERS})"
  hf download "${repo}" --repo-type dataset --local-dir "${target}" \
    --revision "${revision}" --no-force-download \
    --max-workers "${MAX_WORKERS}" --format human
  mkdir -p "${target}/meta"
  if [[ ! -e "${target}/meta/modality.json" ]]; then
    cp -n "${STARVLA_DIR}/examples/simBenchmarks/LIBERO/train_files/modality.json" "${target}/meta/modality.json"
  fi
  echo "[$(date '+%F %T')] Finished ${repo}"
}

PIDS=()
for i in "${!DATASET_REPOS[@]}"; do
  repo="${DATASET_REPOS[$i]}"
  download_one "${repo}" "${DATASET_REVISIONS[$i]}" >>"${LOG_ROOT}/${repo##*/}.log" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" != 0 ]]; then
  echo "One or more downloads failed. Re-run this command; completed files are skipped and partial files resume." >&2
  exit "${status}"
fi
echo "Training datasets are under ${DATA_ROOT}"
