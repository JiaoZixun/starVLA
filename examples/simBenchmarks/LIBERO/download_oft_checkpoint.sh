#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MODEL_ROOT="${MODEL_ROOT:-${STARVLA_DIR}/playground/Checkpoints/Qwen3-VL-OFT-LIBERO-4in1}"
CHECKPOINT="${MODEL_ROOT}/checkpoints/steps_50000_pytorch_model.pt"
EXPECTED_BYTES=9785061050
REVISION=39177146f142f8805d86fcc29d2a17c2228e6073
MAX_WORKERS="${MAX_WORKERS:-2}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Hugging Face CLI 'hf' is required in the active Python environment." >&2
  exit 1
fi

mkdir -p "${MODEL_ROOT}"
echo "Downloading StarVLA/Qwen3-VL-OFT-LIBERO-4in1 to ${MODEL_ROOT}"
echo "The checkpoint is about 9.8 GB. Re-running this command resumes the download."
hf download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  config.yaml dataset_statistics.json checkpoints/steps_50000_pytorch_model.pt \
  --revision "${REVISION}" --max-workers "${MAX_WORKERS}" \
  --format human --local-dir "${MODEL_ROOT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Download finished without the expected checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
ACTUAL_BYTES="$(stat -c '%s' "${CHECKPOINT}")"
if [[ "${ACTUAL_BYTES}" != "${EXPECTED_BYTES}" ]]; then
  echo "Checkpoint is incomplete: ${ACTUAL_BYTES}/${EXPECTED_BYTES} bytes" >&2
  exit 1
fi
echo "Checkpoint verified: ${CHECKPOINT} (${ACTUAL_BYTES} bytes)"
