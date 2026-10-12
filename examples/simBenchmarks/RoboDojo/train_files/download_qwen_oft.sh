#!/usr/bin/env bash
set -euo pipefail

# Download the pinned StarVLA QwenOFT RoboDojo run directory.
# The checkpoint is about 9.8 GB. Re-running this script resumes an
# interrupted download and verifies the exact release after completion.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

repo_id="StarVLA/Qwen3vl4b-OFT-RoboDojo"
revision="97123d6ef9f88b99801d07f3ad0720ca7c539777"
hf_endpoint="${STARVLA_DOWNLOAD_ENDPOINT:-https://huggingface.co}"
output_dir="${OUT_DIR:-${REPO_ROOT}/playground/Checkpoints/huggingface/oft}"

checkpoint_rel="checkpoints/steps_100000_pytorch_model.pt"
checkpoint_size=9785133528
checkpoint_sha256="62b8333c9af58467a911150e908cc726dd3c2660fd461871ae9300296391a3cd"
config_sha256="492d10ce090277760be03c39e1b68f31588a497f1d3cadc946e14312e2a63b10"
config_full_sha256="02bb182358a29a1d4beb25da59adef2655abcddbf8cfd1b4e805d827b8d71a68"
stats_sha256="88f104f05efac1c73be6bfccd0209fb74562ff3b65ff9a92f58cf3ea428aea0d"

declare -A expected_sizes=(
    ["config.yaml"]=1168
    ["config.full.yaml"]=2511
    ["dataset_statistics.json"]=5256
    ["README.md"]=6683
    ["${checkpoint_rel}"]="${checkpoint_size}"
)

if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required" >&2
    exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "sha256sum is required" >&2
    exit 1
fi

base_url="${hf_endpoint%/}/${repo_id}/resolve/${revision}"

download() {
    local relative_path="$1"
    local destination="${output_dir}/${relative_path}"
    local url="${base_url}/${relative_path}?download=true"
    mkdir -p "$(dirname "${destination}")"

    if [[ -f "${destination}" && "$(stat -c '%s' "${destination}")" == "${expected_sizes[${relative_path}]}" ]]; then
        echo "[download] ${relative_path}: already complete"
        return 0
    fi

    echo "[download] ${relative_path} (resumable)"
    # -C - resumes an existing partial file. --progress-bar remains visible
    # in normal terminals and in saved logs.
    curl --fail --location --show-error --progress-bar \
        --retry 8 --retry-delay 3 \
        -C - "${url}" -o "${destination}"
}

echo "[download] repo=${repo_id}"
echo "[download] revision=${revision}"
echo "[download] endpoint=${hf_endpoint}"
echo "[download] output=${output_dir}"

download "config.yaml"
download "config.full.yaml"
download "dataset_statistics.json"
download "README.md"
download "${checkpoint_rel}"

actual_size="$(stat -c '%s' "${output_dir}/${checkpoint_rel}")"
if [[ "${actual_size}" != "${checkpoint_size}" ]]; then
    echo "[verify][ERROR] checkpoint size: expected ${checkpoint_size}, got ${actual_size}" >&2
    exit 1
fi

declare -A expected_hashes=(
    ["config.yaml"]="${config_sha256}"
    ["config.full.yaml"]="${config_full_sha256}"
    ["dataset_statistics.json"]="${stats_sha256}"
    ["${checkpoint_rel}"]="${checkpoint_sha256}"
)

for relative_path in "${!expected_hashes[@]}"; do
    expected="${expected_hashes[${relative_path}]}"
    actual="$(sha256sum "${output_dir}/${relative_path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "[verify][ERROR] ${relative_path} SHA256 mismatch" >&2
        echo "[verify][ERROR] expected=${expected}" >&2
        echo "[verify][ERROR] actual=${actual}" >&2
        exit 1
    fi
    echo "[verify] ${relative_path}: SHA256 OK"
done

echo "[verify] QwenOFT RoboDojo checkpoint is complete: ${output_dir}"
