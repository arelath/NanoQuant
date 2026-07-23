#!/usr/bin/env bash
# Bootstrap a persistent RunPod workspace and run one complete NanoQuant experiment.
# Defaults to the Llama 3.2 1B Instruct compression, quality, and publish experiment.
set -Eeuo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${NANOQUANT_WORKSPACE_ROOT:-/workspace}"
VENV_OVERRIDE="${NANOQUANT_VENV:-}"
VENV="${VENV_OVERRIDE:-${WORKSPACE_ROOT}/nanoquant-venv}"
SYSTEM_PYTHON="${NANOQUANT_SYSTEM_PYTHON:-python3}"
EXPERIMENT="${NANOQUANT_EXPERIMENT:-025}"
MINIMUM_TORCH_VERSION="2.6"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/huggingface}"
export NANOQUANT_LLAMA_CPP_ROOT="${NANOQUANT_LLAMA_CPP_ROOT:-${WORKSPACE_ROOT}/llama.cpp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${WORKSPACE_ROOT}/pip-cache}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${WORKSPACE_ROOT}/torchinductor-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORKSPACE_ROOT}/triton-cache}"
export CCE_AUTOTUNE="${CCE_AUTOTUNE:-0}"

LLAMA_CPP_REPOSITORY="${NANOQUANT_LLAMA_CPP_REPOSITORY:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_REVISION="${NANOQUANT_LLAMA_CPP_REVISION:-68a521b591edd2f36a456809230d63aa81003dfc}"
VENDORED_CONVERTER="${REPOSITORY_ROOT}/tools/llamacpp/convert_nanoquant_to_gguf.py"
VENDORED_CONVERTER_SHA256="c2e1fd064bbd46f38e9e3c5f739865d198ca75bd0bb9db16f72530d378d11304"
REQUIRES_HF_WRITE=0
PREFLIGHT_CCE=0

case "${EXPERIMENT}" in
  001)
    MODEL_ID="google/gemma-3-1b-it"
    MODEL_REVISION="dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    LAUNCHER="experiments/001-compress-gemma-3-1b-it.py"
    ;;
  003)
    MODEL_ID="google/gemma-3-4b-it"
    MODEL_REVISION="093f9f388b31de276ce2de164bdc2081324b9767"
    LAUNCHER="experiments/003-compress-and-benchmark-gemma-3-4b-it.py"
    ;;
  006)
    MODEL_ID="google/gemma-3-1b-it"
    MODEL_REVISION="dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    LAUNCHER="experiments/006-compress-and-benchmark-gemma-3-1b-it.py"
    ;;
  007)
    MODEL_ID="unsloth/gemma-3-270m-it"
    MODEL_REVISION="23cf460f6bb16954176b3ddcc8d4f250501458a9"
    LAUNCHER="experiments/007-compress-and-benchmark-gemma-3-270m-it.py"
    ;;
  008)
    MODEL_ID="unsloth/gemma-3-12b-it"
    MODEL_REVISION="9478e665381f42974aa06177b019352fb6291876"
    LAUNCHER="experiments/008-compress-and-benchmark-gemma-3-12b-it.py"
    ;;
  009)
    MODEL_ID="unsloth/gemma-3-270m-it"
    MODEL_REVISION="23cf460f6bb16954176b3ddcc8d4f250501458a9"
    LAUNCHER="experiments/009-compress-benchmark-and-publish-gemma-3-270m-it.py"
    REQUIRES_HF_WRITE=1
    ;;
  017)
    MODEL_ID="google/gemma-3-1b-it"
    MODEL_REVISION="dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    LAUNCHER="experiments/017-compress-and-benchmark-gemma-3-1b-it.py"
    REQUIRES_HF_WRITE=1
    PREFLIGHT_CCE=1
    ;;
  018)
    MODEL_ID="google/gemma-3-4b-it"
    MODEL_REVISION="093f9f388b31de276ce2de164bdc2081324b9767"
    LAUNCHER="experiments/018-compress-and-benchmark-gemma-3-4b-it.py"
    REQUIRES_HF_WRITE=1
    PREFLIGHT_CCE=1
    ;;
  025)
    MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
    MODEL_REVISION="9213176726f574b556790deb65791e0c5aa438b6"
    LAUNCHER="experiments/025-compress-and-benchmark-llama-3-2-1b-instruct.py"
    REQUIRES_HF_WRITE=1
    PREFLIGHT_CCE=1
    ;;
  *)
    echo "Unsupported NANOQUANT_EXPERIMENT=${EXPERIMENT}; choose 001, 003, 006, 007, 008, 009, 017, 018, or 025." >&2
    exit 2
    ;;
esac

if [[ ${REQUIRES_HF_WRITE} -eq 1 && -z "${HF_TOKEN:-}" ]]; then
  echo "Experiment ${EXPERIMENT} publishes its validated GGUF to Hugging Face." >&2
  echo "Set HF_TOKEN to a token with write permission before starting this long-running workflow." >&2
  exit 2
fi

mkdir -p \
  "${WORKSPACE_ROOT}" \
  "${HF_HOME}" \
  "${PIP_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${REPOSITORY_ROOT}/outputs/runpod-logs"
cd "${REPOSITORY_ROOT}"

if ! command -v cmake >/dev/null || ! command -v c++ >/dev/null; then
  if ! command -v apt-get >/dev/null; then
    echo "cmake and a C++ compiler are required, and apt-get is unavailable" >&2
    exit 1
  fi
  echo "==> Installing system build dependencies"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake git python3-venv
fi

echo "==> Repository: ${REPOSITORY_ROOT}"
echo "==> Experiment: ${EXPERIMENT} (${MODEL_ID}@${MODEL_REVISION})"
if [[ ${REQUIRES_HF_WRITE} -eq 1 ]]; then
  echo "==> Hugging Face publication enabled; repository access will be validated by the workflow before compression"
fi

if ! IMAGE_TORCH_VERSION="$("${SYSTEM_PYTHON}" -c 'import torch; print(torch.__version__)')"; then
  echo "The RunPod image must provide a CUDA-enabled PyTorch installation." >&2
  exit 1
fi
IMAGE_TORCH_CUDA="$("${SYSTEM_PYTHON}" -c 'import torch; print(torch.version.cuda or "none")')"
IMAGE_TORCH_BASE_VERSION="${IMAGE_TORCH_VERSION%%+*}"
if ! "${SYSTEM_PYTHON}" - "${IMAGE_TORCH_BASE_VERSION}" "${MINIMUM_TORCH_VERSION}" <<'PY'
import re
import sys


def major_minor(value: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise SystemExit(2)
    return int(match.group(1)), int(match.group(2))


installed, minimum = sys.argv[1:]
raise SystemExit(0 if major_minor(installed) >= major_minor(minimum) else 1)
PY
then
  echo "RunPod image PyTorch ${IMAGE_TORCH_VERSION} is unsupported; NanoQuant requires PyTorch >=${MINIMUM_TORCH_VERSION}." >&2
  echo "Select a newer CUDA-enabled PyTorch image and rerun the bootstrap." >&2
  echo "The bootstrap intentionally preserves the image's tested CUDA/PyTorch stack instead of replacing it with pip." >&2
  exit 1
fi
if ! "${SYSTEM_PYTHON}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "Image PyTorch ${IMAGE_TORCH_VERSION} (CUDA ${IMAGE_TORCH_CUDA}) cannot initialize CUDA." >&2
  echo "Check that the pod's NVIDIA driver supports the image's CUDA major version." >&2
  exit 1
fi

venv_uses_image_torch() {
  "${VENV}/bin/python" - "${IMAGE_TORCH_VERSION}" "${IMAGE_TORCH_CUDA}" <<'PY'
import sys
try:
    import torch
except ImportError:
    raise SystemExit(1)
expected_version, expected_cuda = sys.argv[1:]
raise SystemExit(
    0
    if torch.__version__ == expected_version and (torch.version.cuda or "none") == expected_cuda
    else 1
)
PY
}

if [[ -x "${VENV}/bin/python" ]] && ! venv_uses_image_torch; then
  if [[ -n "${VENV_OVERRIDE}" ]]; then
    echo "Existing NANOQUANT_VENV=${VENV} does not use image PyTorch ${IMAGE_TORCH_VERSION}." >&2
    echo "Choose a new empty NANOQUANT_VENV so the image installation can be reused." >&2
    exit 1
  fi
  VENV="${WORKSPACE_ROOT}/nanoquant-venv-torch-${IMAGE_TORCH_BASE_VERSION}-cu${IMAGE_TORCH_CUDA//./}"
  echo "==> Existing default environment has a different PyTorch; preserving it and using ${VENV}"
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${SYSTEM_PYTHON}" -m venv --system-site-packages "${VENV}"
fi
if ! venv_uses_image_torch; then
  echo "Virtual environment ${VENV} does not expose image PyTorch ${IMAGE_TORCH_VERSION}." >&2
  echo "Remove that environment or select a new NANOQUANT_VENV path." >&2
  exit 1
fi

echo "==> Image PyTorch: ${IMAGE_TORCH_VERSION} (CUDA ${IMAGE_TORCH_CUDA})"
echo "==> Persistent environment: ${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,evaluation]" "huggingface-hub[hf_transfer]>=0.30,<1" \
  --constraint <(printf 'torch==%s\n' "${IMAGE_TORCH_BASE_VERSION}")
if ! venv_uses_image_torch; then
  echo "Dependency installation replaced image PyTorch ${IMAGE_TORCH_VERSION}; refusing to continue." >&2
  exit 1
fi

if [[ "${NANOQUANT_RUN_TESTS:-1}" == "1" ]]; then
  echo "==> Running fast recipe preflight tests"
  tests=(tests/unit/test_base_compression_recipe.py)
  experiment_test="tests/unit/test_experiment${EXPERIMENT}.py"
  if [[ -f "${experiment_test}" ]]; then
    tests+=("${experiment_test}")
  fi
  python -m pytest -q "${tests[@]}"
fi

python - <<'PY'
import sys
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; select a CUDA RunPod image and GPU pod")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
PY

# Download the exact model snapshot now, so authentication or network errors happen
# before llama.cpp compilation or a long compression stage.
export NANOQUANT_BOOTSTRAP_MODEL_ID="${MODEL_ID}"
export NANOQUANT_BOOTSTRAP_MODEL_REVISION="${MODEL_REVISION}"
MODEL_SNAPSHOT="$(${VENV}/bin/python - <<'PY'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id=os.environ["NANOQUANT_BOOTSTRAP_MODEL_ID"],
    revision=os.environ["NANOQUANT_BOOTSTRAP_MODEL_REVISION"],
))
PY
)"
export NANOQUANT_BOOTSTRAP_MODEL_SNAPSHOT="${MODEL_SNAPSHOT}"
echo "==> Model snapshot: ${MODEL_SNAPSHOT}"

# Compile and execute the exact-size causal-loss kernels in a disposable,
# time-bounded process. A bad Torch/Triton/image combination must fail during
# setup instead of appearing to hang forever on calibration batch zero.
if [[ ${PREFLIGHT_CCE} -eq 1 && "${NANOQUANT_PREFLIGHT_CCE:-1}" == "1" ]]; then
  if ! command -v timeout >/dev/null; then
    echo "GNU timeout is required for the bounded cut-cross-entropy preflight." >&2
    exit 1
  fi
  CCE_TIMEOUT_SECONDS="${NANOQUANT_CCE_PREFLIGHT_TIMEOUT_SECONDS:-300}"
  echo "==> Preflighting cut-cross-entropy kernels (timeout: ${CCE_TIMEOUT_SECONDS}s)"
  set +e
  timeout --signal=TERM --kill-after=30s "${CCE_TIMEOUT_SECONDS}s" \
    "${VENV}/bin/python" - <<'PY'
import os
import time

import torch
from cut_cross_entropy import linear_cross_entropy
from transformers import AutoConfig

snapshot = os.environ["NANOQUANT_BOOTSTRAP_MODEL_SNAPSHOT"]
config = AutoConfig.from_pretrained(snapshot, local_files_only=False)
text_config = getattr(config, "text_config", config)
hidden_size = int(text_config.hidden_size)
vocab_size = int(text_config.vocab_size)
sequence_length = 2048

hidden = torch.randn(
    (1, sequence_length, hidden_size),
    device="cuda",
    dtype=torch.bfloat16,
    requires_grad=True,
)
weight = torch.randn((vocab_size, hidden_size), device="cuda", dtype=torch.bfloat16)
targets = torch.randint(0, vocab_size, (1, sequence_length), device="cuda")
torch.cuda.synchronize()
started = time.monotonic()
loss = linear_cross_entropy(hidden, weight, targets, shift=True, filter_eps=None)
loss.backward()
torch.cuda.synchronize()
print(
    "cut-cross-entropy preflight completed "
    f"for hidden={hidden_size}, vocab={vocab_size}, sequence={sequence_length} "
    f"in {time.monotonic() - started:.1f}s",
    flush=True,
)
PY
  cce_status=$?
  set -e
  if [[ ${cce_status} -eq 124 || ${cce_status} -eq 137 ]]; then
    echo "Cut-cross-entropy preflight exceeded ${CCE_TIMEOUT_SECONDS}s; refusing to start calibration." >&2
    echo "Use a compatible CUDA PyTorch image or inspect TorchInductor/Triton compiler processes and logs." >&2
    exit 1
  fi
  if [[ ${cce_status} -ne 0 ]]; then
    echo "Cut-cross-entropy preflight failed with status ${cce_status}; refusing to start calibration." >&2
    exit "${cce_status}"
  fi
fi

# Populate every pinned evaluation dataset during setup to avoid later network
# latency. Launchers may still download a missing pinned file when necessary.
if [[ "${NANOQUANT_PREFETCH_QUALITY:-1}" == "1" ]]; then
  echo "==> Caching pinned quality datasets"
  "${VENV}/bin/python" - <<'PY'
from datasets import load_dataset
from nanoquant.application.task_evaluation import pinned_legacy_multiple_choice_tasks
from nanoquant.quality_evaluation import WIKITEXT_CONFIG, WIKITEXT_DATASET, WIKITEXT_REVISION

load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, revision=WIKITEXT_REVISION, split="test")
for task in pinned_legacy_multiple_choice_tasks():
    load_dataset(
        task.dataset_name,
        task.dataset_config,
        revision=task.dataset_revision,
        split=task.split,
    )
    print(f"cached {task.task_name}")
PY
fi

if [[ ! -d "${NANOQUANT_LLAMA_CPP_ROOT}/.git" ]]; then
  echo "==> Fetching pinned upstream llama.cpp conversion toolchain"
  mkdir -p "${NANOQUANT_LLAMA_CPP_ROOT}"
  git -C "${NANOQUANT_LLAMA_CPP_ROOT}" init
  git -C "${NANOQUANT_LLAMA_CPP_ROOT}" remote add origin "${LLAMA_CPP_REPOSITORY}"
  git -C "${NANOQUANT_LLAMA_CPP_ROOT}" fetch --depth 1 origin "${LLAMA_CPP_REVISION}"
  git -C "${NANOQUANT_LLAMA_CPP_ROOT}" checkout --detach FETCH_HEAD
fi
if [[ "$(git -C "${NANOQUANT_LLAMA_CPP_ROOT}" rev-parse HEAD)" != "${LLAMA_CPP_REVISION}" ]]; then
  if [[ -n "$(git -C "${NANOQUANT_LLAMA_CPP_ROOT}" status --porcelain)" ]]; then
    echo "llama.cpp has local changes and is not at ${LLAMA_CPP_REVISION}; refusing to overwrite it" >&2
    exit 1
  fi
  git -C "${NANOQUANT_LLAMA_CPP_ROOT}" fetch origin "${LLAMA_CPP_REVISION}"
  git -C "${NANOQUANT_LLAMA_CPP_ROOT}" checkout --detach "${LLAMA_CPP_REVISION}"
fi

python - "${VENDORED_CONVERTER}" "${VENDORED_CONVERTER_SHA256}" <<'PY'
import hashlib
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"vendored NanoQuant converter hash differs: {actual} != {sys.argv[2]}")
PY
cp "${VENDORED_CONVERTER}" "${NANOQUANT_LLAMA_CPP_ROOT}/convert_nanoquant_to_gguf.py"

echo "==> Using repository-vendored NanoQuant GGUF converter"
python - <<'PY'
import sentencepiece
PY
python "${NANOQUANT_LLAMA_CPP_ROOT}/convert_nanoquant_to_gguf.py" --help >/dev/null
if [[ ! -x "${NANOQUANT_LLAMA_CPP_ROOT}/build/bin/llama-quantize" ]]; then
  echo "==> Building upstream llama.cpp token-embedding quantizer"
  cmake -S "${NANOQUANT_LLAMA_CPP_ROOT}" -B "${NANOQUANT_LLAMA_CPP_ROOT}/build" \
    -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "${NANOQUANT_LLAMA_CPP_ROOT}/build" --target llama-quantize \
    --config Release -j"$(nproc)"
fi

if [[ "${NANOQUANT_SETUP_ONLY:-0}" == "1" ]]; then
  echo "==> Setup complete (NANOQUANT_SETUP_ONLY=1); experiment was not launched"
  exit 0
fi

LOG="${REPOSITORY_ROOT}/outputs/runpod-logs/experiment-${EXPERIMENT}-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "==> Launching ${LAUNCHER}"
echo "==> Console log: ${LOG}"
set +e
"${VENV}/bin/python" "${LAUNCHER}" 2>&1 | tee "${LOG}"
status=${PIPESTATUS[0]}
set -e
if [[ ${status} -ne 0 ]]; then
  echo "Experiment exited with status ${status}. Run this script again to resume durable work." >&2
  exit "${status}"
fi
echo "==> Experiment ${EXPERIMENT} complete"
if [[ ${REQUIRES_HF_WRITE} -eq 1 ]]; then
  echo "==> Hugging Face upload complete; stopping RunPod pod ${RUNPOD_POD_ID}"
  runpodctl pod stop "${RUNPOD_POD_ID}"
fi
