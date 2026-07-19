# Building and running NanoQuant on RunPod

This guide sets up a persistent RunPod environment and runs a complete, resumable NanoQuant compression experiment.
The repository includes [`tools/runpod_bootstrap.sh`](tools/runpod_bootstrap.sh), which performs the setup and launches
the selected experiment.

## 1. Recommended pod

For the default Gemma 3 1B compression or the optional Gemma 3 4B workflow:

- a recent Ubuntu PyTorch/CUDA image;
- Python 3.10 or newer;
- an NVIDIA GPU with at least 24 GiB VRAM recommended for the 1B workflow;
- at least 32 GiB system RAM;
- at least 100 GiB of persistent storage mounted at `/workspace`.

Use a persistent volume. The virtual environment, Hugging Face cache, llama.cpp export tools, model snapshots, and
resumable NanoQuant evidence are retained there. Do not run two compression workers on the same GPU.

## 2. Hugging Face access

Accept the model license on Hugging Face before launching gated Google Gemma models. The default Experiment 017 and
Experiment 018 publish their validated GGUF artifacts, so provide a token with write access through the
RunPod secret/environment-variable configuration or export it in the shell:

```bash
export HF_TOKEN="hf_your_write_token"
```

Never add the token to this repository, a launcher, or a log file. `huggingface_hub` reads `HF_TOKEN` automatically.
The bootstrap stores downloaded files under persistent `HF_HOME`; it does not write the token into NanoQuant run
artifacts.

## 3. Sync the repository and run

From a RunPod terminal:

```bash
cd /workspace
git clone <YOUR-NANOQUANT-REPOSITORY-URL> NanoQuantRewrite
cd NanoQuantRewrite

export HF_TOKEN="hf_your_write_token"
bash tools/runpod_bootstrap.sh
```

The default is Experiment 017 (`google/gemma-3-1b-it`), which runs the architecture-protected stacked-QKV rank
policy with sensitivity strength 0.5, evaluates quality, and uploads the validated GGUF. To run its Gemma 3 4B
variant instead:

```bash
cd /workspace/NanoQuantRewrite
export HF_TOKEN="hf_your_write_token"
NANOQUANT_EXPERIMENT=018 bash tools/runpod_bootstrap.sh
```

For a long run, start the command inside `tmux`:

```bash
tmux new -s nanoquant
NANOQUANT_EXPERIMENT=017 bash tools/runpod_bootstrap.sh
```

Detach with `Ctrl-B`, then `D`. Reconnect with:

```bash
tmux attach -t nanoquant
```

## 4. What the bootstrap script does

[`tools/runpod_bootstrap.sh`](tools/runpod_bootstrap.sh) performs these steps:

1. Installs missing system build prerequisites when `apt-get` is available.
2. Creates a persistent Python virtual environment.
3. Installs NanoQuant with development and evaluation dependencies.
4. Runs fast recipe preflight tests.
5. Verifies that PyTorch can access the selected CUDA GPU.
6. Downloads the exact pinned Hugging Face model snapshot.
7. Recreates and hash-validates the pinned 256 × 2,048 calibration artifact when it is absent.
8. Downloads the pinned quality datasets required by the launchers' offline evaluation mode.
9. Fetches a pinned upstream llama.cpp conversion toolchain.
10. Copies the repository-vendored NanoQuant GGUF converter into that toolchain and verifies its SHA-256.
11. Builds only the CPU `llama-quantize` target used to quantize `token_embd.weight` during export.
12. Launches the selected numbered experiment and records its console log.

The llama.cpp build uses `GGML_CUDA=OFF` intentionally. It is only an export utility build and does not affect CUDA
calibration, factorization, tuning, quality evaluation, NanoQuant runtime performance, or the final GGUF's ability to
run through a CUDA-enabled inference executable.

## 5. Environment variables

### Common variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | unset | Hugging Face token. Experiments 009, 017, and 018 require write permission for publication; gated snapshots require model access. |
| `NANOQUANT_EXPERIMENT` | `017` | Numbered experiment to launch. Supported values are `001`, `003`, `006`, `007`, `008`, `009`, `017`, and `018`. |
| `NANOQUANT_SETUP_ONLY` | `0` | Set to `1` to prepare everything without starting compression. |
| `NANOQUANT_RUN_TESTS` | `1` | Set to `0` to skip the fast recipe preflight tests. |
| `NANOQUANT_PREFETCH_QUALITY` | `1` | Set to `0` to skip evaluation-dataset downloads. Complete quality launchers may then fail unless those datasets are already cached. |
| `NANOQUANT_WORKSPACE_ROOT` | `/workspace` | Persistent RunPod storage root. |
| `NANOQUANT_VENV` | `/workspace/nanoquant-venv` | Persistent Python virtual environment. |
| `HF_HOME` | `/workspace/huggingface` | Persistent Hugging Face model and dataset cache. |
| `PIP_CACHE_DIR` | `/workspace/pip-cache` | Persistent pip download/build cache. |
| `NANOQUANT_LLAMA_CPP_ROOT` | `/workspace/llama.cpp` | Pinned upstream llama.cpp conversion-tool checkout and quantizer build. |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | PyTorch allocator policy used to reduce fragmentation. |

### Expert-only toolchain overrides

| Variable | Default | Purpose |
| --- | --- | --- |
| `NANOQUANT_LLAMA_CPP_REPOSITORY` | `https://github.com/ggml-org/llama.cpp.git` | Source of the upstream conversion support and standard quantizer. |
| `NANOQUANT_LLAMA_CPP_REVISION` | pinned in the script | Exact upstream revision compatible with the vendored converter. |

Do not override the llama.cpp revision for an ordinary run. Export validates the vendored converter hash, and an
arbitrary upstream revision may have incompatible Python conversion APIs or GGUF behavior.

Example setup-only command with explicit persistent paths:

```bash
export HF_TOKEN="hf_your_write_token"
export HF_HOME=/workspace/cache/huggingface
export NANOQUANT_VENV=/workspace/envs/nanoquant
export NANOQUANT_LLAMA_CPP_ROOT=/workspace/tools/llama.cpp

NANOQUANT_SETUP_ONLY=1 bash tools/runpod_bootstrap.sh
```

After setup succeeds, launch without rebuilding the environment:

```bash
NANOQUANT_EXPERIMENT=017 bash tools/runpod_bootstrap.sh
```

## 6. Experiment selection

| Number | Model/workflow |
| --- | --- |
| `001` | Gemma 3 1B compression, GGUF export, and benchmark |
| `003` | Gemma 3 4B compression, quality evaluation, and GGUF export |
| `006` | Gemma 3 1B complete compression and quality workflow |
| `007` | Gemma 3 270M complete compression and quality workflow; recommended first run |
| `008` | Gemma 3 12B CPU-offloaded large-model workflow |
| `009` | Gemma 3 270M complete compression, quality, and Hugging Face publication workflow |
| `017` | Gemma 3 1B architecture-protected stacked-QKV compression at sensitivity 0.5, quality, and publication; default |
| `018` | Gemma 3 4B variant of Experiment 017 with bounded-memory safeguards, quality, and publication |

The 4B and 12B workflows require substantially more host memory, storage, and runtime than the default setup check.
Confirm the selected pod's resources before launching them.

## 7. Resume and outputs

The numbered workflows write authoritative resumable state below `evidence/NNN`. Re-run the identical command after a
pod restart, SSH disconnect, or interrupted process. Do not delete or rewrite the evidence directory to force a clean
run.

For Experiment 017, the main locations are:

```text
evidence/017/   durable journal, commits, and resumable artifacts
outputs/017/    logical/packed artifacts, checkpoint, GGUF, summaries, and logs
Results/017/    publishable GGUF and reports
```

Bootstrap console logs are written to:

```text
outputs/runpod-logs/experiment-NNN-<UTC timestamp>.log
```

Before restarting after an unexpected disconnect, verify that the prior Python worker is no longer running and check
GPU ownership:

```bash
pgrep -af 'python.*experiments/' || true
nvidia-smi
```

If a worker is still healthy, reconnect to its terminal rather than launching a second worker.

## 8. Updating the repository

Stop or finish the active worker before updating source code:

```bash
cd /workspace/NanoQuantRewrite
git pull --ff-only
export HF_TOKEN="hf_your_write_token"
NANOQUANT_EXPERIMENT=017 bash tools/runpod_bootstrap.sh
```

The script reuses the virtual environment, caches, calibration artifact, upstream export toolchain, and completed
NanoQuant commits when their identities still match. Semantic recipe changes are not treated as a resume of old
numerical work.

## 9. Troubleshooting

### Gated model or HTTP 401/403

Confirm that the Hugging Face account has accepted the model license and that `HF_TOKEN` is present:

```bash
python - <<'PY'
import os
print("HF_TOKEN is set:", bool(os.environ.get("HF_TOKEN")))
PY
```

Do not print the token itself.

### CUDA unavailable

Use a CUDA-enabled RunPod image and confirm that the pod has an attached GPU:

```bash
nvidia-smi
/workspace/nanoquant-venv/bin/python -c 'import torch; print(torch.cuda.is_available())'
```

### Out of memory

Stop other GPU processes first. For tuning memory pressure, use a larger GPU rather than changing semantic recipe
settings in an existing numbered experiment. Experiment 008 already uses the CPU-offload large-model policy.

### Interrupted run

Run the same command again. The journal and committed block/layer artifacts are authoritative, and valid work will be
reused automatically.
