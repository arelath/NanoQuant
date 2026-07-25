# NanoQuant Rewrite

This repository implements the typed, auditable NanoQuant pipeline specified in
[`Docs/`](Docs/README.md). The implementation is intentionally split into pure
domain code, application services, ports, infrastructure adapters, a standalone
runtime, configuration, and CLI surfaces.

Development setup:

```powershell
python -m pip install -e ".[dev]"
pytest
```

RunPod setup and complete compression can be bootstrapped from a fresh repository sync on a persistent
`/workspace` volume. See [`RUNPOD.md`](RUNPOD.md) for pod sizing, environment variables, secrets, experiment
selection, output locations, and troubleshooting. The default is Experiment 029 on
`Qwen/Qwen3-8B`; rerunning the same command resumes its durable commits and publishes the
validated GGUF when complete:

The RunPod image must provide CUDA-enabled PyTorch 2.6 or newer; the bootstrap preserves that image installation.

```bash
export HF_TOKEN=<hugging-face-write-token>  # gated model access and final publication
bash tools/runpod_bootstrap.sh
```

Select the Gemma 3 1B or 4B workflows with `NANOQUANT_EXPERIMENT=017` or `018`. The script creates a persistent
virtual environment and Hugging Face cache, recreates and verifies the ignored pinned calibration artifact,
prefetches the pinned quality datasets, installs the repository-vendored NanoQuant converter into the NanoQuant
llama.cpp fork, builds its CUDA runtime and quantizer, and launches the numbered experiment. Useful controls are:

```bash
NANOQUANT_SETUP_ONLY=1 bash tools/runpod_bootstrap.sh
NANOQUANT_EXPERIMENT=017 bash tools/runpod_bootstrap.sh
NANOQUANT_EXPERIMENT=018 bash tools/runpod_bootstrap.sh
NANOQUANT_LLAMA_CPP_ROOT=/workspace/llama.cpp bash tools/runpod_bootstrap.sh
```

The RunPod bootstrap syncs `https://github.com/arelath/llama.cpp.git` at the
`nanoquants` branch by default, records the resolved commit, builds its CUDA runtime and quantizer, and builds the
protocol-matched GGUF quality runner. Override `NANOQUANT_LLAMA_CPP_REPOSITORY` or
`NANOQUANT_LLAMA_CPP_REVISION` only when intentionally testing another fork identity.

Artifact cleanup is dry-run by default. The collector keeps artifacts referenced by evidence files and follows
artifact-to-artifact references transitively:

```powershell
.\.venv\Scripts\python.exe tools/cleanup_artifacts.py `
  --artifact-root evidence/m4/gemma-full-fisher-quantization/artifacts `
  --evidence-root evidence
```

Use `--ignore-evidence-path evidence/m4/<retired-run>` to make artifacts referenced only by a retired run eligible
without deleting that run's journal, logs, reports, or other evidence. Review with `--list-candidates`, then repeat
with `--apply` to delete. The default 24-hour minimum age protects recent/in-flight results; only reduce it after all
writers using that artifact store have stopped.
