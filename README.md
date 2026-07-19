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
selection, output locations, and troubleshooting. The default is the Gemma 3 1B architecture-protected Experiment
017; rerunning the same command resumes its durable commits and publishes the validated GGUF when complete:

```bash
export HF_TOKEN=<hugging-face-write-token>  # gated model access and final publication
bash tools/runpod_bootstrap.sh
```

Select the corresponding Gemma 3 4B workflow with `NANOQUANT_EXPERIMENT=018`. The script creates a persistent
virtual environment and Hugging Face cache, recreates and verifies the ignored pinned calibration artifact,
prefetches the offline quality datasets, installs the repository-vendored NanoQuant converter into a pinned upstream
llama.cpp conversion toolchain, builds its standard token-embedding quantizer, and launches the numbered experiment. Useful controls are:

```bash
NANOQUANT_SETUP_ONLY=1 bash tools/runpod_bootstrap.sh
NANOQUANT_EXPERIMENT=018 bash tools/runpod_bootstrap.sh
NANOQUANT_LLAMA_CPP_ROOT=/workspace/llama.cpp bash tools/runpod_bootstrap.sh
```

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
