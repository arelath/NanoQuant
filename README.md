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
