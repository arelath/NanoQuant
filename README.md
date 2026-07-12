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

