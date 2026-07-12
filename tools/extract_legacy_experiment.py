"""Materialize the complete effective legacy dataclass config without running quantization."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("launcher", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    launcher = args.launcher.resolve()
    sys.path.insert(0, str(launcher.parent / "src"))
    specification = importlib.util.spec_from_file_location("legacy_experiment_capture", launcher)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {launcher}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    import torch
    from nanoquant.modules.hub import NanoQuantModel

    captured: dict[str, Any] = {}

    def fake_quantize(cls: type[Any], **kwargs: Any) -> object:
        captured.update(kwargs)
        return SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(model_type="capture-only")))

    original_quantize = NanoQuantModel.from_pretrained_quantize
    original_available = torch.cuda.is_available
    original_name = torch.cuda.get_device_name
    NanoQuantModel.from_pretrained_quantize = classmethod(fake_quantize)  # type: ignore[method-assign]
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_name = lambda *_: "capture-only"
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.main()
    finally:
        NanoQuantModel.from_pretrained_quantize = original_quantize  # type: ignore[method-assign]
        torch.cuda.is_available = original_available
        torch.cuda.get_device_name = original_name
    config = captured.pop("quant_config")
    payload = {
        "schema_version": 1,
        "launcher": launcher.name,
        "model_id_argument": captured.get("model_id"),
        "dtype_argument": str(captured.get("dtype")),
        "device_map_argument": captured.get("device_map"),
        "effective_legacy_config": asdict(config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
