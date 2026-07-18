"""Run legacy Experiment 018 against the rewrite's pinned Gemma inputs.

The legacy implementation is intentionally executed in its own virtual environment and
process.  The worker owns the rewrite device lease for its entire lifetime, which keeps
detached runs safe without importing both packages named ``nanoquant`` into one process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import Any

from safetensors import safe_open

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
EXPECTED_CALIBRATION_SHAPE = (256, 2048)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _calibration_tensor_path(root: Path, artifact_id: str) -> tuple[Path, dict[str, Any]]:
    manifest_root = root / "artifacts" / artifact_id[7:9] / artifact_id
    manifest_path = manifest_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_artifact = str(manifest["tensor_artifact"])
    tensor_path = root / "artifacts" / tensor_artifact[7:9] / tensor_artifact / "tensors.safetensors"
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        input_ids = handle.get_tensor("input_ids")
        if tuple(input_ids.shape) != EXPECTED_CALIBRATION_SHAPE:
            raise ValueError(
                f"pinned calibration shape is {tuple(input_ids.shape)}; expected {EXPECTED_CALIBRATION_SHAPE}"
            )
        if str(input_ids.dtype) != "torch.int64":
            raise ValueError(f"pinned calibration dtype is {input_ids.dtype}; expected torch.int64")
    return tensor_path.resolve(), manifest


def _git_value(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(path)


def _detach(output: Path) -> int:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output.with_name(output.name + ".launcher.stdout.log")
    stderr_path = output.with_name(output.name + ".launcher.stderr.log")
    if stdout_path.exists() or stderr_path.exists():
        raise FileExistsError("refusing to overwrite existing detached launcher logs")
    command = [sys.executable, str(Path(__file__).resolve()), *(value for value in sys.argv[1:] if value != "--detach")]
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        options["start_new_session"] = True
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            **options,
        )
    print(
        json.dumps(
            {
                "process_id": process.pid,
                "output": str(output),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _patch_legacy_inputs(module: ModuleType, snapshot: Path, calibration_tensors: Path) -> None:
    import numpy as np
    import torch

    legacy_src = Path(module.SRC).resolve()
    sys.path.insert(0, str(legacy_src))
    from nanoquant.modules import hub  # type: ignore[import-not-found]
    from nanoquant.utils.utils import set_seed  # type: ignore[import-not-found]

    with safe_open(calibration_tensors, framework="pt", device="cpu") as handle:
        pinned_input_ids = handle.get_tensor("input_ids").long()
    if tuple(pinned_input_ids.shape) != EXPECTED_CALIBRATION_SHAPE:
        raise ValueError("worker received an incompatible pinned calibration tensor")

    original_config = hub.NanoQuantConfigDataclass

    def pinned_config(*args: Any, **kwargs: Any) -> Any:
        kwargs["calib_dataset"] = "pinned-direct-v1"
        kwargs["num_calib_samples"] = EXPECTED_CALIBRATION_SHAPE[0]
        kwargs["seqlen"] = EXPECTED_CALIBRATION_SHAPE[1]
        return original_config(*args, **kwargs)

    def pinned_prepare_dataset(_model_id: str, _quant_config: dict[str, Any]) -> object:
        return object()

    def pinned_loader(
        _dataset: object,
        _tokenizer: object,
        n_samples: int = 128,
        seed: int = 0,
        seqlen: int = 2048,
    ) -> torch.Tensor:
        if (n_samples, seqlen) != EXPECTED_CALIBRATION_SHAPE:
            raise ValueError(f"legacy requested calibration shape {(n_samples, seqlen)}")
        # Preserve the source loader's RNG boundary.  Its sampled indices are
        # deliberately ignored because the supplied tensor is already the final,
        # pinned dataloader shared with the rewrite.
        set_seed(seed)
        np.random.randint(0, n_samples, size=(n_samples,))
        print(f"Using pinned calibration dataloader with shape: {tuple(pinned_input_ids.shape)}")
        return pinned_input_ids.clone()

    hub.NanoQuantConfigDataclass = pinned_config
    hub.prepare_dataset = pinned_prepare_dataset
    hub.get_calib_loader = pinned_loader
    module.MODEL_ID = str(snapshot)


def _configure_legacy_outputs(module: ModuleType, output: Path) -> None:
    module.OUTPUT_DIR = output
    module.QMODEL_PATH = output / "legacy018-contemporary.pt"
    module.LOG_DIR = output
    module.LOG_PATH = output / "legacy018-contemporary.log"
    module.WEIGHT_ERROR_LOG_PATH = output / "weight-errors.csv"
    module.WEIGHT_ERROR_TABLE_PATH = output / "weight-errors.md"
    module.RANK_UTILITY_LOG_PATH = output / "rank-utility.csv"


def _worker(args: argparse.Namespace) -> int:
    started = time.time()
    output = args.output.resolve()
    result_path = output / "run-result.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_unix": started,
        "worker_pid": os.getpid(),
    }
    _atomic_json(result_path, status)
    try:
        lease_module = _load_module(
            "_nanoquant_rewrite_device_lease",
            args.rewrite_root / "src" / "nanoquant" / "infrastructure" / "device_lease.py",
        )
        with lease_module.acquire_device_lease(args.device):
            import torch

            torch.cuda.reset_peak_memory_stats()
            legacy = _load_module("_nanoquant_legacy_experiment018", args.legacy_launcher)
            _configure_legacy_outputs(legacy, output)
            receipt = json.loads((output / "calibration-input.json").read_text(encoding="utf-8"))
            calibration_tensors, _ = _calibration_tensor_path(output, str(receipt["artifact_id"]))
            _patch_legacy_inputs(legacy, args.snapshot, calibration_tensors)
            if args.probe:
                print(
                    json.dumps(
                        {
                            "cuda": torch.cuda.get_device_name(0),
                            "legacy_launcher": str(args.legacy_launcher),
                            "snapshot": str(args.snapshot),
                            "calibration_tensors": str(calibration_tensors),
                        },
                        sort_keys=True,
                    )
                )
            else:
                log_path = output / "legacy018-contemporary.log"
                with log_path.open("w", encoding="utf-8") as log_file:
                    tee_out = legacy.Tee(sys.stdout, log_file)
                    tee_err = legacy.Tee(sys.stderr, log_file)
                    with redirect_stdout(tee_out), redirect_stderr(tee_err):
                        legacy.main()
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        checkpoint = output / "legacy018-contemporary.pt"
        status.update(
            {
                "status": "probe-complete" if args.probe else "complete",
                "checkpoint": None
                if args.probe
                else {
                    "path": str(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": _sha256(checkpoint),
                },
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
            }
        )
        return_code = 0
    except BaseException as exc:
        status.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    finally:
        status["finished_unix"] = time.time()
        status["wall_seconds"] = status["finished_unix"] - started
        _atomic_json(result_path, status)
    return return_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rewrite-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--legacy-launcher", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker:
        required = (args.rewrite_root, args.legacy_launcher)
        if any(value is None for value in required):
            raise ValueError("worker arguments are incomplete")
        return _worker(args)
    if args.detach:
        return _detach(args.output)

    from nanoquant.infrastructure.hf_calibration_dataset import load_or_prepare_calibration

    rewrite_root = Path(__file__).resolve().parents[1]
    legacy_root = args.legacy_root.resolve()
    legacy_launcher = legacy_root / "018-compress-gemma-3-1b-it-phase1-no-hessian.py"
    legacy_python = legacy_root / ".venv" / "Scripts" / "python.exe"
    snapshot = args.snapshot.resolve()
    for path in (legacy_launcher, legacy_python, snapshot / "config.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty evidence directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    generated = load_or_prepare_calibration(snapshot, output)
    calibration_tensors, calibration_manifest = _calibration_tensor_path(
        output,
        generated.reference.artifact_id,
    )
    manifest = {
        "schema_version": 1,
        "protocol": "legacy018-contemporary-pinned-direct-v1",
        "legacy_root": str(legacy_root),
        "legacy_commit": _git_value(legacy_root, "rev-parse", "HEAD"),
        "legacy_status": _git_value(legacy_root, "status", "--short"),
        "legacy_launcher": str(legacy_launcher),
        "legacy_launcher_sha256": _sha256(legacy_launcher),
        "legacy_python": str(legacy_python),
        "rewrite_commit": _git_value(rewrite_root, "rev-parse", "HEAD"),
        "snapshot": str(snapshot),
        "model_revision": MODEL_REVISION,
        "calibration_artifact": generated.reference.artifact_id,
        "calibration_fingerprint": calibration_manifest["fingerprint"],
        "calibration_tensors": str(calibration_tensors),
        "calibration_loader_override": "pinned-direct-v1",
        "output": str(output),
    }
    _atomic_json(output / "run-manifest.json", manifest)

    command = [
        str(legacy_python),
        str(Path(__file__).resolve()),
        "--worker",
        "--legacy-root",
        str(legacy_root),
        "--snapshot",
        str(snapshot),
        "--output",
        str(output),
        "--device",
        args.device,
        "--rewrite-root",
        str(rewrite_root),
        "--legacy-launcher",
        str(legacy_launcher),
    ]
    if args.validate_only:
        command.append("--probe")
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    completed = subprocess.run(command, cwd=rewrite_root, env=environment, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
