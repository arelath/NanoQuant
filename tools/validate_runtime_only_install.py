"""Install the deployment-only wheel in isolation and generate from a self-contained bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from nanoquant.infrastructure.device_lease import wait_for_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json

_CHILD = r"""
import hashlib
import json
import sys
from pathlib import Path

site = Path(sys.argv[1]).resolve()
bundle_path = Path(sys.argv[2]).resolve()
device = sys.argv[3]
prompt = sys.argv[4]
max_new_tokens = int(sys.argv[5])
reference_path = None if sys.argv[6] == "-" else Path(sys.argv[6]).resolve()
sys.path.insert(0, str(site))

import nanoquant
import torch
from nanoquant.runtime import (
    CudaPackedBackend,
    GenerationRequest,
    SamplingConfig,
    TransformersGenerationModel,
    batch_prompts,
    generate,
    hybrid_cache_factory,
    load_transformers_runtime,
    open_runtime_bundle,
)

package_file = Path(nanoquant.__file__).resolve()
if not package_file.is_relative_to(site):
    raise RuntimeError(f"nanoquant resolved outside isolated target: {package_file}")
forbidden = (
    "nanoquant.application",
    "nanoquant.config",
    "nanoquant.domain",
    "nanoquant.infrastructure",
    "nanoquant.ports",
)
loaded_forbidden = sorted(name for name in sys.modules if name.startswith(forbidden))
if loaded_forbidden:
    raise RuntimeError(f"runtime-only import loaded research packages: {loaded_forbidden}")

bundle = open_runtime_bundle(bundle_path, verify_hashes=True)
tokenizer = bundle.load_tokenizer()
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True,
    add_generation_prompt=True,
)
pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    raise RuntimeError("runtime tokenizer has no pad token")
input_ids, attention_mask = batch_prompts((prompt_ids,), pad_token_id=pad_token_id)
target = torch.device(device)
if target.index is None:
    target = torch.device("cuda", torch.cuda.current_device())

with torch.inference_mode(), torch.cuda.device(target):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    loaded = load_transformers_runtime(
        bundle,
        CudaPackedBackend(),
        device=target,
        input_dtype="float32",
        batch_size=1,
        prefill_tokens=input_ids.shape[1],
    )
    request = GenerationRequest(
        input_ids.to(target),
        attention_mask.to(target),
        max_new_tokens,
        (int(loaded.model.config.vocab_size),),
        pad_token_id,
        sampling=SamplingConfig(mode="greedy"),
        stopping_check_interval=8,
    )
    created_caches = []
    cache_factory = hybrid_cache_factory(loaded.model.config)
    def capture_cache(*args):
        cache = cache_factory(*args)
        created_caches.append(cache)
        return cache
    shell = TransformersGenerationModel(loaded.model, capture_cache)
    result = generate(request, shell)
    torch.cuda.synchronize(target)
    allocated = torch.cuda.memory_allocated(target)
    peak = torch.cuda.max_memory_allocated(target)

tokens = result.token_ids[0, : result.lengths[0]].detach().cpu().tolist()
text = tokenizer.decode(tokens)
reference = None
if reference_path is not None:
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    count = payload["generated_token_count"]
    expected = payload["generated"]
    actual = tokenizer.decode(tokens[:count])
    if actual != expected:
        raise RuntimeError(f"clean-install reference prefix differs: {actual!r} != {expected!r}")
    reference = {"path": str(reference_path), "token_count": count, "text": expected, "exact": True}

digest = hashlib.sha256()
for token in tokens:
    digest.update(int(token).to_bytes(8, "little"))
print(json.dumps({
    "nanoquant_package_member": package_file.relative_to(site).as_posix(),
    "forbidden_modules_loaded": loaded_forbidden,
    "bundle": str(bundle.root),
    "packed_layer_count": bundle.packed.manifest.layer_count,
    "shell_tensor_count": len(bundle.manifest.shell_tensors),
    "replaced_linear_count": loaded.replaced_linear_count,
    "fused_rms_norm_count": loaded.fused_rms_norm_count,
    "fused_decode_rope_count": loaded.fused_decode_rope_count,
    "short_sliding_mask_count": loaded.short_sliding_mask_count,
    "fast_sliding_update_count": sum(
        getattr(cache, "nanoquant_fast_sliding_update_count", 0) for cache in created_caches
    ),
    "prefill_fallback_count": loaded.plans.prefill.plan.fallback_count,
    "decode_fallback_count": loaded.plans.decode.plan.fallback_count,
    "prompt_tokens": len(prompt_ids),
    "generated_tokens": len(tokens),
    "generated_token_ids": tokens,
    "generated_token_sha256": digest.hexdigest(),
    "generated_text": text,
    "reference_prefix": reference,
    "allocated_bytes": allocated,
    "peak_allocated_bytes": peak,
    "device": torch.cuda.get_device_name(target),
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "passed": True,
}, sort_keys=True))
"""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _wheel_members(path: Path) -> tuple[str, ...]:
    with ZipFile(path) as archive:
        members = tuple(sorted(archive.namelist()))
    forbidden = tuple(
        name
        for name in members
        if name.startswith("nanoquant/")
        and not (name == "nanoquant/__init__.py" or name.startswith("nanoquant/runtime/"))
    )
    if forbidden:
        raise ValueError(f"runtime wheel contains research package members: {forbidden}")
    if "nanoquant/__init__.py" not in members or not any(
        name.startswith("nanoquant/runtime/") for name in members
    ):
        raise ValueError("runtime wheel is missing its deployment package surface")
    return members


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="Write a short paragraph about quantization.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--triton-cache", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path(".tmp"))
    parser.add_argument("--wait-for-device-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("max_new_tokens must be positive")
    wheel = args.wheel.resolve()
    bundle = args.bundle.resolve()
    members = _wheel_members(wheel)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.triton_cache.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="nanoquant-runtime-install-", dir=args.work_root
    ) as temporary_name:
        temporary = Path(temporary_name).resolve()
        site = temporary / "site"
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-compile",
                "--target",
                str(site),
                str(wheel),
            ],
            cwd=temporary,
            text=True,
            capture_output=True,
            check=False,
        )
        if install.returncode != 0:
            raise RuntimeError(f"runtime-only wheel installation failed:\n{install.stderr}")
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["TRITON_CACHE_DIR"] = str(args.triton_cache.resolve())
        reference = "-" if args.reference_output is None else str(args.reference_output.resolve())
        with wait_for_device_lease(args.device, args.wait_for_device_seconds):
            child = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _CHILD,
                    str(site),
                    str(bundle),
                    args.device,
                    args.prompt,
                    str(args.max_new_tokens),
                    reference,
                ],
                cwd=temporary,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
                timeout=600,
            )
        if child.returncode != 0:
            raise RuntimeError(
                "runtime-only installed generation failed:\n"
                f"stdout:\n{child.stdout}\nstderr:\n{child.stderr}"
            )
        child_result = json.loads(child.stdout)
        result = {
            "schema_version": 1,
            "wheel": str(wheel),
            "wheel_bytes": wheel.stat().st_size,
            "wheel_sha256": _hash_file(wheel),
            "wheel_member_count": len(members),
            "wheel_members": list(members),
            "installation_mode": "isolated-target-with-host-runtime-dependencies",
            "research_package_members": [],
            "child": child_result,
            "passed": True,
        }
    if args.output is not None:
        atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
