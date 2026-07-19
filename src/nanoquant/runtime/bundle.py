"""Self-contained packed runtime bundles and Transformers model-shell loading."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from nanoquant.runtime.artifact import MINIMUM_RUNTIME_VERSION, RuntimeModelMetadata
from nanoquant.runtime.backend import DeviceLike, RuntimeBackend, WorkloadSpec
from nanoquant.runtime.logical import canonical_torch_dtype
from nanoquant.runtime.packed_artifact import OpenPackedArtifact, open_packed_artifact
from nanoquant.runtime.planning import (
    PreparedExecutionPlans,
    plan_execution_workloads,
    prepare_execution_workloads,
)
from nanoquant.runtime.torch_model import (
    bind_fused_decode_rope,
    bind_grouped_decode_mlp,
    bind_native_bfloat16_tied_projection,
    bind_prepared_linears,
    bind_prepared_rms_norms,
    bind_short_sliding_masks,
    grouped_decode_qkv_count,
    transformers_decoder_module_paths,
)

RUNTIME_BUNDLE_SCHEMA_VERSION = 1
RUNTIME_BUNDLE_FORMAT = "nanoquant-runtime-bundle"
RUNTIME_BUNDLE_DESCRIPTOR = "nanoquant-runtime-bundle.json"
_MAXIMUM_DESCRIPTOR_BYTES = 16 * 1024 * 1024
_MODEL_ASSET_NAMES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
_REQUIRED_MODEL_ASSETS = ("config.json", "tokenizer.json", "tokenizer_config.json")
_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class RuntimeBundleError(ValueError):
    pass


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _member_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or candidate.is_absolute()
        or "\\" in value
        or any(part in ("", ".", "..") for part in candidate.parts)
        or candidate.as_posix() != value
    ):
        raise RuntimeBundleError(f"runtime bundle member path is unsafe: {value!r}")
    return value


def _digest(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeBundleError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeBundleMember:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _member_path(self.path)
        if self.bytes < 0:
            raise RuntimeBundleError("runtime bundle member bytes must be non-negative")
        _digest(self.sha256, f"member {self.path} hash")


@dataclass(frozen=True, slots=True)
class RuntimeShellTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str
    kind: str = "state"

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith(".") or ".." in self.name.split("."):
            raise RuntimeBundleError("runtime shell tensor name must be canonical")
        if any(dimension < 0 for dimension in self.shape):
            raise RuntimeBundleError("runtime shell tensor shape must be non-negative")
        if self.dtype not in _DTYPES:
            raise RuntimeBundleError(f"runtime shell tensor dtype is unsupported: {self.dtype}")
        _member_path(self.shard)
        if self.kind not in ("state", "buffer"):
            raise RuntimeBundleError(f"runtime shell tensor kind is unsupported: {self.kind}")


@dataclass(frozen=True, slots=True)
class RuntimeBundleManifest:
    schema_version: int
    artifact_format: str
    minimum_runtime_version: str
    model: RuntimeModelMetadata
    packed_path: str
    packed_descriptor_sha256: str
    members: tuple[RuntimeBundleMember, ...]
    shell_tensors: tuple[RuntimeShellTensor, ...]
    excluded_linear_modules: tuple[str, ...]
    total_member_bytes: int

    def __post_init__(self) -> None:
        if self.schema_version != RUNTIME_BUNDLE_SCHEMA_VERSION:
            raise RuntimeBundleError(f"unsupported runtime bundle schema: {self.schema_version}")
        if self.artifact_format != RUNTIME_BUNDLE_FORMAT:
            raise RuntimeBundleError(f"unsupported runtime bundle format: {self.artifact_format}")
        if self.minimum_runtime_version != MINIMUM_RUNTIME_VERSION:
            raise RuntimeBundleError(f"unsupported runtime bundle minimum version: {self.minimum_runtime_version}")
        _member_path(self.packed_path)
        _digest(self.packed_descriptor_sha256, "packed descriptor hash")
        paths = tuple(member.path for member in self.members)
        if len(paths) != len(set(paths)) or tuple(sorted(paths)) != paths:
            raise RuntimeBundleError("runtime bundle members must be unique and sorted")
        if sum(member.bytes for member in self.members) != self.total_member_bytes:
            raise RuntimeBundleError("runtime bundle member byte total is inconsistent")
        names = tuple(tensor.name for tensor in self.shell_tensors)
        if not names or len(names) != len(set(names)) or tuple(sorted(names)) != names:
            raise RuntimeBundleError("runtime shell tensor names must be non-empty, unique, and sorted")
        member_paths = set(paths)
        if any(tensor.shard not in member_paths for tensor in self.shell_tensors):
            raise RuntimeBundleError("runtime shell tensor refers to an undeclared shard")
        if (
            not self.excluded_linear_modules
            or len(self.excluded_linear_modules) != len(set(self.excluded_linear_modules))
            or tuple(sorted(self.excluded_linear_modules)) != self.excluded_linear_modules
        ):
            raise RuntimeBundleError("runtime bundle excluded linear modules must be non-empty, unique, and sorted")


@dataclass(frozen=True, slots=True)
class OpenRuntimeBundle:
    root: Path
    manifest: RuntimeBundleManifest
    packed: OpenPackedArtifact

    @property
    def model_assets(self) -> Path:
        return self.root / "model"

    def load_tokenizer(self) -> object:
        from transformers.models.auto.tokenization_auto import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_assets, local_files_only=True)


@dataclass(frozen=True, slots=True)
class LoadedTransformersRuntime:
    bundle: OpenRuntimeBundle
    model: nn.Module
    plans: PreparedExecutionPlans
    replaced_linear_count: int
    fused_rms_norm_count: int = 0
    fused_decode_rope_count: int = 0
    fused_decode_attention_count: int = 0
    grouped_decode_qkv_count: int = 0
    grouped_decode_mlp_count: int = 0
    short_sliding_mask_count: int = 0
    native_bfloat16_tied_projection_count: int = 0


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RuntimeBundleError(f"{path} must be an object with string keys")
    return cast(dict[str, object], value)


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise RuntimeBundleError(f"{path} must be an array")
    return cast(list[object], value)


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise RuntimeBundleError(f"{path} must be a string")
    return value


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeBundleError(f"{path} must be an integer")
    return value


def _model(value: object) -> RuntimeModelMetadata:
    payload = _mapping(value, "manifest.model")
    return RuntimeModelMetadata(
        _string(payload.get("source"), "manifest.model.source"),
        _string(payload.get("revision"), "manifest.model.revision"),
        _string(payload.get("family"), "manifest.model.family"),
        _string(payload.get("config_hash"), "manifest.model.config_hash"),
        _string(payload.get("tokenizer_hash"), "manifest.model.tokenizer_hash"),
    )


def _manifest(payload: object) -> RuntimeBundleManifest:
    value = _mapping(payload, "manifest")
    members = []
    for index, item in enumerate(_sequence(value.get("members"), "manifest.members")):
        member = _mapping(item, f"manifest.members[{index}]")
        members.append(
            RuntimeBundleMember(
                _string(member.get("path"), f"manifest.members[{index}].path"),
                _integer(member.get("bytes"), f"manifest.members[{index}].bytes"),
                _string(member.get("sha256"), f"manifest.members[{index}].sha256"),
            )
        )
    tensors = []
    for index, item in enumerate(_sequence(value.get("shell_tensors"), "manifest.shell_tensors")):
        tensor = _mapping(item, f"manifest.shell_tensors[{index}]")
        tensors.append(
            RuntimeShellTensor(
                _string(tensor.get("name"), f"manifest.shell_tensors[{index}].name"),
                tuple(
                    _integer(dimension, f"manifest.shell_tensors[{index}].shape")
                    for dimension in _sequence(tensor.get("shape"), f"manifest.shell_tensors[{index}].shape")
                ),
                _string(tensor.get("dtype"), f"manifest.shell_tensors[{index}].dtype"),
                _string(tensor.get("shard"), f"manifest.shell_tensors[{index}].shard"),
                _string(tensor.get("kind"), f"manifest.shell_tensors[{index}].kind"),
            )
        )
    return RuntimeBundleManifest(
        _integer(value.get("schema_version"), "manifest.schema_version"),
        _string(value.get("artifact_format"), "manifest.artifact_format"),
        _string(value.get("minimum_runtime_version"), "manifest.minimum_runtime_version"),
        _model(value.get("model")),
        _string(value.get("packed_path"), "manifest.packed_path"),
        _string(
            value.get("packed_descriptor_sha256"),
            "manifest.packed_descriptor_sha256",
        ),
        tuple(members),
        tuple(tensors),
        tuple(
            _string(item, f"manifest.excluded_linear_modules[{index}]")
            for index, item in enumerate(
                _sequence(
                    value.get("excluded_linear_modules"),
                    "manifest.excluded_linear_modules",
                )
            )
        ),
        _integer(value.get("total_member_bytes"), "manifest.total_member_bytes"),
    )


def open_runtime_bundle(
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> OpenRuntimeBundle:
    destination = Path(root)
    descriptor = destination / RUNTIME_BUNDLE_DESCRIPTOR
    if not descriptor.is_file() or descriptor.stat().st_size > _MAXIMUM_DESCRIPTOR_BYTES:
        raise RuntimeBundleError("runtime bundle descriptor is missing or too large")
    try:
        manifest = _manifest(json.loads(descriptor.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeBundleError("runtime bundle descriptor is invalid JSON") from error
    expected_paths = {member.path for member in manifest.members}
    actual_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path != descriptor
    }
    if actual_paths != expected_paths:
        raise RuntimeBundleError("runtime bundle file inventory differs from its manifest")
    for member in manifest.members:
        path = destination / member.path
        if path.stat().st_size != member.bytes:
            raise RuntimeBundleError(f"runtime bundle member size differs: {member.path}")
        if verify_hashes and _hash_file(path) != member.sha256:
            raise RuntimeBundleError(f"runtime bundle member hash differs: {member.path}")
    packed_descriptor = destination / manifest.packed_path / "nanoquant-packed-model.json"
    if _hash_file(packed_descriptor) != manifest.packed_descriptor_sha256:
        raise RuntimeBundleError("runtime bundle packed descriptor hash differs")
    packed = open_packed_artifact(destination / manifest.packed_path, verify_hashes=verify_hashes)
    if packed.manifest.model != manifest.model:
        raise RuntimeBundleError("runtime bundle and packed model identities differ")
    return OpenRuntimeBundle(destination.resolve(), manifest, packed)


def _checkpoint_files(source: Path) -> dict[Path, tuple[str, ...]]:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = _mapping(json.loads(index_path.read_text(encoding="utf-8")), "index")
            weights = _mapping(payload.get("weight_map"), "index.weight_map")
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeBundleError("source safetensors index is invalid") from error
        grouped: dict[Path, list[str]] = defaultdict(list)
        for name, shard in weights.items():
            grouped[source / _string(shard, f"index.weight_map.{name}")].append(name)
        return {path: tuple(sorted(names)) for path, names in sorted(grouped.items())}
    checkpoint = source / "model.safetensors"
    if not checkpoint.is_file():
        raise RuntimeBundleError("source model has no safetensors checkpoint")
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        return {checkpoint: tuple(handle.keys())}


def _module_paths(packed: OpenPackedArtifact) -> dict[str, str]:
    names = tuple(layer.spec.name for block in packed.manifest.blocks for layer in block.layers)
    return transformers_decoder_module_paths(names)


def _member(root: Path, path: Path) -> RuntimeBundleMember:
    relative = path.relative_to(root).as_posix()
    return RuntimeBundleMember(relative, path.stat().st_size, _hash_file(path))


def _derived_runtime_buffers(source: Path) -> dict[str, torch.Tensor]:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.auto.configuration_auto import AutoConfig

    config = AutoConfig.from_pretrained(source, local_files_only=True)
    if getattr(config, "model_type", None) != "gemma3_text":
        raise RuntimeBundleError(
            f"runtime buffer export does not support model type {getattr(config, 'model_type', None)!r}"
        )
    rope_scaling = getattr(config, "rope_scaling", None)
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type")) if isinstance(rope_scaling, dict) else "default"
    if not isinstance(rope_type, str) or rope_type not in ROPE_INIT_FUNCTIONS:
        raise RuntimeBundleError(f"runtime buffer export has unsupported RoPE type: {rope_type!r}")
    cpu = torch.device("cpu")
    global_frequency, _ = ROPE_INIT_FUNCTIONS[rope_type](config, device=cpu)
    local_config = copy.deepcopy(config)
    local_config.rope_theta = config.rope_local_base_freq
    local_config.rope_scaling = {"rope_type": "default"}
    local_frequency, _ = ROPE_INIT_FUNCTIONS["default"](local_config, device=cpu)
    return {
        "model.embed_tokens.embed_scale": torch.tensor(config.hidden_size**0.5),
        "model.rotary_emb.inv_freq": global_frequency.detach().cpu().contiguous(),
        "model.rotary_emb_local.inv_freq": local_frequency.detach().cpu().contiguous(),
    }


def write_runtime_bundle(
    output: str | Path,
    packed_artifact: str | Path,
    source_model: str | Path,
    *,
    replace: bool = False,
) -> OpenRuntimeBundle:
    """Create one atomic bundle containing packed weights, shell tensors, config, and tokenizer."""

    destination = Path(output)
    if destination.exists() and not replace:
        raise RuntimeBundleError(f"runtime bundle output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    packed = open_packed_artifact(packed_artifact, verify_hashes=True)
    source = Path(source_model)
    for required in _REQUIRED_MODEL_ASSETS:
        if not (source / required).is_file():
            raise RuntimeBundleError(f"source model asset is missing: {required}")
    module_paths = _module_paths(packed)
    member_paths: dict[str, str] = {}
    for block in packed.manifest.blocks:
        for layer in block.layers:
            if layer.spec.members:
                member_paths.update(
                    transformers_decoder_module_paths(tuple(member.name for member in layer.spec.members))
                )
            else:
                member_paths[layer.spec.name] = module_paths[layer.spec.name]
    excluded_modules = tuple(sorted(member_paths.values()))
    excluded_weights = {f"{path}.weight" for path in excluded_modules}
    excluded_biases = {
        f"{member_paths[layer.spec.name]}.bias"
        for block in packed.manifest.blocks
        for layer in block.layers
        if layer.spec.has_bias and not layer.spec.members
    }
    checkpoint_files = _checkpoint_files(source)
    source_keys = {name for names in checkpoint_files.values() for name in names}
    missing_weights = sorted(excluded_weights - source_keys)
    if missing_weights:
        raise RuntimeBundleError(f"source model is missing packed linear weights: {missing_weights[:3]}")
    unexpected_biases = sorted(excluded_biases - source_keys)
    if unexpected_biases:
        raise RuntimeBundleError(f"source model is missing packed linear biases: {unexpected_biases[:3]}")
    shell_names = source_keys - excluded_weights - excluded_biases
    if not shell_names:
        raise RuntimeBundleError("source model has no ordinary shell tensors")
    expected_linear_shapes: dict[str, tuple[int, ...]] = {}
    for block in packed.manifest.blocks:
        for layer in block.layers:
            if layer.spec.members:
                for member in layer.spec.members:
                    expected_linear_shapes[f"{member_paths[member.name]}.weight"] = (
                        member.row_end - member.row_start,
                        layer.spec.in_features,
                    )
            else:
                expected_linear_shapes[f"{module_paths[layer.spec.name]}.weight"] = (
                    layer.spec.out_features,
                    layer.spec.in_features,
                )
    expected_linear_shapes.update(
        {
            f"{member_paths[layer.spec.name]}.bias": (layer.spec.out_features,)
            for block in packed.manifest.blocks
            for layer in block.layers
            if layer.spec.has_bias and not layer.spec.members
        }
    )
    for source_shard, names in checkpoint_files.items():
        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            for name in names:
                expected_shape = expected_linear_shapes.get(name)
                if expected_shape is not None and tuple(handle.get_slice(name).get_shape()) != expected_shape:
                    raise RuntimeBundleError(f"source packed linear tensor shape differs: {name}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=str(destination.parent)))
    try:
        shutil.copytree(packed.root, temporary / "packed")
        model_root = temporary / "model"
        model_root.mkdir()
        for name in _MODEL_ASSET_NAMES:
            source_asset = source / name
            if source_asset.is_file():
                shutil.copy2(source_asset, model_root / name, follow_symlinks=True)

        shell_entries: list[RuntimeShellTensor] = []
        shard_number = 0
        for source_shard, names in checkpoint_files.items():
            selected = tuple(name for name in names if name in shell_names)
            if not selected:
                continue
            shard_number += 1
            shard_path = model_root / f"nanoquant-shell-{shard_number:05d}.safetensors"
            with safe_open(source_shard, framework="pt", device="cpu") as handle:
                values = {name: handle.get_tensor(name).detach().cpu().contiguous() for name in selected}
            save_file(values, shard_path, metadata={"format": "pt"})
            relative = shard_path.relative_to(temporary).as_posix()
            shell_entries.extend(
                RuntimeShellTensor(
                    name,
                    tuple(int(dimension) for dimension in value.shape),
                    canonical_torch_dtype(value.dtype),
                    relative,
                )
                for name, value in values.items()
            )
            del values
            gc.collect()

        buffer_values = _derived_runtime_buffers(source)
        buffer_path = model_root / "nanoquant-runtime-buffers.safetensors"
        save_file(buffer_values, buffer_path, metadata={"format": "pt"})
        buffer_relative = buffer_path.relative_to(temporary).as_posix()
        shell_entries.extend(
            RuntimeShellTensor(
                name,
                tuple(int(dimension) for dimension in value.shape),
                canonical_torch_dtype(value.dtype),
                buffer_relative,
                "buffer",
            )
            for name, value in buffer_values.items()
        )

        members = tuple(
            sorted(
                (_member(temporary, path) for path in temporary.rglob("*") if path.is_file()),
                key=lambda value: value.path,
            )
        )
        manifest = RuntimeBundleManifest(
            RUNTIME_BUNDLE_SCHEMA_VERSION,
            RUNTIME_BUNDLE_FORMAT,
            MINIMUM_RUNTIME_VERSION,
            packed.manifest.model,
            "packed",
            _hash_file(temporary / "packed" / "nanoquant-packed-model.json"),
            members,
            tuple(sorted(shell_entries, key=lambda value: value.name)),
            excluded_modules,
            sum(member.bytes for member in members),
        )
        descriptor = temporary / RUNTIME_BUNDLE_DESCRIPTOR
        descriptor.write_text(
            json.dumps(asdict(manifest), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        open_runtime_bundle(temporary, verify_hashes=True)
        if destination.exists():
            backup = destination.with_name(f".{destination.name}-previous")
            if backup.exists():
                raise RuntimeBundleError(f"runtime bundle replacement backup exists: {backup}")
            os.replace(destination, backup)
            try:
                os.replace(temporary, destination)
            except BaseException:
                os.replace(backup, destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return open_runtime_bundle(destination, verify_hashes=True)


def load_transformers_runtime(
    bundle: str | Path | OpenRuntimeBundle,
    backend: RuntimeBackend,
    *,
    device: DeviceLike,
    input_dtype: str,
    batch_size: int,
    prefill_tokens: int,
    fuse_rms_norm: bool = True,
    fuse_decode_rope: bool = True,
    fuse_decode_attention: bool = True,
    group_decode_qkv: bool = True,
    group_decode_mlp: bool = True,
    optimize_short_sliding_masks: bool = True,
    native_bfloat16_tied_projection: bool = True,
) -> LoadedTransformersRuntime:
    """Build a prepared model shell without loading source dense linear weights."""

    if input_dtype not in _DTYPES:
        raise RuntimeBundleError(f"runtime bundle input dtype is unsupported: {input_dtype}")
    opened = bundle if isinstance(bundle, OpenRuntimeBundle) else open_runtime_bundle(bundle, verify_hashes=True)
    target = torch.device(device)
    use_native_bfloat16_tied_projection = (
        native_bfloat16_tied_projection and target.type == "cuda" and input_dtype == "float32"
    )
    use_fused_decode_attention = fuse_decode_attention and target.type == "cuda" and input_dtype == "float32"
    use_group_decode_qkv = group_decode_qkv and target.type == "cuda" and input_dtype == "float32"
    use_group_decode_mlp = group_decode_mlp and target.type == "cuda" and input_dtype == "float32"
    entries = tuple(layer for block in opened.packed.manifest.blocks for layer in block.layers)
    specs = tuple(entry.spec for entry in entries)
    plans = plan_execution_workloads(
        specs,
        prefill=WorkloadSpec("prefill", target.type, input_dtype, batch_size, prefill_tokens, deterministic=True),
        decode=WorkloadSpec("decode", target.type, input_dtype, batch_size, 1, deterministic=True),
        prefill_backends=(backend,),
        decode_backends=(backend,),
        strict=True,
    )
    states = {entry.spec.name: opened.packed.load_layer(entry.spec.name) for entry in entries}
    prepared = prepare_execution_workloads(plans, states, (backend,), target)
    del states

    from transformers.models.auto.configuration_auto import AutoConfig
    from transformers.models.auto.modeling_auto import AutoModelForCausalLM

    config = AutoConfig.from_pretrained(opened.model_assets, local_files_only=True)
    dtype = _DTYPES[input_dtype]
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(  # type: ignore[no-untyped-call]
            config,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    replaced = bind_prepared_linears(
        model,
        prepared,
        transformers_decoder_module_paths(tuple(entry.spec.name for entry in entries)),
    )
    native_bfloat16_tied_projection_count = (
        bind_native_bfloat16_tied_projection(model) if use_native_bfloat16_tied_projection else 0
    )
    model.to_empty(device=target)
    model.tie_weights()
    state = model.state_dict()
    expected = {tensor.name for tensor in opened.manifest.shell_tensors if tensor.kind == "state"}
    expected_buffers = {tensor.name for tensor in opened.manifest.shell_tensors if tensor.kind == "buffer"}
    if not expected <= set(state):
        raise RuntimeBundleError(
            f"runtime shell tensors are absent from the model: {sorted(expected - set(state))[:3]}"
        )
    buffers = dict(model.named_buffers())
    if set(buffers) != expected_buffers:
        raise RuntimeBundleError(
            "runtime model derived buffer inventory differs: "
            f"missing={sorted(expected_buffers - set(buffers))[:3]}, "
            f"unexpected={sorted(set(buffers) - expected_buffers)[:3]}"
        )
    by_shard: dict[str, list[RuntimeShellTensor]] = defaultdict(list)
    for tensor in opened.manifest.shell_tensors:
        by_shard[tensor.shard].append(tensor)
    with torch.no_grad():
        for shard, tensors in sorted(by_shard.items()):
            with safe_open(opened.root / shard, framework="pt", device="cpu") as handle:
                for tensor in tensors:
                    value = handle.get_tensor(tensor.name)
                    target_value = state[tensor.name] if tensor.kind == "state" else buffers[tensor.name]
                    if tuple(value.shape) != tensor.shape or canonical_torch_dtype(value.dtype) != tensor.dtype:
                        raise RuntimeBundleError(f"runtime shell tensor metadata differs: {tensor.name}")
                    if tuple(target_value.shape) != tensor.shape:
                        raise RuntimeBundleError(f"runtime model shell tensor shape differs: {tensor.name}")
                    target_value.copy_(value.to(device=target, dtype=target_value.dtype))
    model.tie_weights()
    for name in expected_buffers:
        if not name.endswith(".inv_freq"):
            continue
        module_path, _, _attribute = name.rpartition(".")
        module = model.get_submodule(module_path)
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = module.inv_freq
    state = model.state_dict()
    loaded_pointers = {state[name].data_ptr() for name in expected}
    uninitialized = sorted(
        name for name, value in state.items() if name not in expected and value.data_ptr() not in loaded_pointers
    )
    if uninitialized:
        raise RuntimeBundleError(f"runtime model shell has uninitialized tensors: {uninitialized[:3]}")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeBundleError("runtime model retained meta parameters after shell loading")
    fused_rms_norm_count = bind_prepared_rms_norms(model) if fuse_rms_norm else 0
    grouped_mlp_count = bind_grouped_decode_mlp(model) if use_group_decode_mlp else 0
    fused_decode_rope_count = (
        bind_fused_decode_rope(
            model,
            fuse_decode_attention=use_fused_decode_attention,
            group_decode_qkv=use_group_decode_qkv,
        )
        if fuse_decode_rope
        else 0
    )
    fused_decode_attention_count = fused_decode_rope_count if use_fused_decode_attention else 0
    grouped_qkv_count = grouped_decode_qkv_count(model)
    short_sliding_mask_count = bind_short_sliding_masks(model) if optimize_short_sliding_masks else 0
    model.eval()
    return LoadedTransformersRuntime(
        opened,
        model,
        prepared,
        replaced,
        fused_rms_norm_count,
        fused_decode_rope_count,
        fused_decode_attention_count,
        grouped_qkv_count,
        grouped_mlp_count,
        short_sliding_mask_count,
        native_bfloat16_tied_projection_count,
    )
