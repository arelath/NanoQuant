#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / "gguf-py"))
import gguf

from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
)
from convert_hf_to_gguf import split_str_to_n_bytes


SUPPORTED_GGUF_WEIGHT_SUFFIXES = (
    ".attn_q.weight",
    ".attn_k.weight",
    ".attn_v.weight",
    ".attn_output.weight",
    ".ffn_gate.weight",
    ".ffn_up.weight",
    ".ffn_down.weight",
)


@dataclass
class NanoQuantSidecar:
    v_packed: np.ndarray
    u_packed: np.ndarray
    scale_pre: np.ndarray
    scale_mid: np.ndarray
    scale_post: np.ndarray
    salient_idx: np.ndarray | None = None
    salient_weight: np.ndarray | None = None
    salient_scale: np.ndarray | None = None


@dataclass
class Int8Embedding:
    weight: np.ndarray
    scale: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Hugging Face model plus NanoQuant checkpoint to GGUF")
    parser.add_argument("model", type=Path, help="directory containing the base Hugging Face model")
    parser.add_argument("--nanoquant-checkpoint", type=Path, required=True, help="NanoQuant checkpoint file or directory")
    parser.add_argument("--outfile", type=Path, help="path to write to; default: based on input")
    parser.add_argument(
        "--outtype",
        choices=["f32", "f16", "bf16", "auto"],
        default="auto",
        help="floating point type for non-NanoQuant tensors",
    )
    parser.add_argument("--bigendian", action="store_true", help="write a big endian GGUF")
    parser.add_argument("--use-temp-file", action="store_true", help="use a temporary file for tensor data")
    parser.add_argument("--no-lazy", action="store_true", help="materialize tensors eagerly")
    parser.add_argument("--model-name", type=str, default=None, help="override the model name metadata")
    parser.add_argument("--metadata", type=Path, help="authorship metadata override file")
    parser.add_argument("--split-max-tensors", type=int, default=0, help="max tensors in each split")
    parser.add_argument("--split-max-size", type=str, default="0", help="max size per split N(M|G)")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    return parser.parse_args()


def checkpoint_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"NanoQuant checkpoint path does not exist: {path}")

    patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(path.glob(pattern)))
    if not files:
        raise FileNotFoundError(f"No checkpoint files found in {path}")
    return files


def unwrap_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            value = obj.get(key)
            if isinstance(value, dict):
                try:
                    return unwrap_state_dict(value)
                except TypeError:
                    pass
    raise TypeError("checkpoint does not contain a tensor state dict")


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for file in checkpoint_files(path):
        logger.info(f"Loading NanoQuant checkpoint shard: {file}")
        if file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError("safetensors is required to read NanoQuant .safetensors checkpoints") from exc
            shard = load_file(file, device="cpu")
        else:
            try:
                shard_obj = torch.load(file, map_location="cpu", weights_only=True)
            except TypeError:
                shard_obj = torch.load(file, map_location="cpu")
            shard = unwrap_state_dict(shard_obj)
        tensors.update(shard)
    return tensors


def tensor_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return tuple(int(v) for v in value.reshape(-1).cpu().tolist())
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return None


def scale_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().reshape(-1).to(torch.float32).numpy().astype(np.float32, copy=False)


def index_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().reshape(-1).to(torch.int32).numpy().astype(np.int32, copy=False)


def salient_weight_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    cpu = tensor.detach().cpu()
    if cpu.dtype == torch.int8:
        return cpu.numpy().astype(np.int8, copy=False)
    return cpu.to(torch.float16).numpy().astype(np.float16, copy=False)


def int8_weight_to_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    cpu = tensor.detach().cpu()
    if cpu.dtype != torch.int8:
        raise ValueError(f"{name}: expected int8 embedding weight, got {cpu.dtype}")
    if cpu.ndim != 2:
        raise ValueError(f"{name}: expected a 2D embedding weight, got shape {tuple(cpu.shape)}")
    return cpu.numpy().astype(np.int8, copy=False)


def pack_q8_0_from_rowwise_int8(weight: np.ndarray, scale: np.ndarray, name: str) -> np.ndarray:
    if weight.ndim != 2:
        raise ValueError(f"{name}: expected a 2D embedding weight, got shape {weight.shape}")
    if scale.ndim != 1:
        raise ValueError(f"{name}: expected a 1D embedding scale, got shape {scale.shape}")
    if scale.size != weight.shape[0]:
        raise ValueError(f"{name}: scale length {scale.size} does not match row count {weight.shape[0]}")
    if weight.shape[1] % 32 != 0:
        raise ValueError(f"{name}: Q8_0 requires embedding width to be a multiple of 32, got {weight.shape[1]}")

    n_rows, n_cols = weight.shape
    n_blocks = n_cols//32

    d = scale.astype(np.float16, copy=False).reshape(n_rows, 1).view(np.uint8).reshape(n_rows, 1, 2)
    d = np.repeat(d, n_blocks, axis=1)
    q = weight.reshape(n_rows, n_blocks, 32).view(np.uint8)
    return np.concatenate([d, q], axis=2).reshape(n_rows, n_blocks*34)


def packed_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    cpu = tensor.detach().cpu()
    if cpu.dtype == torch.int32:
        return cpu.numpy().astype(np.int32, copy=False)
    return cpu.to(torch.int64).numpy().astype(np.uint32, copy=False).view(np.int32)


def pack_binary_matrix(tensor: torch.Tensor) -> np.ndarray:
    data = tensor.detach().cpu().to(torch.float32).numpy()
    if data.ndim != 2:
        raise ValueError(f"expected a 2D binary matrix, got shape {data.shape}")

    # GGUF stores the canonical NanoQuant bit layout: a cleared bit means +1,
    # a set bit means -1, packed least-significant bit first in each word.
    n_rows, n_cols = data.shape
    n_words = (n_cols + 31)//32
    padded = np.zeros((n_rows, n_words*32), dtype=np.uint8)
    padded[:, :n_cols] = data < 0

    bits = padded.reshape(n_rows, n_words, 32).astype(np.uint32, copy=False)
    powers = (np.uint32(1) << np.arange(32, dtype=np.uint32)).reshape(1, 1, 32)
    packed = np.sum(bits*powers, axis=2, dtype=np.uint32)
    return packed.view(np.int32)


def get_tensor(state: dict[str, torch.Tensor], prefix: str, suffixes: Iterable[str]) -> torch.Tensor | None:
    for suffix in suffixes:
        tensor = state.get(prefix + suffix)
        if tensor is not None:
            return tensor
    return None


def get_shape_value(state: dict[str, torch.Tensor], prefix: str, suffixes: Iterable[str]) -> tuple[int, ...] | None:
    for suffix in suffixes:
        key = prefix + suffix
        if key in state:
            return tensor_shape(state[key])
    return None


def build_sidecars(state: dict[str, torch.Tensor]) -> dict[str, NanoQuantSidecar]:
    prefixes: set[str] = set()
    for name in state:
        for suffix in (".scale_pre", ".nq_scale_pre"):
            if name.endswith(suffix):
                prefixes.add(name.removesuffix(suffix))

    sidecars: dict[str, NanoQuantSidecar] = {}
    for prefix in sorted(prefixes):
        scale_pre_t = get_tensor(state, prefix, (".scale_pre", ".nq_scale_pre"))
        scale_mid_t = get_tensor(state, prefix, (".scale_mid", ".nq_scale_mid"))
        scale_post_t = get_tensor(state, prefix, (".scale_post", ".nq_scale_post"))
        v_t = get_tensor(state, prefix, (".V", ".v"))
        u_t = get_tensor(state, prefix, (".U", ".u"))
        v_packed_t = get_tensor(state, prefix, (".V_packed", ".v_packed", ".nq_v"))
        u_packed_t = get_tensor(state, prefix, (".U_packed", ".u_packed", ".nq_u"))
        salient_idx_t = get_tensor(state, prefix, (".salient_idx", ".nq_salient_idx"))
        salient_weight_t = get_tensor(state, prefix, (".salient_weight", ".nq_salient_weight"))
        salient_scale_t = get_tensor(state, prefix, (".salient_scale", ".nq_salient_scale"))

        if scale_pre_t is None or scale_post_t is None:
            continue
        if (v_t is None and v_packed_t is None) or (u_t is None and u_packed_t is None):
            continue
        if (salient_idx_t is None) != (salient_weight_t is None):
            raise ValueError(f"{prefix}: salient_idx and salient_weight must both be present")
        if salient_scale_t is not None and (salient_idx_t is None or salient_weight_t is None):
            raise ValueError(f"{prefix}: salient_scale requires salient_idx and salient_weight")

        scale_pre = scale_to_numpy(scale_pre_t)
        scale_post = scale_to_numpy(scale_post_t)

        v_shape = get_shape_value(state, prefix, (".V_shape", ".v_shape"))
        u_shape = get_shape_value(state, prefix, (".U_shape", ".u_shape"))

        if v_t is not None:
            v_packed = pack_binary_matrix(v_t)
            rank, n_in = tuple(int(v) for v in v_t.shape)
        else:
            assert v_packed_t is not None
            v_packed = packed_to_numpy(v_packed_t)
            if v_shape is not None:
                rank, n_in = v_shape
            else:
                rank, n_in = v_packed.shape[0], scale_pre.size

        if u_t is not None:
            u_packed = pack_binary_matrix(u_t)
            n_out, rank_u = tuple(int(v) for v in u_t.shape)
        else:
            assert u_packed_t is not None
            u_packed = packed_to_numpy(u_packed_t)
            if u_shape is not None:
                n_out, rank_u = u_shape
            else:
                n_out, rank_u = u_packed.shape[0], scale_mid.size

        if rank != rank_u:
            raise ValueError(f"{prefix}: V rank {rank} does not match U rank {rank_u}")
        scale_mid = scale_to_numpy(scale_mid_t) if scale_mid_t is not None else np.ones(rank, dtype=np.float32)
        if scale_pre.size != n_in or scale_mid.size != rank or scale_post.size != n_out:
            raise ValueError(f"{prefix}: scale lengths do not match NanoQuant factor shapes")

        expected_v = (rank, (n_in + 31)//32)
        expected_u = (n_out, (rank + 31)//32)
        if tuple(v_packed.shape) != expected_v:
            raise ValueError(f"{prefix}: V packed shape {v_packed.shape} does not match {expected_v}")
        if tuple(u_packed.shape) != expected_u:
            raise ValueError(f"{prefix}: U packed shape {u_packed.shape} does not match {expected_u}")

        salient_idx = None
        salient_weight = None
        salient_scale = None
        if salient_idx_t is not None and salient_weight_t is not None:
            salient_idx = index_to_numpy(salient_idx_t)
            salient_weight = salient_weight_to_numpy(salient_weight_t)
            k = salient_idx.size

            if k == 0:
                salient_idx = None
                salient_weight = None
            else:
                expected_salient = (n_out, k)
                if tuple(salient_weight.shape) != expected_salient:
                    raise ValueError(f"{prefix}: salient_weight shape {salient_weight.shape} does not match {expected_salient}")
                if np.any(salient_idx < 0) or np.any(salient_idx >= n_in):
                    raise ValueError(f"{prefix}: salient_idx values must be in [0, {n_in})")
                if salient_weight.dtype == np.int8 and salient_scale_t is None:
                    raise ValueError(f"{prefix}: int8 salient_weight requires salient_scale")
                if salient_scale_t is not None:
                    if salient_weight.dtype != np.int8:
                        raise ValueError(f"{prefix}: salient_scale is only supported with int8 salient_weight")
                    salient_scale = scale_to_numpy(salient_scale_t)
                    if salient_scale.size != k:
                        raise ValueError(f"{prefix}: salient_scale length {salient_scale.size} does not match {k}")

        sidecars[prefix] = NanoQuantSidecar(
            v_packed=np.ascontiguousarray(v_packed, dtype=np.int32),
            u_packed=np.ascontiguousarray(u_packed, dtype=np.int32),
            scale_pre=np.ascontiguousarray(scale_pre),
            scale_mid=np.ascontiguousarray(scale_mid),
            scale_post=np.ascontiguousarray(scale_post),
            salient_idx=None if salient_idx is None else np.ascontiguousarray(salient_idx, dtype=np.int32),
            salient_weight=None if salient_weight is None else np.ascontiguousarray(salient_weight),
            salient_scale=None if salient_scale is None else np.ascontiguousarray(salient_scale),
        )

    if not sidecars:
        raise ValueError("No NanoQuant sidecars found in checkpoint")
    return sidecars


def build_int8_embeddings(state: dict[str, torch.Tensor]) -> dict[str, Int8Embedding]:
    embeddings: dict[str, Int8Embedding] = {}
    for name, tensor in sorted(state.items()):
        if not name.endswith("_int8"):
            continue
        base = name.removesuffix("_int8")
        if not base.endswith("embed_tokens.weight"):
            continue

        scale_name = base + "_int8_scale"
        scale_t = state.get(scale_name)
        if scale_t is None:
            raise ValueError(f"{name}: missing embedding scale tensor {scale_name}")

        weight = int8_weight_to_numpy(tensor, name)
        scale = scale_to_numpy(scale_t)
        if scale.size != weight.shape[0]:
            raise ValueError(f"{scale_name}: scale length {scale.size} does not match row count {weight.shape[0]}")

        embeddings[base] = Int8Embedding(
            weight=np.ascontiguousarray(weight),
            scale=np.ascontiguousarray(scale),
        )
    return embeddings


def prefix_candidates(prefix: str) -> Iterable[str]:
    seen: set[str] = set()
    candidates = [
        prefix,
        prefix.removeprefix("module."),
        prefix.removeprefix("base_model."),
        prefix.removeprefix("base_model.model."),
    ]
    if not prefix.startswith("model."):
        candidates.append("model." + prefix)
    if prefix.startswith("model."):
        candidates.append(prefix.removeprefix("model."))
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def find_sidecar(sidecars: dict[str, NanoQuantSidecar], prefix: str) -> tuple[str, NanoQuantSidecar] | None:
    for candidate in prefix_candidates(prefix):
        sidecar = sidecars.get(candidate)
        if sidecar is not None:
            return candidate, sidecar
    return None


def find_int8_embedding(embeddings: dict[str, Int8Embedding], name: str) -> tuple[str, Int8Embedding] | None:
    for candidate in prefix_candidates(name):
        embedding = embeddings.get(candidate)
        if embedding is not None:
            return candidate, embedding
    return None


def maybe_permute_output_rows(model: Any, hf_weight_name: str, sidecar: NanoQuantSidecar) -> NanoQuantSidecar:
    if not getattr(model, "undo_permute", False) or not hasattr(model, "permute"):
        return sidecar
    if not hf_weight_name.endswith(("q_proj.weight", "k_proj.weight")):
        return sidecar

    # The base converter may undo attention row permutation.  Apply the same row
    # order to U/post/outlier weights so the virtual dense weight matches it.
    n_head = model.find_hparam(["n_heads", "num_attention_heads"], optional=True)
    if n_head is None:
        return sidecar
    n_kv_head = n_head
    if hf_weight_name.endswith("k_proj.weight"):
        n_kv_head = model.find_hparam(["n_kv_heads", "num_key_value_heads"], optional=True)

    order = torch.arange(sidecar.u_packed.shape[0], dtype=torch.int64).reshape(sidecar.u_packed.shape[0], 1)
    order = model.permute(order, n_head, n_kv_head).reshape(-1).cpu().numpy()

    return NanoQuantSidecar(
        v_packed=sidecar.v_packed,
        u_packed=np.ascontiguousarray(sidecar.u_packed[order]),
        scale_pre=sidecar.scale_pre,
        scale_mid=sidecar.scale_mid,
        scale_post=np.ascontiguousarray(sidecar.scale_post[order]),
        salient_idx=sidecar.salient_idx,
        salient_weight=None if sidecar.salient_weight is None else np.ascontiguousarray(sidecar.salient_weight[order]),
        salient_scale=sidecar.salient_scale,
    )


def write_sidecar(model: Any, base_name: str, hf_weight_name: str, sidecar: NanoQuantSidecar) -> None:
    sidecar = maybe_permute_output_rows(model, hf_weight_name, sidecar)
    writer = model.gguf_writer

    writer.add_tensor(base_name + ".nq_v", sidecar.v_packed)
    writer.add_tensor(base_name + ".nq_u", sidecar.u_packed)
    writer.add_tensor(base_name + ".nq_scale_pre", sidecar.scale_pre)
    writer.add_tensor(base_name + ".nq_scale_mid", sidecar.scale_mid)
    writer.add_tensor(base_name + ".nq_scale_post", sidecar.scale_post)

    outliers = 0
    if sidecar.salient_idx is not None and sidecar.salient_weight is not None:
        writer.add_tensor(base_name + ".nq_salient_idx", sidecar.salient_idx)
        writer.add_tensor(base_name + ".nq_salient_weight", sidecar.salient_weight)
        if sidecar.salient_scale is not None:
            writer.add_tensor(base_name + ".nq_salient_scale", sidecar.salient_scale)
        outliers = sidecar.salient_idx.size

    logger.info(f"{base_name + '.nq_*,' :<30} NanoQuant rank = {sidecar.scale_mid.size}, outliers = {outliers}")


def write_int8_embedding(model: Any, new_name: str, mapped_tensor: torch.Tensor, embedding: Int8Embedding) -> None:
    expected_shape = tuple(int(v) for v in mapped_tensor.shape)
    if len(expected_shape) != 2:
        raise ValueError(f"{new_name}: expected mapped embedding to be 2D, got {expected_shape}")

    n_rows, n_cols = expected_shape
    if embedding.weight.shape[1] != n_cols:
        raise ValueError(f"{new_name}: int8 embedding width {embedding.weight.shape[1]} does not match mapped width {n_cols}")
    if embedding.weight.shape[0] < n_rows:
        raise ValueError(f"{new_name}: int8 embedding has {embedding.weight.shape[0]} rows, mapped tensor needs {n_rows}")

    weight = embedding.weight[:n_rows]
    scale = embedding.scale[:n_rows]
    # Embeddings use regular GGUF Q8_0 so token lookup stays on existing
    # get_rows kernels instead of going through the NanoQuant linear op.
    q8_0 = pack_q8_0_from_rowwise_int8(weight, scale, new_name)

    model.gguf_writer.add_tensor(new_name, q8_0, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    shape_str = f"{{{', '.join(str(n) for n in reversed(expected_shape))}}}"
    logger.info(f"{new_name + ',':<30} int8 rowwise --> Q8_0, shape = {shape_str}")


def install_nanoquant_hooks(
        model: Any,
        sidecars: dict[str, NanoQuantSidecar],
        int8_embeddings: dict[str, Int8Embedding],
) -> tuple[set[str], set[str]]:
    original_modify_tensors = model.modify_tensors
    original_prepare_metadata = model.prepare_metadata
    written: set[str] = set()
    written_embeddings: set[str] = set()

    def modify_tensors(data_torch: torch.Tensor, name: str, bid: int | None):
        if not name.endswith(".weight"):
            yield from original_modify_tensors(data_torch, name, bid)
            return

        # Hook the normal converter mapping first, then replace only supported
        # dense weights with NanoQuant sidecars under the mapped GGUF name.
        embedding_match = find_int8_embedding(int8_embeddings, name)
        if embedding_match is not None:
            prefix, embedding = embedding_match
            mapped = list(original_modify_tensors(data_torch, name, bid))
            if len(mapped) == 1:
                new_name, mapped_tensor = mapped[0]
                if new_name.endswith("token_embd.weight"):
                    write_int8_embedding(model, new_name, mapped_tensor, embedding)
                    written_embeddings.add(prefix)
                    return
            yield from mapped
            return

        match = find_sidecar(sidecars, name.removesuffix(".weight"))
        if match is None:
            yield from original_modify_tensors(data_torch, name, bid)
            return

        prefix, sidecar = match
        mapped = list(original_modify_tensors(data_torch, name, bid))
        if len(mapped) != 1:
            yield from mapped
            return

        new_name, mapped_tensor = mapped[0]
        if not any(new_name.endswith(suffix) for suffix in SUPPORTED_GGUF_WEIGHT_SUFFIXES):
            yield new_name, mapped_tensor
            return

        write_sidecar(model, new_name.removesuffix(".weight"), name, sidecar)
        written.add(prefix)
        return

    def prepare_metadata(vocab_only: bool):
        original_prepare_metadata(vocab_only)
        has_outliers = any(sidecar.salient_idx is not None for sidecar in sidecars.values())
        model.gguf_writer.add_uint32("quantization.nanoquant.version", 2 if has_outliers else 1)
        model.gguf_writer.add_string("quantization.nanoquant.bit_order", "lsb_first")
        model.gguf_writer.add_string("quantization.nanoquant.positive_bit", "0")
        model.gguf_writer.add_uint32("quantization.nanoquant.tensor_count", len(written))
        if written_embeddings:
            model.gguf_writer.add_string("quantization.nanoquant.embedding_format", "q8_0_rowwise_int8")
            model.gguf_writer.add_uint32("quantization.nanoquant.embedding_tensor_count", len(written_embeddings))
        if has_outliers:
            model.gguf_writer.add_string("quantization.nanoquant.outlier_format", "column_side_path")

    model.modify_tensors = modify_tensors
    model.prepare_metadata = prepare_metadata
    return written, written_embeddings


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    dir_model = args.model
    if not dir_model.is_dir():
        raise FileNotFoundError(f"model path is not a directory: {dir_model}")

    ftype_map: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "auto": gguf.LlamaFileType.GUESSED,
    }

    state = load_checkpoint(args.nanoquant_checkpoint)
    sidecars = build_sidecars(state)
    logger.info(f"Found {len(sidecars)} NanoQuant tensor groups")
    int8_embeddings = build_int8_embeddings(state)
    if int8_embeddings:
        logger.info(f"Found {len(int8_embeddings)} row-wise int8 embedding tensor(s)")

    hparams = ModelBase.load_hparams(dir_model, is_mistral_format=False)
    model_architecture = get_model_architecture(hparams, ModelType.TEXT)
    logger.info(f"Model architecture: {model_architecture}")
    model_class = get_model_class(model_architecture, mmproj=False)

    fname_out = args.outfile if args.outfile is not None else dir_model
    model = model_class(
        dir_model,
        ftype_map[args.outtype],
        fname_out,
        is_big_endian=args.bigendian,
        use_temp_file=args.use_temp_file,
        eager=args.no_lazy,
        metadata_override=args.metadata,
        model_name=args.model_name,
        split_max_tensors=args.split_max_tensors,
        split_max_size=split_str_to_n_bytes(args.split_max_size),
    )
    written, written_embeddings = install_nanoquant_hooks(model, sidecars, int8_embeddings)

    with torch.inference_mode():
        model.write()

    unused = sorted(set(sidecars) - written)
    if unused:
        logger.warning(f"Unused NanoQuant tensor groups: {len(unused)}")
        for name in unused[:20]:
            logger.warning(f"  {name}")

    unused_embeddings = sorted(set(int8_embeddings) - written_embeddings)
    if unused_embeddings:
        logger.warning(f"Unused row-wise int8 embedding tensors: {len(unused_embeddings)}")
        for name in unused_embeddings[:20]:
            logger.warning(f"  {name}")

    logger.info(f"NanoQuant GGUF successfully exported to {model.fname_out}")


if __name__ == "__main__":
    main()
