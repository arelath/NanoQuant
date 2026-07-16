"""Shared GGUF weight loading and color PNG generation helpers."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import struct
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

DEFAULT_BF16_MODEL = Path(r"D:\dev\research\NanoQuant\models\gemma-3-1b-it-BF16.gguf")
DEFAULT_NANOQUANT_MODEL = Path(
    r"D:\dev\research\NanoQuantRewrite\evidence\m6\gemma-pageable-v28-nanoquant.gguf"
)
DEFAULT_LLAMA_ROOT = Path(r"D:\dev\research\llama.cpp")

# Numeric --block values follow the source model's projection order.
LAYER_NAMES = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)
_LAYER_ALIASES = {
    "q": "attn_q",
    "q_proj": "attn_q",
    "self_attn.q_proj": "attn_q",
    "k": "attn_k",
    "k_proj": "attn_k",
    "self_attn.k_proj": "attn_k",
    "v": "attn_v",
    "v_proj": "attn_v",
    "self_attn.v_proj": "attn_v",
    "o": "attn_output",
    "o_proj": "attn_output",
    "self_attn.o_proj": "attn_output",
    "gate": "ffn_gate",
    "gate_proj": "ffn_gate",
    "mlp.gate_proj": "ffn_gate",
    "up": "ffn_up",
    "up_proj": "ffn_up",
    "mlp.up_proj": "ffn_up",
    "down": "ffn_down",
    "down_proj": "ffn_down",
    "mlp.down_proj": "ffn_down",
}
_NANOQUANT_BASE = re.compile(r"blk\.(?P<block>[0-9]+)\.(?P<layer>[^.]+)\.nq_v\Z")

Mode = Literal["bf16", "nanoquant", "difference"]
WeightScale = Literal["full-range", "percentile-01-99", "signed-asinh"]
WEIGHT_SCALES: tuple[WeightScale, ...] = (
    "full-range",
    "percentile-01-99",
    "signed-asinh",
)
AXIS_SPECTRUM_DYNAMIC_RANGE_DB = 6.0


@dataclass(frozen=True, slots=True)
class ImageRecord:
    transformer_layer: int
    weight_block: str
    variant: str
    domain: str
    palette: str
    tensor: str
    path: str
    width: int
    height: int
    minimum: float
    maximum: float
    scaling_minimum: float
    scaling_maximum: float
    transform_scale: float | None


def _existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def load_gguf_module(llama_root: Path) -> Any:
    """Import the GGUF reader from the selected llama.cpp checkout."""

    gguf_py = llama_root.expanduser().resolve() / "gguf-py"
    if not gguf_py.is_dir():
        raise FileNotFoundError(f"llama.cpp gguf-py directory does not exist: {gguf_py}")
    gguf_path = str(gguf_py)
    if gguf_path not in sys.path:
        sys.path.insert(0, gguf_path)
    return importlib.import_module("gguf")


def tensor_map(reader: Any) -> dict[str, Any]:
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("GGUF contains duplicate tensor names")
    return tensors


def nanoquant_inventory(tensors: dict[str, Any]) -> dict[int, tuple[str, ...]]:
    """Return the transformer-layer/projection inventory declared by NanoQuant sidecars."""

    found: dict[int, set[str]] = {}
    for name in tensors:
        match = _NANOQUANT_BASE.fullmatch(name)
        if match is not None:
            found.setdefault(int(match.group("block")), set()).add(match.group("layer"))
    if not found:
        raise ValueError("GGUF contains no NanoQuant .nq_v tensors")
    unknown = sorted({layer for layers in found.values() for layer in layers} - set(LAYER_NAMES))
    if unknown:
        raise ValueError(f"GGUF contains unsupported NanoQuant layer names: {unknown}")
    return {
        block: tuple(layer for layer in LAYER_NAMES if layer in layers)
        for block, layers in sorted(found.items())
    }


def resolve_weight_block(value: str) -> tuple[int, str]:
    normalized = value.strip().lower().removesuffix(".weight")
    try:
        index = int(normalized)
    except ValueError:
        name = _LAYER_ALIASES.get(normalized, normalized)
        if name not in LAYER_NAMES:
            choices = ", ".join(f"{index}={name}" for index, name in enumerate(LAYER_NAMES))
            raise ValueError(f"unknown weight block {value!r}; choices are {choices}") from None
        return LAYER_NAMES.index(name), name
    if index < 0 or index >= len(LAYER_NAMES):
        raise ValueError(f"weight block index must be between 0 and {len(LAYER_NAMES) - 1}: {index}")
    return index, LAYER_NAMES[index]


def select_bases(
    inventory: dict[int, tuple[str, ...]], layer: str, block: str
) -> tuple[tuple[int, str, str], ...]:
    normalized_layer = layer.strip().lower()
    if normalized_layer == "all":
        layer_indexes = tuple(inventory)
    else:
        try:
            layer_index = int(normalized_layer)
        except ValueError:
            raise ValueError(f"layer must be a non-negative integer or 'all': {layer!r}") from None
        if layer_index < 0:
            raise ValueError(f"layer must be non-negative: {layer_index}")
        if layer_index not in inventory:
            raise ValueError(f"transformer layer {layer_index} contains no NanoQuant weights")
        layer_indexes = (layer_index,)

    normalized_block = block.strip().lower()
    selected_block = None if normalized_block == "all" else resolve_weight_block(block)[1]
    selected = tuple(
        (layer_index, weight_block, f"blk.{layer_index}.{weight_block}")
        for layer_index in layer_indexes
        for weight_block in inventory[layer_index]
        if selected_block is None or weight_block == selected_block
    )
    if not selected:
        raise ValueError(
            f"selected transformer layer(s) do not contain NanoQuant weight block {selected_block}"
        )
    return selected


def _float_tensor(tensor: Any, gguf: Any) -> np.ndarray:
    data = np.asarray(tensor.data)
    tensor_type_name = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
    if tensor_type_name in {"I8", "I16", "I32", "I64"}:
        value = data
    else:
        try:
            value = gguf.dequantize(data, tensor.tensor_type)
        except NotImplementedError as error:
            raise ValueError(
                f"tensor {tensor.name} has unsupported type {tensor.tensor_type}"
            ) from error
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"tensor {tensor.name} contains a non-finite value")
    return result


def load_dense_weight(tensors: dict[str, Any], base: str, gguf: Any) -> np.ndarray:
    name = f"{base}.weight"
    try:
        tensor = tensors[name]
    except KeyError as error:
        raise ValueError(f"BF16 GGUF is missing source weight {name}") from error
    weight = _float_tensor(tensor, gguf)
    if weight.ndim != 2:
        raise ValueError(f"source weight {name} is not a matrix: {weight.shape}")
    return weight


def unpack_sign_words(words: np.ndarray, rows: int, columns: int) -> np.ndarray:
    """Unpack llama.cpp I32, least-significant-bit-first NanoQuant signs."""

    value = np.asarray(words)
    expected = (rows, (columns + 31) // 32)
    if value.dtype != np.int32 or value.shape != expected:
        raise ValueError(f"packed sign words must be int32 {expected}, got {value.dtype} {value.shape}")
    unsigned = value.view(np.uint32)
    lanes = np.arange(32, dtype=np.uint32)
    bits = ((unsigned[..., np.newaxis] >> lanes) & np.uint32(1)).reshape(rows, -1)[:, :columns]
    return np.float32(1.0) - np.float32(2.0) * bits.astype(np.float32)


def reconstruct_nanoquant_weight(tensors: dict[str, Any], base: str, gguf: Any) -> np.ndarray:
    """Materialize one effective dense weight from a GGUF NanoQuant sidecar group."""

    def required(suffix: str) -> Any:
        name = f"{base}.{suffix}"
        try:
            return tensors[name]
        except KeyError as error:
            raise ValueError(f"NanoQuant GGUF is missing tensor {name}") from error

    scale_pre = _float_tensor(required("nq_scale_pre"), gguf).reshape(-1).copy()
    scale_mid = _float_tensor(required("nq_scale_mid"), gguf).reshape(-1)
    scale_post = _float_tensor(required("nq_scale_post"), gguf).reshape(-1)
    in_features = int(scale_pre.size)
    rank = int(scale_mid.size)
    out_features = int(scale_post.size)
    if min(in_features, rank, out_features) <= 0:
        raise ValueError(f"NanoQuant tensor {base} has an empty dimension")

    index_tensor = tensors.get(f"{base}.nq_salient_idx")
    value_tensor = tensors.get(f"{base}.nq_salient_weight")
    scale_tensor = tensors.get(f"{base}.nq_salient_scale")
    if (index_tensor is None) != (value_tensor is None):
        raise ValueError(f"NanoQuant tensor {base} has an incomplete salient side path")
    indices: np.ndarray | None = None
    salient: np.ndarray | None = None
    if index_tensor is not None and value_tensor is not None:
        indices = np.asarray(index_tensor.data, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= in_features):
            raise ValueError(f"NanoQuant tensor {base} has an out-of-range salient index")
        if indices.size > 1 and np.any(indices[1:] <= indices[:-1]):
            raise ValueError(f"NanoQuant tensor {base} salient indices are not strictly increasing")
        scale_pre[indices] = 0.0
        salient = _float_tensor(value_tensor, gguf)
        if salient.shape != (out_features, indices.size):
            raise ValueError(
                f"NanoQuant tensor {base} salient values have shape {salient.shape}, "
                f"expected {(out_features, indices.size)}"
            )
        if scale_tensor is not None:
            salient_scale = _float_tensor(scale_tensor, gguf).reshape(-1)
            if salient_scale.shape != (indices.size,):
                raise ValueError(f"NanoQuant tensor {base} salient scales have the wrong shape")
            salient = salient * salient_scale.reshape(1, -1)
    elif scale_tensor is not None:
        raise ValueError(f"NanoQuant tensor {base} has salient scales without salient values")

    right_words = np.asarray(required("nq_v").data)
    left_words = np.asarray(required("nq_u").data)
    right = unpack_sign_words(right_words, rank, in_features)
    left = unpack_sign_words(left_words, out_features, rank)
    right *= scale_mid.reshape(-1, 1)
    right *= scale_pre.reshape(1, -1)
    left *= scale_post.reshape(-1, 1)
    weight = left @ right
    if indices is not None and salient is not None:
        weight[:, indices] += salient
    if weight.shape != (out_features, in_features) or not np.isfinite(weight).all():
        raise ValueError(f"reconstructed NanoQuant weight {base} is invalid")
    return np.asarray(weight, dtype=np.float32)


def scale_to_grayscale(
    weight: np.ndarray,
    *,
    scaling_minimum: float | None = None,
    scaling_maximum: float | None = None,
) -> tuple[np.ndarray, float, float]:
    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"weight image input must be a non-empty matrix: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("weight image input contains a non-finite value")
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    lower = minimum if scaling_minimum is None else float(scaling_minimum)
    upper = maximum if scaling_maximum is None else float(scaling_maximum)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError(f"grayscale scaling range is invalid: [{lower}, {upper}]")
    if upper == lower:
        pixels = np.zeros(value.shape, dtype=np.uint8)
    else:
        normalized = (value - np.float32(lower)) / np.float32(upper - lower)
        pixels = np.rint(normalized * np.float32(255.0)).clip(0, 255).astype(np.uint8)
    return pixels, minimum, maximum


def scale_difference_to_diverging(
    difference: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Stretch signed error from its extrema to black-centered blue/red RGB."""

    value = np.asarray(difference, dtype=np.float32)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"difference image input must be a non-empty matrix: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("difference image input contains a non-finite value")
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    pixels = np.zeros((*value.shape, 3), dtype=np.uint8)
    if maximum > 0.0:
        positive = np.rint(value / np.float32(maximum) * np.float32(255.0)).clip(0, 255).astype(np.uint8)
        pixels[..., 0] = np.where(value > 0.0, positive, 0)
    if minimum < 0.0:
        negative = (
            np.rint(-value / np.float32(-minimum) * np.float32(255.0))
            .clip(0, 255)
            .astype(np.uint8)
        )
        pixels[..., 2] = np.where(value < 0.0, negative, 0)
    return pixels, minimum, maximum


def _topographic_pixels(normalized: np.ndarray) -> np.ndarray:
    stops = np.array([0.0, 0.25, 0.5, 0.625, 0.75, 0.875, 1.0], dtype=np.float32)
    colors = np.array(
        [
            (0, 20, 110),
            (0, 170, 210),
            (35, 130, 55),
            (100, 180, 60),
            (225, 205, 55),
            (150, 90, 45),
            (255, 255, 255),
        ],
        dtype=np.float32,
    )
    pixels = np.empty((*normalized.shape, 3), dtype=np.uint8)
    for channel in range(3):
        pixels[..., channel] = np.rint(
            np.interp(normalized, stops, colors[:, channel])
        ).astype(np.uint8)
    return pixels


def scale_weight_to_topographic(
    weight: np.ndarray,
    *,
    variant: WeightScale = "full-range",
) -> tuple[np.ndarray, float, float, float, float, float | None]:
    """Map signed weights through one terrain-colored normalization variant."""

    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"weight image input must be a non-empty matrix: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("weight image input contains a non-finite value")
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    maximum_magnitude = max(abs(minimum), abs(maximum))
    transform_scale: float | None = None
    if variant == "full-range":
        scaling_magnitude = maximum_magnitude
        signed = (
            np.zeros(value.shape, dtype=np.float32)
            if scaling_magnitude == 0.0
            else value / np.float32(scaling_magnitude)
        )
    elif variant == "percentile-01-99":
        percentile_minimum, percentile_maximum = np.percentile(value, (1.0, 99.0))
        scaling_magnitude = max(abs(float(percentile_minimum)), abs(float(percentile_maximum)))
        signed = (
            np.zeros(value.shape, dtype=np.float32)
            if scaling_magnitude == 0.0
            else value / np.float32(scaling_magnitude)
        )
    elif variant == "signed-asinh":
        scaling_magnitude = maximum_magnitude
        absolute = np.abs(value)
        nonzero = absolute[absolute > 0.0]
        transform_scale = float(np.median(nonzero)) if nonzero.size else 0.0
        if scaling_magnitude == 0.0 or transform_scale == 0.0:
            signed = np.zeros(value.shape, dtype=np.float32)
        else:
            transformed = np.arcsinh(value / np.float32(transform_scale))
            transformed_peak = float(np.arcsinh(np.float32(scaling_magnitude / transform_scale)))
            signed = transformed / np.float32(transformed_peak)
    else:
        raise ValueError(f"unsupported topographic weight scale: {variant}")
    normalized = ((signed.clip(-1.0, 1.0) + np.float32(1.0)) * np.float32(0.5)).astype(np.float32)
    return (
        _topographic_pixels(normalized),
        minimum,
        maximum,
        -scaling_magnitude,
        scaling_magnitude,
        transform_scale,
    )


def scale_frequency_spectrum(
    weight: np.ndarray,
    *,
    dynamic_range_db: float = AXIS_SPECTRUM_DYNAMIC_RANGE_DB,
) -> tuple[np.ndarray, float, float]:
    """Render independent centered X/Y FFT prominence as vertical/horizontal bands."""

    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"frequency image input must be a non-empty matrix: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("frequency image input contains a non-finite value")
    if not np.isfinite(dynamic_range_db) or dynamic_range_db <= 0.0:
        raise ValueError(f"frequency dynamic range must be positive: {dynamic_range_db}")
    _x_frequency, x_decibels, _y_frequency, y_decibels = axis_frequency_profiles(value)
    x_normalized = (np.maximum(x_decibels, 0.0) / np.float32(dynamic_range_db)).clip(0.0, 1.0)
    y_normalized = (np.maximum(y_decibels, 0.0) / np.float32(dynamic_range_db)).clip(0.0, 1.0)
    normalized = np.maximum(y_normalized.reshape(-1, 1), x_normalized.reshape(1, -1))
    stops = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
    colors = np.array(
        [
            (0, 0, 0),
            (35, 0, 70),
            (110, 10, 105),
            (205, 45, 45),
            (250, 155, 20),
            (255, 255, 220),
        ],
        dtype=np.float32,
    )
    pixels = np.empty((*value.shape, 3), dtype=np.uint8)
    for channel in range(3):
        pixels[..., channel] = np.rint(
            np.interp(normalized, stops, colors[:, channel])
        ).astype(np.uint8)
    return pixels, 0.0, float(dynamic_range_db)


def axis_frequency_profiles(
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centered X/Y frequencies and power prominence over each axis median."""

    value = np.asarray(weight, dtype=np.float32)
    if value.ndim != 2 or value.size == 0:
        raise ValueError(f"frequency profile input must be a non-empty matrix: {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("frequency profile input contains a non-finite value")

    def profile(samples: np.ndarray, *, transform_axis: int, reduce_axis: int) -> tuple[np.ndarray, np.ndarray]:
        centered = samples - np.mean(samples, axis=transform_axis, keepdims=True, dtype=np.float64)
        transformed = np.fft.fftshift(np.fft.fft(centered, axis=transform_axis), axes=transform_axis)
        power = np.mean(np.abs(transformed) ** 2, axis=reduce_axis, dtype=np.float64)
        positive = power[power > 0.0]
        if positive.size == 0:
            prominence = np.zeros(power.shape, dtype=np.float32)
        else:
            baseline = float(np.median(positive))
            prominence = (10.0 * np.log10(np.maximum(power / baseline, np.finfo(np.float64).tiny))).astype(
                np.float32
            )
        frequency = np.fft.fftshift(np.fft.fftfreq(samples.shape[transform_axis])).astype(np.float32)
        return frequency, prominence

    x_frequency, x_decibels = profile(value, transform_axis=1, reduce_axis=0)
    y_frequency, y_decibels = profile(value, transform_axis=0, reduce_axis=1)
    return x_frequency, x_decibels, y_frequency, y_decibels


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


def write_grayscale_png(path: Path, pixels: np.ndarray) -> None:
    """Write an 8-bit, non-interlaced grayscale PNG using only the standard library."""

    value = np.asarray(pixels)
    if value.dtype != np.uint8 or value.ndim != 2 or value.size == 0:
        raise ValueError("PNG pixels must be a non-empty uint8 matrix")
    height, width = value.shape
    if width > 0x7FFFFFFF or height > 0x7FFFFFFF:
        raise ValueError(f"PNG dimensions are too large: {width}x{height}")
    rows = b"".join(b"\x00" + row.tobytes() for row in value)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def write_rgb_png(path: Path, pixels: np.ndarray) -> None:
    """Write an 8-bit, non-interlaced RGB PNG using only the standard library."""

    value = np.asarray(pixels)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or value.size == 0:
        raise ValueError("PNG pixels must be a non-empty uint8 RGB matrix")
    height, width, _channels = value.shape
    if width > 0x7FFFFFFF or height > 0x7FFFFFFF:
        raise ValueError(f"PNG dimensions are too large: {width}x{height}")
    rows = b"".join(b"\x00" + row.tobytes() for row in value)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def _default_llama_root() -> Path:
    configured = os.environ.get("NANOQUANT_LLAMA_CPP")
    return DEFAULT_LLAMA_ROOT if configured is None else Path(configured)


def build_parser(mode: Mode) -> argparse.ArgumentParser:
    descriptions = {
        "bf16": "Render source BF16 weights for NanoQuant-quantized Gemma projections.",
        "nanoquant": "Reconstruct and render NanoQuant GGUF weights.",
        "difference": "Render NanoQuant-minus-BF16 weight differences.",
    }
    parser = argparse.ArgumentParser(description=descriptions[mode])
    parser.add_argument(
        "--layer",
        default="5",
        help="transformer layer index or 'all' (default: 0)",
    )
    block_choices = ", ".join(f"{index}={name}" for index, name in enumerate(LAYER_NAMES))
    parser.add_argument(
        "--block",
        default="all",
        help=f"weight block ordinal, name, or 'all' (default: all; {block_choices})",
    )
    parser.add_argument(
        "--llama-root",
        type=Path,
        default=_default_llama_root(),
        help="llama.cpp checkout containing gguf-py",
    )
    default_output = Path("weight_images") / mode
    parser.add_argument("--output-dir", type=Path, default=default_output)
    if mode == "nanoquant":
        parser.add_argument("--model", type=Path, default=DEFAULT_NANOQUANT_MODEL)
    else:
        parser.add_argument("--bf16-model", type=Path, default=DEFAULT_BF16_MODEL)
        parser.add_argument(
            "--quantized-model",
            type=Path,
            default=DEFAULT_NANOQUANT_MODEL,
            help="NanoQuant GGUF used as the authoritative quantized block/layer inventory",
        )
    return parser


def run(mode: Mode) -> None:
    args = build_parser(mode).parse_args()
    gguf = load_gguf_module(args.llama_root)

    quantized_path = _existing_file(
        args.model if mode == "nanoquant" else args.quantized_model,
        "NanoQuant GGUF",
    )
    quantized_reader = gguf.GGUFReader(quantized_path, "r")
    quantized_tensors = tensor_map(quantized_reader)
    selected = select_bases(nanoquant_inventory(quantized_tensors), args.layer, args.block)

    bf16_path: Path | None = None
    bf16_tensors: dict[str, Any] | None = None
    if mode != "nanoquant":
        bf16_path = _existing_file(args.bf16_model, "BF16 GGUF")
        bf16_reader = gguf.GGUFReader(bf16_path, "r")
        bf16_tensors = tensor_map(bf16_reader)

    output_dir = args.output_dir.expanduser().resolve()
    records: list[ImageRecord] = []
    suffix = {"bf16": "bf16", "nanoquant": "nanoquant", "difference": "nanoquant-minus-bf16"}[mode]
    for layer_index, weight_block, base in selected:
        if mode == "bf16":
            assert bf16_tensors is not None
            weight = load_dense_weight(bf16_tensors, base, gguf)
        elif mode == "nanoquant":
            weight = reconstruct_nanoquant_weight(quantized_tensors, base, gguf)
        else:
            assert bf16_tensors is not None
            source = load_dense_weight(bf16_tensors, base, gguf)
            quantized = reconstruct_nanoquant_weight(quantized_tensors, base, gguf)
            if source.shape != quantized.shape:
                raise ValueError(
                    f"source and NanoQuant weight shapes differ for {base}: "
                    f"{source.shape} != {quantized.shape}"
                )
            weight = quantized - source
        variants: list[tuple[str, np.ndarray, float, float, float, float, float | None]] = []
        if mode == "difference":
            pixels, minimum, maximum = scale_difference_to_diverging(weight)
            variants.append(("difference-extrema", pixels, minimum, maximum, minimum, maximum, None))
        else:
            for variant in WEIGHT_SCALES:
                pixels, minimum, maximum, scaling_minimum, scaling_maximum, transform_scale = (
                    scale_weight_to_topographic(weight, variant=variant)
                )
                variants.append(
                    (
                        variant,
                        pixels,
                        minimum,
                        maximum,
                        scaling_minimum,
                        scaling_maximum,
                        transform_scale,
                    )
                )
        spectrum, spectrum_minimum, spectrum_maximum = scale_frequency_spectrum(weight)
        variants.append(
            (
                "frequency-spectrum",
                spectrum,
                spectrum_minimum,
                spectrum_maximum,
                spectrum_minimum,
                spectrum_maximum,
                AXIS_SPECTRUM_DYNAMIC_RANGE_DB,
            )
        )
        for variant, pixels, minimum, maximum, scaling_minimum, scaling_maximum, transform_scale in variants:
            variant_suffix = "" if variant == "difference-extrema" else f"-{variant}"
            path = output_dir / f"layer-{layer_index:02d}-{weight_block}-{suffix}{variant_suffix}.png"
            write_rgb_png(path, pixels)
            frequency_domain = variant == "frequency-spectrum"
            records.append(
                ImageRecord(
                    transformer_layer=layer_index,
                    weight_block=weight_block,
                    variant=variant,
                    domain="frequency-xy" if frequency_domain else "weight",
                    palette=(
                        "spectrometer_black_purple_red_yellow_white"
                        if frequency_domain
                        else (
                            "negative_blue_zero_black_positive_red"
                            if mode == "difference"
                            else "topographic_deep_blue_cyan_green_yellow_brown_white"
                        )
                    ),
                    tensor=f"{base}.weight",
                    path=str(path),
                    width=int(weight.shape[1]),
                    height=int(weight.shape[0]),
                    minimum=minimum,
                    maximum=maximum,
                    scaling_minimum=scaling_minimum,
                    scaling_maximum=scaling_maximum,
                    transform_scale=transform_scale,
                )
            )

    report = {
        "mode": mode,
        "layer": args.layer,
        "block": args.block,
        "bf16_model": None if bf16_path is None else str(bf16_path),
        "nanoquant_model": str(quantized_path),
        "difference": "nanoquant_minus_bf16" if mode == "difference" else None,
        "scaling": "per-image; see variant, scaling bounds, and transform scale",
        "palette": (
            "per-image; see each image's palette"
        ),
        "weight_variants": (
            ["difference-extrema", "frequency-spectrum"]
            if mode == "difference"
            else [*WEIGHT_SCALES, "frequency-spectrum"]
        ),
        "images": [asdict(record) for record in records],
    }
    print(json.dumps(report, sort_keys=True, indent=2))
