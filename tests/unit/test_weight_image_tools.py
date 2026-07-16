from __future__ import annotations

import struct
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tools.weight_image_common import (
    axis_frequency_profiles,
    nanoquant_inventory,
    reconstruct_nanoquant_weight,
    resolve_weight_block,
    scale_difference_to_diverging,
    scale_frequency_spectrum,
    scale_to_grayscale,
    scale_weight_to_topographic,
    select_bases,
    unpack_sign_words,
    write_grayscale_png,
    write_rgb_png,
)


class _DirectGGUF:
    @staticmethod
    def dequantize(data: np.ndarray, _tensor_type: object) -> np.ndarray:
        return data


def _tensor(name: str, data: np.ndarray) -> Any:
    return SimpleNamespace(name=name, data=data, tensor_type="fixture")


def _pack(signs: np.ndarray) -> np.ndarray:
    rows, columns = signs.shape
    result = np.zeros((rows, (columns + 31) // 32), dtype=np.uint32)
    for row in range(rows):
        for column in range(columns):
            if signs[row, column] == -1:
                result[row, column // 32] |= np.uint32(1 << (column % 32))
    return result.view(np.int32)


def test_layer_and_block_selection_uses_only_nanoquant_inventory() -> None:
    names = (
        "blk.0.attn_q.nq_v",
        "blk.0.ffn_down.nq_v",
        "blk.1.attn_q.nq_v",
        "blk.1.attn_norm.weight",
    )
    inventory = nanoquant_inventory({name: object() for name in names})

    assert inventory == {0: ("attn_q", "ffn_down"), 1: ("attn_q",)}
    assert resolve_weight_block("0") == (0, "attn_q")
    assert resolve_weight_block("mlp.down_proj.weight") == (6, "ffn_down")
    assert select_bases(inventory, "all", "attn_q") == (
        (0, "attn_q", "blk.0.attn_q"),
        (1, "attn_q", "blk.1.attn_q"),
    )
    assert select_bases(inventory, "0", "All") == (
        (0, "attn_q", "blk.0.attn_q"),
        (0, "ffn_down", "blk.0.ffn_down"),
    )


def test_unpack_and_reconstruct_nanoquant_weight_with_salient_column() -> None:
    base = "blk.0.attn_q"
    left = np.array([[1, -1], [1, 1]], dtype=np.int8)
    right = np.array([[1, 1, -1], [-1, 1, 1]], dtype=np.int8)
    scale_pre = np.array([2.0, 0.0, 4.0], dtype=np.float32)
    scale_mid = np.array([0.5, 1.5], dtype=np.float32)
    scale_post = np.array([3.0, 5.0], dtype=np.float32)
    salient = np.array([[7.0], [11.0]], dtype=np.float32)
    tensors = {
        f"{base}.nq_v": _tensor(f"{base}.nq_v", _pack(right)),
        f"{base}.nq_u": _tensor(f"{base}.nq_u", _pack(left)),
        f"{base}.nq_scale_pre": _tensor(f"{base}.nq_scale_pre", scale_pre),
        f"{base}.nq_scale_mid": _tensor(f"{base}.nq_scale_mid", scale_mid),
        f"{base}.nq_scale_post": _tensor(f"{base}.nq_scale_post", scale_post),
        f"{base}.nq_salient_idx": _tensor(
            f"{base}.nq_salient_idx", np.array([1], dtype=np.int32)
        ),
        f"{base}.nq_salient_weight": _tensor(f"{base}.nq_salient_weight", salient),
    }

    assert np.array_equal(unpack_sign_words(_pack(left), 2, 2), left.astype(np.float32))
    actual = reconstruct_nanoquant_weight(tensors, base, _DirectGGUF())
    expected = (left * scale_post[:, None]) @ (
        right * scale_mid[:, None] * scale_pre[None, :]
    )
    expected[:, 1] += salient[:, 0]
    assert np.array_equal(actual, expected)


def test_grayscale_scaling_and_png_encoding(tmp_path: Path) -> None:
    pixels, minimum, maximum = scale_to_grayscale(
        np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
    )
    assert minimum == -2.0
    assert maximum == 2.0
    assert pixels.tolist() == [[0, 128, 255]]
    constant, _, _ = scale_to_grayscale(np.full((2, 3), 4.0, dtype=np.float32))
    assert not np.any(constant)

    difference_pixels, difference_minimum, difference_maximum = scale_to_grayscale(
        np.array([[-1.0, 0.0, 1.0]], dtype=np.float32),
        scaling_minimum=-2.0,
        scaling_maximum=2.0,
    )
    assert (difference_minimum, difference_maximum) == (-1.0, 1.0)
    assert difference_pixels.tolist() == [[64, 128, 191]]

    output = tmp_path / "weights.png"
    write_grayscale_png(output, pixels)
    encoded = output.read_bytes()
    assert encoded.startswith(b"\x89PNG\r\n\x1a\n")
    ihdr_length = struct.unpack(">I", encoded[8:12])[0]
    assert ihdr_length == 13
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", encoded[16:29]
    )
    assert (width, height, depth, color, compression, filtering, interlace) == (3, 1, 8, 0, 0, 0, 0)
    idat_offset = 8 + 12 + ihdr_length
    idat_length = struct.unpack(">I", encoded[idat_offset : idat_offset + 4])[0]
    assert encoded[idat_offset + 4 : idat_offset + 8] == b"IDAT"
    compressed = encoded[idat_offset + 8 : idat_offset + 8 + idat_length]
    assert zlib.decompress(compressed) == b"\x00\x00\x80\xff"


def test_diverging_difference_scaling_and_rgb_png_encoding(tmp_path: Path) -> None:
    pixels, minimum, maximum = scale_difference_to_diverging(
        np.array([[-4.0, -2.0, 0.0, 1.0, 2.0]], dtype=np.float32),
    )
    assert (minimum, maximum) == (-4.0, 2.0)
    assert pixels.tolist() == [
        [[0, 0, 255], [0, 0, 128], [0, 0, 0], [128, 0, 0], [255, 0, 0]]
    ]

    output = tmp_path / "difference.png"
    write_rgb_png(output, pixels)
    encoded = output.read_bytes()
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", encoded[16:29]
    )
    assert (width, height, depth, color, compression, filtering, interlace) == (5, 1, 8, 2, 0, 0, 0)
    ihdr_length = struct.unpack(">I", encoded[8:12])[0]
    idat_offset = 8 + 12 + ihdr_length
    idat_length = struct.unpack(">I", encoded[idat_offset : idat_offset + 4])[0]
    compressed = encoded[idat_offset + 8 : idat_offset + 8 + idat_length]
    assert zlib.decompress(compressed) == b"\x00" + pixels.tobytes()


def test_topographic_weight_scaling_uses_symmetric_multicolor_range() -> None:
    pixels, minimum, maximum, scaling_minimum, scaling_maximum, transform_scale = scale_weight_to_topographic(
        np.array([[-4.0, -2.0, 0.0, 1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    )
    assert (minimum, maximum) == (-4.0, 4.0)
    assert (scaling_minimum, scaling_maximum, transform_scale) == (-4.0, 4.0, None)
    assert pixels.tolist() == [
        [
            [0, 20, 110],
            [0, 170, 210],
            [35, 130, 55],
            [100, 180, 60],
            [225, 205, 55],
            [150, 90, 45],
            [255, 255, 255],
        ]
    ]


def test_topographic_weight_scaling_emits_distinct_percentile_and_asinh_views() -> None:
    weight = np.array([[-100.0, -2.0, -1.0, 0.0, 1.0, 2.0, 100.0]], dtype=np.float32)
    full = scale_weight_to_topographic(weight, variant="full-range")[0]
    percentile = scale_weight_to_topographic(weight, variant="percentile-01-99")[0]
    asinh = scale_weight_to_topographic(weight, variant="signed-asinh")[0]
    assert not np.array_equal(full, percentile)
    assert not np.array_equal(full, asinh)
    assert not np.array_equal(percentile, asinh)


def test_frequency_spectrum_separates_x_and_y_frequencies_into_bands() -> None:
    coordinates = np.arange(8, dtype=np.float32)
    x_wave = np.cos(2.0 * np.pi * 2.0 * coordinates / 8.0)
    y_wave = np.cos(2.0 * np.pi * 1.0 * coordinates / 8.0)
    weight = x_wave.reshape(1, -1) + y_wave.reshape(-1, 1)
    pixels, minimum, maximum = scale_frequency_spectrum(weight)
    assert pixels.shape == (8, 8, 3)
    assert pixels.dtype == np.uint8
    assert (minimum, maximum) == (0.0, 6.0)
    assert pixels[4, 4].tolist() == [0, 0, 0]
    brightest = np.all(pixels == [255, 255, 220], axis=2)
    assert np.all(brightest[:, (2, 6)])
    assert np.all(brightest[(3, 5), :])

    x_frequency, x_decibels, y_frequency, y_decibels = axis_frequency_profiles(weight)
    assert abs(float(x_frequency[int(np.argmax(x_decibels))])) == 0.25
    assert abs(float(y_frequency[int(np.argmax(y_decibels))])) == 0.125
