"""Canonical llama.cpp-compatible NanoQuant sign-word packing."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nanoquant.runtime.backend import QuantizedLinearSpec
from nanoquant.runtime.logical import LogicalLayerState, canonical_torch_dtype

PACKED_LAYOUT_VERSION = "llama.cpp-i32-lsb-v1"
PACKED_WORD_BITS = 32
PACKED_WORD_DTYPE = "int32"
PACKED_TENSOR_NAMESPACE = f"layouts.{PACKED_LAYOUT_VERSION}"
PACKED_REFERENCE_REPOSITORY = "modified-llama.cpp"
PACKED_REFERENCE_COMMIT = "da52148384591f4b0d87d58c12862e30f43014f1"
PACKED_REFERENCE_DIRTY_DIFF_GIT_OBJECT = "cf463b9266db4e1f162ad8970e8ddcc1abfb5fbd"
PACKED_REFERENCE_CUDA_SHA256 = (
    "5c87336c2b6b8fb33805c6ee6a8752d4bd364beed63fd4cca03c2b36be966619"
)
PACKED_REFERENCE_CONVERTER_SHA256 = (
    "c2e1fd064bbd46f38e9e3c5f739865d198ca75bd0bb9db16f72530d378d11304"
)
PACKED_REFERENCE_DOCUMENTATION_SHA256 = (
    "12c46863a480a04b1ba449bb7bcb2f637419b677678b60f93522c531bd3f9ac8"
)
PACKED_REFERENCE_MODEL_LOADER_SHA256 = (
    "11175fca67ecd8b97f6d6ffa7d2e8b848839d768669d63f7ec629a69d8d704aa"
)
PACKED_REFERENCE_CPU_SHA256 = (
    "f51195610b4c533e4f606f984c7083bb542e4c3b3c8e740fdc647f8a5b0eff1c"
)
GGUF_TENSOR_SUFFIXES = (
    ("factor_right_words", "nq_v"),
    ("factor_left_words", "nq_u"),
    ("scale_pre", "nq_scale_pre"),
    ("scale_mid", "nq_scale_mid"),
    ("scale_post", "nq_scale_post"),
    ("outlier_indices", "nq_salient_idx"),
    ("outlier_values", "nq_salient_weight"),
    ("outlier_scales", "nq_salient_scale"),
    ("bias", "bias"),
)


@dataclass(frozen=True, slots=True)
class PackedReferenceProvenance:
    repository: str = PACKED_REFERENCE_REPOSITORY
    commit: str = PACKED_REFERENCE_COMMIT
    dirty_diff_git_object: str = PACKED_REFERENCE_DIRTY_DIFF_GIT_OBJECT
    cuda_sha256: str = PACKED_REFERENCE_CUDA_SHA256
    converter_sha256: str = PACKED_REFERENCE_CONVERTER_SHA256
    documentation_sha256: str = PACKED_REFERENCE_DOCUMENTATION_SHA256
    model_loader_sha256: str = PACKED_REFERENCE_MODEL_LOADER_SHA256
    cpu_sha256: str = PACKED_REFERENCE_CPU_SHA256

    def __post_init__(self) -> None:
        actual = (
            self.repository,
            self.commit,
            self.dirty_diff_git_object,
            self.cuda_sha256,
            self.converter_sha256,
            self.documentation_sha256,
            self.model_loader_sha256,
            self.cpu_sha256,
        )
        expected = (
            PACKED_REFERENCE_REPOSITORY,
            PACKED_REFERENCE_COMMIT,
            PACKED_REFERENCE_DIRTY_DIFF_GIT_OBJECT,
            PACKED_REFERENCE_CUDA_SHA256,
            PACKED_REFERENCE_CONVERTER_SHA256,
            PACKED_REFERENCE_DOCUMENTATION_SHA256,
            PACKED_REFERENCE_MODEL_LOADER_SHA256,
            PACKED_REFERENCE_CPU_SHA256,
        )
        if actual != expected:
            raise ValueError("packed layout reference provenance differs from schema 1")


@dataclass(frozen=True, slots=True)
class PackedLayoutMetadata:
    version: str = PACKED_LAYOUT_VERSION
    word_dtype: str = PACKED_WORD_DTYPE
    word_bits: int = PACKED_WORD_BITS
    bit_order: str = "least-significant-bit-first"
    positive_bit: int = 0
    negative_bit: int = 1
    padding_bit: int = 0
    minimum_alignment_bytes: int = 4
    vector_alignment_bytes: int = 16
    tensor_namespace: str = PACKED_TENSOR_NAMESPACE
    left_sidecar_name: str = "nq_u"
    right_sidecar_name: str = "nq_v"
    scale_pre_sidecar_name: str = "nq_scale_pre"
    scale_mid_sidecar_name: str = "nq_scale_mid"
    scale_post_sidecar_name: str = "nq_scale_post"
    outlier_index_sidecar_name: str = "nq_salient_idx"
    outlier_value_sidecar_name: str = "nq_salient_weight"
    outlier_scale_sidecar_name: str = "nq_salient_scale"
    bias_storage: str = "separate-additive-tensor"
    reference: PackedReferenceProvenance = PackedReferenceProvenance()

    def __post_init__(self) -> None:
        if self.version != PACKED_LAYOUT_VERSION:
            raise ValueError(f"unsupported packed layout version: {self.version}")
        if self.word_dtype != PACKED_WORD_DTYPE or self.word_bits != PACKED_WORD_BITS:
            raise ValueError("packed layout word representation differs from the runtime")
        if self.bit_order != "least-significant-bit-first":
            raise ValueError("packed layout bit order differs from the runtime")
        if (self.positive_bit, self.negative_bit, self.padding_bit) != (0, 1, 0):
            raise ValueError("packed layout sign or padding encoding differs from the runtime")
        if self.minimum_alignment_bytes != 4 or self.vector_alignment_bytes != 16:
            raise ValueError("packed layout alignment metadata differs from the reference kernel")
        if self.tensor_namespace != PACKED_TENSOR_NAMESPACE:
            raise ValueError("packed tensor namespace differs from the runtime")
        expected_names = dict(GGUF_TENSOR_SUFFIXES)
        actual_names = {
            "factor_left_words": self.left_sidecar_name,
            "factor_right_words": self.right_sidecar_name,
            "scale_pre": self.scale_pre_sidecar_name,
            "scale_mid": self.scale_mid_sidecar_name,
            "scale_post": self.scale_post_sidecar_name,
            "outlier_indices": self.outlier_index_sidecar_name,
            "outlier_values": self.outlier_value_sidecar_name,
            "outlier_scales": self.outlier_scale_sidecar_name,
        }
        if actual_names != {role: expected_names[role] for role in actual_names}:
            raise ValueError("packed GGUF sidecar names differ from the reference format")
        if self.bias_storage != "separate-additive-tensor":
            raise ValueError("packed bias storage differs from the runtime")
        if self.reference != PackedReferenceProvenance():
            raise ValueError("packed layout reference provenance differs from the runtime")


def packed_word_count(columns: int) -> int:
    if columns <= 0:
        raise ValueError("packed sign column count must be positive")
    return (columns + PACKED_WORD_BITS - 1) // PACKED_WORD_BITS


def packed_row_stride_bytes(columns: int) -> int:
    return packed_word_count(columns) * torch.int32.itemsize


def supports_vector_word_loads(words: torch.Tensor) -> bool:
    """Match llama.cpp's optional aligned uint4 sign-word load predicate."""

    return (
        words.ndim == 2
        and words.dtype == torch.int32
        and words.is_contiguous()
        and words.data_ptr() % 16 == 0
        and words.stride(0) * words.element_size() % 16 == 0
    )


def _validate_sign_matrix(signs: torch.Tensor) -> None:
    if signs.ndim != 2 or not signs.is_contiguous():
        raise ValueError("logical signs must be a contiguous matrix")
    if signs.shape[0] <= 0 or signs.shape[1] <= 0:
        raise ValueError("logical sign matrix dimensions must be positive")
    if not bool(torch.all((signs == 1) | (signs == -1))):
        raise ValueError("logical signs contain a value other than -1 or +1")


def pack_sign_matrix(signs: torch.Tensor) -> torch.Tensor:
    """Pack rows into I32 words matching llama.cpp's `col / 32`, `col % 32` lookup."""

    _validate_sign_matrix(signs)
    rows, columns = (int(value) for value in signs.shape)
    words = packed_word_count(columns)
    padded_columns = words * PACKED_WORD_BITS
    negative = signs < 0
    if padded_columns != columns:
        negative = torch.nn.functional.pad(negative, (0, padded_columns - columns), value=False)
    lanes = torch.arange(PACKED_WORD_BITS, dtype=torch.int64, device=signs.device)
    weights = torch.bitwise_left_shift(torch.ones_like(lanes), lanes)
    packed = torch.sum(
        negative.reshape(rows, words, PACKED_WORD_BITS).to(torch.int64) * weights,
        dim=-1,
    )
    return packed.to(dtype=torch.int32, device="cpu").contiguous()


def _validate_packed_matrix(words: torch.Tensor, rows: int, columns: int) -> None:
    expected = (rows, packed_word_count(columns))
    if words.dtype != torch.int32 or tuple(words.shape) != expected or not words.is_contiguous():
        raise ValueError(
            f"packed sign matrix must be contiguous int32 with shape {expected}, got "
            f"{words.dtype} {tuple(words.shape)}"
        )
    tail = columns % PACKED_WORD_BITS
    if tail:
        valid_mask = (1 << tail) - 1
        last = torch.bitwise_and(words[:, -1].to(torch.int64), 0xFFFFFFFF)
        if bool(torch.any(torch.bitwise_and(last, 0xFFFFFFFF ^ valid_mask) != 0)):
            raise ValueError("packed sign matrix has a non-zero padding bit")


def unpack_sign_matrix(
    words: torch.Tensor,
    rows: int,
    columns: int,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unpack canonical I32 sign words to a dense -1/+1 matrix."""

    _validate_packed_matrix(words, rows, columns)
    lanes = torch.arange(PACKED_WORD_BITS, dtype=torch.int64, device=words.device)
    unsigned = torch.bitwise_and(words.to(torch.int64), 0xFFFFFFFF)
    bits = torch.bitwise_and(torch.bitwise_right_shift(unsigned.unsqueeze(-1), lanes), 1)
    signs = 1 - 2 * bits.reshape(rows, -1)[:, :columns]
    return signs.to(dtype=dtype).contiguous()


def _factor_dtype(name: str) -> torch.dtype:
    try:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"packed layout does not support logical factor dtype: {name}") from error


@dataclass(frozen=True, slots=True)
class PackedLayerState:
    spec: QuantizedLinearSpec
    layout: str
    left_words: torch.Tensor
    right_words: torch.Tensor
    scale_pre: torch.Tensor
    scale_mid: torch.Tensor
    scale_post: torch.Tensor
    bias: torch.Tensor | None = None
    outlier_indices: torch.Tensor | None = None
    outlier_values: torch.Tensor | None = None
    outlier_scales: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.layout != PACKED_LAYOUT_VERSION:
            raise ValueError(f"unsupported packed layer layout: {self.layout}")
        _validate_packed_matrix(
            self.left_words,
            self.spec.out_features,
            self.spec.rank,
        )
        _validate_packed_matrix(
            self.right_words,
            self.spec.rank,
            self.spec.in_features,
        )
        expected_shapes = (
            ("scale_pre", self.scale_pre, (self.spec.in_features,)),
            ("scale_mid", self.scale_mid, (self.spec.rank,)),
            ("scale_post", self.scale_post, (self.spec.out_features,)),
        )
        for name, value, shape in expected_shapes:
            if tuple(value.shape) != shape or not value.is_contiguous():
                raise ValueError(f"packed layer {name} shape or layout differs")
            if canonical_torch_dtype(value.dtype) != self.spec.scale_dtype:
                raise ValueError(f"packed layer {name} dtype differs")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"packed layer {name} contains a non-finite value")
        if (self.bias is None) != (not self.spec.has_bias):
            raise ValueError("packed layer bias presence differs from its specification")
        if self.bias is not None:
            if tuple(self.bias.shape) != (self.spec.out_features,) or not self.bias.is_contiguous():
                raise ValueError("packed layer bias shape differs")
            if canonical_torch_dtype(self.bias.dtype) != self.spec.scale_dtype:
                raise ValueError("packed layer bias dtype differs")
            if not bool(torch.all(torch.isfinite(self.bias))):
                raise ValueError("packed layer bias contains a non-finite value")
        if (self.outlier_indices is None) != (self.outlier_values is None):
            raise ValueError("packed layer outlier indices and values must be paired")
        if (self.outlier_indices is None) != (self.spec.outlier_count == 0):
            raise ValueError("packed layer outlier presence differs from its specification")
        if self.outlier_indices is not None and self.outlier_values is not None:
            if self.outlier_indices.dtype != torch.int32:
                raise ValueError("packed llama.cpp salient indices must be int32")
            if tuple(self.outlier_indices.shape) != (
                self.spec.outlier_count,
            ) or not self.outlier_indices.is_contiguous():
                raise ValueError("packed layer outlier index shape differs")
            if tuple(self.outlier_values.shape) != (
                self.spec.out_features,
                self.spec.outlier_count,
            ) or not self.outlier_values.is_contiguous():
                raise ValueError("packed layer outlier value shape differs")
            if canonical_torch_dtype(self.outlier_values.dtype) != self.spec.outlier_value_dtype:
                raise ValueError("packed layer outlier value dtype differs")
            indexes = self.outlier_indices.to(torch.int64)
            if bool(torch.any(indexes < 0)) or bool(torch.any(indexes >= self.spec.in_features)):
                raise ValueError("packed layer outlier index is outside the input dimension")
            if not bool(torch.all(indexes[1:] > indexes[:-1])):
                raise ValueError("packed layer outlier indices must be strictly increasing")
            salient_pre = self.scale_pre.index_select(0, indexes.to(self.scale_pre.device))
            if bool(torch.any(salient_pre != 0)):
                raise ValueError(
                    "packed llama.cpp salient columns require exactly zero scale_pre entries"
                )
            if self.outlier_values.dtype not in (
                torch.int8,
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ):
                raise ValueError("packed llama.cpp salient value dtype is unsupported")
            if self.outlier_values.is_floating_point() and not bool(
                torch.all(torch.isfinite(self.outlier_values))
            ):
                raise ValueError("packed layer outlier value contains a non-finite value")
        if (self.outlier_scales is not None) != self.spec.has_outlier_scales:
            raise ValueError("packed layer outlier scale presence differs")
        if self.outlier_scales is not None:
            if self.outlier_values is None or self.outlier_values.dtype != torch.int8:
                raise ValueError("packed llama.cpp salient scales require int8 values")
            if tuple(self.outlier_scales.shape) != (
                self.spec.outlier_count,
            ) or not self.outlier_scales.is_contiguous():
                raise ValueError("packed llama.cpp salient scales must have one value per column")
            if canonical_torch_dtype(self.outlier_scales.dtype) != self.spec.scale_dtype:
                raise ValueError("packed layer outlier scale dtype differs")
            if not bool(torch.all(torch.isfinite(self.outlier_scales))):
                raise ValueError("packed layer outlier scale contains a non-finite value")
        if self.outlier_values is not None and self.outlier_values.dtype == torch.int8:
            if self.outlier_scales is None:
                raise ValueError("packed llama.cpp int8 salient values require scales")

    def to_logical(self) -> LogicalLayerState:
        factors = {
            "left": unpack_sign_matrix(
                self.left_words,
                self.spec.out_features,
                self.spec.rank,
                dtype=_factor_dtype(self.spec.factor_dtype),
            ),
            "right": unpack_sign_matrix(
                self.right_words,
                self.spec.rank,
                self.spec.in_features,
                dtype=_factor_dtype(self.spec.factor_dtype),
            ),
        }
        return LogicalLayerState(
            self.spec,
            factors["left"],
            factors["right"],
            self.scale_pre,
            self.scale_mid,
            self.scale_post,
            self.bias,
            self.outlier_indices,
            self.outlier_values,
            self.outlier_scales,
        )


def _owned(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().cpu().clone().contiguous()


def _owned_required(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().clone().contiguous()


def pack_logical_layer(state: LogicalLayerState) -> PackedLayerState:
    indices = _owned(state.outlier_indices)
    if indices is not None:
        indices = indices.to(torch.int32)
    return PackedLayerState(
        state.spec,
        PACKED_LAYOUT_VERSION,
        pack_sign_matrix(state.left_binary),
        pack_sign_matrix(state.right_binary),
        _owned_required(state.scale_pre),
        _owned_required(state.scale_mid),
        _owned_required(state.scale_post),
        _owned(state.bias),
        indices,
        _owned(state.outlier_values),
        _owned(state.outlier_scales),
    )
