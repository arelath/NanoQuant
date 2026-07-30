# mypy: ignore-errors
"""Benchmark exact packed mixed-V decoding against the current packed kernel."""

from __future__ import annotations

import argparse
import math
import platform
from pathlib import Path
from typing import Any

import _paths  # noqa: F401
import torch
import triton
import triton.language as tl

from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.runtime.benchmark import benchmark_wall
from nanoquant.runtime.cuda_kernels import (
    _nanoquant_stage2,
    launch_packed_linear,
)

OUT_FEATURES = 1152
IN_FEATURES = 6912
BASELINE_RANK = 970
MIXED_RANK = 1344
FREE_RIGHT_ROWS = 256
INDEX_BITS = 10
CORRECTION_BITS = 9
CODED_PAYLOAD_BITS = INDEX_BITS + CORRECTION_BITS
WORD_BITS = 32
WORDS_PER_RIGHT_ROW = IN_FEATURES // WORD_BITS
CORRECTION_PAIR_COUNT = math.comb(WORD_BITS, 2)


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "token counts must be unique positive integers"
        )
    return result


def _random_words(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randint(
        0,
        2**32,
        shape,
        generator=generator,
        dtype=torch.int64,
    ).to(torch.int32)


def _correction_pair_table() -> torch.Tensor:
    pairs = tuple(
        (first, second)
        for first in range(WORD_BITS)
        for second in range(first + 1, WORD_BITS)
    )
    if len(pairs) != CORRECTION_PAIR_COUNT:
        raise AssertionError("correction-pair inventory differs")
    table = torch.zeros((1 << CORRECTION_BITS,), dtype=torch.int32)
    for index, (first, second) in enumerate(pairs):
        table[index] = first | (second << 8)
    return table


def _pack_records(records: torch.Tensor) -> torch.Tensor:
    """Pack row-major unsigned 19-bit records with one safe sentinel word."""

    flattened = records.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    if flattened.numel() == 0 or bool(
        torch.any((flattened < 0) | (flattened >= (1 << CODED_PAYLOAD_BITS)))
    ):
        raise ValueError("mixed-V records must be non-empty unsigned 19-bit values")
    positions = torch.arange(flattened.numel(), dtype=torch.int64)
    bit_offsets = positions * CODED_PAYLOAD_BITS
    word_indices = bit_offsets // WORD_BITS
    shifts = bit_offsets & (WORD_BITS - 1)
    word_count = math.ceil(flattened.numel() * CODED_PAYLOAD_BITS / WORD_BITS)
    packed = torch.zeros((word_count + 1,), dtype=torch.int64)
    packed.index_add_(
        0,
        word_indices,
        (flattened << shifts) & 0xFFFF_FFFF,
    )
    crosses = shifts > WORD_BITS - CODED_PAYLOAD_BITS
    packed.index_add_(
        0,
        word_indices[crosses] + 1,
        flattened[crosses] >> (WORD_BITS - shifts[crosses]),
    )
    if bool(torch.any(packed > 0xFFFF_FFFF)):
        raise AssertionError("mixed-V record packing produced overlapping fields")
    return packed.to(torch.int32)


def _unpack_records(packed: torch.Tensor, count: int) -> torch.Tensor:
    if packed.ndim != 1 or count <= 0:
        raise ValueError("mixed-V packed record request is invalid")
    unsigned = packed.detach().to(device="cpu", dtype=torch.int64) & 0xFFFF_FFFF
    positions = torch.arange(count, dtype=torch.int64)
    bit_offsets = positions * CODED_PAYLOAD_BITS
    word_indices = bit_offsets // WORD_BITS
    shifts = bit_offsets & (WORD_BITS - 1)
    result = unsigned.index_select(0, word_indices) >> shifts
    crosses = shifts > WORD_BITS - CODED_PAYLOAD_BITS
    result[crosses] |= (
        unsigned.index_select(0, word_indices[crosses] + 1)
        << (WORD_BITS - shifts[crosses])
    )
    return result & ((1 << CODED_PAYLOAD_BITS) - 1)


def _corrected_words(
    codebook_words: torch.Tensor,
    indices: torch.Tensor,
    pair_ids: torch.Tensor,
    correction_pairs: torch.Tensor,
) -> torch.Tensor:
    flattened_indices = indices.reshape(-1).long()
    pairs = correction_pairs.index_select(0, pair_ids.reshape(-1).long()).long()
    first = pairs & 0xFF
    second = (pairs >> 8) & 0xFF
    masks = (torch.ones_like(first) << first) | (torch.ones_like(second) << second)
    base = codebook_words.index_select(0, flattened_indices).long() & 0xFFFF_FFFF
    return (base ^ masks).to(torch.int32).reshape_as(indices)


@triton.jit
def _mixed_v_stage1(
    value,
    free_right_words,
    coded_payload,
    codebook_words,
    correction_pairs,
    scale_pre,
    scale_mid,
    latent,
    N_IN: tl.constexpr,
    N_RANK: tl.constexpr,
    FREE_ROWS: tl.constexpr,
    WORDS_PER_ROW: tl.constexpr,
    WORD_WIDTH: tl.constexpr,
    PAYLOAD_WIDTH: tl.constexpr,
    INDEX_WIDTH: tl.constexpr,
    BLOCK_IN: tl.constexpr,
    BLOCK_RANK: tl.constexpr,
):
    rank_block = tl.program_id(0)
    token = tl.program_id(1)
    ranks = rank_block * BLOCK_RANK + tl.arange(0, BLOCK_RANK)
    rank_mask = ranks < N_RANK
    free_mask = ranks < FREE_ROWS
    accumulator = tl.zeros((BLOCK_RANK,), dtype=tl.float32)
    for start in range(0, N_IN, BLOCK_IN):
        columns = start + tl.arange(0, BLOCK_IN)
        column_mask = columns < N_IN
        word_columns = columns // WORD_WIDTH
        free_words = tl.load(
            free_right_words
            + ranks[:, None] * WORDS_PER_ROW
            + word_columns[None, :],
            mask=rank_mask[:, None]
            & free_mask[:, None]
            & column_mask[None, :],
            other=0,
        ).to(tl.uint32)

        coded_ranks = ranks - FREE_ROWS
        records = coded_ranks[:, None] * WORDS_PER_ROW + word_columns[None, :]
        bit_offsets = records * PAYLOAD_WIDTH
        payload_words = bit_offsets // WORD_WIDTH
        shifts = bit_offsets & (WORD_WIDTH - 1)
        coded_mask = (
            rank_mask[:, None]
            & (~free_mask[:, None])
            & column_mask[None, :]
        )
        low = tl.load(
            coded_payload + payload_words,
            mask=coded_mask,
            other=0,
        ).to(tl.uint32)
        crosses = shifts > WORD_WIDTH - PAYLOAD_WIDTH
        high = tl.load(
            coded_payload + payload_words + 1,
            mask=coded_mask & crosses,
            other=0,
        ).to(tl.uint32)
        packed = low >> shifts
        packed |= tl.where(
            crosses,
            high << ((WORD_WIDTH - shifts) & (WORD_WIDTH - 1)),
            0,
        )
        packed &= (1 << PAYLOAD_WIDTH) - 1
        indices = packed & ((1 << INDEX_WIDTH) - 1)
        pair_ids = packed >> INDEX_WIDTH
        coded_words = tl.load(
            codebook_words + indices,
            mask=coded_mask,
            other=0,
        ).to(tl.uint32)
        pairs = tl.load(
            correction_pairs + pair_ids,
            mask=coded_mask,
            other=0,
        ).to(tl.uint32)
        first = pairs & 0xFF
        second = (pairs >> 8) & 0xFF
        words = tl.where(free_mask[:, None], free_words, coded_words)
        positions = columns[None, :] & (WORD_WIDTH - 1)
        bits = (words >> positions) & 1
        corrected = (~free_mask[:, None]) & (
            (positions == first) | (positions == second)
        )
        bits = tl.where(corrected, 1 - bits, bits)
        signs = 1.0 - 2.0 * bits.to(tl.float32)
        inputs = tl.load(
            value + token * N_IN + columns,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        pre = tl.load(
            scale_pre + columns,
            mask=column_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.sum(signs * (inputs * pre)[None, :], axis=1)
    mid = tl.load(
        scale_mid + ranks,
        mask=rank_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        latent + token * N_RANK + ranks,
        accumulator * mid,
        mask=rank_mask,
    )


@triton.jit
def _decode_mixed_v_words(
    free_right_words,
    coded_payload,
    codebook_words,
    correction_pairs,
    output_words,
    TOTAL_WORDS: tl.constexpr,
    FREE_ROWS: tl.constexpr,
    WORDS_PER_ROW: tl.constexpr,
    WORD_WIDTH: tl.constexpr,
    PAYLOAD_WIDTH: tl.constexpr,
    INDEX_WIDTH: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    output_mask = offsets < TOTAL_WORDS
    ranks = offsets // WORDS_PER_ROW
    word_columns = offsets - ranks * WORDS_PER_ROW
    free_mask = ranks < FREE_ROWS
    free_words = tl.load(
        free_right_words + ranks * WORDS_PER_ROW + word_columns,
        mask=output_mask & free_mask,
        other=0,
    ).to(tl.uint32)

    records = (ranks - FREE_ROWS) * WORDS_PER_ROW + word_columns
    bit_offsets = records * PAYLOAD_WIDTH
    payload_words = bit_offsets // WORD_WIDTH
    shifts = bit_offsets & (WORD_WIDTH - 1)
    coded_mask = output_mask & (~free_mask)
    low = tl.load(
        coded_payload + payload_words,
        mask=coded_mask,
        other=0,
    ).to(tl.uint32)
    crosses = shifts > WORD_WIDTH - PAYLOAD_WIDTH
    high = tl.load(
        coded_payload + payload_words + 1,
        mask=coded_mask & crosses,
        other=0,
    ).to(tl.uint32)
    packed = low >> shifts
    packed |= tl.where(
        crosses,
        high << ((WORD_WIDTH - shifts) & (WORD_WIDTH - 1)),
        0,
    )
    packed &= (1 << PAYLOAD_WIDTH) - 1
    indices = packed & ((1 << INDEX_WIDTH) - 1)
    pair_ids = packed >> INDEX_WIDTH
    code_words = tl.load(
        codebook_words + indices,
        mask=coded_mask,
        other=0,
    ).to(tl.uint32)
    pairs = tl.load(
        correction_pairs + pair_ids,
        mask=coded_mask,
        other=0,
    ).to(tl.uint32)
    first = pairs & 0xFF
    second = (pairs >> 8) & 0xFF
    correction_mask = (1 << first) | (1 << second)
    corrected_words = code_words ^ correction_mask
    tl.store(
        output_words + offsets,
        tl.where(free_mask, free_words, corrected_words),
        mask=output_mask,
    )


def _predecode_mixed_right_words(
    free_right_words: torch.Tensor,
    coded_payload: torch.Tensor,
    codebook_words: torch.Tensor,
    correction_pairs: torch.Tensor,
) -> torch.Tensor:
    total_words = MIXED_RANK * WORDS_PER_RIGHT_ROW
    output = torch.empty(
        (MIXED_RANK, WORDS_PER_RIGHT_ROW),
        dtype=torch.int32,
        device=free_right_words.device,
    )
    _decode_mixed_v_words[(triton.cdiv(total_words, 256),)](
        free_right_words,
        coded_payload,
        codebook_words,
        correction_pairs,
        output,
        TOTAL_WORDS=total_words,
        FREE_ROWS=FREE_RIGHT_ROWS,
        WORDS_PER_ROW=WORDS_PER_RIGHT_ROW,
        WORD_WIDTH=WORD_BITS,
        PAYLOAD_WIDTH=CODED_PAYLOAD_BITS,
        INDEX_WIDTH=INDEX_BITS,
        BLOCK_WORDS=256,
        num_warps=4,
    )
    return output


def _launch_mixed_linear(
    value: torch.Tensor,
    left_words: torch.Tensor,
    free_right_words: torch.Tensor,
    coded_payload: torch.Tensor,
    codebook_words: torch.Tensor,
    correction_pairs: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
) -> torch.Tensor:
    token_count = value.numel() // IN_FEATURES
    flattened = value.view(token_count, IN_FEATURES)
    latent = torch.empty(
        (token_count, MIXED_RANK),
        dtype=torch.float32,
        device=value.device,
    )
    output = torch.empty(
        (token_count, OUT_FEATURES),
        dtype=torch.float32,
        device=value.device,
    )
    _mixed_v_stage1[(triton.cdiv(MIXED_RANK, 8), token_count)](
        flattened,
        free_right_words,
        coded_payload,
        codebook_words,
        correction_pairs,
        scale_pre,
        scale_mid,
        latent,
        N_IN=IN_FEATURES,
        N_RANK=MIXED_RANK,
        FREE_ROWS=FREE_RIGHT_ROWS,
        WORDS_PER_ROW=WORDS_PER_RIGHT_ROW,
        WORD_WIDTH=WORD_BITS,
        PAYLOAD_WIDTH=CODED_PAYLOAD_BITS,
        INDEX_WIDTH=INDEX_BITS,
        BLOCK_IN=256,
        BLOCK_RANK=8,
        num_warps=4,
    )
    _nanoquant_stage2[(triton.cdiv(OUT_FEATURES, 8), token_count)](
        flattened,
        latent,
        left_words,
        scale_post,
        scale_pre,
        scale_pre,
        scale_pre,
        scale_pre,
        output,
        N_IN=IN_FEATURES,
        N_OUT=OUT_FEATURES,
        N_RANK=MIXED_RANK,
        N_SALIENT=0,
        WORDS_PER_ROW=left_words.shape[1],
        HAS_SALIENT_SCALES=False,
        HAS_BIAS=False,
        BLOCK_OUT=8,
        BLOCK_RANK=256,
        BLOCK_SALIENT=32,
        num_warps=4,
    )
    return output.view(*value.shape[:-1], OUT_FEATURES)


def _tensor_bytes(*values: torch.Tensor) -> int:
    return sum(value.numel() * value.element_size() for value in values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-counts", type=_parse_ints, default=(1, 16, 128))
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def run(args: argparse.Namespace) -> int:
    if (
        not args.device.startswith("cuda")
        or args.warmups < 1
        or args.repetitions < 3
    ):
        raise ValueError("mixed-V benchmark requires CUDA and bounded timing")
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    coded_rows = MIXED_RANK - FREE_RIGHT_ROWS
    codebook_words_cpu = _random_words((1 << INDEX_BITS,), generator)
    free_right_words_cpu = _random_words(
        (FREE_RIGHT_ROWS, WORDS_PER_RIGHT_ROW),
        generator,
    )
    indices = torch.randint(
        0,
        1 << INDEX_BITS,
        (coded_rows, WORDS_PER_RIGHT_ROW),
        generator=generator,
        dtype=torch.int64,
    )
    pair_ids = torch.randint(
        0,
        CORRECTION_PAIR_COUNT,
        (coded_rows, WORDS_PER_RIGHT_ROW),
        generator=generator,
        dtype=torch.int64,
    )
    correction_pairs_cpu = _correction_pair_table()
    records = indices | (pair_ids << INDEX_BITS)
    coded_payload_cpu = _pack_records(records)
    if not torch.equal(
        _unpack_records(coded_payload_cpu, records.numel()),
        records.reshape(-1),
    ):
        raise AssertionError("mixed-V record packing does not round trip")
    corrected_words_cpu = _corrected_words(
        codebook_words_cpu,
        indices,
        pair_ids,
        correction_pairs_cpu,
    )
    right_words_mixed_expanded_cpu = torch.cat(
        (free_right_words_cpu, corrected_words_cpu),
        dim=0,
    )
    left_words_baseline_cpu = _random_words(
        (OUT_FEATURES, math.ceil(BASELINE_RANK / WORD_BITS)),
        generator,
    )
    right_words_baseline_cpu = _random_words(
        (BASELINE_RANK, WORDS_PER_RIGHT_ROW),
        generator,
    )
    left_words_mixed_cpu = _random_words(
        (OUT_FEATURES, MIXED_RANK // WORD_BITS),
        generator,
    )

    with acquire_device_lease(args.device), torch.inference_mode():
        left_words_baseline = left_words_baseline_cpu.to(device)
        right_words_baseline = right_words_baseline_cpu.to(device)
        left_words_mixed = left_words_mixed_cpu.to(device)
        right_words_mixed_expanded = right_words_mixed_expanded_cpu.to(device)
        free_right_words = free_right_words_cpu.to(device)
        coded_payload = coded_payload_cpu.to(device)
        codebook_words = codebook_words_cpu.to(device)
        correction_pairs = correction_pairs_cpu.to(device)
        baseline_pre = torch.ones((IN_FEATURES,), dtype=torch.float32, device=device)
        baseline_mid = torch.ones((BASELINE_RANK,), dtype=torch.float32, device=device)
        mixed_mid = torch.ones((MIXED_RANK,), dtype=torch.float32, device=device)
        post = torch.ones((OUT_FEATURES,), dtype=torch.float32, device=device)

        check_input = torch.randn(
            (2, IN_FEATURES),
            generator=torch.Generator(device=device).manual_seed(args.seed + 1),
            dtype=torch.float32,
            device=device,
        )
        expected = launch_packed_linear(
            check_input,
            left_words_mixed,
            right_words_mixed_expanded,
            baseline_pre,
            mixed_mid,
            post,
            None,
            None,
            None,
            None,
        )
        actual = _launch_mixed_linear(
            check_input,
            left_words_mixed,
            free_right_words,
            coded_payload,
            codebook_words,
            correction_pairs,
            baseline_pre,
            mixed_mid,
            post,
        )
        torch.cuda.synchronize(device)
        maximum_absolute_error = float((expected - actual).abs().max())
        if maximum_absolute_error != 0.0:
            raise AssertionError(
                "mixed-V packed kernel differs from expanded packed execution: "
                f"{maximum_absolute_error}"
            )
        predecoded = _predecode_mixed_right_words(
            free_right_words,
            coded_payload,
            codebook_words,
            correction_pairs,
        )
        torch.cuda.synchronize(device)
        if not torch.equal(predecoded, right_words_mixed_expanded):
            raise AssertionError(
                "one-time mixed-V predecode differs from expanded words"
            )

        def predecode() -> torch.Tensor:
            return _predecode_mixed_right_words(
                free_right_words,
                coded_payload,
                codebook_words,
                correction_pairs,
            )

        def synchronize() -> None:
            torch.cuda.synchronize(device)

        predecode_timing = benchmark_wall(
            predecode,
            warmups=args.warmups,
            repetitions=args.repetitions,
            synchronize=synchronize,
            unit_name="layers",
            units_per_sample=1,
        ).as_dict()

        cases: dict[str, Any] = {}
        for token_count in args.token_counts:
            value = torch.randn(
                (token_count, IN_FEATURES),
                generator=torch.Generator(device=device).manual_seed(
                    args.seed + token_count + 17
                ),
                dtype=torch.float32,
                device=device,
            )

            def baseline(input_value: torch.Tensor = value) -> torch.Tensor:
                return launch_packed_linear(
                    input_value,
                    left_words_baseline,
                    right_words_baseline,
                    baseline_pre,
                    baseline_mid,
                    post,
                    None,
                    None,
                    None,
                    None,
                )

            def expanded(input_value: torch.Tensor = value) -> torch.Tensor:
                return launch_packed_linear(
                    input_value,
                    left_words_mixed,
                    right_words_mixed_expanded,
                    baseline_pre,
                    mixed_mid,
                    post,
                    None,
                    None,
                    None,
                    None,
                )

            def mixed(input_value: torch.Tensor = value) -> torch.Tensor:
                return _launch_mixed_linear(
                    input_value,
                    left_words_mixed,
                    free_right_words,
                    coded_payload,
                    codebook_words,
                    correction_pairs,
                    baseline_pre,
                    mixed_mid,
                    post,
                )

            timings = {
                "rank970_packed": benchmark_wall(
                    baseline,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    synchronize=synchronize,
                    unit_name="tokens",
                    units_per_sample=token_count,
                ).as_dict(),
                "rank1344_expanded_packed": benchmark_wall(
                    expanded,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    synchronize=synchronize,
                    unit_name="tokens",
                    units_per_sample=token_count,
                ).as_dict(),
                "rank1344_mixed_packed": benchmark_wall(
                    mixed,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                    synchronize=synchronize,
                    unit_name="tokens",
                    units_per_sample=token_count,
                ).as_dict(),
            }
            baseline_median = timings["rank970_packed"]["latency_seconds"]["p50"]
            expanded_median = timings["rank1344_expanded_packed"][
                "latency_seconds"
            ]["p50"]
            mixed_median = timings["rank1344_mixed_packed"]["latency_seconds"][
                "p50"
            ]
            cases[str(token_count)] = {
                "timings": timings,
                "median_latency_ratios": {
                    "expanded_to_rank970": expanded_median / baseline_median,
                    "mixed_to_rank970": mixed_median / baseline_median,
                    "mixed_to_expanded": mixed_median / expanded_median,
                    "five_of_26_predecoded_hybrid_to_all_rank970": (
                        5 * expanded_median + 21 * baseline_median
                    )
                    / (26 * baseline_median),
                    "five_of_26_hybrid_to_all_rank970": (
                        5 * mixed_median + 21 * baseline_median
                    )
                    / (26 * baseline_median),
                    "all_mixed_to_all_rank970": mixed_median / baseline_median,
                },
            }

        baseline_runtime_bytes = _tensor_bytes(
            left_words_baseline,
            right_words_baseline,
            baseline_pre,
            baseline_mid,
            post,
        )
        expanded_runtime_bytes = _tensor_bytes(
            left_words_mixed,
            right_words_mixed_expanded,
            baseline_pre,
            mixed_mid,
            post,
        )
        mixed_runtime_bytes = _tensor_bytes(
            left_words_mixed,
            free_right_words,
            coded_payload,
            codebook_words,
            baseline_pre,
            mixed_mid,
            post,
        )
        shared_lookup_bytes = _tensor_bytes(correction_pairs)

    payload = {
        "schema_version": 1,
        "status": "completed",
        "role": "analysis-only exact packed mixed-V runtime microbenchmark",
        "protocol": {
            "out_features": OUT_FEATURES,
            "in_features": IN_FEATURES,
            "baseline_rank": BASELINE_RANK,
            "mixed_rank": MIXED_RANK,
            "free_right_rows": FREE_RIGHT_ROWS,
            "index_bits": INDEX_BITS,
            "correction_bits": CORRECTION_BITS,
            "coded_payload_bits": CODED_PAYLOAD_BITS,
            "token_counts": list(args.token_counts),
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "device": args.device,
        },
        "environment": {
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "triton_version": triton.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
        },
        "correctness": {
            "expanded_vs_mixed_maximum_absolute_error": maximum_absolute_error,
            "one_time_predecode_matches_expanded_words": True,
            "record_round_trip": True,
        },
        "one_time_predecode": predecode_timing,
        "resident_bytes": {
            "rank970_packed": baseline_runtime_bytes,
            "rank1344_expanded_packed": expanded_runtime_bytes,
            "rank1344_mixed_packed": mixed_runtime_bytes,
            "shared_correction_lookup": shared_lookup_bytes,
            "mixed_to_rank970": mixed_runtime_bytes / baseline_runtime_bytes,
            "expanded_to_rank970": expanded_runtime_bytes
            / baseline_runtime_bytes,
        },
        "cases": cases,
    }
    atomic_write_json(args.output, payload)
    print(
        {
            token_count: case["median_latency_ratios"]
            for token_count, case in cases.items()
        }
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
