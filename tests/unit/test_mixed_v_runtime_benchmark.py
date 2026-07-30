from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def _benchmark_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("benchmark_mixed_v_runtime")


def test_mixed_v_records_pack_across_word_boundaries() -> None:
    benchmark = _benchmark_module()
    records = torch.tensor(
        [0, 1, (1 << 19) - 1, 17, 301_221, 42, 98_765],
        dtype=torch.int64,
    )

    packed = benchmark._pack_records(records)

    assert torch.equal(benchmark._unpack_records(packed, records.numel()), records)


def test_mixed_v_correction_pair_table_covers_every_unordered_pair() -> None:
    benchmark = _benchmark_module()
    table = benchmark._correction_pair_table()
    represented = {
        (int(value) & 0xFF, (int(value) >> 8) & 0xFF)
        for value in table[: benchmark.CORRECTION_PAIR_COUNT]
    }

    assert len(represented) == 496
    assert all(first < second < 32 for first, second in represented)
