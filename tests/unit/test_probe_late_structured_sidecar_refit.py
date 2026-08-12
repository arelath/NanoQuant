import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from probe_late_structured_sidecar_refit import (  # noqa: E402
    Shape,
    _count_for_budget,
    _parse_shapes,
)

from nanoquant.domain.structured_sidecar import whole_column_int8_cost  # noqa: E402


def test_narrow_shapes_are_matched_below_one_column_budget() -> None:
    out_features, in_features = 1152, 6912
    budget = whole_column_int8_cost(out_features, 1, in_features).total
    for shape in _parse_shapes("16x1,32x1,64x1,8x2,4x4,1x32"):
        count, cost = _count_for_budget(shape, budget, out_features, in_features)
        assert count > 0
        assert cost.total <= budget


def test_column_control_uses_exact_budget() -> None:
    budget = whole_column_int8_cost(32, 1, 16).total
    count, cost = _count_for_budget(Shape("column"), budget, 32, 16)
    assert count == 1
    assert cost.total == budget


def test_shape_parser_interprets_32_by_1_as_column_segment() -> None:
    shape = _parse_shapes("32x1")[0]
    assert (shape.rows, shape.columns) == (32, 1)
    assert torch.tensor([shape.rows, shape.columns]).tolist() == [32, 1]


def test_control_is_reserved_for_the_no_new_sidecar_overlay() -> None:
    assert all(shape.name != "control" for shape in _parse_shapes("column,16x1"))
