from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("gguf")


def _tool():
    tools = Path(__file__).resolve().parents[2] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location("probe_byte_budget_allocation", tools /
                                                  "probe_byte_budget_allocation.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_estimated_kl_is_quadratic_below_the_measured_range() -> None:
    tool = _tool()
    curve = {"coefficient": 2.0, "exponent": 1.5, "measured_error_range": [0.05, 0.35]}
    assert tool.estimated_kl(curve, 0.2) == pytest.approx(2.0 * 0.2 ** 1.5)
    assert tool.estimated_kl(curve, 0.025) == pytest.approx(2.0 * 0.05 ** 1.5 / 4)


def test_efficient_drops_dominated_options_and_those_above_the_hull() -> None:
    tool = _tool()
    options = [tool.Option("a", 1, 10), tool.Option("dominated", 2, 11), tool.Option("above", 2, 9),
               tool.Option("b", 3, 2), tool.Option("c", 6, 1)]
    assert [option.label for option in tool.efficient(options)] == ["a", "b", "c"]


def test_allocate_spends_bytes_where_they_save_the_most_kl() -> None:
    tool = _tool()
    units = {
        "sensitive": [tool.Option("small", 1, 10), tool.Option("large", 3, 1)],
        "tolerant": [tool.Option("small", 1, 1), tool.Option("large", 3, 0.5)],
    }
    assert {name: option.label for name, option in tool.allocate(units, 4).items()} == {
        "sensitive": "large", "tolerant": "small"}
    assert {option.label for option in tool.allocate(units, 6).values()} == {"large"}
    assert {option.label for option in tool.allocate(units, 2).values()} == {"small"}
    with pytest.raises(ValueError, match="over the budget"):
        tool.allocate(units, 1)


def test_unit_options_mix_member_formats_and_extend_a_third_stage() -> None:
    tool = _tool()
    entry = {
        "members": ["blk.0.attn_q.weight", "blk.0.attn_k.weight"],
        "staged": [{"bytes": 10, "members": [{"kl": 4.0}, {"kl": 1.0}]},
                   {"bytes": 20, "members": [{"kl": 1.0}, {"kl": 0.5}]}],
        "standard": {"blk.0.attn_q.weight": {"Q8_0": {"bytes": 40, "kl": 0.1}, "Q4_0": {"bytes": 20, "kl": 0.4}},
                     "blk.0.attn_k.weight": {"Q8_0": {"bytes": 8, "kl": 0.0}}},
    }
    options = {option.label: option for option in tool.unit_options(entry, third_stage=True)}
    assert options["2 stages"].kl == pytest.approx(1.5)
    assert options["3 stages (assumed)"].bytes == 30
    assert options["3 stages (assumed)"].kl == pytest.approx(0.25 + 0.25)
    assert options["Q4_0 Q8_0"].bytes == 28 and options["Q4_0 Q8_0"].kl == pytest.approx(0.4)
    assert "3 stages (assumed)" not in {o.label for o in tool.unit_options(entry, third_stage=False)}
