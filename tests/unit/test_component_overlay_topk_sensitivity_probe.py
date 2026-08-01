from pathlib import Path

import pytest

from tools.probe_component_overlay_topk_sensitivity import _parse_arm


def test_topk_sensitivity_arm_parser_supports_baseline_and_overlay() -> None:
    assert _parse_arm("postkd") == ("postkd", None)
    assert _parse_arm("fresh25=evidence/fresh25") == (
        "fresh25",
        Path("evidence/fresh25"),
    )
    with pytest.raises(Exception, match="arm must use"):
        _parse_arm("broken=")
