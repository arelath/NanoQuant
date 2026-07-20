from __future__ import annotations

import pytest
from run_error_budget_quality import _quality_claim_passed

GATES = (
    "protocol_identity_matched",
    "base_results_reproduced",
    "packed_evaluation_identity_matched",
    "global_tuning_identity_matched",
    "same_or_lower_budget",
    "nll_improved",
    "exact_packed_conversion",
    "exact_packed_reference_parity",
)


def _passing_gates() -> dict[str, bool]:
    return {name: True for name in GATES}


def test_error_budget_quality_claim_requires_every_gate() -> None:
    assert _quality_claim_passed(**_passing_gates())


@pytest.mark.parametrize("failed_gate", GATES)
def test_error_budget_quality_claim_fails_closed_for_each_gate(failed_gate: str) -> None:
    gates = _passing_gates()
    gates[failed_gate] = False

    assert not _quality_claim_passed(**gates)
