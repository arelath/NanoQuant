from __future__ import annotations

from tools.fold_global_tuning_final_norm_scale import calibrated_protocol_hash


def test_calibrated_protocol_hash_binds_base_and_scale() -> None:
    first = calibrated_protocol_hash("sha256:base", 1.06)

    assert first.startswith("sha256:")
    assert first == calibrated_protocol_hash("sha256:base", 1.06)
    assert first != calibrated_protocol_hash("sha256:other", 1.06)
    assert first != calibrated_protocol_hash("sha256:base", 1.075)
