"""Explicit flat tensor roles for independently scaled residual factor terms."""

from nanoquant.runtime.backend import QuantizedLinearSpec

RESIDUAL_FORMAT = "nanoquant-residual-v2"
RESIDUAL_MINIMUM_RUNTIME = "0.2.0"
FACTOR_ROLES = ("factor_left", "factor_right", "scale_pre", "scale_mid", "scale_post")
PACKED_FACTOR_ROLES = ("factor_left_words", "factor_right_words", "scale_pre", "scale_mid", "scale_post")


def stage_role(index: int, role: str) -> str:
    return f"residual.{index}.{role}"


def is_stage_role(role: str, roles: tuple[str, ...]) -> bool:
    parts = role.split(".")
    return (len(parts) == 3 and parts[0] == "residual" and parts[1].isascii()
            and parts[1].isdigit() and str(int(parts[1])) == parts[1] and parts[2] in roles)


def stage_spec(spec: QuantizedLinearSpec, index: int) -> QuantizedLinearSpec:
    return QuantizedLinearSpec(
        f"{spec.name}.residual.{index}", "nanoquant-v1", spec.in_features, spec.out_features,
        spec.residual_ranks[index], spec.factor_dtype, spec.scale_dtype,
    )


def validate_stage_spec(parent: QuantizedLinearSpec, child: QuantizedLinearSpec, index: int) -> None:
    if child != stage_spec(parent, index):
        raise ValueError("residual stage specification differs from its owning layer inventory")
