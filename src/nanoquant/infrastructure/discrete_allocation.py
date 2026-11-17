"""Solve an additive objective over one integer-cost option per unit."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DiscreteOption:
    label: str
    cost: int
    error: float


@dataclass(frozen=True, slots=True)
class DiscreteAllocation:
    choices: dict[str, DiscreteOption]
    cost: int
    error: float
    lower_bound: float
    relative_gap: float
    status: int
    message: str


def solve_discrete_allocation(
    units: Mapping[str, Sequence[DiscreteOption]], budget: int, *, seconds: float = 60.0,
) -> DiscreteAllocation:
    """Minimize summed error with one discrete choice per unit and a shared budget."""
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore[import-untyped]
        from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("discrete allocation requires the optional 'allocation' dependencies") from error

    if budget < 0 or not units:
        raise ValueError("a nonnegative budget and at least one unit are required")
    if any(not options for options in units.values()):
        raise ValueError("every unit must have at least one discrete option")
    flat = [(name, option) for name, options in units.items() for option in options]
    if any(option.cost < 0 or not math.isfinite(option.error) or option.error < 0 for _, option in flat):
        raise ValueError("discrete options require nonnegative costs and finite nonnegative errors")
    rows_by_name = {name: index for index, name in enumerate(units)}
    columns = np.arange(len(flat))
    costs = np.array([option.cost for _, option in flat], dtype=np.float64)
    rows = np.array([rows_by_name[name] for name, _ in flat])
    unit_count = len(units)
    # Keep the budget row numerically comparable to the unit constraints.
    scale = 1e6
    matrix = coo_matrix((
        np.concatenate((np.ones(len(flat)), costs / scale)),
        (np.concatenate((rows, np.full(len(flat), unit_count))), np.tile(columns, 2)),
    ), shape=(unit_count + 1, len(flat))).tocsc()
    result = milp(
        np.array([option.error for _, option in flat]), integrality=np.ones(len(flat)), bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, np.r_[np.ones(unit_count), -np.inf],
                                     np.r_[np.ones(unit_count), budget / scale]),
        options={"time_limit": seconds, "mip_rel_gap": 1e-8},
    )
    if result.x is None:
        raise ValueError(f"no feasible allocation: {result.message}")
    selected = [(name, option) for value, (name, option) in zip(result.x, flat, strict=True) if value > 0.5]
    chosen = dict(selected)
    used = sum(option.cost for option in chosen.values())
    if len(selected) != unit_count or chosen.keys() != units.keys() or used > budget:
        raise ValueError("solver returned an invalid discrete allocation")
    return DiscreteAllocation(
        chosen, used, math.fsum(option.error for option in chosen.values()), float(result.mip_dual_bound),
        float(result.mip_gap), int(result.status), str(result.message),
    )
