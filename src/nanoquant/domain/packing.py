"""Pack-independent logical sign encoding used by domain property tests."""

from __future__ import annotations

import torch


def pack_sign_bits(signs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    flat = (signs.detach().reshape(-1) >= 0).to(torch.uint8)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat((flat, torch.zeros(padding, dtype=torch.uint8, device=flat.device)))
    bits = flat.reshape(-1, 8)
    shifts = torch.arange(8, dtype=torch.uint8, device=flat.device)
    packed = (bits << shifts).sum(dim=1).to(torch.uint8)
    return packed, tuple(signs.shape)


def unpack_sign_bits(packed: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("shape dimensions must not be negative")
        count *= dimension
    if packed.numel() * 8 < count:
        raise ValueError("packed data is too short for shape")
    shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
    bits = ((packed.detach().reshape(-1, 1).to(torch.uint8) >> shifts) & 1).reshape(-1)[:count]
    return torch.where(
        bits.reshape(shape).bool(), torch.ones((), device=packed.device), -torch.ones((), device=packed.device)
    )
