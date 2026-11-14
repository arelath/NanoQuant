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
    packed = (bits << shifts).sum(dim=1, dtype=torch.int16).to(torch.uint8)
    return packed, tuple(signs.shape)


def unpack_sign_bits(
    packed: torch.Tensor, shape: tuple[int, ...], *, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError("shape dimensions must not be negative")
        count *= dimension
    if packed.numel() * 8 < count:
        raise ValueError("packed data is too short for shape")
    # Compact distillation decodes every factor twice per step, so this is a hot
    # kernel. Lift each inverted bit into the sign position of its own byte and
    # let copysign emit the +/-1 values directly: two passes over the output
    # instead of the five of shift, mask, cast, scale and offset.
    lifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=packed.device)
    clear_is_negative = ((~packed.detach().reshape(-1, 1).to(torch.uint8)) << lifts).view(torch.int8)
    one = torch.ones((), dtype=dtype, device=packed.device)
    return torch.copysign(one, clear_is_negative).reshape(-1)[:count].reshape(shape).to(dtype)
