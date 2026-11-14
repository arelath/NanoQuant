"""Bounded CUDA comparison of nested, block-only, and host-activation checkpointing."""
from __future__ import annotations

import json
import time

import torch
from torch.utils.checkpoint import checkpoint

from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.infrastructure.device_lease import acquire_device_lease


def main() -> None:
    with acquire_device_lease("cuda"):
        torch.manual_seed(72)
        module = TrainableFactorizedLinear(
            torch.randint(0, 2, (10240, 2560)).to(torch.bfloat16).mul_(2).sub_(1),
            torch.randint(0, 2, (2560, 2560)).to(torch.bfloat16).mul_(2).sub_(1),
            torch.ones(2560, dtype=torch.bfloat16),
            torch.full((2560,), 0.01, dtype=torch.bfloat16),
            torch.full((10240,), 0.01, dtype=torch.bfloat16),
            immutable_binary_factors=True,
        )
        module.pack_fixed_signs()
        module.cuda()
        value = torch.randn(1, 512, 2560, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        reference = None
        for mode in ("nested", "block_only", "host_activations"):
            module.checkpoint_fixed_signs = mode != "block_only"
            times = []
            torch.cuda.reset_peak_memory_stats()
            for iteration in range(13):
                module.zero_grad(set_to_none=True)
                value.grad = None
                torch.cuda.synchronize()
                start = time.perf_counter()
                if mode == "host_activations":
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        output = module(value)
                else:
                    output = checkpoint(module, value, use_reentrant=True)
                output.float().square().mean().backward()
                torch.cuda.synchronize()
                if iteration >= 3:
                    times.append(time.perf_counter() - start)
            measured = [output.detach().cpu(), value.grad.detach().cpu()]
            measured.extend(p.grad.detach().cpu() for p in module.parameters() if p.grad is not None)
            if reference is None:
                reference = measured
            assert all(torch.equal(a, b) for a, b in zip(reference, measured, strict=True))
            print(json.dumps({"mode": mode, "mean_seconds": sum(times) / len(times),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "bitwise_forward_and_gradient_parity": True}), flush=True)


if __name__ == "__main__":
    main()
