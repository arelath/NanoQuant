"""Measure joint-pattern screen recall against exhaustive full scale refits."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import _paths  # noqa: F401
import torch

from nanoquant.config.codec import semantic_hash
from nanoquant.domain.binary_factor_search import _joint_scale_screen_batch, _patterns
from nanoquant.domain.scale_fit import fit_scales
from nanoquant.infrastructure.io_utils import atomic_write_json


def _signs(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 2, shape, generator=generator).float().mul_(2).sub_(1)


def _candidate_population(
    left: torch.Tensor,
    right: torch.Tensor,
    patterns: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = patterns.shape[0]
    candidate_left = left.expand(count, -1, -1).clone()
    candidate_right = right.expand(count, -1, -1).clone()
    split = left_indices.shape[0]
    candidate_left[:, left_indices[:, 0], left_indices[:, 1]] = patterns[:, :split]
    candidate_right[:, right_indices[:, 0], right_indices[:, 1]] = patterns[:, split:]
    return candidate_left, candidate_right


def _full_refit_errors(
    target: torch.Tensor,
    candidate_left: torch.Tensor,
    candidate_right: torch.Tensor,
    pre: torch.Tensor,
    mid: torch.Tensor,
    post: torch.Tensor,
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    passes: int,
) -> torch.Tensor:
    errors = torch.empty(candidate_left.shape[0], device=target.device)
    for index in range(candidate_left.shape[0]):
        errors[index] = fit_scales(
            target,
            candidate_left[index],
            candidate_right[index],
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            alternating_passes=passes,
        ).after_error
    return errors


def _winner_rank(errors: torch.Tensor, winner: int) -> int:
    return int((errors < errors[winner]).sum()) + 1


def run(args: argparse.Namespace) -> int:
    if args.cases <= 0 or args.size <= 1 or args.rank <= 0 or args.rank > args.size:
        raise ValueError("joint screen recall dimensions are invalid")
    available_left = (args.size - 1) * args.rank
    available_right = args.rank * (args.size - 1)
    if args.bits < 2 or args.bits > available_left + available_right or args.bits > 16:
        raise ValueError("joint screen recall bit count is invalid")
    if args.screen_passes < 0 or args.refit_passes < args.screen_passes:
        raise ValueError("full refit cannot be shallower than the screen")

    device = torch.device(args.device)
    cutoffs = (1, 4, 16)
    scale_hits = {cutoff: 0 for cutoff in cutoffs}
    fixed_hits = {cutoff: 0 for cutoff in cutoffs}
    scale_ranks: list[int] = []
    fixed_ranks: list[int] = []
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for case in range(args.cases):
        generator = torch.Generator().manual_seed(args.seed + case)
        target = torch.randn((args.size, args.size), generator=generator).to(device)
        left = _signs((args.size, args.rank), generator).to(device)
        right = _signs((args.rank, args.size), generator).to(device)
        pre = torch.exp(0.8 * torch.randn(args.size, generator=generator)).to(device)
        mid = torch.exp(0.8 * torch.randn(args.rank, generator=generator)).to(device)
        post = torch.exp(0.8 * torch.randn(args.size, generator=generator)).to(device)
        input_weight = torch.exp(0.5 * torch.randn(args.size, generator=generator)).to(device)
        output_weight = torch.exp(0.5 * torch.randn(args.size, generator=generator)).to(device)

        left_free = torch.cartesian_prod(
            torch.arange(1, args.size),
            torch.arange(args.rank),
        )
        right_free = torch.cartesian_prod(
            torch.arange(args.rank),
            torch.arange(1, args.size),
        )
        left_count = min(left_free.shape[0], args.bits // 2)
        right_count = args.bits - left_count
        if right_count > right_free.shape[0]:
            right_count = right_free.shape[0]
            left_count = args.bits - right_count
        left_indices = left_free[torch.randperm(left_free.shape[0], generator=generator)[:left_count]].to(device)
        right_indices = right_free[
            torch.randperm(right_free.shape[0], generator=generator)[:right_count]
        ].to(device)
        patterns = _patterns(args.bits, device, target.dtype)
        candidate_left, candidate_right = _candidate_population(
            left,
            right,
            patterns,
            left_indices,
            right_indices,
        )

        scale_errors = _joint_scale_screen_batch(
            target,
            candidate_left,
            candidate_right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            args.screen_passes,
            1e-8,
        )
        fixed_prediction = torch.bmm(
            candidate_left * post[None, :, None],
            candidate_right * (mid[None, :, None] * pre[None, None, :]),
        )
        fixed_errors = (
            (fixed_prediction - target[None]).square()
            * output_weight[None, :, None]
            * input_weight[None, None, :]
        ).sum(dim=(1, 2))
        full_errors = _full_refit_errors(
            target,
            candidate_left,
            candidate_right,
            pre,
            mid,
            post,
            input_weight,
            output_weight,
            args.refit_passes,
        )
        winner = int(full_errors.argmin())
        scale_rank = _winner_rank(scale_errors, winner)
        fixed_rank = _winner_rank(fixed_errors, winner)
        scale_ranks.append(scale_rank)
        fixed_ranks.append(fixed_rank)
        for cutoff in cutoffs:
            scale_hits[cutoff] += int(scale_rank <= cutoff)
            fixed_hits[cutoff] += int(fixed_rank <= cutoff)
        rows.append(
            {
                "case": case,
                "full_refit_winner": winner,
                "scale_profiled_rank": scale_rank,
                "fixed_scale_rank": fixed_rank,
                "full_refit_error": float(full_errors[winner]),
                "scale_profiled_error": float(scale_errors[winner]),
                "fixed_scale_error": float(fixed_errors[winner]),
            }
        )

    protocol = {
        "cases": args.cases,
        "size": args.size,
        "rank": args.rank,
        "bits": args.bits,
        "patterns": 2**args.bits,
        "screen_passes": args.screen_passes,
        "refit_passes": args.refit_passes,
        "seed": args.seed,
        "device": args.device,
    }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "protocol": protocol,
        "protocol_hash": semantic_hash(protocol),
        "scale_profiled": {
            "winner_recall": {str(key): scale_hits[key] / args.cases for key in cutoffs},
            "mean_winner_rank": sum(scale_ranks) / args.cases,
            "maximum_winner_rank": max(scale_ranks),
        },
        "fixed_scale": {
            "winner_recall": {str(key): fixed_hits[key] / args.cases for key in cutoffs},
            "mean_winner_rank": sum(fixed_ranks) / args.cases,
            "maximum_winner_rank": max(fixed_ranks),
        },
        "wall_seconds": time.perf_counter() - started,
        "cases": rows,
    }
    atomic_write_json(args.output, payload)
    print(
        "scale recall="
        + ", ".join(f"top-{key} {scale_hits[key]}/{args.cases}" for key in cutoffs)
        + f" mean-rank={sum(scale_ranks) / args.cases:.3f}",
        flush=True,
    )
    print(
        "fixed recall="
        + ", ".join(f"top-{key} {fixed_hits[key]}/{args.cases}" for key in cutoffs)
        + f" mean-rank={sum(fixed_ranks) / args.cases:.3f}",
        flush=True,
    )
    print(f"wall={payload['wall_seconds']:.3f}s output={args.output}", flush=True)
    if not math.isfinite(float(payload["wall_seconds"])):
        raise ValueError("joint screen recall timing is nonfinite")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=32)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--screen-passes", type=int, default=4)
    parser.add_argument("--refit-passes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
