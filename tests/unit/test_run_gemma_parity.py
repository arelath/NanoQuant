from pathlib import Path

from tools.run_gemma_parity import _parser


def test_parity_launcher_inherits_logical_tuning_batch_for_microbatch_default() -> None:
    args = _parser().parse_args(["--output", "run", "--snapshot", "snapshot"])

    assert args.factorized_tuning_batch_size == 8
    assert args.nonfactorized_tuning_batch_size == 8
    assert args.post_block_refit_batch_size == 8
    assert args.tuning_microbatch_size is None
    assert args.output == Path("run")


def test_parity_launcher_keeps_explicit_memory_fallback_microbatch() -> None:
    args = _parser().parse_args(
        ["--output", "run", "--snapshot", "snapshot", "--tuning-microbatch-size", "1"]
    )

    assert args.tuning_microbatch_size == 1
