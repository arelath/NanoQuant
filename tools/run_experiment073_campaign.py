"""Run Experiment 073 in resumable one-block slices, then export and evaluate."""

from run_experiment070_campaign import main as run_campaign

RUN_NAME = "073-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it"


def main(arguments: list[str] | None = None) -> int:
    return run_campaign(
        number=73, run_name=RUN_NAME, control_name="073-d2-uniform-control-gemma-3-1b-it",
        maximum_slices=64, arguments=arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
