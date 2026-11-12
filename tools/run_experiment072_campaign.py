"""Run Experiment 072 with room for two 34-block passes and final evaluation."""

from run_experiment070_campaign import main as run_campaign

RUN_NAME = "072-two-full-rank-stages-compress-and-benchmark-gemma-3-4b-it"


def main(arguments: list[str] | None = None) -> int:
    return run_campaign(
        number=72, run_name=RUN_NAME, control_name="072-d2-uniform-control-gemma-3-4b-it",
        maximum_slices=80, arguments=arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
