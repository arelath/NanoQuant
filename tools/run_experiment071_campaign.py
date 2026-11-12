"""Run Experiment 071 in one-block slices; --dry-run only validates its setup."""

from run_experiment070_campaign import main as run_campaign

RUN_NAME = "071-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it"


def main(arguments: list[str] | None = None) -> int:
    return run_campaign(
        number=71, run_name=RUN_NAME, control_name="071-d2-uniform-control-gemma-3-1b-it", arguments=arguments,
    )


if __name__ == "__main__":
    raise SystemExit(main())
