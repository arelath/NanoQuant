from tools.compare_quality_task_reports import compare_reports


def _report(first: tuple[bool, bool], second: tuple[bool, bool]) -> dict[str, object]:
    def task(name: str, metric: str, values: tuple[bool, bool]) -> dict[str, object]:
        field = "normalized_correct" if metric == "acc_norm" else "raw_correct"
        examples = [
            {"sample_id": f"{name}:{index}", field: value}
            for index, value in enumerate(values)
        ]
        return {
            "result": {
                "task_name": name,
                "primary_metric": metric,
                "primary_value": sum(values) / len(values),
                "prompt_hash": f"{name}-hash",
                "task_semantic_key": f"{name}-semantic",
                "examples": examples,
            }
        }

    return {
        "protocol": {
            "task_names": ["first", "second"],
            "task_limit": 2,
            "task_batch_size": 1,
            "tokenizer_hash": "tokenizer",
        },
        "results": {
            "frozen": {
                "tasks": [task("first", "acc_norm", first), task("second", "acc", second)]
            }
        },
    }


def test_task_report_comparison_is_paired_and_task_stratified() -> None:
    baseline = _report((False, False), (True, False))
    candidate = _report((True, False), (True, True))

    result = compare_reports(
        baseline,
        candidate,
        baseline_result="frozen",
        candidate_result="frozen",
        confidence=0.95,
        resamples=1_000,
        seed=0,
    )

    assert result["aggregate"]["baseline_mean"] == 0.25
    assert result["aggregate"]["candidate_mean"] == 0.75
    assert result["aggregate"]["candidate_minus_baseline"] == 0.5
    assert result["aggregate"]["regression_established"] is False
    assert result["aggregate"]["improvement_established"] is False
