"""Pair by example and preserve equal task weights in transfer comparisons."""

import pytest
from probe_objective_transfer import paired_task_accuracy


def score(first, second):
    return dict(tasks=[dict(task_input_identity=name, result=dict(task_name=name, primary_metric="acc",
                           examples=[dict(sample_id=str(i), raw_correct=value) for i, value in enumerate(values)]))
                       for name, values in (("a", first), ("b", second))])


def test_paired_task_interval_uses_task_mean_not_pooled_example_mean():
    before = score([False], [True, True, True])
    after = score([True], [True, True, False])
    result = paired_task_accuracy(before, after)
    assert result["point_delta_pp"] == pytest.approx(100 / 3)
    assert result["lower_delta_pp"] >= 0
    assert result["task_count"] == 2
    after["tasks"][1]["result"]["examples"][0]["sample_id"] = "different"
    with pytest.raises(ValueError, match="identities"):
        paired_task_accuracy(before, after)
