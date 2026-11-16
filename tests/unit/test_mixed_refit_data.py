import numpy as np
from audit_mixed_refit_data import exact_matches


def test_overlap_audit_finds_offsets_but_does_not_cross_rows():
    rows = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert exact_matches(rows, [[2, 3], [4, 5], [7, 8]]) == [
        {"probe": 0, "row": 0, "start": 1}, {"probe": 2, "row": 1, "start": 2},
    ]
