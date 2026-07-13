from nanoquant.infrastructure.resource_usage import peak_process_memory_bytes, process_memory_snapshot


def test_process_memory_snapshot_reports_current_and_peak_working_set() -> None:
    snapshot = process_memory_snapshot()

    assert snapshot.working_set_bytes > 0
    assert snapshot.peak_working_set_bytes >= snapshot.working_set_bytes
    assert snapshot.private_bytes >= 0
    assert snapshot.peak_private_bytes >= snapshot.private_bytes
    assert peak_process_memory_bytes() >= snapshot.peak_working_set_bytes
