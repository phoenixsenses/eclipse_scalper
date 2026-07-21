from tools.s34_v_engine_failed_rebound import failure_conditions, ledger_by_id, source_counts


def test_ledger_by_id_indexes_only_rows_with_ids():
    rows = [{"observation_id": "a", "x": 1}, {"x": 2}]

    assert ledger_by_id(rows) == {"a": {"observation_id": "a", "x": 1}}


def test_failure_conditions_flag_failed_v_15m():
    conditions = failure_conditions(min_mfe_bps=20.0)
    trigger_min, fn = conditions["failed_v_15m"]

    assert trigger_min == 15
    assert fn({"ret_15m_bps": -1.0, "anchor_reclaimed_15m": False})
    assert not fn({"ret_15m_bps": 5.0, "anchor_reclaimed_15m": False})
    assert not fn({"ret_15m_bps": -1.0, "anchor_reclaimed_15m": True})


def test_no_rebound_uses_min_mfe_threshold():
    conditions = failure_conditions(min_mfe_bps=20.0)
    _, fn = conditions["no_rebound_mfe15"]

    assert fn({"mfe_15m_bps": 19.9})
    assert not fn({"mfe_15m_bps": 20.0})


def test_source_counts():
    assert source_counts([{"source": "a"}, {"source": "a"}, {"source": None}], "source") == {"a": 2, "none": 1}
