from tools.s34_v_engine_shadow_observer import build_brief, merge_rows, observation_status


def _row(oid, ts_ms, *, status="CLOSED", sim_status="FILLED", net_bps=10.0):
    return {
        "observation_id": oid,
        "signal_ts_ms": ts_ms,
        "signal_utc": "2026-06-01T00:00:00+00:00",
        "observation_status": status,
        "sim_status": sim_status,
        "net_bps": net_bps,
        "counterfactual_anchor_mark_net_bps": -5.0,
    }


def test_merge_rows_dedupes_and_updates_existing_observation():
    existing = [_row("a", 1000, net_bps=1.0)]
    observed = [_row("a", 1000, net_bps=2.0), _row("b", 2000, net_bps=3.0)]

    merged, added = merge_rows(existing, observed)

    assert added == 1
    assert [r["observation_id"] for r in merged] == ["a", "b"]
    assert merged[0]["net_bps"] == 2.0


def test_build_brief_separates_filled_no_fill_and_pending_rows():
    rows = [
        _row("a", 1000, net_bps=12.0),
        _row("b", 2000, net_bps=-4.0),
        _row("c", 3000, status="CLOSED", sim_status="NO_MAKER_FILL", net_bps=None),
        _row("d", 4000, status="PENDING", sim_status="FILLED", net_bps=30.0),
    ]

    brief = build_brief(rows, brief_days=60, source_db={"path": "x"}, added_n=2)

    assert brief["ledger"]["rows_total"] == 4
    assert brief["ledger"]["rows_added_this_run"] == 2
    assert brief["ledger"]["status_counts"] == {"CLOSED": 3, "PENDING": 1}
    assert brief["overall"]["closed_filled"] == 2
    assert brief["overall"]["closed_no_fill_n"] == 1
    assert brief["overall"]["summary"]["sum_bps"] == 8.0


def test_observation_status_separates_old_missing_exit_book_from_pending():
    assert observation_status({"status": "NO_EXIT_BOOK"}, 10_000) == "DATA_INCOMPLETE"
    assert observation_status({"status": "NO_MAKER_FILL", "anchor_ts_ms": 1_000}, 10_000) == "PENDING"
