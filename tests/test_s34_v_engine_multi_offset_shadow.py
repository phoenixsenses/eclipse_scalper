from tools.s34_v_engine_multi_offset_shadow import build_brief, config_id, observation_id


def _row(cid, oid, *, status="CLOSED", sim_status="FILLED", net_bps=10.0):
    return {
        "config_id": cid,
        "observation_id": oid,
        "signal_ts_ms": 1000,
        "observation_status": status,
        "sim_status": sim_status,
        "net_bps": net_bps,
        "fill_delay_sec": 12.0,
        "counterfactual_anchor_mark_net_bps": 20.0,
    }


def test_config_id_is_stable_and_readable():
    assert config_id(20.0, 2.0) == "O20_C2"
    assert config_id(15.5, 1.0) == "O15.5_C1"


def test_observation_id_changes_by_offset_and_cross():
    a = observation_id(signal_ts_ms=1000, bucket=1, offset_bps=20.0, cross_margin_bps=1.0)
    b = observation_id(signal_ts_ms=1000, bucket=1, offset_bps=25.0, cross_margin_bps=1.0)
    c = observation_id(signal_ts_ms=1000, bucket=1, offset_bps=20.0, cross_margin_bps=2.0)

    assert a != b
    assert a != c


def test_build_brief_summarizes_by_config():
    rows = [
        _row("O20_C1", "a", net_bps=10.0),
        _row("O20_C1", "b", net_bps=-3.0),
        _row("O25_C1", "c", status="CLOSED", sim_status="NO_MAKER_FILL", net_bps=None),
    ]

    brief = build_brief(rows, source_db={"path": "x"}, added_n=3)

    by_id = {row["config_id"]: row for row in brief["configs"]}
    assert brief["ledger"]["rows_total"] == 3
    assert by_id["O20_C1"]["closed_filled"] == 2
    assert by_id["O20_C1"]["summary"]["sum_bps"] == 7.0
    assert by_id["O25_C1"]["closed_no_fill"] == 1
