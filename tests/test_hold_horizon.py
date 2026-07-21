"""Tests for the hold-horizon sweep gates and the forward-ledger aggregation in the monitor.

Pure-function coverage (no DB): the hour17/echo causal gates mirror the live executor, and the
dashboard's forward-ledger aggregation buckets RESOLVE rows correctly by (signal, horizon).
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.research_s34_hold_horizon_sweep import cand_hour17, HORIZONS_H, STOP_VARIANTS
from tools.research_s34_echo_causal_vs_lookahead import cand_causal
from tools import s34_leads_monitor_dashboard as dash


def _ev(**kw):
    base = {"bull": False, "sess": "US", "btc4h": -10.0, "btc7d": -10.0,
            "hour": 18, "dow": 3, "echo_30_90": True, "noisy": False}
    base.update(kw)
    return base


def test_cand_hour17_matches_executor_gate():
    # qualifies: not bull, not EUROPE, bearish regime, hour>=17
    assert cand_hour17(_ev()) is True
    # fails hour<17
    assert cand_hour17(_ev(hour=16)) is False
    # fails EUROPE session
    assert cand_hour17(_ev(sess="EUROPE")) is False
    # fails when bull
    assert cand_hour17(_ev(bull=True)) is False
    # fails when regime not bearish (both >= 0)
    assert cand_hour17(_ev(btc4h=5.0, btc7d=5.0)) is False
    # passes if either horizon bearish
    assert cand_hour17(_ev(btc4h=5.0, btc7d=-1.0)) is True


def test_cand_causal_is_t0_only_no_noisy_dependency():
    # causal echo gate must NOT depend on the lookahead `noisy` field
    assert cand_causal(_ev(noisy=True)) == cand_causal(_ev(noisy=False))
    # dow Monday(0)/Wednesday(2) excluded
    assert cand_causal(_ev(dow=0)) is False
    assert cand_causal(_ev(dow=2)) is False
    # needs echo_30_90
    assert cand_causal(_ev(echo_30_90=False)) is False


def test_horizons_and_stops_defined():
    assert HORIZONS_H == [2, 4, 6, 12, 24, 48]
    labels = [lbl for _, lbl in STOP_VARIANTS]
    assert labels == ["nostop", "s150", "s300"]


def test_forward_agg_buckets_by_gate_control_and_intersection(tmp_path, monkeypatch):
    led = tmp_path / "hh.jsonl"
    rows = [
        # qualified hour17 + echo, hour 18 (>=17) -> hour17, echo_causal, echo_hi
        {"event": "RESOLVE", "anchor_ts_ms": 1, "hold_h": 6, "net_bps": 40.0, "hour_utc": 18,
         "qualified_hour17": True, "qualified_echo": True},
        # qualified hour17 only (echo false) -> hour17 + echo_ctrl
        {"event": "RESOLVE", "anchor_ts_ms": 2, "hold_h": 6, "net_bps": -60.0, "hour_utc": 19,
         "qualified_hour17": True, "qualified_echo": False},
        # NON-qualified both (control), hour 3 -> hour17_ctrl + echo_ctrl
        {"event": "RESOLVE", "anchor_ts_ms": 3, "hold_h": 6, "net_bps": 90.0, "hour_utc": 3,
         "qualified_hour17": False, "qualified_echo": False},
        # qualified echo but hour<17 -> echo_causal but NOT echo_hi; hour17 false -> hour17_ctrl
        {"event": "RESOLVE", "anchor_ts_ms": 4, "hold_h": 6, "net_bps": 20.0, "hour_utc": 13,
         "qualified_hour17": False, "qualified_echo": True},
    ]
    led.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(dash, "HOLD_FWD_LEDGER", led)
    agg = dash._hold_forward_agg()
    assert agg["available"] is True
    arms = agg["arms"]
    assert arms["hour17|h6"]["n"] == 2 and arms["hour17|h6"]["avg"] == -10.0   # anchors 1,2
    assert arms["hour17_ctrl|h6"]["n"] == 2                                     # control anchors 3,4
    assert arms["echo_causal|h6"]["n"] == 2                                     # anchors 1,4
    assert arms["echo_hi|h6"]["n"] == 1 and arms["echo_hi|h6"]["total"] == 40.0  # only anchor 1 (echo & h>=17)
    assert arms["echo_ctrl|h6"]["n"] == 2                                       # anchors 2,3


def test_forward_agg_empty_when_no_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "HOLD_FWD_LEDGER", tmp_path / "missing.jsonl")
    agg = dash._hold_forward_agg()
    assert agg["available"] is False


def test_load_hold_horizons_degrades_without_sweep(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "HOLD_SWEEP_JSON", tmp_path / "nope.json")
    monkeypatch.setattr(dash, "HOLD_FWD_LEDGER", tmp_path / "nope.jsonl")
    out = dash.load_hold_horizons()
    assert out["historical"]["available"] is False
    assert out["forward"]["available"] is False


# ── forward ledger: hour tagging + echo∩hour>=17 capture ──────────────────────

class _FakeCur:
    def __init__(self, now_ms):
        self._now = now_ms

    def execute(self, *a, **k):
        return self

    def fetchone(self):
        return (self._now,)


class _FakeConn:
    def __init__(self, now_ms):
        self._now = now_ms

    def cursor(self):
        return _FakeCur(self._now)


def _fake_book(bid, ask, *, age_ms=0, depth=500_000.0):
    """Build a _book_at return dict; age_ms controls the outage-staleness guard."""
    mid = (bid + ask) / 2.0
    return {"quote_ts": 0, "age_ms": age_ms, "bid": bid, "ask": ask, "mid": mid,
            "spread_bps": (ask - bid) / mid * 1e4, "bid_depth_usd": depth}


def _wire_common(hh, monkeypatch, ts, led, tmp_path):
    """Shared monkeypatch scaffold: qualifying echo∩hour>=17 anchor, no real DB touched."""
    monkeypatch.setattr(hh, "LEDGER", led)
    monkeypatch.setattr(hh, "STATE", tmp_path / "hh_state.json")
    monkeypatch.setattr(hh, "_detect_fresh_anchors", lambda cur, now: ([(ts, 250_000.0)], [ts - 40 * 60_000, ts]))
    monkeypatch.setattr(hh, "_mark_bps", lambda cur, sym, t, lb: -50.0)      # bearish regime, not bull
    monkeypatch.setattr(hh, "_echo_check", lambda ts_list, t, lo, hi: True)  # echo_30_90 present
    monkeypatch.setattr(hh, "_session", lambda hour: "US")                   # not EUROPE
    monkeypatch.setattr(hh, "_mark_at", lambda cur, t, sym="ETHUSDT": 1800.0 if t == ts else 1810.0)
    monkeypatch.setattr(hh, "_min_mark", lambda cur, sym, lo, hi: 1790.0)    # above stop levels -> no stop
    monkeypatch.setattr(hh, "_book_max", lambda cur, sym="ETHUSDT": ts + 100 * 3600_000)  # book clock past all horizons


def test_forward_ledger_measured_cost_and_intersection(tmp_path, monkeypatch):
    """run_once on a synthetic echo∩hour>=17 anchor (no DB): every RESOLVE self-tags hour_utc across
    all 6 horizons AND carries a MEASURED-cost net_bps (buy@ask/sell@bid) with spread PAID + slip attribution,
    with nothing quarantined when quotes are fresh and continuous."""
    import datetime as dt
    from tools import research_s34_hold_horizon_forward_ledger as hh

    ts = int(dt.datetime(2026, 7, 21, 18, 0, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)  # hour=18, > cutoff
    now_ms = ts + 49 * 3600_000  # all 6 horizons (<=48h) matured
    led = tmp_path / "hh.jsonl"
    _wire_common(hh, monkeypatch, ts, led, tmp_path)
    # fresh quotes: entry ask 1800.5 (t==ts), exit bid 1809.5 (t>ts); age 0 -> not stale
    monkeypatch.setattr(hh, "_book_at",
                        lambda cur, t, sym="ETHUSDT": _fake_book(1799.5, 1800.5) if t == ts
                        else _fake_book(1809.5, 1810.5))
    monkeypatch.setattr(hh, "_min_bid", lambda cur, lo, hi, sym="ETHUSDT": 1789.0)  # above stop levels
    monkeypatch.setattr(hh, "_window_has_gap", lambda cur, lo, hi, thr, sym="ETHUSDT": False)  # continuous

    st = {"processed": [], "pending": {}}
    opened, resolved, pending, quarantined = hh.run_once(_FakeConn(now_ms), st)

    recs = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines() if x.strip()]
    opens = [r for r in recs if r["event"] == "OPEN"]
    resolves = [r for r in recs if r["event"] == "RESOLVE"]

    assert opened == 1 and resolved == 6 and pending == 0 and quarantined == 0
    assert len(opens) == 1 and opens[0]["hour_utc"] == 18 and opens[0]["cost_model"] == "measured_v2"
    assert opens[0]["ask_entry"] == 1800.5 and opens[0]["qualified_hour17"] and opens[0]["qualified_echo"]
    # ONE resolve per horizon; every row carries gate flags + measured fields
    assert len(resolves) == 6
    assert all(r["hour_utc"] == 18 and r["qualified_hour17"] and r["qualified_echo"] for r in resolves)
    echo_hi = [r for r in resolves if r["qualified_echo"] and r["hour_utc"] >= 17]
    assert len(echo_hi) == 6
    e6 = next(r for r in resolves if r["hold_h"] == 6)
    # B1: net_bps is the AUTHORITATIVE consumer-facing field (= measured, spread PAID) and equals net_bps_measured
    expect = round((1809.5 - 1800.5) / 1800.5 * 1e4 - 5.0, 2)
    assert e6["net_bps"] == expect and e6["net_bps_measured"] == expect
    assert e6["quarantined"] is False and e6["quarantine_reasons"] == []
    # slip = mark-net minus measured-net > 0 (measured pays the spread the mark-fill ignored)
    assert e6["slip_bps"] > 0 and e6["net_bps"] < e6["net_bps_mark"]
    # no stop triggered (min_bid above levels) -> authoritative stop nets equal nostop net_bps
    assert e6["net_bps_s150"] == e6["net_bps"] and e6["net_bps_s300"] == e6["net_bps"]


def test_forward_ledger_quarantines_stale_exit_quote(tmp_path, monkeypatch):
    """A stale exit quote (age > 60s, book clock already past exit_ts) IS the §166 fake-price mechanism
    -> every RESOLVE quarantined AND its authoritative net_bps is None (auto-excluded by consumers)."""
    import datetime as dt
    from tools import research_s34_hold_horizon_forward_ledger as hh

    ts = int(dt.datetime(2026, 7, 21, 18, 0, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    now_ms = ts + 49 * 3600_000
    led = tmp_path / "hh.jsonl"
    _wire_common(hh, monkeypatch, ts, led, tmp_path)
    # entry fresh (age 0), exit quote 200s stale -> exit_quote_stale (book_max is ahead, so NOT a defer)
    monkeypatch.setattr(hh, "_book_at",
                        lambda cur, t, sym="ETHUSDT": _fake_book(1799.5, 1800.5, age_ms=0) if t == ts
                        else _fake_book(1809.5, 1810.5, age_ms=200_000))
    monkeypatch.setattr(hh, "_min_bid", lambda cur, lo, hi, sym="ETHUSDT": 1789.0)
    monkeypatch.setattr(hh, "_window_has_gap", lambda cur, lo, hi, thr, sym="ETHUSDT": False)

    st = {"processed": [], "pending": {}}
    opened, resolved, pending, quarantined = hh.run_once(_FakeConn(now_ms), st)
    resolves = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines()
                if x.strip() and json.loads(x)["event"] == "RESOLVE"]
    assert quarantined == 6 and len(resolves) == 6
    assert all(r["quarantined"] and "exit_quote_stale" in r["quarantine_reasons"] for r in resolves)
    assert all(r["net_bps"] is None for r in resolves)   # B1: quarantined -> authoritative net excluded


def test_forward_ledger_quarantines_in_window_gap(tmp_path, monkeypatch):
    """A book_ticker gap > 5min inside the hold makes the path/stop untrustworthy -> quarantined, net_bps None."""
    import datetime as dt
    from tools import research_s34_hold_horizon_forward_ledger as hh

    ts = int(dt.datetime(2026, 7, 21, 18, 0, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    now_ms = ts + 49 * 3600_000
    led = tmp_path / "hh.jsonl"
    _wire_common(hh, monkeypatch, ts, led, tmp_path)
    monkeypatch.setattr(hh, "_book_at",
                        lambda cur, t, sym="ETHUSDT": _fake_book(1799.5, 1800.5) if t == ts
                        else _fake_book(1809.5, 1810.5))
    monkeypatch.setattr(hh, "_min_bid", lambda cur, lo, hi, sym="ETHUSDT": 1789.0)
    monkeypatch.setattr(hh, "_window_has_gap", lambda cur, lo, hi, thr, sym="ETHUSDT": True)  # gap present

    st = {"processed": [], "pending": {}}
    opened, resolved, pending, quarantined = hh.run_once(_FakeConn(now_ms), st)
    resolves = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines()
                if x.strip() and json.loads(x)["event"] == "RESOLVE"]
    assert quarantined == 6
    assert all(r["quarantined"] and "in_window_gap" in r["quarantine_reasons"] for r in resolves)
    assert all(r["net_bps"] is None for r in resolves)


def test_forward_ledger_defers_when_book_feed_lags(tmp_path, monkeypatch):
    """B3: if the book feed hasn't reached a horizon's exit_ts (transient lag behind the mark clock),
    that horizon is DEFERRED (stays pending), never quarantined/dropped — only horizons the book clock
    has passed resolve this pass. The exact §166-adjacent bug: a recoverable lag must not lose a trade."""
    import datetime as dt
    from tools import research_s34_hold_horizon_forward_ledger as hh

    ts = int(dt.datetime(2026, 7, 21, 18, 0, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    now_ms = ts + 49 * 3600_000                       # mark clock: all horizons matured
    led = tmp_path / "hh.jsonl"
    _wire_common(hh, monkeypatch, ts, led, tmp_path)
    monkeypatch.setattr(hh, "_book_max", lambda cur, sym="ETHUSDT": ts + 5 * 3600_000)  # book only reached +5h
    monkeypatch.setattr(hh, "_book_at",
                        lambda cur, t, sym="ETHUSDT": _fake_book(1799.5, 1800.5) if t == ts
                        else _fake_book(1809.5, 1810.5))
    monkeypatch.setattr(hh, "_min_bid", lambda cur, lo, hi, sym="ETHUSDT": 1789.0)
    monkeypatch.setattr(hh, "_window_has_gap", lambda cur, lo, hi, thr, sym="ETHUSDT": False)

    st = {"processed": [], "pending": {}}
    opened, resolved, pending, quarantined = hh.run_once(_FakeConn(now_ms), st)
    resolves = [json.loads(x) for x in led.read_text(encoding="utf-8").splitlines()
                if x.strip() and json.loads(x)["event"] == "RESOLVE"]
    # only 2h & 4h (book clock passed) resolved; 6/12/24/48h DEFERRED; none quarantined; anchor still pending
    assert opened == 1 and resolved == 2 and quarantined == 0 and pending == 1
    assert sorted(r["hold_h"] for r in resolves) == [2, 4]
    assert all(r["net_bps"] is not None and not r["quarantined"] for r in resolves)
