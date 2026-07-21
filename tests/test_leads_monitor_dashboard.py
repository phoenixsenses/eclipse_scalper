"""Tests for the two-lead read-only monitor dashboard.

Focus: honest aggregation (gap/outage artifacts quarantined out of the forward headline,
backfill never summed into forward), correct OPEN/CLOSE join + id dedup, and the read-only
HTTP contract. No network, no real DB; sources are monkeypatched to temp JSONL.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import s34_leads_monitor_dashboard as dash


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


# ── hour17 aggregation ────────────────────────────────────────────────────────

def test_hour17_quarantines_gap_artifact(tmp_path, monkeypatch):
    """A ~49h force-closed +900bps outage trade must NOT enter the forward headline."""
    led = tmp_path / "sm.jsonl"
    base = 1_784_000_000_000
    rows = [
        # clean 6h HOLD6H winner
        {"event": "OPEN", "signal": "LONG_HOUR17_HOLD6H", "id": "A",
         "entry_ts_ms": base, "entry_price": 100.0, "status": "OPEN"},
        {"event": "CLOSE", "signal": "LONG_HOUR17_HOLD6H", "id": "A",
         "entry_ts_ms": base, "exit_ts_ms": base + 6 * 3600_000, "net_bps": 40.0,
         "close_reason": "TIME_EXIT", "closed_utc": "2026-07-16T00:00:00+00:00"},
        # clean 6h HOLD6H loser
        {"event": "OPEN", "signal": "LONG_HOUR17_HOLD6H", "id": "B",
         "entry_ts_ms": base, "entry_price": 100.0, "status": "OPEN"},
        {"event": "CLOSE", "signal": "LONG_HOUR17_HOLD6H", "id": "B",
         "entry_ts_ms": base, "exit_ts_ms": base + 6 * 3600_000, "net_bps": -60.0,
         "close_reason": "TIME_EXIT", "closed_utc": "2026-07-16T06:00:00+00:00"},
        # GAP artifact: 49h hold, fake +900
        {"event": "OPEN", "signal": "LONG_HOUR17_HOLD6H", "id": "G",
         "entry_ts_ms": base, "entry_price": 100.0, "status": "OPEN"},
        {"event": "CLOSE", "signal": "LONG_HOUR17_HOLD6H", "id": "G",
         "entry_ts_ms": base, "exit_ts_ms": base + 49 * 3600_000, "net_bps": 905.0,
         "close_reason": "TIME_EXIT", "closed_utc": "2026-07-15T19:00:00+00:00"},
    ]
    _write_jsonl(led, rows)
    monkeypatch.setattr(dash, "SM_LEDGER", led)

    out = dash.load_hour17(mark=None)
    # headline = clean HOLD6H only: 40 and -60 → n=2, avg=-10
    assert out["agg_forward"]["n"] == 2
    assert out["agg_forward"]["avg"] == -10.0
    # gap bucket isolated
    assert out["agg_gap"]["n"] == 1
    assert out["agg_gap"]["avg"] == 905.0
    assert out["quarantine_note"] and "1" in out["quarantine_note"]


def test_hour17_backfill_not_summed_into_forward(tmp_path, monkeypatch):
    led = tmp_path / "sm.jsonl"
    base = 1_784_000_000_000
    rows = [
        {"event": "CLOSE", "signal": "LONG_HOUR17_HOLD6H", "id": "F",
         "entry_ts_ms": base, "exit_ts_ms": base + 6 * 3600_000, "net_bps": 30.0,
         "closed_utc": "2026-07-16T00:00:00+00:00"},
        {"event": "CLOSE", "signal": "LONG_HOUR17_HOLD6H", "id": "BF", "backfill": True,
         "entry_ts_ms": base, "exit_ts_ms": base + 6 * 3600_000, "net_bps": 500.0,
         "closed_utc": "2026-06-01T00:00:00+00:00"},
    ]
    _write_jsonl(led, rows)
    monkeypatch.setattr(dash, "SM_LEDGER", led)
    out = dash.load_hour17(mark=None)
    assert out["agg_forward"]["n"] == 1
    assert out["agg_forward"]["total"] == 30.0
    assert out["agg_backfill"]["n"] == 1
    assert out["agg_backfill"]["total"] == 500.0


def test_hour17_close_dedup_by_id(tmp_path, monkeypatch):
    led = tmp_path / "sm.jsonl"
    base = 1_784_000_000_000
    dup = {"event": "CLOSE", "signal": "LONG_HOUR17_HOLD6H", "id": "X",
           "entry_ts_ms": base, "exit_ts_ms": base + 6 * 3600_000, "net_bps": 10.0,
           "closed_utc": "2026-07-16T00:00:00+00:00"}
    _write_jsonl(led, [dup, dict(dup)])  # same id twice
    monkeypatch.setattr(dash, "SM_LEDGER", led)
    out = dash.load_hour17(mark=None)
    assert out["agg_forward"]["n"] == 1  # not 2


def test_hour17_open_position_detected(tmp_path, monkeypatch):
    led = tmp_path / "sm.jsonl"
    base = 1_784_000_000_000
    rows = [
        {"event": "OPEN", "signal": "LONG_HOUR17_HOLD6H", "id": "OPN",
         "entry_ts_ms": base, "entry_price": 100.0, "exit_due_ms": base + 6 * 3600_000,
         "status": "OPEN", "direction": "LONG"},
    ]
    _write_jsonl(led, rows)
    monkeypatch.setattr(dash, "SM_LEDGER", led)
    out = dash.load_hour17(mark={"price": 101.0, "ts_ms": base})
    assert len(out["open_positions"]) == 1
    # unrealized = (101-100)/100 * 1e4 = 100 bps
    assert out["open_positions"][0]["unrealized_bps"] == 100.0


# ── echo aggregation ──────────────────────────────────────────────────────────

def test_echo_open_close_join_and_qualified_split(tmp_path, monkeypatch):
    led = tmp_path / "echo.jsonl"
    ats = 1_784_600_000_000
    rows = [
        {"event": "OPEN", "anchor_ts_ms": ats, "utc": "2026-07-21T00:00:00+00:00",
         "entry_mark": 1800.0, "qualified_t0": True, "session": "US", "echo_30_90": True},
        {"event": "CLOSE", "anchor_ts_ms": ats, "entry_mark": 1800.0, "exit_mark": 1810.0,
         "net_bps": 50.0, "qualified_t0": True, "qualified_full": True, "noisy_T30m": False},
        # a t0-only (noisy) close → in t0 agg, NOT in full agg
        {"event": "OPEN", "anchor_ts_ms": ats + 1, "utc": "2026-07-21T01:00:00+00:00",
         "entry_mark": 1800.0, "qualified_t0": True},
        {"event": "CLOSE", "anchor_ts_ms": ats + 1, "entry_mark": 1800.0, "exit_mark": 1780.0,
         "net_bps": -20.0, "qualified_t0": True, "qualified_full": False, "noisy_T30m": True},
        # a NON-qualified anchor (control) — recorded but must NOT pool into qualified_t0
        {"event": "OPEN", "anchor_ts_ms": ats + 2, "utc": "2026-07-21T02:00:00+00:00",
         "entry_mark": 1800.0, "qualified_t0": False},
        {"event": "CLOSE", "anchor_ts_ms": ats + 2, "entry_mark": 1800.0, "exit_mark": 1830.0,
         "net_bps": 90.0, "qualified_t0": False, "qualified_full": False, "noisy_T30m": False},
    ]
    _write_jsonl(led, rows)
    monkeypatch.setattr(dash, "ECHO_LEDGER", led)
    out = dash.load_echo(mark=None)
    assert out["agg_forward_t0"]["n"] == 2          # qualified_t0 only (the +90 control excluded)
    assert out["agg_forward_t0"]["total"] == 30.0   # 50 + (-20), NOT +90
    assert out["agg_control_nonqual"]["n"] == 1      # the non-qualified control
    assert out["agg_control_nonqual"]["total"] == 90.0
    assert out["agg_forward_full"]["n"] == 1        # only the not-noisy one
    assert out["agg_forward_full"]["total"] == 50.0


def test_echo_empty_ledger_still_has_context(tmp_path, monkeypatch):
    led = tmp_path / "empty.jsonl"
    led.write_text("", encoding="utf-8")
    monkeypatch.setattr(dash, "ECHO_LEDGER", led)
    out = dash.load_echo(mark=None)
    assert out["agg_forward_t0"]["n"] == 0
    assert out["empty_note"]  # informative, not a crash
    # context card loads from the real causal json if present; degrade gracefully otherwise
    assert "context" in out


# ── read-only HTTP contract ───────────────────────────────────────────────────

def test_payload_contract_is_read_only():
    payload = dash.build_payload()
    c = payload["contract"]
    assert c["read_only"] is True
    assert c["control_actions_available"] is False
    assert c["db_mode"] == "ro"


def test_agg_tail_and_wr():
    a = dash._agg([50.0, -120.0, 10.0, -30.0])
    assert a["n"] == 4
    assert a["wr"] == 50.0
    assert a["tail_n"] == 1          # only -120 <= -100
    assert a["worst"] == -120.0


def test_serve_refuses_non_loopback():
    assert dash.serve("0.0.0.0", 8771) == 2   # non-loopback bind refused, no server started


# ── event feed + tail-rate scoreboard ─────────────────────────────────────────

def test_event_feed_tail_rate_scoreboard(tmp_path, monkeypatch):
    led = tmp_path / "hh.jsonl"
    rows = [
        # a1 (newest): echo+hour17 qualified, hour 18 → echo_hi; h4 +40 (no tail), h6 +50
        {"event": "OPEN", "anchor_ts_ms": 4000, "utc": "2026-07-21T18:00:00+00:00", "entry_mark": 1800.0,
         "qualified_hour17": True, "qualified_echo": True, "hour_utc": 18},
        {"event": "RESOLVE", "anchor_ts_ms": 4000, "hold_h": 4, "net_bps": 40.0,
         "qualified_hour17": True, "qualified_echo": True, "hour_utc": 18},
        {"event": "RESOLVE", "anchor_ts_ms": 4000, "hold_h": 6, "net_bps": 50.0,
         "qualified_hour17": True, "qualified_echo": True, "hour_utc": 18},
        # a2: echo qualified, hour 13 (not hi); h4 -150 TAIL
        {"event": "OPEN", "anchor_ts_ms": 3000, "utc": "2026-07-21T13:00:00+00:00", "entry_mark": 1800.0,
         "qualified_hour17": False, "qualified_echo": True, "hour_utc": 13},
        {"event": "RESOLVE", "anchor_ts_ms": 3000, "hold_h": 4, "net_bps": -150.0,
         "qualified_hour17": False, "qualified_echo": True, "hour_utc": 13},
        # a3: control (both false), hour 3; h4 +90
        {"event": "OPEN", "anchor_ts_ms": 2000, "utc": "2026-07-21T03:00:00+00:00", "entry_mark": 1800.0,
         "qualified_hour17": False, "qualified_echo": False, "hour_utc": 3},
        {"event": "RESOLVE", "anchor_ts_ms": 2000, "hold_h": 4, "net_bps": 90.0,
         "qualified_hour17": False, "qualified_echo": False, "hour_utc": 3},
        # a4: echo qualified, hour 18 → echo_hi; h4 -200 TAIL (an echo_hi tail = refutation signal)
        {"event": "OPEN", "anchor_ts_ms": 1000, "utc": "2026-07-21T18:30:00+00:00", "entry_mark": 1800.0,
         "qualified_hour17": False, "qualified_echo": True, "hour_utc": 18},
        {"event": "RESOLVE", "anchor_ts_ms": 1000, "hold_h": 4, "net_bps": -200.0,
         "qualified_hour17": False, "qualified_echo": True, "hour_utc": 18},
    ]
    led.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(dash, "HOLD_FWD_LEDGER", led)
    F = dash.load_event_feed()
    assert F["available"] is True and F["n_events"] == 4
    assert F["events"][0]["id"] == "4000"          # newest first
    sb = F["scoreboard"]
    # echo primary horizon = 4h: qualified {a1 +40, a2 -150 tail, a4 -200 tail} → 2/3 tails
    assert sb["echo_qual"] == {"n": 3, "tail": 2, "rate": 66.7}
    assert sb["echo_ctrl"] == {"n": 1, "tail": 0, "rate": 0.0}
    assert sb["echo_hi"] == {"n": 2, "tail": 1, "rate": 50.0}   # a1 clean, a4 tail
    a1 = next(e for e in F["events"] if e["id"] == "4000")
    assert a1["open"] is True and a1["res"]["12"] is None       # 12h/24h/48h pending
    a2 = next(e for e in F["events"] if e["id"] == "3000")
    assert a2["any_tail"] is True and a2["res"]["4"]["tail"] is True


def test_event_feed_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "HOLD_FWD_LEDGER", tmp_path / "missing.jsonl")
    F = dash.load_event_feed()
    assert F["available"] is False and F["events"] == []
