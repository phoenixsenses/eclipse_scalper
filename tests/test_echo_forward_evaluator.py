"""Tests for research_s34_echo_forward_evaluator (Part 2 scoring machine).

Pure-function, no DB, synthetic ledger: verifies the FROZEN verdict rules (ACCUMULATING / CONFIRMED /
REFUTED), quarantine exclusion, and the §169 hour17 tail-gate overlay. Reads the real burned baseline
only for the CONFIRM bar (present in repo).
"""
from __future__ import annotations

import json

from tools import research_s34_echo_forward_evaluator as ev

H6_MS = 6 * 3600_000


def _row(anchor_ts, h, net, *, hour=18, qecho=True, qh17=True, quar=False):
    v = None if quar else net
    return {"event": "RESOLVE", "cost_model": "measured_v2", "anchor_ts_ms": anchor_ts, "hold_h": h,
            "net_bps": v, "net_bps_s150": v, "net_bps_s300": v, "quarantined": quar,
            "qualified_echo": qecho, "qualified_hour17": qh17, "hour_utc": hour}


def _write(tmp_path, monkeypatch, rows):
    led = tmp_path / "fwd.jsonl"
    led.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(ev, "LEDGER", led)


def test_no_data_is_fail_closed(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, [])
    sc = ev.evaluate()
    assert sc["status"] == "NO_DATA"
    assert sc["primary_hypotheses"]["P1_echo_6h_nostop"] == "ACCUMULATING"
    assert sc["primary_hypotheses"]["P2_hour17_tailgate_6h"] == "GATE_ACCUMULATING"


def test_below_n_min_is_accumulating(tmp_path, monkeypatch):
    base = 2_000_000_000_000
    rows = [_row(base + i * 7 * 3600_000, 6, 50.0) for i in range(5)]  # only 5 noov obs < N_MIN
    _write(tmp_path, monkeypatch, rows)
    sc = ev.evaluate()
    assert sc["cells"]["echo|h6|nostop"]["noov_n"] == 5
    assert sc["primary_hypotheses"]["P1_echo_6h_nostop"] == "ACCUMULATING"


def test_strong_positive_is_confirmed(tmp_path, monkeypatch):
    base = 2_000_000_000_000
    rows = [_row(base + i * 7 * 3600_000, 6, 50.0) for i in range(25)]  # 25 noov, +50 >> bar, CI_LB>0
    _write(tmp_path, monkeypatch, rows)
    sc = ev.evaluate()
    c = sc["cells"]["echo|h6|nostop"]
    assert c["noov_n"] == 25 and c["ci_lo"] > 0 and c["verdict"] == "CONFIRMED"
    assert c["primary"] is True
    assert sc["primary_hypotheses"]["P1_echo_6h_nostop"] == "CONFIRMED"


def test_negative_is_refuted(tmp_path, monkeypatch):
    base = 2_000_000_000_000
    rows = [_row(base + i * 7 * 3600_000, 6, -20.0) for i in range(25)]
    _write(tmp_path, monkeypatch, rows)
    sc = ev.evaluate()
    c = sc["cells"]["echo|h6|nostop"]
    assert c["ci_hi"] <= 0 and c["verdict"] == "REFUTED"


def test_quarantined_rows_excluded(tmp_path, monkeypatch):
    base = 2_000_000_000_000
    clean = [_row(base + i * 7 * 3600_000, 6, 50.0) for i in range(22)]
    quar = [_row(base + (100 + i) * 7 * 3600_000, 6, 50.0, quar=True) for i in range(10)]  # net_bps=None
    _write(tmp_path, monkeypatch, clean + quar)
    sc = ev.evaluate()
    c = sc["cells"]["echo|h6|nostop"]
    assert c["noov_n"] == 22                       # quarantined not counted
    assert sc["forward"]["quarantined"] == 10 and sc["forward"]["resolve_rows"] == 32


def test_overlapping_anchors_collapse_to_noov(tmp_path, monkeypatch):
    base = 2_000_000_000_000
    # 40 anchors only 1h apart at h6 -> no-overlap (>=6h) keeps ~ every 6th
    rows = [_row(base + i * 3600_000, 6, 50.0) for i in range(40)]
    _write(tmp_path, monkeypatch, rows)
    sc = ev.evaluate()
    assert sc["cells"]["echo|h6|nostop"]["noov_n"] < 40  # overlap collapsed


def test_hour17_tailgate_confirmed(tmp_path, monkeypatch):
    base = 2_000_000_000_000
    # echo_hi (hour>=17): 25 noov, tail-free (+40)
    hi = [_row(base + i * 7 * 3600_000, 6, 40.0, hour=18) for i in range(25)]
    # echo_lo (hour<17): 25 noov, ~16% tails (4 of 25 at -150)
    lo = []
    for i in range(25):
        ts = base + (1000 + i) * 7 * 3600_000
        lo.append(_row(ts, 6, -150.0 if i < 4 else 40.0, hour=13))
    _write(tmp_path, monkeypatch, hi + lo)
    sc = ev.evaluate()
    g = sc["gate"]["h6"]
    assert g["hi_noov_n"] == 25 and g["hi_tail_rate"] == 0.0
    assert g["lo_tail_rate"] >= 0.10 and g["primary"] is True
    assert g["verdict"] == "GATE_CONFIRMED"
    assert sc["primary_hypotheses"]["P2_hour17_tailgate_6h"] == "GATE_CONFIRMED"


def test_hour17_tailgate_refuted_when_hi_has_tails(tmp_path, monkeypatch):
    base = 2_000_000_000_000
    # echo_hi with >=10% tails -> gate provides no protection -> REFUTED
    hi = [_row(base + i * 7 * 3600_000, 6, -150.0 if i < 5 else 40.0, hour=18) for i in range(25)]
    _write(tmp_path, monkeypatch, hi)
    sc = ev.evaluate()
    assert sc["gate"]["h6"]["hi_tail_rate"] >= 0.10
    assert sc["gate"]["h6"]["verdict"] == "GATE_REFUTED"
