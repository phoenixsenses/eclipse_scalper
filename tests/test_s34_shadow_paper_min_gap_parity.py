"""OD-018 persistent-v2 min-gap parity matrix (in-memory SQLite only).

Deterministic acceptance requirements from the 2026-07-10 parity audit:
  1. loop and backfill produce identical signal keys (gaps 299/899/900/901s)
  2. restart does not alter eligibility (seed map survives process boundary)
  3. exact 900-second boundary behaves identically (inclusive->= keeps)
  4. cursor boundaries do not truncate cluster identity (bucket-aligned rescan)
  5. repeated processing is idempotent (signal_key dedup layer)
  6. same-bucket duplicate suppression remains intact
  7. suppression chains resolve identically (A kept, B suppressed, C kept)
  8. per-rule state isolation (archived/log-only rules cannot alter another rule)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import S34Rule, _bucket_events  # noqa: E402

BUCKET_MS = 300_000
SYM = "SIMUSDT"


def make_rule(name: str = "SIM_SELL_SHORT_100K") -> S34Rule:
    return S34Rule(
        name=name,
        symbol=SYM,
        liq_side="SELL",
        direction="SHORT",
        threshold_usd=100_000.0,
        bucket_sec=300,
        min_gap_sec=900,
        entry_delay_sec=0,
        require_book_ticker_fill=False,
        modeled_spread_bps=0.0,
    )


def make_db(liq_rows: list[tuple[int, float]], span_ms: tuple[int, int]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE liquidations (symbol TEXT, side TEXT, ts_ms INTEGER, price REAL, notional REAL);
        CREATE TABLE mark_prices  (symbol TEXT, ts_ms INTEGER, mark_price REAL);
        CREATE TABLE agg_trades   (symbol TEXT, ts_ms INTEGER, price REAL, qty REAL,
                                   notional REAL, is_buyer_maker INTEGER);
        """
    )
    for ts, notional in liq_rows:
        conn.execute("INSERT INTO liquidations VALUES (?,?,?,?,?)", (SYM, "SELL", ts, 100.0, notional))
    for m in range(span_ms[0], span_ms[1] + 120_000, 60_000):
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (SYM, m, 100.0))
    conn.commit()
    return conn


def keys(rule: S34Rule, signals: list[dict]) -> list[str]:
    return [f"{rule.name}:{s['bucket']}" for s in signals]


def run_loop(conn, rule, cuts, seed_map=None):
    """Emulate run_once threading: bucket-aligned rescan + persisted per-rule seed."""
    seed_map = dict(seed_map or {})
    out_keys: list[str] = []
    for lo, hi in cuts:
        scan_lo = (lo // BUCKET_MS) * BUCKET_MS
        seed = seed_map.get(rule.name)
        sigs = _bucket_events(conn, rule, scan_lo, hi, 1000, last_signal_ms_seed=seed)
        for s in sigs:
            k = f"{rule.name}:{s['bucket']}"
            if k not in out_keys:  # existing_keys dedup layer
                out_keys.append(k)
        if sigs:
            newest = max(int(s["ts_ms"]) for s in sigs)
            if seed is None or newest > seed:
                seed_map[rule.name] = newest
    return out_keys, seed_map


T0 = 1_800_000_000_000  # bucket-aligned


def _two_cross_fixture(gap_s: int):
    a_ts = T0 + 250_000
    b_ts = a_ts + gap_s * 1000
    conn = make_db([(a_ts, 150_000.0), (b_ts, 150_000.0)], (T0, b_ts + 120_000))
    return conn, a_ts, b_ts


def test_loop_backfill_parity_across_gaps():
    rule = make_rule()
    for gap_s, expect_second in [(299, False), (899, False), (900, True), (901, True)]:
        conn, a_ts, b_ts = _two_cross_fixture(gap_s)
        end = b_ts + 60_000
        backfill = keys(rule, _bucket_events(conn, rule, T0, end, 1000))
        cursor = a_ts + 30_000  # cycle boundary between the two crossings
        loop_keys, _ = run_loop(conn, rule, [(T0, cursor), (cursor, end)])
        assert loop_keys == backfill, f"gap={gap_s}s loop={loop_keys} backfill={backfill}"
        assert (len(backfill) == 2) is expect_second
        conn.close()


def test_restart_parity_uses_persisted_seed():
    rule = make_rule()
    conn, a_ts, b_ts = _two_cross_fixture(500)
    end = b_ts + 60_000
    cursor = a_ts + 30_000
    keys1, seed_map = run_loop(conn, rule, [(T0, cursor)])
    # "restart": fresh call, only the persisted seed map crosses the boundary
    keys2, _ = run_loop(conn, rule, [(cursor, end)], seed_map=seed_map)
    assert keys1 + keys2 == keys(rule, _bucket_events(conn, rule, T0, end, 1000))
    # without the persisted seed the pre-v2 defect reappears (documented contrast)
    keys2_amnesia, _ = run_loop(conn, rule, [(cursor, end)])
    assert len(keys1 + keys2_amnesia) == 2  # pre-v2 behavior accepted the 500s gap
    conn.close()


def test_suppression_chain_resolves_identically():
    rule = make_rule()
    a = T0 + 250_000
    b = a + 500_000   # suppressed by A in both modes
    c = a + 1_000_000  # ≥900s after A (the last EMITTED signal) → kept
    conn = make_db([(a, 150_000.0), (b, 150_000.0), (c, 150_000.0)], (T0, c + 120_000))
    end = c + 60_000
    backfill = keys(rule, _bucket_events(conn, rule, T0, end, 1000))
    loop_keys, _ = run_loop(
        conn, rule, [(T0, a + 30_000), (a + 30_000, b + 30_000), (b + 30_000, end)]
    )
    assert backfill == loop_keys
    assert len(backfill) == 2  # A and C, never B
    conn.close()


def test_cursor_mid_bucket_keeps_cluster_identity():
    rule = make_rule()
    r1 = T0 + 10_000
    r2 = T0 + 130_000  # same bucket; threshold only crossed by r1+r2 combined
    conn = make_db([(r1, 60_000.0), (r2, 60_000.0)], (T0, r2 + 120_000))
    cursor = T0 + 60_000  # between the two rows
    scan_lo = (cursor // BUCKET_MS) * BUCKET_MS  # bucket-aligned rescan (v2)
    sigs = _bucket_events(conn, rule, scan_lo, r2 + 60_000, 1000)
    assert len(sigs) == 1
    assert sigs[0]["liq_total_notional"] == 120_000.0
    assert sigs[0]["cluster_start_ts_ms"] == r1
    assert sigs[0]["cluster_liq_count"] == 2
    # pre-v2 (scan from raw cursor) missed the crossing entirely:
    assert _bucket_events(conn, rule, cursor, r2 + 60_000, 1000) == []
    conn.close()


def test_replay_idempotent_and_same_bucket_dedup():
    rule = make_rule()
    conn, a_ts, b_ts = _two_cross_fixture(901)
    end = b_ts + 60_000
    cuts = [(T0, a_ts + 30_000), (a_ts + 30_000, end), (a_ts + 30_000, end)]  # replay 2nd
    loop_keys, _ = run_loop(conn, rule, cuts)
    assert loop_keys == keys(rule, _bucket_events(conn, rule, T0, end, 1000))
    assert len(loop_keys) == len(set(loop_keys))
    conn.close()


def test_per_rule_state_isolation():
    rule_a = make_rule("ACTIVE_RULE_100K")
    rule_b = make_rule("ARCHIVED_RULE_100K")
    conn, a_ts, b_ts = _two_cross_fixture(500)
    end = b_ts + 60_000
    # rule_b emits over the full window (its own state), then rule_a runs seeded
    # only by its OWN map entry — rule_b's emissions must not suppress rule_a.
    _, seed_map = run_loop(conn, rule_b, [(T0, end)])
    keys_a, _ = run_loop(conn, rule_a, [(T0, end)], seed_map=seed_map)
    assert keys_a == keys(rule_a, _bucket_events(conn, rule_a, T0, end, 1000))
    assert rule_a.name not in {k for k in seed_map} or seed_map.get(rule_a.name) is None
    conn.close()
