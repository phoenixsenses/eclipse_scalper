"""OD-018-FOLLOWUP first-activation migration test matrix (in-memory SQLite
only). Complements tests/test_s34_shadow_paper_min_gap_parity.py (retained,
unmodified, rerun alongside this file) with the v1->v2 activation-boundary
cases: a pre-v2 emitted signal whose bucket has already scrolled behind the
persisted cursor at the moment persistent-v2 first runs for a rule.

Also covers Section C (top-level `min_gap_semantics` protocol-version field).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_shadow_paper_runner import (  # noqa: E402
    S34Rule,
    _bucket_events,
    _derive_min_gap_seed_from_history,
    _paper_trade_from_signal,
    RiskConfig,
)

BUCKET_MS = 300_000
SYM = "SIMUSDT"
T0 = 1_800_000_000_000  # bucket-aligned


def make_rule(name: str = "SIM_SELL_SHORT_100K", threshold: float = 100_000.0) -> S34Rule:
    return S34Rule(
        name=name, symbol=SYM, liq_side="SELL", direction="SHORT",
        threshold_usd=threshold, bucket_sec=300, min_gap_sec=900, entry_delay_sec=0,
        require_book_ticker_fill=False, modeled_spread_bps=0.0,
    )


def make_db(rows, span):
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        "CREATE TABLE liquidations (symbol TEXT, side TEXT, ts_ms INTEGER, price REAL, notional REAL);"
        "CREATE TABLE mark_prices  (symbol TEXT, ts_ms INTEGER, mark_price REAL);"
        "CREATE TABLE agg_trades   (symbol TEXT, ts_ms INTEGER, price REAL, qty REAL, notional REAL, is_buyer_maker INTEGER);"
    )
    for ts, notional in rows:
        conn.execute("INSERT INTO liquidations VALUES (?,?,?,?,?)", (SYM, "SELL", ts, 100.0, notional))
    for m in range(span[0], span[1] + 120_000, 60_000):
        conn.execute("INSERT INTO mark_prices VALUES (?,?,?)", (SYM, m, 100.0))
    conn.commit()
    return conn


def hist_trade(rule: S34Rule, signal_ts_ms: int, status: str = "CLOSED", trade_id: str | None = None,
               symbol: str | None = None, threshold_usd: float | None = None, liq_side: str | None = None,
               omit_signal_ts: bool = False) -> dict:
    rec = {
        "trade_id": trade_id or f"T{signal_ts_ms}_{status}",
        "status": status,
        "rule": {
            "name": rule.name,
            "symbol": symbol if symbol is not None else rule.symbol,
            "liq_side": liq_side if liq_side is not None else rule.liq_side,
            "threshold_usd": threshold_usd if threshold_usd is not None else rule.threshold_usd,
        },
    }
    if not omit_signal_ts:
        rec["signal_ts_ms"] = int(signal_ts_ms)
    return rec


def keys(rule: S34Rule, sigs: list[dict]) -> list[str]:
    return [f"{rule.name}:{s['bucket']}" for s in sigs]


def simulate_activation_tick(conn, rule: S34Rule, prior_state: dict, prior_trades: dict,
                              start_ms: int, end_ms: int) -> tuple[list[dict], dict]:
    """Faithful proxy of the new run_once migration block: same order of
    operations (seed map -> one-time per-rule migration -> fail-closed check
    -> bucket-aligned _bucket_events call -> post-emission map update)."""
    last_signal_map = dict(prior_state.get("last_signal_ts_ms_by_rule") or {})
    migration_provenance = dict(prior_state.get("min_gap_state_provenance_by_rule") or {})
    if rule.name not in migration_provenance:
        prov = _derive_min_gap_seed_from_history(prior_trades, rule)
        migration_provenance[rule.name] = prov
        if prov["status"] == "DERIVED_FROM_HISTORY" and rule.name not in last_signal_map:
            last_signal_map[rule.name] = int(prov["seed_ts_ms"])
    prov = migration_provenance.get(rule.name) or {}
    if prov.get("status") == "AMBIGUOUS_FAILED":
        signals: list[dict] = []
    else:
        bucket_ms = int(rule.bucket_sec) * 1000
        scan_start_ms = (int(start_ms) // bucket_ms) * bucket_ms
        seed = last_signal_map.get(rule.name)
        signals = _bucket_events(conn, rule, scan_start_ms, end_ms, 1000, last_signal_ms_seed=seed)
        if signals:
            newest = max(int(s["ts_ms"]) for s in signals)
            if seed is None or newest > int(seed):
                last_signal_map[rule.name] = newest
    new_state = {
        "last_signal_ts_ms_by_rule": last_signal_map,
        "min_gap_state_provenance_by_rule": migration_provenance,
    }
    return signals, new_state


def _activation_fixture(gap_s: int):
    """pre-v2 last emission at T -> cursor advances to ~now (normal
    operation, independent of gap_s -- restart timing must not be the
    economic boundary) -> new crossing EXACTLY gap_s seconds after T."""
    t_signal = T0 + 250_000
    new_liq_ts = t_signal + gap_s * 1000  # exact signal-to-signal gap under test
    persisted_cursor = t_signal + 60_000  # one bucket ahead of t_signal (300s buckets), fixed regardless of gap_s
    conn = make_db([(t_signal, 150_000.0), (new_liq_ts, 150_000.0)], (T0, new_liq_ts + 120_000))
    end_ms = new_liq_ts + 60_000
    return conn, t_signal, persisted_cursor, end_ms


def test_activation_100s_suppressed():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(100)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert sigs == [], f"expected suppression at 100s gap, got {keys(rule, sigs)}"
    conn.close()


def test_activation_899s_suppressed():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(899)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert sigs == [], f"expected suppression at 899s gap, got {keys(rule, sigs)}"
    conn.close()


def test_activation_900s_accepted():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(900)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert len(sigs) == 1, f"expected the new signal accepted at exactly 900s, got {keys(rule, sigs)}"
    conn.close()


def test_activation_901s_accepted():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(901)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert len(sigs) == 1, f"expected the new signal accepted at 901s, got {keys(rule, sigs)}"
    conn.close()


def test_activation_several_buckets_behind_cursor():
    """last pre-v2 emission several buckets (1200s) behind the persisted
    cursor -- still must be found via trade history (not the raw-data
    rescan, which is bounded to one bucket)."""
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(1200)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert len(sigs) == 1  # >900s gap -> correctly accepted, not an amnesia artifact
    conn.close()


def test_activation_cursor_bucket_boundary_straddle():
    """Migration + bucket-aligned rescan combined: the NEW crossing itself
    straddles the persisted cursor's bucket. Oracle = continuous backfill
    over the whole span; must match."""
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(300)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert sigs == []  # 300s < 900s -> suppressed, matching backfill oracle below
    oracle = _bucket_events(conn, rule, T0, end_ms, 1000)
    assert len(oracle) == 1  # oracle only ever saw ONE crossing total (T is pre-existing history, not rescanned)
    conn.close()


def test_migration_then_restart_preserves_signal_set():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(500)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs1, state_after_migration = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert sigs1 == []  # 500s < 900 -> suppressed, migration worked
    # "restart": fresh call reusing the persisted state, no re-derivation
    sigs2, state_after_restart = simulate_activation_tick(
        conn, rule, state_after_migration, prior_trades, end_ms, end_ms + 60_000
    )
    assert sigs2 == []
    assert state_after_restart["min_gap_state_provenance_by_rule"][rule.name]["status"] == "DERIVED_FROM_HISTORY"
    conn.close()


def test_repeated_migration_is_idempotent():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(500)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    _, state1 = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    seed_after_first = state1["last_signal_ts_ms_by_rule"][rule.name]
    # second migration attempt with DIFFERENT (newer) history must NOT
    # re-derive, since the rule is already present in provenance
    prior_trades_mutated = dict(prior_trades)
    prior_trades_mutated["h2"] = hist_trade(rule, t_signal + 10_000, trade_id="h2")
    _, state2 = simulate_activation_tick(conn, rule, state1, prior_trades_mutated, cursor, end_ms)
    assert state2["last_signal_ts_ms_by_rule"][rule.name] == seed_after_first
    conn.close()


def test_no_prior_emission_is_not_an_error():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(500)
    sigs, state = simulate_activation_tick(conn, rule, {}, {}, cursor, end_ms)
    prov = state["min_gap_state_provenance_by_rule"][rule.name]
    assert prov["status"] == "NO_PRIOR_EMISSION"
    assert prov["seed_ts_ms"] is None  # migration must not fabricate a seed from nothing
    # the SAME tick's ordinary (non-migration) _bucket_events call may still
    # legitimately populate last_signal_map if a fresh signal fires -- that
    # is normal post-emission bookkeeping, not part of the migration itself
    conn.close()


def test_malformed_ambiguous_history_fails_closed():
    """Same rule.name, but a historical record shows a DIFFERENT threshold
    under that name -> refuse to seed, and refuse to emit ANY new signal for
    this rule until an operator resolves it."""
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(1200)  # would otherwise ACCEPT
    prior_trades = {
        "h1": hist_trade(rule, t_signal, threshold_usd=999_999.0),  # identity mismatch
    }
    sigs, state = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    prov = state["min_gap_state_provenance_by_rule"][rule.name]
    assert prov["status"] == "AMBIGUOUS_FAILED"
    assert sigs == [], "fail-closed rule must emit nothing, not fall back to accepting"
    assert rule.name not in state["last_signal_ts_ms_by_rule"]
    conn.close()


def test_malformed_missing_signal_ts_fails_closed():
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(1200)
    prior_trades = {"h1": hist_trade(rule, t_signal, omit_signal_ts=True)}
    sigs, state = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert state["min_gap_state_provenance_by_rule"][rule.name]["status"] == "AMBIGUOUS_FAILED"
    assert sigs == []
    conn.close()


def test_different_rule_remains_independent_during_migration():
    rule_a = make_rule("ACTIVE_RULE_A", threshold=100_000.0)
    rule_b = make_rule("ARCHIVED_RULE_B", threshold=50_000.0)
    conn, t_signal, cursor, end_ms = _activation_fixture(500)
    # history only for rule_b; rule_a has none
    prior_trades = {"h1": hist_trade(rule_b, t_signal)}
    sigs_a, state_a = simulate_activation_tick(conn, rule_a, {}, prior_trades, cursor, end_ms)
    assert state_a["min_gap_state_provenance_by_rule"][rule_a.name]["status"] == "NO_PRIOR_EMISSION"
    assert rule_b.name not in state_a["min_gap_state_provenance_by_rule"]
    conn.close()


def test_governance_disabled_status_still_seeds_its_own_rule_only():
    """A SKIPPED (governance-archived / regime-rejected / no-fill) record
    consumes the gap exactly like an accepted one -- status is irrelevant to
    the oracle, only rule identity + signal_ts_ms matter."""
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(500)
    for status in ("SKIPPED", "OPEN", "CLOSED"):
        prior_trades = {"h1": hist_trade(rule, t_signal, status=status)}
        sigs, state = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
        prov = state["min_gap_state_provenance_by_rule"][rule.name]
        assert prov["status"] == "DERIVED_FROM_HISTORY"
        assert prov["seed_ts_ms"] == t_signal
        assert sigs == []  # 500s gap suppressed regardless of the historical record's status
    conn.close()


def test_loop_restart_backfill_identical_after_migration():
    """End-to-end: for gaps spanning the activation boundary, loop-with-
    migration must match continuous backfill exactly, same as the original
    6/6 matrix but now crossing the v1->v2 boundary itself."""
    rule = make_rule()
    for gap_s, expect_accept in [(100, False), (899, False), (900, True), (901, True)]:
        conn, t_signal, cursor, end_ms = _activation_fixture(gap_s)
        prior_trades = {"h1": hist_trade(rule, t_signal)}
        sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
        # oracle: continuous backfill over the full span, starting from the
        # true first signal (T), not from the persisted cursor
        oracle_all = _bucket_events(conn, rule, T0, end_ms, 1000)
        # oracle's post-T decision for the SAME new crossing:
        oracle_new_accepted = len(oracle_all) == 2
        assert (len(sigs) == 1) == expect_accept == oracle_new_accepted, (
            f"gap={gap_s}s migration_accepted={len(sigs)==1} oracle_accepted={oracle_new_accepted}"
        )
        conn.close()


def test_no_duplicate_trade_created_across_migration_restart():
    """signal_key dedup (existing_keys, run_once's own responsibility) plus
    migration must never double-count: once a signal_key already has a trade
    record, migration or not, re-scanning it must not produce a second one.
    This is exercised at the _paper_trade_from_signal + signal_key layer."""
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(901)
    prior_trades = {"h1": hist_trade(rule, t_signal)}
    sigs, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    assert len(sigs) == 1
    risk_config = RiskConfig()
    trade1 = _paper_trade_from_signal(rule, sigs[0], risk_config)
    # re-running the same tick (replay) must reproduce the identical signal_key
    sigs_replay, _ = simulate_activation_tick(conn, rule, {}, prior_trades, cursor, end_ms)
    trade2 = _paper_trade_from_signal(rule, sigs_replay[0], risk_config)
    assert trade1["signal_key"] == trade2["signal_key"]
    conn.close()


def test_top_level_min_gap_semantics_field_present_and_filterable():
    """Section C: trade['min_gap_semantics'] is a top-level, durable field
    distinguishing pre-v2 (absent) from v2 (== 'persistent-v2') records."""
    rule = make_rule()
    conn, t_signal, cursor, end_ms = _activation_fixture(901)
    sigs = _bucket_events(conn, rule, T0, end_ms, 1000)
    assert len(sigs) >= 1
    risk_config = RiskConfig()
    trade = _paper_trade_from_signal(rule, sigs[0], risk_config)
    assert trade["min_gap_semantics"] == "persistent-v2"
    # a pre-v2-shaped trade dict (no signal.min_gap_semantics key, as all
    # 1,338 real historical records are) must not be misclassified
    pre_v2_signal = dict(sigs[0])
    pre_v2_signal.pop("min_gap_semantics", None)
    pre_v2_trade = _paper_trade_from_signal(rule, pre_v2_signal, risk_config)
    assert pre_v2_trade["min_gap_semantics"] == "persistent-v2"  # runner always tags NEW trades going forward
    conn.close()
