"""S34 intelligence ledger.

The ledger is a small append/upsert SQLite database that records what the
shadow runner saw, why it accepted or rejected a signal, and how each accepted
paper trade resolved. It intentionally lives outside microstructure.db.
"""

from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LEDGER_PATH = Path("data/s34_intelligence.db")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def connect(path: str | Path = DEFAULT_LEDGER_PATH) -> sqlite3.Connection:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS s34_signals (
            signal_id TEXT PRIMARY KEY,
            first_seen_utc TEXT NOT NULL,
            signal_ts_ms INTEGER NOT NULL,
            signal_ts_utc TEXT NOT NULL,
            bucket_ts_ms INTEGER,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            liq_side TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            cluster_notional REAL,
            cluster_liq_count INTEGER,
            cluster_shape_label TEXT,
            features_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_signals_ts ON s34_signals(signal_ts_ms);
        CREATE INDEX IF NOT EXISTS idx_s34_signals_rule ON s34_signals(rule_name, signal_ts_ms);

        CREATE TABLE IF NOT EXISTS s34_decisions (
            decision_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            trade_id TEXT,
            decision_ts_utc TEXT NOT NULL,
            signal_ts_ms INTEGER NOT NULL,
            rule_name TEXT NOT NULL,
            decision TEXT NOT NULL,
            reason TEXT,
            context_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_decisions_ts ON s34_decisions(signal_ts_ms);
        CREATE INDEX IF NOT EXISTS idx_s34_decisions_rule ON s34_decisions(rule_name, decision);

        CREATE TABLE IF NOT EXISTS s34_rejected_signals (
            decision_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            trade_id TEXT,
            rejected_ts_utc TEXT NOT NULL,
            signal_ts_ms INTEGER NOT NULL,
            rule_name TEXT NOT NULL,
            reason TEXT NOT NULL,
            context_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_rejected_reason ON s34_rejected_signals(reason, signal_ts_ms);

        CREATE TABLE IF NOT EXISTS s34_trades (
            trade_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            status TEXT NOT NULL,
            opened_at_utc TEXT,
            entry_ts_ms INTEGER,
            entry_price REAL,
            tp_price REAL,
            sl_price REAL,
            be_trigger_price REAL,
            exit_ts_ms INTEGER,
            exit_reason TEXT,
            exit_price REAL,
            net_bps REAL,
            trade_json TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_trades_rule_status ON s34_trades(rule_name, status);
        CREATE INDEX IF NOT EXISTS idx_s34_trades_entry ON s34_trades(entry_ts_ms);

        CREATE TABLE IF NOT EXISTS s34_outcomes (
            trade_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            outcome_ts_utc TEXT NOT NULL,
            exit_ts_ms INTEGER NOT NULL,
            exit_reason TEXT NOT NULL,
            gross_bps REAL,
            entry_adverse_bps REAL,
            exit_adverse_bps REAL,
            spread_cost_bps REAL,
            fee_cost_bps REAL,
            net_bps REAL,
            outcome_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_outcomes_rule ON s34_outcomes(rule_name, exit_ts_ms);

        CREATE TABLE IF NOT EXISTS s34_predictions (
            prediction_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            predicted_at_utc TEXT NOT NULL,
            prediction_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS s34_model_audit (
            audit_id TEXT PRIMARY KEY,
            signal_id TEXT,
            model_name TEXT NOT NULL,
            audit_ts_utc TEXT NOT NULL,
            audit_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS s34_model_guardrails (
            guardrail_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            guardrail_ts_utc TEXT NOT NULL,
            level TEXT NOT NULL,
            headline TEXT NOT NULL,
            guardrail_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_model_guardrails_signal ON s34_model_guardrails(signal_id);
        CREATE INDEX IF NOT EXISTS idx_s34_model_guardrails_level ON s34_model_guardrails(level, guardrail_ts_utc);

        CREATE TABLE IF NOT EXISTS s34_shadow_guardrails (
            shadow_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            guardrail_name TEXT NOT NULL,
            shadow_ts_utc TEXT NOT NULL,
            action TEXT NOT NULL,
            level TEXT NOT NULL,
            headline TEXT NOT NULL,
            shadow_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_shadow_guardrails_signal ON s34_shadow_guardrails(signal_id);
        CREATE INDEX IF NOT EXISTS idx_s34_shadow_guardrails_name ON s34_shadow_guardrails(guardrail_name, action, shadow_ts_utc);

        CREATE TABLE IF NOT EXISTS s34_ab_results (
            result_id TEXT PRIMARY KEY,
            signal_id TEXT NOT NULL,
            model_a TEXT NOT NULL,
            model_b TEXT NOT NULL,
            result_ts_utc TEXT NOT NULL,
            result_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS s34_prediction_errors (
            error_id TEXT PRIMARY KEY,
            trade_id TEXT NOT NULL,
            signal_id TEXT NOT NULL,
            prediction_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_version TEXT,
            rule_id TEXT,
            predicted_net_bps REAL,
            realized_net_bps REAL,
            forecast_error_bps REAL,
            abs_error_bps REAL,
            direction_correct INTEGER,
            predicted_at_utc TEXT,
            outcome_at_utc TEXT,
            scored_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_prediction_errors_model ON s34_prediction_errors(model_id, scored_at_utc);
        CREATE INDEX IF NOT EXISTS idx_s34_prediction_errors_signal ON s34_prediction_errors(signal_id);

        CREATE TABLE IF NOT EXISTS s34_cascade_correlations (
            correlation_id TEXT PRIMARY KEY,
            signal_id_a TEXT NOT NULL,
            signal_id_b TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            signal_ts_gap_sec REAL NOT NULL,
            notional_a REAL,
            notional_b REAL,
            correlation_type TEXT NOT NULL,
            tagged_at_utc TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_s34_cascade_correlations_a ON s34_cascade_correlations(signal_id_a);
        CREATE INDEX IF NOT EXISTS idx_s34_cascade_correlations_b ON s34_cascade_correlations(signal_id_b);
        CREATE INDEX IF NOT EXISTS idx_s34_cascade_correlations_type ON s34_cascade_correlations(correlation_type, tagged_at_utc);

        CREATE TABLE IF NOT EXISTS s34_shadow_geometry_tags (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT NOT NULL,
            trade_id     TEXT NOT NULL,
            rule_name    TEXT NOT NULL,
            tag          TEXT NOT NULL,
            reasons      TEXT,
            cascade_usd  REAL,
            liq_count    INTEGER,
            single_share REAL,
            net_bps      REAL,
            exit_reason  TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_sgt_trade ON s34_shadow_geometry_tags(trade_id, tag);
        """
    )
    conn.commit()


def correlation_id_for(signal_id_a: str, signal_id_b: str) -> str:
    left, right = sorted([str(signal_id_a), str(signal_id_b)])
    return hashlib.sha256(f"{left}|{right}".encode("utf-8")).hexdigest()[:16]


def classify_signal_gap(gap_sec: float) -> str | None:
    if gap_sec <= 300:
        return "SAME_CASCADE"
    if gap_sec <= 900:
        return "ADJACENT_CASCADE"
    return None


def record_cascade_correlation(
    conn: sqlite3.Connection,
    signal_a: sqlite3.Row | dict[str, Any],
    signal_b: sqlite3.Row | dict[str, Any],
) -> bool:
    a = dict(signal_a)
    b = dict(signal_b)
    if str(a.get("signal_id")) == str(b.get("signal_id")):
        return False
    if str(a.get("symbol")) != str(b.get("symbol")) or str(a.get("direction")) != str(b.get("direction")):
        return False
    gap_sec = abs(int(a.get("signal_ts_ms") or 0) - int(b.get("signal_ts_ms") or 0)) / 1000.0
    corr_type = classify_signal_gap(gap_sec)
    if corr_type is None:
        return False
    left_id, right_id = sorted([str(a.get("signal_id")), str(b.get("signal_id"))])
    left = a if str(a.get("signal_id")) == left_id else b
    right = b if str(b.get("signal_id")) == right_id else a
    conn.execute(
        """
        INSERT OR IGNORE INTO s34_cascade_correlations (
            correlation_id, signal_id_a, signal_id_b, symbol, direction,
            signal_ts_gap_sec, notional_a, notional_b, correlation_type, tagged_at_utc
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            correlation_id_for(left_id, right_id),
            left_id,
            right_id,
            str(left.get("symbol") or ""),
            str(left.get("direction") or ""),
            gap_sec,
            _maybe_float(left.get("cluster_notional")),
            _maybe_float(right.get("cluster_notional")),
            corr_type,
            utc_now_iso(),
        ),
    )
    return True


def tag_recent_cascade_correlations(conn: sqlite3.Connection, signal_id: str, lookback_sec: int = 900) -> int:
    current = conn.execute("SELECT * FROM s34_signals WHERE signal_id=?", (signal_id,)).fetchone()
    if current is None:
        return 0
    min_ts = int(current["signal_ts_ms"]) - int(lookback_sec) * 1000
    max_ts = int(current["signal_ts_ms"]) + int(lookback_sec) * 1000
    peers = conn.execute(
        """
        SELECT * FROM s34_signals
        WHERE signal_id != ?
          AND symbol=?
          AND direction=?
          AND signal_ts_ms BETWEEN ? AND ?
        """,
        (signal_id, current["symbol"], current["direction"], min_ts, max_ts),
    ).fetchall()
    written = 0
    for peer in peers:
        if record_cascade_correlation(conn, current, peer):
            written += 1
    return written


def get_correlated_signals(signal_id: str, db_path: str | Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT
              c.*,
              sa.signal_ts_utc AS signal_a_utc,
              sb.signal_ts_utc AS signal_b_utc,
              ta.trade_id AS trade_a_id,
              ta.status AS trade_a_status,
              tb.trade_id AS trade_b_id,
              tb.status AS trade_b_status
            FROM s34_cascade_correlations c
            JOIN s34_signals sa ON sa.signal_id=c.signal_id_a
            JOIN s34_signals sb ON sb.signal_id=c.signal_id_b
            LEFT JOIN s34_trades ta ON ta.signal_id=c.signal_id_a
            LEFT JOIN s34_trades tb ON tb.signal_id=c.signal_id_b
            WHERE (c.signal_id_a=? OR c.signal_id_b=?)
              AND c.correlation_type='SAME_CASCADE'
            ORDER BY c.signal_ts_gap_sec ASC
            """,
            (signal_id, signal_id),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def signal_id_for(rule_name: str, signal: dict[str, Any]) -> str:
    bucket = int(signal.get("bucket") or signal.get("ts_ms") or 0)
    return f"{rule_name}:{bucket}"


def signal_id_from_trade(trade: dict[str, Any]) -> str:
    raw = str(trade.get("signal_key") or "")
    if raw:
        return raw
    rule_name = str((trade.get("rule") or {}).get("name") or "UNKNOWN")
    signal = trade.get("signal") or {}
    return signal_id_for(rule_name, signal)


def record_signal(conn: sqlite3.Connection, rule: Any, signal: dict[str, Any]) -> str:
    rule_name = str(getattr(rule, "name", None) or (rule.get("name") if isinstance(rule, dict) else "UNKNOWN"))
    signal_id = signal_id_for(rule_name, signal)
    conn.execute(
        """
        INSERT OR IGNORE INTO s34_signals (
            signal_id, first_seen_utc, signal_ts_ms, signal_ts_utc, bucket_ts_ms,
            symbol, direction, liq_side, rule_name, cluster_notional,
            cluster_liq_count, cluster_shape_label, features_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            signal_id,
            utc_now_iso(),
            int(signal.get("ts_ms") or 0),
            str(signal.get("ts_utc") or ""),
            int(signal.get("bucket") or signal.get("ts_ms") or 0),
            str(getattr(rule, "symbol", None) or (rule.get("symbol") if isinstance(rule, dict) else signal.get("symbol") or "")),
            str(getattr(rule, "direction", None) or (rule.get("direction") if isinstance(rule, dict) else "")),
            str(getattr(rule, "liq_side", None) or (rule.get("liq_side") if isinstance(rule, dict) else "")),
            rule_name,
            float(signal.get("liq_total_notional") or 0.0),
            int(signal.get("liq_count") or 0),
            str(signal.get("cluster_shape_label") or ""),
            _json(signal),
        ),
    )
    tag_recent_cascade_correlations(conn, signal_id)
    return signal_id


def record_decision(conn: sqlite3.Connection, trade: dict[str, Any], decision: str, reason: str = "") -> None:
    rule_name = str((trade.get("rule") or {}).get("name") or "UNKNOWN")
    signal_id = signal_id_from_trade(trade)
    trade_id = str(trade.get("trade_id") or "")
    decision_id = f"{trade_id or signal_id}:{decision}:{reason or 'OK'}"
    signal_ts_ms = int(trade.get("signal_ts_ms") or (trade.get("signal") or {}).get("ts_ms") or 0)
    payload = {
        "status": trade.get("status"),
        "risk_gate_status": trade.get("risk_gate_status"),
        "risk_gate_reason": trade.get("risk_gate_reason"),
        "regime": trade.get("regime"),
        "cluster_owner_trade_id": trade.get("cluster_owner_trade_id"),
        "cluster_owner_rule": trade.get("cluster_owner_rule"),
        "fill_error": trade.get("fill_error"),
    }
    conn.execute(
        """
        INSERT OR REPLACE INTO s34_decisions (
            decision_id, signal_id, trade_id, decision_ts_utc, signal_ts_ms,
            rule_name, decision, reason, context_json
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (decision_id, signal_id, trade_id, utc_now_iso(), signal_ts_ms, rule_name, decision, reason, _json(payload)),
    )
    if decision == "REJECT":
        conn.execute(
            """
            INSERT OR REPLACE INTO s34_rejected_signals (
                decision_id, signal_id, trade_id, rejected_ts_utc, signal_ts_ms,
                rule_name, reason, context_json
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (decision_id, signal_id, trade_id, utc_now_iso(), signal_ts_ms, rule_name, reason or "UNKNOWN", _json(payload)),
        )


def record_trade(conn: sqlite3.Connection, trade: dict[str, Any]) -> None:
    rule_name = str((trade.get("rule") or {}).get("name") or "UNKNOWN")
    signal_id = signal_id_from_trade(trade)
    conn.execute(
        """
        INSERT OR REPLACE INTO s34_trades (
            trade_id, signal_id, rule_name, symbol, direction, status, opened_at_utc,
            entry_ts_ms, entry_price, tp_price, sl_price, be_trigger_price,
            exit_ts_ms, exit_reason, exit_price, net_bps, trade_json, updated_at_utc
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            str(trade.get("trade_id") or ""),
            signal_id,
            rule_name,
            str(trade.get("symbol") or ""),
            str(trade.get("direction") or ""),
            str(trade.get("status") or ""),
            trade.get("opened_at_utc"),
            int(trade.get("entry_ts_ms") or 0),
            _maybe_float(trade.get("entry_price")),
            _maybe_float(trade.get("tp_price")),
            _maybe_float(trade.get("sl_price")),
            _maybe_float(trade.get("be_trigger_price")),
            int(trade.get("exit_ts_ms") or 0) or None,
            trade.get("exit_reason"),
            _maybe_float(trade.get("exit_price")),
            _maybe_float(trade.get("net_bps")),
            _json(trade),
            utc_now_iso(),
        ),
    )


def record_outcome(conn: sqlite3.Connection, trade: dict[str, Any]) -> None:
    if trade.get("status") != "CLOSED" or not trade.get("trade_id"):
        return
    rule_name = str((trade.get("rule") or {}).get("name") or "UNKNOWN")
    signal_id = signal_id_from_trade(trade)
    conn.execute(
        """
        INSERT OR REPLACE INTO s34_outcomes (
            trade_id, signal_id, rule_name, outcome_ts_utc, exit_ts_ms, exit_reason,
            gross_bps, entry_adverse_bps, exit_adverse_bps, spread_cost_bps,
            fee_cost_bps, net_bps, outcome_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            str(trade.get("trade_id") or ""),
            signal_id,
            rule_name,
            utc_now_iso(),
            int(trade.get("exit_ts_ms") or 0),
            str(trade.get("exit_reason") or ""),
            _maybe_float(trade.get("gross_bps")),
            _maybe_float(trade.get("entry_adverse_bps")),
            _maybe_float(trade.get("exit_adverse_bps")),
            _maybe_float(trade.get("spread_cost_bps")),
            _maybe_float(trade.get("fee_cost_bps")),
            _maybe_float(trade.get("net_bps")),
            _json(trade),
        ),
    )


def record_trade_lifecycle(conn: sqlite3.Connection, trade: dict[str, Any], decision: str, reason: str = "") -> None:
    record_decision(conn, trade, decision, reason)
    if trade.get("status") in {"OPEN", "CLOSED"}:
        record_trade(conn, trade)
    if trade.get("status") == "CLOSED":
        record_outcome(conn, trade)
        _tag_eth_weak_signal(conn, trade)
        _tag_btc_microtrend(conn, trade)
        _tag_market_context(conn, trade)


_ETH_500K_RULE = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30"
_ETH_WEAK_CNT_TAG   = "ETH_WEAK_COUNT_SHADOW"
_ETH_HIGH_SHARE_TAG = "ETH_HIGH_SHARE_SHADOW"

_BTC_MICRO_COLD_TAG  = "BTC_MICRO_COLD_SHADOW"   # BTC 10s ret < 2 bps
_BTC_MICRO_SWEET_TAG = "BTC_MICRO_SWEET_SHADOW"   # BTC 10s ret 2-5 bps  (sweet spot)
_BTC_MICRO_HOT_TAG   = "BTC_MICRO_HOT_SHADOW"     # BTC 10s ret >= 5 bps
_MICRO_DB_PATH       = Path("data/microstructure.db")


def _tag_eth_weak_signal(conn: sqlite3.Connection, trade: dict[str, Any]) -> None:
    """Write ETH weak-signal shadow tags on trade close. Shadow observe only."""
    rule = str((trade.get("rule") or {}).get("name") or trade.get("rule_name") or "")
    if rule != _ETH_500K_RULE:
        return
    tid = trade.get("trade_id")
    if not tid:
        return
    sig = trade.get("signal") or {}
    cnt   = sig.get("liq_count")
    cas   = sig.get("liq_total_notional")
    mx    = sig.get("liq_max_notional")
    share = (mx / cas * 100) if (cas and mx and cas > 0) else None
    net   = _maybe_float(trade.get("net_bps"))
    exit_r = trade.get("exit_reason")

    tags_to_write = []
    reasons = []
    if cnt is not None and cnt <= 7:
        reasons.append(f"cnt={cnt}<=7")
        tags_to_write.append(_ETH_WEAK_CNT_TAG)
    if share is not None and share >= 80.0:
        reasons.append(f"share={share:.0f}%>=80%")
        tags_to_write.append(_ETH_HIGH_SHARE_TAG)

    if not tags_to_write:
        return

    now = utc_now_iso()
    for tag in tags_to_write:
        conn.execute(
            """
            INSERT INTO s34_shadow_geometry_tags
              (created_at, trade_id, rule_name, tag, reasons,
               cascade_usd, liq_count, single_share, net_bps, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(trade_id, tag) DO UPDATE SET
              created_at  = excluded.created_at,
              reasons     = excluded.reasons,
              net_bps     = excluded.net_bps,
              exit_reason = excluded.exit_reason
            """,
            (now, tid, _ETH_500K_RULE, tag, ";".join(reasons),
             cas, cnt, share, net, exit_r),
        )


def _tag_btc_microtrend(conn: sqlite3.Connection, trade: dict[str, Any]) -> None:
    """Tag ETH 500K trades with BTC 10s micro-trend at entry. Shadow observe only.

    Historical backtest (N=348) shows:
      COLD (<2 bps):  WR=35%  SL=32%  -- weak momentum
      SWEET (2-5):    WR=70%  SL=12%  -- confirmed but not overheated
      HOT  (>=5):     WR=52%  SL=23%  -- overextended, ETH catch-up slows
    All three tagged so OOS evidence accumulates without blocking anything.
    """
    rule = str((trade.get("rule") or {}).get("name") or trade.get("rule_name") or "")
    if rule != _ETH_500K_RULE:
        return
    tid = trade.get("trade_id")
    entry_ms = (trade.get("signal") or {}).get("entry_ts_ms") or trade.get("entry_ts_ms")
    if not tid or not entry_ms:
        return

    try:
        micro = sqlite3.connect(f"file:{_MICRO_DB_PATH}?mode=ro", uri=True)
        r_before = micro.execute(
            "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (entry_ms - 10_000,)).fetchone()
        r_at = micro.execute(
            "SELECT mark_price FROM mark_prices WHERE symbol='BTCUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
            (entry_ms,)).fetchone()
        micro.close()
    except Exception:
        return

    if not r_before or not r_at or r_before[0] == 0:
        return

    btc_ret = (r_at[0] - r_before[0]) / r_before[0] * 10000  # bps

    if btc_ret < 2.0:
        tag = _BTC_MICRO_COLD_TAG
    elif btc_ret < 5.0:
        tag = _BTC_MICRO_SWEET_TAG
    else:
        tag = _BTC_MICRO_HOT_TAG

    net    = _maybe_float(trade.get("net_bps"))
    exit_r = trade.get("exit_reason")
    now    = utc_now_iso()
    conn.execute(
        """
        INSERT INTO s34_shadow_geometry_tags
          (created_at, trade_id, rule_name, tag, reasons,
           cascade_usd, liq_count, single_share, net_bps, exit_reason)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(trade_id, tag) DO UPDATE SET
          created_at  = excluded.created_at,
          reasons     = excluded.reasons,
          net_bps     = excluded.net_bps,
          exit_reason = excluded.exit_reason
        """,
        (now, tid, _ETH_500K_RULE, tag, f"btc_10s={btc_ret:+.2f}bps",
         None, None, None, net, exit_r),
    )


def record_prediction(
    conn: sqlite3.Connection,
    signal_id: str,
    model_name: str,
    model_version: str,
    prediction: dict[str, Any],
) -> None:
    prediction_id = f"{signal_id}:{model_name}:{model_version}"
    conn.execute(
        """
        INSERT OR REPLACE INTO s34_predictions (
            prediction_id, signal_id, model_name, model_version, predicted_at_utc, prediction_json
        ) VALUES (?,?,?,?,?,?)
        """,
        (prediction_id, signal_id, model_name, model_version, utc_now_iso(), _json(prediction)),
    )


def record_model_audit(
    conn: sqlite3.Connection,
    signal_id: str,
    model_name: str,
    audit: dict[str, Any],
) -> None:
    audit_id = f"{signal_id}:{model_name}:{audit.get('audit_version', 'v1')}"
    conn.execute(
        """
        INSERT OR REPLACE INTO s34_model_audit (
            audit_id, signal_id, model_name, audit_ts_utc, audit_json
        ) VALUES (?,?,?,?,?)
        """,
        (audit_id, signal_id, model_name, utc_now_iso(), _json(audit)),
    )


def record_model_guardrail(conn: sqlite3.Connection, signal_id: str, guardrail: dict[str, Any]) -> None:
    guardrail_id = f"{signal_id}:model_guardrail:{guardrail.get('version', 'v1')}"
    conn.execute(
        """
        INSERT OR REPLACE INTO s34_model_guardrails (
            guardrail_id, signal_id, guardrail_ts_utc, level, headline, guardrail_json
        ) VALUES (?,?,?,?,?,?)
        """,
        (
            guardrail_id,
            signal_id,
            utc_now_iso(),
            str(guardrail.get("level") or "unknown"),
            str(guardrail.get("headline") or ""),
            _json(guardrail),
        ),
    )


def record_shadow_guardrail(conn: sqlite3.Connection, signal_id: str, guardrail: dict[str, Any]) -> None:
    name = str(guardrail.get("name") or "unknown_shadow_guardrail")
    version = str(guardrail.get("version") or "v1")
    shadow_id = f"{signal_id}:{name}:{version}"
    conn.execute(
        """
        INSERT OR REPLACE INTO s34_shadow_guardrails (
            shadow_id, signal_id, guardrail_name, shadow_ts_utc,
            action, level, headline, shadow_json
        ) VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            shadow_id,
            signal_id,
            name,
            utc_now_iso(),
            str(guardrail.get("action") or "observe"),
            str(guardrail.get("level") or "unknown"),
            str(guardrail.get("headline") or ""),
            _json(guardrail),
        ),
    )


def _tag_market_context(conn: sqlite3.Connection, trade: dict[str, Any]) -> None:
    """Tag every closed trade with session, co-cascade type, intraday nth, and gap.

    Four shadow tags written per trade (UPSERT):
      CONTEXT_SESSION     — Asia / Europe / US / Late  (UTC hour bands)
      CONTEXT_CO_CASCADE  — IDIO / PARTIAL / SYSTEMIC  (cross-symbol activity ±60s)
      CONTEXT_INTRADAY_NTH — nth=N  (Nth signal of this rule on this calendar day)
      CONTEXT_GAP_LAST_SEC — gap=Xs  (seconds since previous same-rule closed trade)

    Research basis (2026-06-27):
      Session: Europe 08-14 UTC best for ETH BUY (OOS WR=82%)
      Co-cascade: SELL IDIO → OOS WR=36% SL=27%; SYSTEMIC → OOS WR=75%
      Gap: BUY 5-30min danger zone (all-data SL=27-33%)
      Nth: SELL #1 of day weakest; BUY #5 degrades
    """
    rule = str((trade.get("rule") or {}).get("name") or trade.get("rule_name") or "")
    tid = trade.get("trade_id")
    sig = trade.get("signal") or {}
    entry_ms = sig.get("entry_ts_ms") or trade.get("entry_ts_ms")
    if not tid or not entry_ms:
        return

    entry_ms = int(entry_ms)
    net    = _maybe_float(trade.get("net_bps"))
    exit_r = trade.get("exit_reason")
    now    = utc_now_iso()

    # --- 1. UTC Session ---
    dt = datetime.fromtimestamp(entry_ms / 1000, tz=timezone.utc)
    h  = dt.hour
    if h < 8:
        session = "Asia"
    elif h < 14:
        session = "Europe"
    elif h < 22:
        session = "US"
    else:
        session = "Late"

    # --- 2. Co-cascade type ---
    co_type: str | None = None
    try:
        micro = sqlite3.connect(f"file:{_MICRO_DB_PATH}?mode=ro", uri=True)
        liq_side = sig.get("liq_side") or ("BUY" if "_BUY_" in rule else "SELL")
        symbol   = sig.get("symbol")   or ("SOLUSDT" if "SOL" in rule else "ETHUSDT")
        CO_WIN   = 60_000  # ±60s

        # BTC co-cascade (same direction, $1M threshold)
        btc_row = micro.execute(
            "SELECT COALESCE(SUM(notional),0) FROM liquidations"
            " WHERE symbol='BTCUSDT' AND side=? AND ts_ms BETWEEN ? AND ?",
            (liq_side, entry_ms - CO_WIN, entry_ms + CO_WIN)).fetchone()
        has_btc = (btc_row[0] or 0) >= 1_000_000

        # Cross-asset co-cascade: ETH↔SOL ($100K for SOL, $500K for ETH)
        if symbol == "ETHUSDT":
            co_sym, co_thr = "SOLUSDT", 100_000
        else:
            co_sym, co_thr = "ETHUSDT", 500_000
        cross_row = micro.execute(
            "SELECT COALESCE(SUM(notional),0) FROM liquidations"
            " WHERE symbol=? AND side=? AND ts_ms BETWEEN ? AND ?",
            (co_sym, liq_side, entry_ms - CO_WIN, entry_ms + CO_WIN)).fetchone()
        has_cross = (cross_row[0] or 0) >= co_thr
        micro.close()

        if has_btc and has_cross:
            co_type = "SYSTEMIC"
        elif has_btc or has_cross:
            co_type = "PARTIAL"
        else:
            co_type = "IDIO"
    except Exception:
        pass

    # --- 3. Intraday Nth + gap since last same-rule trade ---
    day_start_ms = int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp() * 1000)
    day_end_ms   = day_start_ms + 86_400_000

    prev = conn.execute(
        "SELECT entry_ts_ms FROM s34_trades"
        " WHERE rule_name=? AND entry_ts_ms>=? AND entry_ts_ms<? AND entry_ts_ms<?"
        " ORDER BY entry_ts_ms",
        (rule, day_start_ms, day_end_ms, entry_ms)).fetchall()

    nth      = len(prev) + 1
    gap_sec: float | None = None
    if prev:
        gap_sec = (entry_ms - prev[-1][0]) / 1000.0

    # --- Write tags ---
    def _upsert(tag: str, reasons: str) -> None:
        conn.execute(
            """
            INSERT INTO s34_shadow_geometry_tags
              (created_at, trade_id, rule_name, tag, reasons,
               cascade_usd, liq_count, single_share, net_bps, exit_reason)
            VALUES (?,?,?,?,?,NULL,NULL,NULL,?,?)
            ON CONFLICT(trade_id, tag) DO UPDATE SET
              created_at  = excluded.created_at,
              reasons     = excluded.reasons,
              net_bps     = excluded.net_bps,
              exit_reason = excluded.exit_reason
            """,
            (now, tid, rule, tag, reasons, net, exit_r),
        )

    _upsert("CONTEXT_SESSION",      f"session={session}")
    if co_type:
        _upsert("CONTEXT_CO_CASCADE", f"co_type={co_type}")
    _upsert("CONTEXT_INTRADAY_NTH", f"nth={nth}")
    if gap_sec is not None:
        _upsert("CONTEXT_GAP_LAST_SEC", f"gap={gap_sec:.0f}s")


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
