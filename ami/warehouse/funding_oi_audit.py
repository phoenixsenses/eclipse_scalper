"""BATCH-P2-003: funding/OI data-coverage audit (OD-006).

Read-only investigation of the three sources that hold funding/OI data:
  - data/funding_history.db (funding_rates, open_interest_hist -- one-time
    bulk backfill; `fetch_runs` shows a single completed run, 2026-05-12)
  - data/oi_history.db (oi_history, futures_klines_5m -- one-time bulk
    backfill, `fetch_runs` shows a single completed run, 2026-05-14)
  - data/microstructure.db (funding_rates, open_interest, spot_prices --
    the live stack; open_interest/spot_prices ARE fed by the running
    `data.oi_spot_poller` process (PID confirmed running at Phase 0 audit),
    funding_rates is NOT -- no INSERT into that table exists anywhere in
    the current codebase)

This batch performs NO operational change: no collector started, no config
touched, no process restarted. Findings are recorded as data_quality_events
(read-only sources -> new warehouse rows only). OD-006 is updated with the
concrete findings below and stays OPEN -- activating a new funding/OI
collector remains an operator decision.
"""
from __future__ import annotations
import sqlite3
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SEVEN_DAYS_MS = 7 * 86_400_000


def _range(conn: sqlite3.Connection, table: str, ts_col: str, where: str = "") -> tuple:
    q = f"SELECT MIN([{ts_col}]), MAX([{ts_col}]), COUNT(*) FROM [{table}]"
    if where:
        q += " " + where
    return conn.execute(q).fetchone()


def audit(repo_root: Path = REPO_ROOT) -> list[dict]:
    now_ms = int(time.time() * 1000)
    findings: list[dict] = []

    fh = sqlite3.connect(f"file:{repo_root / 'data' / 'funding_history.db'}?mode=ro", uri=True)
    lo, hi, n = _range(fh, "funding_rates", "funding_time_ms")
    findings.append({
        "event_id": "DQE-FUNDING-HIST-STALE",
        "source": "data/funding_history.db:funding_rates",
        "event_type": "STALE_TABLE",
        "period_start_ms": lo, "period_end_ms": hi,
        "description": (
            f"One-time bulk backfill ({n} rows, BTC/ETH/SOL). fetch_runs shows a single "
            "completed run (2026-05-12); no live successor collector writes to this table."
        ),
        "severity": "MEDIUM",
    })
    lo, hi, n = _range(fh, "open_interest_hist", "timestamp_ms")
    findings.append({
        "event_id": "DQE-OI-HIST-STALE",
        "source": "data/funding_history.db:open_interest_hist",
        "event_type": "STALE_TABLE",
        "period_start_ms": lo, "period_end_ms": hi,
        "description": (
            f"One-time bulk backfill ({n} rows). Superseded operationally by the live "
            "oi_spot_poller writing into data/microstructure.db:open_interest, but this "
            "specific table itself is frozen and will not receive new rows."
        ),
        "severity": "LOW",
    })
    fh.close()

    oh = sqlite3.connect(f"file:{repo_root / 'data' / 'oi_history.db'}?mode=ro", uri=True)
    lo, hi, n = _range(oh, "oi_history", "timestamp_ms")
    findings.append({
        "event_id": "DQE-OI-HISTORY-DB-STALE",
        "source": "data/oi_history.db:oi_history",
        "event_type": "STALE_TABLE",
        "period_start_ms": lo, "period_end_ms": hi,
        "description": (
            f"One-time bulk backfill ({n} rows, 5m/15m/1h/4h, ~30 days). No live successor "
            "writes to this file."
        ),
        "severity": "LOW",
    })
    oh.close()

    ms = sqlite3.connect(f"file:{repo_root / 'data' / 'microstructure.db'}?mode=ro", uri=True)
    lo, hi, n = _range(ms, "funding_rates", "ts_ms")
    findings.append({
        "event_id": "DQE-MICRO-FUNDING-ORPHANED",
        "source": "data/microstructure.db:funding_rates",
        "event_type": "STALE_TABLE",
        "period_start_ms": lo, "period_end_ms": hi,
        "description": (
            f"{n} rows, ETHUSDT only, last write 2026-04-13. No INSERT into this table found "
            "anywhere in the current codebase -- orphaned, no producer process exists. There is "
            "currently NO live funding-rate collector anywhere in the stack."
        ),
        "severity": "HIGH",
    })
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        lo, hi, n = _range(ms, "open_interest", "ts_ms", where=f"WHERE symbol='{sym}'")
        short_window = bool(lo) and (now_ms - lo) < SEVEN_DAYS_MS
        findings.append({
            "event_id": f"DQE-MICRO-OI-COVERAGE-{sym}",
            "source": f"data/microstructure.db:open_interest[{sym}]",
            "event_type": "GAPPED" if short_window else "COVERAGE_NOTE",
            "period_start_ms": lo, "period_end_ms": hi,
            "description": (
                f"{n} rows live via oi_spot_poller (60s public endpoint, HEALTHY at Phase 0 audit). "
                + ("Coverage window is short (<7 days) as of this audit." if short_window
                   else "Coverage window established (>=7 days) as of this audit.")
            ),
            "severity": "MEDIUM" if short_window else "LOW",
        })
    ms.close()

    return findings


def seed(conn, provenance: str = "batch-p2-003-funding-oi-audit") -> int:
    """Idempotent upsert by event_id; re-running refreshes counts/ranges as sources change."""
    now = int(time.time() * 1000)
    findings = audit()
    for f in findings:
        conn.execute(
            "INSERT INTO data_quality_events (event_id, source, event_type, detected_ms, "
            "period_start_ms, period_end_ms, description, severity, schema_version, provenance, created_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(event_id) DO UPDATE SET detected_ms=excluded.detected_ms, "
            "event_type=excluded.event_type, period_start_ms=excluded.period_start_ms, "
            "period_end_ms=excluded.period_end_ms, description=excluded.description, "
            "severity=excluded.severity",
            (f["event_id"], f["source"], f["event_type"], now, f["period_start_ms"], f["period_end_ms"],
             f["description"], f["severity"], 2, provenance, now),
        )
    conn.commit()
    return len(findings)


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed(conn)
        print(f"recorded {n} data_quality_events (funding/OI coverage audit)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
