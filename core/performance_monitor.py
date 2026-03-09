from __future__ import annotations

import math
import sqlite3
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DailyMetrics:
    day_utc: str
    trade_count: int
    win_rate: float
    pnl_bps: float
    scratch_rate: float
    blocked_count: int
    regime_up_sec: float
    regime_down_sec: float
    regime_unknown_sec: float


def _sqlite_rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def _utc_day_range(ts: float | None = None) -> tuple[float, float, str]:
    now = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    end = start + 86400.0
    return start, end, now.strftime("%Y-%m-%d")


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return float(statistics.pstdev(values))
    except Exception:
        return 0.0


def _daily_trade_metrics(conn: sqlite3.Connection, start_ts: float, end_ts: float) -> tuple[int, float, float, float]:
    row = conn.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(pnl_bps),0) pnl, "
        "COALESCE(AVG(CASE WHEN pnl_bps>0 THEN 1.0 ELSE 0.0 END),0) wr, "
        "COALESCE(AVG(CASE WHEN ABS(pnl_bps)<=1.0 THEN 1.0 ELSE 0.0 END),0) scratch "
        "FROM trades WHERE exit_time>=? AND exit_time<?",
        (start_ts, end_ts),
    ).fetchone()
    if not row:
        return 0, 0.0, 0.0, 0.0
    return int(row[0] or 0), float(row[1] or 0.0), float(row[2] or 0.0), float(row[3] or 0.0)


def _daily_regime_durations(health_history_path: Path, start_ts: float, end_ts: float) -> tuple[float, float, float]:
    if not health_history_path.exists():
        return 0.0, 0.0, 0.0
    up = 0.0
    down = 0.0
    unk = 0.0
    try:
        import json

        _TAIL = 512 * 1024  # 512 KB tail to avoid OOM on large history files
        _sz = health_history_path.stat().st_size
        with open(health_history_path, "r", encoding="utf-8", errors="replace") as _fh:
            if _sz > _TAIL:
                _fh.seek(_sz - _TAIL)
                _fh.readline()  # skip partial first line
            lines = _fh.read().splitlines()
        for raw in lines:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            ts = float(obj.get("ts") or 0.0)
            if ts < start_ts or ts >= end_ts:
                continue
            reg = str(obj.get("regime") or obj.get("regime_state") or "UNKNOWN").upper()
            if reg == "UP":
                up += 1.0
            elif reg == "DOWN":
                down += 1.0
            else:
                unk += 1.0
    except Exception:
        return 0.0, 0.0, 0.0
    return up, down, unk


def _daily_blocked_count(journal_path: Path, start_ts: float, end_ts: float) -> int:
    if not journal_path.exists():
        return 0
    count = 0
    try:
        import json

        _TAIL_J = 512 * 1024  # 512 KB tail to avoid OOM
        _sz_j = journal_path.stat().st_size
        with open(journal_path, "r", encoding="utf-8", errors="replace") as _fhj:
            if _sz_j > _TAIL_J:
                _fhj.seek(_sz_j - _TAIL_J)
                _fhj.readline()
            _journal_lines = _fhj.read().splitlines()
        for raw in _journal_lines:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            evt = str(obj.get("event") or "")
            if evt not in ("entry.blocked", "risk.blocked"):
                continue
            ts = float(obj.get("ts_epoch") or obj.get("ts") or 0.0)
            if start_ts <= ts < end_ts:
                count += 1
    except Exception:
        return count
    return count


def compute_daily_metrics(
    *,
    trades_db: str | Path = "data/paper_trades.db",
    now_ts: float | None = None,
    health_history_path: str | Path = "logs/health/history.jsonl",
    journal_path: str | Path = "logs/execution_journal.jsonl",
) -> DailyMetrics:
    start_ts, end_ts, day_utc = _utc_day_range(now_ts)
    db = Path(trades_db)
    n = 0
    pnl = 0.0
    wr = 0.0
    scratch = 0.0
    if db.exists():
        conn = sqlite3.connect(str(db), check_same_thread=False)
        try:
            n, pnl, wr, scratch = _daily_trade_metrics(conn, start_ts, end_ts)
        finally:
            conn.close()
    up, down, unk = _daily_regime_durations(Path(health_history_path), start_ts, end_ts)
    blocked = _daily_blocked_count(Path(journal_path), start_ts, end_ts)
    return DailyMetrics(
        day_utc=day_utc,
        trade_count=n,
        win_rate=wr,
        pnl_bps=pnl,
        scratch_rate=scratch,
        blocked_count=blocked,
        regime_up_sec=up,
        regime_down_sec=down,
        regime_unknown_sec=unk,
    )


def detect_anomalies(
    daily: DailyMetrics,
    *,
    history_db: str | Path = "data/paper_trades.db",
    lookback_days: int = 30,
    expected_pnl_bps: float = 0.0,
    expected_sigma_bps: float = 15.0,
) -> list[str]:
    out: list[str] = []
    if daily.trade_count > 0 and daily.scratch_rate > 0.50:
        out.append("scratch_rate_gt_50pct")
    reg_total = daily.regime_up_sec + daily.regime_down_sec + daily.regime_unknown_sec
    if reg_total > 0:
        max_share = max(daily.regime_up_sec, daily.regime_down_sec, daily.regime_unknown_sec) / reg_total
        if max_share > 0.80:
            out.append("regime_distribution_one_sided")
    # history-based 2-sigma check
    db = Path(history_db)
    if db.exists():
        conn = sqlite3.connect(str(db), check_same_thread=False)
        try:
            cutoff = time.time() - max(1, int(lookback_days)) * 86400.0
            rows = conn.execute(
                "SELECT CAST(exit_time/86400 AS INTEGER) d, COALESCE(SUM(pnl_bps),0) pnl "
                "FROM trades WHERE exit_time>=? GROUP BY CAST(exit_time/86400 AS INTEGER)",
                (cutoff,),
            ).fetchall()
        finally:
            conn.close()
        vals = [float(r[1] or 0.0) for r in rows]
        sigma = _safe_std(vals) if vals else float(expected_sigma_bps)
        mu = (sum(vals) / len(vals)) if vals else float(expected_pnl_bps)
        if sigma > 0:
            z = abs((daily.pnl_bps - mu) / sigma)
            if z > 2.0:
                out.append("pnl_gt_2sigma")
    return out


def weekly_digest_markdown(dailies: list[DailyMetrics]) -> str:
    if not dailies:
        return "# Weekly Digest\n\nNo daily metrics available.\n"
    trades = sum(d.trade_count for d in dailies)
    pnl = sum(d.pnl_bps for d in dailies)
    wr_num = sum(d.win_rate * d.trade_count for d in dailies)
    wr = (wr_num / trades) if trades > 0 else 0.0
    scratch_num = sum(d.scratch_rate * d.trade_count for d in dailies)
    scratch = (scratch_num / trades) if trades > 0 else 0.0
    return "\n".join(
        [
            "# Weekly Digest",
            "",
            f"- days: {len(dailies)}",
            f"- trades: {trades}",
            f"- pnl_bps: {pnl:+.2f}",
            f"- win_rate: {wr*100.0:.1f}%",
            f"- scratch_rate: {scratch*100.0:.1f}%",
            "",
            "## Daily",
            *[
                f"- {d.day_utc}: trades={d.trade_count} pnl={d.pnl_bps:+.2f} wr={d.win_rate*100.0:.1f}% scratch={d.scratch_rate*100.0:.1f}%"
                for d in dailies
            ],
            "",
        ]
    )

