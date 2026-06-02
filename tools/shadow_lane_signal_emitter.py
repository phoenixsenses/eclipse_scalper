"""Shadow-only emitter for lane-conditioned alpha candidates.

The emitter reads existing local DB tables and appends JSONL events for future
matching candidates. It never places orders and does not touch live routing.

Default behavior is safe for first run: if no state file exists, it initializes
each cursor at the current DB frontier and emits zero historical signals. Use
`--backfill-existing` only for research/replay.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShadowSpec:
    family: str
    source: str
    symbol: str
    direction: str
    horizon_sec: int
    liq_side: str = ""
    min_notional: float = 0.0
    utc_hour: int | None = None
    session_us: bool = False
    funding_negative: bool = False
    basis_positive: bool = False


SPECS: tuple[ShadowSpec, ...] = (
    ShadowSpec(
        family="ETH_BUY250K_SHORT_900_UTC14",
        source="liquidations",
        symbol="ETHUSDT",
        liq_side="BUY",
        min_notional=250000.0,
        utc_hour=14,
        direction="SHORT",
        horizon_sec=900,
    ),
    ShadowSpec(
        family="ETH_BUY500K_SHORT_900_SESSION_US",
        source="liquidations",
        symbol="ETHUSDT",
        liq_side="BUY",
        min_notional=500000.0,
        session_us=True,
        direction="SHORT",
        horizon_sec=900,
    ),
    ShadowSpec(
        family="SOL_BUY50K_SHORT_900_FUNDING_NEGATIVE",
        source="liquidations",
        symbol="SOLUSDT",
        liq_side="BUY",
        min_notional=50000.0,
        funding_negative=True,
        direction="SHORT",
        horizon_sec=900,
    ),
    ShadowSpec(
        family="S34_SHORT_900_SESSION_US",
        source="detector_signals",
        symbol="ETHUSDT",
        session_us=True,
        direction="SHORT",
        horizon_sec=900,
    ),
    ShadowSpec(
        family="S34_SHORT_900_BASIS_POSITIVE",
        source="detector_signals",
        symbol="ETHUSDT",
        basis_positive=True,
        direction="SHORT",
        horizon_sec=900,
    ),
)


def _utc_hour(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).hour


def _is_us_session(ts_ms: int) -> bool:
    hour = _utc_hour(ts_ms)
    return 14 <= hour < 21


def _mark_at(conn: sqlite3.Connection, symbol: str, ts_ms: int, *, before: bool) -> float | None:
    op = "<=" if before else ">="
    order = "DESC" if before else "ASC"
    row = conn.execute(
        f"SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms {op} ? ORDER BY ts_ms {order} LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _funding_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = conn.execute(
        """
        SELECT funding_rate FROM mark_prices
        WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _return_bps(conn: sqlite3.Connection, symbol: str, ts_ms: int, direction: str, horizon_sec: int) -> float | None:
    entry = _mark_at(conn, symbol, ts_ms, before=True)
    exit_px = _mark_at(conn, symbol, ts_ms + horizon_sec * 1000, before=False)
    if entry is None or exit_px is None or entry <= 0:
        return None
    raw = (exit_px - entry) / entry * 1e4
    return -raw if direction == "SHORT" else raw


def _overlap_count(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    ts_ms: int,
    window_sec: int = 60,
    min_notional: float = 100000.0,
) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM liquidations
        WHERE symbol=? AND side=? AND notional>=? AND ABS(ts_ms - ?) <= ?
        """,
        (symbol, side, float(min_notional), int(ts_ms), int(window_sec * 1000)),
    ).fetchone()
    return int(row[0] or 0)


def _forward_labels(conn: sqlite3.Connection, symbol: str, ts_ms: int, direction: str) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    for horizon in (60, 120, 300, 900):
        labels[f"return_bps_{horizon}s"] = _return_bps(conn, symbol, ts_ms, direction, horizon)
    return labels


def _base_event(spec: ShadowSpec, ts_ms: int, source_id: Any) -> dict[str, Any]:
    return {
        "ts": time.time(),
        "event": "research.shadow_signal",
        "symbol": spec.symbol,
        "data": {
            "status": "SHADOW_ONLY",
            "signal_family": spec.family,
            "source": spec.source,
            "source_id": source_id,
            "ts_ms": int(ts_ms),
            "direction": spec.direction,
            "horizon_sec": int(spec.horizon_sec),
            "utc_hour": _utc_hour(int(ts_ms)),
            "session_us": _is_us_session(int(ts_ms)),
        },
    }


def _enrich_common(conn: sqlite3.Connection, event: dict[str, Any], spec: ShadowSpec) -> dict[str, Any]:
    data = event["data"]
    ts_ms = int(data["ts_ms"])
    entry = _mark_at(conn, spec.symbol, ts_ms, before=True)
    data["entry_reference_price"] = entry
    data["forward_labels"] = _forward_labels(conn, spec.symbol, ts_ms, spec.direction)
    data["fee_rt_bps"] = {
        "2": _net_label(data["forward_labels"].get("return_bps_900s"), 2.0),
        "4": _net_label(data["forward_labels"].get("return_bps_900s"), 4.0),
        "8": _net_label(data["forward_labels"].get("return_bps_900s"), 8.0),
        "10": _net_label(data["forward_labels"].get("return_bps_900s"), 10.0),
    }
    data["overlap_60s"] = {
        "eth_buy_100k": _overlap_count(conn, symbol="ETHUSDT", side="BUY", ts_ms=ts_ms),
        "eth_sell_100k": _overlap_count(conn, symbol="ETHUSDT", side="SELL", ts_ms=ts_ms),
        "btc_buy_100k": _overlap_count(conn, symbol="BTCUSDT", side="BUY", ts_ms=ts_ms),
        "btc_sell_100k": _overlap_count(conn, symbol="BTCUSDT", side="SELL", ts_ms=ts_ms),
        "sol_buy_100k": _overlap_count(conn, symbol="SOLUSDT", side="BUY", ts_ms=ts_ms),
        "sol_sell_100k": _overlap_count(conn, symbol="SOLUSDT", side="SELL", ts_ms=ts_ms),
    }
    return event


def _net_label(value: Any, fee_bps: float) -> float | None:
    if value is None:
        return None
    return float(value) - float(fee_bps)


def _load_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cursors = raw.get("cursors", raw if isinstance(raw, dict) else {})
    out: dict[str, int] = {}
    for key, value in dict(cursors or {}).items():
        try:
            out[str(key)] = int(value)
        except Exception:
            pass
    return out


def _write_state(path: Path, cursors: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": time.time(), "cursors": dict(sorted(cursors.items()))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _frontier(conn: sqlite3.Connection, spec: ShadowSpec) -> int:
    if spec.source == "liquidations":
        row = conn.execute(
            "SELECT MAX(ts_ms) FROM liquidations WHERE symbol=? AND side=? AND notional>=?",
            (spec.symbol, spec.liq_side, float(spec.min_notional)),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT MAX(signal_ts_ms) FROM detector_signals WHERE symbol=? AND signal_ts_ms IS NOT NULL",
            (spec.symbol,),
        ).fetchone()
    return int(row[0] or 0)


def _liquidation_events(conn: sqlite3.Connection, spec: ShadowSpec, after_ts_ms: int, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, ts_ms, side, price, quantity, notional
        FROM liquidations
        WHERE symbol=? AND side=? AND notional>=? AND ts_ms>?
        ORDER BY ts_ms ASC
        LIMIT ?
        """,
        (spec.symbol, spec.liq_side, float(spec.min_notional), int(after_ts_ms), int(limit)),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        source_id, ts_ms, side, price, qty, notional = row
        ts = int(ts_ms)
        funding = _funding_at(conn, spec.symbol, ts)
        if spec.utc_hour is not None and _utc_hour(ts) != int(spec.utc_hour):
            continue
        if spec.session_us and not _is_us_session(ts):
            continue
        if spec.funding_negative and not (funding is not None and funding < 0):
            continue
        event = _base_event(spec, ts, source_id)
        event["data"].update(
            {
                "liq_side": str(side),
                "liq_price": float(price or 0.0),
                "liq_quantity": float(qty or 0.0),
                "liq_notional": float(notional or 0.0),
                "funding_rate": funding,
                "lane_fields": _lane_fields(spec, ts, funding=funding),
            }
        )
        out.append(_enrich_common(conn, event, spec))
    return out


def _detector_events(conn: sqlite3.Connection, spec: ShadowSpec, after_ts_ms: int, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, signal_id, signal_ts_ms, entry_price, basis_at_entry, liq_composition,
               confidence_band, session_tag
        FROM detector_signals
        WHERE symbol=? AND signal_ts_ms IS NOT NULL AND entry_price IS NOT NULL AND signal_ts_ms>?
        ORDER BY signal_ts_ms ASC
        LIMIT ?
        """,
        (spec.symbol, int(after_ts_ms), int(limit)),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        source_id, signal_id, ts_ms, entry_price, basis, comp, confidence, session = row
        ts = int(ts_ms)
        basis_value = float(basis) if basis is not None else None
        if spec.session_us and not _is_us_session(ts):
            continue
        if spec.basis_positive and not (basis_value is not None and basis_value > 0):
            continue
        event = _base_event(spec, ts, signal_id or source_id)
        event["data"].update(
            {
                "detector_row_id": int(source_id),
                "detector_signal_id": signal_id,
                "detector_entry_price": float(entry_price or 0.0),
                "basis_at_entry": basis_value,
                "liq_composition": str(comp or ""),
                "confidence_band": str(confidence or ""),
                "session_tag": str(session or ""),
                "lane_fields": _lane_fields(spec, ts, basis=basis_value),
            }
        )
        out.append(_enrich_common(conn, event, spec))
    return out


def _lane_fields(spec: ShadowSpec, ts_ms: int, *, funding: float | None = None, basis: float | None = None) -> dict[str, Any]:
    return {
        "utc_hour_required": spec.utc_hour,
        "utc_hour_actual": _utc_hour(ts_ms),
        "session_us_required": bool(spec.session_us),
        "session_us_actual": _is_us_session(ts_ms),
        "funding_negative_required": bool(spec.funding_negative),
        "funding_rate": funding,
        "basis_positive_required": bool(spec.basis_positive),
        "basis_at_entry": basis,
    }


def _append_jsonl(path: Path, events: list[dict[str, Any]], *, dry_run: bool) -> None:
    if dry_run or not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n")


def emit_shadow_signals(args: argparse.Namespace) -> dict[str, Any]:
    state_path = Path(str(args.state))
    output_path = Path(str(args.output_jsonl))
    cursors = _load_state(state_path)
    emitted: list[dict[str, Any]] = []
    initialized: dict[str, int] = {}
    matched_counts: dict[str, int] = {}

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        for spec in SPECS:
            cursor = int(cursors.get(spec.family, 0) or 0)
            if cursor <= 0 and not bool(args.backfill_existing):
                cursor = _frontier(conn, spec)
                cursors[spec.family] = cursor
                initialized[spec.family] = cursor
                matched_counts[spec.family] = 0
                continue
            if spec.source == "liquidations":
                events = _liquidation_events(conn, spec, cursor, int(args.limit_per_family))
            else:
                events = _detector_events(conn, spec, cursor, int(args.limit_per_family))
            matched_counts[spec.family] = len(events)
            emitted.extend(events)
            if events:
                cursors[spec.family] = max(int(e["data"]["ts_ms"]) for e in events)
    finally:
        conn.close()

    _append_jsonl(output_path, emitted, dry_run=bool(args.dry_run))
    if not bool(args.dry_run):
        _write_state(state_path, cursors)
    return {
        "output_jsonl": str(output_path),
        "state": str(state_path),
        "dry_run": bool(args.dry_run),
        "backfill_existing": bool(args.backfill_existing),
        "initialized": initialized,
        "matched_counts": matched_counts,
        "emitted_count": len(emitted),
        "families": [spec.family for spec in SPECS],
        "events": emitted[: int(args.preview_limit)],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Emit shadow-only lane-conditioned alpha signals.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--output-jsonl", default="logs/telemetry.jsonl")
    p.add_argument("--state", default="reports/LANE_SHADOW_EMITTER_STATE.json")
    p.add_argument("--limit-per-family", type=int, default=100)
    p.add_argument("--preview-limit", type=int, default=5)
    p.add_argument("--backfill-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out-json", default="reports/LANE_SHADOW_EMITTER_RUN.json")
    args = p.parse_args()
    payload = emit_shadow_signals(args)
    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"emitted_count={payload['emitted_count']}")
    print(f"wrote {out_json}")
    if payload["initialized"]:
        print(f"initialized={payload['initialized']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
