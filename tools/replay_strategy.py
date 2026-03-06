from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol, Tuple

from tools.check_data_ready import detect_ts_col, list_tables, table_columns
from tools.strategies.baseline import BaselineStrategy
from tools.strategies.micro_edge_pocket import MicroEdgePocketStrategy


class StrategyProtocol(Protocol):
    def on_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        ...

    def on_tick(self, ts_utc: str) -> List[Dict[str, Any]]:
        ...


def _parse_iso_utc(text: str) -> float:
    s = str(text or "").strip()
    if not s:
        raise ValueError("empty datetime")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _ts_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _table_specs(conn: sqlite3.Connection) -> List[Tuple[str, str, str | None, List[str]]]:
    out: List[Tuple[str, str, str | None, List[str]]] = []
    for table in ("agg_trades", "mark_prices", "liquidations"):
        if table not in list_tables(conn):
            continue
        cols = table_columns(conn, table)
        ts_col = detect_ts_col(cols)
        if not ts_col:
            continue
        lower = [c.lower() for c in cols]
        sym_col = cols[lower.index("symbol")] if "symbol" in lower else None
        out.append((table, ts_col, sym_col, cols))
    return out


def extract_events(
    db: Path,
    symbols: Iterable[str],
    start_iso: str,
    end_iso: str,
) -> List[Dict[str, Any]]:
    if not db.exists():
        raise FileNotFoundError(str(db))
    start_ts = _parse_iso_utc(start_iso)
    end_ts = _parse_iso_utc(end_iso)
    if end_ts <= start_ts:
        raise ValueError("invalid range")

    conn = sqlite3.connect(str(db))
    try:
        specs = _table_specs(conn)
        if not specs:
            raise RuntimeError("no supported timestamped tables")
        syms = list(symbols)
        raw_rows: List[Tuple[float, str, str, int, Dict[str, Any]]] = []
        for table, ts_col, sym_col, cols in specs:
            select_cols = ["rowid"] + cols
            base = (
                f"SELECT {', '.join(select_cols)} FROM {table} "
                f"WHERE (({ts_col} >= ? AND {ts_col} <= ?) OR ({ts_col} >= ? AND {ts_col} <= ?))"
            )
            params: List[Any] = [start_ts, end_ts, int(start_ts * 1000), int(end_ts * 1000)]
            if sym_col and syms:
                placeholders = ",".join("?" for _ in syms)
                base += f" AND {sym_col} IN ({placeholders})"
                params.extend(syms)
            rows = conn.execute(base, tuple(params)).fetchall()
            for row in rows:
                rowid = int(row[0])
                payload: Dict[str, Any] = {}
                for i, col in enumerate(cols, start=1):
                    payload[col] = row[i]
                ts_val = payload.get(ts_col)
                if ts_val is None:
                    continue
                try:
                    ts = float(ts_val)
                except Exception:
                    continue
                if ts > 1e12:
                    ts = ts / 1000.0
                symbol = str(payload.get(sym_col) or "ALL").upper() if sym_col else "ALL"
                raw_rows.append((ts, table, symbol, rowid, payload))
    finally:
        conn.close()

    raw_rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    events: List[Dict[str, Any]] = []
    for idx, (ts, table, symbol, rowid, payload) in enumerate(raw_rows, start=1):
        events.append(
            {
                "event_index": idx,
                "ts_utc": _ts_to_iso(ts),
                "symbol": symbol,
                "source_table": table,
                "rowid": int(rowid),
                "payload": payload,
            }
        )
    return events


def _strategy_from_name(name: str, strategy_config: Dict[str, Any]) -> StrategyProtocol:
    n = str(name or "baseline").strip().lower()
    if n == "baseline":
        return BaselineStrategy(
            period=int(strategy_config.get("period", 5)),
            action=str(strategy_config.get("action", "signal")),
        )
    if n == "micro_edge_pocket":
        return MicroEdgePocketStrategy(
            rule=str(strategy_config.get("rule", "micro_edge_v3_passive_alpha")),
            side=str(strategy_config.get("side", "buy")),
            symbol_whitelist=tuple(strategy_config.get("symbol_whitelist", ()) or ()),
            event_source_table=str(strategy_config.get("event_source_table", "agg_trades")),
            min_trade_count_window=int(strategy_config.get("min_trade_count_window", 1)),
            horizon_sec=int(strategy_config.get("horizon_sec", 120)),
            cooldown_ms=int(strategy_config.get("cooldown_ms", 250)),
            filters=dict(strategy_config.get("filters", {}) or {}),
            action=str(strategy_config.get("action", "signal")),
        )
    raise ValueError(f"unknown strategy: {name}")


def _load_strategy_config(raw: str) -> Dict[str, Any]:
    s = str(raw or "").strip()
    if not s:
        return {}
    p = Path(s)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("strategy-config must be JSON object")
    return data


def _decision_id(ts_utc: str, symbol: str, action: str, params: Dict[str, Any]) -> str:
    raw = json.dumps(
        {"ts_utc": ts_utc, "symbol": symbol, "action": action, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def replay_to_decisions(
    db: Path,
    symbols: Iterable[str],
    start_iso: str,
    end_iso: str,
    strategy_name: str = "baseline",
    strategy_config: Dict[str, Any] | None = None,
) -> tuple[List[Dict[str, Any]], int]:
    events = extract_events(db=db, symbols=symbols, start_iso=start_iso, end_iso=end_iso)
    strat = _strategy_from_name(strategy_name, strategy_config or {})
    decisions: List[Dict[str, Any]] = []
    for ev in events:
        out = strat.on_event(ev) or []
        for d in out:
            action = str(d.get("action") or "noop")
            params = d.get("params") if isinstance(d.get("params"), dict) else {}
            rec = {
                "ts_utc": str(ev["ts_utc"]),
                "symbol": str(ev["symbol"]),
                "action": action,
                "decision_id": _decision_id(str(ev["ts_utc"]), str(ev["symbol"]), action, params),
                "params": params,
            }
            decisions.append(rec)
    return decisions, len(events)


def write_decisions_jsonl(path: Path, decisions: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for d in decisions:
            f.write(json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            f.write("\n")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Deterministic replay strategy harness producing decisions JSONL.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--strategy", default="baseline")
    p.add_argument("--strategy-config", default="{}", help="JSON dict or path to JSON file for strategy configuration.")
    p.add_argument("--out", required=True, help="decisions.jsonl output path")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        cfg = _load_strategy_config(str(args.strategy_config))
        decisions, events_count = replay_to_decisions(
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            start_iso=str(args.start),
            end_iso=str(args.end),
            strategy_name=str(args.strategy),
            strategy_config=cfg,
        )
        out = Path(str(args.out))
        write_decisions_jsonl(out, decisions)
        print(
            f"replay_strategy ok events={events_count} decisions={len(decisions)} "
            f"strategy={args.strategy} out={out}"
        )
        return 0
    except Exception as e:
        print(f"replay_strategy error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
