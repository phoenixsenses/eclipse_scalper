from __future__ import annotations

import argparse
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

from tools.s34_shadow_paper_runner import _evaluate_trade, _iso_from_ms


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mark_at_exact(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = conn.execute(
        "SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms=? LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return None if row is None else float(row[0])


def _reset_for_recompute(trade: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(trade)
    rule = dict(out.get("rule") or {})
    rule["require_book_ticker_fill"] = False
    out["rule"] = rule
    out["status"] = "OPEN"
    out["be_active"] = False
    out["exit_price"] = None
    out["exit_reference_price"] = None
    out["exit_fill"] = None
    out["exit_ts_ms"] = None
    out["exit_ts_utc"] = None
    out["exit_reason"] = None
    out["gross_bps"] = None
    out["executable_gross_bps"] = None
    out["spread_cost_bps"] = None
    out["fee_cost_bps"] = None
    out["slippage_cost_bps"] = None
    out["net_bps"] = None
    out["gross_usdt"] = None
    out["net_usdt"] = None
    out.pop("closed_at_utc", None)
    out.pop("be_activated_ts_ms", None)
    out.pop("be_activated_ts_utc", None)
    out.pop("last_evaluated_mark_ts_ms", None)
    out.pop("last_evaluated_mark_ts_utc", None)
    return out


def _anomalies(conn: sqlite3.Connection, trade: dict[str, Any]) -> list[str]:
    out: list[str] = []
    exit_ts = int(trade.get("exit_ts_ms") or 0)
    be_ts = int(trade.get("be_activated_ts_ms") or 0)
    reason = str(trade.get("exit_reason") or "")
    if reason == "BE" and not be_ts:
        out.append("BE_MISSING_ACTIVATION")
    if exit_ts and be_ts and exit_ts < be_ts:
        out.append("EXIT_BEFORE_BE_ACTIVATION")
    exit_reference = trade.get("exit_reference_price")
    if exit_ts and (exit_reference is not None or trade.get("exit_price") is not None):
        mark = _mark_at_exact(conn, str(trade.get("symbol")), exit_ts)
        if mark is None:
            out.append("EXIT_TS_NO_EXACT_MARK")
        elif abs(float(exit_reference if exit_reference is not None else trade.get("exit_price")) - mark) > 1e-9:
            out.append("EXIT_REFERENCE_MARK_MISMATCH")
    return out


def audit(db: Path, trades_json: Path) -> dict[str, Any]:
    payload = _read_json(trades_json)
    trades = payload.get("trades", []) if isinstance(payload, dict) else []
    rows: list[dict[str, Any]] = []
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for trade in trades:
            if trade.get("status") == "SKIPPED":
                continue
            original_anomalies = _anomalies(conn, trade)
            recompute = _reset_for_recompute(trade)
            horizon_end = int(recompute["signal_ts_ms"]) + int(recompute["rule"]["max_horizon_sec"]) * 1000
            corrected = _evaluate_trade(conn, recompute, horizon_end)
            rows.append(
                {
                    "trade_id": trade.get("trial_id") or trade.get("trade_id"),
                    "status": trade.get("status"),
                    "exit_reason": trade.get("exit_reason"),
                    "anomaly": ",".join(original_anomalies) if original_anomalies else "",
                    "original_exit_ts_utc": trade.get("exit_ts_utc"),
                    "corrected_exit_ts_utc": corrected.get("exit_ts_utc"),
                    "original_net_bps": trade.get("net_bps"),
                    "corrected_net_bps": corrected.get("net_bps"),
                    "original_gross_bps": trade.get("gross_bps"),
                    "corrected_gross_bps": corrected.get("gross_bps"),
                    "original_exit_price": trade.get("exit_price"),
                    "corrected_exit_price": corrected.get("exit_price"),
                    "corrected_exit_reference_price": corrected.get("exit_reference_price"),
                    "corrected_executable_gross_bps": corrected.get("executable_gross_bps"),
                    "corrected_spread_cost_bps": corrected.get("spread_cost_bps"),
                    "corrected_fee_cost_bps": corrected.get("fee_cost_bps"),
                    "corrected_slippage_cost_bps": corrected.get("slippage_cost_bps"),
                    "corrected_entry_fill_source": (corrected.get("entry_fill") or {}).get("source"),
                    "corrected_exit_fill_source": (corrected.get("exit_fill") or {}).get("source"),
                    "be_activated_ts_utc": trade.get("be_activated_ts_utc"),
                    "corrected_be_activated_ts_utc": corrected.get("be_activated_ts_utc"),
                    "corrected_status": corrected.get("status"),
                    "corrected_exit_reason": corrected.get("exit_reason"),
                }
            )
    finally:
        conn.close()
    return {
        "trades_checked": len(rows),
        "anomalies": sum(1 for row in rows if row["anomaly"]),
        "rows": rows,
    }


def write_reports(result: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = [
        "# S34 Paper Chronology Audit",
        "",
        f"- trades_checked: `{result['trades_checked']}`",
        f"- anomaly_count: `{result['anomalies']}`",
        "",
        "| trade_id | anomaly | corrected exit | reason | gross | spread | fee | slippage | net | fill source |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in result["rows"]:
        corr_net = row["corrected_net_bps"]
        lines.append(
            "| `{trade_id}` | `{anomaly}` | `{corr_exit}` | `{corr_reason}` | {gross} | {spread} | {fee} | {slip} | {net} | `{entry_src}->{exit_src}` |".format(
                trade_id=row["trade_id"],
                anomaly=row["anomaly"] or "",
                corr_exit=row["corrected_exit_ts_utc"] or "",
                corr_reason=row["corrected_exit_reason"] or "",
                gross="" if row["corrected_gross_bps"] is None else f"{float(row['corrected_gross_bps']):.4f}",
                spread="" if row["corrected_spread_cost_bps"] is None else f"{float(row['corrected_spread_cost_bps']):.4f}",
                fee="" if row["corrected_fee_cost_bps"] is None else f"{float(row['corrected_fee_cost_bps']):.4f}",
                slip="" if row["corrected_slippage_cost_bps"] is None else f"{float(row['corrected_slippage_cost_bps']):.4f}",
                net="" if corr_net is None else f"{float(corr_net):.4f}",
                entry_src=row["corrected_entry_fill_source"] or "",
                exit_src=row["corrected_exit_fill_source"] or "",
            )
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit S34 paper trades for chronological close corruption.")
    parser.add_argument("--db", default="data/microstructure.db")
    parser.add_argument("--trades-json", default="reports/research/s34/S34_SHADOW_PAPER_TRADES.json")
    parser.add_argument("--out-json", default="reports/research/s34/S34_PAPER_CHRONOLOGY_AUDIT.json")
    parser.add_argument("--out-md", default="reports/research/s34/S34_PAPER_CHRONOLOGY_AUDIT.md")
    args = parser.parse_args()
    result = audit(Path(args.db), Path(args.trades_json))
    write_reports(result, Path(args.out_json), Path(args.out_md))
    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md, **{k: result[k] for k in ("trades_checked", "anomalies")}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
