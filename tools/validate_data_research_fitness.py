from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List

from data.features.micro_features import compute_features, load_symbol_window
from tools.check_data_ready import check_db_fresh
from tools.run_summary import build_run_summary
from tools.validate_microstructure_contract import analyze_contract


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _csv_status(csv_path: Path, fresh_sec: int, now: float) -> Dict[str, Any]:
    if not csv_path.exists():
        return {"present": False, "fresh": False, "age_sec": None, "detail": f"missing:{csv_path}"}
    age = max(0.0, float(now) - float(csv_path.stat().st_mtime))
    return {
        "present": True,
        "fresh": bool(age <= max(1, int(fresh_sec))),
        "age_sec": age,
        "detail": "fresh" if age <= max(1, int(fresh_sec)) else f"stale:age_sec={age:.1f}",
    }


def _symbol_sample_stats(conn: sqlite3.Connection, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    tables = set()
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        tables = {str(r[0]) for r in rows if r and r[0]}
    except Exception:
        tables = set()
    for symbol in symbols:
        trades = (
            int(conn.execute("SELECT COUNT(*) FROM agg_trades WHERE symbol = ?", (symbol,)).fetchone()[0])
            if "agg_trades" in tables
            else 0
        )
        marks = (
            int(conn.execute("SELECT COUNT(*) FROM mark_prices WHERE symbol = ?", (symbol,)).fetchone()[0])
            if "mark_prices" in tables
            else 0
        )
        liqs = (
            int(conn.execute("SELECT COUNT(*) FROM liquidations WHERE symbol = ?", (symbol,)).fetchone()[0])
            if "liquidations" in tables
            else 0
        )
        out[symbol] = {
            "agg_trade_rows": trades,
            "mark_price_rows": marks,
            "liquidation_rows": liqs,
        }
    return out


def _feature_fitness(conn: sqlite3.Connection, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for symbol in symbols:
        records = load_symbol_window(conn, symbol)
        features = compute_features(records, volatility_window=5) if records else []
        out[symbol] = {
            "records": int(len(records)),
            "feature_rows": int(len(features)),
            "has_mid": bool(any(row.get("mid") is not None for row in features)),
            "has_spread": bool(any(row.get("spread") is not None for row in features)),
            "has_trade_intensity": bool(any((row.get("trade_intensity") or 0) > 0 for row in features)),
        }
    return out


def analyze_research_fitness(
    db_path: Path,
    csv_path: Path,
    symbols: List[str],
    *,
    fresh_sec: int = 120,
    min_trade_rows_per_symbol: int = 10,
    now: float | None = None,
) -> Dict[str, Any]:
    now_ts = float(now if now is not None else time.time())
    failures: List[str] = []
    warnings: List[str] = []

    if not db_path.exists():
        failures.append(f"missing_db:{db_path}")
        payload = {
            "status": "fail",
            "db": str(db_path),
            "csv": str(csv_path),
            "symbols": list(symbols),
            "failures": failures,
            "warnings": warnings,
        }
        payload["run_summary"] = build_run_summary(
            run_type="validate_data_research_fitness",
            inputs={"db": str(db_path), "csv": str(csv_path), "symbols": list(symbols)},
            metrics={"status": "fail", "failure_count": len(failures), "warning_count": len(warnings)},
            artifacts={"json": "reports/DATA_RESEARCH_FITNESS.json", "md": "reports/DATA_RESEARCH_FITNESS.md"},
        )
        return payload

    csv_status = _csv_status(csv_path, fresh_sec=fresh_sec, now=now_ts)
    if not csv_status["present"]:
        failures.append("missing_event_diary_csv")
    elif not csv_status["fresh"]:
        warnings.append("stale_event_diary_csv")

    conn = sqlite3.connect(str(db_path))
    try:
        ready_ok, ready_details, _table_diags = check_db_fresh(conn, symbols=symbols, fresh_sec=fresh_sec, now=now_ts)
        if not ready_ok:
            failures.append("db_not_ready")

        contract = analyze_contract(db_path, symbols)
        if contract["status"] == "fail":
            failures.append("contract_fail")
        elif contract["status"] == "warn":
            warnings.append("contract_warn")

        sample_stats = _symbol_sample_stats(conn, symbols)
        feature_stats = _feature_fitness(conn, symbols)
    finally:
        conn.close()

    for symbol in symbols:
        sample = sample_stats[symbol]
        feature = feature_stats[symbol]
        if int(sample["agg_trade_rows"]) < int(min_trade_rows_per_symbol):
            warnings.append(f"low_trade_rows:{symbol}")
        if not feature["feature_rows"]:
            failures.append(f"no_feature_rows:{symbol}")
            continue
        if not feature["has_mid"]:
            failures.append(f"no_mid:{symbol}")
        if not feature["has_trade_intensity"]:
            failures.append(f"no_trade_intensity:{symbol}")
        if not feature["has_spread"]:
            warnings.append(f"no_spread:{symbol}")

    status = "pass"
    if failures:
        status = "fail"
    elif warnings:
        status = "warn"

    payload = {
        "status": status,
        "db": str(db_path),
        "csv": str(csv_path),
        "symbols": list(symbols),
        "fresh_sec": int(fresh_sec),
        "min_trade_rows_per_symbol": int(min_trade_rows_per_symbol),
        "csv_status": csv_status,
        "db_ready": bool(ready_ok),
        "db_ready_details": ready_details,
        "contract": {
            "status": contract["status"],
            "tier": contract["feature_capability"]["tier"],
            "requires_book": contract["feature_capability"]["requires_book"],
        },
        "sample_stats": sample_stats,
        "feature_stats": feature_stats,
        "warnings": warnings,
        "failures": failures,
    }
    payload["run_summary"] = build_run_summary(
        run_type="validate_data_research_fitness",
        inputs={
            "db": str(db_path),
            "csv": str(csv_path),
            "symbols": list(symbols),
            "fresh_sec": int(fresh_sec),
            "min_trade_rows_per_symbol": int(min_trade_rows_per_symbol),
        },
        metrics={
            "status": status,
            "db_ready": bool(ready_ok),
            "contract_status": contract["status"],
            "warning_count": len(warnings),
            "failure_count": len(failures),
        },
        artifacts={"json": "reports/DATA_RESEARCH_FITNESS.json", "md": "reports/DATA_RESEARCH_FITNESS.md"},
    )
    return payload


def _write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Data Research Fitness Report")
    lines.append("")
    lines.append(f"- Status: `{payload['status']}`")
    lines.append(f"- DB ready: `{payload['db_ready']}`")
    lines.append(f"- Contract: `{payload['contract']['status']}`")
    lines.append(f"- Capability tier: `{payload['contract']['tier']}`")
    lines.append(f"- Event diary CSV: `{payload['csv_status']['detail']}`")
    lines.append("")
    lines.append("## Symbol Fitness")
    lines.append("")
    lines.append("| symbol | trade_rows | mark_rows | liq_rows | feature_rows | mid | spread | trade_intensity |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for symbol in payload["symbols"]:
        sample = payload["sample_stats"][symbol]
        feature = payload["feature_stats"][symbol]
        lines.append(
            f"| `{symbol}` | {sample['agg_trade_rows']} | {sample['mark_price_rows']} | {sample['liquidation_rows']} | "
            f"{feature['feature_rows']} | {feature['has_mid']} | {feature['has_spread']} | {feature['has_trade_intensity']} |"
        )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if payload["failures"]:
        for item in payload["failures"]:
            lines.append(f"- fail: `{item}`")
    if payload["warnings"]:
        for item in payload["warnings"]:
            lines.append(f"- warn: `{item}`")
    if not payload["failures"] and not payload["warnings"]:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate whether microstructure data is fit for research use.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--csv", default="data/event_diary.csv")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--fresh-sec", type=int, default=120)
    p.add_argument("--min-trade-rows-per-symbol", type=int, default=10)
    p.add_argument("--out-json", default="reports/DATA_RESEARCH_FITNESS.json")
    p.add_argument("--out-md", default="reports/DATA_RESEARCH_FITNESS.md")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    payload = analyze_research_fitness(
        db_path=Path(str(args.db)),
        csv_path=Path(str(args.csv)),
        symbols=_parse_symbols(args.symbols),
        fresh_sec=int(args.fresh_sec),
        min_trade_rows_per_symbol=int(args.min_trade_rows_per_symbol),
    )
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(out_md, payload)
    print(
        f"validate_data_research_fitness status={payload['status']} "
        f"warnings={len(payload['warnings'])} failures={len(payload['failures'])}"
    )
    return 0 if payload["status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
