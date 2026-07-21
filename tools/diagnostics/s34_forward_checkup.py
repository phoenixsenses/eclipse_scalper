from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_calibration_check import DEFAULT_EXCLUDE, DEFAULT_RULE_NAME, valid_closed_trade


TRADES = Path("reports/research/s34/S34_SHADOW_PAPER_TRADES.json")
STATUS = Path("reports/research/s34/S34_SHADOW_PAPER_STATUS.json")
DB = Path("data/microstructure.db")


def iso_ms(ts_ms: int | float | None) -> str:
    if ts_ms is None:
        return ""
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def fnum(x: object) -> float:
    return float(x or 0.0)


def trade_rule_name(t: dict) -> str:
    return str((t.get("rule") or {}).get("name") or "")


def main() -> None:
    trades = json.loads(TRADES.read_text(encoding="utf-8")).get("trades", [])
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    rule_trades = [t for t in trades if trade_rule_name(t) == DEFAULT_RULE_NAME]
    valid = [t for t in rule_trades if valid_closed_trade(t, DEFAULT_EXCLUDE, DEFAULT_RULE_NAME)[0]]
    valid.sort(key=lambda t: (int(t.get("signal_ts_ms") or 0), str(t.get("trade_id") or "")))

    skip_reasons = Counter(str(t.get("risk_gate_reason") or t.get("exit_reason") or "UNKNOWN") for t in trades if t.get("status") == "SKIPPED")
    rule_skip_reasons = Counter(str(t.get("risk_gate_reason") or t.get("exit_reason") or "UNKNOWN") for t in rule_trades if t.get("status") == "SKIPPED")

    nets = [fnum(t.get("net_bps")) for t in valid]
    tps = [t for t in valid if t.get("exit_reason") == "TP"]
    losses = [t for t in valid if t.get("exit_reason") in {"SL", "BE", "TIME"}]
    by_exit: dict[str, dict[str, float | int]] = {}
    for reason, rows in defaultdict(list, {}).items():
        pass
    exit_groups: dict[str, list[dict]] = defaultdict(list)
    for t in valid:
        exit_groups[str(t.get("exit_reason") or "UNKNOWN")].append(t)
    for reason, rows in sorted(exit_groups.items()):
        by_exit[reason] = {"count": len(rows), "net_bps_sum": round(sum(fnum(t.get("net_bps")) for t in rows), 4)}

    by_day: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "net_bps_sum": 0.0})
    for t in valid:
        day = iso_ms(t.get("signal_ts_ms"))[:10]
        by_day[day]["count"] = int(by_day[day]["count"]) + 1
        by_day[day]["net_bps_sum"] = float(by_day[day]["net_bps_sum"]) + fnum(t.get("net_bps"))

    table = []
    for t in valid:
        table.append({
            "trade_id": t.get("trade_id"),
            "signal_utc": iso_ms(t.get("signal_ts_ms")),
            "exit_utc": t.get("exit_ts_utc") or iso_ms(t.get("exit_ts_ms")),
            "exit": t.get("exit_reason"),
            "entry": round(fnum(t.get("entry_price")), 4),
            "exit_price": round(fnum(t.get("exit_price")), 4),
            "gross": round(fnum(t.get("gross_bps")), 4),
            "entry_adv": round(fnum(t.get("entry_adverse_bps")), 4),
            "exit_adv": round(fnum(t.get("exit_adverse_bps")), 4),
            "spread": round(fnum(t.get("spread_cost_bps")), 4),
            "fee": round(fnum(t.get("fee_cost_bps")), 4),
            "net": round(fnum(t.get("net_bps")), 4),
        })

    now_ms = int(time.time() * 1000)
    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True, timeout=10)
    cur = con.cursor()
    streams = {}
    for table_name in ("liquidations", "book_ticker", "mark_prices", "agg_trades"):
        latest = cur.execute(f"SELECT MAX(ts_ms) FROM {table_name}").fetchone()[0]
        rows_1h = cur.execute(f"SELECT COUNT(*) FROM {table_name} WHERE ts_ms>=?", (now_ms - 3_600_000,)).fetchone()[0]
        streams[table_name] = {
            "latest_utc": iso_ms(latest),
            "minutes_since": None if latest is None else round((now_ms - int(latest)) / 60000.0, 3),
            "rows_1h": rows_1h,
        }
    con.close()

    k1 = None if len(valid) < 40 else statistics.fmean(nets) <= 0.0
    gross_abs = [abs(fnum(t.get("gross_bps"))) for t in valid]
    entry_adv = [fnum(t.get("entry_adverse_bps")) for t in valid]
    k2 = None if len(valid) < 40 or not gross_abs else statistics.median(entry_adv) >= statistics.fmean(gross_abs)
    quarantine = [t for t in trades if str(t.get("risk_gate_reason") or t.get("exit_reason") or "") == "NO_FILL_DATA"]
    k3 = False if not quarantine else None

    out = {
        "status": {
            "total_trials": status.get("total_trades"),
            "closed": status.get("closed_trades"),
            "open": status.get("open_trades"),
            "skipped": status.get("risk_skipped_trades"),
            "valid_n": len(valid),
            "remaining_to_40": max(40 - len(valid), 0),
        },
        "skip_reasons_all": dict(skip_reasons.most_common()),
        "skip_reasons_s34_rule": dict(rule_skip_reasons.most_common()),
        "valid_trades": table,
        "summary": {
            "cum_net_bps": round(sum(nets), 4) if nets else 0.0,
            "avg_net_bps": round(statistics.fmean(nets), 4) if nets else None,
            "median_net_bps": round(statistics.median(nets), 4) if nets else None,
            "win_rate_tp_over_valid": round(len(tps) / len(valid), 4) if valid else None,
            "avg_win_bps_tp": round(statistics.fmean([fnum(t.get("net_bps")) for t in tps]), 4) if tps else None,
            "avg_loss_bps_sl_be_time": round(statistics.fmean([fnum(t.get("net_bps")) for t in losses]), 4) if losses else None,
            "exit_distribution": by_exit,
            "best_winner": max(table, key=lambda x: x["net"], default=None),
            "worst_loser": min(table, key=lambda x: x["net"], default=None),
        },
        "regime_days": {k: {"count": v["count"], "net_bps_sum": round(float(v["net_bps_sum"]), 4)} for k, v in sorted(by_day.items())},
        "regime_day_count": len(by_day),
        "streams": streams,
        "runner_config": status.get("regime_config"),
        "kill_status": {
            "K1_mean_net_le_zero": k1,
            "K2_median_entry_adverse_ge_mean_abs_gross": k2,
            "K3_quarantine_selection_bias": k3,
            "quarantine_count": len(quarantine),
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
