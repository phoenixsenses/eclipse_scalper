"""S34 next research results.

Observation-only research around the current v0.2 live-matching alpha:

1. Tail-injected compounding stress for CURRENT_ENV / BALANCED / SURVIVAL.
2. Exit expansion summary from the existing forward research pack.
3. Bull-pullback shadow screen: in bull regime, does shallow ETH SELL-liq
   pullback continuation/rebound exist at lower knowable thresholds?

No live executor, live size, leverage, order logic, or .env changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MIRROR_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
FORWARD_PACK = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_RESEARCH_PACK.json"
DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_NEXT_RESEARCH_RESULTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_NEXT_RESEARCH_RESULTS.md"

EQUITY_START = 35.0
RISK_RATIOS = {
    "CURRENT_ENV": 1190.0 / 35.0,
    "STOP_ASSISTED": 39.8 / 35.0,
    "BALANCED": 16.3 / 35.0,
    "SURVIVAL": 11.0 / 35.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def r1(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 1)


def r2(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 2)


def r3(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 3)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            out.append(json.loads(text))
    return out


def closed_net_bps(path: Path = MIRROR_LEDGER) -> list[float]:
    rows = [
        r for r in load_jsonl(path)
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "FILLED"
        and r.get("net_bps") is not None
    ]
    rows.sort(key=lambda r: str(r.get("signal_utc") or ""))
    return [float(r["net_bps"]) for r in rows]


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "sum_bps": 0.0, "median_bps": None, "win_rate": None, "max_loss_bps": None, "t3r_bps": 0.0}
    t3r = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else sum(vals)
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "max_loss_bps": r1(min(vals)),
        "t3r_bps": r1(t3r),
    }


def compound(vals: list[float], ratio: float) -> dict[str, Any]:
    equity = EQUITY_START
    peak = equity
    max_dd = 0.0
    path = []
    ruined_at = None
    for i, bps in enumerate(vals, start=1):
        equity *= 1.0 + float(ratio) * float(bps) / 10_000.0
        peak = max(peak, equity)
        dd = equity - peak
        max_dd = min(max_dd, dd)
        path.append({"i": i, "bps": r1(bps), "equity": r3(equity), "drawdown": r3(dd)})
        if ruined_at is None and equity <= 0:
            ruined_at = i
    return {
        "end_equity": r3(equity),
        "multiple": r3(equity / EQUITY_START),
        "max_drawdown_usdt": r3(max_dd),
        "max_drawdown_pct": r3(abs(max_dd) / EQUITY_START * 100.0),
        "ruined_at": ruined_at,
        "path": path,
    }


def inject_tails(vals: list[float], *, every: int | None = None, tail_bps: float | None = None, append_tail_bps: float | None = None) -> list[float]:
    out = []
    for i, v in enumerate(vals, start=1):
        out.append(float(v))
        if every and tail_bps is not None and i % int(every) == 0:
            out.append(float(tail_bps))
    if append_tail_bps is not None:
        out.append(float(append_tail_bps))
    return out


def tail_compounding_report(vals: list[float]) -> dict[str, Any]:
    scenarios = {
        "observed_11": vals,
        "append_minus150": inject_tails(vals, append_tail_bps=-150.0),
        "append_minus300": inject_tails(vals, append_tail_bps=-300.0),
        "append_minus507": inject_tails(vals, append_tail_bps=-507.0),
        "every5_minus150": inject_tails(vals, every=5, tail_bps=-150.0),
        "every10_minus300": inject_tails(vals, every=10, tail_bps=-300.0),
    }
    return {
        name: {mode: compound(seq, ratio) for mode, ratio in RISK_RATIOS.items()}
        for name, seq in scenarios.items()
    }


def mark_at_or_before(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> tuple[int, float] | None:
    row = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return (int(row[0]), float(row[1])) if row else None


def ret_bps(conn: sqlite3.Connection, symbol: str, ts_ms: int, window_sec: int) -> float | None:
    a = mark_at_or_before(conn, symbol, int(ts_ms) - int(window_sec) * 1000)
    b = mark_at_or_before(conn, symbol, int(ts_ms))
    if not a or not b or a[1] <= 0:
        return None
    return (b[1] - a[1]) / a[1] * 10_000.0


def forward_ret_bps(conn: sqlite3.Connection, symbol: str, ts_ms: int, horizon_sec: int, fee_bps: float) -> float | None:
    a = mark_at_or_before(conn, symbol, int(ts_ms))
    b = mark_at_or_before(conn, symbol, int(ts_ms) + int(horizon_sec) * 1000)
    if not a or not b or a[1] <= 0:
        return None
    return (b[1] - a[1]) / a[1] * 10_000.0 - float(fee_bps)


def eth_mark(conn: sqlite3.Connection, ts_ms: int) -> float | None:
    row = mark_at_or_before(conn, "ETHUSDT", ts_ms)
    return row[1] if row else None


def latest_mark_ts(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol='ETHUSDT'").fetchone()
    return int(row[0])


def bull_ok(conn: sqlite3.Connection, ts_ms: int) -> bool:
    eth_1h = ret_bps(conn, "ETHUSDT", ts_ms, 3600)
    eth_4h = ret_bps(conn, "ETHUSDT", ts_ms, 4 * 3600)
    btc_1h = ret_bps(conn, "BTCUSDT", ts_ms, 3600)
    btc_4h = ret_bps(conn, "BTCUSDT", ts_ms, 4 * 3600)
    if None in (eth_1h, eth_4h, btc_1h, btc_4h):
        return False
    return float(eth_1h) >= 20.0 and float(eth_4h) >= 80.0 and float(btc_1h) >= 0.0 and float(btc_4h) >= 50.0


def shallow_vdepth_bps(conn: sqlite3.Connection, first_ts: int, ts_ms: int) -> float | None:
    first = eth_mark(conn, first_ts)
    cur = eth_mark(conn, ts_ms)
    if first is None or cur is None or first <= 0:
        return None
    return (first - cur) / first * 10_000.0


def bull_pullback_screen(conn: sqlite3.Connection, *, days: int = 120) -> dict[str, Any]:
    end_ms = latest_mark_ts(conn)
    start_ms = end_ms - int(days) * 86_400_000
    liqs = [
        {"ts_ms": int(ts), "notional": float(notional)}
        for ts, notional in conn.execute(
            "SELECT ts_ms, notional FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' "
            "AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
            (start_ms, end_ms),
        ).fetchall()
    ]
    thresholds = [50_000.0, 100_000.0, 150_000.0]
    horizons = [1800, 3600, 7200]
    events = []
    bucket_ms = 300_000
    min_gap_ms = 900_000
    last_by_threshold = {t: -10**18 for t in thresholds}
    bucket_start = None
    first_ts = None
    running = 0.0
    crossed: set[float] = set()
    for liq in liqs:
        ts = int(liq["ts_ms"])
        if bucket_start is None or ts >= bucket_start + bucket_ms:
            bucket_start = (ts // bucket_ms) * bucket_ms
            first_ts = ts
            running = 0.0
            crossed = set()
        running += float(liq["notional"])
        for threshold in thresholds:
            if threshold in crossed or running < threshold:
                continue
            crossed.add(threshold)
            if ts - last_by_threshold[threshold] < min_gap_ms:
                continue
            last_by_threshold[threshold] = ts
            if not bull_ok(conn, ts):
                continue
            vdepth = shallow_vdepth_bps(conn, int(first_ts or ts), ts)
            if vdepth is None or not (5.0 <= float(vdepth) < 28.0):
                continue
            row = {
                "threshold": threshold,
                "signal_ts_ms": ts,
                "signal_utc": iso_ms(ts),
                "running_notional": r1(running),
                "vdepth_bps": r1(vdepth),
                "eth_1h_bps": r1(ret_bps(conn, "ETHUSDT", ts, 3600)),
                "eth_4h_bps": r1(ret_bps(conn, "ETHUSDT", ts, 4 * 3600)),
                "btc_1h_bps": r1(ret_bps(conn, "BTCUSDT", ts, 3600)),
                "btc_4h_bps": r1(ret_bps(conn, "BTCUSDT", ts, 4 * 3600)),
            }
            for h in horizons:
                row[f"net_bps_{h}s"] = r1(forward_ret_bps(conn, "ETHUSDT", ts, h, 5.0))
            events.append(row)
    by_key: dict[str, list[float]] = {}
    for e in events:
        for h in horizons:
            val = e.get(f"net_bps_{h}s")
            if val is not None:
                by_key.setdefault(f"thr{int(e['threshold'])}_h{h}", []).append(float(val))
    return {
        "status": "BULL_PULLBACK_SHADOW_SCREEN",
        "definition": "Bull regime, ETH SELL-liq running threshold 50/100/150K, shallow vdepth 5-28bps, mark-entry forward label minus 5bps.",
        "days": days,
        "events_n": len(events),
        "events": events[:100],
        "by_threshold_horizon": {k: summary(v) for k, v in sorted(by_key.items())},
        "read": "Exploratory shadow only. This is not maker-fill validated and not a live candidate.",
    }


def exit_summary() -> dict[str, Any]:
    pack = load_json(FORWARD_PACK, {})
    return pack.get("exit_management") or {"status": "MISSING_FORWARD_PACK"}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    vals = closed_net_bps(args.shadow_ledger)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        bull = bull_pullback_screen(conn, days=int(args.days))
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "source_shadow_n": len(vals),
        "tail_injected_compounding": tail_compounding_report(vals),
        "exit_expansion": exit_summary(),
        "bull_pullback_shadow": bull,
        "read": "No live executor, leverage, size, order logic, or .env changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Next Research Results",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## Tail-Injected Compounding",
        "",
        "| Scenario | Mode | End Equity | Multiple | Max DD % | Ruined At |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for scenario, modes in report["tail_injected_compounding"].items():
        for mode in ("CURRENT_ENV", "STOP_ASSISTED", "BALANCED", "SURVIVAL"):
            row = modes[mode]
            lines.append(
                f"| {scenario} | {mode} | {row.get('end_equity')} | {row.get('multiple')} | "
                f"{row.get('max_drawdown_pct')} | {row.get('ruined_at')} |"
            )
    lines.extend([
        "",
        "## Exit Expansion Top",
        "",
        "| Variant | N | Sum bps | Median | Win | T3R | Max loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in (report["exit_expansion"].get("best_by_sum") or [])[:7]:
        lines.append(
            f"| {row.get('variant')} | {row.get('n')} | {row.get('sum_bps')} | {row.get('median_bps')} | "
            f"{row.get('win_rate')} | {row.get('t3r_bps')} | {row.get('max_loss_bps')} |"
        )
    lines.extend([
        "",
        "## Bull Pullback Shadow Screen",
        "",
        f"Definition: {report['bull_pullback_shadow']['definition']}",
        "",
        f"Events: `{report['bull_pullback_shadow']['events_n']}`",
        "",
        "| Cell | N | Sum bps | Median | Win | T3R | Max loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cell, row in report["bull_pullback_shadow"]["by_threshold_horizon"].items():
        lines.append(
            f"| {cell} | {row.get('n')} | {row.get('sum_bps')} | {row.get('median_bps')} | "
            f"{row.get('win_rate')} | {row.get('t3r_bps')} | {row.get('max_loss_bps')} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 next research result pack.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--shadow-ledger", type=Path, default=MIRROR_LEDGER)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    md = render_md(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
