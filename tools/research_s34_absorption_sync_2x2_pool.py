from __future__ import annotations

import bisect
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_cross_asset_absorption_pool import DEFAULT_DB, metrics, summarize_rows
from tools.research_s34_knowable_anchor_continuation import load_liquidations, r1


IN_JSON = ROOT / "reports" / "research" / "s34" / "S34_CROSS_ASSET_ABSORPTION_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.md"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SYNC_WINDOW_SEC = 600
SYNC_THRESHOLD_K = 200.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def window_liq(ts: list[int], rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> float:
    lo = bisect.bisect_right(ts, int(start_ms))
    hi = bisect.bisect_right(ts, int(end_ms))
    return sum(float(rows[i]["notional"]) for i in range(lo, hi))


def load_sell_liqs(conn: sqlite3.Connection) -> dict[str, tuple[list[int], list[dict[str, Any]]]]:
    out = {}
    for symbol in SYMBOLS:
        rows = load_liquidations(conn, symbol, "SELL", None, None)
        out[symbol] = ([int(r["ts_ms"]) for r in rows], rows)
    return out


def enrich_sync(rows: list[dict[str, Any]], liqs: dict[str, tuple[list[int], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        symbol = str(row["symbol"])
        entry_ts = int(row["entry_ts_ms"])
        start_ts = entry_ts - SYNC_WINDOW_SEC * 1000
        other_k = {}
        for other in SYMBOLS:
            if other == symbol:
                continue
            ts, lrows = liqs[other]
            other_k[other] = window_liq(ts, lrows, start_ts, entry_ts) / 1000.0
        market_k = sum(other_k.values())
        enriched = dict(row)
        for other, value in other_k.items():
            enriched[f"{other.lower()}_sell_liq_k"] = r1(value)
        enriched["market_concurrent_k"] = r1(market_k)
        enriched["sync_gate"] = "sync" if market_k >= SYNC_THRESHOLD_K else "idio"
        enriched["other_asset_count_200k"] = sum(1 for value in other_k.values() if value >= 200.0)
        out.append(enriched)
    return out


def split_rows(rows: list[dict[str, Any]], split: dict[str, Any]) -> set[str]:
    return set(split.get("holdout_months", []))


def cell(summary: dict[str, Any]) -> str:
    s = summary["all"] if "all" in summary else summary
    return (
        f"N={s['n']} sum={s['sum_bps']} med={s['median_bps']} "
        f"T3R={s['t3r_bps']} max_loss={s['max_loss_bps']} tail<-100={s['tail_n_lt_-100']}"
    )


def group(rows: list[dict[str, Any]], hold_months: set[str], **where: str) -> dict[str, Any]:
    sub = [r for r in rows if all(str(r.get(k)) == str(v) for k, v in where.items())]
    return summarize_rows(sub, hold_months)


def metric_only(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return metrics([float(r["net_bps"]) for r in rows])


def group_metric(rows: list[dict[str, Any]], **where: str) -> dict[str, Any]:
    return metric_only([r for r in rows if all(str(r.get(k)) == str(v) for k, v in where.items())])


def build_report(rows: list[dict[str, Any]], split: dict[str, Any]) -> dict[str, Any]:
    hold_months = split_rows(rows, split)
    sync_values = ("idio", "sync")
    absorption_values = ("vacuum_like", "mixed", "absorbed")
    bid_values = ("shallow_bid", "deep_bid")
    imbalance_values = ("ask_heavy", "bid_support")

    two_by_three = []
    for sync in sync_values:
        for absorption in absorption_values:
            two_by_three.append(
                {
                    "sync_gate": sync,
                    "absorption_gate": absorption,
                    "summary": group(rows, hold_months, sync_gate=sync, absorption_gate=absorption),
                }
            )

    sync_by_bid = []
    for sync in sync_values:
        for bid in bid_values:
            sync_by_bid.append(
                {
                    "sync_gate": sync,
                    "bid_depth_gate": bid,
                    "summary": group(rows, hold_months, sync_gate=sync, bid_depth_gate=bid),
                }
            )

    sync_by_imbalance = []
    for sync in sync_values:
        for imb in imbalance_values:
            sync_by_imbalance.append(
                {
                    "sync_gate": sync,
                    "imbalance_gate": imb,
                    "summary": group(rows, hold_months, sync_gate=sync, imbalance_gate=imb),
                }
            )

    by_symbol = []
    for symbol in sorted({str(r["symbol"]) for r in rows}):
        srows = [r for r in rows if str(r["symbol"]) == symbol]
        by_symbol.append(
            {
                "symbol": symbol,
                "all": summarize_rows(srows, hold_months),
                "sync": group(srows, hold_months, sync_gate="sync"),
                "idio": group(srows, hold_months, sync_gate="idio"),
                "sync_absorbed": group(srows, hold_months, sync_gate="sync", absorption_gate="absorbed"),
                "sync_mixed_or_vacuum": summarize_rows(
                    [r for r in srows if r["sync_gate"] == "sync" and r["absorption_gate"] != "absorbed"],
                    hold_months,
                ),
            }
        )

    thresholds = []
    for threshold in (0.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0):
        sub = [r for r in rows if float(r.get("market_concurrent_k") or 0.0) >= threshold]
        thresholds.append({"threshold_k": threshold, "summary": summarize_rows(sub, hold_months)})

    return {
        "generated_at_utc": utc_now(),
        "input": str(IN_JSON),
        "sync_window_sec": SYNC_WINDOW_SEC,
        "sync_threshold_k": SYNC_THRESHOLD_K,
        "event_n": len(rows),
        "split": split,
        "overall": summarize_rows(rows, hold_months),
        "sync_gate": {
            "sync": group(rows, hold_months, sync_gate="sync"),
            "idio": group(rows, hold_months, sync_gate="idio"),
        },
        "absorption_gate": {
            value: group(rows, hold_months, absorption_gate=value) for value in absorption_values
        },
        "two_by_three": two_by_three,
        "sync_by_bid_depth": sync_by_bid,
        "sync_by_imbalance": sync_by_imbalance,
        "sync_threshold_sweep": thresholds,
        "by_symbol": by_symbol,
        "rows": rows,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Absorption x Synchronization 2x2 Pool",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. Cross-asset pooled SELL fade rows with knowable prior-window sync.",
        "",
        f"- Rows: `{report['event_n']}`",
        f"- Sync window: `{report['sync_window_sec'] // 60}m`",
        f"- Sync threshold: `{report['sync_threshold_k']}K` other-asset SELL liq",
        f"- Overall: {cell(report['overall'])}",
        "",
        "## Sync Gate",
        "",
        "| Gate | All | Cal | Hold |",
        "| --- | --- | --- | --- |",
    ]
    for gate in ("idio", "sync"):
        s = report["sync_gate"][gate]
        lines.append(f"| `{gate}` | {cell(s)} | {cell(s['cal'])} | {cell(s['hold'])} |")

    lines += [
        "",
        "## Absorption Gate",
        "",
        "| Gate | All | Cal | Hold |",
        "| --- | --- | --- | --- |",
    ]
    for gate in ("vacuum_like", "mixed", "absorbed"):
        s = report["absorption_gate"][gate]
        lines.append(f"| `{gate}` | {cell(s)} | {cell(s['cal'])} | {cell(s['hold'])} |")

    lines += [
        "",
        "## Sync x Absorption",
        "",
        "| Sync | Absorption | All | Cal | Hold |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["two_by_three"]:
        s = row["summary"]
        lines.append(f"| `{row['sync_gate']}` | `{row['absorption_gate']}` | {cell(s)} | {cell(s['cal'])} | {cell(s['hold'])} |")

    lines += [
        "",
        "## Sync x Bid Depth",
        "",
        "| Sync | Bid Depth | All | Cal | Hold |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["sync_by_bid_depth"]:
        s = row["summary"]
        lines.append(f"| `{row['sync_gate']}` | `{row['bid_depth_gate']}` | {cell(s)} | {cell(s['cal'])} | {cell(s['hold'])} |")

    lines += [
        "",
        "## By Symbol",
        "",
        "| Symbol | All | Sync | Idio | Sync+Absorbed | Sync+NotAbsorbed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["by_symbol"]:
        lines.append(
            f"| `{row['symbol']}` | {cell(row['all'])} | {cell(row['sync'])} | {cell(row['idio'])} | "
            f"{cell(row['sync_absorbed'])} | {cell(row['sync_mixed_or_vacuum'])} |"
        )

    lines += [
        "",
        "## Sync Threshold Sweep",
        "",
        "| Threshold K | All | Cal | Hold |",
        "| ---: | --- | --- | --- |",
    ]
    for row in report["sync_threshold_sweep"]:
        s = row["summary"]
        lines.append(f"| {row['threshold_k']} | {cell(s)} | {cell(s['cal'])} | {cell(s['hold'])} |")

    lines += [
        "",
        "## Read",
        "",
        "- If sync improves all-sample but fails holdout, treat it as route/time-period structure rather than a robust resonance gate.",
        "- If absorbed only works inside one symbol, the next model must be hierarchical by route/symbol, not globally pooled.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    payload = json.loads(IN_JSON.read_text(encoding="utf-8"))
    conn = sqlite3.connect(DEFAULT_DB)
    try:
        liqs = load_sell_liqs(conn)
    finally:
        conn.close()
    rows = enrich_sync(list(payload["rows"]), liqs)
    report = build_report(rows, payload.get("split", {}))
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    text = render(report)
    OUT_MD.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
