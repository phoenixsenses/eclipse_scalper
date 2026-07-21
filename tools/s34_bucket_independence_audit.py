from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRADES_PATH = ROOT / "reports" / "research" / "s34" / "S34_SHADOW_PAPER_TRADES.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_BUCKET_INDEPENDENCE_AUDIT.md"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_BUCKET_INDEPENDENCE_AUDIT.json"
OUT_CSV = ROOT / "reports" / "research" / "s34" / "S34_BUCKET_INDEPENDENCE_PAIRS.csv"

HIDDEN_RULES = {"ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"}
EXCLUDED_TRADE_IDS = {"P013", "P056"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_from_ms(ts_ms: int | float | None) -> str:
    if not ts_ms:
        return ""
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).date().isoformat()


def _rule_name(trade: dict[str, Any]) -> str:
    return str((trade.get("rule") or {}).get("name") or "")


def _rule_side(rule_name: str) -> str:
    return "SHORT" if "_SHORT_" in rule_name else "LONG"


def _rule_symbol(trade: dict[str, Any]) -> str:
    return str(trade.get("symbol") or (trade.get("rule") or {}).get("symbol") or "")


def _clean_closed(trade: dict[str, Any]) -> bool:
    if str(trade.get("trade_id") or "") in EXCLUDED_TRADE_IDS:
        return False
    if _rule_name(trade) in HIDDEN_RULES:
        return False
    if trade.get("status") != "CLOSED":
        return False
    if trade.get("net_bps") is None:
        return False
    if (trade.get("entry_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    if (trade.get("exit_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    return True


def _signal_ts(trade: dict[str, Any]) -> int:
    return int(trade.get("signal_ts_ms") or 0)


def _entry_ts(trade: dict[str, Any]) -> int:
    return int(trade.get("entry_ts_ms") or trade.get("signal_ts_ms") or 0)


def _exit_ts(trade: dict[str, Any]) -> int:
    return int(trade.get("exit_ts_ms") or 0)


def _bucket(trade: dict[str, Any]) -> int | None:
    signal = trade.get("signal") or {}
    if signal.get("bucket") is None:
        return None
    try:
        return int(signal.get("bucket"))
    except (TypeError, ValueError):
        return None


def _liq_side(trade: dict[str, Any]) -> str:
    rule = trade.get("rule") or {}
    signal = trade.get("signal") or {}
    return str(rule.get("liq_side") or signal.get("liq_side") or "").upper()


def _corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _family_verdict(pair: dict[str, Any]) -> str:
    if pair["same_cascade_pairs"] > 0:
        return "SAME_FAMILY"
    if pair["co_trigger_15m_pairs"] >= 2 or (pair["daily_overlap_days"] >= 3 and (pair["daily_corr"] or 0.0) >= 0.35):
        return "RELATED"
    if pair["daily_overlap_days"] >= 3 and (pair["daily_corr"] or 0.0) <= -0.35:
        return "ANTI_CORRELATED"
    return "LIKELY_INDEPENDENT"


def load_trades(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("trades", []) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError(f"unexpected trades JSON shape: {type(rows).__name__}")
    return [row for row in rows if isinstance(row, dict)]


def build_audit(trades: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [t for t in trades if _clean_closed(t)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in closed:
        grouped[_rule_name(trade)].append(trade)
    for values in grouped.values():
        values.sort(key=lambda t: (_entry_ts(t), str(t.get("trade_id") or "")))

    route_stats = []
    for rule, rows in sorted(grouped.items()):
        nets = [float(t.get("net_bps") or 0.0) for t in rows]
        days = {_date_from_ms(_signal_ts(t)) for t in rows if _date_from_ms(_signal_ts(t))}
        route_stats.append(
            {
                "rule": rule,
                "symbol": _rule_symbol(rows[0]) if rows else "",
                "side": _rule_side(rule),
                "n": len(rows),
                "days": len(days),
                "cum_net_bps": sum(nets),
                "median_net_bps": statistics.median(nets) if nets else None,
                "mean_net_bps": statistics.fmean(nets) if nets else None,
                "win_rate": sum(1 for value in nets if value > 0.0) / len(nets) if nets else None,
            }
        )

    pairs = []
    for rule_a, rule_b in itertools.combinations(sorted(grouped), 2):
        a = grouped[rule_a]
        b = grouped[rule_b]
        same_cascade = 0
        co_5 = 0
        co_15 = 0
        overlap_life = 0
        for ta in a:
            for tb in b:
                gap_sec = abs(_signal_ts(ta) - _signal_ts(tb)) / 1000.0
                if gap_sec <= 300:
                    co_5 += 1
                if gap_sec <= 900:
                    co_15 += 1
                if (
                    _rule_symbol(ta) == _rule_symbol(tb)
                    and _rule_side(rule_a) == _rule_side(rule_b)
                    and _liq_side(ta) == _liq_side(tb)
                    and _bucket(ta) is not None
                    and _bucket(ta) == _bucket(tb)
                ):
                    same_cascade += 1
                a_start, a_end = _entry_ts(ta), _exit_ts(ta)
                b_start, b_end = _entry_ts(tb), _exit_ts(tb)
                if a_start and a_end and b_start and b_end and max(a_start, b_start) <= min(a_end, b_end):
                    overlap_life += 1

        by_day_a: dict[str, float] = defaultdict(float)
        by_day_b: dict[str, float] = defaultdict(float)
        for trade in a:
            by_day_a[_date_from_ms(_signal_ts(trade))] += float(trade.get("net_bps") or 0.0)
        for trade in b:
            by_day_b[_date_from_ms(_signal_ts(trade))] += float(trade.get("net_bps") or 0.0)
        overlap_days = sorted(set(by_day_a) & set(by_day_b))
        xs = [by_day_a[day] for day in overlap_days]
        ys = [by_day_b[day] for day in overlap_days]
        daily_corr = _corr(xs, ys)
        both_loss_days = sum(1 for x, y in zip(xs, ys) if x < 0 and y < 0)
        both_win_days = sum(1 for x, y in zip(xs, ys) if x > 0 and y > 0)
        pair = {
            "route_a": rule_a,
            "route_b": rule_b,
            "n_a": len(a),
            "n_b": len(b),
            "same_cascade_pairs": same_cascade,
            "co_trigger_5m_pairs": co_5,
            "co_trigger_15m_pairs": co_15,
            "overlapping_lifecycle_pairs": overlap_life,
            "daily_overlap_days": len(overlap_days),
            "daily_corr": daily_corr,
            "both_loss_days": both_loss_days,
            "both_win_days": both_win_days,
        }
        pair["verdict"] = _family_verdict(pair)
        pairs.append(pair)

    return {
        "generated_at_utc": _utc_now(),
        "closed_clean_n": len(closed),
        "route_count": len(grouped),
        "route_stats": route_stats,
        "pairs": pairs,
    }


def write_outputs(audit: dict[str, Any], md_path: Path, json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "route_a",
                "route_b",
                "n_a",
                "n_b",
                "same_cascade_pairs",
                "co_trigger_5m_pairs",
                "co_trigger_15m_pairs",
                "overlapping_lifecycle_pairs",
                "daily_overlap_days",
                "daily_corr",
                "both_loss_days",
                "both_win_days",
                "verdict",
            ],
        )
        writer.writeheader()
        writer.writerows(audit["pairs"])

    route_lines = [
        "| Route | Side | N | Days | Median | Mean | Cum | WR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(audit["route_stats"], key=lambda x: x["cum_net_bps"], reverse=True):
        route_lines.append(
            f"| `{row['rule']}` | {row['side']} | {row['n']} | {row['days']} | "
            f"{_fmt(row['median_net_bps'])} | {_fmt(row['mean_net_bps'])} | "
            f"{_fmt(row['cum_net_bps'])} | {_fmt((row['win_rate'] or 0.0) * 100, 1)}% |"
        )

    pair_lines = [
        "| Route A | Route B | same cascade | co-trigger 15m | overlap days | daily corr | both loss days | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(audit["pairs"], key=lambda x: (x["verdict"], -x["same_cascade_pairs"], -x["co_trigger_15m_pairs"])):
        pair_lines.append(
            f"| `{row['route_a']}` | `{row['route_b']}` | {row['same_cascade_pairs']} | "
            f"{row['co_trigger_15m_pairs']} | {row['daily_overlap_days']} | "
            f"{_fmt(row['daily_corr'])} | {row['both_loss_days']} | **{row['verdict']}** |"
        )

    verdict_counts: dict[str, int] = defaultdict(int)
    for row in audit["pairs"]:
        verdict_counts[row["verdict"]] += 1
    verdict_text = ", ".join(f"{key}={value}" for key, value in sorted(verdict_counts.items())) or "none"

    md = [
        "# S34 Bucket Independence Audit",
        "",
        f"- generated_at_utc: `{audit['generated_at_utc']}`",
        f"- clean_closed_trades: `{audit['closed_clean_n']}`",
        f"- routes: `{audit['route_count']}`",
        f"- pair_verdict_counts: `{verdict_text}`",
        "",
        "## Route Stats",
        "",
        *route_lines,
        "",
        "## Pairwise Independence",
        "",
        *pair_lines,
        "",
        "## Interpretation Rules",
        "",
        "- `SAME_FAMILY`: same symbol/direction/liq-side bucket overlap was observed.",
        "- `RELATED`: repeated 15-minute co-triggers or positive daily PnL correlation.",
        "- `ANTI_CORRELATED`: repeated overlap days with negative daily correlation.",
        "- `LIKELY_INDEPENDENT`: no current evidence of tight coupling, but small-N routes remain provisional.",
        "",
        "This is an audit report only. It does not change runner rules, route thresholds, or risk sizing.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit independence across S34 bucket routes.")
    parser.add_argument("--trades", default=str(TRADES_PATH))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    audit = build_audit(load_trades(Path(args.trades)))
    write_outputs(audit, Path(args.out_md), Path(args.out_json), Path(args.out_csv))
    print("=== S34 BUCKET INDEPENDENCE AUDIT ===")
    print(f"Clean closed trades : {audit['closed_clean_n']}")
    print(f"Routes              : {audit['route_count']}")
    counts: dict[str, int] = defaultdict(int)
    for row in audit["pairs"]:
        counts[row["verdict"]] += 1
    for key in sorted(counts):
        print(f"{key:20s}: {counts[key]}")
    print(f"Report              : {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
