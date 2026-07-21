from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable


DEFAULT_DB = Path("data/s34_intelligence.db")
DEFAULT_MD = Path("reports/research/s34/S34_50K_LOSS_POSTMORTEM.md")
DEFAULT_JSON = Path("reports/research/s34/S34_50K_LOSS_POSTMORTEM.json")
RULE_50K = "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"


@dataclass(frozen=True)
class Row:
    trade_id: str
    signal_id: str
    signal_ts_utc: str
    exit_reason: str
    net_bps: float
    gross_bps: float | None
    entry_adverse_bps: float | None
    exit_adverse_bps: float | None
    fee_cost_bps: float | None
    cluster_notional: float | None
    cluster_liq_count: int | None
    shape_label: str | None
    guardrail_level: str
    features: dict[str, Any]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_loads(value: Any) -> dict[str, Any]:
    try:
        data = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _feature(row: Row, key: str) -> float | None:
    aliases = {
        "max_single_liq_share": ("max_single_liq_share", "cluster_max_single_liq_share"),
        "intensity_per_sec": ("intensity_per_sec",),
        "inter_cluster_gap_sec": ("inter_cluster_gap_sec", "prev_liq_gap_sec"),
        "day_trend_bps": ("day_trend_bps",),
        "day_range_bps": ("day_range_bps",),
    }
    for name in aliases.get(key, (key,)):
        value = _float_or_none(row.features.get(name))
        if value is not None:
            return value
    return None


def _load_rows(db_path: Path) -> list[Row]:
    with _connect(db_path) as conn:
        raw = conn.execute(
            """
            SELECT
              o.trade_id, o.signal_id, o.exit_reason, o.net_bps,
              o.gross_bps, o.entry_adverse_bps, o.exit_adverse_bps, o.fee_cost_bps,
              s.signal_ts_utc, s.cluster_notional, s.cluster_liq_count,
              s.cluster_shape_label, s.features_json,
              g.level AS guardrail_level
            FROM s34_outcomes o
            JOIN s34_signals s ON s.signal_id=o.signal_id
            LEFT JOIN s34_model_guardrails g ON g.signal_id=o.signal_id
            WHERE o.rule_name=?
            ORDER BY o.exit_ts_ms ASC, o.trade_id ASC
            """,
            (RULE_50K,),
        ).fetchall()
    rows: list[Row] = []
    for item in raw:
        rows.append(
            Row(
                trade_id=str(item["trade_id"]),
                signal_id=str(item["signal_id"]),
                signal_ts_utc=str(item["signal_ts_utc"]),
                exit_reason=str(item["exit_reason"] or ""),
                net_bps=float(item["net_bps"] or 0.0),
                gross_bps=_float_or_none(item["gross_bps"]),
                entry_adverse_bps=_float_or_none(item["entry_adverse_bps"]),
                exit_adverse_bps=_float_or_none(item["exit_adverse_bps"]),
                fee_cost_bps=_float_or_none(item["fee_cost_bps"]),
                cluster_notional=_float_or_none(item["cluster_notional"]),
                cluster_liq_count=int(item["cluster_liq_count"]) if item["cluster_liq_count"] is not None else None,
                shape_label=str(item["cluster_shape_label"] or "") or None,
                guardrail_level=str(item["guardrail_level"] or "missing"),
                features=_json_loads(item["features_json"]),
            )
        )
    return rows


def _metrics(rows: Iterable[Row]) -> dict[str, Any]:
    items = list(rows)
    nets = [r.net_bps for r in items]
    wins = [x for x in nets if x > 0]
    return {
        "n": len(items),
        "cum_net_bps": round(sum(nets), 2) if nets else 0.0,
        "mean_net_bps": round(mean(nets), 2) if nets else 0.0,
        "median_net_bps": round(median(nets), 2) if nets else 0.0,
        "win_rate_pct": round(100.0 * len(wins) / len(items), 2) if items else 0.0,
    }


def _sum_costs(rows: Iterable[Row]) -> dict[str, Any]:
    items = list(rows)
    def avg(attr: str) -> float | None:
        values = [getattr(row, attr) for row in items if getattr(row, attr) is not None]
        return round(mean(values), 2) if values else None
    return {
        "avg_gross_bps": avg("gross_bps"),
        "avg_entry_adverse_bps": avg("entry_adverse_bps"),
        "avg_exit_adverse_bps": avg("exit_adverse_bps"),
        "avg_fee_bps": avg("fee_cost_bps"),
    }


def _bin_notional(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 100_000:
        return "<100K"
    if value < 200_000:
        return "100K-200K"
    if value < 500_000:
        return "200K-500K"
    if value < 1_000_000:
        return "500K-1M"
    return ">=1M"


def _bin_count(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    return ">10"


def _bin_share(value: float | None) -> str:
    if value is None:
        return "missing"
    if value >= 90:
        return ">=90"
    if value >= 80:
        return "80-90"
    if value >= 50:
        return "50-80"
    return "<50"


def _group(rows: list[Row], key_fn: Callable[[Row], str]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        buckets[key_fn(row)].append(row)
    return {
        key: {**_metrics(items), **_sum_costs(items)}
        for key, items in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def _scenario(rows: list[Row], name: str, pred: Callable[[Row], bool]) -> dict[str, Any]:
    blocked = [r for r in rows if pred(r)]
    kept_ids = {(r.trade_id, r.signal_id) for r in blocked}
    kept = [r for r in rows if (r.trade_id, r.signal_id) not in kept_ids]
    base = _metrics(rows)
    kept_m = _metrics(kept)
    return {
        "name": name,
        "blocked": {**_metrics(blocked), **_sum_costs(blocked)},
        "kept": {
            **kept_m,
            "delta_cum_vs_base_bps": round(kept_m["cum_net_bps"] - base["cum_net_bps"], 2),
            "blocked_n": len(blocked),
        },
        "examples": [
            _row_dict(row)
            for row in sorted(blocked, key=lambda r: r.net_bps)[:20]
        ],
    }


def _row_dict(row: Row) -> dict[str, Any]:
    return {
        "trade_id": row.trade_id,
        "signal_ts_utc": row.signal_ts_utc,
        "exit_reason": row.exit_reason,
        "net_bps": round(row.net_bps, 2),
        "gross_bps": None if row.gross_bps is None else round(row.gross_bps, 2),
        "entry_adverse_bps": None if row.entry_adverse_bps is None else round(row.entry_adverse_bps, 2),
        "exit_adverse_bps": None if row.exit_adverse_bps is None else round(row.exit_adverse_bps, 2),
        "fee_cost_bps": row.fee_cost_bps,
        "cluster_notional": row.cluster_notional,
        "cluster_liq_count": row.cluster_liq_count,
        "max_single_liq_share": _feature(row, "max_single_liq_share"),
        "guardrail_level": row.guardrail_level,
        "day_trend_bps": _feature(row, "day_trend_bps"),
        "intensity_per_sec": _feature(row, "intensity_per_sec"),
        "inter_cluster_gap_sec": _feature(row, "inter_cluster_gap_sec"),
    }


def build_payload(db_path: Path) -> dict[str, Any]:
    rows = _load_rows(db_path)
    scenarios = [
        _scenario(
            rows,
            "warning_100k_200k",
            lambda r: r.guardrail_level == "warning"
            and r.cluster_notional is not None
            and 100_000 <= r.cluster_notional < 200_000,
        ),
        _scenario(
            rows,
            "warning_lt200k",
            lambda r: r.guardrail_level == "warning"
            and r.cluster_notional is not None
            and r.cluster_notional < 200_000,
        ),
        _scenario(
            rows,
            "max_single_share_ge80",
            lambda r: (_feature(r, "max_single_liq_share") or -1.0) >= 80.0,
        ),
        _scenario(
            rows,
            "liq_count_le2",
            lambda r: r.cluster_liq_count is not None and r.cluster_liq_count <= 2,
        ),
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rule": RULE_50K,
        "overall": {**_metrics(rows), **_sum_costs(rows)},
        "by_exit": _group(rows, lambda r: r.exit_reason or "unknown"),
        "by_guardrail": _group(rows, lambda r: r.guardrail_level),
        "by_notional": _group(rows, lambda r: _bin_notional(r.cluster_notional)),
        "by_liq_count": _group(rows, lambda r: _bin_count(r.cluster_liq_count)),
        "by_max_single_share": _group(rows, lambda r: _bin_share(_feature(r, "max_single_liq_share"))),
        "scenarios": scenarios,
        "latest_12": [_row_dict(row) for row in rows[-12:]],
        "worst_12": [_row_dict(row) for row in sorted(rows, key=lambda r: r.net_bps)[:12]],
        "read": (
            "This report is diagnostic only. It identifies weak 50K/TP120 sub-buckets; "
            "it does not change runner config or promote a live filter."
        ),
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _metric_rows(grouped: dict[str, dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            key,
            m["n"],
            m["cum_net_bps"],
            m["mean_net_bps"],
            m["median_net_bps"],
            m["win_rate_pct"],
            m.get("avg_gross_bps"),
            m.get("avg_entry_adverse_bps"),
            m.get("avg_exit_adverse_bps"),
        ]
        for key, m in grouped.items()
    ]


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    overall = payload["overall"]
    scenario_rows = [
        [
            s["name"],
            s["blocked"]["n"],
            s["blocked"]["cum_net_bps"],
            s["blocked"]["median_net_bps"],
            s["blocked"]["win_rate_pct"],
            s["kept"]["cum_net_bps"],
            s["kept"]["delta_cum_vs_base_bps"],
        ]
        for s in payload["scenarios"]
    ]
    worst_rows = [
        [
            row["trade_id"],
            row["exit_reason"],
            row["net_bps"],
            row["gross_bps"],
            row["entry_adverse_bps"],
            row["exit_adverse_bps"],
            row["cluster_notional"],
            row["cluster_liq_count"],
            row["max_single_liq_share"],
            row["guardrail_level"],
        ]
        for row in payload["worst_12"]
    ]
    lines = [
        "# S34 50K/TP120 Loss Postmortem",
        "",
        f"Generated at: `{payload['generated_at_utc']}`",
        "",
        "Scope: closed `ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30` trades only. Read-only diagnostic.",
        "",
        "## Overall",
        "",
        _table(
            ["N", "Cum", "Mean", "Median", "WR %", "Avg Gross", "Avg Entry Adv", "Avg Exit Adv", "Avg Fee"],
            [[overall["n"], overall["cum_net_bps"], overall["mean_net_bps"], overall["median_net_bps"], overall["win_rate_pct"], overall["avg_gross_bps"], overall["avg_entry_adverse_bps"], overall["avg_exit_adverse_bps"], overall["avg_fee_bps"]]],
        ),
        "",
        "## Breakdown",
        "",
        "### By Exit",
        "",
        _table(["Exit", "N", "Cum", "Mean", "Median", "WR %", "Gross", "EntryAdv", "ExitAdv"], _metric_rows(payload["by_exit"])),
        "",
        "### By Guardrail",
        "",
        _table(["Guard", "N", "Cum", "Mean", "Median", "WR %", "Gross", "EntryAdv", "ExitAdv"], _metric_rows(payload["by_guardrail"])),
        "",
        "### By Cluster Notional",
        "",
        _table(["Notional", "N", "Cum", "Mean", "Median", "WR %", "Gross", "EntryAdv", "ExitAdv"], _metric_rows(payload["by_notional"])),
        "",
        "### By Liquidation Count",
        "",
        _table(["Liq Count", "N", "Cum", "Mean", "Median", "WR %", "Gross", "EntryAdv", "ExitAdv"], _metric_rows(payload["by_liq_count"])),
        "",
        "### By Max Single Liquidation Share",
        "",
        _table(["Share", "N", "Cum", "Mean", "Median", "WR %", "Gross", "EntryAdv", "ExitAdv"], _metric_rows(payload["by_max_single_share"])),
        "",
        "## Shadow Removal Scenarios",
        "",
        _table(["Scenario", "Blocked N", "Blocked Cum", "Blocked Median", "Blocked WR %", "Kept Cum", "Kept Delta"], scenario_rows),
        "",
        "## Worst Trades",
        "",
        _table(["Trade", "Exit", "Net", "Gross", "EntryAdv", "ExitAdv", "Cluster", "Count", "Share", "Guard"], worst_rows),
        "",
        "## Read",
        "",
        payload["read"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="S34 50K route loss postmortem.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()
    payload = build_payload(args.db)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(args.out_md, payload)
    print(
        json.dumps(
            {
                "n": payload["overall"]["n"],
                "cum_net_bps": payload["overall"]["cum_net_bps"],
                "median_net_bps": payload["overall"]["median_net_bps"],
                "warning_100k_200k": payload["scenarios"][0]["blocked"],
                "out_md": str(args.out_md),
                "out_json": str(args.out_json),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
