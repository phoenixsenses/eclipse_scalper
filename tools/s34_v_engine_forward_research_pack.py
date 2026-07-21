"""S34 V Engine forward research pack.

Runs the next observation-only research layer around the live-matching v0.2
shadow mirror:

1. Bull/neutral/risk-off regime tags.
2. Exit-management variants on the same entries.
3. Live-vs-system sizing equity comparison.
4. Fill/queue realism diagnostics.
5. 30/60-day forward decision gate snapshot.

No live executor, order logic, .env, or size is changed.
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

DEFAULT_DB = ROOT / "data" / "microstructure.db"
MIRROR_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
SIZING_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_SIZING_SHADOW_PAPER_LEDGER.jsonl"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_RESEARCH_PACK.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_RESEARCH_PACK.md"

STATUS = "RESEARCH_OBSERVATION_ONLY_NO_LIVE_CHANGE"
RULE_ID = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
FEE_BPS_DEFAULT = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def parse_iso_ms(text: str) -> int:
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def r1(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 1)


def r2(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 2)


def r3(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 3)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def closed_fills(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [
        r
        for r in rows
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "FILLED"
        and r.get("entry_price") is not None
        and r.get("maker_fill_ts_ms") is not None
    ]
    out.sort(key=lambda r: int(r["maker_fill_ts_ms"]))
    return out


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate": None,
            "max_loss_bps": None,
            "t3r_bps": 0.0,
        }
    ordered = sorted(vals)
    t3r = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else sum(vals)
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "max_loss_bps": r1(min(vals)),
        "p10_bps": r1(ordered[max(0, int(len(ordered) * 0.1) - 1)]),
        "p90_bps": r1(ordered[min(len(ordered) - 1, int(math.ceil(len(ordered) * 0.9)) - 1)]),
        "t3r_bps": r1(t3r),
    }


def mark_at_or_before(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> tuple[int, float] | None:
    row = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return (int(row[0]), float(row[1])) if row else None


def mark_range(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    return [
        (int(ts), float(px))
        for ts, px in conn.execute(
            "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms",
            (symbol, int(start_ms), int(end_ms)),
        ).fetchall()
    ]


def ret_bps(conn: sqlite3.Connection, symbol: str, ts_ms: int, window_sec: int) -> float | None:
    start = mark_at_or_before(conn, symbol, int(ts_ms) - int(window_sec) * 1000)
    end = mark_at_or_before(conn, symbol, int(ts_ms))
    if not start or not end or start[1] <= 0:
        return None
    return (end[1] - start[1]) / start[1] * 10_000.0


def funding_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = conn.execute(
        "SELECT funding_rate FROM mark_prices WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL "
        "ORDER BY ts_ms DESC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def regime_tag(conn: sqlite3.Connection, row: dict[str, Any]) -> dict[str, Any]:
    ts = int(row["signal_ts_ms"])
    eth_1h = ret_bps(conn, "ETHUSDT", ts, 3600)
    eth_4h = ret_bps(conn, "ETHUSDT", ts, 4 * 3600)
    btc_1h = ret_bps(conn, "BTCUSDT", ts, 3600)
    btc_4h = ret_bps(conn, "BTCUSDT", ts, 4 * 3600)
    funding = funding_at(conn, "ETHUSDT", ts)
    if eth_4h is not None and btc_4h is not None and eth_1h is not None and btc_1h is not None:
        if eth_4h >= 80.0 and btc_4h >= 50.0 and eth_1h >= 20.0 and btc_1h >= 0.0:
            tag = "BULL"
        elif eth_4h <= -80.0 or btc_4h <= -50.0:
            tag = "RISK_OFF"
        else:
            tag = "NEUTRAL"
    else:
        tag = "UNKNOWN"
    return {
        "observation_id": row.get("observation_id"),
        "signal_utc": row.get("signal_utc"),
        "tag": tag,
        "eth_1h_bps": r1(eth_1h),
        "eth_4h_bps": r1(eth_4h),
        "btc_1h_bps": r1(btc_1h),
        "btc_4h_bps": r1(btc_4h),
        "funding_rate": funding,
        "net_bps": r1(float(row["net_bps"])),
    }


def bull_regime_report(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    tagged = [regime_tag(conn, r) for r in rows]
    by_tag: dict[str, list[float]] = {}
    for row in tagged:
        by_tag.setdefault(str(row["tag"]), []).append(float(row["net_bps"]))
    return {
        "status": "TAGGED_OBSERVATION_ONLY",
        "rows": tagged,
        "by_tag": {tag: summary(vals) for tag, vals in sorted(by_tag.items())},
        "read": "Bull tags are labels only. The current v0.2 rule is not changed by these tags.",
    }


def exit_net_bps(entry: float, exit_px: float, fee_bps: float) -> float:
    return (float(exit_px) - float(entry)) / float(entry) * 10_000.0 - float(fee_bps)


def first_touch(path: list[tuple[int, float]], *, entry: float, up_bps: float | None, down_bps: float | None) -> tuple[str, int, float] | None:
    up_px = entry * (1.0 + float(up_bps) / 10_000.0) if up_bps is not None else None
    down_px = entry * (1.0 - abs(float(down_bps)) / 10_000.0) if down_bps is not None else None
    for ts, px in path:
        if down_px is not None and px <= down_px:
            return ("SL", ts, down_px)
        if up_px is not None and px >= up_px:
            return ("TP", ts, up_px)
    return None


def simulate_exit_variant(conn: sqlite3.Connection, row: dict[str, Any], variant: str) -> dict[str, Any]:
    entry = float(row["entry_price"])
    fee = float(row.get("fee_bps") or FEE_BPS_DEFAULT)
    start = int(row["maker_fill_ts_ms"])
    horizon_sec = {
        "fixed_2h": 2 * 3600,
        "fixed_4h": 4 * 3600,
        "fixed_8h": 8 * 3600,
        "sl150_2h": 2 * 3600,
        "tp300_sl150_4h": 4 * 3600,
        "trail100_after150_4h": 4 * 3600,
        "partial_tp150_2h": 2 * 3600,
    }[variant]
    path = mark_range(conn, "ETHUSDT", start, start + horizon_sec * 1000)
    if not path:
        return {"variant": variant, "status": "NO_MARK_PATH", "net_bps": None}
    fallback_ts, fallback_px = path[-1]
    reason = "TIME"
    exit_ts = fallback_ts
    exit_px = fallback_px

    if variant.startswith("fixed_"):
        pass
    elif variant == "sl150_2h":
        touch = first_touch(path, entry=entry, up_bps=None, down_bps=150.0)
        if touch:
            reason, exit_ts, exit_px = touch
    elif variant == "tp300_sl150_4h":
        touch = first_touch(path, entry=entry, up_bps=300.0, down_bps=150.0)
        if touch:
            reason, exit_ts, exit_px = touch
    elif variant == "trail100_after150_4h":
        armed = False
        peak = entry
        for ts, px in path:
            peak = max(peak, px)
            if not armed and peak >= entry * 1.015:
                armed = True
            if armed and px <= peak * 0.99:
                reason, exit_ts, exit_px = "TRAIL", ts, peak * 0.99
                break
    elif variant == "partial_tp150_2h":
        touch = first_touch(path, entry=entry, up_bps=150.0, down_bps=None)
        if touch:
            _, tp_ts, tp_px = touch
            half_tp = exit_net_bps(entry, tp_px, fee)
            half_time = exit_net_bps(entry, fallback_px, fee)
            return {
                "variant": variant,
                "status": "OK",
                "exit_reason": "PARTIAL_TP_TIME",
                "exit_ts_ms": int(max(tp_ts, fallback_ts)),
                "exit_utc": iso_ms(max(tp_ts, fallback_ts)),
                "net_bps": r1((half_tp + half_time) / 2.0),
            }
    return {
        "variant": variant,
        "status": "OK",
        "exit_reason": reason,
        "exit_ts_ms": int(exit_ts),
        "exit_utc": iso_ms(exit_ts),
        "net_bps": r1(exit_net_bps(entry, exit_px, fee)),
    }


def exit_management_report(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [
        "fixed_2h",
        "fixed_4h",
        "fixed_8h",
        "sl150_2h",
        "tp300_sl150_4h",
        "trail100_after150_4h",
        "partial_tp150_2h",
    ]
    detailed = []
    by_variant: dict[str, list[float]] = {v: [] for v in variants}
    reasons: dict[str, dict[str, int]] = {v: {} for v in variants}
    for row in rows:
        item = {"observation_id": row.get("observation_id"), "signal_utc": row.get("signal_utc"), "variants": {}}
        for variant in variants:
            sim = simulate_exit_variant(conn, row, variant)
            item["variants"][variant] = sim
            if sim.get("net_bps") is not None:
                by_variant[variant].append(float(sim["net_bps"]))
                reason = str(sim.get("exit_reason") or "UNKNOWN")
                reasons[variant][reason] = int(reasons[variant].get(reason, 0)) + 1
        detailed.append(item)
    return {
        "status": "EXIT_VARIANTS_OBSERVATION_ONLY",
        "by_variant": {
            v: {**summary(vals), "exit_reasons": reasons[v]}
            for v, vals in by_variant.items()
        },
        "best_by_sum": sorted(
            [{"variant": v, **summary(vals)} for v, vals in by_variant.items()],
            key=lambda x: float(x.get("sum_bps") or -1e9),
            reverse=True,
        ),
        "rows": detailed,
        "read": "Exit variants are simulated on the same filled entries. They are not live order logic.",
    }


def sizing_equity_report(sizing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in sizing_rows:
        by_mode.setdefault(str(row.get("risk_mode")), []).append(row)
    out = {}
    for mode, rows in sorted(by_mode.items()):
        rows.sort(key=lambda r: str(r.get("signal_utc") or ""))
        vals = [float(r["net_bps"]) for r in rows if r.get("net_bps") is not None]
        out[mode] = {
            **summary(vals),
            "notional_usdt": rows[0].get("notional_usdt") if rows else None,
            "margin_usdt": rows[0].get("margin_usdt") if rows else None,
            "leverage": rows[0].get("leverage") if rows else None,
            "sum_pnl_usdt": r3(sum(float(r.get("pnl_usdt") or 0.0) for r in rows)),
            "ending_equity_usdt": rows[-1].get("equity_after_usdt") if rows else None,
            "max_drawdown_pct_equity": r3(max(float(r.get("drawdown_pct_equity") or 0.0) for r in rows)) if rows else None,
        }
    return {
        "status": "SIZING_EQUITY_COMPARISON",
        "modes": out,
        "read": "CURRENT_ENV mirrors configured live sizing; BALANCED/SURVIVAL are system risk sizing shadows.",
    }


def fill_realism_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["net_bps"]) for r in rows]
    delay = [float(r.get("fill_delay_sec") or 0.0) for r in rows]
    exec_gain = [
        float(r["net_bps"]) - float(r.get("counterfactual_anchor_mark_net_bps") or 0.0)
        for r in rows
    ]
    stressed_10 = [v - 10.0 for v in vals]
    stressed_20 = [v - 20.0 for v in vals]
    by_leg: dict[str, list[float]] = {}
    for row in rows:
        by_leg.setdefault(str(row.get("fill_leg") or "UNKNOWN"), []).append(float(row["net_bps"]))
    return {
        "status": "FILL_QUEUE_PROXY_DIAGNOSTIC",
        "fill_delay_sec": {
            "n": len(delay),
            "median": r1(median(delay)) if delay else None,
            "max": r1(max(delay)) if delay else None,
            "over_300s_n": sum(1 for d in delay if d > 300.0),
            "over_900s_n": sum(1 for d in delay if d > 900.0),
        },
        "by_fill_leg": {leg: summary(v) for leg, v in sorted(by_leg.items())},
        "maker_vs_anchor_counterfactual_gain_bps": summary(exec_gain),
        "queue_penalty_stress": {
            "minus_10bps": summary(stressed_10),
            "minus_20bps": summary(stressed_20),
        },
        "read": "Queue stress is a proxy, not proof. Real exchange queue position and actual fee tier remain binding.",
    }


def decision_gate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "NO_ROWS"}
    start = min(int(r["signal_ts_ms"]) for r in rows)
    end = max(int(r["signal_ts_ms"]) for r in rows)
    days = max(1.0, (end - start) / 86_400_000.0)
    vals = [float(r["net_bps"]) for r in rows]
    independent_weeks = len(
        {
            datetime.fromtimestamp(int(r["signal_ts_ms"]) / 1000.0, tz=timezone.utc).isocalendar()[:2]
            for r in rows
        }
    )
    status = "WAIT_FORWARD_OOS"
    reasons = []
    if days < 30 or len(rows) < 10:
        reasons.append("30D_INTERIM_NOT_MET")
    if days < 60 or len(rows) < 20 or independent_weeks < 2:
        reasons.append("60D_DECISION_NOT_MET")
    if sum(vals) <= 0:
        reasons.append("SUM_NOT_POSITIVE")
    if summary(vals)["t3r_bps"] <= 0:
        reasons.append("T3R_NOT_POSITIVE")
    if not reasons:
        status = "REVIEW_READY_NOT_LIVE_AUTHORIZATION"
    return {
        "status": status,
        "calendar_span_days": r1(days),
        "closed_n": len(rows),
        "independent_weeks": independent_weeks,
        "summary": summary(vals),
        "reasons": reasons,
        "policy": {
            "30d_interim": ">=30 calendar days AND >=10 closed forward fills",
            "60d_decision": ">=60 calendar days AND >=20 closed forward fills across >=2 UTC weeks",
            "promotion": "requires positive sum and positive T3R; still operator decision only",
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    mirror_rows = closed_fills(load_jsonl(args.shadow_ledger))
    sizing_rows = load_jsonl(args.sizing_ledger)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        bull = bull_regime_report(conn, mirror_rows)
        exits = exit_management_report(conn, mirror_rows)
    sizing = sizing_equity_report(sizing_rows)
    fill = fill_realism_report(mirror_rows)
    gate = decision_gate_report(mirror_rows)
    return {
        "generated_at_utc": utc_now(),
        "status": STATUS,
        "rule_id": RULE_ID,
        "source_shadow_rows": len(mirror_rows),
        "bull_regime": bull,
        "exit_management": exits,
        "sizing_equity": sizing,
        "fill_realism": fill,
        "decision_gate": gate,
        "read": "Research only. No live executor, order logic, leverage, size, or .env changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Forward Research Pack",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## Bull Regime Tags",
        "",
        "| Tag | N | Sum bps | Median | Win | T3R | Max loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for tag, row in (report["bull_regime"].get("by_tag") or {}).items():
        lines.append(
            f"| {tag} | {row.get('n')} | {row.get('sum_bps')} | {row.get('median_bps')} | "
            f"{row.get('win_rate')} | {row.get('t3r_bps')} | {row.get('max_loss_bps')} |"
        )
    lines.extend(
        [
            "",
            "## Exit Management Sweep",
            "",
            "| Variant | N | Sum bps | Median | Win | T3R | Max loss | Exit reasons |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["exit_management"]["best_by_sum"]:
        lines.append(
            f"| {row.get('variant')} | {row.get('n')} | {row.get('sum_bps')} | {row.get('median_bps')} | "
            f"{row.get('win_rate')} | {row.get('t3r_bps')} | {row.get('max_loss_bps')} | "
            f"{(report['exit_management']['by_variant'].get(row.get('variant')) or {}).get('exit_reasons')} |"
        )
    lines.extend(
        [
            "",
            "## Sizing Equity",
            "",
            "| Mode | N | Notional | Margin | Sum bps | PnL USDT | End equity | Max DD % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, row in (report["sizing_equity"].get("modes") or {}).items():
        lines.append(
            f"| {mode} | {row.get('n')} | {row.get('notional_usdt')} | {row.get('margin_usdt')} | "
            f"{row.get('sum_bps')} | {row.get('sum_pnl_usdt')} | {row.get('ending_equity_usdt')} | "
            f"{row.get('max_drawdown_pct_equity')} |"
        )
    fill = report["fill_realism"]
    lines.extend(
        [
            "",
            "## Fill / Queue Realism",
            "",
            f"- Fill delay: median `{fill['fill_delay_sec']['median']}` sec, max `{fill['fill_delay_sec']['max']}` sec, >300s `{fill['fill_delay_sec']['over_300s_n']}`, >900s `{fill['fill_delay_sec']['over_900s_n']}`.",
            f"- Maker vs anchor counterfactual gain: N={fill['maker_vs_anchor_counterfactual_gain_bps']['n']} sum={fill['maker_vs_anchor_counterfactual_gain_bps']['sum_bps']} median={fill['maker_vs_anchor_counterfactual_gain_bps']['median_bps']} bps.",
            f"- Queue stress -10bps: sum={fill['queue_penalty_stress']['minus_10bps']['sum_bps']} median={fill['queue_penalty_stress']['minus_10bps']['median_bps']} T3R={fill['queue_penalty_stress']['minus_10bps']['t3r_bps']}.",
            f"- Queue stress -20bps: sum={fill['queue_penalty_stress']['minus_20bps']['sum_bps']} median={fill['queue_penalty_stress']['minus_20bps']['median_bps']} T3R={fill['queue_penalty_stress']['minus_20bps']['t3r_bps']}.",
            "",
            "## Decision Gate",
            "",
        ]
    )
    gate = report["decision_gate"]
    lines.append(
        f"- Status: `{gate.get('status')}`; span `{gate.get('calendar_span_days')}` days; "
        f"N `{gate.get('closed_n')}`; weeks `{gate.get('independent_weeks')}`; reasons `{gate.get('reasons')}`."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V Engine forward research pack.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--shadow-ledger", type=Path, default=MIRROR_LEDGER)
    p.add_argument("--sizing-ledger", type=Path, default=SIZING_LEDGER)
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
