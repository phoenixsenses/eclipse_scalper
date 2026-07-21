"""S34 v0.2 live-like momentum arming tests.

Research-only. Converts the prior hindsight momentum-onset precursor into a
causal scan: after each frozen v0.2 anchor, scan forward for the first time a
knowable "momentum arming" condition appears, then evaluate outcomes from that
timestamp and maker lifecycle alternatives.

This script writes reports only. It does not touch the live executor, order
logic, size, leverage, config, runtime state, or environment files.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import book_at, load_mark_index, r1, r3, signed_return_bps  # noqa: E402
from tools.research_s34_maker_fade import maker_limit_price  # noqa: E402
from tools.research_s34_wave_absorption import book_features_at  # noqa: E402
from tools.s34_navigation_full_followup import DEFAULT_DB, mark_at_or_after, summary  # noqa: E402
from tools.s34_v_engine_cancel_replace import find_fill_between, simulate_cancel_replace  # noqa: E402
from tools.s34_v_engine_execution_frontier import collect_v01_events, prior_return_bps  # noqa: E402
from tools.s34_v_engine_shadow_observer import HORIZON_SEC, PRIOR4H_LT_BPS, SYMBOL  # noqa: E402
from tools.s34_v_engine_v02_shadow_mirror import (  # noqa: E402
    CROSS_MARGIN_BPS,
    INITIAL_OFFSET_BPS,
    MIN_BID_DEPTH_USD,
    REPLACE_OFFSET_BPS,
    WAIT_SEC,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V02_MOMENTUM_ARMING_LIVE_LIKE.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V02_MOMENTUM_ARMING_LIVE_LIKE.md"

BTC = "BTCUSDT"
FEE_BPS = 5.0
SCAN_MAX_SEC = 600
SCAN_STEP_SEC = 1
WINDOW_SEC = 5
MAKER_FEE_BPS = -0.5
TAKER_FEE_BPS = 5.0
MAX_BOOK_STALENESS_SEC = 10
HORIZONS = {
    "30s": 30,
    "60s": 60,
    "5m": 300,
    "15m": 900,
    "2h": 7200,
}


@dataclass(frozen=True)
class ArmingConfig:
    name: str
    eth_min_bps: float = 0.0
    btc_min_bps: float = 0.0
    max_sell_liq_usd: float = 1.0
    min_taker_imbalance: float | None = 0.0
    min_taker_buy_usd: float | None = None


ARMING_CONFIGS = [
    ArmingConfig("ARM_BASE"),
    ArmingConfig("ETH_BTC_UP_ONLY", max_sell_liq_usd=1e18, min_taker_imbalance=None),
    ArmingConfig("QUIET_ETH_BTC_UP", min_taker_imbalance=None),
    ArmingConfig("FLOW_POSITIVE_ONLY", eth_min_bps=-1e18, btc_min_bps=-1e18, max_sell_liq_usd=1e18),
    ArmingConfig("ARM_FLOW_STRONG", min_taker_imbalance=0.5),
    ArmingConfig("ARM_BUY_250K", min_taker_buy_usd=250_000.0),
    ArmingConfig("ARM_BUY_500K", min_taker_buy_usd=500_000.0),
    ArmingConfig("ARM_ETHBTC_2BPS", eth_min_bps=2.0, btc_min_bps=2.0),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def safe_vals(vals: list[float | None]) -> list[float]:
    return [float(v) for v in vals if v is not None and math.isfinite(float(v))]


def t3r(vals: list[float]) -> float:
    vals = safe_vals(vals)
    return float(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else float(sum(vals))


def mark_ret_bps(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    a = mark_at_or_after(conn, symbol, int(start_ms))
    b = mark_at_or_after(conn, symbol, int(end_ms))
    if not a or not b or float(a[1]) <= 0:
        return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0


def long_net_mark(conn: sqlite3.Connection, start_ms: int, horizon_sec: int) -> float | None:
    raw = mark_ret_bps(conn, SYMBOL, int(start_ms), int(start_ms) + int(horizon_sec) * 1000)
    return None if raw is None else float(raw) - FEE_BPS


def trade_flow_5s(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT is_buyer_maker, COALESCE(SUM(notional),0.0), COUNT(*)
        FROM agg_trades
        WHERE symbol=? AND ts_ms>=? AND ts_ms<?
        GROUP BY is_buyer_maker
        """,
        (SYMBOL, int(start_ms), int(end_ms)),
    ).fetchall()
    taker_buy = 0.0
    taker_sell = 0.0
    count = 0
    for is_buyer_maker, notion, c in rows:
        count += int(c or 0)
        if int(is_buyer_maker) == 0:
            taker_buy += float(notion or 0.0)
        else:
            taker_sell += float(notion or 0.0)
    total = taker_buy + taker_sell
    return {
        "taker_buy_usd": taker_buy,
        "taker_sell_usd": taker_sell,
        "taker_imbalance": (taker_buy - taker_sell) / total if total > 0 else None,
        "agg_trade_count": count,
    }


def sell_liq_5s(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> float:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(notional),0.0)
        FROM liquidations
        WHERE symbol=? AND side='SELL' AND ts_ms>=? AND ts_ms<?
        """,
        (SYMBOL, int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0] or 0.0) if row else 0.0


def arming_snapshot(conn: sqlite3.Connection, ts_ms: int, window_sec: int = WINDOW_SEC) -> dict[str, Any]:
    start_ms = int(ts_ms) - int(window_sec) * 1000
    flow = trade_flow_5s(conn, start_ms, int(ts_ms))
    return {
        "ts_ms": int(ts_ms),
        "utc": iso_ms(int(ts_ms)),
        "eth_ret_bps": mark_ret_bps(conn, SYMBOL, start_ms, int(ts_ms)),
        "btc_ret_bps": mark_ret_bps(conn, BTC, start_ms, int(ts_ms)),
        "sell_liq_usd": sell_liq_5s(conn, start_ms, int(ts_ms)),
        **flow,
    }


def passes_config(snap: dict[str, Any], cfg: ArmingConfig) -> bool:
    eth = snap.get("eth_ret_bps")
    btc = snap.get("btc_ret_bps")
    if eth is None or btc is None:
        return False
    if float(eth) <= float(cfg.eth_min_bps):
        return False
    if float(btc) <= float(cfg.btc_min_bps):
        return False
    if float(snap.get("sell_liq_usd") or 0.0) > float(cfg.max_sell_liq_usd):
        return False
    imb = snap.get("taker_imbalance")
    if cfg.min_taker_imbalance is not None:
        if imb is None or float(imb) <= float(cfg.min_taker_imbalance):
            return False
    if cfg.min_taker_buy_usd is not None:
        if float(snap.get("taker_buy_usd") or 0.0) < float(cfg.min_taker_buy_usd):
            return False
    return True


def scan_first_arming(conn: sqlite3.Connection, anchor_ms: int, cfg: ArmingConfig, max_sec: int = SCAN_MAX_SEC) -> dict[str, Any] | None:
    start_sec = WINDOW_SEC
    for sec in range(start_sec, int(max_sec) + 1, SCAN_STEP_SEC):
        ts_ms = int(anchor_ms) + sec * 1000
        snap = arming_snapshot(conn, ts_ms)
        if passes_config(snap, cfg):
            snap["delay_sec"] = sec
            snap["config"] = cfg.name
            return snap
    return None


def build_v02_events(conn: sqlite3.Connection) -> list[Any]:
    marks = load_mark_index(conn, SYMBOL)
    out = []
    for event in collect_v01_events(conn):
        anchor_ms = int(event.anchor.anchor_ts_ms)
        prior4h = prior_return_bps(marks, anchor_ms, 4 * 3600)
        if prior4h is None or not math.isfinite(float(prior4h)) or not (float(prior4h) < PRIOR4H_LT_BPS):
            continue
        book = book_features_at(conn, SYMBOL, anchor_ms, MAX_BOOK_STALENESS_SEC)
        if not book or float(book.get("bid_depth_usd") or 0.0) < MIN_BID_DEPTH_USD:
            continue
        out.append(event)
    out.sort(key=lambda e: int(e.anchor.anchor_ts_ms))
    return out


def exit_from_fill(
    conn: sqlite3.Connection,
    event: Any,
    *,
    fill_ts_ms: int,
    entry_px: float,
    maker_fee_bps: float = MAKER_FEE_BPS,
    taker_fee_bps: float = TAKER_FEE_BPS,
) -> dict[str, Any]:
    exit_ts_ms = int(fill_ts_ms) + HORIZON_SEC * 1000
    exit_book = book_at(conn, event.symbol, exit_ts_ms, MAX_BOOK_STALENESS_SEC)
    if not exit_book:
        return {"status": "NO_EXIT_BOOK", "net_bps": None}
    exit_px = float(exit_book.bid if event.fade_direction == "LONG" else exit_book.ask)
    gross = signed_return_bps(event.fade_direction, float(entry_px), exit_px)
    net = gross - float(maker_fee_bps) - float(taker_fee_bps)
    return {
        "status": "FILLED",
        "maker_fill_ts_ms": int(fill_ts_ms),
        "maker_fill_utc": iso_ms(int(fill_ts_ms)),
        "fill_delay_sec": r1((int(fill_ts_ms) - int(event.anchor_mark_ts_ms)) / 1000.0),
        "entry_price": float(entry_px),
        "exit_ts_ms": int(exit_ts_ms),
        "exit_utc": iso_ms(int(exit_ts_ms)),
        "exit_price": exit_px,
        "gross_bps": r1(gross),
        "net_bps": float(net),
        "fee_bps": r1(float(maker_fee_bps) + float(taker_fee_bps)),
    }


def simulate_replace_at(
    conn: sqlite3.Connection,
    event: Any,
    *,
    replace_ts_ms: int | None,
    replace_offset_bps: float | None,
) -> dict[str, Any]:
    """Simulate O20 until replace_ts, then replacement offset or cancel."""
    anchor_ts = int(event.anchor_mark_ts_ms)
    stop_ts = int(replace_ts_ms) if replace_ts_ms is not None else anchor_ts + WAIT_SEC * 1000
    initial_limit = maker_limit_price(event.anchor_mark_price, event.fade_direction, INITIAL_OFFSET_BPS)
    initial_fill = find_fill_between(
        event,
        limit_px=initial_limit,
        cross_margin_bps=CROSS_MARGIN_BPS,
        start_ts_ms=anchor_ts,
        end_ts_ms=stop_ts,
    )
    base: dict[str, Any] = {
        "status": "NO_MAKER_FILL",
        "fill_leg": None,
        "replace_ts_ms": stop_ts,
        "replace_delay_sec": r1((stop_ts - anchor_ts) / 1000.0),
        "initial_limit_price": float(initial_limit),
        "replace_offset_bps": replace_offset_bps,
        "replace_limit_price": None,
        "net_bps": None,
    }
    if initial_fill is not None:
        fill_ts, entry_px = initial_fill
        base["fill_leg"] = "initial"
        base.update(exit_from_fill(conn, event, fill_ts_ms=fill_ts, entry_px=entry_px))
        return base
    if replace_offset_bps is None:
        base["status"] = "CANCELLED_UNFILLED"
        return base
    replace_limit = maker_limit_price(event.anchor_mark_price, event.fade_direction, float(replace_offset_bps))
    base["replace_limit_price"] = float(replace_limit)
    replacement_fill = find_fill_between(
        event,
        limit_px=replace_limit,
        cross_margin_bps=CROSS_MARGIN_BPS,
        start_ts_ms=stop_ts,
        end_ts_ms=None,
    )
    if replacement_fill is None:
        return base
    fill_ts, entry_px = replacement_fill
    base["fill_leg"] = "replacement"
    base.update(exit_from_fill(conn, event, fill_ts_ms=fill_ts, entry_px=entry_px))
    return base


def summarize_sim(rows: list[dict[str, Any]]) -> dict[str, Any]:
    filled = [r for r in rows if r.get("status") == "FILLED" and r.get("net_bps") is not None]
    vals = [float(r["net_bps"]) for r in filled]
    return {
        "signals": len(rows),
        "filled": len(filled),
        "fill_rate": r3(len(filled) / len(rows)) if rows else None,
        "initial_filled": sum(1 for r in filled if r.get("fill_leg") == "initial"),
        "replacement_filled": sum(1 for r in filled if r.get("fill_leg") == "replacement"),
        "summary": summary(vals),
    }


def split_summary(vals: list[tuple[int, float]]) -> dict[str, Any]:
    vals = sorted([(int(t), float(v)) for t, v in vals if math.isfinite(float(v))], key=lambda x: x[0])
    if not vals:
        return {"all": summary([]), "cal": summary([]), "hold": summary([])}
    cut = max(1, int(len(vals) * 0.6))
    return {
        "all": summary([v for _, v in vals]),
        "cal": summary([v for _, v in vals[:cut]]),
        "hold": summary([v for _, v in vals[cut:]]),
    }


def run() -> dict[str, Any]:
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        events = build_v02_events(conn)
        event_rows: list[dict[str, Any]] = []
        config_rows: dict[str, list[dict[str, Any]]] = {cfg.name: [] for cfg in ARMING_CONFIGS}
        lifecycle_rows: dict[str, list[dict[str, Any]]] = {
            "CURRENT_O20_W300_O5": [],
            "ARM_BASE_O5_ELSE_CURRENT": [],
            "ARM_BASE_O0_ELSE_CURRENT": [],
            "ARM_BASE_O5_ELSE_CANCEL300": [],
            "ARM_BASE_CANCEL_IF_NOARM300": [],
        }

        for event in events:
            anchor_ms = int(event.anchor.anchor_ts_ms)
            anchor_row: dict[str, Any] = {
                "event_id": f"V02:{event.anchor.bucket}:{anchor_ms}",
                "anchor_ts_ms": anchor_ms,
                "anchor_utc": iso_ms(anchor_ms),
                "anchor_price": float(event.anchor_mark_price),
                "vdepth_bps": r1(float(event.vdepth_bps)),
                "anchor_to_2h_net_bps": r1(long_net_mark(conn, anchor_ms, HORIZONS["2h"])),
            }
            arm_by_cfg: dict[str, dict[str, Any] | None] = {}
            for cfg in ARMING_CONFIGS:
                arm = scan_first_arming(conn, anchor_ms, cfg)
                arm_by_cfg[cfg.name] = arm
                row = {**anchor_row, "config": cfg.name, "armed": arm is not None}
                if arm is not None:
                    row.update(
                        {
                            "arming_ts_ms": int(arm["ts_ms"]),
                            "arming_utc": arm["utc"],
                            "arming_delay_sec": int(arm["delay_sec"]),
                            "eth_5s_bps": r1(arm.get("eth_ret_bps")),
                            "btc_5s_bps": r1(arm.get("btc_ret_bps")),
                            "sell_liq_5s_usd": r1(arm.get("sell_liq_usd")),
                            "taker_buy_5s_usd": r1(arm.get("taker_buy_usd")),
                            "taker_imbalance_5s": r3(arm.get("taker_imbalance")),
                            "anchor_to_arm_gross_bps": r1(mark_ret_bps(conn, SYMBOL, anchor_ms, int(arm["ts_ms"]))),
                        }
                    )
                    for label, sec in HORIZONS.items():
                        row[f"arm_to_{label}_net_bps"] = r1(long_net_mark(conn, int(arm["ts_ms"]), sec))
                config_rows[cfg.name].append(row)
            anchor_row["arm_base_delay_sec"] = (
                int(arm_by_cfg["ARM_BASE"]["delay_sec"]) if arm_by_cfg.get("ARM_BASE") else None
            )
            event_rows.append(anchor_row)

            current = simulate_cancel_replace(
                conn,
                event,
                initial_offset_bps=INITIAL_OFFSET_BPS,
                replace_offset_bps=REPLACE_OFFSET_BPS,
                wait_sec=WAIT_SEC,
                cross_margin_bps=CROSS_MARGIN_BPS,
                maker_fee_bps=MAKER_FEE_BPS,
                taker_fee_bps=TAKER_FEE_BPS,
                max_book_staleness_sec=MAX_BOOK_STALENESS_SEC,
            )
            lifecycle_rows["CURRENT_O20_W300_O5"].append(current)
            arm_base = arm_by_cfg.get("ARM_BASE")
            arm_ts = int(arm_base["ts_ms"]) if arm_base else None
            current_replace_ts = anchor_ms + WAIT_SEC * 1000
            early_ts = min(arm_ts, current_replace_ts) if arm_ts is not None else current_replace_ts
            lifecycle_rows["ARM_BASE_O5_ELSE_CURRENT"].append(
                simulate_replace_at(conn, event, replace_ts_ms=early_ts, replace_offset_bps=5.0)
            )
            lifecycle_rows["ARM_BASE_O0_ELSE_CURRENT"].append(
                simulate_replace_at(conn, event, replace_ts_ms=early_ts, replace_offset_bps=0.0)
            )
            if arm_ts is not None and arm_ts <= current_replace_ts:
                lifecycle_rows["ARM_BASE_O5_ELSE_CANCEL300"].append(
                    simulate_replace_at(conn, event, replace_ts_ms=arm_ts, replace_offset_bps=5.0)
                )
                lifecycle_rows["ARM_BASE_CANCEL_IF_NOARM300"].append(
                    simulate_replace_at(conn, event, replace_ts_ms=arm_ts, replace_offset_bps=5.0)
                )
            else:
                lifecycle_rows["ARM_BASE_O5_ELSE_CANCEL300"].append(
                    simulate_replace_at(conn, event, replace_ts_ms=current_replace_ts, replace_offset_bps=None)
                )
                lifecycle_rows["ARM_BASE_CANCEL_IF_NOARM300"].append(
                    simulate_replace_at(conn, event, replace_ts_ms=current_replace_ts, replace_offset_bps=None)
                )

    config_summary: dict[str, Any] = {}
    for cfg in ARMING_CONFIGS:
        rows = config_rows[cfg.name]
        armed = [r for r in rows if r.get("armed")]
        noarm = [r for r in rows if not r.get("armed")]
        horizon_summary = {}
        for label in HORIZONS:
            pairs = [
                (int(r["anchor_ts_ms"]), float(r[f"arm_to_{label}_net_bps"]))
                for r in armed
                if r.get(f"arm_to_{label}_net_bps") is not None
            ]
            horizon_summary[label] = split_summary(pairs)
        config_summary[cfg.name] = {
            "armed_n": len(armed),
            "noarm_n": len(noarm),
            "arm_rate": r3(len(armed) / len(rows)) if rows else None,
            "delay_median_sec": r1(median([float(r["arming_delay_sec"]) for r in armed])) if armed else None,
            "delay_p25_sec": r1(sorted([float(r["arming_delay_sec"]) for r in armed])[int(0.25 * (len(armed) - 1))]) if armed else None,
            "delay_p75_sec": r1(sorted([float(r["arming_delay_sec"]) for r in armed])[int(0.75 * (len(armed) - 1))]) if armed else None,
            "anchor_to_arm_gross_bps": summary([float(r["anchor_to_arm_gross_bps"]) for r in armed if r.get("anchor_to_arm_gross_bps") is not None]),
            "armed_anchor_2h": summary([float(r["anchor_to_2h_net_bps"]) for r in armed if r.get("anchor_to_2h_net_bps") is not None]),
            "noarm_anchor_2h": summary([float(r["anchor_to_2h_net_bps"]) for r in noarm if r.get("anchor_to_2h_net_bps") is not None]),
            "from_arming_horizons": horizon_summary,
        }

    lifecycle_summary = {name: summarize_sim(rows) for name, rows in lifecycle_rows.items()}

    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "rule": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
        "event_n": len(event_rows),
        "anchor_2h_summary": summary([float(r["anchor_to_2h_net_bps"]) for r in event_rows if r.get("anchor_to_2h_net_bps") is not None]),
        "arming_config_summary": config_summary,
        "lifecycle_summary": lifecycle_summary,
        "config_event_rows": config_rows,
        "lifecycle_rows": lifecycle_rows,
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"WR={s.get('win_rate')} T3R={s.get('t3r_bps')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 v0.2 Momentum Arming Live-Like Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Rule: `{result['rule']}`",
        "",
        f"Events: `{result['event_n']}`",
        "",
        f"Anchor 2h mark outcome: {fmt(result['anchor_2h_summary'])}",
        "",
        "## 1. Causal Arming Screens",
        "",
        "| Config | Armed | No-arm | Delay med | Anchor->Arm | Armed anchor 2h | No-arm anchor 2h | Arm->60s | Arm->15m | Arm->2h |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in result["arming_config_summary"].items():
        h = row["from_arming_horizons"]
        lines.append(
            f"| `{name}` | {row['armed_n']} | {row['noarm_n']} | {row['delay_median_sec']} | "
            f"{fmt(row['anchor_to_arm_gross_bps'])} | {fmt(row['armed_anchor_2h'])} | {fmt(row['noarm_anchor_2h'])} | "
            f"{fmt(h['60s']['all'])} | {fmt(h['15m']['all'])} | {fmt(h['2h']['all'])} |"
        )

    lines.extend(
        [
            "",
            "## 2. Chronological Split From Arming",
            "",
            "60/40 chronological split. This is small-N only, but catches obvious regime concentration.",
            "",
        ]
    )
    for name, row in result["arming_config_summary"].items():
        lines.append(f"### `{name}`")
        lines.append("| Horizon | Cal | Hold |")
        lines.append("| --- | --- | --- |")
        for horizon, split in row["from_arming_horizons"].items():
            lines.append(f"| `{horizon}` | {fmt(split['cal'])} | {fmt(split['hold'])} |")
        lines.append("")

    lines.extend(
        [
            "## 3. Maker Lifecycle Alternatives",
            "",
            "Offline maker-fill simulation using the same O20/O5 lifecycle primitives. These are not live changes.",
            "",
            "| Variant | Signals | Fill rate | Initial | Replacement | Filled summary |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, row in result["lifecycle_summary"].items():
        lines.append(
            f"| `{name}` | {row['signals']} | {row['fill_rate']} | {row['initial_filled']} | "
            f"{row['replacement_filled']} | {fmt(row['summary'])} |"
        )

    lines.extend(["", "## 4. ARM_BASE Event Rows", ""])
    for row in result["config_event_rows"]["ARM_BASE"]:
        compact = {
            k: row.get(k)
            for k in [
                "event_id",
                "anchor_utc",
                "armed",
                "arming_delay_sec",
                "eth_5s_bps",
                "btc_5s_bps",
                "sell_liq_5s_usd",
                "taker_buy_5s_usd",
                "taker_imbalance_5s",
                "anchor_to_arm_gross_bps",
                "anchor_to_2h_net_bps",
                "arm_to_60s_net_bps",
                "arm_to_15m_net_bps",
                "arm_to_2h_net_bps",
            ]
        }
        lines.append(f"- `{compact}`")

    lines.extend(
        [
            "",
            "## 5. Interpretation",
            "",
            "- This is causal/live-like: the arming timestamp is found by scanning forward from the anchor, not by labeling a future rebound onset.",
            "- Treat as navigation/management evidence only. N is still 11, so no live gating or order-logic change is justified by this report alone.",
        ]
    )

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = run()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_report(result)
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
