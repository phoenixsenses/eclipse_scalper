"""S34 v0.2 momentum-arming retest/pullback tests.

Research-only. Tests whether the post-arming short-term weakness can be used as
a better passive fill point for the frozen v0.2 ETH SELL maker LONG rule.

Important distinction:
- Base signal under test: S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID
- Scalp horizons are reported only as diagnostics. This is not the rejected
  stress-scalp candidate.
"""

from __future__ import annotations

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

from tools.s34_navigation_full_followup import DEFAULT_DB, mark_at_or_after, summary  # noqa: E402
from tools.s34_stress_reaction_deep_tests import mark_series  # noqa: E402
from tools.s34_v02_momentum_arming_live_like import (  # noqa: E402
    ARMING_CONFIGS,
    FEE_BPS,
    HORIZONS,
    SYMBOL,
    build_v02_events,
    scan_first_arming,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V02_ARMING_RETEST_PULLBACK.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V02_ARMING_RETEST_PULLBACK.md"

CONFIG_NAMES = ("FLOW_POSITIVE_ONLY", "ARM_BASE", "QUIET_ETH_BTC_UP")
PULLBACK_BPS = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0)
WAIT_WINDOWS_SEC = (30, 60, 90, 180)
CROSS_MARGIN_BPS = 1.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def iso_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def r1(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 1)


def r3(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 3)


def mark_net_long(conn: sqlite3.Connection, start_ms: int, entry_px: float, horizon_sec: int) -> float | None:
    exit_ = mark_at_or_after(conn, SYMBOL, int(start_ms) + int(horizon_sec) * 1000)
    if not exit_ or float(entry_px) <= 0:
        return None
    return (float(exit_[1]) - float(entry_px)) / float(entry_px) * 10_000.0 - FEE_BPS


def find_pullback_fill(
    conn: sqlite3.Connection,
    *,
    arming_ts_ms: int,
    arming_px: float,
    pullback_bps: float,
    wait_sec: int,
) -> dict[str, Any]:
    limit_px = float(arming_px) * (1.0 - float(pullback_bps) / 10_000.0)
    required_px = limit_px * (1.0 - CROSS_MARGIN_BPS / 10_000.0)
    series = mark_series(conn, int(arming_ts_ms), int(arming_ts_ms) + int(wait_sec) * 1000)
    min_px = None
    min_ts = None
    for ts_ms, px in series:
        px = float(px)
        if min_px is None or px < min_px:
            min_px = px
            min_ts = int(ts_ms)
        if px <= required_px:
            return {
                "status": "FILLED",
                "fill_ts_ms": int(ts_ms),
                "fill_utc": iso_ms(int(ts_ms)),
                "fill_delay_from_arm_sec": r1((int(ts_ms) - int(arming_ts_ms)) / 1000.0),
                "entry_px": limit_px,
                "limit_px": limit_px,
                "required_px": required_px,
                "min_px_in_wait": min_px,
                "min_ts_ms": min_ts,
            }
    return {
        "status": "NO_FILL",
        "fill_ts_ms": None,
        "fill_utc": None,
        "fill_delay_from_arm_sec": None,
        "entry_px": None,
        "limit_px": limit_px,
        "required_px": required_px,
        "min_px_in_wait": min_px,
        "min_ts_ms": min_ts,
    }


def chronological_split(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [(int(r["anchor_ts_ms"]), float(r[key])) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    vals.sort(key=lambda x: x[0])
    if not vals:
        return {"cal": summary([]), "hold": summary([])}
    cut = max(1, int(len(vals) * 0.6))
    return {
        "cal": summary([v for _, v in vals[:cut]]),
        "hold": summary([v for _, v in vals[cut:]]),
    }


def summarize_rows(rows: list[dict[str, Any]], horizon_key: str = "net_2h_bps") -> dict[str, Any]:
    filled = [r for r in rows if r.get("status") == "FILLED"]
    vals = [float(r[horizon_key]) for r in filled if r.get(horizon_key) is not None]
    missed_anchor = [float(r["anchor_to_2h_net_bps"]) for r in rows if r.get("status") != "FILLED" and r.get("anchor_to_2h_net_bps") is not None]
    return {
        "signals": len(rows),
        "filled": len(filled),
        "fill_rate": r3(len(filled) / len(rows)) if rows else None,
        "fill_delay_median_sec": r1(median([float(r["fill_delay_from_arm_sec"]) for r in filled])) if filled else None,
        "filled_summary": summary(vals),
        "filled_split": chronological_split(filled, horizon_key),
        "missed_n": len(rows) - len(filled),
        "missed_anchor_2h": summary(missed_anchor),
    }


def run() -> dict[str, Any]:
    cfg_by_name = {cfg.name: cfg for cfg in ARMING_CONFIGS}
    result_rows: dict[str, list[dict[str, Any]]] = {}
    event_cards: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        events = build_v02_events(conn)
        for cfg_name in CONFIG_NAMES:
            cfg = cfg_by_name[cfg_name]
            for wait_sec in WAIT_WINDOWS_SEC:
                for pullback_bps in PULLBACK_BPS:
                    key = f"{cfg_name}_PB{pullback_bps:g}_W{wait_sec}"
                    result_rows[key] = []
            for event in events:
                anchor_ts = int(event.anchor.anchor_ts_ms)
                anchor_entry = mark_at_or_after(conn, SYMBOL, anchor_ts)
                anchor_to_2h = None
                if anchor_entry and float(anchor_entry[1]) > 0:
                    anchor_to_2h = mark_net_long(conn, anchor_ts, float(anchor_entry[1]), HORIZONS["2h"])
                arm = scan_first_arming(conn, anchor_ts, cfg)
                if arm is None:
                    for wait_sec in WAIT_WINDOWS_SEC:
                        for pullback_bps in PULLBACK_BPS:
                            key = f"{cfg_name}_PB{pullback_bps:g}_W{wait_sec}"
                            result_rows[key].append(
                                {
                                    "status": "NO_ARM",
                                    "anchor_ts_ms": anchor_ts,
                                    "anchor_utc": iso_ms(anchor_ts),
                                    "anchor_to_2h_net_bps": r1(anchor_to_2h),
                                }
                            )
                    continue
                arm_ts = int(arm["ts_ms"])
                arm_entry = mark_at_or_after(conn, SYMBOL, arm_ts)
                if not arm_entry:
                    continue
                arm_px = float(arm_entry[1])
                card_base = {
                    "event_id": f"V02:{event.anchor.bucket}:{anchor_ts}",
                    "anchor_ts_ms": anchor_ts,
                    "anchor_utc": iso_ms(anchor_ts),
                    "config": cfg_name,
                    "arming_delay_sec": int(arm["delay_sec"]),
                    "arming_px": arm_px,
                    "anchor_to_2h_net_bps": r1(anchor_to_2h),
                    "arm_to_2h_net_bps": r1(mark_net_long(conn, arm_ts, arm_px, HORIZONS["2h"])),
                }
                for wait_sec in WAIT_WINDOWS_SEC:
                    for pullback_bps in PULLBACK_BPS:
                        key = f"{cfg_name}_PB{pullback_bps:g}_W{wait_sec}"
                        fill = find_pullback_fill(
                            conn,
                            arming_ts_ms=arm_ts,
                            arming_px=arm_px,
                            pullback_bps=pullback_bps,
                            wait_sec=wait_sec,
                        )
                        row = {
                            **card_base,
                            "pullback_bps": float(pullback_bps),
                            "wait_sec": int(wait_sec),
                            **fill,
                        }
                        if fill["status"] == "FILLED":
                            fill_ts = int(fill["fill_ts_ms"])
                            entry_px = float(fill["entry_px"])
                            for label, sec in HORIZONS.items():
                                row[f"net_{label}_bps"] = r1(mark_net_long(conn, fill_ts, entry_px, sec))
                            row["entry_improvement_vs_arm_bps"] = r1((arm_px - entry_px) / arm_px * 10_000.0)
                            row["entry_deterioration_vs_anchor_bps"] = (
                                r1((entry_px - float(anchor_entry[1])) / float(anchor_entry[1]) * 10_000.0)
                                if anchor_entry and float(anchor_entry[1]) > 0
                                else None
                            )
                        result_rows[key].append(row)
                        if cfg_name == "FLOW_POSITIVE_ONLY" and wait_sec == 60 and pullback_bps in {0.0, 5.0, 10.0}:
                            event_cards.append(row)

    summaries = {
        key: {
            "key": key,
            "summary_30s": summarize_rows(rows, "net_30s_bps"),
            "summary_60s": summarize_rows(rows, "net_60s_bps"),
            "summary_5m": summarize_rows(rows, "net_5m_bps"),
            "summary_15m": summarize_rows(rows, "net_15m_bps"),
            "summary_2h": summarize_rows(rows, "net_2h_bps"),
        }
        for key, rows in result_rows.items()
    }
    ranked_2h = sorted(
        summaries.values(),
        key=lambda r: (
            float(r["summary_2h"]["filled_summary"].get("t3r_bps") or -1e18),
            float(r["summary_2h"]["filled_summary"].get("sum_bps") or -1e18),
            float(r["summary_2h"].get("fill_rate") or 0.0),
        ),
        reverse=True,
    )
    ranked_scalp = sorted(
        summaries.values(),
        key=lambda r: (
            float(r["summary_60s"]["filled_summary"].get("t3r_bps") or -1e18),
            float(r["summary_60s"]["filled_summary"].get("sum_bps") or -1e18),
        ),
        reverse=True,
    )
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "base_rule": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
        "note": "Scalp horizons are diagnostics on the v0.2 base signal, not the rejected stress-scalp candidate.",
        "config_names": list(CONFIG_NAMES),
        "pullback_bps": list(PULLBACK_BPS),
        "wait_windows_sec": list(WAIT_WINDOWS_SEC),
        "summaries": summaries,
        "ranked_2h": ranked_2h[:20],
        "ranked_60s_scalp": ranked_scalp[:20],
        "event_cards": event_cards,
    }


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"WR={s.get('win_rate')} T3R={s.get('t3r_bps')} maxLoss={s.get('max_loss_bps')}"
    )


def row_cell(block: dict[str, Any]) -> str:
    return (
        f"sig={block['signals']} fill={block['filled']} rate={block['fill_rate']} "
        f"{fmt(block['filled_summary'])} missed={fmt(block['missed_anchor_2h'])}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 v0.2 Arming Retest/Pullback Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Base rule: `{result['base_rule']}`",
        "",
        result["note"],
        "",
        "## 1. Best 2h Pullback Variants",
        "",
        "| Rank | Variant | 2h result | Cal 2h | Hold 2h | 60s scalp | Missed anchor 2h |",
        "| ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, row in enumerate(result["ranked_2h"][:15], start=1):
        s2h = row["summary_2h"]
        s60 = row["summary_60s"]
        lines.append(
            f"| {idx} | `{row['key']}` | {row_cell(s2h)} | {fmt(s2h['filled_split']['cal'])} | "
            f"{fmt(s2h['filled_split']['hold'])} | {fmt(s60['filled_summary'])} | {fmt(s2h['missed_anchor_2h'])} |"
        )

    lines.extend(["", "## 2. Best 60s Scalp Diagnostics", ""])
    lines.append("| Rank | Variant | 60s result | 2h result |")
    lines.append("| ---: | --- | --- | --- |")
    for idx, row in enumerate(result["ranked_60s_scalp"][:15], start=1):
        lines.append(
            f"| {idx} | `{row['key']}` | {row_cell(row['summary_60s'])} | {row_cell(row['summary_2h'])} |"
        )

    lines.extend(["", "## 3. FLOW_POSITIVE_ONLY W60 Event Cards", ""])
    for row in result["event_cards"]:
        compact = {
            k: row.get(k)
            for k in [
                "event_id",
                "anchor_utc",
                "pullback_bps",
                "status",
                "fill_delay_from_arm_sec",
                "entry_improvement_vs_arm_bps",
                "entry_deterioration_vs_anchor_bps",
                "anchor_to_2h_net_bps",
                "arm_to_2h_net_bps",
                "net_60s_bps",
                "net_15m_bps",
                "net_2h_bps",
            ]
        }
        lines.append(f"- `{compact}`")

    lines.extend(
        [
            "",
            "## 4. Interpretation Rules",
            "",
            "- A useful pullback variant must improve 2h T3R without simply missing the winners.",
            "- Positive 30s/60s results here would be a v0.2-management scalp diagnostic only, not validation of the separate stress-scalp candidate.",
            "- N remains 11; this is navigation/management research, not live order-logic evidence.",
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
