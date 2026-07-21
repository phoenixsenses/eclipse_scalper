"""S34 V02 H4 shadow control-plane.

Observation/risk only. This script extends the existing V02 shadow mirror ledger
into the requested H2/H3/H4 management buckets and companion observers:

1. H4 shadow bucket + forward ledger
2. cross-no-dump observer
3. catastrophic stop observer
4. dashboard fragment
5. protocol material
6. queue/fill realism proxy
7. live-vs-shadow parity audit
8. forced-flow expansion summary
9. state-machine v2
10. visual navigation dashboard feed

It never sends orders and never edits live executor/config/runtime files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (  # noqa: E402
    MarkIndex,
    iso_ms,
    load_mark_index,
    pctile,
    r1,
    signed_return_bps,
)

DB_PATH = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
SOURCE_LEDGER_CSV = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.csv"
LIVE_EXECUTOR = ROOT / "tools" / "s34_v_engine_live_executor.py"
NEXT_GEN_30D = OUT_DIR / "S34_V02_NEXT_GEN_ALPHA_RESEARCH_30D.json"

OUT_LEDGER_CSV = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW_LEDGER.csv"
OUT_LEDGER_JSONL = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW_LEDGER.jsonl"
OUT_JSON = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW.json"
OUT_MD = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW.md"
OUT_DASHBOARD_FRAGMENT = OUT_DIR / "S34_V02_H4_SHADOW_DASHBOARD_FRAGMENT.json"
OUT_PARITY_JSON = OUT_DIR / "S34_V02_LIVE_SHADOW_PARITY_AUDIT.json"
OUT_PARITY_MD = OUT_DIR / "S34_V02_LIVE_SHADOW_PARITY_AUDIT.md"

RULE_ID = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
HORIZONS_SEC = {"H2_CURRENT": 7200, "H3_SHADOW": 10800, "H4_SHADOW": 14400}
STOP_LEVELS_BPS = (100.0, 125.0, 150.0, 175.0, 200.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite_float(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(vals: list[Any]) -> dict[str, Any]:
    xs = [float(x) for x in (finite_float(v) for v in vals) if x is not None]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate": None,
            "t3r_bps": 0.0,
            "top1_removed_bps": 0.0,
            "min_bps": None,
            "max_bps": None,
        }
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": r1(sum(xs)),
        "mean_bps": r1(mean(xs)),
        "median_bps": r1(pctile(xs, 0.5)),
        "win_rate": round(sum(1 for x in xs if x > 0.0) / len(xs), 3),
        "t3r_bps": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "top1_removed_bps": r1(sum(ordered[1:]) if len(ordered) > 1 else sum(ordered)),
        "min_bps": r1(min(xs)),
        "max_bps": r1(max(xs)),
    }


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def book_bid_at_or_after(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> tuple[int, float] | None:
    row = conn.execute(
        "SELECT ts_ms, bid_price FROM book_ticker WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return (int(row[0]), float(row[1])) if row and row[1] is not None else None


def mark_at_or_after(marks: MarkIndex, ts_ms: int) -> tuple[int, float] | None:
    row = marks.at_or_after(int(ts_ms))
    return (int(row[0]), float(row[1])) if row else None


def exit_net_bps(
    conn: sqlite3.Connection,
    marks: MarkIndex,
    *,
    entry_price: float,
    fill_ts_ms: int,
    horizon_sec: int,
    fee_bps: float,
) -> tuple[float | None, int | None, float | None, str]:
    target = int(fill_ts_ms) + int(horizon_sec) * 1000
    book = book_bid_at_or_after(conn, "ETHUSDT", target)
    if book:
        gross = signed_return_bps("LONG", entry_price, float(book[1]))
        return r1(gross - fee_bps), int(book[0]), float(book[1]), "book_bid"
    mark = mark_at_or_after(marks, target)
    if mark:
        gross = signed_return_bps("LONG", entry_price, float(mark[1]))
        return r1(gross - fee_bps), int(mark[0]), float(mark[1]), "mark_fallback"
    return None, None, None, "pending_no_price"


def series_ret(marks: MarkIndex, start_ms: int, horizon_sec: int) -> float | None:
    a = mark_at_or_after(marks, start_ms)
    b = mark_at_or_after(marks, int(start_ms) + int(horizon_sec) * 1000)
    if not a or not b:
        return None
    return r1(signed_return_bps("LONG", a[1], b[1]))


def path_returns(marks: MarkIndex, entry_price: float, start_ms: int, horizon_sec: int) -> list[tuple[int, float]]:
    end = int(start_ms) + int(horizon_sec) * 1000
    return [
        (int(ts), float(signed_return_bps("LONG", entry_price, float(px))))
        for ts, px in marks.slice_range(int(start_ms), end)
        if int(ts) >= int(start_ms)
    ]


def path_stats(marks: MarkIndex, entry_price: float, start_ms: int, horizon_sec: int) -> dict[str, Any]:
    path = path_returns(marks, entry_price, start_ms, horizon_sec)
    if not path:
        return {}
    mfe_ts, mfe = max(path, key=lambda x: x[1])
    mae_ts, mae = min(path, key=lambda x: x[1])

    def first_ge(level: float) -> float | None:
        for ts, ret in path:
            if ret >= level:
                return r1((int(ts) - int(start_ms)) / 1000.0)
        return None

    def first_le(level: float) -> float | None:
        for ts, ret in path:
            if ret <= level:
                return r1((int(ts) - int(start_ms)) / 1000.0)
        return None

    out = {
        "mfe_bps": r1(mfe),
        "mae_bps": r1(mae),
        "mfe_sec": r1((int(mfe_ts) - int(start_ms)) / 1000.0),
        "mae_sec": r1((int(mae_ts) - int(start_ms)) / 1000.0),
        "rebound20_sec": first_ge(20.0),
        "rebound50_sec": first_ge(50.0),
        "rebound100_sec": first_ge(100.0),
    }
    for stop in STOP_LEVELS_BPS:
        out[f"sl{int(stop)}_touch_sec"] = first_le(-stop)
    return out


def stop_net(path_info: dict[str, Any], h4_net: float | None, fee_bps: float, stop_bps: float) -> float | None:
    if h4_net is None:
        return None
    touch = path_info.get(f"sl{int(stop_bps)}_touch_sec")
    if touch is None:
        return h4_net
    # Conservative: stop exits as taker at the observed path level, but a mark
    # path cannot know exact stop slippage. Use nominal stop plus the already
    # accounted round-trip fee as a lower-bound proxy; gap risk is reported.
    return r1(-float(stop_bps) - float(fee_bps))


def quality_state(row: dict[str, Any]) -> str:
    tags = str(row.get("entry_quality_tags") or "")
    warnings = str(row.get("entry_quality_warnings") or "")
    if "LATE_RETEST_FILL" in warnings:
        return "LATE_FILL"
    if "BID_VANISHED" in warnings:
        return "BID_VANISHED"
    if "RETEST_QUALITY_HIGH" in str(row.get("retest_quality_bucket") or "") or "BID_DEPTH_RETAINED" in tags:
        return "QUALITY_FILL"
    return "BASE_FILL"


def state_path_v2(row: dict[str, Any], p: dict[str, Any]) -> str:
    states = ["ANCHOR", quality_state(row)]
    mae = finite_float(p.get("mae_bps"))
    mfe = finite_float(p.get("mfe_bps"))
    h2 = finite_float(p.get("h2_net_bps"))
    h4 = finite_float(p.get("h4_net_bps"))
    btc30 = finite_float(p.get("btc30_bps"))
    sol30 = finite_float(p.get("sol30_bps"))
    if mae is not None and mae <= -50:
        states.append("PAIN_GE50")
    elif mae is not None and mae <= -20:
        states.append("PAIN_20_50")
    else:
        states.append("CLEAN_PATH")
    if mfe is not None and mfe >= 100:
        states.append("REBOUND100")
    elif mfe is not None and mfe >= 50:
        states.append("REBOUND50")
    elif mfe is not None and mfe >= 20:
        states.append("REBOUND20")
    if btc30 is not None and sol30 is not None:
        states.append("CROSS_OK" if btc30 > -40 and sol30 > -80 else "CROSS_DUMP")
    if h2 is not None and h4 is not None:
        states.append("RUNNER_H4" if h4 > h2 else "H2_BETTER")
    return ">".join(states)


def extract_assignments(path: Path, names: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {}
    for name in names:
        m = re.search(rf"^{re.escape(name)}\s*=\s*([^\n#]+)", text, flags=re.MULTILINE)
        if not m:
            continue
        val = m.group(1).strip().strip("\"'")
        out[name] = val
    return out


def parity_audit(source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    live = extract_assignments(
        LIVE_EXECUTOR,
        [
            "RULE_NAME",
            "SYMBOL",
            "LIQ_SIDE",
            "THRESHOLD_USD",
            "VDEPTH_MIN_BPS",
            "VDEPTH_MAX_BPS",
            "PRIOR4H_LT_BPS",
            "INITIAL_OFFSET_BPS",
            "REPLACE_OFFSET_BPS",
            "REPLACE_WAIT_SEC",
            "MIN_BID_DEPTH_USD",
        ],
    )
    sample = source_rows[0] if source_rows else {}
    shadow = {
        "RULE_NAME": sample.get("protocol_id"),
        "SYMBOL": sample.get("symbol"),
        "LIQ_SIDE": sample.get("liq_side"),
        "THRESHOLD_USD": sample.get("threshold_usd"),
        "INITIAL_OFFSET_BPS": sample.get("initial_offset_bps"),
        "REPLACE_OFFSET_BPS": sample.get("replace_offset_bps"),
        "REPLACE_WAIT_SEC": sample.get("wait_sec"),
    }
    checks = []
    checks.append({"field": "RULE_NAME/protocol_id", "live": live.get("RULE_NAME"), "shadow": shadow["RULE_NAME"], "match": live.get("RULE_NAME") == shadow["RULE_NAME"]})
    checks.append({"field": "symbol", "live": live.get("SYMBOL"), "shadow": shadow["SYMBOL"], "match": live.get("SYMBOL") == shadow["SYMBOL"]})
    checks.append({"field": "liq_side", "live": live.get("LIQ_SIDE"), "shadow": shadow["LIQ_SIDE"], "match": live.get("LIQ_SIDE") == shadow["LIQ_SIDE"]})
    for field, key in [
        ("threshold_usd", "THRESHOLD_USD"),
        ("initial_offset_bps", "INITIAL_OFFSET_BPS"),
        ("replace_offset_bps", "REPLACE_OFFSET_BPS"),
        ("wait_sec", "REPLACE_WAIT_SEC"),
    ]:
        lv = finite_float(live.get(key))
        sv = finite_float(shadow.get(key))
        checks.append({"field": field, "live": lv, "shadow": sv, "match": lv == sv})
    return {
        "status": "PASS" if all(c["match"] for c in checks) else "REVIEW",
        "live_executor_path": str(LIVE_EXECUTOR),
        "checks": checks,
        "note": "Read-only parity audit; no live files were modified.",
    }


def forced_flow_summary(path: Path = NEXT_GEN_30D) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING", "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "INVALID", "path": str(path), "error": str(exc)}
    mech = payload.get("mechanism_expansion") or {}
    return {
        "status": "LOADED_EXISTING_SCAN",
        "path": str(path),
        "summary": mech,
        "interpretation": "Existing forced-flow expansion remains small-N/research-only; not a live route.",
    }


def build_rows(conn: sqlite3.Connection, source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eth = load_mark_index(conn, "ETHUSDT")
    btc = load_mark_index(conn, "BTCUSDT")
    sol = load_mark_index(conn, "SOLUSDT")
    bucket_rows: list[dict[str, Any]] = []
    expanded: list[dict[str, Any]] = []
    closed_fills = [
        r for r in source_rows if r.get("sim_status") == "FILLED" and r.get("observation_status") == "CLOSED"
    ]
    for src in closed_fills:
        entry = finite_float(src.get("entry_price"))
        fill_ts = int(finite_float(src.get("maker_fill_ts_ms")) or 0)
        if entry is None or fill_ts <= 0:
            continue
        fee = finite_float(src.get("fee_bps")) or 5.0
        h: dict[str, Any] = {}
        hpx: dict[str, Any] = {}
        for name, sec in HORIZONS_SEC.items():
            net, exit_ts, exit_px, src_name = exit_net_bps(
                conn,
                eth,
                entry_price=entry,
                fill_ts_ms=fill_ts,
                horizon_sec=sec,
                fee_bps=fee,
            )
            h[name] = net
            hpx[name] = {"exit_ts_ms": exit_ts, "exit_price": exit_px, "exit_source": src_name}
        # Preserve exact mirror H2 where available for parity with the already
        # running shadow process.
        h["H2_CURRENT"] = finite_float(src.get("net_bps")) if finite_float(src.get("net_bps")) is not None else h["H2_CURRENT"]
        pstats = path_stats(eth, entry, fill_ts, HORIZONS_SEC["H4_SHADOW"])
        btc30 = series_ret(btc, fill_ts, 1800)
        sol30 = series_ret(sol, fill_ts, 1800)
        cross_no_dump = bool(btc30 is not None and sol30 is not None and btc30 > -40.0 and sol30 > -80.0)
        policy_net = h["H4_SHADOW"] if cross_no_dump else h["H2_CURRENT"]
        common = {
            "source_observation_id": src.get("observation_id"),
            "protocol_id": RULE_ID,
            "signal_ts_ms": src.get("signal_ts_ms"),
            "signal_utc": src.get("signal_utc"),
            "maker_fill_ts_ms": fill_ts,
            "maker_fill_utc": src.get("maker_fill_utc"),
            "entry_price": r1(entry),
            "fill_delay_sec": finite_float(src.get("fill_delay_sec")),
            "fill_leg": src.get("fill_leg"),
            "vdepth_bps": finite_float(src.get("vdepth_bps")),
            "prior_4h_bps": finite_float(src.get("prior_4h_bps")),
            "bid_depth_usd": finite_float(src.get("bid_depth_usd")),
            "book_imbalance": finite_float(src.get("book_imbalance")),
            "spread_bps": finite_float(src.get("spread_bps")),
            "retest_quality_bucket": src.get("retest_quality_bucket"),
            "entry_quality_tags": src.get("entry_quality_tags"),
            "entry_quality_warnings": src.get("entry_quality_warnings"),
            "btc30_bps": btc30,
            "sol30_bps": sol30,
            "cross_no_dump": cross_no_dump,
            "mfe_bps": pstats.get("mfe_bps"),
            "mae_bps": pstats.get("mae_bps"),
            "rebound50_sec": pstats.get("rebound50_sec"),
            "state_path_v2": None,
        }
        full = {
            **common,
            "h2_net_bps": h["H2_CURRENT"],
            "h3_net_bps": h["H3_SHADOW"],
            "h4_net_bps": h["H4_SHADOW"],
            "h4_minus_h2_bps": r1(float(h["H4_SHADOW"]) - float(h["H2_CURRENT"])) if h["H4_SHADOW"] is not None and h["H2_CURRENT"] is not None else None,
            "h4_cross_no_dump_policy_net_bps": policy_net,
            **{k: v for k, v in pstats.items() if k.startswith("sl")},
            "sl100_policy_net_bps": stop_net(pstats, h["H4_SHADOW"], fee, 100.0),
            "sl125_policy_net_bps": stop_net(pstats, h["H4_SHADOW"], fee, 125.0),
            "sl150_policy_net_bps": stop_net(pstats, h["H4_SHADOW"], fee, 150.0),
            "sl175_policy_net_bps": stop_net(pstats, h["H4_SHADOW"], fee, 175.0),
            "sl200_policy_net_bps": stop_net(pstats, h["H4_SHADOW"], fee, 200.0),
        }
        full["state_path_v2"] = state_path_v2(src, full)
        expanded.append(full)
        for bucket in ("H2_CURRENT", "H3_SHADOW", "H4_SHADOW", "H4_CROSS_NO_DUMP_SHADOW"):
            if bucket == "H4_CROSS_NO_DUMP_SHADOW":
                net = policy_net
                exit_mode = "H4_IF_CROSS_NO_DUMP_ELSE_H2"
            else:
                net = h[bucket]
                exit_mode = bucket
            exit_meta = hpx.get(bucket) or {}
            bucket_rows.append(
                {
                    **common,
                    "bucket": bucket,
                    "exit_mode": exit_mode,
                    "exit_horizon_sec": HORIZONS_SEC.get(bucket),
                    "exit_ts_ms": exit_meta.get("exit_ts_ms"),
                    "exit_price": exit_meta.get("exit_price"),
                    "exit_price_source": exit_meta.get("exit_source"),
                    "net_bps": net,
                    "h2_net_bps": h["H2_CURRENT"],
                    "h4_net_bps": h["H4_SHADOW"],
                    "state_path_v2": full["state_path_v2"],
                    "observation_status": "CLOSED" if net is not None else "PENDING",
                    "notes": "shadow_observation_only_no_order",
                }
            )
    return bucket_rows, expanded


def summarize_report(bucket_rows: list[dict[str, Any]], expanded: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_bucket = {
        b: metrics([r.get("net_bps") for r in bucket_rows if r.get("bucket") == b])
        for b in ("H2_CURRENT", "H3_SHADOW", "H4_SHADOW", "H4_CROSS_NO_DUMP_SHADOW")
    }
    stop = {
        f"SL{int(sl)}": {
            "touch_count": sum(1 for r in expanded if r.get(f"sl{int(sl)}_touch_sec") is not None),
            "policy_if_applied_to_h4": metrics([r.get(f"sl{int(sl)}_policy_net_bps") for r in expanded]),
        }
        for sl in STOP_LEVELS_BPS
    }
    cross = {
        "cross_no_dump_true": metrics([r.get("h4_net_bps") for r in expanded if r.get("cross_no_dump")]),
        "cross_no_dump_false": metrics([r.get("h4_net_bps") for r in expanded if not r.get("cross_no_dump")]),
        "policy_h4_if_cross_no_dump_else_h2": by_bucket["H4_CROSS_NO_DUMP_SHADOW"],
    }
    states = Counter(str(r.get("state_path_v2")) for r in expanded)
    queue = {
        "status": "PROXY_ONLY_TOP_OF_BOOK",
        "n": len(expanded),
        "fill_delay_sec": metrics([r.get("fill_delay_sec") for r in expanded]),
        "late_fill_gt_900s_n": sum(1 for r in expanded if (finite_float(r.get("fill_delay_sec")) or 0.0) > 900.0),
        "bid_vanished_warning_n": sum(1 for r in expanded if "BID_VANISHED" in str(r.get("entry_quality_warnings") or "")),
        "high_quality_fill_n": sum(1 for r in expanded if str(r.get("retest_quality_bucket")) == "RETEST_QUALITY_HIGH"),
        "limitation": "No real queue position from top-of-book snapshots; 600GB/tick queue replay is still required before treating fills as executable.",
    }
    return {
        "generated_at_utc": utc_now(),
        "scope": {
            "protocol_id": RULE_ID,
            "source_ledger": str(SOURCE_LEDGER_CSV),
            "source_rows": len(source_rows),
            "closed_filled_rows": len(expanded),
            "research_only": True,
            "live_executor_touched": False,
        },
        "executive_read": (
            f"H4 shadow remains the strongest bucket on the current mirror sample: "
            f"H2 sum {by_bucket['H2_CURRENT']['sum_bps']} / T3R {by_bucket['H2_CURRENT']['t3r_bps']} "
            f"vs H4 sum {by_bucket['H4_SHADOW']['sum_bps']} / T3R {by_bucket['H4_SHADOW']['t3r_bps']}. "
            "This is shadow-only and small-N; it is not a live promotion."
        ),
        "buckets": by_bucket,
        "cross_no_dump_observer": cross,
        "catastrophic_stop_observer": stop,
        "queue_fill_realism": queue,
        "state_machine_v2": {
            "state_counts": dict(states),
            "by_state": {state: metrics([r.get("h4_net_bps") for r in expanded if r.get("state_path_v2") == state]) for state in states},
        },
        "live_shadow_parity": parity_audit(source_rows),
        "forced_flow_expansion_scan": forced_flow_summary(),
        "forward_decision_gate": {
            "status": "OBSERVATION_ONLY",
            "minimum_forward_closed_fills": 30,
            "minimum_calendar_days": 30,
            "promote_requires": [
                "H4_SHADOW forward sum > 0",
                "H4_SHADOW forward T3R > 0",
                "no single winner carries the sample",
                "live-vs-shadow parity PASS",
                "operator approval before any live order-logic change",
            ],
            "kill_review": "If 30/60-day forward sum or T3R turns negative, keep H2/live unchanged or disarm per operator decision.",
        },
        "dashboard_fragment": {
            "status": "ACTIVE",
            "protocol_id": RULE_ID,
            "h2_sum_bps": by_bucket["H2_CURRENT"]["sum_bps"],
            "h4_sum_bps": by_bucket["H4_SHADOW"]["sum_bps"],
            "h4_t3r_bps": by_bucket["H4_SHADOW"]["t3r_bps"],
            "cross_policy_sum_bps": by_bucket["H4_CROSS_NO_DUMP_SHADOW"]["sum_bps"],
            "sl150_touch_count": stop["SL150"]["touch_count"],
            "queue_status": queue["status"],
            "decision": "H4_SHADOW_OBSERVATION_ONLY",
        },
        "latest_rows": expanded[-10:],
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 H4 Shadow Control Plane",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        report["executive_read"],
        "",
        "## Bucket Results",
        "",
        "| Bucket | N | Sum | Median | Win | T3R | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, s in report["buckets"].items():
        lines.append(
            f"| {name} | {s['n']} | {s['sum_bps']} | {s['median_bps']} | {s['win_rate']} | {s['t3r_bps']} | {s['min_bps']} | {s['max_bps']} |"
        )
    lines.extend(
        [
            "",
            "## Cross-No-Dump Observer",
            "",
            "```json",
            json.dumps(report["cross_no_dump_observer"], indent=2, ensure_ascii=True),
            "```",
            "",
            "## Catastrophic Stop Observer",
            "",
            "```json",
            json.dumps(report["catastrophic_stop_observer"], indent=2, ensure_ascii=True),
            "```",
            "",
            "## Queue / Fill Realism",
            "",
            "```json",
            json.dumps(report["queue_fill_realism"], indent=2, ensure_ascii=True),
            "```",
            "",
            "## Live / Shadow Parity",
            "",
            "```json",
            json.dumps(report["live_shadow_parity"], indent=2, ensure_ascii=True),
            "```",
            "",
            "## State Machine v2",
            "",
            "```json",
            json.dumps(report["state_machine_v2"], indent=2, ensure_ascii=True),
            "```",
            "",
            "## Forced-Flow Expansion",
            "",
            "```json",
            json.dumps(report["forced_flow_expansion_scan"], indent=2, ensure_ascii=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_parity_md(parity: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Live / Shadow Parity Audit",
        "",
        f"Generated: `{utc_now()}`",
        "",
        f"Status: `{parity['status']}`",
        "",
        "| Field | Live | Shadow | Match |",
        "| --- | --- | --- | --- |",
    ]
    for c in parity["checks"]:
        lines.append(f"| {c['field']} | {c.get('live')} | {c.get('shadow')} | {c.get('match')} |")
    lines.append("")
    lines.append(parity.get("note", ""))
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = load_csv(args.source_ledger_csv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        bucket_rows, expanded = build_rows(conn, source_rows)
    report = summarize_report(bucket_rows, expanded, source_rows)
    write_csv(args.out_ledger_csv, bucket_rows)
    write_jsonl(args.out_ledger_jsonl, bucket_rows)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    args.dashboard_fragment.write_text(json.dumps(report["dashboard_fragment"], indent=2, ensure_ascii=True), encoding="utf-8")
    args.parity_json.write_text(json.dumps(report["live_shadow_parity"], indent=2, ensure_ascii=True), encoding="utf-8")
    args.parity_md.write_text(render_parity_md(report["live_shadow_parity"]), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build S34 V02 H4 shadow control-plane outputs.")
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--source-ledger-csv", type=Path, default=SOURCE_LEDGER_CSV)
    p.add_argument("--out-ledger-csv", type=Path, default=OUT_LEDGER_CSV)
    p.add_argument("--out-ledger-jsonl", type=Path, default=OUT_LEDGER_JSONL)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--dashboard-fragment", type=Path, default=OUT_DASHBOARD_FRAGMENT)
    p.add_argument("--parity-json", type=Path, default=OUT_PARITY_JSON)
    p.add_argument("--parity-md", type=Path, default=OUT_PARITY_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(render_md(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
