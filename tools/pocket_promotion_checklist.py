"""
pocket_promotion_checklist.py -- Unified GO/NO-GO readiness check before live activation.

Aggregates all conditions required to promote ETHUSDT h=60 imb>=0.85 pocket to live.

Usage:
    python -m tools.pocket_promotion_checklist --db ../eclipse_scalper/data/microstructure.db
    python -m tools.pocket_promotion_checklist --db ... --json
    python -m tools.pocket_promotion_checklist --db ... --out-json reports/PROMOTION_CHECKLIST.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

POCKET_LABEL = "micro_edge_v3_passive_alpha h=60 imb>=0.85"
REVALIDATE_CMD = (
    "py -3 -m tools.rank_passive_pockets_forward"
    " --candidates-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH_IMB05.md"
    " --db ../eclipse_scalper/data/microstructure.db"
    " --lookback-min 10080 --splits 3"
    " --out-md reports/REVALIDATE_POST_REGIME_7D.md"
    " --out-json reports/REVALIDATE_POST_REGIME_7D.json"
)

_PASS = "PASS"
_FAIL = "FAIL"
_SKIP = "SKIP"


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _item(name: str, result: str, detail: str) -> Dict[str, Any]:
    return {"name": name, "result": result, "detail": detail}


# ---------------------------------------------------------------------------
# Item 1: Event lane gate
# ---------------------------------------------------------------------------

def _call_check_gate(db: str, symbol: str) -> dict:
    try:
        from tools.check_event_lanes import check_gate
        return check_gate(db=db, symbol=symbol)
    except Exception as e:
        return {"gate": "UNKNOWN", "reason": f"error:{e}"}


def check_event_lane_gate(db: str, symbol: str) -> Dict[str, Any]:
    try:
        result = _call_check_gate(db, symbol)
        gate = str(result.get("gate", "UNKNOWN")).upper()
        if gate == "ALLOWED":
            return _item("event_lane_gate", _PASS, f"gate=ALLOWED")
        elif gate == "UNKNOWN":
            return _item("event_lane_gate", _SKIP, f"gate=UNKNOWN reason={result.get('reason', '')}")
        else:
            lanes = result.get("lanes", {})
            active = [k for k, v in lanes.items() if isinstance(v, dict) and v.get("active")]
            return _item("event_lane_gate", _FAIL, f"gate={gate} blocking_lanes={active or 'unknown'}")
    except Exception as e:
        return _item("event_lane_gate", _SKIP, f"error:{e}")


# ---------------------------------------------------------------------------
# Item 2: Re-validation result
# ---------------------------------------------------------------------------

def check_revalidation(report_path: str) -> Dict[str, Any]:
    try:
        import os
        if not os.path.exists(report_path):
            return _item("revalidation_pass", _SKIP, "report not found -- run re-validation first")
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        # Look for pass_rate or pct_pass in results list or top-level
        results = data if isinstance(data, list) else data.get("results", [])
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    rate = r.get("pass_rate") or r.get("pct_pass")
                    if rate is not None and float(rate) >= 0.5:
                        return _item("revalidation_pass", _PASS, f"pass_rate={float(rate):.2f}")
            # No pocket passed
            return _item("revalidation_pass", _FAIL, "no pocket with pass_rate>=0.5 in report")
        # Try top-level fields
        rate = data.get("pass_rate") or data.get("pct_pass")
        if rate is not None:
            rate = float(rate)
            result = _PASS if rate >= 0.5 else _FAIL
            return _item("revalidation_pass", result, f"pass_rate={rate:.2f}")
        return _item("revalidation_pass", _SKIP, "could not find pass_rate in report")
    except Exception as e:
        return _item("revalidation_pass", _SKIP, f"error reading report:{e}")


# ---------------------------------------------------------------------------
# Item 3: Fill density
# ---------------------------------------------------------------------------

def check_fill_density(report_path: str) -> Dict[str, Any]:
    try:
        import os
        if not os.path.exists(report_path):
            return _item("fill_density", _SKIP, "report not found -- run monitor_h120_fill_density first")
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        status = str(data.get("status", "")).upper()
        estimated = data.get("estimated_fills")
        if status == "READY_TO_RANK" or (estimated is not None and int(estimated) >= 30):
            detail = f"status={status} estimated_fills={estimated}"
            return _item("fill_density", _PASS, detail)
        detail = f"status={status} estimated_fills={estimated} (need >= 30)"
        return _item("fill_density", _FAIL, detail)
    except Exception as e:
        return _item("fill_density", _SKIP, f"error reading report:{e}")


# ---------------------------------------------------------------------------
# Item 4: Market regime (not strongly trending)
# ---------------------------------------------------------------------------

def check_market_regime(db: str, symbol: str) -> Dict[str, Any]:
    try:
        now_ms = int(time.time() * 1000)
        lookback_ms = 2 * 3600 * 1000  # 2 hours
        start_ms = now_ms - lookback_ms
        bucket_ms = 60 * 1000  # 1-min buckets

        con = sqlite3.connect(str(db))
        try:
            rows = con.execute(
                """
                SELECT (ts_ms / ?) * ? AS bkt,
                       SUM(price*quantity)/SUM(quantity) AS vwap
                FROM agg_trades
                WHERE symbol=? AND ts_ms BETWEEN ? AND ?
                GROUP BY bkt
                ORDER BY bkt
                """,
                (bucket_ms, bucket_ms, symbol, start_ms, now_ms),
            ).fetchall()
        finally:
            con.close()

        if len(rows) < 10:
            return _item("market_regime", _SKIP, f"insufficient data ({len(rows)} buckets, need >= 10)")

        vwaps = [float(r[1]) for r in rows]
        # Rolling 1h log-return: compare last 60 buckets vs 60 buckets before that
        half = min(60, len(vwaps) // 2)
        if half < 5:
            return _item("market_regime", _SKIP, "insufficient data for 1h window")

        recent_vwap = vwaps[-1]
        old_vwap = vwaps[-half - 1] if len(vwaps) > half else vwaps[0]
        if old_vwap <= 0:
            return _item("market_regime", _SKIP, "invalid vwap")

        ret_1h = math.log(recent_vwap / old_vwap)
        threshold = 0.015
        if abs(ret_1h) < threshold:
            return _item("market_regime", _PASS,
                         f"rolling_1h_return={ret_1h:+.4f} (below {threshold:.1%} threshold)")
        return _item("market_regime", _FAIL,
                     f"rolling_1h_return={ret_1h:+.4f} (above {threshold:.1%} threshold -- trending)")
    except Exception as e:
        return _item("market_regime", _SKIP, f"error:{e}")


# ---------------------------------------------------------------------------
# Overall logic
# ---------------------------------------------------------------------------

def run_checklist(
    db: str,
    symbol: str = "ETHUSDT",
    revalidation_report: str = "reports/REVALIDATE_POST_REGIME_7D.json",
    density_report: str = "reports/H120_DENSITY_STATUS.json",
) -> Dict[str, Any]:
    items = [
        check_event_lane_gate(db, symbol),
        check_revalidation(revalidation_report),
        check_fill_density(density_report),
        check_market_regime(db, symbol),
    ]

    has_fail = any(it["result"] == _FAIL for it in items)
    has_skip = any(it["result"] == _SKIP for it in items)

    if has_fail:
        overall = "NO-GO"
    elif has_skip:
        overall = "HOLD"
    else:
        overall = "GO"

    skip_count = sum(1 for it in items if it["result"] == _SKIP)
    fail_count = sum(1 for it in items if it["result"] == _FAIL)

    return {
        "timestamp_utc": _utc_now(),
        "symbol": symbol,
        "pocket": POCKET_LABEL,
        "overall": overall,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "checklist": items,
        "revalidate_cmd": REVALIDATE_CMD,
    }


# ---------------------------------------------------------------------------
# Human-readable output
# ---------------------------------------------------------------------------

def _print_human(payload: Dict[str, Any]) -> None:
    sym = payload["symbol"]
    pocket = payload["pocket"]
    overall = payload["overall"]
    header = f"Pocket Promotion Checklist: {sym} {pocket}"
    print(header)
    print("=" * len(header))
    for item in payload["checklist"]:
        marker = {"PASS": "[PASS]", "FAIL": "[FAIL]", "SKIP": "[SKIP]"}.get(item["result"], "[????]")
        print(f"  {marker} {item['name']:<24}: {item['detail']}")
    print()
    suffix = ""
    if overall == "HOLD":
        suffix = f" ({payload['skip_count']} SKIP items -- missing data)"
    elif overall == "NO-GO":
        suffix = f" ({payload['fail_count']} FAIL items)"
    print(f"Overall: {overall}{suffix}")
    if overall in ("HOLD", "GO"):
        print()
        print(f"Re-validation cmd:")
        print(f"  {payload['revalidate_cmd']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified GO/NO-GO readiness check for pocket live activation."
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--revalidation-report", default="reports/REVALIDATE_POST_REGIME_7D.json")
    parser.add_argument("--density-report", default="reports/H120_DENSITY_STATUS.json")
    parser.add_argument("--json", action="store_true", dest="json_out")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    payload = run_checklist(
        db=args.db,
        symbol=args.symbol,
        revalidation_report=args.revalidation_report,
        density_report=args.density_report,
    )

    if args.json_out:
        print(json.dumps(payload, indent=2))
    else:
        _print_human(payload)

    if args.out_json:
        try:
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"wrote {args.out_json}")
        except Exception as e:
            print(f"[warn] could not write --out-json: {e}", file=sys.stderr)

    return 0 if payload["overall"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
