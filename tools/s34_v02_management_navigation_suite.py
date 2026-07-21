"""S34 V02 management/navigation test suite.

Research-only. Consolidates the current V02 H4 shadow, V02 mirror, navigation
overlay, sizing shadow, and operational status into the 10 requested tests.
No live executor/config/order logic is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import pctile, r1  # noqa: E402

OUT_DIR = ROOT / "reports" / "research" / "s34"
H4_JSON = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW.json"
H4_LEDGER_JSONL = OUT_DIR / "S34_V02_H4_FORWARD_SHADOW_LEDGER.jsonl"
MIRROR_BRIEF_JSON = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_BRIEF.json"
NAV_OVERLAY_JSON = OUT_DIR / "S34_V02_ALPHA_NAVIGATION_OVERLAY_120D.json"
SIZING_JSON = OUT_DIR / "S34_V_ENGINE_SIZING_SHADOW_PAPER.json"
PAPER_STATUS_JSON = OUT_DIR / "S34_SHADOW_PAPER_STATUS.json"
LIVE_STATE_JSON = ROOT / "runtime" / "s34_v_engine_live_state.json"
OUT_JSON = OUT_DIR / "S34_V02_MANAGEMENT_NAVIGATION_SUITE.json"
OUT_MD = OUT_DIR / "S34_V02_MANAGEMENT_NAVIGATION_SUITE.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def metrics(vals: list[Any]) -> dict[str, Any]:
    xs = [float(x) for x in (f(v) for v in vals) if x is not None]
    if not xs:
        return {"n": 0, "sum": 0.0, "mean": None, "median": None, "win_rate": None, "t3r": 0.0, "min": None, "max": None}
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum": r1(sum(xs)),
        "mean": r1(mean(xs)),
        "median": r1(pctile(xs, 0.5)),
        "win_rate": round(sum(1 for x in xs if x > 0.0) / len(xs), 3),
        "t3r": r1(sum(ordered[3:]) if len(ordered) > 3 else sum(ordered)),
        "min": r1(min(xs)),
        "max": r1(max(xs)),
    }


def group_metrics(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    out = {}
    for val in sorted({str(r.get(key)) for r in rows}):
        out[val] = metrics([r.get(value) for r in rows if str(r.get(key)) == val])
    return out


def month_key(utc_text: Any) -> str:
    text = str(utc_text or "")
    return text[:7] if len(text) >= 7 else "NA"


def verdict_from_bucket(bucket: dict[str, Any]) -> str:
    n = int(bucket.get("n") or 0)
    if n < 30:
        return "SMALL_N_SHADOW_ONLY"
    if (f(bucket.get("sum_bps")) or f(bucket.get("sum")) or 0.0) > 0 and (f(bucket.get("t3r_bps")) or f(bucket.get("t3r")) or 0.0) > 0:
        return "FORWARD_CANDIDATE"
    return "FAIL"


def h4_unique_rows(h4_ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [r for r in h4_ledger if r.get("bucket") == "H4_SHADOW"]
    rows.sort(key=lambda r: int(f(r.get("maker_fill_ts_ms")) or 0))
    for r in rows:
        h2 = f(r.get("h2_net_bps"))
        h4 = f(r.get("h4_net_bps"))
        r["h4_minus_h2_bps"] = None if h2 is None or h4 is None else r1(h4 - h2)
        r["runner_h4"] = bool(r["h4_minus_h2_bps"] is not None and r["h4_minus_h2_bps"] > 0)
        delay = f(r.get("fill_delay_sec")) or 0.0
        r["fill_delay_bucket"] = "FAST_0_60" if delay <= 60 else ("NORMAL_60_900" if delay <= 900 else "LATE_GT900")
        mae = f(r.get("mae_bps"))
        r["mae_bucket"] = "NA" if mae is None else ("PAIN_GE100" if mae <= -100 else ("PAIN_50_100" if mae <= -50 else "CLEAN_LT50"))
        rebound = f(r.get("rebound50_sec"))
        r["rebound50_bucket"] = "NO_REBOUND50" if rebound is None else ("REBOUND50_FAST_30M" if rebound <= 1800 else "REBOUND50_LATE")
        r["month"] = month_key(r.get("maker_fill_utc") or r.get("signal_utc"))
    return rows


def nav_trades(nav: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [r for r in nav.get("sample_filled_trades", []) if r.get("status") == "FILLED"]
    for r in rows:
        r["nav_score_bucket"] = "NAV_HIGH" if (f(r.get("nav_score_fill")) or 0.0) >= 7 else ("NAV_MID" if (f(r.get("nav_score_fill")) or 0.0) >= 5 else "NAV_LOW")
    return rows


def run_suite() -> dict[str, Any]:
    h4 = load_json(H4_JSON, {})
    h4_ledger = load_jsonl(H4_LEDGER_JSONL)
    mirror = load_json(MIRROR_BRIEF_JSON, {})
    nav = load_json(NAV_OVERLAY_JSON, {})
    sizing = load_json(SIZING_JSON, {})
    paper_status = load_json(PAPER_STATUS_JSON, {})
    live_state = load_json(LIVE_STATE_JSON, {})
    h4_rows = h4_unique_rows(h4_ledger)
    nav_rows = nav_trades(nav)
    buckets = h4.get("buckets") or {}
    h2 = buckets.get("H2_CURRENT") or {}
    h3 = buckets.get("H3_SHADOW") or {}
    h4b = buckets.get("H4_SHADOW") or {}
    cross = buckets.get("H4_CROSS_NO_DUMP_SHADOW") or {}
    runner_rows = [r for r in h4_rows if r.get("runner_h4")]
    non_runner_rows = [r for r in h4_rows if not r.get("runner_h4")]

    tests: dict[str, Any] = {}
    tests["01_h4_forward_validation"] = {
        "h2": h2,
        "h3": h3,
        "h4": h4b,
        "h4_minus_h2": metrics([r.get("h4_minus_h2_bps") for r in h4_rows]),
        "verdict": verdict_from_bucket(h4b),
        "read": "H4 leads H2 in current shadow sample, but N<30 so it remains shadow-only.",
    }
    tests["02_h4_runner_predictor"] = {
        "runner_count": len(runner_rows),
        "non_runner_count": len(non_runner_rows),
        "runner_h4_minus_h2": metrics([r.get("h4_minus_h2_bps") for r in runner_rows]),
        "non_runner_h4_minus_h2": metrics([r.get("h4_minus_h2_bps") for r in non_runner_rows]),
        "by_cross_no_dump": group_metrics(h4_rows, "cross_no_dump", "h4_minus_h2_bps"),
        "by_fill_delay": group_metrics(h4_rows, "fill_delay_bucket", "h4_minus_h2_bps"),
        "by_mae": group_metrics(h4_rows, "mae_bucket", "h4_minus_h2_bps"),
        "by_rebound50": group_metrics(h4_rows, "rebound50_bucket", "h4_minus_h2_bps"),
        "verdict": "HYPOTHESIS_ONLY_SMALL_N",
    }
    tests["03_cross_no_dump_observer"] = {
        "observer": h4.get("cross_no_dump_observer") or {},
        "verdict": "HYPOTHESIS_ONLY_SMALL_N",
        "read": "Cross-no-dump improves policy sum in-sample, but false bucket has N=1.",
    }
    tests["04_catastrophic_stop_reality"] = {
        "observer": h4.get("catastrophic_stop_observer") or {},
        "verdict": "SL150_CATASTROPHIC_ONLY_IN_SAMPLE",
        "read": "SL100/125 degrade current sample; SL150+ never touched in current N=11.",
    }
    tests["05_queue_fill_realism"] = {
        "queue": h4.get("queue_fill_realism") or {},
        "late_fill_rows": [r for r in h4_rows if r.get("fill_delay_bucket") == "LATE_GT900"],
        "by_delay_h4": group_metrics(h4_rows, "fill_delay_bucket", "h4_net_bps"),
        "verdict": "PROXY_ONLY_NEEDS_TICK_QUEUE_REPLAY",
    }
    tests["06_shadow_paper_bucket_health"] = {
        "mirror_rows_total": (mirror.get("ledger") or {}).get("rows_total"),
        "mirror_closed_filled": (mirror.get("overall") or {}).get("closed_filled"),
        "h4_closed_filled": (h4.get("scope") or {}).get("closed_filled_rows"),
        "h4_ledger_rows": len(h4_ledger),
        "expected_h4_ledger_rows": 4 * len(h4_rows),
        "paper_status": {
            "updated_at_utc": paper_status.get("updated_at_utc"),
            "total_trades": paper_status.get("total_trades"),
            "open_trades": paper_status.get("open_trades"),
            "closed_trades": paper_status.get("closed_trades"),
        },
        "live_state": {
            "mode": ((live_state.get("status") or {}).get("mode")),
            "active": live_state.get("active"),
            "orders_count": len(live_state.get("orders") or []),
        },
        "verdict": "PARITY_OK" if (mirror.get("overall") or {}).get("closed_filled") == (h4.get("scope") or {}).get("closed_filled_rows") == len(h4_rows) else "REVIEW",
    }
    tests["07_regime_drift"] = {
        "by_month_h2": group_metrics(h4_rows, "month", "h2_net_bps"),
        "by_month_h4": group_metrics(h4_rows, "month", "h4_net_bps"),
        "mirror_weekly": mirror.get("weekly") or [],
        "verdict": "TOO_FEW_MONTHS_FOR_DRIFT_DECISION",
    }
    tests["08_navigation_indicator_context"] = {
        "nav_overlay_baseline": (nav.get("fill_set") or {}).get("baseline_2h"),
        "by_nav_high_fill": (nav.get("navigation_overlay") or {}).get("nav_high_fill") or (nav.get("buy_spike_overlay") or {}).get("nav_high_fill"),
        "by_nav_high_holds_5m": (nav.get("navigation_overlay") or {}).get("nav_high_holds_5m") or (nav.get("buy_spike_overlay") or {}).get("nav_high_holds_5m"),
        "by_nav_score_bucket": group_metrics(nav_rows, "nav_score_bucket", "net_2h_bps"),
        "state_sequences_top": nav.get("state_sequences_top") or {},
        "verdict": "NAV_CONTEXT_ONLY_NOT_ENTRY_RULE",
    }
    tests["09_state_sequence_model"] = {
        "h4_state_counts": dict(Counter(str(r.get("state_path_v2")) for r in h4_rows)),
        "h4_by_state": group_metrics(h4_rows, "state_path_v2", "h4_net_bps"),
        "h4_delta_by_state": group_metrics(h4_rows, "state_path_v2", "h4_minus_h2_bps"),
        "nav_state_sequences_top": nav.get("state_sequences_top") or {},
        "verdict": "PROMISING_STRUCTURE_BUT_SMALL_N",
    }
    tests["10_kill_promote_rules"] = {
        "promote_gate": {
            "min_forward_closed_fills": 30,
            "min_calendar_days": 30,
            "requires_h4_sum_gt_0": True,
            "requires_h4_t3r_gt_0": True,
            "requires_no_single_winner_dependence": True,
            "requires_operator_approval": True,
        },
        "current_status": {
            "closed_fills": len(h4_rows),
            "h4_sum": h4b.get("sum_bps"),
            "h4_t3r": h4b.get("t3r_bps"),
            "decision": "DO_NOT_PROMOTE_YET_SMALL_N",
        },
        "kill_gate": {
            "30_or_60_day_forward_sum_lt_0": "pause_or_disarm_review",
            "forward_t3r_lt_0_after_min_3": "pause_review",
            "tail_or_stop_budget_breach": "operator_size_review",
        },
    }

    result = {
        "generated_at_utc": utc_now(),
        "scope": {
            "rule": "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID",
            "h4_rows": len(h4_rows),
            "nav_rows": len(nav_rows),
            "research_only": True,
            "live_executor_touched": False,
        },
        "executive_read": (
            "All 10 management/navigation tests were run on the current V02 shadow sample. "
            "H4 remains the best management hypothesis, but every positive result is still small-N. "
            "No live promotion is justified before forward N>=30."
        ),
        "tests": tests,
    }
    return result


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# S34 V02 Management / Navigation Test Suite",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        result["executive_read"],
        "",
        "## Summary",
        "",
        f"- H4 rows: `{result['scope']['h4_rows']}`",
        f"- NAV rows: `{result['scope']['nav_rows']}`",
        f"- research only: `{result['scope']['research_only']}`",
        f"- live executor touched: `{result['scope']['live_executor_touched']}`",
        "",
    ]
    for key, value in result["tests"].items():
        lines.extend([f"## {key}", "", "```json", json.dumps(value, indent=2, ensure_ascii=True), "```", ""])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()
    result = run_suite()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    args.out_md.write_text(render_md(result), encoding="utf-8")
    print(render_md(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
