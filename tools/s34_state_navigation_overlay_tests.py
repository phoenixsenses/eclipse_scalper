"""S34 state navigation overlay tests.

Turns the hindsight/proxy labels into practical navigation questions:
- v0.2 conflict resolver / permission layer;
- state transition matrix;
- PANIC risk overlay and RECLAIM permission overlay;
- dashboard coverage and suggested non-order actions.

Research/navigation only. No live executor, order logic, size, leverage, config,
or environment changes.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_hindsight_indicator_proxy_tests import (  # noqa: E402
    BASE_FEE_BPS,
    DEFAULT_DB,
    build_feature_table,
    build_live_like_rows,
    eval_subset,
    fixed_long,
    fmt,
    long_tighten,
    mark_at_or_after,
    outcome_bracket,
    r1,
    r3,
    route_v02,
    summary,
    ts,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_NAVIGATION_OVERLAY_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_NAVIGATION_OVERLAY_TESTS.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label_flags(row: dict[str, Any]) -> dict[str, bool]:
    return {
        "CHAIN_BUILDING": int(row.get("chain_causal_15m_thresholds") or 0) >= 2 and not bool(row.get("seq_complete")),
        "CHAIN_COMPLETE": bool(row.get("seq_complete")),
        "EXHAUSTION_PROXY": bool(row.get("seq_complete")) and float(row.get("decel60_ratio") or 0.0) < -0.5,
        "PANIC_CONTINUES": float(row.get("decel60_ratio") or 0.0) >= 0.0 and float(row.get("ret60_bps") or 0.0) < 0.0,
        "RECLAIM_CONFIRMED": float(row.get("reclaim60_bps") or 0.0) > 20.0,
        "NO_TRADE_HINDSIGHT_ZONE": bool(row.get("is_near3_only")),
    }


def primary_state(row: dict[str, Any]) -> str:
    flags = label_flags(row)
    # Order matters: conflict/risk states first, then constructive states.
    for label in (
        "PANIC_CONTINUES",
        "RECLAIM_CONFIRMED",
        "EXHAUSTION_PROXY",
        "NO_TRADE_HINDSIGHT_ZONE",
        "CHAIN_BUILDING",
        "CHAIN_COMPLETE",
    ):
        if flags[label]:
            return label
    return "OTHER"


def fixed_short(conn: sqlite3.Connection, row: dict[str, Any], sec: int) -> float | None:
    a = mark_at_or_after(conn, "ETHUSDT", ts(row))
    b = mark_at_or_after(conn, "ETHUSDT", ts(row) + int(sec) * 1000)
    if not a or not b or float(a[1]) <= 0:
        return None
    return -(float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0 - BASE_FEE_BPS


def long2h_summary(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    vals = [v for r in rows if selector(r) and (v := fixed_long(conn, r, 7200)) is not None]
    return summary(vals)


def short20m_summary(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    vals = [v for r in rows if selector(r) and (v := fixed_short(conn, r, 1200)) is not None]
    return summary(vals)


def bracket_summary(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool], *, direction: str) -> dict[str, Any]:
    vals = []
    exits: dict[str, int] = defaultdict(int)
    for row in rows:
        if not selector(row):
            continue
        val, ex = outcome_bracket(
            conn,
            row,
            direction="SHORT" if direction == "SHORT" else "LONG",
            tp=200 if direction == "SHORT" else 80,
            sl=40 if direction == "SHORT" else 80,
            horizon_sec=1200,
        )
        if val is not None:
            vals.append(float(val))
            exits[str(ex)] += 1
    return {"summary": summary(vals), "exits": dict(exits)}


def label_coverage(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for label in ("PANIC_CONTINUES", "RECLAIM_CONFIRMED", "EXHAUSTION_PROXY", "CHAIN_BUILDING", "CHAIN_COMPLETE", "NO_TRADE_HINDSIGHT_ZONE"):
        sel = lambda r, label=label: label_flags(r)[label]
        out[label] = {
            "n": len([r for r in rows if sel(r)]),
            "coverage": r3(len([r for r in rows if sel(r)]) / len(rows)) if rows else None,
            "long2h": long2h_summary(conn, rows, sel),
            "long_bracket": bracket_summary(conn, rows, sel, direction="LONG"),
            "short20m": short20m_summary(conn, rows, sel),
            "short_bracket": bracket_summary(conn, rows, sel, direction="SHORT"),
        }
    return out


def v02_conflict_resolver(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    v02 = [r for r in rows if route_v02(r)]
    policies: dict[str, Callable[[dict[str, Any]], bool]] = {
        "baseline_all": lambda r: True,
        "allow_only_reclaim_or_exhaustion": lambda r: label_flags(r)["RECLAIM_CONFIRMED"] or label_flags(r)["EXHAUSTION_PROXY"],
        "skip_panic": lambda r: not label_flags(r)["PANIC_CONTINUES"],
        "skip_panic_and_chain_building": lambda r: not label_flags(r)["PANIC_CONTINUES"] and not label_flags(r)["CHAIN_BUILDING"],
        "skip_hindsight_zone": lambda r: not label_flags(r)["NO_TRADE_HINDSIGHT_ZONE"],
        "allow_tail_low_bid_ok": lambda r: "TAIL_LOW_CONTEXT" in set(r.get("tags") or []) or "BID_DEPTH_OK" in set(r.get("tags") or []),
    }
    out = {}
    for name, policy in policies.items():
        vals = []
        skipped = []
        traded = []
        for row in v02:
            val = fixed_long(conn, row, 7200)
            if val is None:
                continue
            if policy(row):
                vals.append(val)
                traded.append(row)
            else:
                skipped.append(val)
        out[name] = {
            "v02_n": len(v02),
            "traded_n": len(traded),
            "skipped_n": len(skipped),
            "traded_long2h": summary(vals),
            "skipped_counterfactual": summary(skipped),
        }

    # Label-specific v0.2 anatomy.
    out["label_breakdown"] = {}
    for label in ("PANIC_CONTINUES", "RECLAIM_CONFIRMED", "EXHAUSTION_PROXY", "CHAIN_BUILDING", "NO_TRADE_HINDSIGHT_ZONE"):
        sel = lambda r, label=label: route_v02(r) and label_flags(r)[label]
        out["label_breakdown"][label] = {
            "n": len([r for r in rows if sel(r)]),
            "long2h": long2h_summary(conn, rows, sel),
            "tighten_tp120_sl40": summary([v for r in rows if sel(r) and (v := long_tighten(conn, r, tp=120, sl=40, horizon_sec=7200)) is not None]),
            "exit_after_60s": summary([v for r in rows if sel(r) and (v := fixed_long(conn, r, 60)) is not None]),
        }
    return out


def state_transition_matrix(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=ts)
    transitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prev, cur in zip(ordered, ordered[1:]):
        gap_sec = (ts(cur) - ts(prev)) / 1000.0
        if gap_sec < 0 or gap_sec > 900:
            continue
        a = primary_state(prev)
        b = primary_state(cur)
        transitions[f"{a}->{b}"].append(cur)
    out = {}
    for name, items in transitions.items():
        if len(items) < 5:
            continue
        out[name] = {
            "n": len(items),
            "long2h": long2h_summary(conn, items, lambda r: True),
            "long_bracket": bracket_summary(conn, items, lambda r: True, direction="LONG"),
            "short20m": short20m_summary(conn, items, lambda r: True),
            "short_bracket": bracket_summary(conn, items, lambda r: True, direction="SHORT"),
        }
    return dict(sorted(out.items(), key=lambda kv: abs(float(kv[1]["short_bracket"]["summary"].get("t3r_bps") or 0.0)) + abs(float(kv[1]["long_bracket"]["summary"].get("t3r_bps") or 0.0)), reverse=True))


def conflict_focus(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Directly answer the practical question: v0.2 LONG signal + state label.
    v02 = [r for r in rows if route_v02(r)]
    out = {}
    for label in ("PANIC_CONTINUES", "RECLAIM_CONFIRMED", "EXHAUSTION_PROXY", "CHAIN_BUILDING", "CHAIN_COMPLETE"):
        yes = [r for r in v02 if label_flags(r)[label]]
        no = [r for r in v02 if not label_flags(r)[label]]
        out[label] = {
            "yes_n": len(yes),
            "yes_long2h": long2h_summary(conn, yes, lambda r: True),
            "yes_tighten": summary([v for r in yes if (v := long_tighten(conn, r, tp=120, sl=40, horizon_sec=7200)) is not None]),
            "no_n": len(no),
            "no_long2h": long2h_summary(conn, no, lambda r: True),
        }
    return out


def action_recommendations(result: dict[str, Any]) -> list[dict[str, Any]]:
    recs = []
    labels = result["label_coverage"]
    v02 = result["v02_conflict_resolver"]["label_breakdown"]
    panic_short = labels["PANIC_CONTINUES"]["short_bracket"]["summary"]
    reclaim_long = labels["RECLAIM_CONFIRMED"]["long_bracket"]["summary"]
    exhaustion_long = labels["EXHAUSTION_PROXY"]["long_bracket"]["summary"]
    recs.append(
        {
            "label": "PANIC_CONTINUES",
            "suggested_use": "dashboard red-light / SHORT pressure / LONG caution",
            "evidence": panic_short,
            "live_action": "notify_only",
        }
    )
    recs.append(
        {
            "label": "RECLAIM_CONFIRMED",
            "suggested_use": "dashboard green-light for rebound state",
            "evidence": reclaim_long,
            "live_action": "notify_only",
        }
    )
    recs.append(
        {
            "label": "EXHAUSTION_PROXY",
            "suggested_use": "small-N rebound permission; keep shadow",
            "evidence": exhaustion_long,
            "live_action": "shadow_only",
        }
    )
    if float(v02.get("EXHAUSTION_PROXY", {}).get("tighten_tp120_sl40", {}).get("t3r_bps") or 0.0) > float(v02.get("EXHAUSTION_PROXY", {}).get("long2h", {}).get("t3r_bps") or 0.0):
        recs.append(
            {
                "label": "EXHAUSTION_PROXY_ON_V02",
                "suggested_use": "would-tighten observer for v0.2",
                "evidence": v02["EXHAUSTION_PROXY"],
                "live_action": "shadow_observer_only",
            }
        )
    return recs


def run() -> dict[str, Any]:
    rows = build_live_like_rows()
    with sqlite3.connect(DEFAULT_DB) as conn:
        ft = build_feature_table(conn, rows)
        result: dict[str, Any] = {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "rows_n": len(ft),
            "label_coverage": label_coverage(conn, ft),
            "v02_conflict_resolver": v02_conflict_resolver(conn, ft),
            "state_transition_matrix": state_transition_matrix(conn, ft),
            "conflict_focus": conflict_focus(conn, ft),
        }
    result["action_recommendations"] = action_recommendations(result)
    return result


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 State Navigation Overlay Tests",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Rows: `{result['rows_n']}`",
        "",
        "## 1. Label Coverage / Direction",
        "",
        "| Label | N | Coverage | LONG 2h | LONG bracket | SHORT 20m | SHORT bracket |",
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for label, row in result["label_coverage"].items():
        lines.append(
            f"| `{label}` | {row['n']} | {row['coverage']} | {fmt(row['long2h'])} | "
            f"{fmt(row['long_bracket']['summary'])} | {fmt(row['short20m'])} | {fmt(row['short_bracket']['summary'])} |"
        )

    lines.extend(["", "## 2. v0.2 Conflict Resolver Policies", ""])
    lines.append("| Policy | Traded N | Skipped N | Traded long2h | Skipped counterfactual |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for name, row in result["v02_conflict_resolver"].items():
        if name == "label_breakdown":
            continue
        lines.append(f"| `{name}` | {row['traded_n']} | {row['skipped_n']} | {fmt(row['traded_long2h'])} | {fmt(row['skipped_counterfactual'])} |")

    lines.extend(["", "## 3. v0.2 Label Breakdown", ""])
    lines.append("| Label | N | Long2h | Tighten TP120/SL40 | Exit after 60s |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for label, row in result["v02_conflict_resolver"]["label_breakdown"].items():
        lines.append(f"| `{label}` | {row['n']} | {fmt(row['long2h'])} | {fmt(row['tighten_tp120_sl40'])} | {fmt(row['exit_after_60s'])} |")

    lines.extend(["", "## 4. State Transition Matrix", ""])
    lines.append("| Transition | N | LONG 2h | LONG bracket | SHORT 20m | SHORT bracket |")
    lines.append("| --- | ---: | --- | --- | --- | --- |")
    for name, row in list(result["state_transition_matrix"].items())[:30]:
        lines.append(
            f"| `{name}` | {row['n']} | {fmt(row['long2h'])} | {fmt(row['long_bracket']['summary'])} | "
            f"{fmt(row['short20m'])} | {fmt(row['short_bracket']['summary'])} |"
        )

    lines.extend(["", "## 5. Conflict Focus", ""])
    lines.append("| Label | Yes N | Yes long2h | Yes tighten | No N | No long2h |")
    lines.append("| --- | ---: | --- | --- | ---: | --- |")
    for label, row in result["conflict_focus"].items():
        lines.append(f"| `{label}` | {row['yes_n']} | {fmt(row['yes_long2h'])} | {fmt(row['yes_tighten'])} | {row['no_n']} | {fmt(row['no_long2h'])} |")

    lines.extend(["", "## 6. Recommendations", ""])
    for row in result["action_recommendations"]:
        lines.append(f"- `{row['label']}`: {row['suggested_use']}; live_action=`{row['live_action']}`; evidence=`{row['evidence']}`")

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
