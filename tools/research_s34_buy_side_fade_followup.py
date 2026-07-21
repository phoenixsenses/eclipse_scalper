"""Follow-up tests for S34 BUY-side SHORT fade lead.

Research-only. Focuses on the BUY-side result from
research_s34_buy_side_state_machine_gauntlet.py:

    ETH BUY cascade + silence -> SHORT fade, 60m hold.

No live executor, shadow runner, env, order logic, leverage, or sizing changes.
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_buy_side_state_machine_gauntlet import (  # noqa: E402
    DB_PATH,
    FEE_BPS,
    SIL_LO_MS,
    build_buy_dataset,
    first_liq_above,
    load_liq_series,
    load_mark_series,
    ret_bps,
    signed_net,
)
from tools.research_s34_freq_tests import build_dataset as build_sell_dataset  # noqa: E402


OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_BUY_SIDE_FADE_FOLLOWUP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_BUY_SIDE_FADE_FOLLOWUP.md"

MINUTE = 60_000
HOUR = 3_600_000
PROP_THRESH = 50_000.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stat(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return {"n": 0, "wr": None, "avg": None, "sum": 0.0, "median": None, "t3r": 0.0, "worst": None, "tail100_n": 0}
    desc = sorted(vals, reverse=True)
    return {
        "n": len(vals),
        "wr": round(sum(1 for v in vals if v > 0) / len(vals), 3),
        "avg": round(mean(vals), 1),
        "sum": round(sum(vals), 1),
        "median": round(median(vals), 1),
        "t3r": round(sum(desc[3:]) if len(desc) > 3 else sum(desc), 1),
        "worst": round(min(vals), 1),
        "tail100_n": sum(1 for v in vals if v <= -100.0),
    }


def fmt(s: dict[str, Any]) -> str:
    wr = "NA" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
    avg = "NA" if s["avg"] is None else f"{float(s['avg']):+.1f}"
    return f"N={s['n']} WR={wr} avg={avg} sum={float(s['sum']):+.1f} T3R={float(s['t3r']):+.1f} worst={s['worst']} tail100={s['tail100_n']}"


def base_prefilter(row: dict[str, Any]) -> bool:
    return not bool(row["bear_squeeze"]) and str(row["session"]) != "EUROPE"


def base_silence_short(rows: list[dict[str, Any]], hold_min: int = 60) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not base_prefilter(r) or r["state"] != "SILENCE":
            continue
        net = r["short_net_by_hold"].get(str(hold_min))
        if net is None:
            continue
        ts = int(r["anchor_ts_ms"])
        out.append({**r, "entry_ts_ms": ts, "exit_ts_ms": ts + hold_min * MINUTE, "net_bps": float(net), "direction": "SHORT"})
    return out


def group_by(rows: list[dict[str, Any]], name: str, fn: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(fn(r), []).append(r)
    return {"name": name, "groups": {k: stat(v) for k, v in sorted(groups.items())}}


def tail_anatomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tails = [r for r in rows if float(r["net_bps"]) <= -100.0]
    non_tail = [r for r in rows if float(r["net_bps"]) > -100.0]
    splits = [
        group_by(rows, "session", lambda r: str(r["session"])),
        group_by(rows, "dow", lambda r: str(r["dow_name"])),
        group_by(rows, "btc4h", lambda r: "btc4h_pos" if float(r["btc4h_bps"]) > 0 else "btc4h_neg"),
        group_by(rows, "btc7d", lambda r: "btc7d_pos" if float(r["btc7d_bps"]) > 0 else "btc7d_neg"),
        group_by(rows, "sync", lambda r: "sync_ge200k" if float(r["sync_k"]) >= 200_000 else "sync_lt200k"),
        group_by(rows, "prebuild", lambda r: "prebuild_ge2" if int(r["prebuildup"]) >= 2 else "prebuild_lt2"),
        group_by(rows, "echo", lambda r: "echo_45_120" if bool(r["echo_45_120"]) else "no_echo"),
        group_by(rows, "ask_depth", lambda r: "ask_depth_hi" if (r.get("ask_depth_usd") or 0.0) >= 50_000 else "ask_depth_lo"),
        group_by(rows, "book_imbalance", lambda r: "imbalance_bid" if (r.get("book_imbalance") or 0.0) > 0 else "imbalance_ask_or_flat"),
    ]
    return {
        "base": stat(rows),
        "tail": stat(tails),
        "non_tail": stat(non_tail),
        "tail_examples": [
            {
                "anchor_utc": r["anchor_utc"],
                "net_bps": round(float(r["net_bps"]), 1),
                "session": r["session"],
                "dow": r["dow_name"],
                "btc4h": round(float(r["btc4h_bps"]), 1),
                "btc7d": round(float(r["btc7d_bps"]), 1),
                "sync_k": round(float(r["sync_k"]), 1),
                "prebuildup": int(r["prebuildup"]),
                "echo": bool(r["echo_45_120"]),
                "ask_depth": None if r.get("ask_depth_usd") is None else round(float(r["ask_depth_usd"]), 1),
            }
            for r in sorted(tails, key=lambda x: float(x["net_bps"]))[:10]
        ],
        "splits": splits,
    }


def mark_slice(series, lo: int, hi: int) -> list[tuple[int, float]]:
    a = bisect.bisect_left(series.ts, int(lo))
    b = bisect.bisect_right(series.ts, int(hi))
    return list(zip(series.ts[a:b], series.vals[a:b]))


def stop_short_net(marks, entry_ts: int, horizon_min: int, sl_bps: float | None) -> tuple[float | None, str, int | None]:
    entry = None
    i = bisect.bisect_left(marks.ts, int(entry_ts))
    if i < len(marks.vals):
        entry = float(marks.vals[i])
    if entry is None or entry <= 0:
        return None, "NO_ENTRY", None
    exit_target = int(entry_ts) + int(horizon_min) * MINUTE
    if sl_bps is not None:
        for ts, px in mark_slice(marks, int(entry_ts), exit_target):
            adverse = (float(px) - entry) / entry * 10_000.0
            if adverse >= float(sl_bps):
                return -float(sl_bps) - FEE_BPS, f"SL{int(sl_bps)}", int(ts)
    net = signed_net(marks, "SHORT", int(entry_ts), exit_target)
    return net, f"T{horizon_min}", exit_target


def stop_time_sweep(rows: list[dict[str, Any]], marks) -> dict[str, Any]:
    out = {}
    for horizon in (15, 30, 45, 60):
        for sl in (None, 50.0, 75.0, 100.0, 150.0):
            label = f"h{horizon}_sl{'none' if sl is None else int(sl)}"
            sims = []
            for r in rows:
                net, reason, exit_ts = stop_short_net(marks, int(r["entry_ts_ms"]), horizon, sl)
                if net is None:
                    continue
                sims.append({**r, "net_bps": float(net), "exit_reason": reason, "exit_ts_ms": exit_ts})
            out[label] = {
                "summary": stat(sims),
                "sl_exits": sum(1 for r in sims if str(r.get("exit_reason", "")).startswith("SL")),
            }
    return out


def exit_shape(rows: list[dict[str, Any]], marks) -> dict[str, Any]:
    out = {}
    for horizon in (5, 10, 15, 20, 30, 45, 60, 90, 120):
        sims = []
        for r in rows:
            net = signed_net(marks, "SHORT", int(r["entry_ts_ms"]), int(r["entry_ts_ms"]) + horizon * MINUTE)
            if net is not None:
                sims.append({**r, "net_bps": float(net), "exit_ts_ms": int(r["entry_ts_ms"]) + horizon * MINUTE})
        out[f"h{horizon}"] = stat(sims)
    return out


def silence_window_tests(rows: list[dict[str, Any]], eth_buy, marks) -> dict[str, Any]:
    out = {}
    for win in (10, 15, 20, 30, 45):
        sims_t0 = []
        sims_after = []
        for r in rows:
            if not base_prefilter(r):
                continue
            ts = int(r["anchor_ts_ms"])
            follow = first_liq_above(eth_buy, ts + SIL_LO_MS, ts + win * MINUTE, PROP_THRESH)
            if follow is not None:
                continue
            net0 = signed_net(marks, "SHORT", ts, ts + 60 * MINUTE)
            net_after = signed_net(marks, "SHORT", ts + win * MINUTE, ts + (win + 60) * MINUTE)
            if net0 is not None:
                sims_t0.append({**r, "net_bps": float(net0), "entry_ts_ms": ts})
            if net_after is not None:
                sims_after.append({**r, "net_bps": float(net_after), "entry_ts_ms": ts + win * MINUTE})
        out[f"silence{win}_t0_short_h60"] = stat(sims_t0)
        out[f"silence{win}_confirmed_entry_h60"] = stat(sims_after)
    return out


def sell_live_overlap(buy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sell_rows, _ = build_sell_dataset()
    # Approx current SELL state-machine trade opportunities: LONG OR-regime gate plus SHORT score>=4/BTC confirm candidates.
    sell_events = []
    for r in sell_rows:
        if r.get("close_reason") == "TIME_EXIT" and not r["bull"]:
            if (
                float(r["sync_k"]) < 200_000.0
                and r["session"] != "EUROPE"
                and not (r["session"] == "US" and int(r["hour"]) in {13, 14})
                and int(r["dow"]) not in {0, 2}
                and int(r["long_score"]) >= 3
                and ((r.get("btc4h_bps") is not None and float(r["btc4h_bps"]) < 0.0) or (r.get("btc7d_bps") is not None and float(r["btc7d_bps"]) < 0.0))
            ):
                sell_events.append({"ts": int(r["anchor_ts_ms"]), "side": "SELL_LONG"})
        if not r["bull"] and r["session"] != "EUROPE" and int(r["dow"]) != 6 and int(r["base_score"]) >= 4:
            sell_events.append({"ts": int(r["anchor_ts_ms"]), "side": "SELL_SHORT_PREFILTER"})
    sell_ts = sorted(e["ts"] for e in sell_events)

    def has_overlap(ts: int, win_ms: int) -> bool:
        i = bisect.bisect_left(sell_ts, ts - win_ms)
        return i < len(sell_ts) and sell_ts[i] <= ts + win_ms

    blocks = {}
    for win_min in (30, 60, 120, 240):
        w = win_min * MINUTE
        ov = [r for r in buy_rows if has_overlap(int(r["anchor_ts_ms"]), w)]
        no = [r for r in buy_rows if not has_overlap(int(r["anchor_ts_ms"]), w)]
        blocks[f"overlap_{win_min}m"] = {"overlap": stat(ov), "no_overlap": stat(no), "overlap_rate": round(len(ov) / len(buy_rows), 3) if buy_rows else None}
    return {"sell_event_count": len(sell_events), "windows": blocks}


def absorption_mirror(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ask_depth": group_by(rows, "ask_depth", lambda r: "ask_ge100k" if (r.get("ask_depth_usd") or 0) >= 100_000 else ("ask_ge50k" if (r.get("ask_depth_usd") or 0) >= 50_000 else "ask_lt50k")),
        "imbalance": group_by(rows, "book_imbalance", lambda r: "bid_imbalance" if (r.get("book_imbalance") or 0) > 0.1 else ("ask_imbalance" if (r.get("book_imbalance") or 0) < -0.1 else "balanced")),
        "spread": group_by(rows, "spread", lambda r: "spread_gt5" if (r.get("spread_bps") or 0) > 5 else "spread_le5"),
    }


def sync_resonance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sync_buckets": group_by(rows, "sync", lambda r: "sync_ge500k" if float(r["sync_k"]) >= 500_000 else ("sync_200_500k" if float(r["sync_k"]) >= 200_000 else "sync_lt200k")),
        "btc_regime": group_by(rows, "btc_regime", lambda r: "btc4h_pos_7d_pos" if float(r["btc4h_bps"]) > 0 and float(r["btc7d_bps"]) > 0 else ("mixed" if float(r["btc4h_bps"]) * float(r["btc7d_bps"]) < 0 else "both_nonpos")),
    }


def navigation_labels(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = {
        "BUY_CONTINUATION_DANGER": [],
        "BUY_SILENCE_FADE_WATCH": [],
        "BUY_PREBUILD2_FADE_WATCH": [],
        "BUY_SYNC_HIGH_TAIL_WARNING": [],
    }
    for r in rows:
        net = r["short_net_by_hold"].get("60")
        if net is None:
            continue
        row = {**r, "net_bps": float(net), "entry_ts_ms": int(r["anchor_ts_ms"])}
        if r["state"] == "NOISY":
            labels["BUY_CONTINUATION_DANGER"].append(row)
        if r["state"] == "SILENCE":
            labels["BUY_SILENCE_FADE_WATCH"].append(row)
        if r["state"] == "SILENCE" and int(r["prebuildup"]) >= 2:
            labels["BUY_PREBUILD2_FADE_WATCH"].append(row)
        if float(r["sync_k"]) >= 200_000:
            labels["BUY_SYNC_HIGH_TAIL_WARNING"].append(row)
    return {k: stat(v) for k, v in labels.items()}


def render(report: dict[str, Any]) -> str:
    lines = [
        "# S34 BUY-Side Fade Follow-Up",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "Research-only. No live/shadow runner, order logic, env, leverage, or sizing was changed.",
        "",
        "## Baseline",
        f"`F_silence_short_h60`: {fmt(report['baseline'])}",
        "",
        "## 1. Tail Anatomy",
        f"- Tail subset: {fmt(report['tail_anatomy']['tail'])}",
        f"- Non-tail subset: {fmt(report['tail_anatomy']['non_tail'])}",
        "",
        "### Worst Tail Examples",
        "| UTC | Net | Session | DOW | BTC4h | BTC7d | Sync | Prebuild | Echo | AskDepth |",
        "|---|---:|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in report["tail_anatomy"]["tail_examples"]:
        ask_depth = "" if r["ask_depth"] is None else f"{float(r['ask_depth']):.0f}"
        lines.append(f"| {r['anchor_utc']} | {r['net_bps']:+.1f} | {r['session']} | {r['dow']} | {r['btc4h']:+.1f} | {r['btc7d']:+.1f} | {r['sync_k']:.0f} | {r['prebuildup']} | {r['echo']} | {ask_depth} |")
    lines.extend(["", "### Tail Splits"])
    for block in report["tail_anatomy"]["splits"]:
        lines.append(f"#### {block['name']}")
        for k, s in block["groups"].items():
            lines.append(f"- `{k}`: {fmt(s)}")
        lines.append("")
    lines.extend(["## 2. Stop / Time-Stop Sweep", "| Variant | N | WR | Avg | Sum | T3R | Worst | TailN | SL exits |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for name, row in sorted(report["stop_time_sweep"].items(), key=lambda kv: (float(kv[1]["summary"]["t3r"]), float(kv[1]["summary"]["avg"] or -999)), reverse=True)[:20]:
        s = row["summary"]
        wr = "NA" if s["wr"] is None else f"{float(s['wr']) * 100:.1f}%"
        avg = "NA" if s["avg"] is None else f"{float(s['avg']):+.1f}"
        lines.append(f"| {name} | {s['n']} | {wr} | {avg} | {float(s['sum']):+.1f} | {float(s['t3r']):+.1f} | {s['worst']} | {s['tail100_n']} | {row['sl_exits']} |")
    lines.extend(["", "## 3. Exit Shape", "| Horizon | Stats |", "|---|---|"])
    for k, s in report["exit_shape"].items():
        lines.append(f"| {k} | {fmt(s)} |")
    lines.extend(["", "## 4. Silence Window / Confirmation Cost", "| Variant | Stats |", "|---|---|"])
    for k, s in report["silence_windows"].items():
        lines.append(f"| {k} | {fmt(s)} |")
    lines.extend(["", "## 5. SELL Live-Family Overlap"])
    lines.append(f"- Approx SELL-family event count: `{report['sell_overlap']['sell_event_count']}`")
    for k, block in report["sell_overlap"]["windows"].items():
        lines.append(f"- `{k}` overlap_rate={block['overlap_rate']}: overlap {fmt(block['overlap'])}; no_overlap {fmt(block['no_overlap'])}")
    lines.extend(["", "## 6. Ask-Depth / Absorption Mirror"])
    for name, block in report["absorption"].items():
        lines.append(f"### {name}")
        for k, s in block["groups"].items():
            lines.append(f"- `{k}`: {fmt(s)}")
    lines.extend(["", "## 7. Cross-Asset BUY Resonance"])
    for name, block in report["sync"].items():
        lines.append(f"### {name}")
        for k, s in block["groups"].items():
            lines.append(f"- `{k}`: {fmt(s)}")
    lines.extend(["", "## 8. Navigation Labels", "| Label | Stats |", "|---|---|"])
    for k, s in report["navigation_labels"].items():
        lines.append(f"| {k} | {fmt(s)} |")
    lines.extend(["", "## Conclusions"])
    for item in report["conclusions"]:
        lines.append(f"- {item}")
    return "\n".join(lines)


def main() -> int:
    rows, meta = build_buy_dataset()
    base = base_silence_short(rows, 60)
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        eth_marks = load_mark_series(conn, "ETHUSDT")
        eth_buy = load_liq_series(conn, "ETHUSDT", "BUY")
    report = {
        "generated_at_utc": utc_now(),
        "dataset": meta,
        "baseline": stat(base),
        "tail_anatomy": tail_anatomy(base),
        "stop_time_sweep": stop_time_sweep(base, eth_marks),
        "exit_shape": exit_shape(base, eth_marks),
        "silence_windows": silence_window_tests(rows, eth_buy, eth_marks),
        "sell_overlap": sell_live_overlap(base),
        "absorption": absorption_mirror(base),
        "sync": sync_resonance(base),
        "navigation_labels": navigation_labels(rows),
    }
    best_stop = max(report["stop_time_sweep"].items(), key=lambda kv: float(kv[1]["summary"]["t3r"]))
    best_h = max(report["exit_shape"].items(), key=lambda kv: float(kv[1]["t3r"]))
    report["conclusions"] = [
        f"Best stop/time variant by T3R is `{best_stop[0]}`: {fmt(best_stop[1]['summary'])}.",
        f"Best fixed exit by T3R is `{best_h[0]}`: {fmt(best_h[1])}.",
        "BUY-side fade has a real-looking 1h edge, but tails are structural enough that it should remain shadow until stop/confirmation rules are forward-tested.",
        "BUY continuation remains a danger/navigation label, not a long alpha.",
    ]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render(report), encoding="utf-8")
    print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
