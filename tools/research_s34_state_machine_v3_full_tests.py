"""S34 state-machine v3 full test suite.

Research-only. No live executor, env, runtime, or order changes.

This extends the v2 gauntlet with the concrete follow-up questions:
- provisional entry + state-resolution actions
- state transition / Markov stability
- conflict variants
- month and regime stability
- exit/stop/latency/slippage/book realism
- timestamp-level shadow/backfill parity approximation
"""

from __future__ import annotations

import bisect
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_state_machine_v2_gauntlet import (  # noqa: E402
    CAL_FRAC,
    DEFAULT_DB,
    DOW,
    FEE_BPS,
    NAV_EVENTS,
    PROP_THRESH,
    SIL_HI_MS,
    SIL_LO_MS,
    SYNC_WIN_MS,
    Config,
    apply_conflict_policy,
    build_signals,
    classify_rows,
    finite,
    first_above,
    iso_ms,
    load_liq,
    load_marks,
    load_nav_events,
    mark_at_or_after,
    recompute_score,
    short_btc_outcome,
    signed_net,
    state_for,
    summarize,
    summary_with_dd,
    utc_now,
    win_cnt,
    win_sum,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V3_FULL_TESTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STATE_MACHINE_V3_FULL_TESTS.md"
SHADOW_LEDGER = ROOT / "reports" / "shadow" / "s34_realtime_shadow.jsonl"


def pct(vals: list[float], p: float) -> float | None:
    vals = sorted(float(v) for v in vals if math.isfinite(float(v)))
    if not vals:
        return None
    idx = max(0, min(int((len(vals) - 1) * p / 100.0), len(vals) - 1))
    return round(vals[idx], 1)


def by_split(signals: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": summary_with_dd(signals),
        "cal": summary_with_dd([s for s in signals if not s["row"]["is_hold"]]),
        "hold": summary_with_dd([s for s in signals if s["row"]["is_hold"]]),
    }


def mark_min(ts: list[int], px: list[float], lo: int, hi: int) -> float | None:
    a = bisect.bisect_left(ts, int(lo))
    b = bisect.bisect_right(ts, int(hi))
    if a >= b:
        return None
    return float(min(px[a:b]))


def mark_max(ts: list[int], px: list[float], lo: int, hi: int) -> float | None:
    a = bisect.bisect_left(ts, int(lo))
    b = bisect.bisect_right(ts, int(hi))
    if a >= b:
        return None
    return float(max(px[a:b]))


def score_at_t0(row: dict[str, Any], *, sync_thr: float = 200_000.0, n2h_thr: int = 3) -> int:
    """Score excluding the unknowable silence bit."""
    return sum(
        [
            int(row["n2h"] >= n2h_thr),
            int(row["b4h"] < 0),
            int(row["vd"] >= 30),
            int(row["sess_us"]),
            int(row["sync_k"] >= sync_thr),
        ]
    )


def eth_follow_on_ts(row: dict[str, Any], eth_ts: list[int], eth_not: list[float]) -> int | None:
    ts = int(row["ts"])
    return first_above(eth_ts, eth_not, ts + SIL_LO_MS, ts + SIL_HI_MS, PROP_THRESH)


def btc_follow_on_ts(row: dict[str, Any], btc_thr: float) -> int | None:
    return row["first_btc_by_thr"].get(str(int(btc_thr)))


def net_between(side: str, mk_ts: list[int], mk_px: list[float], entry_ts: int | None, exit_ts: int | None, fee: float = FEE_BPS) -> float | None:
    if entry_ts is None or exit_ts is None:
        return None
    return signed_net(side, mark_at_or_after(mk_ts, mk_px, int(entry_ts)), mark_at_or_after(mk_ts, mk_px, int(exit_ts)), fee)


def add_nets(*vals: float | None) -> float | None:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if len(clean) != len(vals):
        return None
    return float(sum(clean))


def provisional_suite(rows: list[dict[str, Any]], eth_ts: list[int], eth_not: list[float], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    candidates = [r for r in rows if not r["bull"]]
    for min_score in range(0, 6):
        chosen = [r for r in candidates if score_at_t0(r) >= min_score]
        variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in chosen:
            ts = int(r["ts"])
            eth_fo = eth_follow_on_ts(r, eth_ts, eth_not)
            btc750 = btc_follow_on_ts(r, 750_000.0)
            btc1000 = btc_follow_on_ts(r, 1_000_000.0)
            # 1. enter LONG at T0; if no follow-on, hold 4h; if follow-on, exit.
            exit_ts = eth_fo if eth_fo is not None else ts + 4 * 3600_000
            net = net_between("LONG", mk_ts, mk_px, ts, exit_ts)
            if net is not None:
                variants["long_exit_on_eth_noisy"].append({"entry_ts_ms": ts, "net_bps": net, "row": r})
            # 2. enter LONG at T0; if ETH noisy, flip SHORT at ETH follow-on; else hold LONG 4h.
            if eth_fo is not None:
                net = add_nets(
                    net_between("LONG", mk_ts, mk_px, ts, eth_fo),
                    net_between("SHORT", mk_ts, mk_px, eth_fo, eth_fo + 2 * 3600_000),
                )
            else:
                net = net_between("LONG", mk_ts, mk_px, ts, ts + 4 * 3600_000)
            if net is not None:
                variants["long_flip_short_on_eth_noisy"].append({"entry_ts_ms": ts, "net_bps": net, "row": r})
            # 3. enter LONG at T0; if BTC750 occurs, flip SHORT there; elif ETH noisy, exit; else hold.
            if btc750 is not None:
                net = add_nets(
                    net_between("LONG", mk_ts, mk_px, ts, btc750),
                    net_between("SHORT", mk_ts, mk_px, btc750, btc750 + 2 * 3600_000),
                )
            elif eth_fo is not None:
                net = net_between("LONG", mk_ts, mk_px, ts, eth_fo)
            else:
                net = net_between("LONG", mk_ts, mk_px, ts, ts + 4 * 3600_000)
            if net is not None:
                variants["long_flip_short_on_btc750_else_exit"].append({"entry_ts_ms": ts, "net_bps": net, "row": r})
            # 4. same but BTC1000.
            if btc1000 is not None:
                net = add_nets(
                    net_between("LONG", mk_ts, mk_px, ts, btc1000),
                    net_between("SHORT", mk_ts, mk_px, btc1000, btc1000 + 2 * 3600_000),
                )
            elif eth_fo is not None:
                net = net_between("LONG", mk_ts, mk_px, ts, eth_fo)
            else:
                net = net_between("LONG", mk_ts, mk_px, ts, ts + 4 * 3600_000)
            if net is not None:
                variants["long_flip_short_on_btc1000_else_exit"].append({"entry_ts_ms": ts, "net_bps": net, "row": r})
            # 5. confirmed-only: no T0 risk, enter LONG at T+30 only for silence, SHORT at BTC750 for neither.
            if eth_fo is None:
                net = net_between("LONG", mk_ts, mk_px, ts + SIL_HI_MS, ts + 4 * 3600_000)
                if net is not None:
                    variants["confirmed_only_long_t30_or_short_btc750"].append({"entry_ts_ms": ts + SIL_HI_MS, "net_bps": net, "row": r})
            elif btc750 is not None:
                net = net_between("SHORT", mk_ts, mk_px, btc750, btc750 + 2 * 3600_000)
                if net is not None:
                    variants["confirmed_only_long_t30_or_short_btc750"].append({"entry_ts_ms": btc750, "net_bps": net, "row": r})
        out[f"t0_score_ge{min_score}"] = {k: by_split(v) for k, v in variants.items()}
    return out


def event_sequence_context(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted([r for r in rows if not r["bull"]], key=lambda r: int(r["ts"]))
    enriched = []
    prev_state = None
    prev_ts = None
    for r in ordered:
        st = state_for(r, 750_000.0)
        enriched.append({**r, "state750": st, "prev_state750": prev_state, "prev_gap_min": ((int(r["ts"]) - prev_ts) / 60_000.0 if prev_ts else None)})
        prev_state = st
        prev_ts = int(r["ts"])
    return enriched


def markov_suite(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = event_sequence_context(rows)
    transitions: dict[str, int] = defaultdict(int)
    from_counts: dict[str, int] = defaultdict(int)
    expectancy: dict[str, list[float]] = defaultdict(list)
    for r in ordered:
        prev = r.get("prev_state750")
        if not prev:
            continue
        cur = r["state750"]
        key = f"{prev}->{cur}"
        transitions[key] += 1
        from_counts[str(prev)] += 1
        val = r["long_t0_4h"] if r["sil_eth"] else r["short_anchor_2h"]
        if val is not None:
            expectancy[key].append(float(val))
    return {
        "transition_probabilities": {
            k: {"n": n, "p_from_prev": round(n / from_counts[k.split("->")[0]], 3)}
            for k, n in sorted(transitions.items())
        },
        "transition_expectancy": {k: summarize(v) for k, v in sorted(expectancy.items())},
    }


def conflict_variants(signals: list[dict[str, Any]]) -> dict[str, Any]:
    base = {}
    for policy in ["all_independent", "one_pos_ignore", "short_replace"]:
        taken, blocked = apply_conflict_policy(signals, policy)
        base[policy] = {"taken": by_split(taken), "blocked": by_split(blocked)}

    # Timer reset is approximated by replacing the previous same-side position with the newer signal.
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    selected: list[dict[str, Any]] = []
    active_idx: int | None = None
    active_end = None
    active_side = None
    for sig in ordered:
        side = sig["side"]
        hold_ms = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
        entry = int(sig["entry_ts_ms"])
        if active_end is None or entry >= active_end:
            selected.append(sig)
            active_idx = len(selected) - 1
            active_side = side
            active_end = entry + hold_ms
        elif side == active_side and active_idx is not None:
            selected[active_idx] = {**sig, "conflict_action": "same_side_timer_reset"}
            active_end = entry + hold_ms
        elif side == "SHORT" and active_side == "LONG":
            selected.append({**sig, "conflict_action": "flip_long_to_short"})
            active_idx = len(selected) - 1
            active_side = side
            active_end = entry + hold_ms
    base["same_side_timer_reset"] = {"taken": by_split(selected), "blocked": {"all": summarize([]), "cal": summarize([]), "hold": summarize([])}}
    return base


def monthly_stability(signals: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in signals:
        dt = datetime.fromtimestamp(int(s["entry_ts_ms"]) / 1000.0, tz=timezone.utc)
        buckets[dt.strftime("%Y-%m")].append(s)
    return {k: summary_with_dd(v) for k, v in sorted(buckets.items())}


def dow_stability(signals: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in signals:
        buckets[DOW[int(s["row"]["dow"])]].append(s)
    return {k: summary_with_dd(v) for k, v in sorted(buckets.items(), key=lambda kv: DOW.index(kv[0]))}


def horizon_exit_suite(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float], *, include_by_arm: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for hours in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        vals = []
        for s in signals:
            entry = int(s["entry_ts_ms"])
            net = net_between(s["side"], mk_ts, mk_px, entry, entry + int(hours * 3600_000))
            if net is not None:
                vals.append({"entry_ts_ms": entry, "net_bps": net, "row": s["row"]})
        out[f"h{hours:g}"] = by_split(vals)
    by_arm = {}
    if include_by_arm:
        for arm in sorted({s["arm"] for s in signals}):
            by_arm[arm] = horizon_exit_suite([s for s in signals if s["arm"] == arm], mk_ts, mk_px, include_by_arm=False) if arm else {}
        out["by_arm"] = by_arm
    return out


def stop_suite(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stop in [40, 60, 80, 100, 150, 200]:
        vals = []
        triggered = 0
        for s in signals:
            entry_ts = int(s["entry_ts_ms"])
            side = str(s["side"])
            horizon = 4 * 3600_000 if side == "LONG" else 2 * 3600_000
            entry_px = mark_at_or_after(mk_ts, mk_px, entry_ts)
            if entry_px is None or entry_px <= 0:
                continue
            if side == "LONG":
                lo = mark_min(mk_ts, mk_px, entry_ts, entry_ts + horizon)
                adverse = ((lo - entry_px) / entry_px * 10_000.0) if lo is not None else 0.0
            else:
                hi = mark_max(mk_ts, mk_px, entry_ts, entry_ts + horizon)
                adverse = ((entry_px - hi) / entry_px * 10_000.0) if hi is not None else 0.0
            if adverse <= -float(stop):
                vals.append({"entry_ts_ms": entry_ts, "net_bps": -float(stop) - FEE_BPS, "row": s["row"]})
                triggered += 1
            else:
                vals.append(s)
        sm = by_split(vals)
        sm["triggered"] = triggered
        sm["triggered_pct"] = round(triggered / len(signals), 3) if signals else 0
        out[f"sl{stop}"] = sm
    return out


def tail_cluster_suite(signals: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(signals, key=lambda s: int(s["entry_ts_ms"]))
    next_after_tail = []
    next_after_win = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        if float(a["net_bps"]) <= -100:
            next_after_tail.append(b)
        if float(a["net_bps"]) > 0:
            next_after_win.append(b)
    return {
        "tail_threshold_bps": -100,
        "next_after_tail": summary_with_dd(next_after_tail),
        "next_after_win": summary_with_dd(next_after_win),
        "tail_count": sum(1 for s in ordered if float(s["net_bps"]) <= -100),
    }


def latency_suite(signals: list[dict[str, Any]], mk_ts: list[int], mk_px: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for delay_s in [0, 5, 15, 30, 60]:
        vals = []
        for s in signals:
            entry = int(s["entry_ts_ms"]) + delay_s * 1000
            horizon = 4 * 3600_000 if s["side"] == "LONG" else 2 * 3600_000
            net = net_between(s["side"], mk_ts, mk_px, entry, entry + horizon)
            if net is not None:
                vals.append({"entry_ts_ms": entry, "net_bps": net, "row": s["row"]})
        out[f"delay_{delay_s}s"] = by_split(vals)
    return out


def slippage_suite(signals: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for slip in [0, 5, 10, 20, 30]:
        vals = [{**s, "net_bps": float(s["net_bps"]) - float(slip)} for s in signals]
        out[f"slip_{slip}bps"] = by_split(vals)
    return out


def book_at(conn: sqlite3.Connection, ts_ms: int, max_stale_sec: int = 10) -> dict[str, float] | None:
    row = conn.execute(
        "SELECT ts_ms,bid_price,ask_price,mid_price,spread_pct,book_imbalance,bid_depth_usd "
        "FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
        (int(ts_ms),),
    ).fetchone()
    if not row:
        return None
    stale_ms = int(ts_ms) - int(row[0])
    if stale_ms < 0 or stale_ms > max_stale_sec * 1000:
        return None
    return {
        "ts_ms": int(row[0]),
        "bid": float(row[1]),
        "ask": float(row[2]),
        "mid": float(row[3]),
        "spread_pct": float(row[4]),
        "imbalance": float(row[5]),
        "bid_depth_usd": float(row[6] or 0.0),
        "stale_ms": float(stale_ms),
    }


def book_realism_suite(signals: list[dict[str, Any]], db_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for stale in [1, 5, 10, 30]:
            vals = []
            missing = 0
            spreads = []
            for s in signals:
                entry_ts = int(s["entry_ts_ms"])
                horizon = 4 * 3600_000 if s["side"] == "LONG" else 2 * 3600_000
                ex = book_at(conn, entry_ts, stale)
                ox = book_at(conn, entry_ts + horizon, stale)
                if not ex or not ox:
                    missing += 1
                    continue
                if s["side"] == "LONG":
                    entry = ex["ask"]
                    exit_ = ox["bid"]
                else:
                    entry = ex["bid"]
                    exit_ = ox["ask"]
                net = signed_net(s["side"], entry, exit_)
                if net is not None:
                    vals.append({"entry_ts_ms": entry_ts, "net_bps": net, "row": s["row"]})
                    spreads.append(float(ex["spread_pct"]) * 100.0)
            sm = by_split(vals)
            sm["missing"] = missing
            sm["coverage"] = round(len(vals) / len(signals), 3) if signals else 0
            sm["entry_spread_bps_median"] = pct(spreads, 50)
            out[f"book_stale_{stale}s"] = sm
    return out


def shadow_timestamp_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not SHADOW_LEDGER.exists():
        return {"exists": False}
    ledger_ids = set()
    closes = 0
    with SHADOW_LEDGER.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source") == "BACKFILL" and rec.get("event") == "CLOSE":
                closes += 1
                ledger_ids.add(str(rec.get("id")))
    expected = set()
    for r in rows:
        ts = int(r["ts"])
        base = f"SHD:{ts}"
        if r["sil_eth"]:
            expected.add(f"{base}:LS")
        if (not r["sil_eth"]) and (not r["bull"]):
            expected.add(f"{base}:SP")
            if r["first_btc_by_thr"].get("500000") is not None:
                expected.add(f"{base}:SN")
    return {
        "exists": True,
        "ledger_backfill_closes": closes,
        "expected_ids": len(expected),
        "matching_ids": len(ledger_ids & expected),
        "missing_expected_ids": len(expected - ledger_ids),
        "extra_ledger_ids": len(ledger_ids - expected),
        "parity_ratio": round(len(ledger_ids & expected) / len(expected), 3) if expected else None,
        "note": "ID-level parity only; P&L parity differs because backfill uses NAV labels while this suite recomputes mark/book outcomes.",
    }


def render_stats_table(title: str, items: list[tuple[str, dict[str, Any]]], split: str = "hold") -> list[str]:
    lines = [f"## {title}", "", "| Name | N | WR | Sum | Mean | Median | T3R | Max loss | Max DD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, val in items:
        s = val.get(split, val) if isinstance(val, dict) else {}
        wr = "" if s.get("wr") is None else f"{float(s['wr']) * 100:.1f}%"
        lines.append(
            f"| {name} | {s.get('n',0)} | {wr} | {s.get('sum')} | {s.get('mean')} | {s.get('median')} | {s.get('t3r')} | {s.get('max_loss')} | {s.get('max_dd_bps','')} |"
        )
    lines.append("")
    return lines


def render_md(r: dict[str, Any]) -> str:
    primary = r["primary"]
    lines = [
        "# S34 State Machine V3 Full Tests",
        "",
        f"- generated_at_utc: `{r['generated_at_utc']}`",
        f"- research_only: `{r['research_only']}`",
        f"- primary_config: `{r['primary_config']}`",
        f"- primary_hold: `{primary['hold']}`",
        "",
        "## Executive Read",
        "",
        f"- provisional_best: `{r['top_findings']['best_provisional']}`",
        f"- execution_book_hold: `{r['top_findings']['book_realism_hold']}`",
        f"- slippage_20bps_hold: `{r['top_findings']['slippage_20_hold']}`",
        f"- shadow_id_parity: `{r['shadow_timestamp_parity']['parity_ratio']}`",
        "",
    ]
    # Provisional compact
    best_prov = []
    for score, variants in r["provisional"].items():
        for name, splits in variants.items():
            hold = splits["hold"]
            best_prov.append((f"{score}:{name}", splits))
    best_prov = sorted(best_prov, key=lambda kv: float(kv[1]["hold"].get("t3r") or -1e18), reverse=True)[:12]
    lines += render_stats_table("Top Provisional / State-Resolution Variants", best_prov, "hold")
    lines += render_stats_table("Conflict Variants", [(k, v["taken"]) for k, v in r["conflict_variants"].items()], "hold")
    lines += render_stats_table("Monthly Stability", list(r["monthly_stability"].items()), "all")
    lines += render_stats_table("DOW Stability", list(r["dow_stability"].items()), "all")
    lines += render_stats_table("Exit Horizon Holdout", [(k, v) for k, v in r["exit_horizons"].items() if k != "by_arm"], "hold")
    lines += render_stats_table("Stop Sweep Holdout", list(r["stop_sweep"].items()), "hold")
    lines += render_stats_table("Latency Holdout", list(r["latency"].items()), "hold")
    lines += render_stats_table("Slippage Holdout", list(r["slippage"].items()), "hold")
    lines += render_stats_table("Book Realism Holdout", list(r["book_realism"].items()), "hold")
    lines += [
        "## Markov / Transition Summary",
        "",
        "| Transition | N | P from previous | WR | Mean | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    exp = r["markov"]["transition_expectancy"]
    probs = r["markov"]["transition_probabilities"]
    for key, s in sorted(exp.items(), key=lambda kv: float(kv[1].get("t3r") or -1e18), reverse=True):
        p = probs.get(key, {})
        wr = "" if s.get("wr") is None else f"{float(s['wr']) * 100:.1f}%"
        lines.append(f"| {key} | {s.get('n')} | {p.get('p_from_prev')} | {wr} | {s.get('mean')} | {s.get('t3r')} |")
    lines += [
        "",
        "## Tail Cluster",
        "",
        f"- tail_count: `{r['tail_cluster']['tail_count']}`",
        f"- next_after_tail: `{r['tail_cluster']['next_after_tail']}`",
        f"- next_after_win: `{r['tail_cluster']['next_after_win']}`",
        "",
        "## Shadow Timestamp Parity",
        "",
        f"- {r['shadow_timestamp_parity']}",
        "",
        "## Interpretation",
        "",
        "State-machine still looks strongest as provisional-entry plus state-resolution. The biggest live blocker remains not statistical edge, but executable parity and the fact that SILENCE is not knowable at T=0.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    nav = load_nav_events()
    with sqlite3.connect(f"file:{DEFAULT_DB}?mode=ro", uri=True) as conn:
        eth_ts, eth_not = load_liq(conn, "ETHUSDT", "SELL")
        btc_ts, btc_not = load_liq(conn, "BTCUSDT", "SELL")
        sol_ts, sol_not = load_liq(conn, "SOLUSDT", "SELL")
        event_ts = [int(r["signal_ts_ms"]) for r in nav if finite(r.get("threshold_usd")) is not None]
        mk_ts, mk_px = load_marks(conn, "ETHUSDT", min(event_ts) - 60_000, max(event_ts) + 6 * 3600_000)
    rows = classify_rows(nav, eth_ts=eth_ts, eth_not=eth_not, btc_ts=btc_ts, btc_not=btc_not, sol_ts=sol_ts, sol_not=sol_not, mk_ts=mk_ts, mk_px=mk_px)
    primary_cfg = Config("btc750_dow_score3", btc_thr=750_000, long_score_min=3, short_score_min=3, exclude_long_dow=(0, 2), exclude_short_dow=(6,))
    primary_raw = build_signals(rows, primary_cfg, mk_ts=mk_ts, mk_px=mk_px)
    primary, primary_blocked = apply_conflict_policy(primary_raw, "short_replace")
    report = {
        "generated_at_utc": utc_now(),
        "research_only": True,
        "counts": {
            "nav_events": len(nav),
            "classified": len(rows),
            "cal": sum(1 for r in rows if not r["is_hold"]),
            "hold": sum(1 for r in rows if r["is_hold"]),
            "holdout_cutoff_utc": rows[int(len(rows) * CAL_FRAC)]["holdout_cutoff_utc"] if rows else None,
        },
        "primary_config": primary_cfg.name,
        "primary": by_split(primary),
        "primary_blocked": by_split(primary_blocked),
        "provisional": provisional_suite(rows, eth_ts, eth_not, mk_ts, mk_px),
        "markov": markov_suite(rows),
        "conflict_variants": conflict_variants(primary_raw),
        "monthly_stability": monthly_stability(primary),
        "dow_stability": dow_stability(primary),
        "exit_horizons": horizon_exit_suite(primary, mk_ts, mk_px),
        "stop_sweep": stop_suite(primary, mk_ts, mk_px),
        "tail_cluster": tail_cluster_suite(primary),
        "latency": latency_suite(primary, mk_ts, mk_px),
        "slippage": slippage_suite(primary),
        "book_realism": book_realism_suite(primary, DEFAULT_DB),
        "shadow_timestamp_parity": shadow_timestamp_parity(rows),
    }
    # Small top-line pointers.
    best_prov = []
    for score, variants in report["provisional"].items():
        for name, splits in variants.items():
            best_prov.append((score, name, splits["hold"]))
    best_prov.sort(key=lambda x: float(x[2].get("t3r") or -1e18), reverse=True)
    report["top_findings"] = {
        "best_provisional": {"score": best_prov[0][0], "variant": best_prov[0][1], "hold": best_prov[0][2]} if best_prov else None,
        "book_realism_hold": report["book_realism"].get("book_stale_10s", {}).get("hold"),
        "slippage_20_hold": report["slippage"].get("slip_20bps", {}).get("hold"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    OUT_MD.write_text(render_md(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(json.dumps({"primary": report["primary"]["hold"], "top_findings": report["top_findings"], "shadow_parity": report["shadow_timestamp_parity"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
