"""S34 navigation full follow-up.

Research-only follow-up:
- DANGER reverse stability over K and horizons.
- KNN CLEAN strictness sweep.
- v0.3 shadow readout.
- Bull thin-depth tail anatomy.
- BUY 500K fade-short screen.
- Pattern-ranker criteria sweep.
- Historical navigation card ledger.

No live executor, order logic, size, leverage, or .env changes.
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

from tools.research_s34_maker_fade import collect_events
from tools.research_s34_wave_absorption import book_features_at
from tools.s34_v_engine_execution_frontier import prior_return_bps
from tools.s34_v_engine_shadow_observer import ACCEL_WINDOW_SEC, BUCKET_SEC, MIN_GAP_SEC

DEFAULT_DB = ROOT / "data" / "microstructure.db"
NAV_EVENTS = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"
V03_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V03_SHADOW_LEDGER.jsonl"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_FULL_FOLLOWUP.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_FULL_FOLLOWUP.md"
CARD_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_CARD_LEDGER.jsonl"

FEE_BPS = 5.0
START_EQUITY = 35.0


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "sum_bps": 0.0, "median_bps": None, "win_rate": None, "max_loss_bps": None, "t3r_bps": 0.0}
    t3r = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else sum(vals)
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "max_loss_bps": r1(min(vals)),
        "tail_lte_minus100_n": sum(1 for v in vals if v <= -100.0),
        "tail_lte_minus150_n": sum(1 for v in vals if v <= -150.0),
        "tail_lte_minus300_n": sum(1 for v in vals if v <= -300.0),
        "t3r_bps": r1(t3r),
    }


def feature_vector(row: dict[str, Any]) -> list[float]:
    return [
        float(row.get("threshold_usd") or 0.0) / 200_000.0,
        float(row.get("vdepth_bps") or 0.0) / 40.0,
        float(row.get("prior4h_bps") or 0.0) / 200.0,
        math.log1p(max(0.0, float(row.get("bid_depth_usd") or 0.0))) / 13.0,
        float(row.get("book_imbalance") or 0.0),
        float(row.get("eth1h_bps") or 0.0) / 100.0,
        float(row.get("btc4h_bps") or 0.0) / 100.0,
    ]


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def route_id(row: dict[str, Any]) -> str:
    return f"{row.get('symbol')}_{row.get('liq_side')}_{int(float(row.get('threshold_usd') or 0))}"


def classify_neighbor(s: dict[str, Any], mode: str) -> str:
    if s["n"] <= 0:
        return "UNKNOWN"
    if mode == "loose":
        clean = s["tail_lte_minus150_n"] <= 1 and float(s.get("t3r_bps") or -1e9) > 0
    elif mode == "base":
        clean = s["tail_lte_minus150_n"] == 0 and float(s.get("t3r_bps") or -1e9) > 0 and float(s.get("median_bps") or -1e9) > 0
    elif mode == "strict":
        clean = (
            s["tail_lte_minus150_n"] == 0
            and float(s.get("t3r_bps") or -1e9) > 0
            and float(s.get("median_bps") or -1e9) > 10
            and float(s.get("max_loss_bps") or -1e9) > -100
        )
    else:  # ultra
        clean = (
            s["tail_lte_minus100_n"] == 0
            and float(s.get("t3r_bps") or -1e9) > 100
            and float(s.get("median_bps") or -1e9) > 10
            and float(s.get("max_loss_bps") or -1e9) > -75
        )
    if clean:
        return "CLEAN"
    if s["tail_lte_minus150_n"] >= 2 or float(s.get("max_loss_bps") or 0.0) <= -250:
        return "DANGER"
    return "MIXED"


def knn_cards(rows: list[dict[str, Any]], *, k: int, strictness: str = "base") -> list[dict[str, Any]]:
    usable = [r for r in rows if r.get("net_2h_bps") is not None]
    vecs = [feature_vector(r) for r in usable]
    cards = []
    for i, row in enumerate(usable):
        ds = []
        for j, other in enumerate(usable):
            if i == j:
                continue
            ds.append((distance(vecs[i], vecs[j]), other))
        nn = [r for _, r in sorted(ds, key=lambda x: x[0])[:k]]
        vals = [float(r["net_2h_bps"]) for r in nn]
        s = summary(vals)
        pred = "UNKNOWN" if s["n"] < k else classify_neighbor(s, strictness)
        actual = float(row["net_2h_bps"])
        reverse = -actual - 2.0 * FEE_BPS
        cards.append({"prediction": pred, "row": row, "actual_bps": actual, "reverse_bps": reverse, "neighbor_summary": s})
    return cards


def summarize_cards(cards: list[dict[str, Any]], value_key: str = "actual_bps") -> dict[str, Any]:
    by_pred: dict[str, list[float]] = {}
    for c in cards:
        by_pred.setdefault(c["prediction"], []).append(float(c[value_key]))
    return {k: summary(v) for k, v in sorted(by_pred.items())}


def mark_at_or_after(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> tuple[int, float] | None:
    row = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return (int(row[0]), float(row[1])) if row else None


def horizon_net(conn: sqlite3.Connection, row: dict[str, Any], horizon_sec: int) -> tuple[float | None, float | None]:
    ts = int(row["signal_ts_ms"])
    entry = mark_at_or_after(conn, "ETHUSDT", ts)
    exit_ = mark_at_or_after(conn, "ETHUSDT", ts + int(horizon_sec) * 1000)
    if not entry or not exit_ or entry[1] <= 0:
        return None, None
    raw = (exit_[1] - entry[1]) / entry[1] * 10_000.0
    normal_long = raw - FEE_BPS
    reverse_short = -raw - FEE_BPS
    return normal_long, reverse_short


def danger_reverse_stability(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ks = [5, 8, 10, 12, 15, 20]
    horizons = {"30m": 1800, "1h": 3600, "2h": 7200, "4h": 14_400}
    out = {}
    for k in ks:
        cards = knn_cards(rows, k=k, strictness="base")
        danger = [c for c in cards if c["prediction"] == "DANGER"]
        cells = {}
        for label, sec in horizons.items():
            vals = []
            revs = []
            by_thr: dict[str, list[float]] = {}
            for c in danger:
                normal, reverse = horizon_net(conn, c["row"], sec)
                if normal is None or reverse is None:
                    continue
                vals.append(normal)
                revs.append(reverse)
                by_thr.setdefault(f"thr{int(float(c['row'].get('threshold_usd') or 0))}", []).append(reverse)
            cells[label] = {
                "danger_n": len(danger),
                "normal": summary(vals),
                "reverse": summary(revs),
                "reverse_by_threshold": {kk: summary(vv) for kk, vv in sorted(by_thr.items())},
            }
        out[f"k{k}"] = cells
    return out


def clean_strictness_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for strictness in ("loose", "base", "strict", "ultra"):
        cards = knn_cards(rows, k=20, strictness=strictness)
        out[strictness] = {
            "actual": summarize_cards(cards, "actual_bps"),
            "reverse": summarize_cards(cards, "reverse_bps"),
        }
    return out


def v03_shadow_readout() -> dict[str, Any]:
    rows = load_jsonl(V03_LEDGER)
    by_mode: dict[str, list[float]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode")), []).append(float(row["net_bps"]))
    return {mode: summary(vals) for mode, vals in sorted(by_mode.items())}


def bull_thin_tail_anatomy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [
        r for r in rows
        if {"BULL_PULLBACK", "VDEPTH_CORE", "BID_DEPTH_THIN"}.issubset(set(r.get("tags") or []))
    ]
    tails = [r for r in target if float(r.get("net_2h_bps") or 0.0) <= -100.0]
    winners = [r for r in target if float(r.get("net_2h_bps") or 0.0) > 0.0]
    def avg(key: str, items: list[dict[str, Any]]) -> float | None:
        vals = [float(r[key]) for r in items if r.get(key) is not None]
        return r1(sum(vals) / len(vals)) if vals else None
    return {
        "n": len(target),
        "overall_2h": summary([float(r["net_2h_bps"]) for r in target]),
        "tails_n": len(tails),
        "tail_profile": {
            "avg_threshold": avg("threshold_usd", tails),
            "avg_vdepth": avg("vdepth_bps", tails),
            "avg_bid_depth": avg("bid_depth_usd", tails),
            "avg_book_imbalance": avg("book_imbalance", tails),
            "avg_btc4h": avg("btc4h_bps", tails),
            "avg_eth1h": avg("eth1h_bps", tails),
        },
        "winner_profile": {
            "avg_threshold": avg("threshold_usd", winners),
            "avg_vdepth": avg("vdepth_bps", winners),
            "avg_bid_depth": avg("bid_depth_usd", winners),
            "avg_book_imbalance": avg("book_imbalance", winners),
            "avg_btc4h": avg("btc4h_bps", winners),
            "avg_eth1h": avg("eth1h_bps", winners),
        },
        "tail_rows": tails,
    }


def buy500_fade_screen(conn: sqlite3.Connection) -> dict[str, Any]:
    events = collect_events(
        conn,
        symbol="ETHUSDT",
        threshold=500_000.0,
        sides=("BUY",),
        min_vdepth_bps=5.0,
        bucket_sec=300,
        min_gap_sec=900,
        accel_window_sec=30,
        max_horizon_sec=14_400,
    )
    horizons = {"30m": 1800, "1h": 3600, "2h": 7200, "4h": 14_400}
    rows = []
    for ev in events:
        ts = int(ev.anchor.anchor_ts_ms)
        book = book_features_at(conn, "ETHUSDT", ts, 5)
        row = {
            "signal_utc": datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).isoformat(),
            "vdepth_bps": float(ev.vdepth_bps),
            "bid_depth_usd": float(book.get("bid_depth_usd") or 0.0) if book else 0.0,
            "book_imbalance": float(book.get("book_imbalance") or 0.0) if book else 0.0,
        }
        for label, sec in horizons.items():
            entry = mark_at_or_after(conn, "ETHUSDT", ts)
            exit_ = mark_at_or_after(conn, "ETHUSDT", ts + sec * 1000)
            if entry and exit_ and entry[1] > 0:
                raw = (exit_[1] - entry[1]) / entry[1] * 10_000.0
                row[f"fade_short_{label}"] = r1(-raw - FEE_BPS)
        rows.append(row)
    by_h = {label: summary([float(r[f"fade_short_{label}"]) for r in rows if r.get(f"fade_short_{label}") is not None]) for label in horizons}
    by_vdepth = {
        "v5_20": summary([float(r["fade_short_2h"]) for r in rows if 5 <= float(r["vdepth_bps"]) < 20 and r.get("fade_short_2h") is not None]),
        "v20_40": summary([float(r["fade_short_2h"]) for r in rows if 20 <= float(r["vdepth_bps"]) < 40 and r.get("fade_short_2h") is not None]),
        "v40_plus": summary([float(r["fade_short_2h"]) for r in rows if float(r["vdepth_bps"]) >= 40 and r.get("fade_short_2h") is not None]),
    }
    return {"n": len(rows), "by_horizon": by_h, "by_vdepth_2h": by_vdepth}


def pattern_ranker_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("tag_combo")), []).append(row)
    criteria = {
        "lenient": {"min_n": 20, "max_tail150": 5, "min_t3r": 0},
        "base": {"min_n": 20, "max_tail150": 2, "min_t3r": 0},
        "strict": {"min_n": 20, "max_tail150": 0, "min_t3r": 0},
    }
    out = {}
    for name, c in criteria.items():
        leads = []
        for combo, items in groups.items():
            vals = [float(r["net_2h_bps"]) for r in items if r.get("net_2h_bps") is not None]
            vals4 = [float(r["net_tp300_sl150_4h_bps"]) for r in items if r.get("net_tp300_sl150_4h_bps") is not None]
            s = summary(vals)
            s4 = summary(vals4)
            if s["n"] >= c["min_n"] and s["tail_lte_minus150_n"] <= c["max_tail150"] and float(s.get("t3r_bps") or -1e9) > c["min_t3r"]:
                leads.append({"combo": combo, "net_2h": s, "tp300_sl150_4h": s4})
        leads.sort(key=lambda x: (float(x["net_2h"].get("t3r_bps") or -1e9), float(x["net_2h"].get("sum_bps") or -1e9)), reverse=True)
        out[name] = leads[:10]
    return out


def navigation_card_ledger(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cards = knn_cards(rows, k=20, strictness="base")
    out = []
    for c in cards:
        row = c["row"]
        out.append(
            {
                "event_id": row.get("event_id"),
                "signal_utc": row.get("signal_utc"),
                "route": f"{row.get('symbol')}_{row.get('liq_side')}_{int(float(row.get('threshold_usd') or 0))}",
                "tags": row.get("tags") or [],
                "knn_global_k20": c["prediction"],
                "tail_risk": "LOW" if c["prediction"] == "CLEAN" else ("HIGH" if c["prediction"] == "DANGER" else "MIXED"),
                "actual_2h_bps": r1(c["actual_bps"]),
                "reverse_2h_bps": r1(c["reverse_bps"]),
                "neighbor_median_bps": c["neighbor_summary"].get("median_bps"),
                "neighbor_t3r_bps": c["neighbor_summary"].get("t3r_bps"),
            }
        )
    write_jsonl(CARD_LEDGER, out)
    return {"path": str(CARD_LEDGER), "rows": len(out), "latest": out[-1] if out else None}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.nav_events)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        danger = danger_reverse_stability(conn, rows)
        buy500 = buy500_fade_screen(conn)
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "danger_reverse_stability": danger,
        "clean_strictness_sweep": clean_strictness_sweep(rows),
        "v03_shadow_readout": v03_shadow_readout(),
        "bull_thin_tail_anatomy": bull_thin_tail_anatomy(rows),
        "buy500_fade_screen": buy500,
        "pattern_ranker_sweep": pattern_ranker_sweep(rows),
        "navigation_card_ledger": navigation_card_ledger(rows),
        "read": "Navigation/paper-shadow research only. No live order/config changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Navigation Full Follow-Up",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## DANGER Reverse Stability",
        "",
        "| K | Horizon | DANGER N | Normal Sum | Normal T3R | Reverse Sum | Reverse T3R | Reverse Tail<=150 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, cells in report["danger_reverse_stability"].items():
        for horizon, row in cells.items():
            lines.append(
                f"| {k} | {horizon} | {row['danger_n']} | {row['normal']['sum_bps']} | {row['normal']['t3r_bps']} | "
                f"{row['reverse']['sum_bps']} | {row['reverse']['t3r_bps']} | {row['reverse']['tail_lte_minus150_n']} |"
            )
    lines.extend([
        "",
        "## KNN CLEAN Strictness (k20)",
        "",
        "| Strictness | CLEAN N | CLEAN Sum | CLEAN T3R | CLEAN Tail<=150 | DANGER N | DANGER Sum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for strictness, row in report["clean_strictness_sweep"].items():
        actual = row["actual"]
        clean = actual.get("CLEAN") or summary([])
        danger = actual.get("DANGER") or summary([])
        lines.append(
            f"| {strictness} | {clean['n']} | {clean['sum_bps']} | {clean['t3r_bps']} | {clean['tail_lte_minus150_n']} | {danger['n']} | {danger['sum_bps']} |"
        )
    lines.extend([
        "",
        "## v0.3 Shadow Readout",
        "",
        "| Mode | N | Sum | Median | T3R | Max loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for mode, row in report["v03_shadow_readout"].items():
        lines.append(f"| {mode} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['t3r_bps']} | {row['max_loss_bps']} |")
    b = report["bull_thin_tail_anatomy"]
    lines.extend([
        "",
        "## Bull Thin-Depth Tail Anatomy",
        "",
        f"- N: `{b['n']}`; tails: `{b['tails_n']}`",
        f"- Overall 2h: `{b['overall_2h']}`",
        f"- Tail profile: `{b['tail_profile']}`",
        f"- Winner profile: `{b['winner_profile']}`",
        "",
        "## BUY500 Fade SHORT",
        "",
        "| Cell | N | Sum | Median | Win | Tail<=150 | Max loss | T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for cell, row in report["buy500_fade_screen"]["by_horizon"].items():
        lines.append(f"| horizon_{cell} | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | {row['tail_lte_minus150_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |")
    for cell, row in report["buy500_fade_screen"]["by_vdepth_2h"].items():
        lines.append(f"| {cell}_2h | {row['n']} | {row['sum_bps']} | {row['median_bps']} | {row['win_rate']} | {row['tail_lte_minus150_n']} | {row['max_loss_bps']} | {row['t3r_bps']} |")
    lines.extend([
        "",
        "## Pattern Ranker Sweep",
        "",
        "| Criteria | Combo | N | Sum | T3R | Tail<=150 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for crit, leads in report["pattern_ranker_sweep"].items():
        for lead in leads[:5]:
            s = lead["net_2h"]
            lines.append(f"| {crit} | {lead['combo']} | {s['n']} | {s['sum_bps']} | {s['t3r_bps']} | {s['tail_lte_minus150_n']} |")
    lines.extend([
        "",
        "## Navigation Card Ledger",
        "",
        f"- Path: `{report['navigation_card_ledger']['path']}`",
        f"- Rows: `{report['navigation_card_ledger']['rows']}`",
        f"- Latest: `{report['navigation_card_ledger']['latest']}`",
        "",
    ])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 navigation full follow-up.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--nav-events", type=Path, default=NAV_EVENTS)
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
