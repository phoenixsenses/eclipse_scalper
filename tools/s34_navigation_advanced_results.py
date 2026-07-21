"""S34 navigation advanced results.

Research-only tests:
- KNN robustness over k=10/20/30/50.
- Global vs route-specific KNN.
- DANGER reverse-direction test.
- v0.3 shadow ledger build for 15x/18x TP300/SL150/4h.
- Latest navigation card.
- Pattern candidate ranker.

No live executor, order logic, leverage, size, or .env changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAV_EVENTS = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_EVENTS.jsonl"
FORWARD_PACK = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_RESEARCH_PACK.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_ADVANCED_RESULTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_ADVANCED_RESULTS.md"
V03_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V03_SHADOW_LEDGER.jsonl"
V03_MD = ROOT / "reports" / "research" / "s34" / "S34_V03_SHADOW_LEDGER.md"
NAV_CARD_JSON = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_CARD.json"
NAV_CARD_MD = ROOT / "reports" / "research" / "s34" / "S34_NAVIGATION_CARD.md"

START_EQUITY = 35.0
FEE_BPS = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def r1(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 1)


def r3(v: float | None) -> float | None:
    if v is None or not math.isfinite(float(v)):
        return None
    return round(float(v), 3)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
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
        "win_rate": r3(sum(1 for v in vals if v > 0.0) / len(vals)),
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


def knn_cards(rows: list[dict[str, Any]], *, k: int, route_specific: bool) -> list[dict[str, Any]]:
    usable = [r for r in rows if r.get("net_2h_bps") is not None]
    vecs = [feature_vector(r) for r in usable]
    cards = []
    for i, row in enumerate(usable):
        ds = []
        for j, other in enumerate(usable):
            if i == j:
                continue
            if route_specific and route_id(other) != route_id(row):
                continue
            ds.append((distance(vecs[i], vecs[j]), other))
        nn = [r for _, r in sorted(ds, key=lambda x: x[0])[:k]]
        vals = [float(r["net_2h_bps"]) for r in nn]
        s = summary(vals)
        if s["n"] < k:
            pred = "UNKNOWN"
        elif s["tail_lte_minus150_n"] == 0 and float(s.get("t3r_bps") or -1e9) > 0 and float(s.get("median_bps") or -1e9) > 0:
            pred = "CLEAN"
        elif s["tail_lte_minus150_n"] >= 2 or float(s.get("max_loss_bps") or 0.0) <= -250:
            pred = "DANGER"
        else:
            pred = "MIXED"
        actual = float(row["net_2h_bps"])
        # Reverse direction at the same 2h timestamp, accounting for fees on both sides.
        reverse = -actual - 2.0 * FEE_BPS
        cards.append(
            {
                "prediction": pred,
                "route": route_id(row),
                "actual_bps": actual,
                "reverse_bps": reverse,
                "neighbor_summary": s,
                "row": row,
            }
        )
    return cards


def summarize_cards(cards: list[dict[str, Any]]) -> dict[str, Any]:
    by_pred: dict[str, list[float]] = {}
    by_pred_rev: dict[str, list[float]] = {}
    for c in cards:
        by_pred.setdefault(c["prediction"], []).append(float(c["actual_bps"]))
        by_pred_rev.setdefault(c["prediction"], []).append(float(c["reverse_bps"]))
    return {
        "counts": {k: len(v) for k, v in sorted(by_pred.items())},
        "actual_by_prediction": {k: summary(v) for k, v in sorted(by_pred.items())},
        "reverse_by_prediction": {k: summary(v) for k, v in sorted(by_pred_rev.items())},
    }


def knn_robustness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for k in (10, 20, 30, 50):
        global_cards = knn_cards(rows, k=k, route_specific=False)
        route_cards = knn_cards(rows, k=k, route_specific=True)
        out[f"global_k{k}"] = summarize_cards(global_cards)
        out[f"route_k{k}"] = summarize_cards(route_cards)
    return out


def compound(vals: list[float], ratio: float) -> dict[str, Any]:
    equity = START_EQUITY
    peak = equity
    max_dd = 0.0
    ruined_at = None
    path = []
    for i, bps in enumerate(vals, start=1):
        equity *= 1.0 + float(ratio) * float(bps) / 10_000.0
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        if ruined_at is None and equity <= 0:
            ruined_at = i
        path.append({"i": i, "bps": r1(bps), "equity": r3(equity)})
    return {
        "end_equity": r3(equity),
        "multiple": r3(equity / START_EQUITY),
        "max_drawdown_pct": r3(abs(max_dd) / START_EQUITY * 100.0),
        "ruined_at": ruined_at,
        "path": path,
    }


def build_v03_shadow_ledger() -> dict[str, Any]:
    pack = load_json(FORWARD_PACK, {})
    rows = (pack.get("exit_management") or {}).get("rows") or []
    ledger = []
    vals_by_mode = {"V03_15X": [], "V03_18X": []}
    for row in rows:
        sim = (row.get("variants") or {}).get("tp300_sl150_4h") or {}
        if sim.get("net_bps") is None:
            continue
        net = float(sim["net_bps"])
        for mode, ratio in (("V03_15X", 15.0), ("V03_18X", 18.0)):
            vals_by_mode[mode].append(net)
            ledger.append(
                {
                    "shadow_trade_id": f"{mode}:{row.get('observation_id')}",
                    "mode": mode,
                    "ratio_notional_to_equity": ratio,
                    "protocol_id": "S34_V_ENGINE_V0_3_SHADOW_TP300_SL150_4H",
                    "signal_utc": row.get("signal_utc"),
                    "exit_variant": "tp300_sl150_4h",
                    "net_bps": r1(net),
                    "status": "SHADOW_ONLY_NO_ORDER",
                }
            )
    write_jsonl(V03_LEDGER, ledger)
    modes = {}
    for mode, vals in vals_by_mode.items():
        ratio = 15.0 if mode == "V03_15X" else 18.0
        modes[mode] = {
            "summary": summary(vals),
            "observed": compound(vals, ratio),
            "minus300": compound(vals + [-300.0], ratio),
            "minus507": compound(vals + [-507.0], ratio),
        }
    md = [
        "# S34 v0.3 Shadow Ledger",
        "",
        "Status: `SHADOW_ONLY_NO_ORDER`.",
        "",
        "| Mode | N | Sum bps | Observed End | -300 End | -507 End |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, row in modes.items():
        md.append(f"| {mode} | {row['summary']['n']} | {row['summary']['sum_bps']} | {row['observed']['end_equity']} | {row['minus300']['end_equity']} | {row['minus507']['end_equity']} |")
    V03_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"ledger_path": str(V03_LEDGER), "rows": len(ledger), "modes": modes}


def latest_navigation_card(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "NO_ROWS"}
    latest = sorted(rows, key=lambda r: int(r.get("signal_ts_ms") or 0))[-1]
    cards = knn_cards(rows, k=20, route_specific=True)
    match = None
    for c in cards:
        if c["row"].get("event_id") == latest.get("event_id"):
            match = c
            break
    if match is None:
        match = {"prediction": "UNKNOWN", "neighbor_summary": {}, "reverse_bps": None}
    card = {
        "status": "NAVIGATION_CARD_OBSERVATION_ONLY",
        "event_id": latest.get("event_id"),
        "signal_utc": latest.get("signal_utc"),
        "route": route_id(latest),
        "tags": latest.get("tags") or [],
        "knn_route_k20": {
            "prediction": match.get("prediction"),
            "neighbor_summary": match.get("neighbor_summary"),
        },
        "tail_risk": "LOW" if match.get("prediction") == "CLEAN" else ("HIGH" if match.get("prediction") == "DANGER" else "MIXED_OR_UNKNOWN"),
        "exit_preference": "EXIT_4H" if "EXIT_4H_ACTUAL_BETTER" in (latest.get("tags") or []) else "EXIT_2H",
        "sizing_preference": "SIZE_15X_STABLE" if "SIZE_15X_STABLE" in (latest.get("tags") or []) else "SIZE_34X_FRAGILE",
        "actual_label_2h_bps": latest.get("net_2h_bps"),
        "actual_reverse_2h_bps": r1(match.get("reverse_bps")) if match.get("reverse_bps") is not None else None,
        "read": "Example latest historical navigation card. Live card should be generated point-in-time at signal arrival.",
    }
    NAV_CARD_JSON.write_text(json.dumps(card, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    NAV_CARD_MD.write_text(render_card_md(card), encoding="utf-8")
    return card


def render_card_md(card: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# S34 Navigation Card",
            "",
            f"Status: `{card.get('status')}`",
            f"Route: `{card.get('route')}`",
            f"Signal: `{card.get('signal_utc')}`",
            f"Tags: `{', '.join(card.get('tags') or [])}`",
            f"KNN: `{(card.get('knn_route_k20') or {}).get('prediction')}`",
            f"Tail risk: `{card.get('tail_risk')}`",
            f"Exit: `{card.get('exit_preference')}`",
            f"Size: `{card.get('sizing_preference')}`",
            "",
        ]
    )


def pattern_ranker(rows: list[dict[str, Any]], cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pred_by_event = {c["row"].get("event_id"): c["prediction"] for c in cards}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        combo = row.get("tag_combo") or "+".join(row.get("tags") or [])
        groups.setdefault(str(combo), []).append(row)
    ranked = []
    for combo, items in groups.items():
        if len(items) < 20:
            continue
        vals = [float(r["net_2h_bps"]) for r in items]
        vals4 = [float(r["net_tp300_sl150_4h_bps"]) for r in items]
        clean_n = sum(1 for r in items if pred_by_event.get(r.get("event_id")) == "CLEAN")
        s = summary(vals)
        s4 = summary(vals4)
        score = (float(s.get("t3r_bps") or 0.0) + float(s4.get("t3r_bps") or 0.0)) - 100.0 * float(s.get("tail_lte_minus150_n") or 0.0)
        ranked.append(
            {
                "combo": combo,
                "n": len(items),
                "score": r1(score),
                "clean_frac": r3(clean_n / len(items)),
                "net_2h": s,
                "tp300_sl150_4h": s4,
                "verdict": "SHADOW_RESEARCH_LEAD" if score > 0 and clean_n / len(items) >= 0.25 else "CONTEXT_ONLY",
            }
        )
    ranked.sort(key=lambda r: float(r["score"] or -1e9), reverse=True)
    return ranked[:30]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_jsonl(args.nav_events)
    robustness = knn_robustness(rows)
    global_k20_cards = knn_cards(rows, k=20, route_specific=False)
    v03 = build_v03_shadow_ledger()
    card = latest_navigation_card(rows)
    patterns = pattern_ranker(rows, global_k20_cards)
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "knn_robustness": robustness,
        "danger_reverse_test_global_k20": summarize_cards(global_k20_cards)["reverse_by_prediction"].get("DANGER"),
        "danger_actual_test_global_k20": summarize_cards(global_k20_cards)["actual_by_prediction"].get("DANGER"),
        "v03_shadow_ledger": v03,
        "latest_navigation_card": card,
        "pattern_ranker": patterns,
        "read": "Navigation/paper-shadow research only. No live order/config changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 Navigation Advanced Results",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## KNN Robustness",
        "",
        "| Mode | CLEAN N | CLEAN Sum | CLEAN T3R | DANGER N | DANGER Sum | DANGER T3R | DANGER Reverse Sum | DANGER Reverse T3R |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, res in report["knn_robustness"].items():
        actual = res["actual_by_prediction"]
        reverse = res["reverse_by_prediction"]
        clean = actual.get("CLEAN") or summary([])
        danger = actual.get("DANGER") or summary([])
        danger_rev = reverse.get("DANGER") or summary([])
        lines.append(
            f"| {mode} | {clean['n']} | {clean['sum_bps']} | {clean['t3r_bps']} | "
            f"{danger['n']} | {danger['sum_bps']} | {danger['t3r_bps']} | "
            f"{danger_rev['sum_bps']} | {danger_rev['t3r_bps']} |"
        )
    lines.extend(
        [
            "",
            "## DANGER Reverse Test (Global k20)",
            "",
            f"- DANGER actual: `{report['danger_actual_test_global_k20']}`",
            f"- DANGER reverse: `{report['danger_reverse_test_global_k20']}`",
            "",
            "## v0.3 Shadow Ledger",
            "",
            f"- Ledger: `{report['v03_shadow_ledger']['ledger_path']}`",
            "| Mode | N | Sum | Observed End | -300 End | -507 End |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, row in report["v03_shadow_ledger"]["modes"].items():
        lines.append(
            f"| {mode} | {row['summary']['n']} | {row['summary']['sum_bps']} | "
            f"{row['observed']['end_equity']} | {row['minus300']['end_equity']} | {row['minus507']['end_equity']} |"
        )
    card = report["latest_navigation_card"]
    lines.extend(
        [
            "",
            "## Latest Navigation Card",
            "",
            f"- Route: `{card.get('route')}`",
            f"- Tags: `{', '.join(card.get('tags') or [])}`",
            f"- KNN route k20: `{(card.get('knn_route_k20') or {}).get('prediction')}`",
            f"- Tail risk: `{card.get('tail_risk')}`",
            f"- Exit preference: `{card.get('exit_preference')}`",
            f"- Sizing preference: `{card.get('sizing_preference')}`",
            "",
            "## Pattern Ranker",
            "",
            "| Verdict | Score | Combo | N | CleanFrac | 2h Sum | 2h T3R | 4hTP Sum | 4hTP T3R | Tail<=150 |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["pattern_ranker"][:20]:
        s = row["net_2h"]
        s4 = row["tp300_sl150_4h"]
        lines.append(
            f"| {row['verdict']} | {row['score']} | {row['combo']} | {row['n']} | {row['clean_frac']} | "
            f"{s['sum_bps']} | {s['t3r_bps']} | {s4['sum_bps']} | {s4['t3r_bps']} | {s['tail_lte_minus150_n']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 navigation advanced tests.")
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
