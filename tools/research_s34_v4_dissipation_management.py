"""S34 v4 dissipation management backtest.

Research-only. Tests post-entry management overlays for deep-V SELL fades:
enter as usual, observe book replenishment / liq deceleration at tau, then either
exit early at tau or hold to 4h. No live/paper state is touched.
"""

from __future__ import annotations

import argparse
import bisect
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

from tools.research_s34_cross_asset_absorption_pool import DEFAULT_DB
from tools.research_s34_knowable_anchor_continuation import book_at, file_fingerprint, load_liquidations, signed_return_bps
from tools.research_s34_wave_absorption import book_features_at


IN_JSON = ROOT / "reports" / "research" / "s34" / "S34_ABSORPTION_SYNC_2X2_POOL.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V4_DISSIPATION_MANAGEMENT_BACKTEST.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V4_DISSIPATION_MANAGEMENT_BACKTEST.md"

TAUS_SEC = (60, 90, 120, 180)
HOLD_SEC = 4 * 3600
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def percentile(vals: list[float], q: float) -> float | None:
    xs = sorted(v for v in vals if math.isfinite(v))
    if not xs:
        return None
    idx = int(round((len(xs) - 1) * float(q)))
    return xs[max(0, min(len(xs) - 1, idx))]


def month_split(rows: list[dict[str, Any]], split: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    hold_months = set(split.get("holdout_months", []))
    cal = [r for r in rows if str(r.get("month")) not in hold_months]
    hold = [r for r in rows if str(r.get("month")) in hold_months]
    return cal, hold, hold_months


def metrics(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    if not xs:
        return {
            "n": 0,
            "sum_bps": 0.0,
            "mean_bps": None,
            "median_bps": None,
            "win_rate_pct": None,
            "t3r_bps": 0.0,
            "max_loss_bps": None,
            "tail_lt_100": 0,
            "tail_lt_200": 0,
            "tail_lt_400": 0,
        }
    ordered = sorted(xs, reverse=True)
    return {
        "n": len(xs),
        "sum_bps": round(sum(xs), 1),
        "mean_bps": round(sum(xs) / len(xs), 1),
        "median_bps": round(median(xs), 1),
        "win_rate_pct": round(100.0 * sum(1 for v in xs if v > 0.0) / len(xs), 1),
        "t3r_bps": round(sum(ordered[3:]) if len(ordered) > 3 else sum(xs), 1),
        "max_loss_bps": round(min(xs), 1),
        "tail_lt_100": sum(1 for v in xs if v < -100.0),
        "tail_lt_200": sum(1 for v in xs if v < -200.0),
        "tail_lt_400": sum(1 for v in xs if v < -400.0),
    }


def compare(base_vals: list[float], managed_vals: list[float]) -> dict[str, Any]:
    b = metrics(base_vals)
    m = metrics(managed_vals)
    return {
        "baseline": b,
        "managed": m,
        "delta_sum_bps": round(float(m["sum_bps"] or 0.0) - float(b["sum_bps"] or 0.0), 1),
        "delta_t3r_bps": round(float(m["t3r_bps"] or 0.0) - float(b["t3r_bps"] or 0.0), 1),
        "delta_max_loss_bps": None
        if m["max_loss_bps"] is None or b["max_loss_bps"] is None
        else round(float(m["max_loss_bps"]) - float(b["max_loss_bps"]), 1),
        "delta_tail_lt_100": int(m["tail_lt_100"] or 0) - int(b["tail_lt_100"] or 0),
        "delta_tail_lt_200": int(m["tail_lt_200"] or 0) - int(b["tail_lt_200"] or 0),
        "delta_tail_lt_400": int(m["tail_lt_400"] or 0) - int(b["tail_lt_400"] or 0),
    }


def window_liq(ts: list[int], rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> float:
    lo = bisect.bisect_right(ts, int(start_ms))
    hi = bisect.bisect_right(ts, int(end_ms))
    return sum(float(rows[i]["notional"]) for i in range(lo, hi))


def load_liq_index(conn: sqlite3.Connection) -> dict[str, tuple[list[int], list[dict[str, Any]]]]:
    out = {}
    for symbol in SYMBOLS:
        rows = load_liquidations(conn, symbol, "SELL", None, None)
        out[symbol] = ([int(r["ts_ms"]) for r in rows], rows)
    return out


def enrich_rows(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    *,
    taus_sec: tuple[int, ...],
    max_book_staleness_sec: int,
    fee_bps_side: float,
) -> list[dict[str, Any]]:
    liq_idx = load_liq_index(conn)
    out = []
    for row in rows:
        r = dict(row)
        symbol = str(row["symbol"])
        entry_ts = int(row["entry_ts_ms"])
        entry_ask = finite(row.get("ask"))
        base_total = finite(row.get("total_top_depth_usd"))
        if entry_ask is None or entry_ask <= 0:
            continue
        ts, lrows = liq_idx[symbol]
        r["baseline_4h_net_bps"] = float(row["net_bps"])
        for tau in taus_sec:
            tau_ts = entry_ts + int(tau) * 1000
            tau_book = book_at(conn, symbol, tau_ts, int(max_book_staleness_sec))
            tau_features = book_features_at(conn, symbol, tau_ts, int(max_book_staleness_sec))
            if tau_book:
                gross_tau = signed_return_bps("LONG", float(entry_ask), float(tau_book.bid))
                r[f"exit_tau_{tau}s_net_bps"] = gross_tau - 2.0 * float(fee_bps_side)
                r[f"tau_exit_book_ts_ms_{tau}s"] = int(tau_book.ts_ms)
                r[f"tau_exit_staleness_ms_{tau}s"] = int(tau_book.staleness_ms)
            if tau_features and base_total and base_total > 0:
                r[f"total_replenish_{tau}s_pct"] = (
                    (float(tau_features["total_top_depth_usd"]) - base_total) / base_total * 100.0
                )
                r[f"bid_replenish_{tau}s_pct"] = (
                    (float(tau_features["bid_depth_usd"]) - float(row["bid_depth_usd"])) / float(row["bid_depth_usd"]) * 100.0
                    if float(row["bid_depth_usd"]) > 0
                    else None
                )
                r[f"spread_change_{tau}s_bps"] = float(tau_features["spread_bps"]) - float(row["spread_bps"])
            pre = window_liq(ts, lrows, entry_ts - int(tau) * 1000, entry_ts)
            post = window_liq(ts, lrows, entry_ts, tau_ts)
            r[f"pre_liq_{tau}s_k"] = pre / 1000.0
            r[f"post_liq_{tau}s_k"] = post / 1000.0
            r[f"liq_deceleration_{tau}s"] = (pre - post) / max(pre, 1.0)
        out.append(r)
    return out


def rule_decision(row: dict[str, Any], *, tau: int, rule: str, cuts: dict[str, float]) -> tuple[bool, str]:
    repl = finite(row.get(f"total_replenish_{tau}s_pct"))
    decel = finite(row.get(f"liq_deceleration_{tau}s"))
    if repl is None or decel is None:
        return False, "missing_tau_features"
    repl_ok = repl >= cuts["replenish_cut"]
    decel_ok = decel >= cuts["decel_cut"]
    if rule == "replenish_only":
        return repl_ok, "hold_replenish" if repl_ok else "exit_low_replenish"
    if rule == "dual_and":
        ok = repl_ok and decel_ok
        return ok, "hold_dual_confirm" if ok else "exit_dual_fail"
    if rule == "dual_or":
        ok = repl_ok or decel_ok
        return ok, "hold_any_confirm" if ok else "exit_no_confirm"
    raise ValueError(f"unknown rule {rule}")


def apply_management(rows: list[dict[str, Any]], *, tau: int, rule: str, cuts: dict[str, float]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tau_net = finite(row.get(f"exit_tau_{tau}s_net_bps"))
        base = finite(row.get("baseline_4h_net_bps"))
        if tau_net is None or base is None:
            continue
        hold, reason = rule_decision(row, tau=tau, rule=rule, cuts=cuts)
        r = dict(row)
        r["management_tau_sec"] = tau
        r["management_rule"] = rule
        r["management_decision"] = "HOLD_4H" if hold else "EXIT_EARLY"
        r["management_reason"] = reason
        r["managed_net_bps"] = float(base if hold else tau_net)
        r["management_delta_bps"] = float(r["managed_net_bps"]) - float(base)
        out.append(r)
    return out


def cuts_from_cal(cal: list[dict[str, Any]], *, tau: int, replenish_q: float, decel_q: float) -> dict[str, float] | None:
    repl_vals = [finite(r.get(f"total_replenish_{tau}s_pct")) for r in cal]
    decel_vals = [finite(r.get(f"liq_deceleration_{tau}s")) for r in cal]
    repl = percentile([v for v in repl_vals if v is not None], replenish_q)
    decel = percentile([v for v in decel_vals if v is not None], decel_q)
    if repl is None or decel is None:
        return None
    return {"replenish_cut": float(repl), "decel_cut": float(decel)}


def evaluate_config(
    cal: list[dict[str, Any]],
    hold: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    *,
    tau: int,
    rule: str,
    replenish_q: float,
    decel_q: float,
) -> dict[str, Any] | None:
    cuts = cuts_from_cal(cal, tau=tau, replenish_q=replenish_q, decel_q=decel_q)
    if cuts is None:
        return None
    cal_m = apply_management(cal, tau=tau, rule=rule, cuts=cuts)
    hold_m = apply_management(hold, tau=tau, rule=rule, cuts=cuts)
    all_m = apply_management(all_rows, tau=tau, rule=rule, cuts=cuts)

    def vals(rows: list[dict[str, Any]], key: str) -> list[float]:
        return [float(r[key]) for r in rows if finite(r.get(key)) is not None]

    return {
        "config_id": f"tau{tau}_{rule}_replQ{int(replenish_q*100)}_decelQ{int(decel_q*100)}",
        "tau_sec": tau,
        "rule": rule,
        "replenish_q": replenish_q,
        "decel_q": decel_q,
        "cuts": {k: round(v, 4) for k, v in cuts.items()},
        "coverage": {"all": len(all_m), "cal": len(cal_m), "hold": len(hold_m)},
        "all": compare(vals(all_m, "baseline_4h_net_bps"), vals(all_m, "managed_net_bps")),
        "cal": compare(vals(cal_m, "baseline_4h_net_bps"), vals(cal_m, "managed_net_bps")),
        "hold": compare(vals(hold_m, "baseline_4h_net_bps"), vals(hold_m, "managed_net_bps")),
        "hold_decisions": {
            "hold_4h": sum(1 for r in hold_m if r["management_decision"] == "HOLD_4H"),
            "exit_early": sum(1 for r in hold_m if r["management_decision"] == "EXIT_EARLY"),
        },
        "rows": all_m,
    }


def summarize_by_symbol(rows: list[dict[str, Any]], hold_months: set[str]) -> list[dict[str, Any]]:
    hold = [r for r in rows if str(r.get("month")) in hold_months]
    out = []
    for symbol in sorted({str(r["symbol"]) for r in hold}):
        sub = [r for r in hold if str(r["symbol"]) == symbol]
        out.append(
            {
                "symbol": symbol,
                "n": len(sub),
                "baseline": metrics([float(r["baseline_4h_net_bps"]) for r in sub]),
                "managed": metrics([float(r["managed_net_bps"]) for r in sub]),
                "decisions": {
                    "hold_4h": sum(1 for r in sub if r["management_decision"] == "HOLD_4H"),
                    "exit_early": sum(1 for r in sub if r["management_decision"] == "EXIT_EARLY"),
                },
            }
        )
    return out


def build_report(conn: sqlite3.Connection, args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    source_rows = payload.get("rows", [])
    enriched = enrich_rows(
        conn,
        source_rows,
        taus_sec=tuple(args.taus_sec),
        max_book_staleness_sec=int(args.max_book_staleness_sec),
        fee_bps_side=float(args.fee_bps_side),
    )
    cal, hold, hold_months = month_split(enriched, payload.get("split", {}))
    configs = []
    for tau in args.taus_sec:
        for rule in ("replenish_only", "dual_and", "dual_or"):
            for rq in args.replenish_quantiles:
                for dq in args.deceleration_quantiles:
                    cfg = evaluate_config(
                        cal,
                        hold,
                        enriched,
                        tau=int(tau),
                        rule=rule,
                        replenish_q=float(rq),
                        decel_q=float(dq),
                    )
                    if cfg:
                        configs.append(cfg)
    configs.sort(
        key=lambda c: (
            float(c["hold"]["delta_t3r_bps"]),
            float(c["hold"]["delta_sum_bps"]),
            float(c["hold"]["managed"]["sum_bps"]),
        ),
        reverse=True,
    )
    primary = next((c for c in configs if c["tau_sec"] == 120 and c["rule"] == "dual_and" and c["replenish_q"] == 0.5 and c["decel_q"] == 0.5), None)
    best = configs[0] if configs else None
    consistent = [
        c
        for c in configs
        if c["cal"]["delta_sum_bps"] > 0
        and c["cal"]["delta_t3r_bps"] > 0
        and c["hold"]["delta_sum_bps"] > 0
        and c["hold"]["delta_t3r_bps"] > 0
    ]
    consistent.sort(
        key=lambda c: (
            float(c["hold"]["delta_t3r_bps"]),
            float(c["cal"]["delta_t3r_bps"]),
            float(c["hold"]["delta_sum_bps"]),
        ),
        reverse=True,
    )
    if best:
        best["hold_by_symbol"] = summarize_by_symbol(best["rows"], hold_months)
    if primary:
        primary["hold_by_symbol"] = summarize_by_symbol(primary["rows"], hold_months)
    live_route = []
    if best:
        live_route = [
            r
            for r in best["rows"]
            if r.get("symbol") == "ETHUSDT"
            and float(r.get("threshold_usd") or 0.0) == 200_000.0
            and r.get("vdepth_band") == "v28_40"
            and r.get("bid_depth_gate") == "deep_bid"
        ]
    return {
        "generated_at_utc": utc_now(),
        "mode": "RESEARCH_ONLY_NO_LIVE_NO_PAPER",
        "source_db": file_fingerprint(args.db),
        "input_json": str(args.input_json),
        "split": payload.get("split", {}),
        "config": {
            "taus_sec": list(args.taus_sec),
            "hold_sec": HOLD_SEC,
            "fee_bps_side": float(args.fee_bps_side),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "rules": ["replenish_only", "dual_and", "dual_or"],
            "replenish_quantiles": list(args.replenish_quantiles),
            "deceleration_quantiles": list(args.deceleration_quantiles),
        },
        "coverage": {"rows": len(enriched), "cal": len(cal), "hold": len(hold)},
        "overall_baseline": {
            "all": metrics([float(r["baseline_4h_net_bps"]) for r in enriched]),
            "cal": metrics([float(r["baseline_4h_net_bps"]) for r in cal]),
            "hold": metrics([float(r["baseline_4h_net_bps"]) for r in hold]),
        },
        "primary_tau120_dual_median": primary,
        "best_by_holdout_delta_t3r": best,
        "consistent_cal_hold_positive": [{k: v for k, v in c.items() if k != "rows"} for c in consistent],
        "live_v02_lane_diagnostic_on_best_config": None
        if not best
        else {
            "config_id": best["config_id"],
            "note": "Small-N diagnostic only; this is the currently live v0.2 lane shape.",
            "rows": len(live_route),
            "baseline": metrics([float(r["baseline_4h_net_bps"]) for r in live_route]),
            "managed": metrics([float(r["managed_net_bps"]) for r in live_route]),
            "decisions": {
                "hold_4h": sum(1 for r in live_route if r["management_decision"] == "HOLD_4H"),
                "exit_early": sum(1 for r in live_route if r["management_decision"] == "EXIT_EARLY"),
            },
        },
        "ranked_configs": [{k: v for k, v in c.items() if k != "rows"} for c in configs],
    }


def fmt(m: dict[str, Any]) -> str:
    return (
        f"N={m['n']} sum={m['sum_bps']} med={m['median_bps']} "
        f"T3R={m['t3r_bps']} max_loss={m['max_loss_bps']} "
        f"tail<-100={m['tail_lt_100']} tail<-200={m['tail_lt_200']}"
    )


def compare_cell(c: dict[str, Any]) -> str:
    return (
        f"base {fmt(c['baseline'])}; managed {fmt(c['managed'])}; "
        f"dSum={c['delta_sum_bps']} dT3R={c['delta_t3r_bps']} dMaxLoss={c['delta_max_loss_bps']}"
    )


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 v4 Dissipation Management Backtest",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "`RESEARCH_ONLY_NO_LIVE_NO_PAPER` - this tests post-entry management only.",
        "",
        "## Coverage",
        "",
        f"- rows: `{report['coverage']['rows']}`",
        f"- calibration rows: `{report['coverage']['cal']}`",
        f"- holdout rows: `{report['coverage']['hold']}`",
        f"- split: `{report['split']}`",
        "",
        "## Baseline Hold 4h",
        "",
        f"- all: {fmt(report['overall_baseline']['all'])}",
        f"- cal: {fmt(report['overall_baseline']['cal'])}",
        f"- hold: {fmt(report['overall_baseline']['hold'])}",
        "",
    ]
    primary = report.get("primary_tau120_dual_median")
    if primary:
        lines.extend(
            [
                "## Primary Predefined Rule",
                "",
                f"- config: `{primary['config_id']}`",
                f"- cuts from calibration: `{primary['cuts']}`",
                f"- hold decisions: `{primary['hold_decisions']}`",
                f"- cal: {compare_cell(primary['cal'])}",
                f"- hold: {compare_cell(primary['hold'])}",
                "",
                "### Primary Holdout By Symbol",
                "",
                "| Symbol | N | Baseline | Managed | Decisions |",
                "| --- | ---: | --- | --- | --- |",
            ]
        )
        for row in primary.get("hold_by_symbol", []):
            lines.append(
                f"| `{row['symbol']}` | {row['n']} | {fmt(row['baseline'])} | {fmt(row['managed'])} | `{row['decisions']}` |"
            )
        lines.append("")
    best = report.get("best_by_holdout_delta_t3r")
    if best:
        lines.extend(
            [
                "## Best Exploratory Config By Holdout dT3R",
                "",
                f"- config: `{best['config_id']}`",
                f"- cuts from calibration: `{best['cuts']}`",
                f"- hold decisions: `{best['hold_decisions']}`",
                f"- cal: {compare_cell(best['cal'])}",
                f"- hold: {compare_cell(best['hold'])}",
                "",
                "### Best Holdout By Symbol",
                "",
                "| Symbol | N | Baseline | Managed | Decisions |",
                "| --- | ---: | --- | --- | --- |",
            ]
        )
        for row in best.get("hold_by_symbol", []):
            lines.append(
                f"| `{row['symbol']}` | {row['n']} | {fmt(row['baseline'])} | {fmt(row['managed'])} | `{row['decisions']}` |"
            )
        lines.append("")
    consistent = report.get("consistent_cal_hold_positive") or []
    lines.extend(
        [
            "## Cal + Hold Consistent Improvers",
            "",
            f"Configs with both cal and hold delta_sum > 0 and delta_T3R > 0: `{len(consistent)}`",
            "",
            "| Rank | Config | Cal dSum/dT3R | Hold dSum/dT3R | Hold managed |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for i, cfg in enumerate(consistent[:10], 1):
        lines.append(
            f"| {i} | `{cfg['config_id']}` | {cfg['cal']['delta_sum_bps']}/{cfg['cal']['delta_t3r_bps']} | "
            f"{cfg['hold']['delta_sum_bps']}/{cfg['hold']['delta_t3r_bps']} | {fmt(cfg['hold']['managed'])} |"
        )
    live_diag = report.get("live_v02_lane_diagnostic_on_best_config")
    if live_diag:
        lines.extend(
            [
                "",
                "## Live v0.2 Lane Diagnostic",
                "",
                f"- config applied: `{live_diag['config_id']}`",
                f"- note: {live_diag['note']}",
                f"- rows: `{live_diag['rows']}`",
                f"- decisions: `{live_diag['decisions']}`",
                f"- baseline: {fmt(live_diag['baseline'])}",
                f"- managed: {fmt(live_diag['managed'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## Ranked Configs",
            "",
            "| Rank | Config | Hold decisions | Cal dSum/dT3R | Hold baseline | Hold managed | Hold dSum | Hold dT3R | dMaxLoss |",
            "| ---: | --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for i, cfg in enumerate(report.get("ranked_configs", [])[:30], 1):
        lines.append(
            f"| {i} | `{cfg['config_id']}` | `{cfg['hold_decisions']}` | "
            f"{cfg['cal']['delta_sum_bps']}/{cfg['cal']['delta_t3r_bps']} | "
            f"{fmt(cfg['hold']['baseline'])} | {fmt(cfg['hold']['managed'])} | "
            f"{cfg['hold']['delta_sum_bps']} | {cfg['hold']['delta_t3r_bps']} | {cfg['hold']['delta_max_loss_bps']} |"
        )
    lines.extend(
        [
            "",
            "## Read",
            "",
            "- This is an actual management P&L backtest: rows either exit at tau using bid/ask or hold to 4h.",
            "- Positive descriptive dissipation is not enough; the relevant number is holdout managed total/T3R vs baseline.",
            "- Post-entry features remain illegal as entry inputs. Use only for management/shadow observation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def parse_int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in str(text).split(",") if x.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest S34 v4 dissipation management overlays.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--input-json", type=Path, default=IN_JSON)
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    p.add_argument("--taus-sec", type=parse_int_tuple, default=TAUS_SEC)
    p.add_argument("--replenish-quantiles", type=parse_float_tuple, default=(0.5, 0.75, 0.9))
    p.add_argument("--deceleration-quantiles", type=parse_float_tuple, default=(0.5, 0.75))
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True) as conn:
        report = build_report(conn, args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    args.out_md.write_text(render_md(report), encoding="utf-8")
    print(render_md(report))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
