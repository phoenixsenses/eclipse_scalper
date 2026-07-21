"""S34 stress-scalp promotion gauntlet.

Runs the pre-live validation checklist for the current stress-reaction scalp
candidate. Research only: no live executor, order logic, size, leverage, config,
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
from statistics import median
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_navigation_full_followup import DEFAULT_DB, mark_at_or_after, r1, r3, summary  # noqa: E402
from tools.s34_stress_reaction_deep_tests import BASE_FEE_BPS, bracket_outcome, fixed_horizon, mark_series  # noqa: E402
from tools.s34_stress_reaction_gauntlet import (  # noqa: E402
    anatomy,
    build_candidates,
    enrich_chain_counts,
    eval_candidate,
    max_stat_permutation,
    non_overlap_eval,
    prepare_rows,
)

OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_STRESS_SCALP_PROMOTION_GAUNTLET.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_STRESS_SCALP_PROMOTION_GAUNTLET.md"

PRIMARY_NAME = "S3_BTC75_VLT50_CHAIN3_REV_TP200_SL40_20M"
HORIZON_SEC = 1200
TP_BPS = 200.0
SL_BPS = 40.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ts(row: dict[str, Any]) -> int:
    return int(row.get("signal_ts_ms") or 0)


def stress3(row: dict[str, Any]) -> bool:
    return int(row.get("stress_score") or 0) >= 3


def primary_selector(row: dict[str, Any]) -> bool:
    return (
        stress3(row)
        and float(row.get("btc4h_bps") or 0.0) < -75.0
        and float(row.get("vdepth_bps") or 0.0) < 50.0
        and int(row.get("chain_near_15m_thresholds") or 0) >= 3
    )


def t3r(vals: list[float]) -> float:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(sorted(vals, reverse=True)[3:])) if len(vals) > 3 else float(sum(vals))


def bracket_from_entry(
    conn: sqlite3.Connection,
    *,
    entry_ts_ms: int,
    entry_px: float,
    horizon_sec: int,
    direction: str,
    tp: float,
    sl: float,
    fee_bps: float = BASE_FEE_BPS,
) -> tuple[float | None, str, int | None]:
    series = mark_series(conn, entry_ts_ms, entry_ts_ms + horizon_sec * 1000)
    if not series or entry_px <= 0:
        return None, "NO_SERIES", None
    for t, px in series:
        raw = (float(px) - float(entry_px)) / float(entry_px) * 10_000.0
        pnl = raw if direction == "NORMAL" else -raw
        if pnl >= tp:
            return tp - fee_bps, "TP", int((int(t) - entry_ts_ms) / 1000)
        if pnl <= -sl:
            return -sl - fee_bps, "SL", int((int(t) - entry_ts_ms) / 1000)
    end_ts, end_px = series[-1]
    raw = (float(end_px) - float(entry_px)) / float(entry_px) * 10_000.0
    pnl = raw if direction == "NORMAL" else -raw
    return pnl - fee_bps, "TIME", int((int(end_ts) - entry_ts_ms) / 1000)


def primary_bracket(conn: sqlite3.Connection, row: dict[str, Any], *, tp: float = TP_BPS, sl: float = SL_BPS, horizon_sec: int = HORIZON_SEC, fee_bps: float = BASE_FEE_BPS) -> tuple[float | None, str, int | None]:
    return bracket_outcome(conn, row, horizon_sec=horizon_sec, direction="REVERSE", tp=tp, sl=sl, fee_bps=fee_bps)


def values_for(conn: sqlite3.Connection, rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool] = primary_selector) -> tuple[list[float], dict[str, int]]:
    vals = []
    exits: dict[str, int] = defaultdict(int)
    for row in rows:
        if not selector(row):
            continue
        val, exit_, _ = primary_bracket(conn, row)
        if val is None:
            continue
        vals.append(float(val))
        exits[str(exit_)] += 1
    return vals, dict(exits)


def fold_report(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for fold in sorted({int(r.get("fold") or 0) for r in rows}):
        fold_rows = [r for r in rows if int(r.get("fold") or 0) == fold]
        vals, exits = values_for(conn, fold_rows)
        out[f"fold_{fold}"] = {"matched_n": len([r for r in fold_rows if primary_selector(r)]), "summary": summary(vals), "exits": exits}
    positive_t3r = sum(1 for v in out.values() if float(v["summary"].get("t3r_bps") or 0.0) > 0)
    positive_sum = sum(1 for v in out.values() if float(v["summary"].get("sum_bps") or 0.0) > 0)
    return {
        "folds": out,
        "positive_t3r_folds": positive_t3r,
        "positive_sum_folds": positive_sum,
        "fold_t3r_total": r1(sum(float(v["summary"].get("t3r_bps") or 0.0) for v in out.values())),
        "fold_sum_total": r1(sum(float(v["summary"].get("sum_bps") or 0.0) for v in out.values())),
    }


def exit_robustness(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [
        ("TP150_SL30_15M", 150.0, 30.0, 900),
        ("TP200_SL40_20M", 200.0, 40.0, 1200),
        ("TP250_SL50_30M", 250.0, 50.0, 1800),
        ("TP200_SL30_20M", 200.0, 30.0, 1200),
        ("TP150_SL40_20M", 150.0, 40.0, 1200),
    ]
    out = {}
    for name, tp, sl, sec in variants:
        vals = []
        exits: dict[str, int] = defaultdict(int)
        for row in rows:
            if not primary_selector(row):
                continue
            val, exit_, _ = primary_bracket(conn, row, tp=tp, sl=sl, horizon_sec=sec)
            if val is not None:
                vals.append(float(val))
                exits[str(exit_)] += 1
        out[name] = {"summary": summary(vals), "exits": dict(exits)}
    return out


def fee_sensitivity(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for fee in (0.0, 2.5, 5.0, 8.0, 10.0):
        vals = []
        exits: dict[str, int] = defaultdict(int)
        for row in rows:
            if not primary_selector(row):
                continue
            val, exit_, _ = primary_bracket(conn, row, fee_bps=fee)
            if val is not None:
                vals.append(float(val))
                exits[str(exit_)] += 1
        out[f"fee_{fee:g}bps"] = {"summary": summary(vals), "exits": dict(exits)}
    return out


def passive_short_entry(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    *,
    offset_bps: float,
    wait_sec: int,
    fallback: bool,
) -> tuple[int | None, float | None, str]:
    entry = mark_at_or_after(conn, "ETHUSDT", ts(row))
    if not entry:
        return None, None, "NO_ENTRY"
    entry_ts, entry_px = int(entry[0]), float(entry[1])
    limit_px = entry_px * (1.0 + float(offset_bps) / 10_000.0)
    series = mark_series(conn, entry_ts, entry_ts + int(wait_sec) * 1000)
    for t, px in series:
        if float(px) >= limit_px:
            return int(t), float(limit_px), "PASSIVE_FILL"
    if fallback:
        fb = mark_at_or_after(conn, "ETHUSDT", entry_ts + int(wait_sec) * 1000)
        if fb:
            return int(fb[0]), float(fb[1]), "FALLBACK_TAKER"
    return None, None, "NO_FILL"


def execution_realism(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    target = [r for r in rows if primary_selector(r)]
    out: dict[str, Any] = {}

    # Immediate taker baseline.
    vals, exits = values_for(conn, rows)
    out["taker_immediate"] = {"fill_rate": 1.0 if target else None, "summary": summary(vals), "exits": exits}

    for offset in (5.0, 10.0, 20.0):
        for wait in (15, 30, 60):
            for fallback in (False, True):
                vals2 = []
                exits2: dict[str, int] = defaultdict(int)
                fill_kinds: dict[str, int] = defaultdict(int)
                nofill_counter = []
                for row in target:
                    fill_ts, fill_px, kind = passive_short_entry(conn, row, offset_bps=offset, wait_sec=wait, fallback=fallback)
                    fill_kinds[kind] += 1
                    if fill_ts is None or fill_px is None:
                        counter, _, _ = primary_bracket(conn, row)
                        if counter is not None:
                            nofill_counter.append(float(counter))
                        continue
                    val, exit_, _ = bracket_from_entry(
                        conn,
                        entry_ts_ms=fill_ts,
                        entry_px=fill_px,
                        horizon_sec=HORIZON_SEC,
                        direction="REVERSE",
                        tp=TP_BPS,
                        sl=SL_BPS,
                    )
                    if val is not None:
                        vals2.append(float(val))
                        exits2[str(exit_)] += 1
                fills = len(vals2)
                name = f"{'passive_then_taker' if fallback else 'passive_only'}_off{offset:g}_wait{wait}s"
                out[name] = {
                    "fill_rate": r3(fills / len(target)) if target else None,
                    "fill_kinds": dict(fill_kinds),
                    "summary": summary(vals2),
                    "exits": dict(exits2),
                    "no_fill_counterfactual": summary(nofill_counter),
                }
    return out


def regime_concentration(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in rows:
        if not primary_selector(row):
            continue
        val, exit_, _ = primary_bracket(conn, row)
        if val is None:
            continue
        dt = datetime.fromtimestamp(ts(row) / 1000.0, tz=timezone.utc)
        scored.append((float(val), row, dt.date().isoformat(), int(row.get("fold") or 0), exit_))
    total = sum(v for v, *_ in scored)
    t3r_total = t3r([v for v, *_ in scored])
    by_fold: dict[int, list[float]] = defaultdict(list)
    by_date: dict[str, list[float]] = defaultdict(list)
    by_hour: dict[int, list[float]] = defaultdict(list)
    for v, row, day, fold, _ in scored:
        by_fold[fold].append(v)
        by_date[day].append(v)
        by_hour[datetime.fromtimestamp(ts(row) / 1000.0, tz=timezone.utc).hour].append(v)
    date_rows = [
        {"date": day, "n": len(vals), "sum_bps": r1(sum(vals)), "t3r_bps": r1(t3r(vals))}
        for day, vals in by_date.items()
    ]
    date_rows.sort(key=lambda r: abs(float(r["sum_bps"] or 0.0)), reverse=True)
    fold_rows = {
        f"fold_{fold}": {"n": len(vals), "summary": summary(vals)}
        for fold, vals in sorted(by_fold.items())
    }
    hour_rows = {
        f"hour_{hour:02d}": {"n": len(vals), "summary": summary(vals)}
        for hour, vals in sorted(by_hour.items())
    }
    top_date_share = None
    if total:
        top_date_share = r3(max(abs(sum(vals)) for vals in by_date.values()) / abs(total))
    return {
        "summary": summary([v for v, *_ in scored]),
        "folds": fold_rows,
        "top_dates": date_rows[:10],
        "hours": hour_rows,
        "top_abs_date_sum_share": top_date_share,
        "t3r_total": r1(t3r_total),
        "warning": "candidate appears only in folds with matching stress/BTC regime; needs forward OOS before live",
    }


def big_event_cluster_check(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for row in rows:
        if not primary_selector(row):
            continue
        val, exit_, _ = primary_bracket(conn, row)
        if val is not None:
            scored.append((float(val), row, exit_))
    vals = [v for v, _, _ in scored]
    sorted_vals = sorted(vals, reverse=True)
    top3_removed = sum(sorted_vals[3:]) if len(sorted_vals) > 3 else sum(sorted_vals)
    top10_removed = sum(sorted_vals[10:]) if len(sorted_vals) > 10 else sum(sorted_vals)
    worst_sl = [v for v, _, e in scored if e == "SL"]
    return {
        "summary": summary(vals),
        "top3_removed_sum_bps": r1(top3_removed),
        "top10_removed_sum_bps": r1(top10_removed),
        "sl_summary": summary(worst_sl),
        "best10": [
            {
                "event_id": r.get("event_id"),
                "signal_utc": r.get("signal_utc"),
                "value_bps": r1(v),
                "exit": e,
                "fold": r.get("fold"),
                "vdepth": r.get("vdepth_bps"),
                "btc4h": r.get("btc4h_bps"),
                "chain_thresholds": r.get("chain_near_15m_thresholds"),
            }
            for v, r, e in sorted(scored, key=lambda x: x[0], reverse=True)[:10]
        ],
        "worst10": [
            {
                "event_id": r.get("event_id"),
                "signal_utc": r.get("signal_utc"),
                "value_bps": r1(v),
                "exit": e,
                "fold": r.get("fold"),
                "vdepth": r.get("vdepth_bps"),
                "btc4h": r.get("btc4h_bps"),
                "chain_thresholds": r.get("chain_near_15m_thresholds"),
            }
            for v, r, e in sorted(scored, key=lambda x: x[0])[:10]
        ],
    }


def verdict(result: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    hold = result["causal_holdout"]["summary"]
    wf = result["walkforward"]
    non = result["non_overlap"]["nonoverlap_15m_first"]["summary"]
    perm = result["permutation"]["candidate_p"].get(PRIMARY_NAME, {})
    exits = result["exit_robustness"]
    fees = result["fee_sensitivity"]
    execs = result["execution_realism"]
    regime = result["regime_concentration"]
    big = result["big_winner_loser"]

    checks["causal_holdout"] = float(hold.get("sum_bps") or 0.0) > 0 and float(hold.get("t3r_bps") or 0.0) > 0
    checks["walkforward"] = int(wf.get("positive_t3r_folds") or 0) >= 3 and float(wf.get("fold_t3r_total") or 0.0) > 0
    checks["non_overlap_15m"] = float(non.get("t3r_bps") or 0.0) > 0
    checks["permutation"] = float(perm.get("mc_p") or 1.0) < 0.05
    checks["exit_robustness"] = sum(1 for row in exits.values() if float(row["summary"].get("t3r_bps") or 0.0) > 0) >= 2
    checks["fee_sensitivity"] = float(fees["fee_8bps"]["summary"].get("t3r_bps") or 0.0) > 0
    checks["execution_realism"] = (
        float(execs["taker_immediate"]["summary"].get("t3r_bps") or 0.0) > 0
        and any(
            name.startswith("passive_then_taker") and float(row["summary"].get("t3r_bps") or 0.0) > 0
            for name, row in execs.items()
        )
    )
    checks["big_winner_loser"] = float(big.get("top3_removed_sum_bps") or 0.0) > 0 and float(big["summary"].get("max_loss_bps") or 0.0) >= -60.0
    checks["regime_concentration"] = int(wf.get("positive_t3r_folds") or 0) >= 3 and float(regime.get("top_abs_date_sum_share") or 1.0) < 0.5

    hard_fail_reasons = [k for k, ok in checks.items() if not ok]
    status = "PROMOTE_READY" if not hard_fail_reasons else "SHADOW_ONLY"
    return {"status": status, "checks": checks, "hard_fail_reasons": hard_fail_reasons}


def run() -> dict[str, Any]:
    rows = enrich_chain_counts(prepare_rows())
    candidates = build_candidates()
    primary = next(c for c in candidates if c.name == PRIMARY_NAME)
    final_hold = [r for r in rows if int(r.get("fold") or 0) >= 4]
    with sqlite3.connect(DEFAULT_DB) as conn:
        result: dict[str, Any] = {
            "generated_at_utc": utc_now(),
            "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
            "primary_candidate": PRIMARY_NAME,
            "definition": {
                "selector": "stress_score>=3 AND btc4h_bps<-75 AND vdepth_bps<50 AND chain_near_15m_thresholds>=3",
                "direction": "REVERSE_SHORT",
                "exit": "TP200_SL40_20M",
                "fee_bps": BASE_FEE_BPS,
            },
            "rows_n": len(rows),
            "candidate_n": len([r for r in rows if primary_selector(r)]),
            "causal_holdout": eval_candidate(conn, final_hold, primary),
            "walkforward": fold_report(conn, rows),
            "non_overlap": non_overlap_eval(conn, rows, primary),
            "permutation": max_stat_permutation(conn, final_hold, candidates),
            "exit_robustness": exit_robustness(conn, rows),
            "fee_sensitivity": fee_sensitivity(conn, rows),
            "execution_realism": execution_realism(conn, rows),
            "big_winner_loser": big_event_cluster_check(conn, rows),
            "regime_concentration": regime_concentration(conn, rows),
            "anatomy": anatomy(conn, rows, primary),
        }
    result["promotion_verdict"] = verdict(result)
    return result


def fmt(s: dict[str, Any]) -> str:
    return (
        f"N={s.get('n')} sum={s.get('sum_bps')} med={s.get('median_bps')} "
        f"T3R={s.get('t3r_bps')} tail150={s.get('tail_lte_minus150_n')} maxLoss={s.get('max_loss_bps')}"
    )


def write_report(result: dict[str, Any]) -> None:
    lines = [
        "# S34 Stress Scalp Promotion Gauntlet",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Primary: `{result['primary_candidate']}`",
        "",
        f"Definition: `{result['definition']}`",
        "",
        f"Rows: `{result['rows_n']}`; candidate N: `{result['candidate_n']}`",
        "",
        "## Promotion Verdict",
        "",
        f"Verdict: `{result['promotion_verdict']['status']}`",
        "",
        f"Hard fail reasons: `{result['promotion_verdict']['hard_fail_reasons']}`",
        "",
        "| Check | Pass? |",
        "| --- | ---: |",
    ]
    for name, ok in result["promotion_verdict"]["checks"].items():
        lines.append(f"| `{name}` | `{ok}` |")

    lines.extend(["", "## 1. Causal Holdout", ""])
    lines.append(f"{fmt(result['causal_holdout']['summary'])}; exits `{result['causal_holdout']['exits']}`")

    lines.extend(["", "## 2. Walk-Forward Stability", ""])
    wf = result["walkforward"]
    lines.append(f"Positive T3R folds: `{wf['positive_t3r_folds']}/5`; fold T3R total `{wf['fold_t3r_total']}`")
    lines.append("")
    lines.append("| Fold | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in wf["folds"].items():
        lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## 3. Non-Overlap", ""])
    lines.append("| Policy | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in result["non_overlap"].items():
        lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row.get('exits', {})}` |")

    lines.extend(["", "## 4. Max-Statistic Permutation", ""])
    perm = result["permutation"]
    lines.append(f"95pct max-stat T3R: `{perm['maxstat_95pct_t3r']}`")
    lines.append(f"Primary p: `{perm['candidate_p'][PRIMARY_NAME]}`")

    lines.extend(["", "## 5. Exit Robustness", ""])
    lines.append("| Exit | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in result["exit_robustness"].items():
        lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## 6. Fee Sensitivity", ""])
    lines.append("| Fee | Summary | Exits |")
    lines.append("| --- | --- | --- |")
    for name, row in result["fee_sensitivity"].items():
        lines.append(f"| `{name}` | {fmt(row['summary'])} | `{row['exits']}` |")

    lines.extend(["", "## 7. Execution Realism", ""])
    lines.append("| Model | Fill rate | Summary | Fill kinds | Exits | No-fill counterfactual |")
    lines.append("| --- | ---: | --- | --- | --- | --- |")
    for name, row in result["execution_realism"].items():
        lines.append(
            f"| `{name}` | {row.get('fill_rate')} | {fmt(row['summary'])} | "
            f"`{row.get('fill_kinds', {})}` | `{row.get('exits', {})}` | {fmt(row.get('no_fill_counterfactual', {}))} |"
        )

    lines.extend(["", "## 8. Big Winner / Big Loser", ""])
    big = result["big_winner_loser"]
    lines.append(f"Summary: {fmt(big['summary'])}")
    lines.append(f"Top3 removed sum: `{big['top3_removed_sum_bps']}`; top10 removed sum: `{big['top10_removed_sum_bps']}`")
    lines.append(f"SL summary: {fmt(big['sl_summary'])}")
    lines.append("")
    lines.append("Worst 10:")
    for row in big["worst10"]:
        lines.append(f"- `{row}`")
    lines.append("Best 10:")
    for row in big["best10"]:
        lines.append(f"- `{row}`")

    lines.extend(["", "## 9. Regime Concentration", ""])
    reg = result["regime_concentration"]
    lines.append(f"Summary: {fmt(reg['summary'])}")
    lines.append(f"Top abs date sum share: `{reg['top_abs_date_sum_share']}`")
    lines.append(f"Warning: `{reg['warning']}`")
    lines.append("")
    lines.append("Top dates:")
    for row in reg["top_dates"]:
        lines.append(f"- `{row}`")

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
