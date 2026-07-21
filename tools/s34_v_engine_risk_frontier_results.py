"""S34 V Engine risk frontier results.

Research-only:
1. CURRENT_ENV tail break-even sweep.
2. 2h vs 4h exit overlap and synthetic tail stress.
3. Intermediate equity-notional sizing ladder.

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

MIRROR_LEDGER = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
FORWARD_PACK = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_FORWARD_RESEARCH_PACK.json"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_RISK_FRONTIER_RESULTS.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_V_ENGINE_RISK_FRONTIER_RESULTS.md"

START_EQUITY = 35.0
CURRENT_ENV_RATIO = 34.0
SIZING_RATIOS = [1, 2, 5, 10, 15, 20, 25, 30, 34]
TAILS = [-150, -180, -200, -220, -250, -275, -300, -350, -400, -507]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso_ms(text: str) -> int:
    value = str(text).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def r1(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 1)


def r3(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), 3)


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


def closed_rows() -> list[dict[str, Any]]:
    rows = [
        r for r in load_jsonl(MIRROR_LEDGER)
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "FILLED"
        and r.get("net_bps") is not None
    ]
    rows.sort(key=lambda r: str(r.get("signal_utc") or ""))
    return rows


def compound(vals: list[float], ratio: float, *, start: float = START_EQUITY) -> dict[str, Any]:
    equity = float(start)
    peak = equity
    max_dd = 0.0
    ruined_at = None
    min_equity = equity
    for i, bps in enumerate(vals, start=1):
        equity *= 1.0 + float(ratio) * float(bps) / 10_000.0
        min_equity = min(min_equity, equity)
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        if ruined_at is None and equity <= 0:
            ruined_at = i
    return {
        "end_equity": r3(equity),
        "multiple": r3(equity / float(start)),
        "min_equity": r3(min_equity),
        "max_drawdown_usdt": r3(max_dd),
        "max_drawdown_pct": r3(abs(max_dd) / float(start) * 100.0),
        "ruined_at": ruined_at,
    }


def tail_break_even(vals: list[float]) -> dict[str, Any]:
    observed = compound(vals, CURRENT_ENV_RATIO)
    end_before = float(observed["end_equity"])
    # Equity after appending tail T is end_before * (1 + 34*T/10000).
    break_even_tail = -10_000.0 / CURRENT_ENV_RATIO
    zero_profit_tail = ((START_EQUITY / end_before) - 1.0) * 10_000.0 / CURRENT_ENV_RATIO
    sweep = []
    for tail in TAILS:
        out = compound(vals + [float(tail)], CURRENT_ENV_RATIO)
        sweep.append({"tail_bps": tail, **out})
    return {
        "observed_current_env": observed,
        "mathematical_ruin_tail_bps": r1(break_even_tail),
        "tail_that_returns_to_start_equity_bps": r1(zero_profit_tail),
        "sweep": sweep,
        "read": "At 34x equity notional, any single appended tail worse than about -294.1 bps makes equity <= 0.",
    }


def summary(vals: list[float]) -> dict[str, Any]:
    vals = [float(v) for v in vals if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "sum_bps": 0.0, "median_bps": None, "win_rate": None, "t3r_bps": 0.0, "max_loss_bps": None}
    t3r = sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else sum(vals)
    return {
        "n": len(vals),
        "sum_bps": r1(sum(vals)),
        "mean_bps": r1(sum(vals) / len(vals)),
        "median_bps": r1(median(vals)),
        "win_rate": r3(sum(1 for v in vals if v > 0) / len(vals)),
        "t3r_bps": r1(t3r),
        "max_loss_bps": r1(min(vals)),
    }


def exit_overlap(rows: list[dict[str, Any]], hold_sec: int) -> dict[str, Any]:
    intervals = []
    for row in rows:
        start = int(row.get("maker_fill_ts_ms") or parse_iso_ms(row["signal_utc"]))
        intervals.append((start, start + int(hold_sec) * 1000, row.get("signal_utc")))
    intervals.sort()
    overlaps = []
    max_concurrent = 0
    for i, (s, e, label) in enumerate(intervals):
        conc = 1
        for j, (s2, e2, label2) in enumerate(intervals):
            if i == j:
                continue
            if s2 < e and e2 > s:
                conc += 1
        max_concurrent = max(max_concurrent, conc)
        if i > 0 and s < intervals[i - 1][1]:
            overlaps.append({"signal_utc": label, "overlaps_previous": intervals[i - 1][2]})
    blocked = 0
    last_exit = -10**30
    for s, e, _ in intervals:
        if s < last_exit:
            blocked += 1
        else:
            last_exit = e
    return {
        "hold_sec": hold_sec,
        "signals": len(intervals),
        "overlap_n": len(overlaps),
        "would_block_if_max_one_position_n": blocked,
        "max_concurrent": max_concurrent,
        "overlap_examples": overlaps[:10],
    }


def exit_stress() -> dict[str, Any]:
    pack = load_json(FORWARD_PACK, {})
    by_variant = (pack.get("exit_management") or {}).get("by_variant") or {}
    out: dict[str, Any] = {}
    for variant, row in by_variant.items():
        # We only have aggregate variant stats in forward pack. Stress the
        # observed aggregate by appending single tails to approximate fragility.
        base_sum = float(row.get("sum_bps") or 0.0)
        n = int(row.get("n") or 0)
        med = float(row.get("median_bps") or 0.0)
        proxy_vals = [med] * max(n, 1)
        # Preserve aggregate sum by adjusting last proxy value.
        if n > 0:
            proxy_vals[-1] += base_sum - sum(proxy_vals)
        out[variant] = {
            "base": {
                "n": n,
                "sum_bps": row.get("sum_bps"),
                "median_bps": row.get("median_bps"),
                "t3r_bps": row.get("t3r_bps"),
                "max_loss_bps": row.get("max_loss_bps"),
            },
            "append_minus150_current_env": compound(proxy_vals + [-150.0], CURRENT_ENV_RATIO),
            "append_minus300_current_env": compound(proxy_vals + [-300.0], CURRENT_ENV_RATIO),
            "append_minus507_current_env": compound(proxy_vals + [-507.0], CURRENT_ENV_RATIO),
        }
    return out


def exit_overlap_stress(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overlap_2h": exit_overlap(rows, 2 * 3600),
        "overlap_4h": exit_overlap(rows, 4 * 3600),
        "overlap_8h": exit_overlap(rows, 8 * 3600),
        "variant_tail_stress": exit_stress(),
        "read": "Overlap uses maker-fill start and fixed hold windows. Variant tail stress is aggregate/proxy because full per-variant paths are stored in the forward pack JSON.",
    }


def sizing_frontier(vals: list[float]) -> dict[str, Any]:
    rows = []
    for ratio in SIZING_RATIOS:
        observed = compound(vals, float(ratio))
        minus150 = compound(vals + [-150.0], float(ratio))
        minus300 = compound(vals + [-300.0], float(ratio))
        minus507 = compound(vals + [-507.0], float(ratio))
        ruin_tail = -10_000.0 / float(ratio)
        rows.append(
            {
                "ratio": float(ratio),
                "observed_end": observed["end_equity"],
                "observed_multiple": observed["multiple"],
                "append_minus150_end": minus150["end_equity"],
                "append_minus300_end": minus300["end_equity"],
                "append_minus507_end": minus507["end_equity"],
                "single_trade_ruin_tail_bps": r1(ruin_tail),
                "survives_minus300": minus300["ruined_at"] is None and float(minus300["end_equity"]) > 0,
                "survives_minus507": minus507["ruined_at"] is None and float(minus507["end_equity"]) > 0,
            }
        )
    survive_300 = [r for r in rows if r["survives_minus300"]]
    survive_507 = [r for r in rows if r["survives_minus507"]]
    return {
        "rows": rows,
        "max_ratio_survives_appended_minus300": max((r["ratio"] for r in survive_300), default=None),
        "max_ratio_survives_appended_minus507": max((r["ratio"] for r in survive_507), default=None),
        "read": "Ratio is notional/equity. Current env is 34x equity notional.",
    }


def build_report() -> dict[str, Any]:
    rows = closed_rows()
    vals = [float(r["net_bps"]) for r in rows]
    return {
        "generated_at_utc": utc_now(),
        "status": "RESEARCH_ONLY_NO_LIVE_CHANGE",
        "source_n": len(vals),
        "tail_break_even": tail_break_even(vals),
        "exit_overlap_stress": exit_overlap_stress(rows),
        "sizing_frontier": sizing_frontier(vals),
        "read": "No live executor, leverage, size, order logic, or .env changes.",
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# S34 V Engine Risk Frontier Results",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['status']}`. {report['read']}",
        "",
        "## CURRENT_ENV Tail Break-Even",
        "",
        f"- Observed 11-trade CURRENT_ENV end: `{report['tail_break_even']['observed_current_env']['end_equity']}`.",
        f"- Single appended tail that makes equity <= 0: about `{report['tail_break_even']['mathematical_ruin_tail_bps']}` bps.",
        f"- Tail that gives back profit to starting $35: `{report['tail_break_even']['tail_that_returns_to_start_equity_bps']}` bps.",
        "",
        "| Tail | End Equity | Multiple | Ruined At |",
        "| ---: | ---: | ---: | --- |",
    ]
    for row in report["tail_break_even"]["sweep"]:
        lines.append(f"| {row['tail_bps']} | {row['end_equity']} | {row['multiple']} | {row['ruined_at']} |")
    lines.extend([
        "",
        "## Exit Overlap",
        "",
        "| Hold | Signals | Overlaps | Blocked if max-one | Max concurrent |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for key in ("overlap_2h", "overlap_4h", "overlap_8h"):
        row = report["exit_overlap_stress"][key]
        lines.append(
            f"| {int(row['hold_sec']/3600)}h | {row['signals']} | {row['overlap_n']} | "
            f"{row['would_block_if_max_one_position_n']} | {row['max_concurrent']} |"
        )
    lines.extend([
        "",
        "## Intermediate Sizing Frontier",
        "",
        "| Ratio | Observed End | -150 End | -300 End | -507 End | Ruin Tail | Survive -300 | Survive -507 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in report["sizing_frontier"]["rows"]:
        lines.append(
            f"| {row['ratio']} | {row['observed_end']} | {row['append_minus150_end']} | "
            f"{row['append_minus300_end']} | {row['append_minus507_end']} | "
            f"{row['single_trade_ruin_tail_bps']} | {row['survives_minus300']} | {row['survives_minus507']} |"
        )
    lines.extend([
        "",
        "## Exit Variant Tail Stress (CURRENT_ENV proxy)",
        "",
        "| Variant | Base Sum | Base T3R | -150 End | -300 End | -507 End |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for variant, row in report["exit_overlap_stress"]["variant_tail_stress"].items():
        lines.append(
            f"| {variant} | {row['base']['sum_bps']} | {row['base']['t3r_bps']} | "
            f"{row['append_minus150_current_env']['end_equity']} | "
            f"{row['append_minus300_current_env']['end_equity']} | "
            f"{row['append_minus507_current_env']['end_equity']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run S34 V Engine risk frontier results.")
    p.add_argument("--out-json", type=Path, default=OUT_JSON)
    p.add_argument("--out-md", type=Path, default=OUT_MD)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    md = render_md(report)
    args.out_md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
