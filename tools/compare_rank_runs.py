from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def _pct(v: float) -> str:
    return f"{100.0 * float(v):.2f}%"


def _fmt_opt(v: Any, ndigits: int = 6) -> str:
    if v is None:
        return "-"
    try:
        fv = float(v)
        return f"{fv:.{ndigits}f}"
    except Exception:
        return str(v)


def _load_rank(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ranking = payload.get("ranking")
    if not isinstance(ranking, list):
        raise ValueError(f"{path}: missing 'ranking' list")
    return {"path": path, "payload": payload, "ranking": ranking}


def _top_rows(ranking: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    return ranking[:n]


def _stable_pocket_key(row: Dict[str, Any]) -> str:
    # Stable key across rank outputs for the same pocket definition.
    return (
        f"{str(row.get('symbol', ''))}|"
        f"{int(_safe_float(row.get('horizon_sec', 0)))}|"
        f"{_safe_float(row.get('min_imbalance', 0.0)):.8f}|"
        f"{_safe_float(row.get('min_trade_intensity', 0.0)):.8f}|"
        f"{_safe_float(row.get('max_spread', 0.0)):.8f}|"
        f"{str(row.get('rule', ''))}"
    )


def _failure_share(ranking: List[Dict[str, Any]]) -> Tuple[str, float]:
    if not ranking:
        return "mixed", 0.0
    c = Counter(str(r.get("failure_reason_top", "mixed")) for r in ranking)
    reason, cnt = c.most_common(1)[0]
    return reason, cnt / float(len(ranking))


def _avg_metrics(rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    return (
        mean(_safe_float(r.get("npa_core", 0.0)) for r in rows),
        mean(_safe_float(r.get("pass_rate_core", 0.0)) for r in rows),
    )


def _buy_sell_delta(runs: List[Dict[str, Any]], top_n: int) -> str:
    buy = None
    sell = None
    for run in runs:
        name = run["path"].name.upper()
        if "BUY" in name and buy is None:
            buy = run
        if "SELL" in name and sell is None:
            sell = run
    if buy is None or sell is None:
        return "BUY/SELL delta: insufficient runs (need at least one BUY and one SELL file name)."

    buy_npa, buy_pass = _avg_metrics(_top_rows(buy["ranking"], top_n))
    sell_npa, sell_pass = _avg_metrics(_top_rows(sell["ranking"], top_n))
    return (
        f"BUY/SELL delta (top-{top_n} mean): "
        f"delta_npa_core={buy_npa - sell_npa:+.6e}, "
        f"delta_pass_rate_core={buy_pass - sell_pass:+.2%}"
    )


def _render_table(rows: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | npa_core | pass_rate_core | failure_reason_top | best_fee_survive | gate_reject_ratio | fill_rate_after_gate | avg_fee_bps | avg_adverse_bps_on_fills | avg_net_return_bps_on_fills |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.get('symbol', '-')} | "
            f"{int(_safe_float(r.get('horizon_sec', 0)))} | "
            f"{_fmt_opt(r.get('min_imbalance'), 2)} | "
            f"{_fmt_opt(r.get('min_trade_intensity'), 0)} | "
            f"{_fmt_opt(r.get('max_spread'), 6)} | "
            f"{_safe_float(r.get('npa_core', 0.0)):+.6e} | "
            f"{_pct(_safe_float(r.get('pass_rate_core', 0.0)))} | "
            f"{str(r.get('failure_reason_top', 'mixed'))} | "
            f"{_fmt_opt(r.get('best_fee_survive'), 2)} | "
            f"{_pct(_safe_float(r.get('gate_reject_ratio', 0.0)))} | "
            f"{_pct(_safe_float(r.get('fill_rate_after_gate', 0.0)))} | "
            f"{_fmt_opt(r.get('avg_fee_bps'), 3)} | "
            f"{_fmt_opt(r.get('avg_adverse_bps_on_fills'), 3)} | "
            f"{_fmt_opt(r.get('avg_net_return_bps_on_fills'), 3)} |"
        )
    return lines


def _diagnosis(run: Dict[str, Any], top_n: int) -> List[str]:
    rows = _top_rows(run["ranking"], top_n)
    dom_reason, dom_share = _failure_share(rows)
    avg_npa, avg_pass = _avg_metrics(rows)
    return [
        f"- dominant_failure_reason_top={dom_reason} ({_pct(dom_share)})",
        f"- top{top_n}_mean_npa_core={avg_npa:+.6e}",
        f"- top{top_n}_mean_pass_rate_core={_pct(avg_pass)}",
    ]


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare rank run JSON outputs side-by-side.")
    p.add_argument("--ins", required=True, help="Comma-separated rank json paths.")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--out-md", default="reports/COMPARE_RANK_RUNS.md")
    p.add_argument("--intersect-only", action="store_true", help="Compare only pockets present in all input files (stable key intersection).")
    return p.parse_args()


def main() -> int:
    args = _args()
    paths = [Path(x) for x in _parse_list(args.ins)]
    if not paths:
        print("no input files")
        return 2

    runs: List[Dict[str, Any]] = []
    for p in paths:
        if not p.exists():
            print(f"missing: {p}")
            return 2
        try:
            runs.append(_load_rank(p))
        except Exception as exc:
            print(f"invalid rank json {p}: {exc}")
            return 2

    intersection_count = None
    if bool(args.intersect_only):
        key_sets = []
        for run in runs:
            keys = {_stable_pocket_key(r) for r in run["ranking"] if isinstance(r, dict)}
            key_sets.append(keys)
        inter = set.intersection(*key_sets) if key_sets else set()
        intersection_count = len(inter)
        print(f"intersect_only enabled: intersection_count={intersection_count}")
        if not inter:
            print("intersection is empty")
            return 2
        for run in runs:
            run["ranking"] = [r for r in run["ranking"] if _stable_pocket_key(r) in inter]

    top_n = max(1, int(args.top_n))
    md: List[str] = ["# COMPARE_RANK_RUNS", ""]
    if intersection_count is not None:
        md.append(f"intersect_only=true intersection_count={int(intersection_count)}")
        md.append("")

    for run in runs:
        rows = _top_rows(run["ranking"], top_n)
        md.append(f"## {run['path'].name}")
        md.append("")
        md.append(f"rows_total={len(run['ranking'])} top_n={len(rows)}")
        md.append("")
        md.extend(_render_table(rows))
        md.append("")
        md.append("Diagnosis")
        md.extend(_diagnosis(run, top_n))
        md.append("")

    md.append("## Cross-Run Diagnosis")
    md.append("")
    md.append(f"- {_buy_sell_delta(runs, top_n)}")
    md.append("")

    out_md = Path(str(args.out_md))
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {out_md}")
    for run in runs:
        rows = _top_rows(run["ranking"], top_n)
        dom_reason, dom_share = _failure_share(rows)
        avg_npa, avg_pass = _avg_metrics(rows)
        print(
            f"{run['path'].name}: top_n={len(rows)} "
            f"dominant={dom_reason}({_pct(dom_share)}) "
            f"mean_npa={avg_npa:+.6e} mean_pass={_pct(avg_pass)}"
        )
    print(_buy_sell_delta(runs, top_n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
