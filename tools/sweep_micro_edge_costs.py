from __future__ import annotations

"""
Cost sensitivity sweep from micro-edge debug JSONL (no DB required).

Examples:
  python -m tools.sweep_micro_edge_costs --debug logs/micro_edge_debug_trades.jsonl --fee-bps 0,2,4,6 --slip-bps 0,1,2,3 --group-by rule_side
  python -m tools.sweep_micro_edge_costs --debug logs/micro_edge_debug_trades.jsonl --group-by overall --side LONG
"""

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

from tools.micro_edge_backtest import compute_trade_cost


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        x = float(v)
    except Exception:
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _to_boolish(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return None


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    if q <= 0.0:
        return xs[0]
    if q >= 1.0:
        return xs[-1]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def parse_bps_list(raw: str) -> List[float]:
    out: List[float] = []
    for tok in str(raw or "").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def load_debug_gross_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        g = _to_float(obj.get("gross_ret"))
        if g is None:
            continue
        rows.append(
            {
                "rule_name": str(obj.get("rule_name", "")),
                "resolved_side": str(obj.get("resolved_side", "")).upper(),
                "gross_ret": float(g),
                "direction_match": _to_boolish(obj.get("direction_match")),
            }
        )
    return rows


def group_key(row: Dict[str, Any], group_by: str) -> str:
    if group_by == "overall":
        return "overall"
    if group_by == "side":
        return str(row.get("resolved_side", ""))
    if group_by == "rule":
        return str(row.get("rule_name", ""))
    return f"{row.get('rule_name', '')}|{row.get('resolved_side', '')}"


def summarize_with_cost(rows: Iterable[Dict[str, Any]], fee_bps: float, slip_bps: float) -> Dict[str, Any]:
    xs = list(rows)
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "win_rate": 0.0,
            "avg_net_ret": 0.0,
            "median_net_ret": 0.0,
            "p10_net_ret": 0.0,
            "p90_net_ret": 0.0,
            "avg_cost": 0.0,
            "avg_gross_ret": 0.0,
            "dir_hit": None,
            "break_even_cost_bps_total": 0.0,
            "p90_net_negative": False,
        }
    cost = compute_trade_cost(float(fee_bps), float(slip_bps))
    gross = [float(r["gross_ret"]) for r in xs]
    net = [g - cost for g in gross]
    dm = [bool(r["direction_match"]) for r in xs if r.get("direction_match") is not None]
    avg_g = sum(gross) / n
    return {
        "n": n,
        "win_rate": sum(1 for v in net if v > 0.0) / n,
        "avg_net_ret": sum(net) / n,
        "median_net_ret": float(median(net)),
        "p10_net_ret": _percentile(net, 0.10),
        "p90_net_ret": _percentile(net, 0.90),
        "avg_cost": cost,
        "avg_gross_ret": avg_g,
        "dir_hit": (sum(1 for v in dm if v) / len(dm)) if dm else None,
        "break_even_cost_bps_total": avg_g * 10000.0,
        "p90_net_negative": _percentile(net, 0.90) < 0.0,
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100.0:6.2f}%"


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:+.6f}"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost sensitivity sweep over debug JSONL.")
    p.add_argument("--debug", required=True)
    p.add_argument("--fee-bps", default="0,2,4,6")
    p.add_argument("--slip-bps", default="0,1,2,3")
    p.add_argument("--group-by", choices=["overall", "side", "rule", "rule_side"], default="rule_side")
    p.add_argument("--side", choices=["ALL", "LONG", "SHORT"], default="ALL")
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rank-by", choices=["avg_net_ret", "median_net_ret"], default="avg_net_ret")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    path = Path(str(args.debug))
    if not path.exists():
        print(f"error: debug file not found: {path}")
        return 2
    rows = load_debug_gross_rows(path)
    if str(args.side) != "ALL":
        side = str(args.side).upper()
        rows = [r for r in rows if str(r.get("resolved_side", "")).upper() == side]
    fees = parse_bps_list(args.fee_bps)
    slips = parse_bps_list(args.slip_bps)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = group_key(r, str(args.group_by))
        grouped.setdefault(k, []).append(r)

    ranked: List[Dict[str, Any]] = []
    for grp, grp_rows in grouped.items():
        for f in fees:
            for s in slips:
                sm = summarize_with_cost(grp_rows, fee_bps=f, slip_bps=s)
                if int(sm["n"]) < int(args.min_n):
                    continue
                ranked.append({"group": grp, "fee_bps": f, "slip_bps": s, **sm})
    ranked.sort(key=lambda x: float(x.get(str(args.rank_by), 0.0) or 0.0), reverse=True)
    top = ranked[: max(1, int(args.top_k))]

    print(
        f"sweep_micro_edge_costs debug={path} group_by={args.group_by} side={args.side} "
        f"grid={len(fees)}x{len(slips)} groups={len(grouped)} ranked={len(ranked)}"
    )
    print(
        f"{'rank':>4} {'group':34} {'fee':>5} {'slip':>5} {'n':>6} {'win_rate':>10} "
        f"{'avg_gross':>12} {'avg_cost':>10} {'avg_net':>12} {'median_net':>12} "
        f"{'p10':>12} {'p90':>12} {'be_bps':>10} {'p90<0':>7}"
    )
    for i, r in enumerate(top, start=1):
        print(
            f"{i:4d} {str(r['group'])[:34]:34} {float(r['fee_bps']):5.1f} {float(r['slip_bps']):5.1f} "
            f"{int(r['n']):6d} {_fmt_pct(r['win_rate']):>10} {_fmt_num(r['avg_gross_ret']):>12} "
            f"{_fmt_num(r['avg_cost']):>10} {_fmt_num(r['avg_net_ret']):>12} {_fmt_num(r['median_net_ret']):>12} "
            f"{_fmt_num(r['p10_net_ret']):>12} {_fmt_num(r['p90_net_ret']):>12} {float(r['break_even_cost_bps_total']):10.2f} "
            f"{('YES' if bool(r['p90_net_negative']) else 'NO'):>7}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
