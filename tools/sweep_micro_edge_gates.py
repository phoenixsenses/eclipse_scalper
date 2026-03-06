from __future__ import annotations

"""
Sweep simple feature gates over debug JSONL outputs (no DB access).

Examples:
  python -m tools.sweep_micro_edge_gates --debug logs/micro_edge_debug_trades.jsonl --sweep "spread<=0.0001,0.0002,0.0005" --sweep "trade_intensity>=1000,2500" --side LONG
  python -m tools.sweep_micro_edge_gates --debug logs/micro_edge_debug_trades.jsonl --sweep "imbalance>=0.1,0.3,0.5" --side ALL
"""

import argparse
import itertools
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def _load_rows(path: Path) -> List[Dict[str, Any]]:
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
        feat = obj.get("feature")
        if not isinstance(feat, dict):
            feat = {}
        net = _to_float(obj.get("net_ret"))
        gross = _to_float(obj.get("gross_ret"))
        cost = _to_float(obj.get("cost"))
        if net is None or gross is None or cost is None:
            continue
        rows.append(
            {
                "rule_name": str(obj.get("rule_name", "")),
                "resolved_side": str(obj.get("resolved_side", "")),
                "feature": feat,
                "net_ret": net,
                "gross_ret": gross,
                "cost": cost,
                "direction_match": _to_boolish(obj.get("direction_match")),
            }
        )
    return rows


def parse_sweep_clause(raw: str) -> Tuple[str, str, List[float]]:
    s = str(raw or "").strip()
    if "<=" in s:
        name, values = s.split("<=", 1)
        op = "<="
    elif ">=" in s:
        name, values = s.split(">=", 1)
        op = ">="
    else:
        raise ValueError(f"invalid --sweep clause '{raw}', expected name<=v1,v2 or name>=v1,v2")
    feature = name.strip()
    if not feature:
        raise ValueError(f"invalid --sweep feature in '{raw}'")
    vals = [float(tok.strip()) for tok in values.split(",") if tok.strip()]
    if not vals:
        raise ValueError(f"invalid --sweep values in '{raw}'")
    return feature, op, vals


def event_passes_config(row: Dict[str, Any], config: Sequence[Tuple[str, str, float]]) -> bool:
    feat = row.get("feature") or {}
    if not isinstance(feat, dict):
        return False
    for k, op, thr in config:
        v = _to_float(feat.get(k))
        if v is None:
            return False
        if op == "<=" and not (v <= thr):
            return False
        if op == ">=" and not (v >= thr):
            return False
    return True


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    xs = list(rows)
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "win_rate": 0.0,
            "avg_net": 0.0,
            "median_net": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "avg_cost": 0.0,
            "avg_gross": 0.0,
            "dir_hit": None,
        }
    net = [float(r["net_ret"]) for r in xs]
    gross = [float(r["gross_ret"]) for r in xs]
    cost = [float(r["cost"]) for r in xs]
    dm = [bool(r["direction_match"]) for r in xs if r.get("direction_match") is not None]
    return {
        "n": n,
        "win_rate": sum(1 for x in net if x > 0.0) / n,
        "avg_net": sum(net) / n,
        "median_net": float(median(net)),
        "p10": _percentile(net, 0.10),
        "p90": _percentile(net, 0.90),
        "avg_cost": sum(cost) / n,
        "avg_gross": sum(gross) / n,
        "dir_hit": (sum(1 for x in dm if x) / len(dm)) if dm else None,
    }


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100.0:6.2f}%"


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:+.6f}"


def _config_str(cfg: Sequence[Tuple[str, str, float]]) -> str:
    return " & ".join(f"{k}{op}{v:g}" for k, op, v in cfg)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep feature gates from micro-edge debug JSONL.")
    p.add_argument("--debug", required=True)
    p.add_argument("--sweep", action="append", default=[], help='Repeatable, e.g. "spread<=0.0001,0.0002"')
    p.add_argument("--side", choices=["LONG", "SHORT", "ALL"], default="ALL")
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--rank-by", choices=["avg_net", "median_net"], default="avg_net")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(str(args.debug))
    if not path.exists():
        print(f"error: debug file not found: {path}")
        return 2
    rows = _load_rows(path)
    if str(args.side) != "ALL":
        rows = [r for r in rows if str(r.get("resolved_side", "")).upper() == str(args.side).upper()]
    clauses = [parse_sweep_clause(x) for x in list(args.sweep or [])]
    if not clauses:
        print("error: at least one --sweep clause is required")
        return 2

    grids: List[List[Tuple[str, str, float]]] = []
    for feat, op, vals in clauses:
        grids.append([(feat, op, float(v)) for v in vals])
    configs = list(itertools.product(*grids))

    scored: List[Dict[str, Any]] = []
    for cfg in configs:
        passed = [r for r in rows if event_passes_config(r, cfg)]
        s = summarize(passed)
        n = int(s["n"] or 0)
        if n < int(args.min_n):
            continue
        scored.append({"config": list(cfg), "config_str": _config_str(cfg), **s})

    scored.sort(key=lambda x: float(x.get(str(args.rank_by), 0.0) or 0.0), reverse=True)
    top = scored[: max(1, int(args.top_k))]
    print(
        f"sweep_micro_edge_gates debug={path} side={args.side} "
        f"configs_total={len(configs)} configs_ranked={len(scored)}"
    )
    print(
        f"{'rank':>4} {'config':56} {'n':>6} {'win_rate':>10} {'avg_net':>12} {'median':>12} "
        f"{'p10':>12} {'p90':>12} {'avg_cost':>12} {'avg_gross':>12} {'dir_hit':>10}"
    )
    for i, r in enumerate(top, start=1):
        print(
            f"{i:4d} {str(r['config_str'])[:56]:56} {int(r['n']):6d} {_fmt_pct(r['win_rate']):>10} "
            f"{_fmt_num(r['avg_net']):>12} {_fmt_num(r['median_net']):>12} "
            f"{_fmt_num(r['p10']):>12} {_fmt_num(r['p90']):>12} {_fmt_num(r['avg_cost']):>12} "
            f"{_fmt_num(r['avg_gross']):>12} {_fmt_pct(r['dir_hit']):>10}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

