from __future__ import annotations

import argparse
import json
import math
from statistics import median
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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


def _percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    xs = sorted(vals)
    pos = (len(xs) - 1) * max(0.0, min(1.0, q))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def load_debug_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s:
            continue
        try:
            o = json.loads(s)
        except Exception:
            continue
        if not isinstance(o, dict):
            continue
        net = _to_float(o.get("net_ret"))
        gross = _to_float(o.get("gross_ret"))
        cost = _to_float(o.get("cost"))
        if net is None or gross is None or cost is None:
            continue
        rows.append(
            {
                "ts_bucket": _to_float(o.get("ts_bucket")),
                "net_ret": float(net),
                "gross_ret": float(gross),
                "cost": float(cost),
                "exec_model": str(o.get("exec_model", "")),
                "horizon_sec": o.get("horizon_sec"),
                "spread": _to_float((o.get("feature") or {}).get("spread") if isinstance(o.get("feature"), dict) else None),
                "trade_intensity": _to_float((o.get("feature") or {}).get("trade_intensity") if isinstance(o.get("feature"), dict) else None),
                "micro_volatility": _to_float((o.get("feature") or {}).get("micro_volatility") if isinstance(o.get("feature"), dict) else None),
                "ret_1": _to_float((o.get("feature") or {}).get("ret_1") if isinstance(o.get("feature"), dict) else None),
                "imbalance": _to_float((o.get("feature") or {}).get("imbalance") if isinstance(o.get("feature"), dict) else None),
                "regime_spread_bin": str(o.get("regime_spread_bin", "")),
                "regime_intensity_bin": str(o.get("regime_intensity_bin", "")),
                "regime_vol_bin": str(o.get("regime_vol_bin", "")),
                "regime_imb_bin": str(o.get("regime_imb_bin", "")),
            }
        )
    return rows


def _quantile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(float(v) for v in vals)
    pos = (len(xs) - 1) * max(0.0, min(1.0, float(q)))
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _bin_with_edges(v: Optional[float], edges: List[Optional[float]], labels: List[str]) -> str:
    if v is None:
        return "missing"
    clean = [e for e in edges if e is not None]
    if len(clean) != len(edges):
        return "unknown"
    x = float(v)
    for i, e in enumerate(clean):
        if x <= float(e):
            return labels[i]
    return labels[-1]


def enrich_bins(rows: List[Dict[str, Any]], bins: str) -> None:
    mode = str(bins or "tertiles").lower()
    if mode == "quartiles":
        q = [0.25, 0.50, 0.75]
        labels = ["<=p25", "p25-50", "p50-75", ">p75"]
    else:
        q = [1.0 / 3.0, 2.0 / 3.0]
        labels = ["<=p33", "p33-67", ">p67"]
    spreads = [float(r["spread"]) for r in rows if r.get("spread") is not None]
    ints = [float(r["trade_intensity"]) for r in rows if r.get("trade_intensity") is not None]
    vols: List[float] = []
    for r in rows:
        v = r.get("micro_volatility")
        if v is not None:
            vols.append(float(v))
        elif r.get("ret_1") is not None:
            vols.append(abs(float(r["ret_1"])))
    edges_spread = [_quantile(spreads, x) for x in q]
    edges_int = [_quantile(ints, x) for x in q]
    edges_vol = [_quantile(vols, x) for x in q]
    for r in rows:
        r["regime_spread_bin"] = _bin_with_edges(r.get("spread"), edges_spread, labels)
        r["regime_intensity_bin"] = _bin_with_edges(r.get("trade_intensity"), edges_int, labels)
        vol_v = r.get("micro_volatility")
        if vol_v is None and r.get("ret_1") is not None:
            vol_v = abs(float(r["ret_1"]))
        r["regime_vol_bin"] = _bin_with_edges(vol_v, edges_vol, labels)
        imb = r.get("imbalance")
        if imb is None:
            r["regime_imb_bin"] = "missing"
        else:
            x = float(imb)
            ax = abs(x)
            if ax < 0.3:
                r["regime_imb_bin"] = "abs<0.3"
            elif ax < 0.5:
                r["regime_imb_bin"] = ("+" if x > 0 else "-") + "[0.3,0.5)"
            elif ax < 0.7:
                r["regime_imb_bin"] = ("+" if x > 0 else "-") + "[0.5,0.7)"
            elif ax < 0.9:
                r["regime_imb_bin"] = ("+" if x > 0 else "-") + "[0.7,0.9)"
            else:
                r["regime_imb_bin"] = ("+" if x > 0 else "-") + ">=0.9"


def group_key(row: Dict[str, Any], fields: List[str]) -> str:
    return "|".join(f"{f}={row.get(f, '')}" for f in fields)


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    xs = list(rows)
    n = len(xs)
    if n == 0:
        return {
            "n": 0,
            "avg_gross": 0.0,
            "avg_cost": 0.0,
            "avg_net": 0.0,
            "median_net": 0.0,
            "p10_net": 0.0,
            "p90_net": 0.0,
            "p90_net_negative": False,
            "break_even_bps_total": 0.0,
            "exec_model": "",
            "horizon_sec": "",
        }
    net = [float(r["net_ret"]) for r in xs]
    gross = [float(r["gross_ret"]) for r in xs]
    cost = [float(r["cost"]) for r in xs]
    exec_vals = sorted({str(r.get("exec_model", "")) for r in xs})
    horizon_vals = sorted({str(r.get("horizon_sec", "")) for r in xs})
    avg_g = sum(gross) / n
    p90 = _percentile(net, 0.90)
    return {
        "n": n,
        "avg_gross": avg_g,
        "avg_cost": sum(cost) / n,
        "avg_net": sum(net) / n,
        "median_net": float(median(net)),
        "p10_net": _percentile(net, 0.10),
        "p90_net": p90,
        "p90_net_negative": p90 < 0.0,
        "break_even_bps_total": avg_g * 10000.0,
        "exec_model": exec_vals[0] if len(exec_vals) == 1 else "MIXED",
        "horizon_sec": horizon_vals[0] if len(horizon_vals) == 1 else "MIXED",
    }


def analyze(rows: List[Dict[str, Any]], group_fields: List[str], min_n: int) -> List[Dict[str, Any]]:
    all_stats = group_stats(rows, group_fields)
    return [r for r in all_stats if int(r["n"]) >= int(min_n)]


def group_stats(rows: List[Dict[str, Any]], group_fields: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = group_key(r, group_fields)
        grouped.setdefault(k, []).append(r)
    out: List[Dict[str, Any]] = []
    for k, rs in grouped.items():
        sm = summarize(rs)
        out.append({"group": k, **sm})
    return out


def _fmt_num(x: Any) -> str:
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return "-"


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze micro-edge regimes from debug JSONL.")
    p.add_argument("--debug", required=True)
    p.add_argument(
        "--group-by",
        default="regime_spread_bin,regime_intensity_bin,regime_vol_bin,regime_imb_bin",
        help="Comma-separated regime fields",
    )
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--bins", choices=["tertiles", "quartiles"], default="tertiles")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    path = Path(str(args.debug))
    if not path.exists():
        print(f"error: debug file not found: {path}")
        return 2
    fields = [x.strip() for x in str(args.group_by).split(",") if x.strip()]
    rows = load_debug_rows(path)
    enrich_bins(rows, bins=str(args.bins))
    all_groups = group_stats(rows, group_fields=fields)
    scored = [g for g in all_groups if int(g["n"]) >= int(args.min_n)]
    by_avg = sorted(scored, key=lambda r: float(r.get("avg_net", 0.0)), reverse=True)[: int(args.top_k)]
    by_p90 = sorted(scored, key=lambda r: float(r.get("p90_net", 0.0)), reverse=True)[: int(args.top_k)]

    small = sum(1 for g in all_groups if int(g["n"]) < int(args.min_n))
    ge = sum(1 for g in all_groups if int(g["n"]) >= int(args.min_n))
    print(
        f"analyze_micro_edge_regimes debug={path} rows={len(rows)} bins={args.bins} "
        f"groups_total={len(all_groups)} groups_n_lt_min={small} groups_n_ge_min={ge} min_n={args.min_n}"
    )
    print("TOP_BY_AVG_NET")
    print(
        f"{'group':72} {'n':>6} {'avg_gross':>12} {'avg_cost':>12} {'avg_net':>12} {'median':>12} "
        f"{'p10':>12} {'p90':>12} {'p90<0':>7} {'be_bps':>10} {'exec':>8} {'h':>6}"
    )
    for r in by_avg:
        print(
            f"{str(r['group'])[:72]:72} {int(r['n']):6d} {_fmt_num(r['avg_gross']):>12} {_fmt_num(r['avg_cost']):>12} "
            f"{_fmt_num(r['avg_net']):>12} {_fmt_num(r['median_net']):>12} {_fmt_num(r['p10_net']):>12} {_fmt_num(r['p90_net']):>12} "
            f"{('YES' if bool(r['p90_net_negative']) else 'NO'):>7} {float(r['break_even_bps_total']):10.2f} "
            f"{str(r['exec_model'])[:8]:>8} {str(r['horizon_sec'])[:6]:>6}"
        )
    print("TOP_BY_P90_NET")
    print(
        f"{'group':72} {'n':>6} {'avg_gross':>12} {'avg_cost':>12} {'avg_net':>12} {'median':>12} "
        f"{'p10':>12} {'p90':>12} {'p90<0':>7} {'be_bps':>10} {'exec':>8} {'h':>6}"
    )
    for r in by_p90:
        print(
            f"{str(r['group'])[:72]:72} {int(r['n']):6d} {_fmt_num(r['avg_gross']):>12} {_fmt_num(r['avg_cost']):>12} "
            f"{_fmt_num(r['avg_net']):>12} {_fmt_num(r['median_net']):>12} {_fmt_num(r['p10_net']):>12} {_fmt_num(r['p90_net']):>12} "
            f"{('YES' if bool(r['p90_net_negative']) else 'NO'):>7} {float(r['break_even_bps_total']):10.2f} "
            f"{str(r['exec_model'])[:8]:>8} {str(r['horizon_sec'])[:6]:>6}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
