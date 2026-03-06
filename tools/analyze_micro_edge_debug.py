from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


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


def load_debug_rows(path: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    invalid_json = 0
    invalid_row = 0
    total_lines = 0
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        total_lines += 1
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            invalid_json += 1
            continue
        if not isinstance(obj, dict):
            invalid_row += 1
            continue
        net_ret = _to_float(obj.get("net_ret"))
        gross_ret = _to_float(obj.get("gross_ret"))
        cost = _to_float(obj.get("cost"))
        if net_ret is None or gross_ret is None or cost is None:
            invalid_row += 1
            continue
        feat = obj.get("feature")
        if not isinstance(feat, dict):
            feat = {}
        rows.append(
            {
                "symbol": str(obj.get("symbol", "")),
                "rule_name": str(obj.get("rule_name", "")),
                "resolved_side": str(obj.get("resolved_side", "")),
                "timing": str(obj.get("timing", "")),
                "net_ret": net_ret,
                "gross_ret": gross_ret,
                "cost": cost,
                "direction_match": _to_boolish(obj.get("direction_match")),
                "feature": feat,
            }
        )
    return {
        "rows": rows,
        "invalid_json": invalid_json,
        "invalid_row": invalid_row,
        "total_lines": total_lines,
    }


def summarize_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_l = list(rows)
    n = len(rows_l)
    if n == 0:
        return {
            "n": 0,
            "win_rate": 0.0,
            "avg_net_ret": 0.0,
            "median_net_ret": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "p10_gross": 0.0,
            "p90_gross": 0.0,
            "avg_cost": 0.0,
            "avg_gross_ret": 0.0,
            "dir_match_rate": None,
            "timing_counts": {},
            "p90_net_negative": False,
        }
    net = [float(r["net_ret"]) for r in rows_l]
    gross = [float(r["gross_ret"]) for r in rows_l]
    cost = [float(r["cost"]) for r in rows_l]
    dir_vals = [bool(r["direction_match"]) for r in rows_l if r.get("direction_match") is not None]
    timing_counts: Dict[str, int] = {}
    for r in rows_l:
        k = str(r.get("timing", ""))
        timing_counts[k] = timing_counts.get(k, 0) + 1
    return {
        "n": n,
        "win_rate": sum(1 for x in net if x > 0.0) / n,
        "avg_net_ret": sum(net) / n,
        "median_net_ret": float(median(net)),
        "p10": _percentile(net, 0.10),
        "p90": _percentile(net, 0.90),
        "p10_gross": _percentile(gross, 0.10),
        "p90_gross": _percentile(gross, 0.90),
        "avg_cost": sum(cost) / n,
        "avg_gross_ret": sum(gross) / n,
        "dir_match_rate": (sum(1 for x in dir_vals if x) / len(dir_vals)) if dir_vals else None,
        "timing_counts": timing_counts,
        "p90_net_negative": _percentile(net, 0.90) < 0.0,
    }


def _group_by(rows: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        k = str(r.get(key, ""))
        out.setdefault(k, []).append(r)
    return out


def _group_by_feature(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        feat = r.get("feature") or {}
        if not isinstance(feat, dict):
            continue
        for name, val in feat.items():
            if _to_float(val) is None:
                continue
            out.setdefault(str(name), []).append(r)
    return out


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x * 100.0:6.2f}%"


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:+.6f}"


def _print_table(title: str, rows: List[Dict[str, Any]]) -> None:
    print(title)
    print("  " + "-" * 150)
    print(
        "  "
        f"{'group':28} {'n':>6} {'win_rate':>10} {'avg_net':>12} {'median':>12} "
        f"{'p10':>12} {'p90':>12} {'p10_gross':>12} {'p90_gross':>12} "
        f"{'avg_cost':>12} {'avg_gross':>12} {'dir_hit':>10} {'p90<0':>8}"
    )
    for r in rows:
        print(
            "  "
            f"{str(r['group'])[:28]:28} {int(r['n']):6d} {_fmt_pct(r['win_rate']):>10} "
            f"{_fmt_num(r['avg_net_ret']):>12} {_fmt_num(r['median_net_ret']):>12} "
            f"{_fmt_num(r['p10']):>12} {_fmt_num(r['p90']):>12} {_fmt_num(r['p10_gross']):>12} {_fmt_num(r['p90_gross']):>12} "
            f"{_fmt_num(r['avg_cost']):>12} {_fmt_num(r['avg_gross_ret']):>12} {_fmt_pct(r['dir_match_rate']):>10} "
            f"{('YES' if bool(r.get('p90_net_negative')) else 'NO'):>8}"
        )
    print()


def _to_table_rows(grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, rs in grouped.items():
        s = summarize_rows(rs)
        rows.append({"group": name, **s})
    rows.sort(key=lambda x: int(x["n"]), reverse=True)
    return rows


def _timing_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get("timing", ""))
        out[k] = out.get(k, 0) + 1
    return out


def _write_csv(path: Path, blocks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "group_type",
        "group_value",
        "n",
        "win_rate",
        "avg_net_ret",
        "median_net_ret",
        "p10",
        "p90",
        "p10_gross",
        "p90_gross",
        "avg_cost",
        "avg_gross_ret",
        "dir_match_rate",
        "p90_net_negative",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in blocks:
            w.writerow({k: r.get(k) for k in cols})


def build_report(rows: List[Dict[str, Any]], top_features: int) -> Dict[str, Any]:
    overall = summarize_rows(rows)
    by_side_rows = _to_table_rows(_group_by(rows, "resolved_side"))
    by_rule_rows = _to_table_rows(_group_by(rows, "rule_name"))
    by_feature_rows = _to_table_rows(_group_by_feature(rows))[: max(1, int(top_features))]
    return {
        "overall": overall,
        "by_side": by_side_rows,
        "by_rule": by_rule_rows,
        "by_feature": by_feature_rows,
    }


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze micro-edge debug JSONL.")
    p.add_argument("--debug", required=True, help="Path to debug JSONL.")
    p.add_argument("--out-csv", default="", help="Optional summary CSV output path.")
    p.add_argument("--top-features", type=int, default=20, help="Top feature groups by count.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    debug_path = Path(str(args.debug))
    if not debug_path.exists():
        print(f"error: debug file not found: {debug_path}")
        return 2
    loaded = load_debug_rows(debug_path)
    rows = loaded["rows"]
    report = build_report(rows, top_features=int(args.top_features))

    print(f"analyze_micro_edge_debug file={debug_path}")
    print(
        "parse_stats "
        f"total_lines={loaded['total_lines']} valid_rows={len(rows)} "
        f"invalid_json={loaded['invalid_json']} invalid_row={loaded['invalid_row']}"
    )
    print()

    overall_row = {"group": "overall", **report["overall"]}
    _print_table("Overall", [overall_row])
    if bool(report["overall"].get("p90_net_negative")):
        print("ASSERTION overall_structurally_negative: p90(net_ret) < 0 under current assumptions")
        print()
    print("Timing counts:")
    for k, v in sorted(_timing_counts(rows).items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {k}: {v}")
    print()

    _print_table("By Resolved Side", report["by_side"])
    for r in report["by_side"]:
        if bool(r.get("p90_net_negative")):
            print(f"ASSERTION side_structurally_negative side={r.get('group')}: p90(net_ret) < 0")
    if report["by_side"]:
        print()
    for side_row in report["by_side"]:
        side = str(side_row["group"])
        subset = [r for r in rows if str(r.get("resolved_side", "")) == side]
        print(f"Timing counts side={side}:")
        for k, v in sorted(_timing_counts(subset).items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k}: {v}")
        print()

    _print_table("By Rule Name", report["by_rule"])
    for r in report["by_rule"]:
        if bool(r.get("p90_net_negative")):
            print(f"ASSERTION rule_structurally_negative rule={r.get('group')}: p90(net_ret) < 0")
    if report["by_rule"]:
        print()
    _print_table(f"By Feature (top {int(args.top_features)})", report["by_feature"])

    if str(args.out_csv).strip():
        out_csv = Path(str(args.out_csv))
        csv_rows: List[Dict[str, Any]] = []
        csv_rows.append({"group_type": "overall", "group_value": "overall", **report["overall"]})
        for r in report["by_side"]:
            csv_rows.append({"group_type": "by_side", "group_value": r["group"], **r})
        for r in report["by_rule"]:
            csv_rows.append({"group_type": "by_rule", "group_value": r["group"], **r})
        for r in report["by_feature"]:
            csv_rows.append({"group_type": "by_feature", "group_value": r["group"], **r})
        _write_csv(out_csv, csv_rows)
        print(f"wrote_csv {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
