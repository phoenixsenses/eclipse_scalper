from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from typing import Dict, List

from tools.micro_edge_lib import extract_best_rule_delta_min_n, parse_int_list
from tools.micro_edge_smoke import _parse_symbols, analyze_symbol, build_json_record


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run micro-edge parameter sweep.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", default="1,5,10")
    p.add_argument("--horizon-sec", default="30,60,120")
    p.add_argument("--min-rule-n", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = _parse_symbols(args.symbols)
    buckets = parse_int_list(args.bucket_sec)
    horizons = parse_int_list(args.horizon_sec)
    try:
        conn = sqlite3.connect(str(args.db), check_same_thread=False)
    except Exception as exc:
        print(f"micro_edge_sweep: unable to open db={args.db} err={exc}")
        return 0
    try:
        all_rows: Dict[str, List[dict]] = defaultdict(list)
        for sym in symbols:
            for b in buckets:
                for h in horizons:
                    rep = analyze_symbol(
                        conn,
                        symbol=sym,
                        lookback_min=int(args.lookback_min),
                        bucket_sec=int(b),
                        horizon_sec=int(h),
                    )
                    rec = build_json_record(rep, int(args.lookback_min), int(b), int(h))
                    rec["min_rule_n"] = int(args.min_rule_n)
                    rec["best_rule_delta_vs_baseline"] = extract_best_rule_delta_min_n(
                        rec, min_rule_n=int(args.min_rule_n)
                    )
                    all_rows[sym].append(rec)
        print("micro_edge_sweep ranked summary:")
        for sym in symbols:
            rows = all_rows.get(sym, [])
            ranked = sorted(
                rows,
                key=lambda r: float(r.get("best_rule_delta_vs_baseline") if r.get("best_rule_delta_vs_baseline") is not None else -1e9),
                reverse=True,
            )
            print(f"\n[{sym}] top 5 by best_rule_delta_vs_baseline")
            if not ranked:
                print("  n/a")
                continue
            for i, r in enumerate(ranked[:5], start=1):
                d = r.get("best_rule_delta_vs_baseline")
                print(
                    f"  {i}. bucket={r['bucket_sec']} horizon={r['horizon_sec']} "
                    f"delta={(f'{float(d):+.4f}' if d is not None else 'n/a')} "
                    f"baseline={(f'{float(r['baseline_hit_rate']):.4f}' if r.get('baseline_hit_rate') is not None else 'n/a')}"
                )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
