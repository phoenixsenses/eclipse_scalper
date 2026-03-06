from __future__ import annotations

"""
Export top micro-edge gates (research config only, no live wiring).

Example:
  python -m tools.micro_edge_gate_export --db data/microstructure.db --symbols BTCUSDT,ETHUSDT --lookback-min 240 --bucket-sec 1,5,10 --horizon-sec 30,60,120
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.micro_edge_lib import (
    build_bucket_features,
    compute_rule_thresholds,
    evaluate_naive_rules,
    extract_best_rule_delta_min_n,
    filter_rules_min_n,
    parse_int_list,
    utc_now_iso,
)
from tools.micro_edge_smoke import _load_symbol_trades_and_marks, _parse_symbols


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export micro-edge gate config from top sweep configs.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", default="1,5,10")
    p.add_argument("--horizon-sec", default="30,60,120")
    p.add_argument("--min-rule-n", type=int, default=100)
    p.add_argument("--cooldown-buckets", type=int, default=0)
    p.add_argument("--out", default="state/micro_edge_gates.json")
    return p.parse_args()


def _choose_best_rule(rules: Dict[str, Dict[str, Any]], min_rule_n: int) -> tuple[Optional[str], Optional[float]]:
    best_name = None
    best_delta = None
    for k, v in filter_rules_min_n(rules, min_rule_n).items():
        d = v.get("delta_vs_baseline")
        if d is None:
            continue
        dv = float(d)
        if best_delta is None or dv > best_delta:
            best_delta = dv
            best_name = k
    return best_name, best_delta


def main() -> int:
    args = _parse_args()
    symbols = _parse_symbols(args.symbols)
    buckets = parse_int_list(args.bucket_sec)
    horizons = parse_int_list(args.horizon_sec)
    out_path = Path(str(args.out))
    try:
        conn = sqlite3.connect(str(args.db), check_same_thread=False)
    except Exception as exc:
        print(f"micro_edge_gate_export: unable to open db={args.db} err={exc}")
        return 0
    try:
        gates: Dict[str, Any] = {}
        for sym in symbols:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - int(max(1, args.lookback_min) * 60 * 1000)
            trades, marks = _load_symbol_trades_and_marks(conn, sym, start_ms=start_ms, end_ms=now_ms)
            best_cfg = None
            best_delta = None
            for b in buckets:
                rows = build_bucket_features(trades, marks, bucket_sec=int(b), vol_window=max(4, int(60 / max(1, b))))
                mids = [r.get("mid") for r in rows]
                if len(mids) < 10:
                    continue
                mids_float = [float(m) if m is not None else 0.0 for m in mids]
                for h in horizons:
                    from data.labels.forward_return import direction_label, forward_return

                    horizon_steps = max(1, int(round(float(h) / max(1, b))))
                    fwd = forward_return(mids_float, horizon_steps=horizon_steps)
                    labels = [None if r is None else direction_label(float(r), 0.0002) for r in fwd]
                    lbl = [int(x) for x in labels if x is not None and int(x) != 0]
                    if not lbl:
                        continue
                    up = sum(1 for x in lbl if x > 0)
                    baseline = max(up, len(lbl) - up) / len(lbl)
                    rules = evaluate_naive_rules(rows, labels, baseline_hit_rate=baseline)
                    rule_name, delta = _choose_best_rule(rules, int(args.min_rule_n))
                    if rule_name is None or delta is None:
                        continue
                    if best_delta is None or float(delta) > float(best_delta):
                        thresholds = compute_rule_thresholds(rows)
                        best_delta = float(delta)
                        best_cfg = {
                            "bucket_sec": int(b),
                            "horizon_sec": int(h),
                            "rule_name": rule_name,
                            "thresholds": thresholds,
                            "min_rule_n": int(args.min_rule_n),
                            "cooldown_buckets": int(args.cooldown_buckets),
                            "delta_vs_baseline": float(delta),
                            "rules_filtered": filter_rules_min_n(rules, int(args.min_rule_n)),
                        }
            if best_cfg:
                gates[sym] = best_cfg
        payload = {
            "generated_utc": utc_now_iso(),
            "lookback_min": int(args.lookback_min),
            "symbols": gates,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"micro_edge_gate_export wrote {out_path}")
        print(f"symbols_exported={len(gates)}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
