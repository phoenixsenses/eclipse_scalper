from __future__ import annotations

import argparse
import math
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.micro_edge_lib import (
    append_jsonl,
    build_bucket_features,
    evaluate_naive_rules,
    filter_rules_min_n,
    signal_aligned_labels,
    utc_now_iso,
)
from tools.run_summary import build_run_summary


def _normalize_rule_summary(rules: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float] | int]]:
    out: Dict[str, Dict[str, Optional[float] | int]] = {}
    for key, value in (rules or {}).items():
        if not isinstance(value, dict):
            continue
        try:
            n = int(value.get("n", 0) or 0)
        except Exception:
            n = 0
        hit_rate = value.get("hit_rate")
        delta = value.get("delta_vs_baseline")
        out[str(key)] = {
            "hit_rate": (None if hit_rate is None else float(hit_rate)),
            "n": n,
            "delta_vs_baseline": (None if delta is None else float(delta)),
        }
    return out


def _parse_symbols(raw: str) -> List[str]:
    return [s.strip().upper() for s in str(raw or "").replace(";", ",").split(",") if s.strip()]


def _corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x)
    deny = sum((b - my) ** 2 for b in y)
    den = math.sqrt(max(0.0, denx * deny))
    if den <= 0:
        return None
    return num / den


def _hit_rate(pred: List[int], actual: List[int]) -> Optional[float]:
    if not pred or not actual or len(pred) != len(actual):
        return None
    n = len(pred)
    if n == 0:
        return None
    return sum(1 for p, a in zip(pred, actual) if p == a) / n


def _majority_baseline(actual: List[int]) -> Optional[float]:
    vals = [a for a in actual if a != 0]
    if not vals:
        return None
    up = sum(1 for a in vals if a > 0)
    down = len(vals) - up
    return max(up, down) / len(vals)


def _load_symbol_trades_marks_and_liqs(
    conn: sqlite3.Connection,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    trades_raw = conn.execute(
        "SELECT ts_ms, price, quantity, is_buyer_maker FROM agg_trades "
        "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms ASC",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    marks_raw = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices "
        "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms ASC",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    liq_raw = conn.execute(
        "SELECT ts_ms, side, quantity, price FROM liquidations "
        "WHERE symbol = ? AND ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms ASC",
        (symbol, int(start_ms), int(end_ms)),
    ).fetchall()
    trades = [
        {
            "ts_ms": int(r[0]),
            "price": float(r[1]),
            "quantity": float(r[2]),
            "is_buyer_maker": int(r[3]),
        }
        for r in trades_raw
    ]
    marks = [{"ts_ms": int(r[0]), "mark_price": float(r[1])} for r in marks_raw]
    liqs = [
        {
            "ts_ms": int(r[0]),
            "side": str(r[1]),
            "quantity": float(r[2]),
            "price": float(r[3]),
        }
        for r in liq_raw
    ]
    return trades, marks, liqs


def _load_symbol_trades_and_marks(
    conn: sqlite3.Connection,
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trades, marks, _ = _load_symbol_trades_marks_and_liqs(conn, symbol, start_ms, end_ms)
    return trades, marks


def analyze_symbol(
    conn: sqlite3.Connection,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    horizon_sec: int,
    threshold: float = 0.0002,
) -> Dict[str, Any]:
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(max(1, lookback_min) * 60 * 1000)
    trades, marks, liqs = _load_symbol_trades_marks_and_liqs(conn, symbol, start_ms=start_ms, end_ms=now_ms)
    feats = build_bucket_features(
        trades=trades,
        marks=marks,
        liqs=liqs,
        bucket_sec=max(1, int(bucket_sec)),
        vol_window=max(4, int(60 / max(1, bucket_sec))),
    )
    mids = [f.get("mid") for f in feats]
    horizon_steps = max(1, int(round(float(horizon_sec) / max(1, bucket_sec))))
    fwd, labels = signal_aligned_labels(mids, horizon_steps=horizon_steps, threshold=float(threshold))

    valid_idx = [i for i, r in enumerate(fwd) if r is not None]
    lbl_valid = [int(labels[i]) for i in valid_idx if labels[i] is not None]
    up = sum(1 for x in lbl_valid if x > 0)
    down = sum(1 for x in lbl_valid if x < 0)
    flat = sum(1 for x in lbl_valid if x == 0)
    baseline_hit_rate = _majority_baseline(lbl_valid)
    feature_corr: Dict[str, Optional[float]] = {}
    for key in ("ret_1", "spread", "imbalance", "trade_intensity", "micro_volatility", "volume", "liquidity_proxy"):
        xs: List[float] = []
        ys: List[float] = []
        for i in valid_idx:
            xv = feats[i].get(key)
            yv = fwd[i]
            if xv is None or yv is None:
                continue
            try:
                xs.append(float(xv))
                ys.append(float(yv))
            except Exception:
                continue
        feature_corr[key] = _corr(xs, ys)

    rules = evaluate_naive_rules(feats, labels, baseline_hit_rate=baseline_hit_rate)

    return {
        "symbol": symbol,
        "raw_count": len(trades) + len(marks),
        "bucket_count": len(feats),
        "sample_count": len(valid_idx),
        "up": up,
        "down": down,
        "flat": flat,
        "baseline_hit_rate": baseline_hit_rate,
        "feature_corr": feature_corr,
        "rules": rules,
        "label_definition": {
            "timing": "signal at t, entry at t+1 mark, exit at t+1+h mark",
            "label": "sign((mark[t+1+h]/mark[t+1])-1) with threshold",
            "horizon_steps": int(horizon_steps),
            "threshold": float(threshold),
            "label_values": {"up": 1, "flat": 0, "down": -1},
            "hit_definition": "rule hit_rate = predicted directional sign matches label sign on non-flat labels",
            "baseline_definition": "majority directional class accuracy over non-flat labels",
        },
        "insufficient": len(valid_idx) < 30,
    }


def build_json_record(
    rep: Dict[str, Any],
    lookback_min: int,
    bucket_sec: int,
    horizon_sec: int,
    min_rule_n: int = 100,
) -> Dict[str, Any]:
    label_definition = dict(rep.get("label_definition") or {})
    label_definition.setdefault("timing", "signal at t, entry at t+1 mark, exit at t+1+h mark")
    label_definition.setdefault("label", "sign((mark[t+1+h]/mark[t+1])-1) with threshold")
    label_definition.setdefault("horizon_steps", int(max(1, round(float(horizon_sec) / max(1, bucket_sec)))))
    label_definition.setdefault("threshold", 0.0002)
    label_definition.setdefault("label_values", {"up": 1, "flat": 0, "down": -1})
    label_definition.setdefault(
        "hit_definition",
        "rule hit_rate = predicted directional sign matches label sign on non-flat labels",
    )
    label_definition.setdefault(
        "baseline_definition",
        "majority directional class accuracy over non-flat labels",
    )
    record = {
        "ts_utc": utc_now_iso(),
        "symbol": rep.get("symbol"),
        "lookback_min": int(lookback_min),
        "bucket_sec": int(bucket_sec),
        "horizon_sec": int(horizon_sec),
        "raw_rows": int(rep.get("raw_count", 0)),
        "bucket_rows": int(rep.get("bucket_count", 0)),
        "label_counts": {
            "up": int(rep.get("up", 0)),
            "down": int(rep.get("down", 0)),
            "flat": int(rep.get("flat", 0)),
        },
        "baseline_hit_rate": rep.get("baseline_hit_rate"),
        "min_rule_n": int(min_rule_n),
        "correlations": rep.get("feature_corr", {}),
        "naive_rules": _normalize_rule_summary(rep.get("rules", {})),
        "label_definition": label_definition,
    }
    record["run_summary"] = build_run_summary(
        run_type="micro_edge_smoke",
        inputs={
            "symbol": record["symbol"],
            "lookback_min": int(lookback_min),
            "bucket_sec": int(bucket_sec),
            "horizon_sec": int(horizon_sec),
            "min_rule_n": int(min_rule_n),
        },
        metrics={
            "raw_rows": record["raw_rows"],
            "bucket_rows": record["bucket_rows"],
            "baseline_hit_rate": record["baseline_hit_rate"],
            "rule_count": len(record["naive_rules"]),
        },
        artifacts={},
    )
    return record


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Microstructure edge smoke test.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=240)
    p.add_argument("--bucket-sec", type=int, default=5)
    p.add_argument("--horizon-sec", type=int, default=60)
    p.add_argument("--min-rule-n", type=int, default=100)
    p.add_argument("--out", default="logs/micro_edge_smoke.jsonl")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    db = str(args.db)
    symbols = _parse_symbols(args.symbols)
    out_path = Path(str(args.out))
    try:
        conn = sqlite3.connect(db, check_same_thread=False)
    except Exception as exc:
        print(f"micro_edge_smoke: unable to open db={db} err={exc}")
        return 0
    try:
        print(
            f"micro_edge_smoke db={db} symbols={symbols} lookback_min={args.lookback_min} "
            f"bucket_sec={args.bucket_sec} horizon_sec={args.horizon_sec}"
        )
        print(
            "label_def: signal@t -> entry@t+1 -> exit@t+1+h; "
            "hit=predicted_direction matches future label sign"
        )
        for sym in symbols:
            rep = analyze_symbol(
                conn,
                symbol=sym,
                lookback_min=int(args.lookback_min),
                bucket_sec=int(args.bucket_sec),
                horizon_sec=int(args.horizon_sec),
            )
            append_jsonl(
                out_path,
                build_json_record(
                    rep=rep,
                    lookback_min=int(args.lookback_min),
                    bucket_sec=int(args.bucket_sec),
                    horizon_sec=int(args.horizon_sec),
                    min_rule_n=int(args.min_rule_n),
                ),
            )
            print(f"\n[{sym}] samples={rep['sample_count']} raw={rep['raw_count']} buckets={rep['bucket_count']}")
            if rep["sample_count"] <= 0:
                print("  insufficient data: no forward-return samples")
                continue
            denom = max(1, rep["up"] + rep["down"] + rep["flat"])
            print(
                f"  class_mix up={rep['up']/denom:.2%} down={rep['down']/denom:.2%} flat={rep['flat']/denom:.2%}"
            )
            print("  correlations:")
            for k, v in rep["feature_corr"].items():
                txt = "n/a" if v is None else f"{v:.4f}"
                print(f"    {k}: {txt}")
            shown_rules = filter_rules_min_n(rep.get("rules", {}), int(args.min_rule_n))
            if not shown_rules:
                print("  rules: n/a")
            else:
                print("  rules:")
                for rname, rv in shown_rules.items():
                    hr = rv.get("hit_rate")
                    bl = rv.get("baseline")
                    print(
                        f"    {rname}: n={rv.get('n', 0)} hit_rate={(f'{hr:.2%}' if hr is not None else 'n/a')} "
                        f"baseline={(f'{bl:.2%}' if bl is not None else 'n/a')}"
                    )
            if rep.get("insufficient"):
                print("  note: insufficient data for robust inference (sample_count < 30)")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
