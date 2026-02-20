from __future__ import annotations

import argparse
import math
import sqlite3
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple

from execution.passive_execution_simulator import calibrate_passive_model
from tools.micro_edge_backtest import (
    build_passive_calibration_samples,
    compute_regime_bins,
    compute_rule_thresholds,
    load_passive_profiles,
    resolve_symbol_profile,
    simulate_rule_trades,
)
from tools.micro_edge_lib import build_bucket_features
from tools.micro_edge_signal_v2 import enrich_rows_with_v2
from tools.micro_edge_smoke import _load_symbol_trades_and_marks

_ROWS_CACHE: Dict[Tuple[str, str, int, int, str], Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]] = {}


def _parse_seed_list(raw: str | List[int]) -> List[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    out: List[int] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(int(t))
    return out


def _split_ranges(n: int, splits: int) -> List[Tuple[int, int]]:
    k = max(2, int(splits))
    step = max(1, n // k)
    out: List[Tuple[int, int]] = []
    for i in range(1, k):
        a = i * step
        b = (i + 1) * step if i < (k - 1) else n
        if b - a < 50:
            continue
        out.append((a, b))
    return out


def _aggregate_per_split(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        sid = int(r.get("split", 0) or 0)
        by.setdefault(sid, []).append(r)
    out: List[Dict[str, Any]] = []
    for sid in sorted(by):
        grp = by[sid]
        out.append(
            {
                "split": sid,
                "n_seeds": len(grp),
                "filled_n_mean": mean(float(x.get("filled_n", 0)) for x in grp) if grp else 0.0,
                "filled_avg_net_mean": mean(float(x.get("filled_avg_net", 0.0)) for x in grp) if grp else 0.0,
                "filled_p90_net_mean": mean(float(x.get("filled_p90_net", 0.0)) for x in grp) if grp else 0.0,
                "attempt_fill_rate_mean": mean(float(x.get("attempt_fill_rate", 0.0)) for x in grp) if grp else 0.0,
                "net_per_attempt_mean": mean(float(x.get("net_per_attempt", 0.0)) for x in grp) if grp else 0.0,
                "attempts_per_min_mean": mean(float(x.get("attempts_per_min", 0.0)) for x in grp) if grp else 0.0,
                "pass_rate": (sum(1 for x in grp if bool(x.get("pass", False))) / len(grp)) if grp else 0.0,
            }
        )
    return out


def _q_edges(vals: List[float]) -> Tuple[float, float, float]:
    if not vals:
        return (0.0, 0.0, 0.0)
    xs = sorted(float(v) for v in vals)
    n = len(xs)
    def _at(q: float) -> float:
        idx = int(round((n - 1) * q))
        idx = max(0, min(n - 1, idx))
        return float(xs[idx])
    return (_at(0.25), _at(0.50), _at(0.75))


def _bucket_label(v: float, edges: Tuple[float, float, float]) -> str:
    q1, q2, q3 = edges
    x = float(v)
    if x <= q1:
        return "Q1"
    if x <= q2:
        return "Q2"
    if x <= q3:
        return "Q3"
    return "Q4"


def _regime_value(row: Dict[str, Any], mode: str) -> float:
    m = str(mode or "").strip().lower()
    if m == "spread_q":
        return float(row.get("spread") or 0.0)
    if m == "intensity_q":
        return float(row.get("trade_intensity") or 0.0)
    if m == "vol_q":
        v = row.get("micro_volatility")
        if v is not None:
            return float(v)
        return abs(float(row.get("ret_1") or 0.0))
    return 0.0


def validate_pocket_forward(
    db: str,
    symbol: str,
    lookback_min: int,
    bucket_sec: int,
    horizon_sec: int,
    rule: str,
    side: str,
    min_imbalance: float,
    min_trade_intensity: float,
    max_spread: float,
    splits: int,
    seeds: str | List[int],
    min_n: int,
    min_n_frac: float,
    maker_fee_bps: float,
    passive_profile_in: str | None = None,
    passive_max_wait_buckets: int = 0,
    passive_adverse_mult: float = 1.0,
    v2_min_score: float = 0.0,
    v2_min_persistence: float = 0.0,
    v2_min_confidence: float = 0.0,
    min_intensity_strong: float = 0.0,
    min_imbalance_strong: float = 0.0,
    max_spread_tight: float = 0.0,
    regime_bucket: str = "",
) -> Dict[str, Any]:
    seed_list = _parse_seed_list(seeds)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        cache_key = (str(db), str(symbol), int(lookback_min), int(bucket_sec), str(passive_profile_in or ""))
        cached = _ROWS_CACHE.get(cache_key)
        if cached is not None:
            rows, regime_edges, sym_profile = cached
        else:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - int(max(1, lookback_min) * 60 * 1000)
            trades, marks = _load_symbol_trades_and_marks(conn, str(symbol), start_ms=start_ms, end_ms=now_ms)
            rows = build_bucket_features(
                trades,
                marks,
                bucket_sec=max(1, int(bucket_sec)),
                vol_window=max(4, int(60 / max(1, bucket_sec))),
            )
            rows = enrich_rows_with_v2(
                rows,
                bucket_sec=int(bucket_sec),
                cache_key=(str(db), str(symbol), int(lookback_min), int(bucket_sec), str(rule)),
            )
            sym_profile = resolve_symbol_profile(load_passive_profiles(str(passive_profile_in or "")), str(symbol))
            regime_edges = compute_regime_bins(rows)
            _ROWS_CACHE[cache_key] = (rows, regime_edges, sym_profile)
        if len(rows) < 500:
            return {
                "symbol": str(symbol),
                "rows_total": 0,
                "pass_count": 0,
                "pass_rate": 0.0,
                "per_combo": [],
                "per_split": [],
                "insufficient_data": True,
            }
        tox_cfg = sym_profile.get("toxicity_gate", {}) if isinstance(sym_profile.get("toxicity_gate", {}), dict) else {}
        tox_cfg.setdefault("vol_high_threshold", float(regime_edges.get("vol", (None, None, 0.0))[2] or 0.0))
        tox_cfg.setdefault("intensity_high_threshold", float(regime_edges.get("intensity", (None, None, 0.0))[2] or 0.0))
        tox_cfg.setdefault("imbalance_min_threshold", 0.3)
        tox_cfg.setdefault("enabled", True)
        min_b = {"trade_intensity": float(min_trade_intensity), "abs_imbalance": float(min_imbalance)}
        max_b = {"spread": float(max_spread)}
        hold = max(1, int(round(float(horizon_sec) / max(1, int(bucket_sec)))))
        ranges = _split_ranges(len(rows), int(splits))
        _attempt_gate: Dict[str, float] = {}
        if float(min_intensity_strong) > 0.0:
            _attempt_gate["min_trade_intensity_strong"] = float(min_intensity_strong)
        if float(min_imbalance_strong) > 0.0:
            _attempt_gate["min_imbalance_strong"] = float(min_imbalance_strong)
        if float(max_spread_tight) > 0.0:
            _attempt_gate["max_spread_tight"] = float(max_spread_tight)
        results: List[Dict[str, Any]] = []
        by_regime: Dict[str, Dict[str, float]] = {}
        for seed in seed_list:
            for split_id, (a, b) in enumerate(ranges, start=1):
                train_rows = rows[:a]
                val_rows = rows[a:b]
                reg_mode = str(regime_bucket or "").strip().lower()
                reg_edges = (0.0, 0.0, 0.0)
                if reg_mode:
                    train_vals = [_regime_value(r, reg_mode) for r in train_rows]
                    reg_edges = _q_edges(train_vals)
                th_train = compute_rule_thresholds(train_rows)
                th_val = compute_rule_thresholds(val_rows)
                if float(v2_min_score) > 0.0:
                    th_train["v2_min_score"] = float(v2_min_score)
                    th_val["v2_min_score"] = float(v2_min_score)
                if float(v2_min_persistence) > 0.0:
                    th_train["v2_min_persistence"] = float(v2_min_persistence)
                    th_val["v2_min_persistence"] = float(v2_min_persistence)
                if float(v2_min_confidence) > 0.0:
                    th_train["v2_min_confidence"] = float(v2_min_confidence)
                    th_val["v2_min_confidence"] = float(v2_min_confidence)
                samples = build_passive_calibration_samples(
                    rows=train_rows,
                    rule_name=str(rule),
                    side=str(side),
                    thresholds=th_train,
                    hold_buckets=hold,
                    min_feature_bounds=min_b,
                    max_feature_bounds=max_b,
                    max_wait_buckets=int(passive_max_wait_buckets),
                )
                pparams = calibrate_passive_model(samples, maker_fee_bps=float(maker_fee_bps), seed=int(seed))
                p_over = sym_profile.get("passive", {}) if isinstance(sym_profile.get("passive", {}), dict) else {}
                pparams.update(p_over)
                pparams["passive_adverse_mult"] = float(passive_adverse_mult)
                sim = simulate_rule_trades(
                    rows=val_rows,
                    rule_name=str(rule),
                    side=str(side),
                    thresholds=th_val,
                    labels=None,
                    hold_buckets=hold,
                    cooldown_buckets=0,
                    fee_bps=0.0,
                    slip_bps=0.0,
                    min_feature_bounds=min_b,
                    max_feature_bounds=max_b,
                    exec_model="passive_realistic",
                    maker_fee_bps=float(maker_fee_bps),
                    maker_penalty_bps=0.0,
                    passive_params=pparams,
                    passive_max_wait_buckets=int(passive_max_wait_buckets),
                    toxicity_cfg=tox_cfg,
                    regime_edges=regime_edges,
                    attempt_gate_bounds=_attempt_gate or None,
                )
                fo = sim.get("filled_only_metrics", {})
                al = sim.get("attempt_level_metrics", {})
                n = int(fo.get("n", 0) or 0)
                val_rows_n = int(b - a)
                frac_component = int(math.ceil(float(min_n_frac) * float(val_rows_n)))
                effective_min_n = max(int(min_n), frac_component)
                avg_net = float(fo.get("avg_net", 0.0))
                p90_net = float(fo.get("p90_net", 0.0))
                # capacity fields — derived from attempt_level_metrics; no wall-clock dependency
                _val_attempts = int(al.get("n_attempts", 0))
                _net_per_attempt = float(al.get("net_per_attempt", 0.0))
                _duration_min = max(1e-9, val_rows_n * int(bucket_sec) / 60.0)
                _attempts_per_min = _val_attempts / _duration_min
                _pre_gate_n = int(al.get("n_signals_before_gate", _val_attempts))
                if reg_mode:
                    for ar in list(sim.get("attempt_rows", [])):
                        try:
                            si = int(ar.get("signal_idx", -1))
                        except Exception:
                            si = -1
                        if si < 0 or si >= len(val_rows):
                            continue
                        rv = _regime_value(val_rows[si], reg_mode)
                        lab = _bucket_label(rv, reg_edges)
                        agg = by_regime.setdefault(lab, {"attempts": 0.0, "filled": 0.0, "net_sum": 0.0})
                        agg["attempts"] += 1.0
                        agg["filled"] += 1.0 if bool(ar.get("filled")) else 0.0
                        agg["net_sum"] += float(ar.get("net_return", 0.0) or 0.0)
                if n < effective_min_n:
                    fail_reason = "insufficient_fills"
                elif avg_net <= 0.0:
                    fail_reason = "avg_net"
                elif p90_net <= 0.0:
                    fail_reason = "p90_net"
                else:
                    fail_reason = "ok"
                pass_flag = fail_reason == "ok"
                results.append(
                    {
                        "seed": int(seed),
                        "split": int(split_id),
                        "train_n": int(a),
                        "val_n_rows": val_rows_n,
                        "frac_min_component": int(frac_component),
                        "effective_min_n": int(effective_min_n),
                        "filled_n": n,
                        "filled_avg_net": avg_net,
                        "filled_p90_net": p90_net,
                        "filled_win_rate": float(fo.get("win_rate", 0.0)),
                        "attempt_fill_rate": float(al.get("fill_rate", 0.0)),
                        "val_attempts": _val_attempts,
                        "val_filled": n,
                        "attempts_per_min": _attempts_per_min,
                        "net_per_attempt": _net_per_attempt,
                        "val_attempts_before_gate": _pre_gate_n,
                        "val_attempts_after_gate": _val_attempts,
                        "val_filled_after_gate": n,
                        "net_per_attempt_after_gate": _net_per_attempt,
                        "fail_reason": str(fail_reason),
                        "pass": pass_flag,
                    }
                )
        total = len(results)
        passes = sum(1 for r in results if bool(r["pass"]))
        eff_vals = [int(r.get("effective_min_n", 0) or 0) for r in results]
        frac_vals = [int(r.get("frac_min_component", 0) or 0) for r in results]
        frac_dom = [1 for r in results if int(r.get("frac_min_component", 0) or 0) > int(min_n)]
        per_regime: List[Dict[str, Any]] = []
        for k in sorted(by_regime):
            a = by_regime[k]
            attempts = int(a.get("attempts", 0.0))
            filled = int(a.get("filled", 0.0))
            net_sum = float(a.get("net_sum", 0.0))
            per_regime.append(
                {
                    "bucket": k,
                    "attempts": attempts,
                    "filled": filled,
                    "attempt_fill_rate": (filled / attempts) if attempts > 0 else 0.0,
                    "net_per_attempt": (net_sum / attempts) if attempts > 0 else 0.0,
                }
            )
        return {
            "symbol": str(symbol),
            "horizon_sec": int(horizon_sec),
            "min_imbalance": float(min_imbalance),
            "min_trade_intensity": float(min_trade_intensity),
            "max_spread": float(max_spread),
            "maker_fee_bps": float(maker_fee_bps),
            "passive_adverse_mult": float(passive_adverse_mult),
            "rows_total": int(total),
            "pass_count": int(passes),
            "pass_rate": (passes / total) if total > 0 else 0.0,
            "insufficient_fill_rate": (sum(1 for r in results if str(r.get("fail_reason")) == "insufficient_fills") / total) if total > 0 else 0.0,
            "effective_min_n_median": int(median(eff_vals)) if eff_vals else 0,
            "frac_min_component_median": int(median(frac_vals)) if frac_vals else 0,
            "min_n_frac_dominance_rate": (sum(frac_dom) / total) if total > 0 else 0.0,
            "per_combo": results,
            "per_split": _aggregate_per_split(results),
            "regime_bucket": str(regime_bucket or ""),
            "per_regime": per_regime,
            "insufficient_data": False,
        }
    finally:
        conn.close()


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward validation for top passive pocket.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=1440)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--horizon-sec", type=int, default=60)
    p.add_argument("--rule", default="intensity_spike_imbalance_cont")
    p.add_argument("--side", default="auto")
    p.add_argument("--min-imbalance", type=float, default=0.50)
    p.add_argument("--min-trade-intensity", type=float, default=2500.0)
    p.add_argument("--max-spread", type=float, default=0.00025)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--seeds", default="11,22,33,44,55")
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--min-n-frac", type=float, default=0.0, help="Dynamic min fills: effective_min_n=max(min_n, ceil(min_n_frac*val_rows)).")
    p.add_argument("--maker-fee-bps", type=float, default=0.5)
    p.add_argument("--passive-adverse-mult", type=float, default=1.0)
    p.add_argument("--passive-max-wait-buckets", type=int, default=0)
    p.add_argument("--v2-min-score", type=float, default=0.0)
    p.add_argument("--v2-min-persistence", type=float, default=0.0)
    p.add_argument("--v2-min-confidence", type=float, default=0.0)
    p.add_argument("--passive-profile-in", default="state/passive_realistic_profiles.json")
    p.add_argument("--out-md", default="reports/PASSIVE_POCKET_FORWARD_VALIDATION.md")
    p.add_argument("--min-intensity-strong", type=float, default=0.0, help="Pre-attempt gate: skip when trade_intensity < this.")
    p.add_argument("--min-imbalance-strong", type=float, default=0.0, help="Pre-attempt gate: skip when |imbalance| < this.")
    p.add_argument("--max-spread-tight", type=float, default=0.0, help="Pre-attempt gate: skip when spread > this.")
    p.add_argument("--regime-bucket", default="", choices=["", "spread_q", "intensity_q", "vol_q"], help="Optional per-regime robustness breakdown.")
    return p.parse_args()


def main() -> int:
    args = _args()
    res = validate_pocket_forward(
        db=str(args.db),
        symbol=str(args.symbol),
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        horizon_sec=int(args.horizon_sec),
        rule=str(args.rule),
        side=str(args.side),
        min_imbalance=float(args.min_imbalance),
        min_trade_intensity=float(args.min_trade_intensity),
        max_spread=float(args.max_spread),
        splits=int(args.splits),
        seeds=str(args.seeds),
        min_n=int(args.min_n),
        min_n_frac=float(args.min_n_frac),
        maker_fee_bps=float(args.maker_fee_bps),
        passive_profile_in=str(args.passive_profile_in),
        passive_max_wait_buckets=int(args.passive_max_wait_buckets),
        passive_adverse_mult=float(args.passive_adverse_mult),
        v2_min_score=float(args.v2_min_score),
        v2_min_persistence=float(args.v2_min_persistence),
        v2_min_confidence=float(args.v2_min_confidence),
        min_intensity_strong=float(args.min_intensity_strong),
        min_imbalance_strong=float(args.min_imbalance_strong),
        max_spread_tight=float(args.max_spread_tight),
        regime_bucket=str(args.regime_bucket),
    )
    total = int(res.get("rows_total", 0))
    passes = int(res.get("pass_count", 0))
    print(f"forward_validation rows={total} pass={passes}")
    print(
        "effective_min_n formula: "
        f"max(min_n={int(args.min_n)}, ceil(min_n_frac*val_rows)=ceil({float(args.min_n_frac)}*val_rows)); "
        f"median_frac_component={int(res.get('frac_min_component_median', 0))} "
        f"median_effective_min_n={int(res.get('effective_min_n_median', 0))}"
    )
    if float(res.get("min_n_frac_dominance_rate", 0.0)) > 0.0:
        print(
            f"WARNING min_n_frac dominates in {float(res.get('min_n_frac_dominance_rate', 0.0)):.2%} of rows "
            f"(ceil(min_n_frac*val_rows) > min_n)."
        )
    md = Path(str(args.out_md))
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PASSIVE_POCKET_FORWARD_VALIDATION",
        "",
        f"symbol={args.symbol} horizon_sec={args.horizon_sec} min_imbalance={args.min_imbalance} min_trade_intensity={args.min_trade_intensity} max_spread={args.max_spread}",
        f"seeds={_parse_seed_list(args.seeds)} splits={len(res.get('per_split', []))} min_n={args.min_n} min_n_frac={args.min_n_frac} maker_fee_bps={args.maker_fee_bps} passive_adverse_mult={args.passive_adverse_mult} v2_min_score={args.v2_min_score} v2_min_persistence={args.v2_min_persistence} v2_min_confidence={args.v2_min_confidence}",
        f"effective_min_n_formula=max(min_n={int(args.min_n)}, ceil(min_n_frac*val_rows)=ceil({float(args.min_n_frac)}*val_rows)); median_frac_component={int(res.get('frac_min_component_median', 0))} median_effective_min_n={int(res.get('effective_min_n_median', 0))}",
        f"gate: min_intensity_strong={args.min_intensity_strong} min_imbalance_strong={args.min_imbalance_strong} max_spread_tight={args.max_spread_tight}",
        f"regime_bucket={args.regime_bucket or 'none'}",
        "",
        "| seed | split | train_n | val_rows | effective_min_n | filled_n | filled_avg_net | filled_p90_net | filled_win_rate | attempt_fill_rate | net_per_attempt | attempts_per_min | val_before_gate | val_after_gate | fail_reason | pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in res.get("per_combo", []):
        lines.append(
            f"| {r['seed']} | {r['split']} | {r['train_n']} | {r['val_n_rows']} | {r['effective_min_n']} | {r['filled_n']} | "
            f"{r['filled_avg_net']:+.8f} | {r['filled_p90_net']:+.8f} | {r['filled_win_rate']:.2%} | {r['attempt_fill_rate']:.2%} | "
            f"{r.get('net_per_attempt', 0.0):+.6e} | {r.get('attempts_per_min', 0.0):.2f} | "
            f"{r.get('val_attempts_before_gate', r.get('val_attempts', 0))} | {r.get('val_attempts_after_gate', r.get('val_attempts', 0))} | "
            f"{r['fail_reason']} | {'YES' if r['pass'] else 'NO'} |"
        )
    fail_counts: Dict[str, int] = {}
    for r in res.get("per_combo", []):
        k = str(r.get("fail_reason", "unknown"))
        fail_counts[k] = int(fail_counts.get(k, 0)) + 1
    lines += [
        "",
        f"pass_count={passes}/{total}",
        f"pass_rate={float(res.get('pass_rate', 0.0)):.2%}",
        f"insufficient_fill_rate={float(res.get('insufficient_fill_rate', 0.0)):.2%}",
        f"min_n_frac_dominance_rate={float(res.get('min_n_frac_dominance_rate', 0.0)):.2%}",
        "",
        "## Failure Reasons",
    ]
    for k in sorted(fail_counts):
        lines.append(f"- {k}: {fail_counts[k]}")
    lines += [
        "",
        "## Per-Split Capacity",
        "| split | n_seeds | filled_n_mean | attempt_fill_rate_mean | net_per_attempt_mean | attempts_per_min_mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for s in res.get("per_split", []):
        lines.append(
            f"| {s['split']} | {s['n_seeds']} | {float(s['filled_n_mean']):.2f} | {float(s['attempt_fill_rate_mean']):.2%} | "
            f"{float(s.get('net_per_attempt_mean', 0.0)):+.6e} | {float(s.get('attempts_per_min_mean', 0.0)):.2f} |"
        )
    if float(res.get("insufficient_fill_rate", 0.0)) > 0.5:
        lines += ["", "CAPACITY_WARNING: >50% rows failed due to insufficient fills."]
    if float(res.get("min_n_frac_dominance_rate", 0.0)) > 0.0:
        lines += [
            "",
            "MIN_N_FRAC_WARNING: ceil(min_n_frac*val_rows) exceeded min_n for at least one split/seed row.",
        ]
    if str(args.regime_bucket).strip():
        lines += [
            "",
            "## Per-Regime",
            "| bucket | attempts | filled | attempt_fill_rate | net_per_attempt |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in res.get("per_regime", []):
            lines.append(
                f"| {r['bucket']} | {r['attempts']} | {r['filled']} | {float(r['attempt_fill_rate']):.2%} | {float(r['net_per_attempt']):+.6e} |"
            )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
