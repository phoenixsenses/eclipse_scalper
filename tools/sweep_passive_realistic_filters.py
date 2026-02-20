from __future__ import annotations

import argparse
import itertools
import sqlite3
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

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
from tools.micro_edge_smoke import _load_symbol_trades_and_marks, _parse_symbols
from tools.validate_passive_pocket_forward import validate_pocket_forward


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if not t:
            continue
        out.append(float(t))
    return out


def _parse_int_list(raw: str) -> List[int]:
    return [int(v) for v in _parse_float_list(raw)]


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):+.6f}"
    except Exception:
        return str(x)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Passive-realistic filter sweep with validation ranking.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    p.add_argument("--lookback-min", type=int, default=1440)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--horizons", default="30,60,120")
    p.add_argument("--rule", default="intensity_spike_imbalance_cont")
    p.add_argument("--side", default="auto")
    p.add_argument("--min-imbalance-grid", default="0.2,0.3,0.4,0.5")
    p.add_argument("--min-trade-intensity-grid", default="1500,2500,3500")
    p.add_argument("--max-spread-grid", default="0.0003,0.0005")
    p.add_argument("--passive-seed", type=int, default=42)
    p.add_argument("--maker-fee-bps", type=float, default=0.5)
    p.add_argument("--passive-max-wait-buckets", type=int, default=0)
    p.add_argument("--passive-adverse-mult", type=float, default=1.0)
    p.add_argument("--v2-min-score-grid", default="0.0")
    p.add_argument("--v2-min-persistence-grid", default="0.0")
    p.add_argument("--passive-profile-in", default="state/passive_realistic_profiles.json")
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--min-validation-n", type=int, default=30)
    p.add_argument("--splits", type=int, default=4, help="Capacity check splits (forward-style).")
    p.add_argument("--seeds", default="11,22,33,44,55", help="Capacity check seeds (comma list).")
    p.add_argument("--min-n", type=int, default=50, help="Capacity check min filled per split/seed.")
    p.add_argument("--min-n-frac", type=float, default=0.0, help="Capacity check dynamic min filled fraction.")
    p.add_argument("--min-attempt-fill-rate", type=float, default=0.10)
    p.add_argument("--max-insufficient-fill-rate", type=float, default=0.50)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--out-md", default="reports/FILTER_SWEEP_PASSIVE_REALISTIC.md")
    return p.parse_args()


def main() -> int:
    args = _args()
    symbols = _parse_symbols(args.symbols)
    horizons = _parse_int_list(args.horizons)
    min_imb_grid = _parse_float_list(args.min_imbalance_grid)
    min_int_grid = _parse_float_list(args.min_trade_intensity_grid)
    max_spread_grid = _parse_float_list(args.max_spread_grid)
    v2_min_score_grid = _parse_float_list(args.v2_min_score_grid)
    v2_min_persist_grid = _parse_float_list(args.v2_min_persistence_grid)
    profiles = load_passive_profiles(str(args.passive_profile_in))
    conn = sqlite3.connect(str(args.db), check_same_thread=False)
    out_rows: List[Dict[str, Any]] = []
    try:
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(max(1, args.lookback_min) * 60 * 1000)
        for sym in symbols:
            trades, marks = _load_symbol_trades_and_marks(conn, sym, start_ms=start_ms, end_ms=now_ms)
            rows = build_bucket_features(
                trades,
                marks,
                bucket_sec=max(1, int(args.bucket_sec)),
                vol_window=max(4, int(60 / max(1, args.bucket_sec))),
            )
            rows = enrich_rows_with_v2(
                rows,
                bucket_sec=int(args.bucket_sec),
                cache_key=(str(args.db), str(sym), int(args.lookback_min), int(args.bucket_sec), str(args.rule)),
            )
            if len(rows) < 200:
                continue
            split_i = max(100, int(len(rows) * float(args.train_frac)))
            train_rows = rows[:split_i]
            val_rows = rows[split_i:]
            thresholds_train = compute_rule_thresholds(train_rows)
            thresholds_val = compute_rule_thresholds(val_rows)
            regime_edges = compute_regime_bins(rows)
            sym_profile = resolve_symbol_profile(profiles, sym)
            tox_cfg = sym_profile.get("toxicity_gate", {}) if isinstance(sym_profile.get("toxicity_gate", {}), dict) else {}
            tox_cfg.setdefault("vol_high_threshold", float(regime_edges.get("vol", (None, None, 0.0))[2] or 0.0))
            tox_cfg.setdefault("intensity_high_threshold", float(regime_edges.get("intensity", (None, None, 0.0))[2] or 0.0))
            tox_cfg.setdefault("imbalance_min_threshold", 0.3)
            tox_cfg.setdefault("enabled", True)

            score_grid = v2_min_score_grid if str(args.rule) == "micro_edge_v2_passive_alpha" else [0.0]
            persist_grid = v2_min_persist_grid if str(args.rule) == "micro_edge_v2_passive_alpha" else [0.0]
            for h_sec, min_imb, min_int, max_spread, v2_ms, v2_mp in itertools.product(
                horizons, min_imb_grid, min_int_grid, max_spread_grid, score_grid, persist_grid
            ):
                hold = max(1, int(round(float(h_sec) / max(1, int(args.bucket_sec)))))
                min_b = {"trade_intensity": float(min_int), "abs_imbalance": float(min_imb)}
                max_b = {"spread": float(max_spread)}
                samples = build_passive_calibration_samples(
                    rows=train_rows,
                    rule_name=str(args.rule),
                    side=str(args.side),
                    thresholds=thresholds_train,
                    hold_buckets=hold,
                    min_feature_bounds=min_b,
                    max_feature_bounds=max_b,
                    max_wait_buckets=int(args.passive_max_wait_buckets),
                )
                pparams = calibrate_passive_model(samples, maker_fee_bps=float(args.maker_fee_bps), seed=int(args.passive_seed))
                p_over = sym_profile.get("passive", {}) if isinstance(sym_profile.get("passive", {}), dict) else {}
                pparams.update(p_over)
                pparams["passive_adverse_mult"] = float(args.passive_adverse_mult)
                thresholds_local = dict(thresholds_val)
                if float(v2_ms) > 0.0:
                    thresholds_local["v2_min_score"] = float(v2_ms)
                if float(v2_mp) > 0.0:
                    thresholds_local["v2_min_persistence"] = float(v2_mp)
                sim = simulate_rule_trades(
                    rows=val_rows,
                    rule_name=str(args.rule),
                    side=str(args.side),
                    thresholds=thresholds_local,
                    labels=None,
                    hold_buckets=hold,
                    cooldown_buckets=0,
                    fee_bps=4.0,
                    slip_bps=2.0,
                    min_feature_bounds=min_b,
                    max_feature_bounds=max_b,
                    exec_model="passive_realistic",
                    maker_fee_bps=float(args.maker_fee_bps),
                    maker_penalty_bps=0.0,
                    passive_params=pparams,
                    passive_max_wait_buckets=int(args.passive_max_wait_buckets),
                    toxicity_cfg=tox_cfg,
                    regime_edges=regime_edges,
                )
                fo = sim.get("filled_only_metrics", {})
                al = sim.get("attempt_level_metrics", {})
                n = int(fo.get("n", 0) or 0)
                if n < int(args.min_validation_n):
                    continue
                # Capacity-aware forward-style diagnostics (per split/seed)
                vres = validate_pocket_forward(
                    db=str(args.db),
                    symbol=str(sym),
                    lookback_min=int(args.lookback_min),
                    bucket_sec=int(args.bucket_sec),
                    horizon_sec=int(h_sec),
                    rule=str(args.rule),
                    side=str(args.side),
                    min_imbalance=float(min_imb),
                    min_trade_intensity=float(min_int),
                    max_spread=float(max_spread),
                    splits=int(args.splits),
                    seeds=str(args.seeds),
                    min_n=int(args.min_n),
                    min_n_frac=float(args.min_n_frac),
                    maker_fee_bps=float(args.maker_fee_bps),
                    passive_profile_in=str(args.passive_profile_in),
                    passive_max_wait_buckets=int(args.passive_max_wait_buckets),
                    passive_adverse_mult=float(args.passive_adverse_mult),
                    v2_min_score=float(v2_ms),
                    v2_min_persistence=float(v2_mp),
                )
                per_combo = list(vres.get("per_combo", []))
                if per_combo:
                    val_attempts = int(round(sum(float(x.get("val_attempts", 0) or 0.0) for x in per_combo) / len(per_combo)))
                    val_filled = int(round(sum(float(x.get("val_filled", 0) or 0.0) for x in per_combo) / len(per_combo)))
                    cap_attempt_fill_rate = float(median(float(x.get("attempt_fill_rate", 0.0) or 0.0) for x in per_combo))
                    attempts_per_min = float(median(float(x.get("attempts_per_min", 0.0) or 0.0) for x in per_combo))
                    net_per_attempt = float(median(float(x.get("net_per_attempt", 0.0) or 0.0) for x in per_combo))
                else:
                    val_attempts = int(al.get("n_attempts", 0) or 0)
                    val_filled = int(n)
                    cap_attempt_fill_rate = float(al.get("fill_rate", 0.0) or 0.0)
                    attempts_per_min = 0.0
                    net_per_attempt = float(al.get("net_per_attempt", 0.0) or 0.0)
                insuff_raw = vres.get("insufficient_fill_rate", 1.0)
                insufficient_fill_rate = 1.0 if insuff_raw is None else float(insuff_raw)
                cap_ok = bool(
                    float(net_per_attempt) > 0.0
                    and float(cap_attempt_fill_rate) >= float(args.min_attempt_fill_rate)
                    and float(insufficient_fill_rate) <= float(args.max_insufficient_fill_rate)
                )
                pass_flag = bool(
                    float(fo.get("avg_net", 0.0)) > 0.0
                    and float(fo.get("p90_net", 0.0)) > 0.0
                    and cap_ok
                )
                out_rows.append(
                    {
                        "symbol": sym,
                        "horizon_sec": int(h_sec),
                        "min_imbalance": float(min_imb),
                        "min_trade_intensity": float(min_int),
                        "max_spread": float(max_spread),
                        "v2_min_score": float(v2_ms),
                        "v2_min_persistence": float(v2_mp),
                        "filled_n": n,
                        "filled_avg_net": float(fo.get("avg_net", 0.0)),
                        "filled_p90_net": float(fo.get("p90_net", 0.0)),
                        "filled_win_rate": float(fo.get("win_rate", 0.0)),
                        "attempt_n": int(al.get("n_attempts", 0) or 0),
                        "attempt_net_per_attempt": float(al.get("net_per_attempt", 0.0)),
                        "attempt_fill_rate": float(al.get("fill_rate", 0.0)),
                        "val_attempts": int(val_attempts),
                        "val_filled": int(val_filled),
                        "capacity_attempt_fill_rate": float(cap_attempt_fill_rate),
                        "attempts_per_min": float(attempts_per_min),
                        "net_per_attempt": float(net_per_attempt),
                        "insufficient_fill_rate": float(insufficient_fill_rate),
                        "cap_ok": bool(cap_ok),
                        "pass_flag": bool(pass_flag),
                    }
                )
        out_rows.sort(
            key=lambda r: (
                1 if bool(r.get("pass_flag")) else 0,
                float(r.get("filled_avg_net", 0.0)),
                float(r.get("filled_p90_net", 0.0)),
                int(r.get("filled_n", 0)),
            ),
            reverse=True,
        )
        top = out_rows[: max(1, int(args.top_k))]
        print(
            f"sweep_passive_realistic_filters rows={len(out_rows)} pass={sum(1 for r in out_rows if r.get('pass_flag'))} "
            f"top_k={len(top)}"
        )
        print(
            f"{'symbol':8} {'h':>4} {'imb>=':>7} {'int>=':>8} {'spr<=':>10} "
            f"{'n_fill':>7} {'avg_net':>10} {'p90_net':>10} {'npa':>10} {'cfill%':>8} {'insuf%':>8} {'PASS':>6}"
        )
        for r in top:
            print(
                f"{str(r['symbol']):8} {int(r['horizon_sec']):4d} {float(r['min_imbalance']):7.2f} "
                f"{float(r['min_trade_intensity']):8.0f} {float(r['max_spread']):10.6f} "
                f"{int(r['filled_n']):7d} {float(r['filled_avg_net']):+10.6f} {float(r['filled_p90_net']):+10.6f} "
                f"{float(r.get('net_per_attempt', 0.0)):+10.6f} {float(r.get('capacity_attempt_fill_rate', 0.0)):8.2%} "
                f"{float(r.get('insufficient_fill_rate', 0.0)):8.2%} "
                f"{('YES' if r['pass_flag'] else 'NO'):>6}"
            )
        md = Path(str(args.out_md))
        md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# FILTER_SWEEP_PASSIVE_REALISTIC",
            "",
            f"rows={len(out_rows)} pass={sum(1 for r in out_rows if r.get('pass_flag'))}",
            f"capacity_filter splits={int(args.splits)} seeds={args.seeds} min_n={int(args.min_n)} min_n_frac={float(args.min_n_frac)} min_attempt_fill_rate={float(args.min_attempt_fill_rate)} max_insufficient_fill_rate={float(args.max_insufficient_fill_rate)}",
            "",
            "| symbol | horizon_sec | min_imbalance | min_trade_intensity | max_spread | v2_min_score | v2_min_persistence | filled_n | filled_avg_net | filled_p90_net | net_per_attempt | cap_attempt_fill_rate | insufficient_fill_rate | cap_ok | pass |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
        for r in top:
            lines.append(
                f"| {r['symbol']} | {r['horizon_sec']} | {r['min_imbalance']:.2f} | {r['min_trade_intensity']:.0f} | {r['max_spread']:.6f} | {r['v2_min_score']:.6f} | {r['v2_min_persistence']:.6f} | "
                f"{r['filled_n']} | {r['filled_avg_net']:+.6f} | {r['filled_p90_net']:+.6f} | {r.get('net_per_attempt', 0.0):+.6f} | "
                f"{r.get('capacity_attempt_fill_rate', 0.0):.2%} | {r.get('insufficient_fill_rate', 0.0):.2%} | {'YES' if r.get('cap_ok') else 'NO'} | {'YES' if r['pass_flag'] else 'NO'} |"
            )
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {md}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
