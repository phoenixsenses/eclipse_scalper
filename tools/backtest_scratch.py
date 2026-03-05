from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from core.fee_model import estimate_fee_bps_for_daily_volume
from tools.micro_edge_backtest import compute_rule_thresholds, simulate_rule_trades
from tools.micro_edge_lib import build_bucket_features
from tools.micro_edge_smoke import _load_symbol_trades_and_marks


def _parse_range(raw: str, step: float = 0.5) -> List[float]:
    s = str(raw or "").strip()
    if "," in s:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    if ":" in s:
        a, b = s.split(":", 1)
        lo = float(a.strip())
        hi = float(b.strip())
        out: List[float] = []
        x = lo
        while x <= hi + 1e-9:
            out.append(round(x, 6))
            x += float(step)
        return out
    return [float(s)] if s else []


def _summarize(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    if not trades:
        return {
            "n": 0.0,
            "mean_net": 0.0,
            "scratch_frac": 0.0,
            "horizon_frac": 0.0,
        }
    n = len(trades)
    nets = [float(t.get("net_return", 0.0) or 0.0) for t in trades]
    sc = sum(1 for t in trades if bool(t.get("scratch_triggered")))
    hz = sum(1 for t in trades if not bool(t.get("scratch_triggered")))
    return {
        "n": float(n),
        "mean_net": float(mean(nets)),
        "scratch_frac": float(sc / n),
        "horizon_frac": float(hz / n),
    }


def _load_rows(db: str, symbol: str, lookback_min: int, bucket_sec: int) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        import time as _time

        now_ms = int(_time.time() * 1000)
        start_ms = now_ms - int(max(1, lookback_min) * 60 * 1000)
        trades, marks = _load_symbol_trades_and_marks(conn, symbol=symbol, start_ms=start_ms, end_ms=now_ms)
    finally:
        conn.close()
    return build_bucket_features(trades, marks, bucket_sec=max(1, int(bucket_sec)))


def run_analysis(
    *,
    db: str,
    symbol: str,
    side: str,
    regime: str,
    lookback_min: int,
    bucket_sec: int,
    horizon_sec: int,
    min_imbalance: float,
    min_trade_intensity: float,
    max_spread: float,
    fee_bps: float,
    slip_bps: float,
    scratch_taker_fee_bps: float,
    scratch_slippage_bps: float,
    exec_model: str,
    adverse_vals: List[float],
    trail_vals: List[float],
) -> Dict[str, Any]:
    rows = _load_rows(db=db, symbol=symbol, lookback_min=lookback_min, bucket_sec=bucket_sec)
    if rows and str(regime).strip().lower() in ("up", "down"):
        from tools.validate_passive_pocket_forward import _add_regime_labels

        _add_regime_labels(rows, window_sec=3600)
        rows = [r for r in rows if str(r.get("_regime_label") or "").upper() == str(regime).upper()]
    thresholds = compute_rule_thresholds(rows)
    hold_buckets = max(1, int(round(float(horizon_sec) / max(1, int(bucket_sec)))))

    base = simulate_rule_trades(
        rows=rows,
        rule_name="intensity_spike_imbalance_cont",
        side=str(side).upper(),
        thresholds=thresholds,
        labels=None,
        hold_buckets=hold_buckets,
        cooldown_buckets=0,
        fee_bps=float(fee_bps),
        slip_bps=float(slip_bps),
        min_feature_bounds={
            "imbalance": float(min_imbalance),
            "trade_intensity": float(min_trade_intensity),
        },
        max_feature_bounds={"spread": float(max_spread)},
        exec_model=str(exec_model),
        bucket_sec=int(bucket_sec),
    )
    base_sum = _summarize(base.get("trades", []))

    adverse_rows: List[Dict[str, Any]] = []
    for adv in adverse_vals:
        sim = simulate_rule_trades(
            rows=rows,
            rule_name="intensity_spike_imbalance_cont",
            side=str(side).upper(),
            thresholds=thresholds,
            labels=None,
            hold_buckets=hold_buckets,
            cooldown_buckets=0,
            fee_bps=float(fee_bps),
            slip_bps=float(slip_bps),
            min_feature_bounds={"imbalance": float(min_imbalance), "trade_intensity": float(min_trade_intensity)},
            max_feature_bounds={"spread": float(max_spread)},
            exec_model=str(exec_model),
            bucket_sec=int(bucket_sec),
            scratch_bps=float(adv),
            scratch_window_sec=10,
            scratch_taker_fee_bps=float(scratch_taker_fee_bps),
            scratch_slippage_bps=float(scratch_slippage_bps),
        )
        ss = _summarize(sim.get("trades", []))
        adverse_rows.append(
            {
                "max_adverse_bps": float(adv),
                "n": int(ss["n"]),
                "mean_net": float(ss["mean_net"]),
                "scratch_frac": float(ss["scratch_frac"]),
                "delta_vs_baseline": float(ss["mean_net"] - base_sum["mean_net"]),
            }
        )

    trailing_rows: List[Dict[str, Any]] = []
    for trail in trail_vals:
        sim = simulate_rule_trades(
            rows=rows,
            rule_name="intensity_spike_imbalance_cont",
            side=str(side).upper(),
            thresholds=thresholds,
            labels=None,
            hold_buckets=hold_buckets,
            cooldown_buckets=0,
            fee_bps=float(fee_bps),
            slip_bps=float(slip_bps),
            min_feature_bounds={"imbalance": float(min_imbalance), "trade_intensity": float(min_trade_intensity)},
            max_feature_bounds={"spread": float(max_spread)},
            exec_model=str(exec_model),
            bucket_sec=int(bucket_sec),
            # Approximate trailing via tighter adverse threshold with short reaction window.
            scratch_bps=float(trail),
            scratch_window_sec=2,
            scratch_taker_fee_bps=float(scratch_taker_fee_bps),
            scratch_slippage_bps=float(scratch_slippage_bps),
        )
        ss = _summarize(sim.get("trades", []))
        trailing_rows.append(
            {
                "trailing_stop_bps_proxy": float(trail),
                "n": int(ss["n"]),
                "mean_net": float(ss["mean_net"]),
                "scratch_frac": float(ss["scratch_frac"]),
                "delta_vs_baseline": float(ss["mean_net"] - base_sum["mean_net"]),
            }
        )

    best_adv = max(adverse_rows, key=lambda r: (float(r["mean_net"]), -float(r["max_adverse_bps"]))) if adverse_rows else None
    best_trail = max(trailing_rows, key=lambda r: (float(r["mean_net"]), -float(r["trailing_stop_bps_proxy"]))) if trailing_rows else None
    return {
        "symbol": symbol,
        "side": str(side).upper(),
        "regime": str(regime).upper(),
        "lookback_min": int(lookback_min),
        "bucket_sec": int(bucket_sec),
        "horizon_sec": int(horizon_sec),
        "min_imbalance": float(min_imbalance),
        "min_trade_intensity": float(min_trade_intensity),
        "max_spread": float(max_spread),
        "scratch_taker_fee_bps": float(scratch_taker_fee_bps),
        "scratch_slippage_bps": float(scratch_slippage_bps),
        "exec_model": str(exec_model),
        "baseline": base_sum,
        "adverse_sweep": adverse_rows,
        "trailing_sweep": trailing_rows,
        "best_adverse": best_adv,
        "best_trailing": best_trail,
    }


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# SCRATCH_ANALYSIS")
    lines.append("")
    lines.append(
        f"symbol={report['symbol']} side={report['side']} regime={report['regime']} "
        f"lookback_min={report['lookback_min']} bucket_sec={report['bucket_sec']} horizon_sec={report['horizon_sec']} "
        f"exec_model={report.get('exec_model', 'taker')}"
    )
    lines.append(
        f"scratch_taker_fee_bps={float(report.get('scratch_taker_fee_bps', 0.0)):.3f} "
        f"scratch_slippage_bps={float(report.get('scratch_slippage_bps', 0.0)):.3f}"
    )
    lines.append(
        f"pocket: imb>={report['min_imbalance']:.3f} int>={report['min_trade_intensity']:.0f} spr<={report['max_spread']:.6f}"
    )
    lines.append("")
    b = report["baseline"]
    lines.append("## Baseline")
    lines.append(
        f"n={int(b['n'])} mean_net={float(b['mean_net']):+.6e} "
        f"scratch_frac={float(b['scratch_frac']):.2%} horizon_frac={float(b['horizon_frac']):.2%}"
    )
    lines.append("")
    lines.append("## Max Adverse Sweep")
    lines.append("| max_adverse_bps | n | mean_net | delta_vs_baseline | scratch_frac |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in report["adverse_sweep"]:
        lines.append(
            f"| {float(r['max_adverse_bps']):.2f} | {int(r['n'])} | {float(r['mean_net']):+.6e} | "
            f"{float(r['delta_vs_baseline']):+.6e} | {float(r['scratch_frac']):.2%} |"
        )
    lines.append("")
    if report.get("best_adverse"):
        ba = report["best_adverse"]
        lines.append(
            f"best_max_adverse_bps={float(ba['max_adverse_bps']):.2f} "
            f"mean_net={float(ba['mean_net']):+.6e}"
        )
    lines.append("")
    lines.append("## Trailing Proxy Sweep")
    lines.append("| trailing_stop_bps_proxy | n | mean_net | delta_vs_baseline | scratch_frac |")
    lines.append("|---:|---:|---:|---:|---:|")
    for r in report["trailing_sweep"]:
        lines.append(
            f"| {float(r['trailing_stop_bps_proxy']):.2f} | {int(r['n'])} | {float(r['mean_net']):+.6e} | "
            f"{float(r['delta_vs_baseline']):+.6e} | {float(r['scratch_frac']):.2%} |"
        )
    lines.append("")
    if report.get("best_trailing"):
        bt = report["best_trailing"]
        lines.append(
            f"best_trailing_stop_bps_proxy={float(bt['trailing_stop_bps_proxy']):.2f} "
            f"mean_net={float(bt['mean_net']):+.6e}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backtest scratch/escape impact on micro-edge pocket.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--side", default="SELL", choices=["BUY", "SELL", "buy", "sell"])
    p.add_argument("--regime", default="UP", choices=["UP", "DOWN", "up", "down", "none"])
    p.add_argument("--lookback-min", type=int, default=13 * 24 * 60)
    p.add_argument("--bucket-sec", type=int, default=1)
    p.add_argument("--horizon-sec", type=int, default=120)
    p.add_argument("--min-imbalance", type=float, default=0.50)
    p.add_argument("--min-trade-intensity", type=float, default=3500.0)
    p.add_argument("--max-spread", type=float, default=0.000300)
    p.add_argument("--fee-bps", type=float, default=0.0)
    p.add_argument("--fee-daily-volume-usd", type=float, default=0.0, help="If >0, infer maker/taker fee tier from estimated daily volume.")
    p.add_argument("--slip-bps", type=float, default=0.0)
    p.add_argument("--scratch-taker-fee-bps", type=float, default=0.0, help="Extra one-way taker fee applied on scratch exits.")
    p.add_argument("--scratch-slippage-bps", type=float, default=0.0, help="Extra one-way slippage applied on scratch exits.")
    p.add_argument("--exec-model", default="passive_realistic", choices=["passive_realistic", "taker"])
    p.add_argument("--adverse-sweep", default="3.0:10.0")
    p.add_argument("--trail-sweep", default="2.0,3.0,4.0,5.0")
    p.add_argument("--out-md", default="reports/SCRATCH_ANALYSIS.md")
    p.add_argument("--out-json", default="reports/SCRATCH_ANALYSIS.json")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    fee_bps = float(args.fee_bps)
    if float(args.fee_daily_volume_usd) > 0:
        fee_est = estimate_fee_bps_for_daily_volume(float(args.fee_daily_volume_usd))
        fee_bps = float(fee_est.get("maker_bps", fee_bps))
        print(
            f"scratch_analysis fee_model tier={int(fee_est.get('tier', 0))} "
            f"maker_bps={float(fee_est.get('maker_bps', fee_bps)):.3f} "
            f"taker_bps={float(fee_est.get('taker_bps', 0.0)):.3f}"
        )
    regime = str(args.regime).upper()
    if regime == "NONE":
        regime = ""
    rep = run_analysis(
        db=str(args.db),
        symbol=str(args.symbol).upper(),
        side=str(args.side).upper(),
        regime=regime,
        lookback_min=int(args.lookback_min),
        bucket_sec=int(args.bucket_sec),
        horizon_sec=int(args.horizon_sec),
        min_imbalance=float(args.min_imbalance),
        min_trade_intensity=float(args.min_trade_intensity),
        max_spread=float(args.max_spread),
        fee_bps=float(fee_bps),
        slip_bps=float(args.slip_bps),
        scratch_taker_fee_bps=float(args.scratch_taker_fee_bps),
        scratch_slippage_bps=float(args.scratch_slippage_bps),
        exec_model=str(args.exec_model),
        adverse_vals=_parse_range(str(args.adverse_sweep), step=0.5),
        trail_vals=_parse_range(str(args.trail_sweep), step=1.0),
    )
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))
    _write_md(out_md, rep)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
    print(
        f"scratch_analysis symbol={rep['symbol']} n={int(rep['baseline']['n'])} "
        f"baseline_mean_net={float(rep['baseline']['mean_net']):+.6e} "
        f"out_md={out_md} out_json={out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
