from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Tuple

from tools.eval_run import run_eval
from tools.replay_strategy import _load_strategy_config, _parse_iso_utc, _parse_symbols


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch)
    return "".join(out) or "x"


def _parse_slices(raw: str) -> List[Tuple[str, str]]:
    segs = [x.strip() for x in str(raw or "").split(";") if x.strip()]
    out: List[Tuple[str, str]] = []
    for seg in segs:
        if "," not in seg:
            raise ValueError(f"invalid slice segment: {seg}")
        a, b = seg.split(",", 1)
        start = str(a).strip()
        end = str(b).strip()
        if _parse_iso_utc(end) <= _parse_iso_utc(start):
            raise ValueError(f"invalid slice range: {seg}")
        out.append((start, end))
    if not out:
        raise ValueError("no slices provided")
    return out


def _iter_auto_slices(start: str, end: str, slice_sec: int) -> List[Tuple[str, str]]:
    from datetime import datetime, timezone

    t0 = _parse_iso_utc(start)
    t1 = _parse_iso_utc(end)
    if t1 <= t0:
        raise ValueError("end must be greater than start")
    if int(slice_sec) <= 0:
        raise ValueError("slice_sec must be > 0")
    out: List[Tuple[str, str]] = []
    cur = t0
    while cur < t1:
        nxt = min(t1, cur + float(slice_sec))
        s = datetime.fromtimestamp(cur, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        e = datetime.fromtimestamp(nxt, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        out.append((s, e))
        cur = nxt
    return out


def _slice_id(i: int, start: str, end: str) -> str:
    return f"slice_{i:03d}_{_safe_name(start)}_{_safe_name(end)}"


def _stability_score(mean_v: float, std_v: float, min_v: float) -> float:
    return float(mean_v - (0.5 * std_v) + (0.2 * min_v))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _coerce_ts(raw: Any) -> float | None:
    try:
        if raw is None:
            return None
        ts = float(raw)
        if ts > 1e12:
            ts /= 1000.0
        return ts
    except Exception:
        return None


def _coerce_price(raw: Any) -> float | None:
    try:
        if raw is None:
            return None
        px = float(raw)
        if px <= 0.0 or math.isnan(px) or math.isinf(px):
            return None
        return px
    except Exception:
        return None


def _slice_price_window(db: Path, symbols: List[str], start: str, end: str) -> tuple[float, float]:
    start_ts = _parse_iso_utc(start)
    end_ts = _parse_iso_utc(end)
    if end_ts <= start_ts:
        return 0.0, 0.0
    if not db.exists():
        return 0.0, 0.0
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT ts_ms, symbol, price FROM agg_trades WHERE ts_ms >= ? AND ts_ms <= ? ORDER BY ts_ms ASC, rowid ASC",
            (int(start_ts * 1000), int(end_ts * 1000)),
        ).fetchall()
        if not rows:
            return 0.0, 0.0
        symbol_set = {s.upper() for s in symbols if str(s).strip()}
        first_px: float | None = None
        last_px: float | None = None
        for ts_ms, sym, price in rows:
            s = str(sym or "").upper()
            if symbol_set and s not in symbol_set:
                continue
            _ = _coerce_ts(ts_ms)
            px = _coerce_price(price)
            if px is None:
                continue
            if first_px is None:
                first_px = px
            last_px = px
        if first_px is None or last_px is None:
            return 0.0, 0.0
        return float(first_px), float(last_px)
    except sqlite3.Error:
        return 0.0, 0.0
    finally:
        conn.close()


def _build_stability(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    pnl_values = [float(r.get("pnl_net_sum", 0.0) or 0.0) for r in rows]
    fill_rates = [float(r.get("decision_to_fill_rate", 0.0) or 0.0) for r in rows]
    pnl_per_fill_values = [float(r.get("pnl_net_per_fill", 0.0) or 0.0) for r in rows]
    slices_count = len(rows)
    pos_slices_count = sum(1 for v in pnl_values if v > 0)
    std_v = pstdev(pnl_values) if slices_count > 1 else 0.0
    stability = {
        "slices_count": slices_count,
        "pos_slices_count": int(pos_slices_count),
        "pos_slices_frac": float(pos_slices_count / slices_count) if slices_count > 0 else 0.0,
        "pnl_net_sum_total": float(sum(pnl_values)),
        "pnl_net_sum_mean": float(mean(pnl_values)) if pnl_values else 0.0,
        "pnl_net_sum_median": float(median(pnl_values)) if pnl_values else 0.0,
        "pnl_net_sum_std": float(std_v),
        "pnl_net_sum_min": float(min(pnl_values)) if pnl_values else 0.0,
        "pnl_net_sum_max": float(max(pnl_values)) if pnl_values else 0.0,
        "worst_pnl_net_per_fill": float(min(pnl_per_fill_values)) if pnl_per_fill_values else 0.0,
        "fill_rate_mean": float(mean(fill_rates)) if fill_rates else 0.0,
    }
    stability["stability_score"] = _stability_score(
        float(stability["pnl_net_sum_mean"]),
        float(stability["pnl_net_sum_std"]),
        float(stability["pnl_net_sum_min"]),
    )
    return stability


def run_walkforward(
    db: Path,
    symbols: List[str],
    strategy: str,
    strategy_config: Dict[str, Any],
    out_dir: Path,
    slices: List[Tuple[str, str]],
    fee_bps: float,
    spread_bps: float,
    qty: float,
    horizon_sec: int,
    top_k: int,
    sort_by: str,
    sort_desc: bool,
) -> Dict[str, Any]:
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    index_rows: List[Dict[str, Any]] = []

    for i, (start, end) in enumerate(slices, start=1):
        sid = _slice_id(i, start, end)
        run_dir = runs_dir / sid
        out = run_eval(
            db=db,
            symbols=symbols,
            start=start,
            end=end,
            strategy=strategy,
            strategy_config=strategy_config,
            run_dir=run_dir,
            fee_bps=float(fee_bps),
            spread_bps=float(spread_bps),
            qty=float(qty),
            horizon_sec=int(horizon_sec),
        )
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        fills_count = int(metrics.get("fills_count", 0) or 0)
        decisions_count = int(metrics.get("decisions_count", 0) or 0)
        pnl_net_sum = float(metrics.get("pnl_net_sum", metrics.get("pnl_sum", 0.0)) or 0.0)
        pnl_net_per_fill = float(pnl_net_sum / fills_count) if fills_count > 0 else 0.0
        fill_rate = float(metrics.get("decision_to_fill_rate", 0.0) or 0.0)
        if decisions_count > 0 and fill_rate <= 0:
            fill_rate = float(fills_count) / float(decisions_count)
        alpha_ok = bool((pnl_net_per_fill >= 0.0) and (fill_rate >= 0.2))
        start_px, end_px = _slice_price_window(db=db, symbols=symbols, start=start, end=end)
        slice_return = float((end_px - start_px) / start_px) if start_px > 0.0 else 0.0
        regime = "up" if end_px > start_px else "down"

        row = {
            "slice_id": sid,
            "start": start,
            "end": end,
            "regime": regime,
            "start_px": start_px,
            "end_px": end_px,
            "slice_return": slice_return,
            "run_dir": f"runs/{sid}",
            "pnl_net_sum": pnl_net_sum,
            "pnl_net_per_fill": pnl_net_per_fill,
            "fills_count": fills_count,
            "decision_to_fill_rate": fill_rate,
            "fee_sum": float(metrics.get("fee_sum", 0.0) or 0.0),
            "spread_cost_est_sum": float(metrics.get("spread_cost_est_sum", 0.0) or 0.0),
            "avg_adverse_samples": float(metrics.get("avg_adverse_samples", 0.0) or 0.0),
            "skipped_count": int(metrics.get("skipped_count", 0) or 0),
            "alpha_ok": int(alpha_ok),
            "events_replayed": int(out.get("events_replayed", 0) or 0),
            "decisions_count": decisions_count,
        }
        index_rows.append(row)
    stability = _build_stability(index_rows)
    rows_up = [r for r in index_rows if str(r.get("regime")) == "up"]
    rows_down = [r for r in index_rows if str(r.get("regime")) == "down"]
    stability_up = _build_stability(rows_up)
    stability_down = _build_stability(rows_down)
    score_all = _safe_float(stability.get("stability_score"), 0.0)
    score_up = _safe_float(stability_up.get("stability_score"), score_all)
    score_down = _safe_float(stability_down.get("stability_score"), score_all)
    combined_score = float((0.5 * score_all) + (0.25 * score_up) + (0.25 * score_down))
    stability["combined_score"] = combined_score

    _write_index_csv(out_dir / "index.csv", index_rows)
    _write_stability_csv(out_dir / "stability.csv", stability)
    _write_stability_csv(out_dir / "stability_all.csv", stability)
    _write_stability_csv(out_dir / "stability_up.csv", {**stability_up, "regime": "up"})
    _write_stability_csv(out_dir / "stability_down.csv", {**stability_down, "regime": "down"})
    _write_summary(
        out_dir=out_dir,
        index_rows=index_rows,
        stability=stability,
        db=db,
        symbols=symbols,
        strategy=strategy,
        strategy_config=strategy_config,
        top_k=max(1, int(top_k)),
        sort_by=sort_by,
        sort_desc=sort_desc,
    )
    return {"count": len(index_rows), "out_dir": str(out_dir)}


def _write_index_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "slice_id",
        "start",
        "end",
        "regime",
        "start_px",
        "end_px",
        "slice_return",
        "run_dir",
        "pnl_net_sum",
        "pnl_net_per_fill",
        "fills_count",
        "decision_to_fill_rate",
        "fee_sum",
        "spread_cost_est_sum",
        "avg_adverse_samples",
        "skipped_count",
        "alpha_ok",
        "events_replayed",
        "decisions_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _write_stability_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "slices_count",
        "pos_slices_count",
        "pos_slices_frac",
        "pnl_net_sum_total",
        "pnl_net_sum_mean",
        "pnl_net_sum_median",
        "pnl_net_sum_std",
        "pnl_net_sum_min",
        "pnl_net_sum_max",
        "worst_pnl_net_per_fill",
        "fill_rate_mean",
        "stability_score",
        "combined_score",
        "regime",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({k: row.get(k) for k in cols})


def _write_summary(
    out_dir: Path,
    index_rows: List[Dict[str, Any]],
    stability: Dict[str, Any],
    db: Path,
    symbols: List[str],
    strategy: str,
    strategy_config: Dict[str, Any],
    top_k: int,
    sort_by: str,
    sort_desc: bool,
) -> None:
    rows_sorted = sorted(
        index_rows,
        key=lambda r: (
            -float(r.get(sort_by, 0.0)) if sort_desc else float(r.get(sort_by, 0.0)),
            str(r.get("slice_id", "")),
        ),
    )
    best = rows_sorted[:top_k]
    worst = sorted(index_rows, key=lambda r: (float(r.get("pnl_net_sum", 0.0)), str(r.get("slice_id", ""))))[:top_k]
    lines = [
        "# Walkforward Eval Summary",
        "",
        f"- db: `{db}`",
        f"- symbols: `{','.join(symbols)}`",
        f"- strategy: `{strategy}`",
        f"- strategy_config: `{json.dumps(strategy_config, ensure_ascii=True, sort_keys=True)}`",
        "",
        "## Stability",
        "",
        "| slices_count | pos_slices_count | pos_slices_frac | pnl_net_sum_total | pnl_net_sum_mean | pnl_net_sum_median | pnl_net_sum_std | pnl_net_sum_min | pnl_net_sum_max | worst_pnl_net_per_fill | fill_rate_mean | stability_score |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {stability['slices_count']} | {stability['pos_slices_count']} | {stability['pos_slices_frac']:.6f} | "
            f"{stability['pnl_net_sum_total']:.12f} | {stability['pnl_net_sum_mean']:.12f} | {stability['pnl_net_sum_median']:.12f} | "
            f"{stability['pnl_net_sum_std']:.12f} | {stability['pnl_net_sum_min']:.12f} | {stability['pnl_net_sum_max']:.12f} | "
            f"{stability['worst_pnl_net_per_fill']:.12f} | {stability['fill_rate_mean']:.12f} | {stability['stability_score']:.12f} |"
        ),
        "",
        "## Top slices",
        "",
        "| rank | slice_id | pnl_net_sum | pnl_net_per_fill | fills_count | fill_rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(best, start=1):
        lines.append(
            f"| {i} | `{r['slice_id']}` | {float(r['pnl_net_sum']):.12f} | {float(r['pnl_net_per_fill']):.12f} | "
            f"{int(r['fills_count'])} | {float(r['decision_to_fill_rate']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Worst slices",
            "",
            "| rank | slice_id | pnl_net_sum | pnl_net_per_fill | fills_count | fill_rate |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for i, r in enumerate(worst, start=1):
        lines.append(
            f"| {i} | `{r['slice_id']}` | {float(r['pnl_net_sum']):.12f} | {float(r['pnl_net_per_fill']):.12f} | "
            f"{int(r['fills_count'])} | {float(r['decision_to_fill_rate']):.6f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward eval over multiple slices with stability metrics.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--strategy", default="baseline")
    p.add_argument("--strategy-config", default="{}")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--slices", default="", help="start,end;start,end;...")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--slice-sec", type=int, default=0)
    p.add_argument("--fee-bps", type=float, default=0.0)
    p.add_argument("--spread-bps", type=float, default=0.0)
    p.add_argument("--qty", type=float, default=1.0)
    p.add_argument("--horizon-sec", type=int, default=120)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--sort-by", default="stability_score")
    p.add_argument("--sort-desc", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        cfg = _load_strategy_config(str(args.strategy_config))
        if str(args.slices).strip():
            slices = _parse_slices(str(args.slices))
        else:
            if not (str(args.start).strip() and str(args.end).strip() and int(args.slice_sec) > 0):
                raise ValueError("either --slices or (--start --end --slice-sec) is required")
            slices = _iter_auto_slices(str(args.start), str(args.end), int(args.slice_sec))
        out = run_walkforward(
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            strategy=str(args.strategy),
            strategy_config=cfg,
            out_dir=Path(str(args.out_dir)),
            slices=slices,
            fee_bps=float(args.fee_bps),
            spread_bps=float(args.spread_bps),
            qty=float(args.qty),
            horizon_sec=max(1, int(args.horizon_sec)),
            top_k=max(1, int(args.top_k)),
            sort_by=str(args.sort_by),
            sort_desc=bool(args.sort_desc),
        )
        print(f"walkforward_eval ok out_dir={out['out_dir']} slices={out['count']}")
        return 0
    except Exception as e:
        print(f"walkforward_eval error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
