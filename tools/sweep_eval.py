from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tools.eval_run import run_eval
from tools.replay_strategy import _parse_symbols


SUPPORTED_GRID_KEYS = {"fee_bps", "spread_bps", "horizon_sec", "qty"}


def _parse_grid(raw: str) -> List[Tuple[str, List[str]]]:
    parts = [p.strip() for p in str(raw or "").split(";") if p.strip()]
    out: List[Tuple[str, List[str]]] = []
    for p in parts:
        if "=" not in p:
            raise ValueError(f"invalid grid part: {p}")
        k, v = p.split("=", 1)
        key = k.strip()
        if key not in SUPPORTED_GRID_KEYS:
            raise ValueError(f"unsupported grid key: {key}")
        vals = [x.strip() for x in v.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"no values for grid key: {key}")
        out.append((key, vals))
    if not out:
        raise ValueError("empty grid")
    return out


def _coerce_value(key: str, value: str) -> Any:
    if key in ("fee_bps", "spread_bps", "qty"):
        return float(value)
    if key == "horizon_sec":
        return int(value)
    return value


def _iter_grid(grid_items: List[Tuple[str, List[str]]]) -> Iterable[Dict[str, Any]]:
    keys = [k for k, _ in grid_items]
    values = [vs for _, vs in grid_items]
    for combo in itertools.product(*values):
        rec: Dict[str, Any] = {}
        for k, v in zip(keys, combo):
            rec[k] = _coerce_value(k, v)
        yield rec


def _run_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_strategy_config(raw: str) -> Dict[str, Any]:
    s = str(raw or "").strip()
    if not s:
        return {}
    p = Path(s)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(s)
    if not isinstance(data, dict):
        raise ValueError("strategy-config must be JSON object")
    return data


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def run_sweep(
    db: Path,
    symbols: List[str],
    start: str,
    end: str,
    strategy: str,
    strategy_config: Dict[str, Any],
    out_dir: Path,
    grid: List[Tuple[str, List[str]]],
    base_qty: float,
    top_n: int,
    sort_by: str,
    sort_desc: bool,
) -> Dict[str, Any]:
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for combo in _iter_grid(grid):
        fee_bps = float(combo.get("fee_bps", 0.0))
        spread_bps = float(combo.get("spread_bps", 0.0))
        horizon_sec = int(combo.get("horizon_sec", 120))
        qty = float(combo.get("qty", base_qty))
        sig_payload = {
            "db": str(db),
            "symbols": symbols,
            "start": start,
            "end": end,
            "strategy": strategy,
            "strategy_config": strategy_config,
            "fee_bps": fee_bps,
            "spread_bps": spread_bps,
            "horizon_sec": horizon_sec,
            "qty": qty,
        }
        h = _run_hash(sig_payload)
        run_dir = runs_dir / f"{strategy}_{h}"
        out = run_eval(
            db=db,
            symbols=symbols,
            start=start,
            end=end,
            strategy=strategy,
            strategy_config=strategy_config,
            run_dir=run_dir,
            fee_bps=fee_bps,
            spread_bps=spread_bps,
            qty=qty,
            horizon_sec=horizon_sec,
        )
        cfg = _read_json(run_dir / "config.json")
        metrics = _read_json(run_dir / "metrics.json")
        fills_count = _safe_int(metrics.get("fills_count"))
        pnl_net_sum = _safe_float(metrics.get("pnl_net_sum", metrics.get("pnl_sum", 0.0)))
        row = {
            "run_dir": f"runs/{run_dir.name}",
            "strategy": str(cfg.get("strategy") or strategy),
            "fee_bps": _safe_float(cfg.get("execution_sim", {}).get("fee_bps")),
            "spread_bps": _safe_float(cfg.get("execution_sim", {}).get("spread_bps")),
            "horizon_sec": _safe_int(cfg.get("execution_sim", {}).get("horizon_sec")),
            "qty": _safe_float(cfg.get("execution_sim", {}).get("qty")),
            "events_replayed": _safe_int(metrics.get("events_replayed")),
            "decisions_count": _safe_int(metrics.get("decisions_count")),
            "fills_count": fills_count,
            "decision_to_fill_rate": _safe_float(metrics.get("decision_to_fill_rate")),
            "pnl_gross_sum": _safe_float(metrics.get("pnl_gross_sum")),
            "pnl_net_sum": pnl_net_sum,
            "fee_sum": _safe_float(metrics.get("fee_sum")),
            "spread_cost_est_sum": _safe_float(metrics.get("spread_cost_est_sum")),
            "avg_adverse_samples": _safe_float(metrics.get("avg_adverse_samples")),
            "fee_dominates_count": _safe_int(metrics.get("fee_dominates_count")),
            "adverse_dominates_count": _safe_int(metrics.get("adverse_dominates_count")),
            "skipped_count": _safe_int(metrics.get("skipped_count")),
            "pnl_net_per_fill": float(round((pnl_net_sum / fills_count) if fills_count > 0 else 0.0, 12)),
        }
        rows.append(row)

    def _sort_key(x: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        primary = _safe_float(x.get(sort_by))
        primary_key = -primary if sort_desc else primary
        return (primary_key, -_safe_int(x.get("fills_count")), str(x.get("run_dir")))

    rows_sorted = sorted(rows, key=_sort_key)
    _write_index(out_dir, rows_sorted)
    _write_summary(
        out_dir=out_dir,
        rows=rows_sorted,
        top_n=max(1, int(top_n)),
        db=db,
        symbols=symbols,
        start=start,
        end=end,
        strategy=strategy,
        grid=grid,
        sort_by=sort_by,
        sort_desc=sort_desc,
    )
    return {"count": len(rows_sorted), "out_dir": str(out_dir)}


def _write_index(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [
        "run_dir",
        "strategy",
        "fee_bps",
        "spread_bps",
        "horizon_sec",
        "qty",
        "events_replayed",
        "decisions_count",
        "fills_count",
        "decision_to_fill_rate",
        "pnl_gross_sum",
        "pnl_net_sum",
        "pnl_net_per_fill",
        "fee_sum",
        "spread_cost_est_sum",
        "avg_adverse_samples",
        "fee_dominates_count",
        "adverse_dominates_count",
        "skipped_count",
    ]
    with (out_dir / "index.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    (out_dir / "index.json").write_text(
        json.dumps({"count": len(rows), "rows": rows}, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_summary(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    top_n: int,
    db: Path,
    symbols: List[str],
    start: str,
    end: str,
    strategy: str,
    grid: List[Tuple[str, List[str]]],
    sort_by: str,
    sort_desc: bool,
) -> None:
    top = rows[:top_n]
    lines = [
        "# Sweep Eval Summary",
        "",
        f"- db: `{db}`",
        f"- symbols: `{','.join(symbols)}`",
        f"- slice: `{start}` -> `{end}`",
        f"- strategy: `{strategy}`",
        f"- sort: `{sort_by}` ({'desc' if sort_desc else 'asc'})",
        f"- grid: `{';'.join(f'{k}={','.join(vs)}' for k, vs in grid)}`",
        f"- total_runs: {len(rows)}",
        "",
        "## Top N",
        "",
        "| rank | run_dir | pnl_net_sum | pnl_net_per_fill | fills_count | spread_bps | fee_bps | horizon_sec | avg_adverse_samples |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(top, start=1):
        lines.append(
            f"| {i} | `{r['run_dir']}` | {r['pnl_net_sum']:.12f} | {r['pnl_net_per_fill']:.12f} | "
            f"{r['fills_count']} | {r['spread_bps']} | {r['fee_bps']} | {r['horizon_sec']} | {r['avg_adverse_samples']:.6f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run eval_run across parameter grid and build deterministic leaderboard.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--strategy", default="baseline")
    p.add_argument("--strategy-config", default="{}", help="JSON dict or path to JSON file")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--grid", required=True, help="fee_bps=0,0.6;spread_bps=0,2;horizon_sec=5,10")
    p.add_argument("--qty", type=float, default=1.0)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--sort-by", default="pnl_net_sum")
    p.add_argument("--sort-desc", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        cfg = _load_strategy_config(str(args.strategy_config))
        out = run_sweep(
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            start=str(args.start),
            end=str(args.end),
            strategy=str(args.strategy),
            strategy_config=cfg,
            out_dir=Path(str(args.out_dir)),
            grid=_parse_grid(str(args.grid)),
            base_qty=float(args.qty),
            top_n=int(args.top_n),
            sort_by=str(args.sort_by),
            sort_desc=bool(args.sort_desc),
        )
        print(f"sweep_eval ok out_dir={out['out_dir']} count={out['count']}")
        return 0
    except Exception as e:
        print(f"sweep_eval error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
