from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tools.replay_strategy import _load_strategy_config, _parse_symbols
from tools.set_latest_run import set_latest_run
from tools.sweep_eval import _parse_grid
from tools.walkforward_eval import _iter_auto_slices, _parse_slices, run_walkforward


def _coerce_value(key: str, value: str) -> Any:
    if key in ("fee_bps", "spread_bps", "qty"):
        return float(value)
    if key in ("horizon_sec",):
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


def _parse_grid_strategy(raw: str) -> List[Tuple[str, List[str]]]:
    if not str(raw or "").strip():
        return []
    parts = [p.strip() for p in str(raw).split(";") if p.strip()]
    out: List[Tuple[str, List[str]]] = []
    for p in parts:
        if "=" not in p:
            raise ValueError(f"invalid strategy grid part: {p}")
        key, values_raw = p.split("=", 1)
        dotted = key.strip()
        if not dotted:
            raise ValueError(f"invalid strategy grid key: {p}")
        vals = [v.strip() for v in str(values_raw).split(",") if v.strip()]
        if not vals:
            raise ValueError(f"no values for strategy grid key: {dotted}")
        out.append((dotted, vals))
    return out


def _coerce_strategy_value(raw: str) -> Any:
    s = str(raw).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        return s


def _iter_grid_strategy(grid_items: List[Tuple[str, List[str]]]) -> Iterable[Dict[str, Any]]:
    if not grid_items:
        yield {}
        return
    keys = [k for k, _ in grid_items]
    values = [vs for _, vs in grid_items]
    for combo in itertools.product(*values):
        rec: Dict[str, Any] = {}
        for k, v in zip(keys, combo):
            rec[k] = _coerce_strategy_value(v)
        yield rec


def _set_dotted(dst: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = [p for p in str(dotted).split(".") if p]
    if not parts:
        return
    cur: Dict[str, Any] = dst
    for key in parts[:-1]:
        node = cur.get(key)
        if not isinstance(node, dict):
            node = {}
            cur[key] = node
        cur = node
    cur[parts[-1]] = value


def _apply_strategy_overrides(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key in sorted(overrides.keys()):
        _set_dotted(out, key, overrides[key])
    return out


def _strategy_base_hash(base: Dict[str, Any]) -> str:
    raw = json.dumps(base, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _combo_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _read_stability_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty stability csv: {path}")
    return rows[0]


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_int(v: Any) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def _format_env_lines(latest_dir: Path, enable_alpha_gate: bool) -> List[str]:
    def _win(p: Path) -> str:
        return str(p).replace("/", "\\")

    lines = [f"set ALPHA_GATE_METRICS_PATH={_win(latest_dir / 'metrics.json')}"]
    if enable_alpha_gate:
        lines.append("set ALPHA_GATE_ENABLED=1")
    if (latest_dir / "stability.csv").exists():
        lines.append(f"set ALPHA_GATE_STABILITY_PATH={_win(latest_dir / 'stability.csv')}")
    if (latest_dir / "stability_up.csv").exists():
        lines.append(f"set ALPHA_GATE_STABILITY_UP_PATH={_win(latest_dir / 'stability_up.csv')}")
    if (latest_dir / "stability_down.csv").exists():
        lines.append(f"set ALPHA_GATE_STABILITY_DOWN_PATH={_win(latest_dir / 'stability_down.csv')}")
    return lines


def _promote_rows(
    rows_sorted: List[Dict[str, Any]],
    out_dir: Path,
    *,
    promote_top: int,
    latest_dir: Path,
    latest_candidates_dir: Path,
    include_globs: List[str],
    extra: str,
    strict_extra: bool,
    print_env: bool,
    enable_alpha_gate: bool,
) -> List[Dict[str, Any]]:
    promoted: List[Dict[str, Any]] = []
    top_n = max(0, int(promote_top))
    if top_n <= 0:
        return promoted
    selected = rows_sorted[:top_n]
    for rank, row in enumerate(selected, start=1):
        combo_id = str(row.get("combo_id") or "")
        walkforward_rel = str(row.get("walkforward_dir") or "")
        run_dir = out_dir / walkforward_rel
        warnings: List[str] = []
        target_latest = latest_dir if rank == 1 else (latest_candidates_dir / f"{rank:02d}_{combo_id}")
        set_latest_run(
            run_dir=run_dir,
            latest_dir=target_latest,
            copy_mode="copy",
            overwrite=True,
            extra=extra,
            include_glob=include_globs,
            strict_extra=bool(strict_extra),
            warnings_out=warnings,
        )
        print(f"promote rank={rank} combo_id={combo_id} walkforward_dir={walkforward_rel} latest_dir={target_latest}")
        for w in warnings:
            print(f"promote WARN rank={rank} combo_id={combo_id}: {w}")
        if print_env and rank == 1:
            print("")
            for line in _format_env_lines(target_latest, enable_alpha_gate):
                print(line)
        promoted.append(
            {
                "rank": rank,
                "combo_id": combo_id,
                "walkforward_dir": walkforward_rel,
                "latest_dir": str(target_latest),
                "warnings": warnings,
            }
        )
    return promoted


def run_walkforward_sweep(
    db: Path,
    symbols: List[str],
    strategy: str,
    strategy_config: Dict[str, Any],
    out_dir: Path,
    slices: List[Tuple[str, str]],
    grid: List[Tuple[str, List[str]]],
    grid_strategy: List[Tuple[str, List[str]]],
    top_n: int,
    sort_by: str,
    sort_desc: bool,
    promote_top: int = 0,
    latest_dir: Path | None = None,
    latest_candidates_dir: Path | None = None,
    promote_include_glob: List[str] | None = None,
    promote_extra: str = "",
    promote_strict_extra: bool = False,
    promote_print_env: bool = False,
    promote_enable_alpha_gate: bool = False,
) -> Dict[str, Any]:
    combos_dir = out_dir / "combos"
    combos_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    strategy_base_hash = _strategy_base_hash(strategy_config)

    for combo, strategy_overrides in itertools.product(_iter_grid(grid), _iter_grid_strategy(grid_strategy)):
        fee_bps = float(combo.get("fee_bps", 0.0))
        spread_bps = float(combo.get("spread_bps", 0.0))
        horizon_sec = int(combo.get("horizon_sec", 120))
        qty = float(combo.get("qty", 1.0))
        effective_strategy_config = _apply_strategy_overrides(strategy_config, strategy_overrides)
        cid = _combo_id(
            {
                "db": str(db),
                "symbols": symbols,
                "strategy": strategy,
                "strategy_config": effective_strategy_config,
                "strategy_overrides": strategy_overrides,
                "slices": slices,
                "fee_bps": fee_bps,
                "spread_bps": spread_bps,
                "horizon_sec": horizon_sec,
                "qty": qty,
            }
        )
        combo_dir = combos_dir / cid
        wf_dir = combo_dir / "walkforward"
        combo_dir.mkdir(parents=True, exist_ok=True)
        (combo_dir / "config.json").write_text(
            json.dumps(
                {
                    "combo_id": cid,
                    "strategy": strategy,
                    "strategy_config_base_hash": strategy_base_hash,
                    "strategy_overrides": strategy_overrides,
                    "strategy_config_effective": effective_strategy_config,
                    "execution": {
                        "fee_bps": fee_bps,
                        "spread_bps": spread_bps,
                        "horizon_sec": horizon_sec,
                        "qty": qty,
                    },
                    "slices": slices,
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        run_walkforward(
            db=db,
            symbols=symbols,
            strategy=strategy,
            strategy_config=effective_strategy_config,
            out_dir=wf_dir,
            slices=slices,
            fee_bps=fee_bps,
            spread_bps=spread_bps,
            qty=qty,
            horizon_sec=horizon_sec,
            top_k=max(1, int(top_n)),
            sort_by=sort_by,
            sort_desc=sort_desc,
        )
        stab = _read_stability_csv(wf_dir / "stability_all.csv")
        stab_up = _read_stability_csv(wf_dir / "stability_up.csv")
        stab_down = _read_stability_csv(wf_dir / "stability_down.csv")
        row = {
            "combo_id": cid,
            "fee_bps": fee_bps,
            "spread_bps": spread_bps,
            "horizon_sec": horizon_sec,
            "qty": qty,
            "strategy_overrides_json": json.dumps(strategy_overrides, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            "imbalance_gte": _safe_float(strategy_overrides.get("filters.imbalance_gte")) if "filters.imbalance_gte" in strategy_overrides else "",
            "intensity_gte": _safe_float(strategy_overrides.get("filters.intensity_gte")) if "filters.intensity_gte" in strategy_overrides else "",
            "spread_lte": _safe_float(strategy_overrides.get("filters.spread_lte")) if "filters.spread_lte" in strategy_overrides else "",
            "cooldown_ms": _safe_int(strategy_overrides.get("cooldown_ms")) if "cooldown_ms" in strategy_overrides else "",
            "slices_count": _safe_int(stab.get("slices_count")),
            "pos_slices_count": _safe_int(stab.get("pos_slices_count")),
            "pos_slices_frac": _safe_float(stab.get("pos_slices_frac")),
            "pnl_net_sum_total": _safe_float(stab.get("pnl_net_sum_total")),
            "pnl_net_sum_mean": _safe_float(stab.get("pnl_net_sum_mean")),
            "pnl_net_sum_std": _safe_float(stab.get("pnl_net_sum_std")),
            "pnl_net_sum_min": _safe_float(stab.get("pnl_net_sum_min")),
            "worst_pnl_net_per_fill": _safe_float(stab.get("worst_pnl_net_per_fill")),
            "fill_rate_mean": _safe_float(stab.get("fill_rate_mean")),
            "stability_score": _safe_float(stab.get("stability_score")),
            "combined_score": _safe_float(stab.get("combined_score")),
            "stability_score_up": _safe_float(stab_up.get("stability_score")),
            "stability_score_down": _safe_float(stab_down.get("stability_score")),
            "pos_slices_frac_up": _safe_float(stab_up.get("pos_slices_frac")),
            "pos_slices_frac_down": _safe_float(stab_down.get("pos_slices_frac")),
            "walkforward_dir": f"combos/{cid}/walkforward",
        }
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -_safe_float(r.get(sort_by)) if sort_desc else _safe_float(r.get(sort_by)),
            -_safe_float(r.get("pos_slices_frac")),
            -_safe_float(r.get("pnl_net_sum_total")),
            str(r.get("combo_id")),
        ),
    )
    _write_index(out_dir / "index.csv", rows_sorted)
    (out_dir / "index.json").write_text(json.dumps({"count": len(rows_sorted), "rows": rows_sorted}, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_summary(out_dir / "summary.md", rows_sorted, top_n=max(1, int(top_n)), sort_by=sort_by, sort_desc=sort_desc)
    promoted = _promote_rows(
        rows_sorted,
        out_dir,
        promote_top=max(0, int(promote_top)),
        latest_dir=latest_dir or Path("runs/latest"),
        latest_candidates_dir=latest_candidates_dir or Path("runs/latest_candidates"),
        include_globs=list(promote_include_glob or []),
        extra=str(promote_extra or ""),
        strict_extra=bool(promote_strict_extra),
        print_env=bool(promote_print_env),
        enable_alpha_gate=bool(promote_enable_alpha_gate),
    )
    return {"count": len(rows_sorted), "out_dir": str(out_dir), "promoted": promoted}


def _write_index(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "combo_id",
        "fee_bps",
        "spread_bps",
        "horizon_sec",
        "qty",
        "strategy_overrides_json",
        "imbalance_gte",
        "intensity_gte",
        "spread_lte",
        "cooldown_ms",
        "slices_count",
        "pos_slices_count",
        "pos_slices_frac",
        "pnl_net_sum_total",
        "pnl_net_sum_mean",
        "pnl_net_sum_std",
        "pnl_net_sum_min",
        "worst_pnl_net_per_fill",
        "fill_rate_mean",
        "stability_score",
        "combined_score",
        "stability_score_up",
        "stability_score_down",
        "pos_slices_frac_up",
        "pos_slices_frac_down",
        "walkforward_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})


def _write_summary(path: Path, rows: List[Dict[str, Any]], top_n: int, sort_by: str, sort_desc: bool) -> None:
    top = rows[:top_n]
    lines = [
        "# Walkforward Sweep Summary",
        "",
        f"- total_combos: {len(rows)}",
        f"- sort: `{sort_by}` ({'desc' if sort_desc else 'asc'})",
        "",
        "| rank | combo_id | combined_score | stability_score | pos_slices_frac | pnl_net_sum_total | fee_bps | spread_bps | horizon_sec | qty | strategy_overrides | walkforward_dir |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(top, start=1):
        lines.append(
            f"| {i} | `{r['combo_id']}` | {float(r.get('combined_score', 0.0)):.12f} | {float(r['stability_score']):.12f} | {float(r['pos_slices_frac']):.6f} | "
            f"{float(r['pnl_net_sum_total']):.12f} | {r['fee_bps']} | {r['spread_bps']} | {int(r['horizon_sec'])} | "
            f"{r['qty']} | `{r.get('strategy_overrides_json', '{}')}` | `{r['walkforward_dir']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Walk-forward sweep across execution parameter grid.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--strategy", default="baseline")
    p.add_argument("--strategy-config", default="{}")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--slices", default="")
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--slice-sec", type=int, default=0)
    p.add_argument("--grid", required=True)
    p.add_argument("--grid-strategy", default="")
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--sort-by", default="stability_score")
    p.add_argument("--sort-desc", action="store_true")
    p.add_argument("--promote-top", type=int, default=0)
    p.add_argument("--latest-dir", default="runs/latest")
    p.add_argument("--latest-candidates-dir", default="runs/latest_candidates")
    p.add_argument("--promote-include-glob", action="append", default=[])
    p.add_argument("--promote-extra", default="")
    p.add_argument("--promote-strict-extra", action="store_true")
    p.add_argument("--promote-print-env", action="store_true")
    p.add_argument("--promote-enable-alpha-gate", action="store_true")
    return p


def main() -> int:
    args = _parser().parse_args()
    try:
        if str(args.slices).strip():
            slices = _parse_slices(str(args.slices))
        else:
            if not (str(args.start).strip() and str(args.end).strip() and int(args.slice_sec) > 0):
                raise ValueError("either --slices or (--start --end --slice-sec) is required")
            slices = _iter_auto_slices(str(args.start), str(args.end), int(args.slice_sec))
        out = run_walkforward_sweep(
            db=Path(str(args.db)),
            symbols=_parse_symbols(args.symbols),
            strategy=str(args.strategy),
            strategy_config=_load_strategy_config(str(args.strategy_config)),
            out_dir=Path(str(args.out_dir)),
            slices=slices,
            grid=_parse_grid(str(args.grid)),
            grid_strategy=_parse_grid_strategy(str(args.grid_strategy)),
            top_n=max(1, int(args.top_n)),
            sort_by=str(args.sort_by),
            sort_desc=bool(args.sort_desc),
            promote_top=max(0, int(args.promote_top)),
            latest_dir=Path(str(args.latest_dir)),
            latest_candidates_dir=Path(str(args.latest_candidates_dir)),
            promote_include_glob=list(args.promote_include_glob or []),
            promote_extra=str(args.promote_extra),
            promote_strict_extra=bool(args.promote_strict_extra),
            promote_print_env=bool(args.promote_print_env),
            promote_enable_alpha_gate=bool(args.promote_enable_alpha_gate),
        )
        print(f"walkforward_sweep ok out_dir={out['out_dir']} combos={out['count']} promoted={len(out.get('promoted') or [])}")
        return 0
    except Exception as e:
        print(f"walkforward_sweep error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
