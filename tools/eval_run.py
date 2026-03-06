from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from tools.health_state import atomic_write_json
from tools.replay_strategy import _load_strategy_config, _parse_symbols, extract_events, replay_to_decisions, write_decisions_jsonl
from execution.sim.min_exec_sim import ExecSimConfig, simulate_fills_with_skips
from tools.state_reconstruct import reconstruct_state_vectors, write_state_vectors_jsonl


def _safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch)
        elif ch in ("-", "_"):
            out.append(ch)
    return "".join(out) or "run"


def _derive_run_dir(base: Path, start: str, end: str, strategy: str, cfg: Dict[str, Any], exec_cfg: Dict[str, Any]) -> Path:
    key = {
        "start": start,
        "end": end,
        "strategy": strategy,
        "config": cfg,
        "execution_sim": exec_cfg,
    }
    h = hashlib.sha1(json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()[:8]
    s = _safe_name(start.replace(":", "").replace("+", "").replace("Z", "Z"))
    e = _safe_name(end.replace(":", "").replace("+", "").replace("Z", "Z"))
    return base / f"{s}_{e}_{_safe_name(strategy)}_{h}"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_eval_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        os.replace(tmp_name, str(path))
    finally:
        try:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        except Exception:
            pass


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    _atomic_write_text(path, text)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            f.write("\n")


def run_eval(
    db: Path,
    symbols: List[str],
    start: str,
    end: str,
    strategy: str,
    strategy_config: Dict[str, Any],
    run_dir: Path,
    fee_bps: float = 0.0,
    spread_bps: float = 0.0,
    qty: float = 1.0,
    horizon_sec: int = 120,
) -> Dict[str, Any]:
    events = extract_events(
        db=db,
        symbols=symbols,
        start_iso=start,
        end_iso=end,
    )
    decisions, events_count = replay_to_decisions(
        db=db,
        symbols=symbols,
        start_iso=start,
        end_iso=end,
        strategy_name=strategy,
        strategy_config=strategy_config,
    )
    fills, skipped = simulate_fills_with_skips(
        decisions=decisions,
        events=events,
        cfg=ExecSimConfig(
            fee_bps=float(fee_bps),
            spread_bps=float(spread_bps),
            use_spread_model=(float(spread_bps) > 0.0),
            qty=float(qty),
            horizon_sec=int(horizon_sec),
            side_rule="from_params",
            skip_missing_price=True,
            horizon_or_before_enabled=True,
            fill_or_before_enabled=False,
        ),
    )
    by_symbol_events = Counter(d.get("symbol", "ALL") for d in decisions)
    by_symbol_fills = Counter(f.get("symbol", "ALL") for f in fills if str(f.get("status")) == "filled")
    skipped_reasons = Counter(str(s.get("reason") or "unknown") for s in skipped)
    horizon_sources = Counter(str(f.get("horizon_price_source") or "unknown") for f in fills if str(f.get("status")) == "filled")
    pnl_sum = 0.0
    pnl_gross_sum = 0.0
    fee_sum = 0.0
    adverse_sum = 0.0
    spread_cost_est_sum = 0.0
    adverse_samples_sum = 0
    adverse_samples_zero_count = 0
    fee_dominates_count = 0
    adverse_dominates_count = 0
    wins = 0
    per_symbol = {}
    for f in fills:
        if str(f.get("status")) != "filled":
            continue
        sym = str(f.get("symbol") or "ALL")
        pnl = float(f.get("pnl") or 0.0)
        pnl_gross = float(f.get("pnl_gross") or 0.0)
        fee = float(f.get("fee") or 0.0)
        adv = float(f.get("adverse_move") or 0.0)
        spread_cost_est = float(f.get("spread_cost_est") or 0.0)
        adverse_samples = int(f.get("adverse_samples") or 0)
        pnl_sum += pnl
        pnl_gross_sum += pnl_gross
        fee_sum += fee
        adverse_sum += adv
        spread_cost_est_sum += spread_cost_est
        adverse_samples_sum += adverse_samples
        if adverse_samples == 0:
            adverse_samples_zero_count += 1
        if abs(fee) > abs(pnl_gross):
            fee_dominates_count += 1
        if abs(adv * float(f.get("qty") or 0.0)) > abs(pnl_gross):
            adverse_dominates_count += 1
        if pnl > 0:
            wins += 1
        ps = per_symbol.setdefault(
            sym,
            {
                "fills_count": 0,
                "pnl_sum": 0.0,
                "pnl_gross_sum": 0.0,
                "fee_sum": 0.0,
                "adverse_sum": 0.0,
                "spread_cost_est_sum": 0.0,
                "adverse_samples_sum": 0,
                "adverse_samples_zero_count": 0,
            },
        )
        ps["fills_count"] += 1
        ps["pnl_sum"] += pnl
        ps["pnl_gross_sum"] += pnl_gross
        ps["fee_sum"] += fee
        ps["adverse_sum"] += adv
        ps["spread_cost_est_sum"] += spread_cost_est
        ps["adverse_samples_sum"] += adverse_samples
        if adverse_samples == 0:
            ps["adverse_samples_zero_count"] += 1
    filled_count = sum(1 for f in fills if str(f.get("status")) == "filled")
    decision_count = int(len(decisions))
    state_vectors = reconstruct_state_vectors(
        db=db,
        symbols=symbols,
        start_iso=start,
        end_iso=end,
        window_sec=30.0,
    )
    metrics = {
        "events_replayed": int(events_count),
        "state_vectors_count": int(len(state_vectors)),
        "decisions_count": decision_count,
        "fills_count": int(filled_count),
        "skipped_count": int(len(skipped)),
        "skipped_reasons": dict(sorted((k, int(v)) for k, v in skipped_reasons.items())),
        "decision_to_fill_rate": float(round((filled_count / decision_count) if decision_count > 0 else 0.0, 12)),
        "horizon_price_source_counts": dict(sorted((k, int(v)) for k, v in horizon_sources.items())),
        "pnl_gross_sum": float(round(pnl_gross_sum, 12)),
        "pnl_net_sum": float(round(pnl_sum, 12)),
        "pnl_sum": float(round(pnl_sum, 12)),
        "avg_pnl": float(round((pnl_sum / filled_count) if filled_count > 0 else 0.0, 12)),
        "fee_sum": float(round(fee_sum, 12)),
        "adverse_sum": float(round(adverse_sum, 12)),
        "spread_cost_est_sum": float(round(spread_cost_est_sum, 12)),
        "avg_adverse_samples": float(round((adverse_samples_sum / filled_count) if filled_count > 0 else 0.0, 12)),
        "adverse_samples_zero_count": int(adverse_samples_zero_count),
        "fee_dominates_count": int(fee_dominates_count),
        "adverse_dominates_count": int(adverse_dominates_count),
        "win_rate": float(round((wins / filled_count) if filled_count > 0 else 0.0, 12)),
        "per_symbol_decisions": dict(sorted((str(k), int(v)) for k, v in by_symbol_events.items())),
        "per_symbol_fills": dict(sorted((str(k), int(v)) for k, v in by_symbol_fills.items())),
        "per_symbol_metrics": {
            str(k): {
                "fills_count": int(v["fills_count"]),
                "pnl_sum": float(round(v["pnl_sum"], 12)),
                "pnl_gross_sum": float(round(v["pnl_gross_sum"], 12)),
                "fee_sum": float(round(v["fee_sum"], 12)),
                "adverse_sum": float(round(v["adverse_sum"], 12)),
                "spread_cost_est_sum": float(round(v["spread_cost_est_sum"], 12)),
                "avg_adverse_samples": float(
                    round((v["adverse_samples_sum"] / v["fills_count"]) if v["fills_count"] > 0 else 0.0, 12)
                ),
                "adverse_samples_zero_count": int(v["adverse_samples_zero_count"]),
            }
            for k, v in sorted(per_symbol.items())
        },
    }
    health = {
        "state": "ok",
        "reason": "",
        "events_replayed": int(events_count),
        "decisions_emitted": decision_count,
        "fills_emitted": int(filled_count),
        "skipped_emitted": int(len(skipped)),
        "ts_start_utc": start,
        "ts_end_utc": end,
    }
    config = {
        "db": str(db),
        "symbols": list(symbols),
        "start": start,
        "end": end,
        "strategy": strategy,
        "strategy_config": strategy_config,
        "execution_sim": {
            "mode": "min_exec_sim",
            "fee_bps": float(fee_bps),
            "spread_bps": float(spread_bps),
            "qty": float(qty),
            "horizon_sec": int(horizon_sec),
            "horizon_or_before_enabled": True,
            "fill_or_before_enabled": False,
        },
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "health.json", health)
    write_decisions_jsonl(run_dir / "decisions.jsonl", decisions)
    write_state_vectors_jsonl(run_dir / "state_vector.jsonl", state_vectors)
    _write_jsonl(run_dir / "fills.jsonl", fills)
    _write_jsonl(run_dir / "skipped.jsonl", skipped)
    _write_json(run_dir / "metrics.json", metrics)

    summary_lines = [
        "# Eval Run Summary",
        "",
        f"- slice: `{start}` -> `{end}`",
        f"- symbols: `{','.join(symbols)}`",
        f"- strategy: `{strategy}`",
        "",
        "## Counts",
        f"- events_replayed: {events_count}",
        f"- state_vectors_count: {len(state_vectors)}",
        f"- decisions_count: {len(decisions)}",
        f"- fills_count: {filled_count}",
        f"- skipped_count: {len(skipped)}",
        f"- decision_to_fill_rate: {round((filled_count / decision_count) if decision_count > 0 else 0.0, 12)}",
        f"- horizon_price_source_counts: {dict(sorted((k, int(v)) for k, v in horizon_sources.items()))}",
        f"- pnl_gross_sum: {round(pnl_gross_sum, 12)}",
        f"- pnl_net_sum: {round(pnl_sum, 12)}",
        f"- pnl_sum: {round(pnl_sum, 12)}",
        f"- avg_pnl: {round((pnl_sum / filled_count) if filled_count > 0 else 0.0, 12)}",
        f"- fee_sum: {round(fee_sum, 12)}",
        f"- spread_cost_est_sum: {round(spread_cost_est_sum, 12)}",
        f"- adverse_sum: {round(adverse_sum, 12)}",
        f"- avg_adverse_samples: {round((adverse_samples_sum / filled_count) if filled_count > 0 else 0.0, 12)}",
        f"- adverse_samples_zero_count: {adverse_samples_zero_count}",
        f"- fee_dominates_count: {fee_dominates_count}",
        f"- adverse_dominates_count: {adverse_dominates_count}",
        f"- win_rate: {round((wins / filled_count) if filled_count > 0 else 0.0, 12)}",
        "",
        "## Artifacts",
        "- `config.json`",
        "- `health.json`",
        "- `decisions.jsonl`",
        "- `state_vector.jsonl`",
        "- `fills.jsonl`",
        "- `skipped.jsonl`",
        "- `metrics.json`",
    ]
    _atomic_write_text(run_dir / "summary.md", "\n".join(summary_lines) + "\n")

    return {
        "run_dir": str(run_dir),
        "events_replayed": int(events_count),
        "decisions_count": int(len(decisions)),
        "fills_count": int(filled_count),
    }


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="One-command deterministic eval run folder writer.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--strategy", default="baseline")
    p.add_argument("--strategy-config", default="{}", help="JSON strategy config")
    p.add_argument("--run-dir", default="", help="Optional explicit run dir")
    p.add_argument("--speed", type=float, default=50.0, help="Reserved for compatibility")
    p.add_argument("--fee-bps", type=float, default=0.0)
    p.add_argument("--spread-bps", type=float, default=0.0)
    p.add_argument("--taker-model", action="store_true", help="Compatibility flag; spread is enabled when spread-bps > 0.")
    p.add_argument("--qty", type=float, default=1.0)
    p.add_argument("--horizon-sec", type=int, default=120)
    return p


def main() -> int:
    args = _parser().parse_args()
    _ = float(args.speed)
    try:
        cfg = _load_strategy_config(str(args.strategy_config))
        db = Path(str(args.db))
        syms = _parse_symbols(args.symbols)
        if not syms:
            raise ValueError("no symbols provided")
        if str(args.run_dir).strip():
            run_dir = Path(str(args.run_dir))
        else:
            run_dir = _derive_run_dir(
                Path("runs/eval"),
                str(args.start),
                str(args.end),
                str(args.strategy),
                cfg,
                {
                    "fee_bps": float(args.fee_bps),
                    "spread_bps": float(args.spread_bps),
                    "qty": float(args.qty),
                    "horizon_sec": max(1, int(args.horizon_sec)),
                },
            )
        out = run_eval(
            db=db,
            symbols=syms,
            start=str(args.start),
            end=str(args.end),
            strategy=str(args.strategy),
            strategy_config=cfg,
            run_dir=run_dir,
            fee_bps=float(args.fee_bps),
            spread_bps=float(args.spread_bps),
            qty=float(args.qty),
            horizon_sec=max(1, int(args.horizon_sec)),
        )
        print(
            f"eval_run ok run_dir={out['run_dir']} events={out['events_replayed']} "
            f"decisions={out['decisions_count']} fills={out['fills_count']}"
        )
        return 0
    except Exception as e:
        print(f"eval_run error runtime={type(e).__name__}:{e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
