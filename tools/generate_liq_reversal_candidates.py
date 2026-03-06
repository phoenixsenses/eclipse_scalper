from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tools.run_summary import build_run_summary


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    for tok in str(raw or "").replace(";", ",").split(","):
        t = tok.strip()
        if t:
            out.append(t)
    return out


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(x) for x in _parse_list(raw)]
    uniq: List[float] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(float(x)) for x in _parse_list(raw)]
    uniq: List[int] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        uniq.append(v)
    return uniq


def build_candidates(
    *,
    symbols: List[str],
    horizons: List[int],
    min_imbalances: List[float],
    min_trade_intensities: List[float],
    max_spreads: List[float],
    rule: str,
    regime: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for symbol in symbols:
        for horizon_sec in horizons:
            for min_imbalance in min_imbalances:
                for min_trade_intensity in min_trade_intensities:
                    for max_spread in max_spreads:
                        out.append(
                            {
                                "symbol": str(symbol),
                                "rule": str(rule),
                                "regime": str(regime),
                                "horizon_sec": int(horizon_sec),
                                "min_imbalance": float(min_imbalance),
                                "min_trade_intensity": float(min_trade_intensity),
                                "max_spread": float(max_spread),
                                "pass": "YES",
                            }
                        )
    out.sort(
        key=lambda row: (
            str(row["symbol"]),
            int(row["horizon_sec"]),
            float(row["min_imbalance"]),
            float(row["min_trade_intensity"]),
            float(row["max_spread"]),
        )
    )
    return out


def write_candidates_md(path: Path, candidates: List[Dict[str, Any]], *, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "| symbol | rule | regime | horizon_sec | min_imbalance | min_trade_intensity | max_spread | pass |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['symbol']} | {row['rule']} | {row['regime']} | {int(row['horizon_sec'])} | "
            f"{float(row['min_imbalance']):.2f} | {float(row['min_trade_intensity']):.0f} | "
            f"{float(row['max_spread']):.6f} | {row['pass']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(
    *,
    symbols: List[str],
    horizons: List[int],
    min_imbalances: List[float],
    min_trade_intensities: List[float],
    max_spreads: List[float],
    rule: str,
    regime: str,
    candidates: List[Dict[str, Any]],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    payload = {
        "rule": str(rule),
        "regime": str(regime),
        "symbols": list(symbols),
        "grid": {
            "horizons": [int(x) for x in horizons],
            "min_imbalances": [float(x) for x in min_imbalances],
            "min_trade_intensities": [float(x) for x in min_trade_intensities],
            "max_spreads": [float(x) for x in max_spreads],
        },
        "count": int(len(candidates)),
        "rows": list(candidates),
    }
    payload["run_summary"] = build_run_summary(
        run_type="generate_liq_reversal_candidates",
        inputs={
            "rule": str(rule),
            "regime": str(regime),
            "symbols": list(symbols),
            "horizons": [int(x) for x in horizons],
            "min_imbalances": [float(x) for x in min_imbalances],
            "min_trade_intensities": [float(x) for x in min_trade_intensities],
            "max_spreads": [float(x) for x in max_spreads],
        },
        metrics={"count": int(len(candidates))},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )
    return payload


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate candidate surface for liquidation reversal research.")
    p.add_argument("--symbols", default="ETHUSDT")
    p.add_argument("--horizons-sec", default="30,60,120")
    p.add_argument("--min-imbalances", default="0.30,0.40,0.50")
    p.add_argument("--min-trade-intensities", default="200,400,800")
    p.add_argument("--max-spreads", default="0.00025,0.00035,0.00050")
    p.add_argument("--rule", default="high_liq_reversal_regime")
    p.add_argument("--regime", default="liq_reversal_research")
    p.add_argument("--out-md", default="reports/LIQ_REVERSAL_CANDIDATES.md")
    p.add_argument("--out-json", default="reports/LIQ_REVERSAL_CANDIDATES.json")
    return p.parse_args()


def main() -> int:
    args = _args()
    symbols = _parse_list(args.symbols)
    horizons = _parse_int_list(args.horizons_sec)
    min_imbalances = _parse_float_list(args.min_imbalances)
    min_trade_intensities = _parse_float_list(args.min_trade_intensities)
    max_spreads = _parse_float_list(args.max_spreads)
    out_md = Path(str(args.out_md))
    out_json = Path(str(args.out_json))

    candidates = build_candidates(
        symbols=symbols,
        horizons=horizons,
        min_imbalances=min_imbalances,
        min_trade_intensities=min_trade_intensities,
        max_spreads=max_spreads,
        rule=str(args.rule),
        regime=str(args.regime),
    )
    write_candidates_md(out_md, candidates, title="LIQUIDATION_REVERSAL_CANDIDATES")
    payload = build_payload(
        symbols=symbols,
        horizons=horizons,
        min_imbalances=min_imbalances,
        min_trade_intensities=min_trade_intensities,
        max_spreads=max_spreads,
        rule=str(args.rule),
        regime=str(args.regime),
        candidates=candidates,
        out_json=out_json,
        out_md=out_md,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
