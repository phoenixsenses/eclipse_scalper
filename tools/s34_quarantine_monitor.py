from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDE = {"P013", "P056"}


def read_trades(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("trades", []) if isinstance(payload, dict) else payload


def _intensity(trade: dict[str, Any]) -> float | None:
    signal = trade.get("signal") or {}
    for key in ("liq_total_notional", "liq_notional", "notional", "intensity", "cascade_intensity"):
        if signal.get(key) is not None:
            return float(signal[key])
    if trade.get("liq_total_notional") is not None:
        return float(trade["liq_total_notional"])
    return None


def _is_candidate_signal(trade: dict[str, Any], exclude: set[str]) -> bool:
    tid = str(trade.get("trade_id") or trade.get("trial_id") or "")
    if tid in exclude:
        return False
    return trade.get("status") in {"CLOSED", "OPEN", "SKIPPED"}


def _is_no_fill_data(trade: dict[str, Any]) -> bool:
    reason = str(trade.get("risk_gate_reason") or trade.get("exit_reason") or "")
    fill_error = str(trade.get("fill_error") or "")
    return reason == "NO_FILL_DATA" or "NO_FILL_DATA" in fill_error or "no_book_ticker" in fill_error.lower()


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 4 or len(xs) != len(ys):
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def fisher_p_value(r: float | None, n: int) -> float | None:
    if r is None or n < 4 or abs(r) >= 1:
        return None
    z = math.atanh(r) * math.sqrt(n - 3)
    return math.erfc(abs(z) / math.sqrt(2.0))


def quarantine_monitor(trades: list[dict[str, Any]], *, exclude: set[str] | None = None) -> dict[str, Any]:
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    candidates = [t for t in trades if _is_candidate_signal(t, exclude)]
    usable: list[tuple[float, float]] = []
    quarantined = 0
    for trade in candidates:
        q = 1.0 if _is_no_fill_data(trade) else 0.0
        quarantined += int(q)
        intensity = _intensity(trade)
        if intensity is not None:
            usable.append((q, float(intensity)))
    rate = quarantined / len(candidates) if candidates else 0.0
    q_flags = [x[0] for x in usable]
    intensities = [x[1] for x in usable]
    r = pearson(q_flags, intensities)
    p = fisher_p_value(r, len(usable))
    significant = bool(r is not None and p is not None and r > 0.0 and p < 0.05)
    return {
        "candidate_signals": len(candidates),
        "no_fill_data_count": quarantined,
        "no_fill_data_rate": rate,
        "correlation_sample_n": len(usable),
        "quarantine_intensity_pearson_r": r,
        "quarantine_intensity_p_value": p,
        "correlation_significant_positive": significant,
        "K3_triggered": bool(rate > 0.25 and significant),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor S34 NO_FILL_DATA quarantine rate and intensity selection bias.")
    parser.add_argument("--trades-json", default="reports/research/s34/S34_SHADOW_PAPER_TRADES.json")
    parser.add_argument("--exclude", default="P013,P056")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    exclude = {x.strip() for x in str(args.exclude).split(",") if x.strip()}
    result = quarantine_monitor(read_trades(Path(args.trades_json)), exclude=exclude)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
