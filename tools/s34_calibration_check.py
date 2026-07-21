from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDE = {"P013", "P056"}
DEFAULT_RULE_NAME = "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"


def read_trades(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("trades", []) if isinstance(payload, dict) else payload


def valid_closed_trade(trade: dict[str, Any], exclude: set[str], rule_name: str = DEFAULT_RULE_NAME) -> tuple[bool, str]:
    tid = str(trade.get("trade_id") or trade.get("trial_id") or "")
    if tid in exclude:
        return False, "EXCLUDED"
    actual_rule = str((trade.get("rule") or {}).get("name") or "")
    if actual_rule != rule_name:
        return False, "RULE_EXCLUDED"
    if trade.get("status") != "CLOSED":
        return False, "NOT_CLOSED"
    if (trade.get("entry_fill") or {}).get("source") != "BOOK_TICKER":
        return False, "ENTRY_NOT_BOOK"
    if (trade.get("exit_fill") or {}).get("source") != "BOOK_TICKER":
        return False, "EXIT_NOT_BOOK"
    required = ("gross_bps", "entry_adverse_bps", "exit_adverse_bps", "spread_cost_bps", "fee_cost_bps", "net_bps")
    missing = [key for key in required if trade.get(key) is None]
    if missing:
        return False, "MISSING_" + ",".join(missing)
    identity = (
        float(trade["gross_bps"])
        - float(trade["entry_adverse_bps"])
        - float(trade["exit_adverse_bps"])
        - float(trade["spread_cost_bps"])
        - float(trade["fee_cost_bps"])
    )
    if abs(identity - float(trade["net_bps"])) > 1e-6:
        return False, "COST_IDENTITY_FAIL"
    return True, ""


def valid_closed_trades(trades: list[dict[str, Any]], exclude: set[str], rule_name: str = DEFAULT_RULE_NAME) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trade in trades:
        ok, _ = valid_closed_trade(trade, exclude, rule_name)
        if ok:
            out.append(trade)
    return sorted(out, key=lambda t: (int(t.get("signal_ts_ms") or 0), str(t.get("trade_id") or "")))


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def calibration_check(
    trades: list[dict[str, Any]],
    *,
    exclude: set[str] | None = None,
    n: int = 40,
    rule_name: str = DEFAULT_RULE_NAME,
) -> dict[str, Any]:
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    valid = valid_closed_trades(trades, exclude, rule_name)
    calibration = valid[:n]
    result: dict[str, Any] = {
        "valid_closed_count": len(valid),
        "validation_rule_name": rule_name,
        "calibration_target_n": n,
        "calibration_count": len(calibration),
        "ready": len(calibration) >= n,
        "kills": {},
        "friction_by_exit_reason": {},
    }
    if not calibration:
        result["kills"] = {"K1": None, "K2": None}
        return result

    net = [float(t["net_bps"]) for t in calibration]
    gross_abs = [abs(float(t["gross_bps"])) for t in calibration]
    entry_adv = [float(t["entry_adverse_bps"]) for t in calibration]
    result["net_bps"] = summarize(net)
    result["gross_abs_bps"] = summarize(gross_abs)
    result["entry_adverse_bps"] = summarize(entry_adv)
    result["kills"] = {
        "K1_mean_net_le_zero": statistics.fmean(net) <= 0.0,
        "K2_median_entry_adverse_ge_mean_abs_gross": statistics.median(entry_adv) >= statistics.fmean(gross_abs),
    }

    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for trade in calibration:
        reason = str(trade.get("exit_reason") or "UNKNOWN")
        for key in ("entry_adverse_bps", "exit_adverse_bps", "spread_cost_bps", "fee_cost_bps", "net_bps"):
            buckets[reason][key].append(float(trade[key]))
    result["friction_by_exit_reason"] = {
        reason: {key: summarize(vals) for key, vals in cols.items()}
        for reason, cols in sorted(buckets.items())
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run S34 pre-registered calibration kill checks on first 40 valid closed trades.")
    parser.add_argument("--trades-json", default="reports/research/s34/S34_SHADOW_PAPER_TRADES.json")
    parser.add_argument("--exclude", default="P013,P056")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--rule-name", default=DEFAULT_RULE_NAME)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    exclude = {x.strip() for x in str(args.exclude).split(",") if x.strip()}
    result = calibration_check(read_trades(Path(args.trades_json)), exclude=exclude, n=int(args.n), rule_name=str(args.rule_name))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
