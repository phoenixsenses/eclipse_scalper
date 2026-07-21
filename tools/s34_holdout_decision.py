from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDE = {"P013", "P056"}
DEFAULT_RULE_NAME = "ETH_BUY_LIQ_LONG_50K_TP120_SL40_BE30"


def read_trades(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("trades", []) if isinstance(payload, dict) else payload


def valid_closed_trade(trade: dict[str, Any], exclude: set[str], rule_name: str = DEFAULT_RULE_NAME) -> bool:
    tid = str(trade.get("trade_id") or trade.get("trial_id") or "")
    if tid in exclude or trade.get("status") != "CLOSED":
        return False
    if str((trade.get("rule") or {}).get("name") or "") != rule_name:
        return False
    if (trade.get("entry_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    if (trade.get("exit_fill") or {}).get("source") != "BOOK_TICKER":
        return False
    required = ("gross_bps", "entry_adverse_bps", "exit_adverse_bps", "spread_cost_bps", "fee_cost_bps", "net_bps")
    if any(trade.get(key) is None for key in required):
        return False
    identity = (
        float(trade["gross_bps"])
        - float(trade["entry_adverse_bps"])
        - float(trade["exit_adverse_bps"])
        - float(trade["spread_cost_bps"])
        - float(trade["fee_cost_bps"])
    )
    return abs(identity - float(trade["net_bps"])) <= 1e-6


def valid_closed_trades(trades: list[dict[str, Any]], exclude: set[str], rule_name: str = DEFAULT_RULE_NAME) -> list[dict[str, Any]]:
    return sorted(
        [t for t in trades if valid_closed_trade(t, exclude, rule_name)],
        key=lambda t: (int(t.get("signal_ts_ms") or 0), str(t.get("trade_id") or "")),
    )


def bootstrap_lower_mean(values: list[float], *, resamples: int = 10_000, alpha: float = 0.05, seed: int = 34) -> float:
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += values[rng.randrange(n)]
        means.append(sample_sum / n)
    means.sort()
    return means[int(alpha * (len(means) - 1))]


def probabilistic_sharpe_ratio(values: list[float], benchmark_sr: float = 0.0) -> dict[str, float | None]:
    n = len(values)
    if n < 3:
        return {"sharpe": None, "probability": None, "z": None}
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values)
    if stdev <= 0:
        if mean > 0:
            return {"sharpe": float("inf"), "probability": 1.0, "z": float("inf")}
        return {"sharpe": 0.0, "probability": 0.0, "z": float("-inf")}
    sr = mean / stdev
    # Bailey-Lopez de Prado PSR core. Higher moments are omitted deliberately here;
    # this script is frozen before data, and the conservative deflated benchmark below
    # is the main multiple-testing correction.
    z = (sr - benchmark_sr) * math.sqrt(n - 1)
    probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return {"sharpe": sr, "probability": probability, "z": z}


def deflated_benchmark_sharpe(n_trials: int, sample_n: int) -> float:
    trials = max(1, int(n_trials))
    # Conservative normal maximum benchmark for multiple tried configurations.
    # Uses the expected maximum of trials noise Sharpes at the same sample size.
    p = 1.0 - 1.0 / trials
    # Acklam inverse-normal approximation.
    q = _norm_ppf(min(max(p, 1e-9), 1.0 - 1e-9))
    return q / math.sqrt(max(1, sample_n - 1))


def _norm_ppf(p: float) -> float:
    # Peter J. Acklam's rational approximation, coefficients in public domain.
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def holdout_decision(
    trades: list[dict[str, Any]],
    *,
    exclude: set[str] | None = None,
    calibration_n: int = 40,
    holdout_n: int = 60,
    bootstrap_resamples: int = 10_000,
    n_trials: int = 50,
    rule_name: str = DEFAULT_RULE_NAME,
) -> dict[str, Any]:
    exclude = DEFAULT_EXCLUDE if exclude is None else exclude
    valid = valid_closed_trades(trades, exclude, rule_name)
    required = calibration_n + holdout_n
    if len(valid) < required:
        return {
            "decision": "INSUFFICIENT_SAMPLE_DO_NOT_RUN_HOLDOUT",
            "valid_closed_count": len(valid),
            "required_valid_closed_count": required,
            "validation_rule_name": rule_name,
        }
    holdout = valid[calibration_n:required]
    net = [float(t["net_bps"]) for t in holdout]
    mean_net = statistics.fmean(net)
    lower = bootstrap_lower_mean(net, resamples=bootstrap_resamples)
    benchmark = deflated_benchmark_sharpe(max(50, int(n_trials)), len(net))
    psr = probabilistic_sharpe_ratio(net, benchmark_sr=benchmark)
    economic = mean_net > 0.0 and lower > 0.0
    statistical = bool(psr["probability"] is not None and psr["probability"] >= 0.95)
    return {
        "decision": "PASS" if economic and statistical else "FAIL",
        "valid_closed_count": len(valid),
        "validation_rule_name": rule_name,
        "holdout_count": len(holdout),
        "mean_net_bps": mean_net,
        "bootstrap_lower_95_mean_bps": lower,
        "economic_significance_pass": economic,
        "n_trials": max(50, int(n_trials)),
        "deflated_benchmark_sharpe": benchmark,
        "holdout_sharpe_proxy": psr["sharpe"],
        "deflated_sharpe_probability": psr["probability"],
        "statistical_significance_pass": statistical,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run S34 pre-registered holdout decision once N=100 valid closed trades exists.")
    parser.add_argument("--trades-json", default="reports/research/s34/S34_SHADOW_PAPER_TRADES.json")
    parser.add_argument("--exclude", default="P013,P056")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--rule-name", default=DEFAULT_RULE_NAME)
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    exclude = {x.strip() for x in str(args.exclude).split(",") if x.strip()}
    result = holdout_decision(
        read_trades(Path(args.trades_json)),
        exclude=exclude,
        n_trials=int(args.n_trials),
        bootstrap_resamples=int(args.bootstrap_resamples),
        rule_name=str(args.rule_name),
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
