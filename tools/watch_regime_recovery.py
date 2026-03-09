"""
watch_regime_recovery.py -- Poll event lane gate state and emit readiness report.

Tells you when the ETHUSDT h=60 imb>=0.85 pocket is safe to re-validate after
a trending regime block (e.g. the Mar 1-6 rally). When the gate shows ALLOWED,
prints the exact re-validation command to run.

Usage:
    python -m tools.watch_regime_recovery --db ../eclipse_scalper/data/microstructure.db
    python -m tools.watch_regime_recovery --db ... --watch --interval 3600
    python -m tools.watch_regime_recovery --db ... --json
    python -m tools.watch_regime_recovery --db ... --out-json reports/REGIME_RECOVERY_STATUS.json
    python -m tools.watch_regime_recovery --db ... --consecutive 3 --watch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Re-validation command to run when gate shows ALLOWED
REVALIDATE_CMD = (
    "py -3 -m tools.rank_passive_pockets_forward"
    " --candidates-md reports/FILTER_SWEEP_PASSIVE_REALISTIC_ETH_IMB05.md"
    " --db ../eclipse_scalper/data/microstructure.db"
    " --lookback-min 10080 --splits 3"
    " --out-md reports/REVALIDATE_POST_REGIME_7D.md"
    " --out-json reports/REVALIDATE_POST_REGIME_7D.json"
)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _call_check_gate(db: str, symbol: str) -> dict:
    """Call check_gate from tools.check_event_lanes. Returns raw result or error dict."""
    try:
        from tools.check_event_lanes import check_gate
        return check_gate(db=db, symbol=symbol)
    except Exception as e:
        return {"gate": "UNKNOWN", "reason": f"error:{e}", "lanes": {}}


def evaluate_status(
    db: str,
    symbol: str,
    consecutive_required: int,
    consecutive_state: list,
) -> dict:
    """
    Evaluate current gate state and update consecutive_state (mutated in-place).
    consecutive_state is a single-element list [int] so callers can share state.

    Returns a status payload dict.
    """
    gate_result = _call_check_gate(db, symbol)
    allowed = str(gate_result.get("gate", "")).upper() == "ALLOWED"

    if allowed:
        consecutive_state[0] += 1
    else:
        consecutive_state[0] = 0

    ready = consecutive_state[0] >= consecutive_required

    return {
        "timestamp_utc": _utc_now(),
        "symbol": symbol,
        "gate_result": gate_result,
        "allowed": allowed,
        "consecutive_allowed": consecutive_state[0],
        "consecutive_required": consecutive_required,
        "ready_to_revalidate": ready,
        "revalidate_cmd": REVALIDATE_CMD,
    }


def print_status(status: dict) -> None:
    sym = status["symbol"]
    gate = status["gate_result"].get("gate", "UNKNOWN")
    consec = status["consecutive_allowed"]
    required = status["consecutive_required"]
    ready = status["ready_to_revalidate"]

    if ready:
        print(f"[READY] {sym} gate={gate} (consecutive={consec}/{required}) -- run revalidation now")
        print(f"  cmd: {status['revalidate_cmd']}")
    elif status["allowed"]:
        print(f"[ALLOWED] {sym} gate={gate} (consecutive={consec}/{required}) -- need {required - consec} more checks")
    else:
        print(f"[WAITING] {sym} gate={gate} (consecutive_allowed={consec}/{required}) -- check again later")
        lanes = status["gate_result"].get("lanes", {})
        for lane_name, lane_info in lanes.items():
            if isinstance(lane_info, dict):
                active = lane_info.get("active", False)
                age = lane_info.get("age_sec")
                age_str = f", age={age}s" if age is not None else ""
                state_str = "active" if active else "clear"
                print(f"  {lane_name}: {state_str}{age_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll event lane gate; emit readiness report for post-regime re-validation."
    )
    parser.add_argument("--db", required=True, help="Path to microstructure.db")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol to check (default ETHUSDT)")
    parser.add_argument("--watch", action="store_true", help="Loop mode (poll on interval)")
    parser.add_argument("--interval", type=int, default=3600, help="Seconds between checks in --watch mode")
    parser.add_argument("--consecutive", type=int, default=1, help="Consecutive ALLOWED checks needed for READY (default 1)")
    parser.add_argument("--json", action="store_true", dest="json_out", help="Emit JSON to stdout")
    parser.add_argument("--out-json", default="", help="Write status payload to this JSON file")
    args = parser.parse_args()

    consecutive_state = [0]

    def run_once() -> dict:
        status = evaluate_status(args.db, args.symbol, args.consecutive, consecutive_state)
        if args.json_out:
            print(json.dumps(status, indent=2))
        else:
            print_status(status)
        if args.out_json:
            try:
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=2)
            except Exception as e:
                print(f"[warn] could not write --out-json: {e}", file=sys.stderr)
        return status

    if args.watch:
        while True:
            run_once()
            time.sleep(args.interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
