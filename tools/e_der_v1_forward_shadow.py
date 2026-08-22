"""Dedicated prospective frozen E-DER V1 shadow runner.

Consumes the versioned universe and new core forceOrder partitions. Exact kline
OPEN/CLOSE/quote-volume support is bootstrapped/refreshed from Binance public
REST into bounded RAM. It has no exchange-order code path.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import research_s34_echo_multilane_forward as support
from tools.e_der_forward_common import (
    CONFIG, DATA_ROOT, RUNTIME_ROOT, SHADOW_ROOT, append_jsonl, atomic_json, git_sha,
    latest_universe_snapshot, load_config, now_ms, role_state, sha256_file,
    scientific_forward_valid_since_ms, utc_now,
)
from tools.e_der_v1_frozen import frozen_timing
from tools.research_s34_echo_impact_elasticity_development import multiscale_decision
from tools.research_s34_knowable_anchor_continuation import reconstruct_anchors


STATE = RUNTIME_ROOT / "v1_forward_runner.json"
LEDGER = SHADOW_ROOT / "events.jsonl"
MINUTE_MS = 60_000
HOUR_MS = 3_600_000
IMPACT_KEYS = ["i1_v30", "i3_v30", "i5_v30", "i10_v30"]
PROTOCOL = "E_DER_V1_PROSPECTIVE_FORWARD_2026_08_21"
STATE_CONTRACT_VERSION = 1
SOURCE_CONTRACT = support.SOURCE_RESULT
SOURCE_CONTRACT_SHA = support.EXPECTED_SOURCE_SHA256.upper()


def blank_state() -> dict[str, Any]:
    return {"schema_version": 1, "state_contract_version": STATE_CONTRACT_VERSION,
            "protocol": PROTOCOL, "mode": "PAPER_SHADOW_NO_ORDERS",
            "created_at_utc": utc_now(), "forward_deployment_ms": now_ms(), "processed": {},
            "locks": {}, "pending": {}, "closed": {}, "cascade": {"end_ms": None, "id": None},
            "status": "STARTING", "last_error": None}


def load_state(path: Path = STATE) -> dict[str, Any]:
    if not path.exists():
        return blank_state()
    value = json.loads(path.read_text(encoding="utf-8"))
    # role_state() owns the operational schema_version in persisted runtime
    # state. Accept the already-deployed legacy state once, then preserve an
    # explicit runner contract version that the envelope cannot overwrite.
    version = value.get("state_contract_version")
    if version is None and value.get("protocol") == PROTOCOL:
        version = STATE_CONTRACT_VERSION
    if value.get("protocol") != PROTOCOL or version != STATE_CONTRACT_VERSION:
        raise RuntimeError("FORWARD_STATE_PROTOCOL_DRIFT")
    value["state_contract_version"] = STATE_CONTRACT_VERSION
    return value


def load_frozen_thresholds() -> tuple[list[str], dict[str, float]]:
    if sha256_file(SOURCE_CONTRACT) != SOURCE_CONTRACT_SHA:
        raise RuntimeError("FROZEN_THRESHOLD_CONTRACT_DRIFT")
    return support.load_contract()


def extract_forceorder(record: dict[str, Any]) -> dict[str, Any] | None:
    payload = record.get("payload", record)
    data = payload.get("data", payload) if isinstance(payload, dict) else None
    if not isinstance(data, dict) or data.get("e") != "forceOrder" or not isinstance(data.get("o"), dict):
        return None
    order = data["o"]
    try:
        price, quantity = float(order["p"]), float(order["q"])
        event_ms, trade_ms = int(data["E"]), int(order["T"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(price, quantity) <= 0 or not all(math.isfinite(x) for x in (price, quantity)):
        return None
    return {"ts_ms": event_ms, "trade_time_ms": trade_ms, "symbol": str(order["s"]).upper(),
            "side": str(order["S"]).upper(), "price": price, "quantity": quantity,
            "notional": price*quantity, "payload_hash": record.get("payload_sha256"),
            "universe_version": record.get("universe_version")}


def recent_forceorders(since_ms: int, until_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((DATA_ROOT / "forceorder_raw").glob("*/*.jsonl")):
        # Path date filtering is deliberately coarse; exact E filtering is below.
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = extract_forceorder(json.loads(line))
                except (json.JSONDecodeError, OSError):
                    continue
                if item is not None and since_ms <= item["ts_ms"] <= until_ms:
                    rows.append(item)
    rows.sort(key=lambda item: (item["ts_ms"], item["trade_time_ms"], item["symbol"],
                                item["price"], item["quantity"]))
    return rows


def load_persisted_klines(cache: dict[str, dict[int, dict[str, float]]],
                          symbols: list[str], since_ms: int) -> int:
    wanted = set(symbols); loaded = 0
    for path in sorted((DATA_ROOT / "kline_1m").glob("*/*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    index = json.loads(line).get("kline_index")
                except (json.JSONDecodeError, OSError):
                    continue
                if not isinstance(index, dict):
                    continue
                symbol, timestamp = str(index.get("symbol","")).upper(), int(index.get("open_time",-1))
                if symbol not in wanted or timestamp < since_ms:
                    continue
                value = {key:float(index[key]) for key in ("open","close","quote_volume")}
                prior = cache.setdefault(symbol, {}).get(timestamp)
                if prior is not None and prior != value:
                    raise RuntimeError(f"PERSISTED_KLINE_CONFLICT:{symbol}:{timestamp}")
                if prior is None:
                    cache[symbol][timestamp] = value; loaded += 1
    return loaded


def anchors_by_symbol(rows: list[dict[str, Any]], symbols: list[str],
                      thresholds: dict[str, float]) -> dict[str, list[Any]]:
    grouped = {symbol: [] for symbol in symbols}
    for row in rows:
        if row["side"] == "SELL" and row["symbol"] in grouped:
            grouped[row["symbol"]].append(row)
    return {symbol: reconstruct_anchors(grouped[symbol], bucket_sec=300, min_gap_sec=900,
            thresholds=(thresholds[symbol],), accel_window_sec=30) for symbol in symbols}


def eligible_e_der(row: dict[str, Any]) -> tuple[bool, tuple[float, int] | None]:
    decision = multiscale_decision(row, IMPACT_KEYS)
    metric = row.get("i3_v30")
    eligible = bool(row.get("qualified") and decision is not None and decision[0] < 0 and
                    metric is not None and float(metric["q_echo"]) >= float(metric["q_parent"]) and
                    int(row.get("pre_parent_stress_count", 0)) >= 2)
    return eligible, decision


def product_cohort(metadata: dict[str, Any]) -> str:
    underlying = str(metadata.get("underlyingType") or "").upper()
    subtypes = {str(x).upper() for x in metadata.get("underlyingSubType") or []}
    if underlying == "COIN" and not subtypes.intersection({"STOCK", "INDEX", "COMMODITY"}):
        return "NATIVE_CRYPTO"
    mapping = {"STOCK": "OTHER_TRADFI", "INDEX": "BROAD_INDEX_ETF",
               "COMMODITY": "COMMODITY_ENERGY_INDUSTRIAL"}
    return next((mapping[x] for x in sorted(subtypes) if x in mapping), "OTHER_TRADFI")


def cascade_id(state: dict[str, Any], anchor_ms: int, boundary_ms: int) -> str:
    current_end = state["cascade"].get("end_ms")
    if current_end is None or anchor_ms > int(current_end):
        state["cascade"] = {"id": f"CASCADE:{anchor_ms}", "end_ms": boundary_ms}
    else:
        state["cascade"]["end_ms"] = max(int(current_end), boundary_ms)
    return str(state["cascade"]["id"])


def make_event(row: dict[str, Any], state: dict[str, Any], universe: dict[str, Any]) -> dict[str, Any]:
    anchor = int(row["anchor_ts_ms"])
    base, entry, boundary = frozen_timing(anchor)
    metadata = next(item for item in universe["symbols"] if item["symbol"] == row["symbol"])
    metric = row["i3_v30"]
    votes = {key: (-1 if row[key]["G"] > 0 else 1) for key in IMPACT_KEYS}
    event_id = f"E:{row['symbol']}:{anchor}"
    onboard = metadata.get("onboardDate")
    return {"event": "DETECTED", "protocol": PROTOCOL, "classification": "PROSPECTIVE_FORWARD",
            "event_id": event_id, "symbol": row["symbol"], "anchor_ts": anchor,
            "base_ms": base, "entry_ms": entry, "boundary_ms": boundary,
            "parent_id": f"P:{row['symbol']}:{row['parent_ts_ms']}",
            "parent_ts_ms": row["parent_ts_ms"], "echo_id": row["anchor_identity"],
            "q_parent": metric["q_parent"], "q_echo": metric["q_echo"],
            "prior_stress_count": row["pre_parent_stress_count"], "multiscale_votes": votes,
            "multiscale_vote_sum": sum(votes.values()), "status": "AWAITING_ENTRY",
            "entry_open": None, "boundary_open": None, "gross_return_bps": None, "net_return_bps": None,
            "cost_bps": 10.0, "data_quality_status": "PENDING_EXACT_OPENS",
            "universe_version": universe["universe_version"], "code_sha": git_sha(),
            "contract_sha": sha256_file(CONFIG), "cascade_id": cascade_id(state, anchor, boundary),
            "product_cohort": product_cohort(metadata),
            "listing_age_days": ((anchor-int(onboard))/86_400_000 if onboard else None),
            "session_state": "ALWAYS_OPEN" if product_cohort(metadata) == "NATIVE_CRYPTO" else "UNAVAILABLE",
            "paper_only": True, "real_order_sent": False, "created_at_utc": utc_now()}


def exact_open(cache: dict[str, dict[int, dict[str, float]]], symbol: str, timestamp: int) -> float | None:
    value = cache.get(symbol, {}).get(timestamp)
    if not value:
        return None
    price = float(value["open"])
    return price if price > 0 and math.isfinite(price) else None


def mature(state: dict[str, Any], cache: dict[str, dict[int, dict[str, float]]], now: int) -> None:
    for event_id, event in list(state["pending"].items()):
        if event["status"] == "AWAITING_ENTRY" and now >= int(event["entry_ms"]):
            value = exact_open(cache, event["symbol"], int(event["entry_ms"]))
            if value is not None:
                event.update({"event": "ENTRY", "status": "OPEN", "entry_open": value,
                              "data_quality_status": "ENTRY_EXACT_OPEN", "updated_at_utc": utc_now()})
                append_jsonl(LEDGER, event)
            elif now > int(event["entry_ms"]) + 2*MINUTE_MS:
                event.update({"event": "ENTRY_UNAVAILABLE", "status": "UNAVAILABLE",
                              "data_quality_status": "EXACT_ENTRY_OPEN_UNAVAILABLE", "updated_at_utc": utc_now()})
                append_jsonl(LEDGER, event)
        if event["status"] == "OPEN" and now >= int(event["boundary_ms"]):
            value = exact_open(cache, event["symbol"], int(event["boundary_ms"]))
            if value is not None:
                gross = math.log(value/float(event["entry_open"]))*10_000
                event.update({"event": "CLOSE", "status": "CLOSED", "boundary_open": value,
                              "gross_return_bps": gross, "net_return_bps": gross-10.0,
                              "data_quality_status": "COMPLETE_EXACT_OPENS", "updated_at_utc": utc_now()})
                append_jsonl(LEDGER, event)
            elif now > int(event["boundary_ms"]) + 2*MINUTE_MS:
                event.update({"event": "BOUNDARY_UNAVAILABLE", "status": "UNAVAILABLE",
                              "data_quality_status": "EXACT_BOUNDARY_OPEN_UNAVAILABLE", "updated_at_utc": utc_now()})
                append_jsonl(LEDGER, event)
        if event["status"] in {"CLOSED", "UNAVAILABLE"}:
            state["closed"][event_id] = event["status"]
            del state["pending"][event_id]


def run_cycle(state: dict[str, Any], cache: dict[str, dict[int, dict[str, float]]], now: int | None = None) -> dict[str, Any]:
    now = now or now_ms()
    _, universe = latest_universe_snapshot()
    frozen_symbols, thresholds = load_frozen_thresholds()
    active = set(universe["symbol_set"])
    evaluable = sorted(active.intersection(frozen_symbols))
    unavailable = sorted(active.difference(frozen_symbols))
    valid_since = scientific_forward_valid_since_ms()
    persisted_loaded = load_persisted_klines(
        cache, ["BTCUSDT", *[s for s in evaluable if s != "BTCUSDT"]], now-8*24*HOUR_MS)
    btc_history = cache.get("BTCUSDT", {})
    bootstrap_needed = not btc_history or min(btc_history) > now-7*24*HOUR_MS
    price_errors = support.refresh_prices(
        cache, ["BTCUSDT", *[s for s in evaluable if s != "BTCUSDT"]], now,
        bootstrap=bootstrap_needed, finalized_only=True,
        strict_existing_parity=True, finalization_lag_ms=MINUTE_MS)
    force_start = max(now-8*HOUR_MS, valid_since) if valid_since is not None else now+1
    rows = recent_forceorders(force_start, now)
    anchors = anchors_by_symbol(rows, evaluable, thresholds)
    candidates: list[dict[str, Any]] = []
    for symbol in evaluable:
        for anchor in anchors[symbol]:
            timestamp = int(anchor.anchor_ts_ms)
            identity = f"{symbol}:{anchor.event_id}:{timestamp}"
            if (valid_since is None or timestamp < valid_since or
                    timestamp < int(state["forward_deployment_ms"]) or identity in state["processed"]):
                continue
            row = support.build_candidate(symbol, anchor, anchors[symbol], cache)
            if row is not None:
                candidates.append(row)
                state["processed"][identity] = timestamp
    for row in sorted(candidates, key=lambda x: (x["anchor_ts_ms"], x["anchor_identity"])):
        if not row.get("qualified"):
            continue
        symbol, timestamp = row["symbol"], int(row["anchor_ts_ms"])
        prior_lock = state["locks"].get(symbol)
        if prior_lock is not None and timestamp-int(prior_lock) < 4*HOUR_MS:
            continue
        state["locks"][symbol] = timestamp
        eligible, _ = eligible_e_der(row)
        if not eligible:
            continue
        event = make_event(row, state, universe)
        if event["event_id"] not in state["pending"] and event["event_id"] not in state["closed"]:
            state["pending"][event["event_id"]] = event
            append_jsonl(LEDGER, event)
    mature(state, cache, now)
    if valid_since is None:
        state["status"] = "FORWARD_SOURCE_UNOBSERVABLE"
    else:
        state["status"] = "RUNNING_PAPER_SHADOW" if not price_errors else "DEGRADED_PRICE_SUPPORT"
    state["last_error"] = None
    state["updated_at_utc"] = utc_now()
    state["source"] = {"forceorder": "forward_v2_append_only_partitions",
                       "klines": "FORWARD_V2_PERSISTED_PLUS_FINALIZED_BINANCE_PUBLIC_REST_RAM_RECOVERY",
                       "rest_kline_semantics": "CLOSED_PLUS_60S_GRACE_STRICT_FINALIZED_PARITY",
                       "persisted_kline_rows_loaded_this_cycle":persisted_loaded,
                       "forceorder_rows_in_8h": len(rows), "price_errors": price_errors,
                       "scientific_forward_valid_since_ms":valid_since}
    state["universe_version"] = universe["universe_version"]
    state["symbol_count"] = len(active)
    state["evaluable_frozen_threshold_symbols"] = len(evaluable)
    state["unavailable_frozen_threshold_symbols"] = unavailable
    state["pending_n"] = len(state["pending"])
    atomic_json(STATE, {**state, **role_state("e_der_v1_forward_runner", status=state["status"],
        last_source_timestamp=max((r["ts_ms"] for r in rows), default=None), last_event="CYCLE",
        universe_version=universe["universe_version"], symbol_count=len(active),
        data_quality_state=("FORWARD_SOURCE_UNOBSERVABLE" if valid_since is None else
                            ("GREEN" if not price_errors else "AMBER")), last_error=None,
        restart_count=state.get("restart_count",0))})
    return state


def main(argv: list[str] | None = None) -> int:
    load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=30)
    args = parser.parse_args(argv)
    state, cache = load_state(), {}
    while True:
        try:
            run_cycle(state, cache)
            print(json.dumps({"status":state["status"],"pending":len(state["pending"]),
                              "symbol_count":state["symbol_count"]}, sort_keys=True), flush=True)
        except Exception as exc:
            state["status"] = "ERROR_RETRYING"
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            state["restart_count"] = int(state.get("restart_count",0))+1
            atomic_json(STATE, {**state, **role_state("e_der_v1_forward_runner", status=state["status"],
                                                     data_quality_state="RED", last_error=state["last_error"],
                                                     restart_count=state["restart_count"])})
            if args.once:
                return 2
        if args.once:
            return 0
        time.sleep(max(10,args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
