# Codex Task 3: Implement execution/event_lane_gate.py (Shadow Mode)

## Context

Analysis is complete (see `docs/LIVE_INTEGRATION_PLAN.md` and
`docs/EVENT_LANE_GATE_PORT.md`). The validated pocket is:
- ETHUSDT, micro_edge_v3_passive_alpha, h=60, imb>=0.85
- Block lanes: book_proxy_pressure + volatility_burst

This task implements the gate helper in **shadow mode only** — it computes
and logs the gate decision but does NOT block any orders yet.

Reference implementation for lane detection logic:
- `tools/check_event_lanes.py` in the research repo (or read the copy at
  `eclipse_scalper-research/tools/check_event_lanes.py`) — copy the
  `_detect_book_proxy_pressure`, `_detect_volatility_burst`, and `_quantile`
  functions verbatim. Do not import from research tools.

---

## Step 1 — Create `execution/event_lane_gate.py`

Implement these functions exactly:

```python
def load_current_event_gate(
    *,
    db: str,
    symbol: str,
    lookback_min: int = 60,
    bucket_sec: int = 5,
    pocket_horizon_sec: int = 60,
    pocket_min_abs_imbalance: float = 0.85,
) -> dict:
    """
    Load recent buckets from microstructure DB and compute current lane state.
    Returns a gate payload dict (see EVENT_LANE_GATE_PORT.md for schema).
    Never raises — returns {"gate": "no_data", "allow_trade": True} on any error.
    """
```

```python
def should_block_event_gate(
    gate_payload: dict,
    *,
    symbol: str,
    rule_name: str,
    horizon_sec: int,
) -> tuple[bool, str, dict]:
    """
    Given a gate payload, decide if this entry should be blocked.
    Returns (blocked: bool, reason: str, details: dict).
    Only applies to ETHUSDT + micro_edge_v3_passive_alpha + h=60.
    Always returns (False, "gate_not_applicable", {}) for other pockets.
    """
```

```python
def applies_to_live_event_gate(
    symbol: str,
    rule_name: str,
    horizon_sec: int,
) -> bool:
    """
    Returns True only for ETHUSDT + micro_edge_v3_passive_alpha + h=60.
    """
    return (
        symbol.upper() == "ETHUSDT"
        and rule_name == "micro_edge_v3_passive_alpha"
        and int(horizon_sec) == 60
    )
```

### Gate payload schema

```python
{
    "symbol": str,
    "gate": "allowed" | "blocked" | "inactive" | "no_data",
    "allow_trade": bool,
    "blocked_lanes": list[str],   # e.g. ["book_proxy_pressure"]
    "reason": str,
    "latest_ts_ms": int | None,
    "latest_abs_imbalance": float | None,
    "lanes": {
        "book_proxy_pressure": {"rule_fired": bool, "severity": str},
        "volatility_burst":    {"rule_fired": bool, "severity": str},
    },
}
```

### DB loading

Load 1-sec or `bucket_sec`-sec bucket features from SQLite:
- Table: `agg_trades` (ts_ms, symbol, price, quantity, is_buyer_maker)
- Table: `mark_prices` (ts_ms, symbol, mark_price)
- Bucket: group by `(ts_ms / (bucket_sec*1000)) * (bucket_sec*1000)`
- Compute per-bucket: imbalance, trade_intensity (count * 60/bucket_sec),
  spread (|vwap - mark| / vwap), ret_1 (vwap return vs previous bucket)
- Use only the last `lookback_min` minutes

### Lane detection

Copy verbatim from `tools/check_event_lanes.py`:
- `_quantile(values, q)` — linear interpolation quantile
- `_detect_book_proxy_pressure(buckets)` — returns list[bool]
- `_detect_volatility_burst(buckets)` — returns list[bool]

Current-bucket gate: only look at the LAST bucket in the list.
Do NOT use stale summary state or recent_alert_count for blocking.

### Error handling

`load_current_event_gate` must never raise. Wrap in try/except:
```python
except Exception as e:
    return {"gate": "no_data", "allow_trade": True, "reason": f"error:{e}", ...}
```

---

## Step 2 — Add env flag support

Read from environment in `load_current_event_gate`:
```python
import os
GATE_ENABLED = os.getenv("ENTRY_EVENT_LANE_GATE_ENABLED", "0") == "1"
GATE_SHADOW  = os.getenv("ENTRY_EVENT_LANE_GATE_SHADOW", "1") == "1"
GATE_DB      = os.getenv("ENTRY_EVENT_LANE_GATE_DB", "")
GATE_LOOKBACK = int(os.getenv("ENTRY_EVENT_LANE_GATE_LOOKBACK_MIN", "60"))
GATE_BUCKET  = int(os.getenv("ENTRY_EVENT_LANE_GATE_BUCKET_SEC", "5"))
```

If `GATE_ENABLED=0`: `load_current_event_gate` returns `{"gate": "inactive", "allow_trade": True}` immediately.

---

## Step 3 — Hook into entry_loop.py (shadow only)

Find the point in `execution/entry_loop.py` where a micro_edge signal (`sig`)
is available and before order routing. Add ONE narrow block:

```python
# Event lane gate (shadow mode — logs only, does not block)
if event_lane_gate.applies_to_live_event_gate(symbol, rule_name, horizon_sec):
    _gate_db = os.getenv("ENTRY_EVENT_LANE_GATE_DB", str(getattr(bot.cfg, "DB_PATH", "")))
    if _gate_db:
        _gate = event_lane_gate.load_current_event_gate(db=_gate_db, symbol=symbol)
        _blocked, _reason, _details = event_lane_gate.should_block_event_gate(
            _gate, symbol=symbol, rule_name=rule_name, horizon_sec=horizon_sec
        )
        shadow = os.getenv("ENTRY_EVENT_LANE_GATE_SHADOW", "1") == "1"
        if _blocked:
            _log_event("entry.event_lane_gate", {
                "symbol": symbol, "rule_name": rule_name, "horizon_sec": horizon_sec,
                "gate_status": "blocked", "gate_reason": _reason,
                "blocking_lanes": _gate.get("blocked_lanes", []),
                "shadow": shadow, **_details,
            })
            if not shadow:
                continue  # actual block — only when SHADOW=0
        else:
            _log_event("entry.event_lane_gate", {
                "symbol": symbol, "rule_name": rule_name, "horizon_sec": horizon_sec,
                "gate_status": "allowed", "shadow": shadow,
            })
```

**Important**: place this AFTER the kill-switch check and AFTER feature bounds
checks. NEVER apply to reduce-only intents. If you cannot find the right
insertion point without risk, leave a `# TODO: insert gate here` comment and
describe the location in your PR description.

---

## Step 4 — Tests

Create `tests/test_event_lane_gate.py` with at least 4 tests:

1. `test_applies_to_live_event_gate_matches` — correct symbol/rule/horizon returns True
2. `test_applies_to_live_event_gate_no_match` — wrong symbol or horizon returns False
3. `test_should_block_returns_false_when_not_applicable` — gate not applicable returns (False, "gate_not_applicable", {})
4. `test_load_current_event_gate_no_data_returns_safe_default` — pass a non-existent DB path, confirm returns `{"gate": "no_data", "allow_trade": True}` without raising

Do NOT use a real DB in tests. Mock or pass a non-existent path for the no_data test.

---

## Deliverables

1. `execution/event_lane_gate.py`
2. `tests/test_event_lane_gate.py` (4+ tests, all passing)
3. Minimal edit to `execution/entry_loop.py` — shadow hook only (or TODO comment with location)

## Branch

`codex/feat/event-lane-gate-shadow`
Base: `main` in `eclipse_scalper` repo

## Hard constraints

- `ENTRY_EVENT_LANE_GATE_SHADOW=1` by default — no orders blocked until explicitly set to 0
- `ENTRY_EVENT_LANE_GATE_ENABLED=0` by default — gate inactive unless explicitly enabled
- Gate must NEVER apply to reduce-only intents
- Gate must come AFTER kill-switch check in entry_loop
- `load_current_event_gate` must never raise
- Do not import from research tools (`eclipse_scalper-research/`)
- Do not touch `risk/`, `brain/`, `execution/reconcile.py`, `execution/order_router.py`
