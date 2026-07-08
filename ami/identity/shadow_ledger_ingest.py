"""BATCH-P3-001: real event ingestion from the S34 shadow ledger.

Source: reports/shadow/s34_state_machine_shadow.jsonl (read-only). Each
logical trade has a ledger "id" with an OPEN row and, once resolved, a CLOSE
row. Protocol §8.1 is explicit that ledger rows are NOT independent events:
multiple ledger ids can share the same underlying market anchor (e.g. a
LONG_SILENCE route and a SHORT_NOISY route both attached to the same
liquidation trigger). This module dedupes on the true anchor key
(symbol, event_family, anchor_ts_ms) -- one ami_events row per anchor,
event_count = how many ledger ids collapsed into it. Per-route signal-level
detail (which routes attached, their individual outcomes) belongs in a
future signal_instances ingestion (Protocol §7.6) and is out of this
batch's scope.

FABLE-REVIEW-A F1: `event_family` here is the live route's `rule_name`
(route-scoped), not a settled market-event taxonomy (e.g. "ETH_SELL_CASCADE"
independent of which route watches it). Deciding the real taxonomy is a
Phase 6 W1 (cycle-integrity wave) question, not this batch's. Because
event_id = hash(symbol, event_family, anchor_ts_ms, source_artifact_id),
changing event_family for existing rows is structurally impossible without
also changing event_id -- there is no in-place-mutation code path. A future
taxonomy correction MUST bump EVENT_DEFINITION_VERSION and mint new
event_ids; existing event_definition_version="s34-shadow-ledger-v1" rows
are never rewritten in place (see test_generate_event_id_differs_on_
different_inputs for the id-changes-with-family proof).

source_quality = REAL_LIQUIDATION for every row here: anchor_ts_ms comes from
the live state machine's real liquidation-cascade detection (ETH SELL
threshold cross against the real liquidation feed), not a synthetic or
approximated trigger. This is distinct from the *cycle*-grouping layer
(cooldown_sensitivity.py), where the 6h-gap proxy label applies -- events
here are real; only the higher-level grouping of events into "cycles" is
currently only approximated (Protocol §8.1).

No fabrication: fields not directly present in the ledger (venue,
liquidation_side, feature_available_ts_ms, max_single_share,
displacement_bps, structural_location, collector_version) are left NULL.
Rows without a usable rule_name (observer-only "prop" monitoring ids) are
skipped -- symbol cannot be derived from them without guessing.

FABLE-REVIEW-A F2: a ledger "id" may appear with more than one OPEN row.
Real cause found while implementing this guard: a delayed/conditional-entry
observer starts MONITORING with entry_ts_ms=None, then legitimately gets a
second OPEN row once the entry actually triggers (entry_ts_ms/entry_price/
status change -- e.g. "prop_delay_ms" in the source row). That is an
expected update, not a conflict, so entry_ts_ms (and other fill-dependent
fields) are intentionally NOT in the identity check below. Only the fields
that define *which anchor this is* (anchor_ts_ms, rule_name, signal) must
never diverge across duplicate OPEN rows for the same id; if they ever do,
_load_logical_trades raises DuplicateOpenConflict instead of silently
keeping whichever happened to be read last.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from ami.identity.event_identity import SourceQuality, generate_event_id

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER_PATH = REPO_ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
EVENT_DEFINITION_VERSION = "s34-shadow-ledger-v1"

_DUP_OPEN_CRITICAL_FIELDS = ("anchor_ts_ms", "rule_name", "signal")


class DuplicateOpenConflict(Exception):
    """Raised when the same ledger id has two OPEN rows with divergent critical fields."""


def _symbol_from_rule_name(rule_name: str) -> str:
    # All current shadow routes are ETH-denominated (rule_name contains "ETH").
    # Kept as a defensive parse rather than a hardcoded constant so a future
    # non-ETH route ledger entry surfaces as an explicit ValueError, not a
    # silently wrong symbol.
    if "ETH" in rule_name.upper():
        return "ETHUSDT"
    raise ValueError(f"cannot determine symbol from rule_name={rule_name!r} -- no fabrication allowed")


def _load_logical_trades(path: Path) -> list[dict]:
    """One dict per ledger 'id': {id, open, close (optional)}."""
    by_id: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rid = row["id"]
            rec = by_id.setdefault(rid, {"id": rid})
            if row.get("event") == "OPEN":
                prev_open = rec.get("open")
                if prev_open is not None:
                    for field in _DUP_OPEN_CRITICAL_FIELDS:
                        if prev_open.get(field) != row.get(field):
                            raise DuplicateOpenConflict(
                                f"ledger id={rid!r} has divergent OPEN rows on {field!r}: "
                                f"{prev_open.get(field)!r} != {row.get(field)!r}"
                            )
                rec["open"] = row
            elif row.get("event") == "CLOSE":
                rec["close"] = row
    trades = []
    for rec in by_id.values():
        open_row = rec.get("open")
        if open_row is None:
            continue  # a CLOSE without an OPEN is not a valid anchor; skip rather than fabricate one
        if open_row.get("rule_name") is None:
            continue  # observer-only monitoring row; symbol not derivable, no fabrication
        trades.append(rec)
    return trades


def parse_shadow_ledger(path: Path = DEFAULT_LEDGER_PATH) -> list[dict]:
    """Read-only parse: dedupes ledger rows onto true (symbol, family, anchor) events."""
    trades = _load_logical_trades(path)
    try:
        source_artifact_id = str(path.relative_to(REPO_ROOT))
    except ValueError:
        source_artifact_id = str(path)

    groups: dict[tuple, list[dict]] = {}
    for rec in trades:
        open_row = rec["open"]
        symbol = _symbol_from_rule_name(open_row["rule_name"])
        anchor_ts_ms = int(open_row["anchor_ts_ms"])
        groups.setdefault((symbol, open_row["rule_name"], anchor_ts_ms), []).append(rec)

    events = []
    for (symbol, event_family, anchor_ts_ms), recs in groups.items():
        event_id = generate_event_id(symbol, event_family, anchor_ts_ms, source_artifact_id)

        entry_times = [int(r["open"]["entry_ts_ms"]) for r in recs if r["open"].get("entry_ts_ms") is not None]
        exit_times = [int(r["close"]["exit_ts_ms"]) for r in recs if r.get("close") is not None]
        # FABLE-REVIEW-A F6: running_notional is a cumulative counter observed
        # at each route's own OPEN moment; multiple routes attached to one
        # anchor observe it at slightly different offsets, so max() is used as
        # the representative (largest-observed, i.e. most-complete) reading of
        # the same underlying cascade size -- not a sum across routes.
        notionals = [r["open"]["running_notional"] for r in recs if r["open"].get("running_notional") is not None]
        signals = sorted({r["open"].get("signal") for r in recs if r["open"].get("signal")})

        event_start_ts_ms = min(entry_times) if entry_times else None
        event_end_ts_ms = max(exit_times) if exit_times else None
        duration_seconds = ((event_end_ts_ms - event_start_ts_ms) / 1000.0
                             if event_start_ts_ms is not None and event_end_ts_ms is not None else None)
        # aggregate-level censor: complete only if every logical trade attached
        # to this anchor has resolved (closed); otherwise at least one is still
        # being monitored as of this ledger snapshot.
        all_closed = all(r.get("close") is not None for r in recs)
        censor_status = "COMPLETED" if all_closed else "RIGHT_CENSORED"

        events.append({
            "event_id": event_id,
            "event_family": event_family,
            "symbol": symbol,
            "venue": None,
            "liquidation_side": None,
            "anchor_ts_ms": anchor_ts_ms,
            "event_start_ts_ms": event_start_ts_ms,
            "event_end_ts_ms": event_end_ts_ms,
            "feature_available_ts_ms": None,
            "notional": max(notionals) if notionals else None,
            "event_count": len(recs),
            "duration_seconds": duration_seconds,
            "max_single_share": None,
            "displacement_bps": None,
            "structural_location": None,
            "source_quality": SourceQuality.REAL_LIQUIDATION.value,
            "event_definition_version": EVENT_DEFINITION_VERSION,
            "censor_status": censor_status,
            "source_artifact_id": source_artifact_id,
            "route_version": ",".join(signals) if signals else None,
            "collector_version": None,
        })
    return events


def seed(conn, path: Path = DEFAULT_LEDGER_PATH, provenance: str = "batch-p3-001-shadow-ledger-ingest") -> int:
    now = int(time.time() * 1000)
    events = parse_shadow_ledger(path)
    for e in events:
        conn.execute(
            "INSERT INTO ami_events (event_id, event_family, symbol, venue, liquidation_side, "
            "anchor_ts_ms, event_start_ts_ms, event_end_ts_ms, feature_available_ts_ms, notional, "
            "event_count, duration_seconds, max_single_share, displacement_bps, structural_location, "
            "source_quality, event_definition_version, censor_status, source_artifact_id, "
            "route_version, collector_version, schema_version, provenance, created_ms, updated_ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(event_id) DO UPDATE SET event_end_ts_ms=excluded.event_end_ts_ms, "
            "event_count=excluded.event_count, duration_seconds=excluded.duration_seconds, "
            "censor_status=excluded.censor_status, route_version=excluded.route_version, "
            "updated_ms=excluded.updated_ms",
            (e["event_id"], e["event_family"], e["symbol"], e["venue"], e["liquidation_side"],
             e["anchor_ts_ms"], e["event_start_ts_ms"], e["event_end_ts_ms"], e["feature_available_ts_ms"],
             e["notional"], e["event_count"], e["duration_seconds"], e["max_single_share"],
             e["displacement_bps"], e["structural_location"], e["source_quality"],
             e["event_definition_version"], e["censor_status"], e["source_artifact_id"],
             e["route_version"], e["collector_version"], 3, provenance, now, now),
        )
    conn.commit()
    return len(events)


def main() -> None:
    from ami.warehouse.schema import DEFAULT_PATH, connect, init_schema

    conn = connect(DEFAULT_PATH)
    try:
        init_schema(conn)
        n = seed(conn)
        print(f"ingested {n} real events (deduped anchors) from shadow ledger (source_quality=REAL_LIQUIDATION)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
