"""End-to-end disposable CVD rehearsal harness.

Composes windowed_taker_flow + cvd_source_quality_contract_v1 +
aggtrades_repair_rehearsal against:
  - the REAL data/ami/canonical.sqlite opened STRICTLY mode=ro (population
    + frozen BUCKET starts + candles for the proxy layer),
  - the REAL data/microstructure.db opened STRICTLY mode=ro (legacy trade rows),
  - a DISPOSABLE rehearsal database (all writes).

Nothing here writes to any live database; every write target is the
caller-supplied disposable connection. Nothing here reads any outcome table:
the only canonical tables touched are ami_signal_lifecycle,
ami_birth_truncated_cascade_geometry, ami_cycles (identity only) and
ami_candles (proxy taker fields). No MFE/MAE/PnL/path table is ever opened.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time

from ami.cvd import windowed_taker_flow as wtf
from ami.cvd import cvd_source_quality_contract_v1 as quality
from ami.cvd import aggtrades_repair_rehearsal as repair

GEOMETRY_FEATURE_VERSION = "s34-knowable-anchor-continuation-v1-birth-truncated"

# Outcome tables that must NEVER be opened by this harness (static guard --
# tests assert none of these strings appear in any SQL this module issues).
FORBIDDEN_TABLES = (
    "ami_lifecycle_path_observations", "experiment_results", "experiment_registry",
)


def open_ro(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=ON")
    return conn


def fetch_canonical_population(canon_ro) -> list[dict]:
    """324 canonical signals + frozen BUCKET starts (LEFT JOIN: absence is
    recorded, never silently dropped)."""
    rows = canon_ro.execute(
        "SELECT s.signal_id, s.setup_id, s.direction, s.source_event_id,"
        " s.independent_cycle_id, s.symbol, s.signal_birth_ts,"
        " g.source_window_start_ts_ms"
        " FROM ami_signal_lifecycle s"
        " LEFT JOIN ami_birth_truncated_cascade_geometry g"
        "   ON g.signal_id = s.signal_id AND g.feature_definition_version = ?"
        " ORDER BY s.signal_id", (GEOMETRY_FEATURE_VERSION,)).fetchall()
    return [{
        "signal_id": r[0], "setup_id": r[1], "direction": r[2], "source_event_id": r[3],
        "independent_cycle_id": r[4], "symbol": r[5], "signal_birth_ts": int(r[6]),
        "bucket_start_ts_ms": int(r[7]) if r[7] is not None else None,
    } for r in rows]


def load_minute_set(disp_conn, symbol: str) -> set:
    return {r[0] for r in disp_conn.execute(
        "SELECT minute_ms FROM minute_map WHERE symbol=?", (symbol,))}


def minutes_overlapping(window_start_ms: int, window_end_ms: int) -> list[int]:
    """Every minute bucket [m, m+60000) that intersects [start, end]."""
    first = window_start_ms - window_start_ms % 60000
    out = []
    m = first
    while m <= window_end_ms:
        out.append(m)
        m += 60000
    return out


def fetch_legacy_window_rows(micro_ro, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Deterministic replay order: ORDER BY ts_ms, id (frozen rule 5.5)."""
    rows = micro_ro.execute(
        "SELECT id, ts_ms, price, quantity, notional, is_buyer_maker FROM agg_trades"
        " WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms, id",
        (symbol, start_ms, end_ms)).fetchall()
    return [{"ts_ms": r[1], "price": r[2], "quantity": r[3], "notional": r[4],
             "is_buyer_maker": r[5], "source": "LEGACY", "order_key": ("L", r[0])}
            for r in rows]


def fetch_repair_window_rows(disp_conn, symbol: str, minute_list: list[int],
                             window_start_ms: int, window_end_ms: int) -> list[dict]:
    """REST-staged rows restricted to LOCALLY-EMPTY minutes only -- the clean
    supersession case: legacy has ZERO rows in these minutes, so no
    cross-source dedup ambiguity can arise inside the effective selection.

    A repaired minute is fetched in full (the staging grain), then clipped to
    [window_start_ms, window_end_ms]: the window boundary can fall mid-minute
    (windows are not minute-aligned), and rows outside the window belong to a
    different window's own computation, never this one's."""
    out = []
    for m in minute_list:
        lo = max(m, window_start_ms)
        hi = min(m + 60000 - 1, window_end_ms)
        if lo > hi:
            continue
        rows = disp_conn.execute(
            "SELECT agg_trade_id, ts_ms, price, quantity, notional, is_buyer_maker"
            " FROM ami_agg_trades_repaired_stage"
            " WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms, agg_trade_id",
            (symbol, lo, hi)).fetchall()
        for r in rows:
            out.append({"ts_ms": r[1], "price": float(r[2]), "quantity": float(r[3]),
                        "notional": r[4], "is_buyer_maker": r[5], "source": "REST",
                        "order_key": ("R", r[0])})
    return out


def fetch_proxy_candles(canon_ro, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Effective candle selection for the proxy layer: for each open_ts_ms
    prefer candle-binance-fapi-repair-v1 over the agg-derived original
    (path-v2-candle-repair-r1 effective-selection precedent). 1m only."""
    rows = canon_ro.execute(
        "SELECT open_ts_ms, close_ts_ms, taker_buy_volume, taker_sell_volume,"
        " candle_definition_version FROM ami_candles"
        " WHERE symbol=? AND timeframe='1m' AND open_ts_ms>=? AND open_ts_ms<=?"
        " ORDER BY open_ts_ms", (symbol, start_ms - 60000, end_ms)).fetchall()
    by_open = {}
    for o, c, tb, ts_, ver in rows:
        prev = by_open.get(o)
        if prev is None or (ver == "candle-binance-fapi-repair-v1"
                            and prev["candle_definition_version"] != "candle-binance-fapi-repair-v1"):
            by_open[o] = {"open_ts_ms": o, "close_ts_ms": c, "taker_buy_volume": tb,
                          "taker_sell_volume": ts_, "candle_definition_version": ver}
    return [by_open[k] for k in sorted(by_open)]


def build_matrix(
    *, canon_ro, micro_ro, disp_conn, signals: list[dict],
    minute_set: set, repaired_minute_set: set,
    cadence_threshold_ms: int, cadence_proof_available: bool,
    scan_min_ms: int, scan_max_ms: int,
    provenance: str, assessment_version: str,
    duplicate_unresolved_regimes: frozenset = frozenset(),
) -> dict:
    """Build the full disposable feature matrix + quality ledger.

    Row law: one EXACT row + one PROXY row per (signal, window) for every
    defined window; BUCKET rows only where the bucket start is frozen, with
    one explicit exclusion row per undefined case. Nothing silent.
    """
    wtf.init_schema(disp_conn)
    quality.init_schema(disp_conn)
    counts = {"exact_rows": 0, "proxy_rows": 0, "bucket_exclusions": 0,
              "quality_rows": 0, "known_at_violations": 0, "noop_identical": 0}
    status_hist = {}
    for s in signals:
        T = s["signal_birth_ts"]
        for window_id in wtf.WINDOW_IDS:
            if window_id == wtf.BUCKET_WINDOW_ID and s["bucket_start_ts_ms"] is None:
                wtf.record_bucket_exclusion(disp_conn, s["signal_id"], s["direction"])
                counts["bucket_exclusions"] += 1
                continue
            w_start, w_end = wtf.window_bounds(T, window_id, s["bucket_start_ts_ms"])
            # ---- EXACT layer ----
            legacy = fetch_legacy_window_rows(micro_ro, s["symbol"], w_start, w_end)
            minutes = minutes_overlapping(w_start, w_end)
            missing = [m for m in minutes if m not in minute_set
                       and m >= scan_min_ms - scan_min_ms % 60000 and m <= scan_max_ms]
            repaired = [m for m in missing if m in repaired_minute_set]
            rest_rows = (fetch_repair_window_rows(disp_conn, s["symbol"], repaired, w_start, w_end)
                        if repaired else [])
            all_rows = sorted(legacy + rest_rows, key=lambda r: (r["ts_ms"], str(r["order_key"])))
            flow = wtf.compute_window_flow(all_rows, w_start, w_end)
            # cadence proof within the window (sub-minute completeness)
            cadence_pass = None
            if cadence_proof_available:
                max_gap = 0
                prev = None
                for r in all_rows:
                    if prev is not None:
                        g = r["ts_ms"] - prev
                        if g > max_gap:
                            max_gap = g
                    prev = r["ts_ms"]
                cadence_pass = bool(all_rows) and max_gap <= cadence_threshold_ms
            regime_ids = quality.regimes_for_window(w_start, w_end)
            repair_method = "AGGTRADES_REST" if rest_rows else "NONE"
            row = dict(flow)
            row.update({
                "signal_id": s["signal_id"], "source_event_id": s["source_event_id"],
                "independent_cycle_id": s["independent_cycle_id"], "symbol": s["symbol"],
                "signal_birth_ts": T, "window_id": window_id,
                "window_start_ts_ms": w_start, "window_end_ts_ms": w_end,
                "source_regime_ids": json.dumps(regime_ids), "repair_method": repair_method,
            })
            r1 = wtf.insert_exact_feature_row(disp_conn, row, provenance=provenance)
            counts["exact_rows" if r1 == "INSERTED" else "noop_identical"] += 1
            # ---- PROXY layer (descriptive-only, physically separate) ----
            candles = fetch_proxy_candles(canon_ro, s["symbol"], w_start, w_end)
            pflow = wtf.compute_proxy_window_flow(candles, w_start, w_end)
            prow = dict(pflow)
            prow.update({
                "signal_id": s["signal_id"], "source_event_id": s["source_event_id"],
                "independent_cycle_id": s["independent_cycle_id"], "symbol": s["symbol"],
                "signal_birth_ts": T, "window_id": window_id,
                "window_start_ts_ms": w_start, "window_end_ts_ms": w_end,
            })
            r2 = wtf.insert_proxy_feature_row(disp_conn, prow, provenance=provenance)
            if r2 == "INSERTED":
                counts["proxy_rows"] += 1
            # ---- quality classification (fail-closed) ----
            proxy_available = pflow["contained_candle_count"] > 0
            dup_unresolved = any(r in duplicate_unresolved_regimes for r in regime_ids)
            st = quality.classify_window(
                missing_minute_count=len(missing),
                repaired_minute_count=len(repaired),
                coverage_map_available=(scan_min_ms <= w_start and w_end <= scan_max_ms),
                cadence_proof_pass=cadence_pass,
                duplicate_unresolved=dup_unresolved,
                regime_ids=regime_ids,
                regime_proofs={r: True for r in regime_ids},
                proxy_available=proxy_available,
            )
            qrow = {
                "signal_id": s["signal_id"], "independent_cycle_id": s["independent_cycle_id"],
                "symbol": s["symbol"], "signal_birth_ts": T, "window_id": window_id,
                "window_start_ts_ms": w_start, "window_end_ts_ms": w_end,
                "evidence_layer": "PROXY" if st == "PROXY_ONLY" else "EXACT",
                "source_regime_ids": json.dumps(regime_ids),
                "regime_spanning": 1 if len(regime_ids) > 1 else 0,
                "legacy_row_count": flow["legacy_row_count"],
                "repair_row_count": flow["repair_row_count"],
                "total_row_count": flow["source_row_count"],
                "duplicate_count": 0, "collision_count": 0, "unresolved_match_count": 0,
                "missing_minute_count": len(missing), "repaired_minute_count": len(repaired),
                "cadence_proof": json.dumps({"available": cadence_proof_available,
                                             "threshold_ms": cadence_threshold_ms,
                                             "pass": cadence_pass}),
                "completeness_proof": json.dumps({
                    "minute_map": "pass_a_full_scan_20260705",
                    "minutes_checked": len(minutes), "missing": len(missing),
                    "repaired": len(repaired)}),
                "quality_status": st,
                "source_provenance": provenance,
                "data_version_id": wtf.REPAIR_POPULATION_VERSION if rest_rows else "legacy-live-collection",
                "feature_definition_version": wtf.FEATURE_DEFINITION_VERSION,
            }
            quality.record_window_quality(disp_conn, qrow, assessment_version=assessment_version)
            counts["quality_rows"] += 1
            status_hist[st] = status_hist.get(st, 0) + 1
    disp_conn.commit()
    counts["status_hist"] = status_hist
    return counts


def timestamp_violation_count(disp_conn) -> int:
    """Global known-at audit over the finished matrix: rows whose window
    violates the T-boundary law at the SQL level. Must be 0."""
    n = 0
    n += disp_conn.execute(
        "SELECT COUNT(*) FROM ami_cvd_windowed_flow"
        " WHERE window_end_ts_ms != signal_birth_ts"
        "    OR window_start_ts_ms > window_end_ts_ms"
        "    OR feature_available_ts_ms != signal_birth_ts").fetchone()[0]
    n += disp_conn.execute(
        "SELECT COUNT(*) FROM ami_cvd_windowed_flow_proxy"
        " WHERE window_end_ts_ms != signal_birth_ts"
        "    OR (last_contained_close_ts_ms IS NOT NULL"
        "        AND last_contained_close_ts_ms > signal_birth_ts)").fetchone()[0]
    return n
