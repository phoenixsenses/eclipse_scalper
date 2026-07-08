"""PHASE 7B-0 / 7B-0.1 / 7B-CANON: Path/MFE/MAE/progress observation engine.

[PHASE 7B-CANON, APPLIED 2026-07-04] `freeze_and_record()` has now been run
against the real data/ami/canonical.sqlite (via
ami.lifecycle.path_canonical_migration.run_canonical_migration(), see
MIGRATION_LOG.md) -- 1080 rows backfilled for the 270 existing signals x 4
horizons. `conn` here is still ALWAYS a caller-supplied connection (this
module itself never opens ami.warehouse.schema.DEFAULT_PATH or calls
init_schema()) -- the deliberate real-DB write happens one layer up, in the
explicit migration entry point, never as a side effect of importing or
calling this module's functions directly.

NO_PHASE_8 / NO_OBSERVER_START / NO_BINDING_CHANGE / NO_ORDER /
NO_PERMISSION_CHANGE / NO_LIVE_SHADOW_CHANGE: this module computes
descriptive, fixed-horizon path measurements only. It never derives a
terminal_ts/invalidation_ts/last_confirmation_ts, never suggests an
entry/exit/stop, and 270 OPEN signals are never treated as active trades --
count_effective_closed_signals() (ami/lifecycle/canonical_schema.py) is 0
today, so no interval/duration claim is made anywhere here; every
observation is against a FIXED horizon frozen verbatim from
ami.research.w4_post_event_path_taxonomy.PATH_HORIZONS_MS
(scalp_30m/scalp_1h/swing_4h/swing_24h), never "until the signal closed".

[PHASE 7B-0.1 semantic closure] Two status axes, not one (see path_schema.py's
module docstring for the full taxonomy and the DB-level CHECK that couples
them): `observation_status` answers "is the price PATH computable" and
`volatility_status` answers "is the vol-normalization baseline usable" --
these are independent questions. A signal whose path is perfectly clean but
whose 60-candle realized-vol baseline is degenerate/insufficient is
observation_status=OK / volatility_status=INVALID_VOLATILITY_BASELINE, with
endpoint_return_bps/mfe_bps/mae_bps/timing/horizon_outcome_class all
populated -- only the 3 *_anchor_vol_units fields are NULL.

[PHASE 7B-0.1 semantic closure] Zero-extremum timing: the reference point
(t=0, signal_birth_ts itself) is a legitimate member of the path -- it always
trivially contributes a candidate (0.0, signal_birth_ts) to both the
favorable and adverse excursion series before the extremum is taken. This
means mfe_bps/mae_bps are ALREADY guaranteed non-negative/non-positive by
construction (the explicit round(max(0.0, ...))/round(min(0.0, ...)) calls
below are now a redundant but harmless defense-in-depth, matching the DB
CHECK's own belt-and-suspenders philosophy) -- and, critically,
time_to_mfe_ms/time_to_mae_ms are 0 (not some arbitrary real candle's
timestamp) whenever the real path never actually reaches a favorable/adverse
excursion. Ties (multiple points, real or virtual, sharing the extremal
value) are broken by EARLIEST timestamp -- never by iteration/insertion order.

FIELD-LEVEL PROVENANCE: see ami/lifecycle/path_field_provenance.py -- the
per-field DETERMINISTIC_HISTORICAL_SAFE/HISTORICAL_PROXY mapping (with
derivation_method and is_proxy) lives there, written into the EXISTING
ami_lifecycle_field_provenance table (never module-docstring-only).

COUNTING DISCIPLINE (operator lock #8): rows in
ami_lifecycle_path_observations are SIGNAL-level (270 signals x 4 horizons =
1080 max rows). This is deliberately NOT deduplicated by event -- a
LONG_SILENCE and a SHORT_NEITHER signal on the SAME source_event_id are two
independent trading hypotheses over the SAME price path and legitimately get
two independent rows (this is exactly what the LONG/SHORT sign-symmetry
proof needs: same path, opposite-signed mfe/mae). What must NOT happen is
treating 270 (or 1080) as the independent-sample-size denominator for any
future statistical test over this table -- that denominator is
independent_cycle_id (ami_signal_lifecycle.independent_cycle_id), exactly
as W1/W4/W6/W7A already established. compute_all()/freeze_and_record()
below report BOTH the raw row count and the distinct independent_cycle_id
count so this distinction is never silently lost. state_age (a
source_event_id-level quantity, one computation fanned out to both routes
of a paired event) is explicitly OUT OF SCOPE for this table -- it is not
one of the operator-specified columns; if/when a future batch adds it, the
same source_event_id-dedup-then-fan-out discipline applies.
"""
from __future__ import annotations

import bisect
import hashlib
import math
import time

from ami.lifecycle.canonical_schema import ObservationMode
from ami.lifecycle.path_schema import PATH_DEFINITION_VERSION, PATH_SCHEMA_VERSION
from ami.research.w4_post_event_path_taxonomy import PATH_HORIZONS_MS, classify_path

CANDLE_MS = 60_000  # 1m grid, verified against real ami_candles (open_ts_ms = close_ts_ms - 60000)
VOL_WINDOW_CANDLES = 60  # same window W7A's realized_vol_clock uses at the anchor
MIN_VOL_LOG_RETURNS = VOL_WINDOW_CANDLES - 1  # 59, matches W7A

_VALID_DIRECTIONS = ("LONG", "SHORT")

_NULL_PATH_FIELDS = {
    "reference_price": None, "reference_price_ts": None, "effective_path_start_ts": None,
    "endpoint_return_bps": None, "mfe_bps": None, "mae_bps": None,
    "time_to_mfe_ms": None, "time_to_mae_ms": None, "intrabar_order_status": None,
    "horizon_outcome_class": None, "expected_candle_count": 0, "observed_candle_count": 0,
    "gap_count": 0, "candle_definition_version": None,
}
_NULL_VOL_FIELDS = {
    "realized_vol_at_anchor": None, "endpoint_return_anchor_vol_units": None,
    "mfe_anchor_vol_units": None, "mae_anchor_vol_units": None,
}


def _observation_id(signal_id: str, horizon_name: str, path_definition_version: str) -> str:
    key = f"{signal_id}|{horizon_name}|{path_definition_version}"
    return "OBS-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


class _CandleOHLCIndex:
    """Sorted 1m candle series (open_ts_ms/close_ts_ms/high/low/close/
    data_quality/candle_definition_version), known-at-safe by construction
    (every row is already a CLOSED candle -- Phase 4 candle_builder never
    stores a still-forming bucket)."""

    def __init__(self, rows: list[dict]):
        ordered = sorted(rows, key=lambda c: c["close_ts_ms"])
        self._rows = ordered
        self._open_ts = [c["open_ts_ms"] for c in ordered]
        self._close_ts = [c["close_ts_ms"] for c in ordered]

    def maturity_cutoff_ts_ms(self) -> int | None:
        return self._close_ts[-1] if self._close_ts else None

    def reference_before(self, ts_ms: int) -> tuple[float, int] | None:
        """Last candle whose close_ts_ms <= ts_ms (fully closed at/before ts_ms)."""
        i = bisect.bisect_right(self._close_ts, ts_ms) - 1
        if i < 0:
            return None
        r = self._rows[i]
        return r["close"], r["close_ts_ms"]

    def path_window(self, start_ts_ms_inclusive_open: int, end_ts_ms_inclusive_close: int) -> list[dict]:
        """Candles with open_ts_ms >= start (inclusive) AND close_ts_ms <= end
        (inclusive) -- i.e. fully CONTAINED in [start, end], never a candle
        straddling either boundary.

        [PHASE 7B-1 perf fix] `hi` is now found via bisect (O(log n)) instead
        of a linear break-scan over `self._rows[lo:]` -- that unbounded slice
        was itself O(n) (Python copies the WHOLE tail of the list to build
        the slice object, regardless of how quickly the loop then breaks),
        which only stayed cheap by accident for the 270-signal canonical
        population (whose signal_birth_ts values cluster late in the candle
        history, keeping `lo` large and the tail slice small). It became a
        real bottleneck the first time a caller (W8-HOLD-BASELINE's matched
        negative-control construction) called this thousands of times across
        candle-universe slots spread evenly across the FULL ~4.5-month
        history (many with `lo` near 0, each copying nearly all 173k rows).
        `self._rows[lo:hi]` is bounded by the actual window size on both
        ends, average-case O(window length), and returns byte-identical
        results (both `_open_ts` and `_close_ts` are monotonically
        non-decreasing in the same row order, so every index in [lo, hi)
        satisfies both conditions and no index outside it does -- proven by
        the existing test suite re-run unchanged after this fix)."""
        lo = bisect.bisect_left(self._open_ts, start_ts_ms_inclusive_open)
        hi = bisect.bisect_right(self._close_ts, end_ts_ms_inclusive_close)
        return self._rows[lo:hi]


def _realized_vol_at(index: _CandleOHLCIndex, anchor_ts_ms: int) -> float | None:
    """Same formula W7A's _window_metrics()/compute_market_clocks() use for
    realized_vol: sqrt(mean(log-return^2)) over the 60 candles at-or-before
    anchor_ts_ms, gated 60/60 data_quality=='AVAILABLE'. Reimplemented (not
    imported) to avoid a cross-research-module coupling for a single small
    formula, but the arithmetic is IDENTICAL and locked by a cross-check
    test against ami.research.w7a_state_structure_aging_market_clocks. Does
    NOT depend on reference_price or direction -- purely a function of
    signal_birth_ts and the closed-candle series, hence identical across all
    4 horizon rows of the same signal (see path_schema.py's naming-rationale
    docstring and test_realized_vol_at_anchor_identical_across_horizons_
    same_signal)."""
    end_idx = bisect.bisect_right(index._close_ts, anchor_ts_ms)
    if end_idx < VOL_WINDOW_CANDLES:
        return None
    window = index._rows[end_idx - VOL_WINDOW_CANDLES:end_idx]
    if any(c["data_quality"] != "AVAILABLE" for c in window):
        return None
    closes = [c["close"] for c in window]
    rets = [math.log(closes[i + 1] / closes[i]) for i in range(len(closes) - 1)
            if closes[i] > 0 and closes[i + 1] > 0]
    if len(rets) < MIN_VOL_LOG_RETURNS:
        return None
    return math.sqrt(sum(r * r for r in rets) / len(rets))


def _extremum_with_earliest_tie(points: list[tuple[float, int]], want_max: bool) -> tuple[float, int]:
    """(value, ts) of the max (want_max=True) or min excursion value among
    `points`, breaking ties by the EARLIEST timestamp -- never by iteration/
    insertion order. `points` must already include the virtual (0.0,
    signal_birth_ts) reference-point candidate (see module docstring)."""
    target = max(v for v, _ in points) if want_max else min(v for v, _ in points)
    ts = min(t for v, t in points if v == target)
    return target, ts


def compute_observation(index: _CandleOHLCIndex, signal_id: str, direction: str, signal_birth_ts: int,
                        horizon_name: str, horizon_ms: int, as_of_ts: int,
                        path_definition_version: str = PATH_DEFINITION_VERSION) -> dict:
    """Priority-ordered observation_status checks (path-computability ONLY --
    first failing check wins):
      1. direction not in {LONG, SHORT}      -> NOT_COMPUTABLE_DIRECTION
      2. no fully-closed candle before birth -> MISSING_REFERENCE_PRICE
      3. horizon matures beyond as_of_ts     -> EXCLUDED_NO_HORIZON_DATA
      4. path window has any gap/missing slot -> MISSING_INTERNAL_GAP
      5. otherwise observation_status=OK, endpoint/mfe/mae/timing/
         horizon_outcome_class are computed.

    volatility_status is a SEPARATE, independent axis (see path_schema.py):
      - observation_status != OK  -> volatility_status=NOT_APPLICABLE (nothing to normalize)
      - observation_status == OK  -> realized_vol_at_anchor is checked; None/<=0
        -> volatility_status=INVALID_VOLATILITY_BASELINE (only the 3
        *_anchor_vol_units fields are NULL; everything else from step 5 stays
        populated); otherwise volatility_status=OK and all 3 are populated.
    """
    horizon_end_ts = signal_birth_ts + horizon_ms
    known_at_ts = horizon_end_ts  # conservative: this observation cannot be claimed known before
                                  # the horizon itself has elapsed (see path_schema.py docstring)

    def _row(observation_status: str, volatility_status: str, **fields) -> dict:
        base = {
            "observation_id": _observation_id(signal_id, horizon_name, path_definition_version),
            "signal_id": signal_id,
            "horizon_name": horizon_name,
            "horizon_end_ts": horizon_end_ts,
            "known_at_ts": known_at_ts,
            "as_of_ts": as_of_ts,
            "observation_status": observation_status,
            "volatility_status": volatility_status,
            **_NULL_PATH_FIELDS,
            **_NULL_VOL_FIELDS,
            "path_definition_version": path_definition_version,
            "observation_mode": ObservationMode.HISTORICAL_REPLAY.value,
            "schema_version": PATH_SCHEMA_VERSION,
        }
        base.update(fields)
        return base

    # 1. direction
    if direction not in _VALID_DIRECTIONS:
        return _row("NOT_COMPUTABLE_DIRECTION", "NOT_APPLICABLE")

    # 2. reference price
    ref = index.reference_before(signal_birth_ts)
    if ref is None:
        return _row("MISSING_REFERENCE_PRICE", "NOT_APPLICABLE")
    reference_price, reference_price_ts = ref

    # 3. maturity
    if horizon_end_ts > as_of_ts:
        return _row("EXCLUDED_NO_HORIZON_DATA", "NOT_APPLICABLE",
                    reference_price=reference_price, reference_price_ts=reference_price_ts)

    # 4. path window / gap check
    window = index.path_window(signal_birth_ts, horizon_end_ts)
    effective_path_start_ts = window[0]["open_ts_ms"] if window else None
    # expected_candle_count is anchored on effective_path_start_ts, NOT signal_birth_ts: when
    # birth falls mid-candle (the common case -- real signal_birth_ts values are arbitrary ms,
    # not candle-aligned), the straddling candle's exclusion structurally shortens the
    # observable window by up to just under one full candle. That is a documented boundary
    # effect (see path_schema.py's module docstring), not a data gap -- anchoring the
    # expectation on signal_birth_ts instead would count this structural shortening as a
    # spurious gap_count>=1 on almost every non-aligned signal, which would make
    # MISSING_INTERNAL_GAP fire near-universally instead of only on genuine collector gaps.
    if effective_path_start_ts is not None:
        expected_candle_count = max(0, (horizon_end_ts - effective_path_start_ts) // CANDLE_MS)
    else:
        expected_candle_count = max(0, (horizon_end_ts - signal_birth_ts) // CANDLE_MS)
    observed_candle_count = len(window)
    available = [c for c in window if c["data_quality"] == "AVAILABLE"]
    gap_count = expected_candle_count - len(available)

    common_audit = dict(
        reference_price=reference_price, reference_price_ts=reference_price_ts,
        effective_path_start_ts=effective_path_start_ts,
        expected_candle_count=expected_candle_count,
        observed_candle_count=observed_candle_count,
        gap_count=max(gap_count, 0),
    )

    if gap_count > 0 or expected_candle_count == 0:
        return _row("MISSING_INTERNAL_GAP", "NOT_APPLICABLE", **common_audit)

    versions = sorted({c["candle_definition_version"] for c in available})
    candle_definition_version = versions[0] if len(versions) == 1 else ",".join(versions)

    # 5. path computation (endpoint/mfe/mae/timing/horizon_outcome_class)
    last_close = available[-1]["close"]
    endpoint_return_bps = round((last_close - reference_price) / reference_price * 1e4, 4)
    horizon_outcome_class = classify_path(endpoint_return_bps)

    zero_point = (0.0, signal_birth_ts)  # the reference point itself is part of the path (operator lock #2)
    if direction == "LONG":
        fav_series = [((c["high"] - reference_price) / reference_price * 1e4, c["close_ts_ms"]) for c in available]
        adv_series = [((c["low"] - reference_price) / reference_price * 1e4, c["close_ts_ms"]) for c in available]
    else:  # SHORT
        fav_series = [((reference_price - c["low"]) / reference_price * 1e4, c["close_ts_ms"]) for c in available]
        adv_series = [((reference_price - c["high"]) / reference_price * 1e4, c["close_ts_ms"]) for c in available]
    fav_series.append(zero_point)
    adv_series.append(zero_point)

    mfe_val, mfe_ts = _extremum_with_earliest_tie(fav_series, want_max=True)
    mae_val, mae_ts = _extremum_with_earliest_tie(adv_series, want_max=False)
    # mfe_val/mae_val are already >=0/<=0 by construction (zero_point guarantees it) --
    # round()+max/min kept as harmless defense-in-depth, matching the DB CHECK's own
    # belt-and-suspenders philosophy.
    mfe_bps = round(max(0.0, mfe_val), 4)
    mae_bps = round(min(0.0, mae_val), 4)
    time_to_mfe_ms = mfe_ts - signal_birth_ts
    time_to_mae_ms = mae_ts - signal_birth_ts

    if mfe_ts == mae_ts:
        intrabar_order_status = "SAME_CANDLE_UNKNOWN"
    elif mfe_ts < mae_ts:
        intrabar_order_status = "MFE_FIRST"
    else:
        intrabar_order_status = "MAE_FIRST"

    path_fields = dict(
        **common_audit,
        endpoint_return_bps=endpoint_return_bps,
        mfe_bps=mfe_bps,
        mae_bps=mae_bps,
        time_to_mfe_ms=time_to_mfe_ms,
        time_to_mae_ms=time_to_mae_ms,
        intrabar_order_status=intrabar_order_status,
        horizon_outcome_class=horizon_outcome_class,
        candle_definition_version=candle_definition_version,
    )

    # independent, trailing check: volatility baseline (does NOT affect observation_status)
    realized_vol = _realized_vol_at(index, signal_birth_ts)
    if realized_vol is None or realized_vol <= 0:
        return _row("OK", "INVALID_VOLATILITY_BASELINE", **path_fields, realized_vol_at_anchor=realized_vol)

    vol_bps = realized_vol * 1e4  # per-candle sigma expressed in bps -- same units as *_bps fields
    return _row(
        "OK", "OK", **path_fields,
        realized_vol_at_anchor=realized_vol,
        endpoint_return_anchor_vol_units=round(endpoint_return_bps / vol_bps, 4),
        mfe_anchor_vol_units=round(mfe_bps / vol_bps, 4),
        mae_anchor_vol_units=round(mae_bps / vol_bps, 4),
    )


def fetch_symbol_candles(conn_source, symbol: str, timeframe: str = "1m") -> list[dict]:
    """Direct warehouse-layer read -- this is measurement/materialization
    infrastructure over the canonical lifecycle population, not a Phase 6
    research wave (same justification ami/lifecycle/canonical_backfill.py
    already documents for fetch_source_events(): feature_gateway's mandate
    covers Phase 6 research population assembly, not lifecycle/path
    infrastructure)."""
    cols = ["open_ts_ms", "close_ts_ms", "high", "low", "close", "data_quality", "candle_definition_version"]
    rows = conn_source.execute(
        f"SELECT {', '.join(cols)} FROM ami_candles WHERE symbol=? AND timeframe=?", (symbol, timeframe)
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def fetch_signals(conn_source) -> list[dict]:
    cols = ["signal_id", "symbol", "direction", "signal_birth_ts", "independent_cycle_id", "source_event_id"]
    rows = conn_source.execute(f"SELECT {', '.join(cols)} FROM ami_signal_lifecycle").fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _excluded_row(signal_id: str, horizon_name: str, horizon_ms: int, signal_birth_ts: int,
                   path_definition_version: str) -> dict:
    horizon_end_ts = signal_birth_ts + horizon_ms
    return {
        "observation_id": _observation_id(signal_id, horizon_name, path_definition_version),
        "signal_id": signal_id, "horizon_name": horizon_name,
        "horizon_end_ts": horizon_end_ts, "known_at_ts": horizon_end_ts, "as_of_ts": 0,
        "observation_status": "EXCLUDED_NO_HORIZON_DATA", "volatility_status": "NOT_APPLICABLE",
        **_NULL_PATH_FIELDS, **_NULL_VOL_FIELDS,
        "path_definition_version": path_definition_version,
        "observation_mode": ObservationMode.HISTORICAL_REPLAY.value,
        "schema_version": PATH_SCHEMA_VERSION,
    }


def compute_all(conn, timeframe: str = "1m",
                 path_definition_version: str = PATH_DEFINITION_VERSION) -> dict:
    """Computes ami_lifecycle_path_observations rows for every
    (signal, horizon) pair, grouped by the signal's own symbol. Returns the
    row list plus a summary (raw row counts AND independent_cycle_id-
    deduplicated counts -- counting discipline, see module docstring)."""
    signals = fetch_signals(conn)
    by_symbol: dict[str, list[dict]] = {}
    for s in signals:
        by_symbol.setdefault(s["symbol"], []).append(s)

    rows: list[dict] = []
    as_of_by_symbol: dict[str, int | None] = {}
    for symbol, sigs in by_symbol.items():
        candles = fetch_symbol_candles(conn, symbol, timeframe)
        index = _CandleOHLCIndex(candles)
        as_of_ts = index.maturity_cutoff_ts_ms()
        as_of_by_symbol[symbol] = as_of_ts
        for s in sigs:
            for horizon_name, horizon_ms in PATH_HORIZONS_MS.items():
                if as_of_ts is None:
                    row = _excluded_row(s["signal_id"], horizon_name, horizon_ms, s["signal_birth_ts"],
                                        path_definition_version)
                else:
                    row = compute_observation(
                        index, s["signal_id"], s["direction"], s["signal_birth_ts"],
                        horizon_name, horizon_ms, as_of_ts, path_definition_version,
                    )
                row["_independent_cycle_id"] = s["independent_cycle_id"]
                row["_source_event_id"] = s["source_event_id"]
                rows.append(row)

    status_counts: dict[str, int] = {}
    volatility_status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["observation_status"]] = status_counts.get(r["observation_status"], 0) + 1
        volatility_status_counts[r["volatility_status"]] = volatility_status_counts.get(r["volatility_status"], 0) + 1

    distinct_cycle_n = len({r["_independent_cycle_id"] for r in rows if r["_independent_cycle_id"] is not None})
    distinct_event_n = len({r["_source_event_id"] for r in rows if r["_source_event_id"] is not None})
    path_valid_n = sum(1 for r in rows if r["observation_status"] == "OK")
    volatility_valid_n = sum(1 for r in rows if r["volatility_status"] == "OK")

    return {
        "rows": rows,
        "raw_signal_n": len(signals),
        "raw_observation_row_n": len(rows),
        "raw_observation_row_n_expected_max": len(signals) * len(PATH_HORIZONS_MS),
        "distinct_independent_cycle_n": distinct_cycle_n,
        "distinct_source_event_n": distinct_event_n,
        "status_counts": status_counts,
        "volatility_status_counts": volatility_status_counts,
        "path_valid_n": path_valid_n,
        "volatility_valid_n": volatility_valid_n,
        "as_of_by_symbol": as_of_by_symbol,
    }


def freeze_and_record(conn, provenance: str = "phase-7b0-disposable-path-metrics",
                       path_definition_version: str = PATH_DEFINITION_VERSION) -> dict:
    """Idempotent upsert (NOT append-only -- this is a re-derivable
    materialization of ami_candles+ami_signal_lifecycle, not the immutable
    ami_lifecycle_transitions ledger; re-running with unchanged source data
    upserts identical values, row count and content unchanged)."""
    result = compute_all(conn, path_definition_version=path_definition_version)
    now = int(time.time() * 1000)
    cols = [
        "observation_id", "signal_id", "horizon_name", "horizon_end_ts", "known_at_ts", "as_of_ts",
        "observation_status", "volatility_status", "reference_price", "reference_price_ts",
        "effective_path_start_ts", "endpoint_return_bps", "mfe_bps", "mae_bps", "time_to_mfe_ms",
        "time_to_mae_ms", "intrabar_order_status", "realized_vol_at_anchor",
        "endpoint_return_anchor_vol_units", "mfe_anchor_vol_units", "mae_anchor_vol_units",
        "horizon_outcome_class", "expected_candle_count", "observed_candle_count", "gap_count",
        "candle_definition_version", "path_definition_version", "observation_mode", "provenance",
        "schema_version", "created_ms",
    ]
    update_cols = [c for c in cols if c not in ("observation_id", "signal_id", "horizon_name",
                                                 "path_definition_version", "schema_version")]
    placeholders = ",".join("?" for _ in cols)
    update_set = ", ".join(f"{c}=excluded.{c}" for c in update_cols)
    for r in result["rows"]:
        r_full = dict(r, provenance=provenance, created_ms=now)
        values = [r_full[c] for c in cols]
        conn.execute(
            f"INSERT INTO ami_lifecycle_path_observations ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(signal_id, horizon_name, path_definition_version) DO UPDATE SET {update_set}",
            values,
        )
    conn.commit()
    return result
