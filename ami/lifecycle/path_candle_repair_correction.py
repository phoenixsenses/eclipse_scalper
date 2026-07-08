"""BATCH: AMI HISTORICAL CANDLE REPAIR — versioned path correction.

Applies a TARGETED correction to exactly the (signal, horizon) path
observations whose result changed because of the candle-gap repair
(ami/chart/candle_gap_repair_rehearsal.py), WITHOUT touching or deleting the
original "path-v2" rows. This satisfies the project's own documented design
(ami.lifecycle.path_metrics.freeze_and_record()'s docstring explicitly calls
path observations "a re-derivable materialization... NOT append-only",
distinct from the immutable ami_lifecycle_transitions ledger) while still
giving every corrected row explicit, permanent version traceability:

  - candle_data_version: the EXISTING `candle_definition_version` column
    (already present on every ami_lifecycle_path_observations row, populated
    by the unmodified compute_observation() logic -- naturally becomes
    "candle-binance-fapi-repair-v1" or a comma-joined blend with
    "candle-agg_trades-v1" for a window whose candles come from both sources,
    exactly the same multi-version convention that module already uses).
  - path_data_version: a NEW, distinct `path_definition_version` value
    (PATH_DATA_VERSION_CANDLE_REPAIR_R1 = "path-v2-candle-repair-r1") for
    exactly the corrected rows -- the original "path-v2" rows are NEVER
    deleted or overwritten (a different path_definition_version produces a
    different observation_id, since ami.lifecycle.path_metrics._observation_id()
    hashes signal_id+horizon_name+path_definition_version together -- no
    UNIQUE-key collision is structurally possible).

WHY NOT a full parallel duplicate of all 1296 rows under the new version
(the naive reading of "append a new path-data revision"): every population
query in this project's completed research modules (fetch_path_observations
via feature_gateway) does NOT currently filter by path_definition_version by
default. Inserting a full duplicate set would silently double-count every
UNAFFECTED signal for any future, not-yet-run research wave that forgets to
scope the filter. Inserting ONLY the genuinely-changed rows avoids this
hazard entirely -- the 1126 unaffected pairs are correctly represented by
their existing, unchanged "path-v2" row, with nothing new to shadow it.
(This is a documented design tradeoff, not an oversight -- flagged
explicitly for any future batch that revisits this.)

NO_EXPERIMENT_RESULT_OVERWRITE: this module never touches
experiment_registry/experiment_results. Existing completed experiments were
computed against the pre-repair "path-v2" population and remain exactly as
they were -- their own frozen results are the historical record of what the
pre-repair candle data implied, which is correct and intentional (operator
instruction: "Record that they used the pre-repair candle/path version").
"""
from __future__ import annotations
import time

from ami.lifecycle.canonical_schema import ObservationMode
from ami.lifecycle.path_metrics import _CandleOHLCIndex, compute_observation, fetch_signals, fetch_symbol_candles
from ami.research.w4_post_event_path_taxonomy import PATH_HORIZONS_MS

PATH_DATA_VERSION_CANDLE_REPAIR_R1 = "path-v2-candle-repair-r1"
OLD_PATH_DEFINITION_VERSION = "path-v2"

_BPS_FIELDS = ("observation_status", "mfe_bps", "mae_bps", "time_to_mfe_ms", "time_to_mae_ms", "endpoint_return_bps")
_VOL_FIELDS = ("volatility_status", "realized_vol_at_anchor", "endpoint_return_anchor_vol_units",
               "mfe_anchor_vol_units", "mae_anchor_vol_units")

_COLUMNS = [
    "observation_id", "signal_id", "horizon_name", "horizon_end_ts", "known_at_ts", "as_of_ts",
    "observation_status", "volatility_status", "reference_price", "reference_price_ts",
    "effective_path_start_ts", "endpoint_return_bps", "mfe_bps", "mae_bps", "time_to_mfe_ms",
    "time_to_mae_ms", "intrabar_order_status", "realized_vol_at_anchor",
    "endpoint_return_anchor_vol_units", "mfe_anchor_vol_units", "mae_anchor_vol_units",
    "horizon_outcome_class", "expected_candle_count", "observed_candle_count", "gap_count",
    "candle_definition_version", "path_definition_version", "observation_mode", "provenance",
    "schema_version", "created_ms",
]


def identify_affected_pairs(conn, symbol: str = "ETHUSDT",
                             old_path_definition_version: str = OLD_PATH_DEFINITION_VERSION) -> dict:
    """Recomputes every (signal, horizon) pair against the connection's
    CURRENT candle data and compares it to the stored old_path_definition_version
    row. Returns three disjoint classes:
      - class_a: raw bps/status changed (a forward path-window gap resolved)
      - class_b: only volatility-normalized fields changed (the 60-candle
        backward-looking realized-vol window was affected, observation_status
        and raw bps are unchanged)
      - unaffected: byte-identical across every compared field
    """
    old_rows = {}
    for row in conn.execute(
        f"SELECT signal_id, horizon_name, {', '.join(_BPS_FIELDS[1:] + _VOL_FIELDS[1:])} "
        f"FROM ami_lifecycle_path_observations WHERE path_definition_version=?",
        (old_path_definition_version,),
    ).fetchall():
        old_rows[(row[0], row[1])] = row

    signals = {s["signal_id"]: s for s in fetch_signals(conn)}
    candles = fetch_symbol_candles(conn, symbol, "1m")
    index = _CandleOHLCIndex(candles)
    as_of = index.maturity_cutoff_ts_ms()

    field_order = _BPS_FIELDS[1:] + _VOL_FIELDS[1:]
    class_a, class_b, unaffected = [], [], []
    for sid, s in signals.items():
        for horizon_name, horizon_ms in PATH_HORIZONS_MS.items():
            key = (sid, horizon_name)
            old = old_rows.get(key)
            if old is None:
                continue
            new_obs = compute_observation(index, sid, s["direction"], s["signal_birth_ts"], horizon_name,
                                           horizon_ms, as_of, old_path_definition_version)
            old_by_field = dict(zip(field_order, old[2:]))
            bps_changed = (
                old_by_field.get("mfe_bps") != new_obs.get("mfe_bps")
                or old_by_field.get("mae_bps") != new_obs.get("mae_bps")
                or old_by_field.get("time_to_mfe_ms") != new_obs.get("time_to_mfe_ms")
                or old_by_field.get("time_to_mae_ms") != new_obs.get("time_to_mae_ms")
                or old_by_field.get("endpoint_return_bps") != new_obs.get("endpoint_return_bps")
            )
            # observation_status is stored on the row itself, fetch separately for clarity
            old_status_row = conn.execute(
                "SELECT observation_status FROM ami_lifecycle_path_observations "
                "WHERE signal_id=? AND horizon_name=? AND path_definition_version=?",
                (sid, horizon_name, old_path_definition_version),
            ).fetchone()
            status_changed = old_status_row[0] != new_obs.get("observation_status")
            vol_changed = (
                old_by_field.get("realized_vol_at_anchor") != new_obs.get("realized_vol_at_anchor")
                or old_by_field.get("endpoint_return_anchor_vol_units") != new_obs.get("endpoint_return_anchor_vol_units")
                or old_by_field.get("mfe_anchor_vol_units") != new_obs.get("mfe_anchor_vol_units")
                or old_by_field.get("mae_anchor_vol_units") != new_obs.get("mae_anchor_vol_units")
            )
            if bps_changed or status_changed:
                class_a.append(key)
            elif vol_changed:
                class_b.append(key)
            else:
                unaffected.append(key)

    return {"class_a": class_a, "class_b": class_b, "unaffected": unaffected}


def apply_targeted_path_correction(conn, affected_pairs: list[tuple[str, str]], symbol: str = "ETHUSDT",
                                    path_data_version: str = PATH_DATA_VERSION_CANDLE_REPAIR_R1,
                                    provenance: str = "batch-candle-gap-remediation-path-correction-r1") -> dict:
    """The ONLY write path for corrected path rows. INSERT ... ON CONFLICT
    (observation_id) DO UPDATE -- idempotent; since path_data_version is
    unique to this correction, observation_id can never collide with an
    original "path-v2" row (see module docstring)."""
    signals = {s["signal_id"]: s for s in fetch_signals(conn)}
    candles = fetch_symbol_candles(conn, symbol, "1m")
    index = _CandleOHLCIndex(candles)
    as_of = index.maturity_cutoff_ts_ms()

    now = int(time.time() * 1000)
    n_written = 0
    for sid, horizon_name in affected_pairs:
        s = signals[sid]
        horizon_ms = PATH_HORIZONS_MS[horizon_name]
        obs = compute_observation(index, sid, s["direction"], s["signal_birth_ts"], horizon_name, horizon_ms,
                                   as_of, path_data_version)
        values = [obs.get(c) for c in _COLUMNS[:-4]]
        values += [ObservationMode.HISTORICAL_REPLAY.value, provenance, 10, now]
        placeholders = ",".join("?" for _ in _COLUMNS)
        conn.execute(
            f"INSERT INTO ami_lifecycle_path_observations ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            "ON CONFLICT(observation_id) DO UPDATE SET "
            "observation_status=excluded.observation_status, volatility_status=excluded.volatility_status, "
            "reference_price=excluded.reference_price, reference_price_ts=excluded.reference_price_ts, "
            "effective_path_start_ts=excluded.effective_path_start_ts, endpoint_return_bps=excluded.endpoint_return_bps, "
            "mfe_bps=excluded.mfe_bps, mae_bps=excluded.mae_bps, time_to_mfe_ms=excluded.time_to_mfe_ms, "
            "time_to_mae_ms=excluded.time_to_mae_ms, intrabar_order_status=excluded.intrabar_order_status, "
            "realized_vol_at_anchor=excluded.realized_vol_at_anchor, "
            "endpoint_return_anchor_vol_units=excluded.endpoint_return_anchor_vol_units, "
            "mfe_anchor_vol_units=excluded.mfe_anchor_vol_units, mae_anchor_vol_units=excluded.mae_anchor_vol_units, "
            "horizon_outcome_class=excluded.horizon_outcome_class, expected_candle_count=excluded.expected_candle_count, "
            "observed_candle_count=excluded.observed_candle_count, gap_count=excluded.gap_count, "
            "candle_definition_version=excluded.candle_definition_version",
            values,
        )
        n_written += 1
    conn.commit()
    return {"rows_written": n_written}


def rollback_path_correction(conn, path_data_version: str = PATH_DATA_VERSION_CANDLE_REPAIR_R1) -> dict:
    """Deletes ONLY rows carrying path_data_version -- every original
    "path-v2" row is untouched (different path_definition_version, hence a
    different observation_id, no shared key at all)."""
    cur = conn.execute(
        "DELETE FROM ami_lifecycle_path_observations WHERE path_definition_version=?", (path_data_version,)
    )
    conn.commit()
    return {"rows_deleted": cur.rowcount}


def effective_observation(conn, signal_id: str, horizon_name: str,
                           corrected_version: str = PATH_DATA_VERSION_CANDLE_REPAIR_R1,
                           old_version: str = OLD_PATH_DEFINITION_VERSION) -> dict | None:
    """Returns the corrected row if one exists for this (signal, horizon),
    otherwise the original "path-v2" row -- the read-side counterpart of the
    write-side design: callers who want the CURRENT-BEST-KNOWN observation
    (not specifically the historical pre-repair or post-repair record) use
    this instead of assuming path_definition_version='path-v2' unconditionally."""
    row = conn.execute(
        "SELECT * FROM ami_lifecycle_path_observations WHERE signal_id=? AND horizon_name=? "
        "AND path_definition_version=?",
        (signal_id, horizon_name, corrected_version),
    ).fetchone()
    if row is None:
        row = conn.execute(
            "SELECT * FROM ami_lifecycle_path_observations WHERE signal_id=? AND horizon_name=? "
            "AND path_definition_version=?",
            (signal_id, horizon_name, old_version),
        ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM ami_lifecycle_path_observations LIMIT 1"
    ).description]
    return dict(zip(cols, row))


def fetch_effective_path_observations(conn, research_context_id: str, equals: dict | None = None,
                                       corrected_version: str = PATH_DATA_VERSION_CANDLE_REPAIR_R1,
                                       old_version: str = OLD_PATH_DEFINITION_VERSION) -> list[dict]:
    """The bulk, feature_gateway-only counterpart of effective_observation():
    returns exactly ONE row per (signal_id, horizon_name) -- the corrected
    ("path-v2-candle-repair-r1") row where one exists, otherwise the original
    ("path-v2") row. Any `equals` filter EXCEPT path_definition_version is
    applied AFTER the effective row is selected, never before -- filtering
    beforehand would be wrong for exactly the case this function exists to
    fix: a corrected row's observation_status can differ from its own
    original row's, and for the 21 volatility-only-corrected pairs BOTH the
    original and corrected row already carry observation_status='OK', so an
    unscoped equals={"observation_status": "OK"} filter passed straight to
    fetch_path_observations() would return BOTH rows for those 21 pairs
    (a real, confirmed double-count hazard -- this is precisely the reason
    this function exists rather than callers filtering ad hoc).

    Only feature_gateway.fetch_path_observations() ever touches the table
    directly -- this function's own reduction step is pure Python over
    feature_gateway's already-audited result set, never a second SQL access
    path."""
    from ami.research.feature_gateway import fetch_path_observations

    equals = dict(equals or {})
    equals.pop("path_definition_version", None)  # never filter by version before the reduction
    # effective=True: this function's own reduction step (below) is exactly the "explicit effective
    # selector" ami.research.feature_gateway.fetch_path_observations' fail-closed multi-version guard
    # (GOAL A) requires -- it deliberately reads across all physical versions first, then narrows to
    # one row per (signal_id, horizon_name).
    all_rows = fetch_path_observations(conn, research_context_id, equals=equals, effective=True)

    effective: dict[tuple[str, str], dict] = {}
    for r in all_rows:
        key = (r["signal_id"], r["horizon_name"])
        if key not in effective or r["path_definition_version"] == corrected_version:
            effective[key] = r

    post_filter = {k: v for k, v in (equals or {}).items()}
    rows = list(effective.values())
    if post_filter:
        rows = [r for r in rows if all(r.get(k) == v for k, v in post_filter.items())]
    return rows


def effective_path_selection_audit(conn, research_context_id: str,
                                    corrected_version: str = PATH_DATA_VERSION_CANDLE_REPAIR_R1,
                                    old_version: str = OLD_PATH_DEFINITION_VERSION) -> dict:
    """Read-only integrity proof for the effective selector (Part 0 of the
    corrected-data-rerun contract): physical row counts by version, effective
    row count, and an explicit duplicate-pair check -- never assumed, always
    computed from the actual table contents."""
    from ami.research.feature_gateway import fetch_path_observations

    # effective=True: this audit deliberately reads ALL physical versions to report per-version
    # counts -- it is the intentional multi-version read GOAL A's guard expects a caller to
    # explicitly acknowledge, never an implicit/accidental one.
    all_rows = fetch_path_observations(conn, research_context_id, equals={}, effective=True)
    physical_by_version: dict[str, int] = {}
    physical_pair_counts: dict[tuple[str, str], int] = {}
    for r in all_rows:
        physical_by_version[r["path_definition_version"]] = physical_by_version.get(r["path_definition_version"], 0) + 1
        key = (r["signal_id"], r["horizon_name"])
        physical_pair_counts[key] = physical_pair_counts.get(key, 0) + 1
    duplicate_physical_pairs = sum(1 for n in physical_pair_counts.values() if n > 1)

    effective_rows = fetch_effective_path_observations(conn, research_context_id)
    effective_pair_counts: dict[tuple[str, str], int] = {}
    for r in effective_rows:
        key = (r["signal_id"], r["horizon_name"])
        effective_pair_counts[key] = effective_pair_counts.get(key, 0) + 1
    duplicate_effective_pairs = sum(1 for n in effective_pair_counts.values() if n > 1)

    corrected_supersede_n = sum(
        1 for r in effective_rows
        if (r["signal_id"], r["horizon_name"]) in {
            (rr["signal_id"], rr["horizon_name"]) for rr in all_rows if rr["path_definition_version"] == corrected_version
        }
        and r["path_definition_version"] == corrected_version
    )

    return {
        "physical_row_count_by_version": physical_by_version,
        "physical_row_count_total": len(all_rows),
        "duplicate_physical_pair_n": duplicate_physical_pairs,
        "effective_row_count": len(effective_rows),
        "duplicate_effective_pair_n": duplicate_effective_pairs,
        "corrected_rows_supersede_n": corrected_supersede_n,
    }
