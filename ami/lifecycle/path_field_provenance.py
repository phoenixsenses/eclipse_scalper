"""PHASE 7B-0.1 / 7B-CANON: Field-level provenance for
ami_lifecycle_path_observations.

[PHASE 7B-CANON, APPLIED 2026-07-04] `backfill_path_field_provenance()` has
been run against the real data/ami/canonical.sqlite (via
ami.lifecycle.path_canonical_migration.run_canonical_migration()) -- 6210
new `path_observations.*` rows written (270 signals x 23 fields), alongside
the pre-existing 4320 rows from Phase 7A-P1 (total 10530), 0 missing/
duplicate, 0 proxy->safe downgrades. `conn` is still always caller-supplied;
this module never opens DEFAULT_PATH itself.

RECONCILIATION (operator instruction: module docstring or row-level
provenance alone is not sufficient): reuses the EXISTING
ami_lifecycle_field_provenance table (ami/lifecycle/canonical_field_provenance.py,
Phase 7A-P1) verbatim -- no second/parallel provenance table is created. That
table's schema is generic over (signal_id, field_name); it has no
source-table column, so a NEW ami_signal_lifecycle field name could in
principle collide with a NEW ami_lifecycle_path_observations field name. None
of the 23 fields tracked here actually collides with any of the 16 fields
ami.lifecycle.canonical_field_provenance.FIELD_PROVENANCE_SPECS tracks
(source_event_id, independent_cycle_id, signal_birth_ts, terminal_ts, symbol,
setup_id, route_version, evidence_layer, is_proxy, setup_version, direction,
timeframe, first_known_ts, first_executable_ts, last_confirmation_ts,
invalidation_ts) -- but every field_name written here is still prefixed
"path_observations." as an explicit, permanent namespacing convention (belt
and suspenders against any FUTURE name collision, not just today's).

WHICH FIELDS ARE TRACKED: the same exclusion discipline
ami.lifecycle.canonical_field_provenance already applies -- identity/PK
columns (observation_id, signal_id, horizon_name -- a partition/dimension key,
treated like signal_id itself) and pure storage-bookkeeping columns
(path_definition_version, observation_mode, provenance, schema_version,
created_ms) are NOT tracked as "fields needing provenance" (mirrors why
ami_signal_lifecycle's own schema_version/provenance/created_at/updated_ms/
identity_version/code_commit/source_hash are absent from that table's
16-field list). The remaining 23 SUBSTANTIVE columns are tracked below.

CLASSIFICATION RATIONALE (every field gets an explicit, non-guessed reason --
"no unknown/default provenance"):

  DETERMINISTIC_HISTORICAL_SAFE (exact, no approximation/heuristic involved):
    horizon_end_ts, known_at_ts, as_of_ts   -- pure arithmetic on already-safe
                                               timestamps (signal_birth_ts,
                                               frozen horizon_ms, candle-table
                                               maturity snapshot).
    observation_status, volatility_status   -- deterministic, code-derived
                                               rule-based classifications
                                               (priority-ordered checks), not
                                               observations of the market
                                               themselves.
    expected_candle_count, observed_candle_count, gap_count
                                             -- exact counts over an exact
                                               window definition.
    candle_definition_version               -- verbatim copy of the source
                                               candles' own version tag.
    realized_vol_at_anchor                  -- exact log-return stdev over the
                                               60 closed candles at-or-before
                                               signal_birth_ts. Does NOT touch
                                               reference_price or direction at
                                               all -- this is the one
                                               "volatility-derived" field that
                                               is direction-INDEPENDENT and
                                               reference-price-INDEPENDENT, so
                                               it is NOT proxy (contrast with
                                               endpoint_return_anchor_vol_units/
                                               mfe_anchor_vol_units/
                                               mae_anchor_vol_units below,
                                               whose PROXY numerators make
                                               THEM proxy even though this
                                               denominator is not).

  HISTORICAL_PROXY -- direction-INDEPENDENT reason (the excluded straddling
  candle means the true instantaneous price at signal_birth_ts is never
  exactly known, only a last-close-up-to-~60s-stale approximation; this
  reason alone, with no direction involvement at all, already makes these
  fields proxy):
    reference_price, reference_price_ts, effective_path_start_ts
    endpoint_return_bps                      -- derived from reference_price;
                                                 same value for LONG and SHORT
                                                 rows on the same anchor.
    horizon_outcome_class                    -- derived from
                                                 endpoint_return_bps via
                                                 ami.research.w4_post_event_
                                                 path_taxonomy.classify_path
                                                 (reused verbatim, not
                                                 reinvented); inherits the
                                                 same proxy status.
    endpoint_return_anchor_vol_units          -- ratio of a proxy numerator
                                                 (endpoint_return_bps) to a
                                                 deterministic denominator
                                                 (realized_vol_at_anchor);
                                                 direction-independent, but
                                                 still proxy via the numerator.

  HISTORICAL_PROXY -- TWO STACKED reasons (the direction-independent
  reference-price reason above, PLUS ami_signal_lifecycle.direction is
  ITSELF field-level HISTORICAL_PROXY [route-name-prefix heuristic, Phase
  7A-P1] -- mfe/mae orientation depends on that proxy field):
    mfe_bps, mae_bps
    time_to_mfe_ms, time_to_mae_ms            -- derived from which candle
                                                 achieves mfe_bps/mae_bps.
    intrabar_order_status                     -- timestamp comparison of the
                                                 mfe/mae-achieving points.
    mfe_anchor_vol_units, mae_anchor_vol_units -- ratio of a doubly-proxy
                                                 numerator to a deterministic
                                                 denominator; NEVER
                                                 cross-horizon sigma-comparable
                                                 (see path_schema.py's naming
                                                 rationale) -- this
                                                 non-comparability is itself
                                                 part of `limitations` below,
                                                 not a separate axis.

NO SILENT PROXY->SAFE DOWNGRADE (operator lock): backfill_path_field_provenance()
raises PathFieldProvenanceDowngradeViolation if any FIELD_NAME already has an
existing HISTORICAL_PROXY row (any signal, any provenance_version) while the
CURRENT spec for that same field_name says DETERMINISTIC_HISTORICAL_SAFE --
this is a hard, code-enforced guard, not just a documented convention.
"""
from __future__ import annotations
import time

from ami.lifecycle.canonical_field_provenance import _provenance_id

PATH_FIELD_PROVENANCE_SCHEMA_VERSION = 1
PATH_PROVENANCE_VERSION = "path-observations-field-provenance-v1"
FIELD_NAME_PREFIX = "path_observations."


class PathFieldProvenanceDowngradeViolation(Exception):
    """Raised if backfill_path_field_provenance() would silently reclassify a
    field that already has a HISTORICAL_PROXY row (any signal/provenance_version)
    down to DETERMINISTIC_HISTORICAL_SAFE."""


PATH_FIELD_PROVENANCE_SPECS: dict[str, dict] = {
    "horizon_end_ts": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "computed_arithmetic",
        "source_reference": "signal_birth_ts + frozen horizon_ms (ami.research.w4_post_event_path_taxonomy.PATH_HORIZONS_MS)",
        "limitations": None,
    },
    "known_at_ts": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "computed_arithmetic",
        "source_reference": "== horizon_end_ts (conservative: not claimed known before the horizon itself elapses)",
        "limitations": None,
    },
    "as_of_ts": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "computed_arithmetic",
        "source_reference": "MAX(ami_candles.close_ts_ms) snapshot at computation time (maturity cutoff)",
        "limitations": "A frozen snapshot -- re-running later as more candle data accumulates can only "
                       "move EXCLUDED_NO_HORIZON_DATA rows to a computable status, never the reverse.",
    },
    "observation_status": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "rule_based_classification",
        "source_reference": "ami.lifecycle.path_metrics.compute_observation's priority-ordered path-computability checks",
        "limitations": None,
    },
    "volatility_status": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "rule_based_classification",
        "source_reference": "ami.lifecycle.path_metrics.compute_observation's independent vol-baseline check",
        "limitations": "Independent axis from observation_status -- NOT_APPLICABLE whenever "
                       "observation_status != OK (DB CHECK-enforced coupling).",
    },
    "reference_price": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "last_closed_candle_close_before_birth",
        "source_reference": "ami_candles.close of the last candle with close_ts_ms<=signal_birth_ts",
        "limitations": "Proxy for the (unobservable) exact instantaneous price at signal_birth_ts -- "
                       "up to just under 60s stale. Direction-independent reason (applies to both LONG "
                       "and SHORT rows identically).",
    },
    "reference_price_ts": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "last_closed_candle_close_before_birth",
        "source_reference": "close_ts_ms of the same candle as reference_price",
        "limitations": "Timestamp of the proxy observation itself, not of signal_birth_ts.",
    },
    "effective_path_start_ts": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "first_candle_open_at_or_after_birth",
        "source_reference": "ami_candles.open_ts_ms of the first candle with open_ts_ms>=signal_birth_ts",
        "limitations": "The true sub-minute state of the excluded straddling candle "
                       "(open_ts_ms<signal_birth_ts<close_ts_ms) is structurally unrecoverable.",
    },
    "endpoint_return_bps": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "derived_from_reference_price",
        "source_reference": "(last available candle close - reference_price) / reference_price * 1e4",
        "limitations": "Inherits reference_price's proxy status. Direction-INDEPENDENT -- identical "
                       "value for a LONG and a SHORT signal on the same source_event_id.",
    },
    "horizon_outcome_class": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "classify_path_reused_from_w4",
        "source_reference": "ami.research.w4_post_event_path_taxonomy.classify_path(endpoint_return_bps), "
                             "reused verbatim (CONTINUATION/REVERSAL/CHOP, +-20bps band)",
        "limitations": "Inherits endpoint_return_bps's proxy status. Direction-independent.",
    },
    "realized_vol_at_anchor": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "log_return_stdev_60_candle",
        "source_reference": "sqrt(mean(log-return^2)) over the 60 AVAILABLE-quality candles at-or-before "
                             "signal_birth_ts (same formula as W7A's realized_vol_clock)",
        "limitations": "Does not depend on reference_price or direction at all -- identical across all "
                       "4 horizon rows of the same signal (see path_schema.py's naming rationale).",
    },
    "endpoint_return_anchor_vol_units": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "ratio_of_proxy_numerator_to_deterministic_denominator",
        "source_reference": "endpoint_return_bps / (realized_vol_at_anchor * 1e4)",
        "limitations": "Numerator is proxy (endpoint_return_bps); denominator is deterministic but is a "
                       "FIXED anchor-window measure never rescaled by horizon_ms -- NOT sigma-comparable "
                       "across different horizon_name rows of the same signal.",
    },
    "mfe_bps": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "direction_oriented_excursion_extremum",
        "source_reference": "max(0, favorable excursion) over the path window + the t=0 reference point, "
                             "oriented by ami_signal_lifecycle.direction",
        "limitations": "TWO stacked proxy reasons: (1) reference_price's instantaneous-price approximation "
                       "(direction-independent); (2) depends on ami_signal_lifecycle.direction, which is "
                       "itself field-level HISTORICAL_PROXY (route-name-prefix heuristic, Phase 7A-P1).",
    },
    "mae_bps": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "direction_oriented_excursion_extremum",
        "source_reference": "min(0, adverse excursion) over the path window + the t=0 reference point, "
                             "oriented by ami_signal_lifecycle.direction",
        "limitations": "Same two stacked reasons as mfe_bps.",
    },
    "time_to_mfe_ms": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "derived_from_mfe_achieving_point",
        "source_reference": "timestamp of the earliest point (real candle or the virtual t=0 reference "
                             "point) achieving mfe_bps, minus signal_birth_ts",
        "limitations": "Inherits mfe_bps's proxy status. 0 whenever the reference point itself is the "
                       "achieving point (favorable excursion never realized) -- never a fabricated "
                       "first-real-candle timestamp.",
    },
    "time_to_mae_ms": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "derived_from_mae_achieving_point",
        "source_reference": "timestamp of the earliest point achieving mae_bps, minus signal_birth_ts",
        "limitations": "Inherits mae_bps's proxy status. Same t=0 convention as time_to_mfe_ms.",
    },
    "intrabar_order_status": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "timestamp_comparison_of_mfe_mae_achieving_points",
        "source_reference": "compares the mfe_bps- and mae_bps-achieving points' timestamps",
        "limitations": "Inherits mfe_bps/mae_bps proxy status. 1m-candle resolution only -- SAME_CANDLE_"
                       "UNKNOWN whenever both achieving points share one timestamp (real intrabar tick "
                       "order is never claimed).",
    },
    "mfe_anchor_vol_units": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "ratio_of_proxy_numerator_to_deterministic_denominator",
        "source_reference": "mfe_bps / (realized_vol_at_anchor * 1e4)",
        "limitations": "Numerator is proxy (mfe_bps, itself doubly-proxy) AND direction-dependent; "
                       "denominator is a fixed anchor-window measure -- NOT sigma-comparable across "
                       "different horizon_name rows of the same signal.",
    },
    "mae_anchor_vol_units": {
        "field_classification": "HISTORICAL_PROXY",
        "derivation_method": "ratio_of_proxy_numerator_to_deterministic_denominator",
        "source_reference": "mae_bps / (realized_vol_at_anchor * 1e4)",
        "limitations": "Same as mfe_anchor_vol_units.",
    },
    "expected_candle_count": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "computed_arithmetic",
        "source_reference": "(horizon_end_ts - effective_path_start_ts) // 60000 (or signal_birth_ts-anchored "
                             "fallback when the window is entirely empty)",
        "limitations": "Anchored on effective_path_start_ts, not signal_birth_ts, so the excluded "
                       "straddling candle's structural boundary-shortening is never counted as a gap.",
    },
    "observed_candle_count": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "computed_arithmetic",
        "source_reference": "count of ami_candles rows physically present in the path window",
        "limitations": None,
    },
    "gap_count": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "computed_arithmetic",
        "source_reference": "expected_candle_count - count of AVAILABLE-quality candles in the path window",
        "limitations": None,
    },
    "candle_definition_version": {
        "field_classification": "DETERMINISTIC_HISTORICAL_SAFE",
        "derivation_method": "verbatim_copy",
        "source_reference": "ami_candles.candle_definition_version of the candles actually used",
        "limitations": None,
    },
}


def rollback_path_field_provenance(conn_target) -> int:
    """Deletes ONLY the path_observations.*-prefixed rows this module writes
    -- the shared ami_lifecycle_field_provenance table itself, and every
    pre-existing ami_signal_lifecycle-field row (e.g. 'direction',
    'terminal_ts') from Phase 7A-P1, are left completely untouched. Returns
    the number of rows deleted."""
    cur = conn_target.execute(
        "DELETE FROM ami_lifecycle_field_provenance WHERE field_name LIKE ?",
        (f"{FIELD_NAME_PREFIX}%",),
    )
    conn_target.commit()
    return cur.rowcount


def backfill_path_field_provenance(conn_target, signal_ids: list[str],
                                    provenance: str = "phase-7b0-1-path-field-provenance-closure") -> dict:
    """Idempotent: rerunning for the same signal_ids/fields upserts identical
    rows (unchanged row count and content). Every signal_id gets exactly one
    provenance row per field in PATH_FIELD_PROVENANCE_SPECS, field_name
    prefixed "path_observations." (namespacing, see module docstring)."""
    now = int(time.time() * 1000)

    existing_proxy_fields = {
        r[0] for r in conn_target.execute(
            "SELECT DISTINCT field_name FROM ami_lifecycle_field_provenance "
            "WHERE field_classification='HISTORICAL_PROXY' AND field_name LIKE ?",
            (f"{FIELD_NAME_PREFIX}%",),
        ).fetchall()
    }
    for field_name, spec in PATH_FIELD_PROVENANCE_SPECS.items():
        prefixed = f"{FIELD_NAME_PREFIX}{field_name}"
        if prefixed in existing_proxy_fields and spec["field_classification"] != "HISTORICAL_PROXY":
            raise PathFieldProvenanceDowngradeViolation(
                f"{prefixed!r} already has a HISTORICAL_PROXY provenance row, but the current spec "
                f"says {spec['field_classification']!r} -- refusing a silent proxy->safe downgrade"
            )

    n_written = 0
    for signal_id in signal_ids:
        for field_name, spec in PATH_FIELD_PROVENANCE_SPECS.items():
            prefixed = f"{FIELD_NAME_PREFIX}{field_name}"
            pid = _provenance_id(signal_id, prefixed, PATH_PROVENANCE_VERSION)
            is_proxy = 1 if spec["field_classification"] == "HISTORICAL_PROXY" else 0
            conn_target.execute(
                "INSERT INTO ami_lifecycle_field_provenance (provenance_id, signal_id, field_name, "
                "field_classification, is_proxy, derivation_method, source_reference, limitations, "
                "provenance_version, schema_version, code_commit, source_hash, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(signal_id, field_name, provenance_version) DO UPDATE SET "
                "field_classification=excluded.field_classification, is_proxy=excluded.is_proxy, "
                "derivation_method=excluded.derivation_method, source_reference=excluded.source_reference, "
                "limitations=excluded.limitations",
                (pid, signal_id, prefixed, spec["field_classification"], is_proxy,
                 spec["derivation_method"], spec["source_reference"], spec["limitations"],
                 PATH_PROVENANCE_VERSION, PATH_FIELD_PROVENANCE_SCHEMA_VERSION,
                 "ami/lifecycle/path_field_provenance.py", None, now),
            )
            n_written += 1
    conn_target.commit()

    expected = len(signal_ids) * len(PATH_FIELD_PROVENANCE_SPECS)
    existing = conn_target.execute(
        "SELECT COUNT(*) FROM ami_lifecycle_field_provenance WHERE provenance_version=? AND field_name LIKE ?",
        (PATH_PROVENANCE_VERSION, f"{FIELD_NAME_PREFIX}%"),
    ).fetchone()[0]
    dup_check = conn_target.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT signal_id, field_name FROM ami_lifecycle_field_provenance "
        "  WHERE provenance_version=? AND field_name LIKE ? "
        "  GROUP BY signal_id, field_name HAVING COUNT(*) > 1"
        ")",
        (PATH_PROVENANCE_VERSION, f"{FIELD_NAME_PREFIX}%"),
    ).fetchone()[0]
    return {
        "signals_covered": len(signal_ids),
        "fields_per_signal": len(PATH_FIELD_PROVENANCE_SPECS),
        "provenance_rows_written_this_call": n_written,
        "provenance_rows_expected_total": expected,
        "provenance_rows_actual_total": existing,
        "provenance_rows_missing": expected - existing,
        "provenance_rows_duplicate_groups": dup_check,
    }
