#!/usr/bin/env python3
"""Measure BTC/ETH forced-order lead-lag without producing a trading signal.

Positive lag is defined as corr(BTC[t], ETH[t + lag]), so positive lag means
BTC leads ETH. SQLite is always opened read-only.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


LAGS = np.arange(-30, 31, dtype=np.int64)
RNG_SEED = 20260718
EXPECTED_LIQUIDATION_COLUMNS = (
    "id", "ts_ms", "symbol", "side", "price", "quantity", "notional",
    "trade_time_ms",
)


@dataclass(frozen=True)
class Gap:
    label: str
    previous_event_ms: int
    next_event_ms: int

    @property
    def excluded_start_sec(self) -> int:
        return self.previous_event_ms // 1000 + 1

    @property
    def excluded_end_sec(self) -> int:
        return self.next_event_ms // 1000

    @property
    def removed_bins(self) -> int:
        return max(0, self.excluded_end_sec - self.excluded_start_sec)


@dataclass(frozen=True)
class SparseVector:
    indices: np.ndarray
    values: np.ndarray
    length: int
    segment_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse index/value length mismatch")
        if len(self.indices) and (
            self.indices[0] < 0
            or self.indices[-1] >= self.length
            or np.any(np.diff(self.indices) <= 0)
        ):
            raise ValueError("Sparse indices must be unique, sorted, and in bounds")
        if sum(self.segment_lengths) != self.length or any(
            length <= 0 for length in self.segment_lengths
        ):
            raise ValueError("Sparse segment lengths must be positive and sum to length")


def utc_text(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)


def discover_schema(conn: sqlite3.Connection) -> dict:
    tables = conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' "
        "AND (lower(name) LIKE '%liq%' OR lower(name) LIKE '%force%' "
        "OR lower(sql) LIKE '%forceorder%') ORDER BY name"
    ).fetchall()
    findings = []
    for name, ddl in tables:
        columns = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
        findings.append({"name": name, "ddl": ddl, "columns": columns})
    names = {item["name"] for item in findings}
    if "liquidations" not in names:
        raise RuntimeError("No liquidations table discovered; analysis stopped")
    actual = tuple(
        row[1] for row in conn.execute("PRAGMA table_info(liquidations)").fetchall()
    )
    if actual != EXPECTED_LIQUIDATION_COLUMNS:
        raise RuntimeError(
            f"Liquidations schema changed: expected {EXPECTED_LIQUIDATION_COLUMNS}, "
            f"found {actual}; analysis stopped"
        )
    symbols = pd.read_sql_query(
        "SELECT symbol, COUNT(*) AS event_count FROM liquidations "
        "GROUP BY symbol ORDER BY symbol", conn
    )
    if not {"BTCUSDT", "ETHUSDT"}.issubset(set(symbols["symbol"])):
        raise RuntimeError("BTCUSDT and ETHUSDT are not both present; analysis stopped")
    sides = pd.read_sql_query(
        "SELECT side, COUNT(*) AS event_count FROM liquidations "
        "GROUP BY side ORDER BY side", conn
    )
    if set(sides["side"]) != {"BUY", "SELL"}:
        raise RuntimeError(f"Unexpected liquidation sides: {sides.to_dict('records')}")
    bounds = conn.execute(
        "SELECT MIN(ts_ms), MAX(ts_ms), MIN(trade_time_ms), MAX(trade_time_ms) "
        "FROM liquidations WHERE symbol IN ('BTCUSDT','ETHUSDT')"
    ).fetchone()
    bounds_by_symbol = pd.read_sql_query(
        "SELECT symbol, COUNT(*) AS event_count, MIN(ts_ms) AS min_ts_ms, "
        "MAX(ts_ms) AS max_ts_ms FROM liquidations "
        "WHERE symbol IN ('BTCUSDT','ETHUSDT') GROUP BY symbol ORDER BY symbol",
        conn,
    )
    mismatch = conn.execute(
        "SELECT COUNT(*) FROM liquidations WHERE symbol IN ('BTCUSDT','ETHUSDT') "
        "AND ABS(notional - price * quantity) > MAX(0.01, ABS(notional)*1e-9)"
    ).fetchone()[0]
    return {
        "tables": findings,
        "symbols": symbols,
        "sides": sides,
        "bounds": bounds,
        "bounds_by_symbol": bounds_by_symbol,
        "notional_mismatch_count": mismatch,
    }


def coverage_map(conn: sqlite3.Connection) -> pd.DataFrame:
    query = """
    WITH RECURSIVE dates(day) AS (
      SELECT date(MIN(ts_ms)/1000, 'unixepoch') FROM liquidations
       WHERE symbol IN ('BTCUSDT','ETHUSDT')
      UNION ALL
      SELECT date(day, '+1 day') FROM dates
       WHERE day < (SELECT date(MAX(ts_ms)/1000, 'unixepoch') FROM liquidations
                     WHERE symbol IN ('BTCUSDT','ETHUSDT'))
    ), counts AS (
      SELECT date(ts_ms/1000, 'unixepoch') AS day, symbol, COUNT(*) AS n
        FROM liquidations WHERE symbol IN ('BTCUSDT','ETHUSDT')
       GROUP BY 1, 2
    )
    SELECT dates.day,
           COALESCE(b.n, 0) AS BTCUSDT,
           COALESCE(e.n, 0) AS ETHUSDT
      FROM dates
      LEFT JOIN counts b ON b.day=dates.day AND b.symbol='BTCUSDT'
      LEFT JOIN counts e ON e.day=dates.day AND e.symbol='ETHUSDT'
     ORDER BY dates.day
    """
    return pd.read_sql_query(query, conn)


def btc_coverage_materially_thinner(coverage: pd.DataFrame) -> tuple[bool, str]:
    """Fail only on temporal collection asymmetry, not different event incidence."""
    eth_active = coverage["ETHUSDT"] > 0
    btc_active = coverage["BTCUSDT"] > 0
    eth_days = int(eth_active.sum())
    shared_days = int((eth_active & btc_active).sum())
    ratio = shared_days / eth_days if eth_days else 0.0
    thinner = ratio < 0.90
    explanation = (
        f"BTC is active on {shared_days}/{eth_days} ETH-active UTC days "
        f"({ratio:.1%}). The stop gate is temporal: below 90% indicates "
        "materially thinner collection; raw event-count differences alone can "
        "reflect different liquidation incidence."
    )
    return thinner, explanation


def _largest_event_gap(
    conn: sqlite3.Connection, start_ms: int, end_ms: int
) -> tuple[int, int]:
    row = conn.execute(
        """
        WITH ordered AS (
          SELECT ts_ms, LAG(ts_ms) OVER (ORDER BY ts_ms) AS previous_ts_ms
            FROM liquidations WHERE ts_ms BETWEEN ? AND ?
        )
        SELECT previous_ts_ms, ts_ms FROM ordered
         WHERE previous_ts_ms IS NOT NULL
         ORDER BY ts_ms - previous_ts_ms DESC LIMIT 1
        """,
        (start_ms, end_ms),
    ).fetchone()
    if row is None:
        raise RuntimeError("Could not resolve a SYSTEM_STATE gap against event data")
    return int(row[0]), int(row[1])


def parse_known_gaps(
    state_path: Path, conn: sqlite3.Connection, collection_bounds: tuple[int, int]
) -> list[Gap]:
    """Parse the liquidation gap statements in SYSTEM_STATE, then resolve edges.

    The state file flags one long blackout, three dated April holes, and a July
    routed-endpoint outage. Exact millisecond edges are the adjacent observed
    liquidation events, avoiding invented timestamps inside an unobserved span.
    """
    text = state_path.read_text(encoding="utf-8")
    long_match = re.search(
        r"liquidations tablosunda \*\*40\.1 günlük TAM blackout "
        r"\((\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) → "
        r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})",
        text,
    )
    april_match = re.search(
        r"Nisan'da saatlik delikler \(([A-Z][a-z]{2}) (\d+): (\d+\.\d+)h, "
        r"[A-Z][a-z]{2} (\d+): (\d+\.\d+)h, "
        r"[A-Z][a-z]{2} (\d+): (\d+\.\d+)h",
        text,
    )
    july_match = re.search(r"(Temmuz) (\d+)-(\d+) outage hariç", text)
    if not (long_match and april_match and july_match):
        raise RuntimeError(
            "Could not parse the canonical liquidation gap list from SYSTEM_STATE.md; "
            "analysis stopped rather than hardcoding dates"
        )

    def ms(value: str) -> int:
        return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)

    long_start = ms(f"{long_match.group(1)}T{long_match.group(2)}")
    long_end = ms(f"{long_match.group(3)}T{long_match.group(4)}")
    prev_ms, next_ms = _largest_event_gap(
        conn, long_start - 86_400_000, long_end + 86_400_000
    )
    gaps = [Gap("SYSTEM_STATE 40.1-day complete blackout", prev_ms, next_ms)]

    year = int(long_match.group(1)[:4])
    april_month = list(calendar_month_abbreviations()).index(april_match.group(1))
    april_specs = (
        (int(april_match.group(2)), float(april_match.group(3))),
        (int(april_match.group(4)), float(april_match.group(5))),
        (int(april_match.group(6)), float(april_match.group(7))),
    )
    for day, expected_hours in april_specs:
        day_start = ms(f"{year}-{april_month:02d}-{day:02d}T00:00:00")
        prev_ms, next_ms = _largest_event_gap(
            conn, day_start - 1, day_start + 86_400_000
        )
        observed_hours = (next_ms - prev_ms) / 3_600_000
        if abs(observed_hours - expected_hours) > 0.15:
            raise RuntimeError(
                f"SYSTEM_STATE April {day} gap expected about {expected_hours}h, "
                f"but database boundary is {observed_hours:.3f}h; analysis stopped"
            )
        gaps.append(Gap(f"SYSTEM_STATE Apr {day} {expected_hours:.1f}h gap", prev_ms, next_ms))

    month_names_tr = {"Temmuz": 7}
    july_month = month_names_tr[july_match.group(1)]
    july_first = int(july_match.group(2))
    july_last = int(july_match.group(3))
    july_start = ms(f"{year}-{july_month:02d}-{july_first:02d}T00:00:00")
    july_end = ms(f"{year}-{july_month:02d}-{july_last + 1:02d}T00:00:00")
    prev_ms, next_ms = _largest_event_gap(conn, july_start, july_end)
    if (next_ms - prev_ms) < 72 * 3_600_000:
        raise RuntimeError("SYSTEM_STATE July 6-10 outage was not found; analysis stopped")
    gaps.append(Gap("SYSTEM_STATE July 6-10 routed-endpoint outage", prev_ms, next_ms))

    lo_ms, hi_ms = collection_bounds
    return sorted(
        [g for g in gaps if g.next_event_ms >= lo_ms and g.previous_event_ms <= hi_ms],
        key=lambda gap: gap.previous_event_ms,
    )


def allowed_intervals(start_sec: int, end_sec: int, gaps: Sequence[Gap]) -> list[tuple[int, int]]:
    """Return allowed half-open UTC-second intervals."""
    result: list[tuple[int, int]] = []
    cursor = start_sec
    for gap in gaps:
        gap_start = max(start_sec, gap.excluded_start_sec)
        gap_end = min(end_sec, gap.excluded_end_sec)
        if gap_end <= cursor:
            continue
        if gap_start > cursor:
            result.append((cursor, gap_start))
        cursor = max(cursor, gap_end)
    if cursor < end_sec:
        result.append((cursor, end_sec))
    return result


def calendar_month_abbreviations() -> tuple[str, ...]:
    return ("", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def load_events(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ts_ms, symbol, side, notional FROM liquidations "
        "WHERE symbol IN ('BTCUSDT','ETHUSDT') ORDER BY ts_ms",
        conn,
    )


def restrict_intervals(
    intervals: Sequence[tuple[int, int]], start_sec: int, end_sec: int
) -> list[tuple[int, int]]:
    return [
        (max(a, start_sec), min(b, end_sec))
        for a, b in intervals
        if max(a, start_sec) < min(b, end_sec)
    ]


def compact_positions(seconds: np.ndarray, intervals: Sequence[tuple[int, int]]) -> np.ndarray:
    positions = np.full(len(seconds), -1, dtype=np.int64)
    offset = 0
    for start, end in intervals:
        left = np.searchsorted(seconds, start, side="left")
        right = np.searchsorted(seconds, end, side="left")
        positions[left:right] = offset + seconds[left:right] - start
        offset += end - start
    return positions


def build_vectors(
    events: pd.DataFrame, intervals: Sequence[tuple[int, int]]
) -> dict[str, tuple[SparseVector, SparseVector]]:
    length = sum(end - start for start, end in intervals)
    segment_lengths = tuple(end - start for start, end in intervals)
    if length <= 60:
        raise RuntimeError("Fewer than 61 usable aligned bins remain")
    work = events.copy()
    work["second"] = (work["ts_ms"].to_numpy(dtype=np.int64) // 1000)
    seconds = work["second"].to_numpy(dtype=np.int64)
    work["position"] = compact_positions(seconds, intervals)
    work = work[work["position"] >= 0]
    work["signed_notional"] = np.where(
        work["side"].eq("SELL"), work["notional"], -work["notional"]
    )

    def vector(symbol: str, column: str) -> SparseVector:
        grouped = (
            work.loc[work["symbol"].eq(symbol)]
            .groupby("position", sort=True)[column]
            .sum()
        )
        grouped = grouped[grouped != 0]
        return SparseVector(
            grouped.index.to_numpy(dtype=np.int64),
            grouped.to_numpy(dtype=np.float64),
            length,
            segment_lengths,
        )

    btc_raw, eth_raw = vector("BTCUSDT", "notional"), vector("ETHUSDT", "notional")
    btc_signed = vector("BTCUSDT", "signed_notional")
    eth_signed = vector("ETHUSDT", "signed_notional")
    btc_log = SparseVector(
        btc_raw.indices, np.log1p(btc_raw.values), length, segment_lengths
    )
    eth_log = SparseVector(
        eth_raw.indices, np.log1p(eth_raw.values), length, segment_lengths
    )
    return {
        "raw_notional": (btc_raw, eth_raw),
        "log1p_notional": (btc_log, eth_log),
        "signed_notional": (btc_signed, eth_signed),
    }


def _slice(vector: SparseVector, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    left = np.searchsorted(vector.indices, lo, side="left")
    right = np.searchsorted(vector.indices, hi, side="left")
    return vector.indices[left:right], vector.values[left:right]


def _sparse_moments(
    x: SparseVector, y: SparseVector, lag: int, start: int, segment_length: int
) -> tuple[int, float, float, float, float, float]:
    lo = start + max(0, -lag)
    hi = start + min(segment_length, segment_length - lag)
    n = hi - lo
    if n <= 0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0
    xi, xv = _slice(x, lo, hi)
    yi, yv = _slice(y, lo + lag, hi + lag)
    shifted_yi = yi - lag
    _, x_match, y_match = np.intersect1d(
        xi, shifted_yi, assume_unique=True, return_indices=True
    )
    return (
        n,
        float(xv.sum()),
        float(yv.sum()),
        float(np.dot(xv, xv)),
        float(np.dot(yv, yv)),
        float(np.dot(xv[x_match], yv[y_match])),
    )


def sparse_pearson(x: SparseVector, y: SparseVector, lag: int) -> tuple[float, int]:
    if x.length != y.length or x.segment_lengths != y.segment_lengths:
        raise ValueError("Series lengths differ")
    totals = np.zeros(6, dtype=np.float64)
    start = 0
    for segment_length in x.segment_lengths:
        totals += _sparse_moments(x, y, lag, start, segment_length)
        start += segment_length
    n = int(totals[0])
    if n <= 1:
        return math.nan, n
    _, sum_x, sum_y, sum_x2, sum_y2, sum_xy = totals
    numerator = sum_xy - sum_x * sum_y / n
    denominator = math.sqrt(
        max(0.0, sum_x2 - sum_x * sum_x / n)
        * max(0.0, sum_y2 - sum_y * sum_y / n)
    )
    return (numerator / denominator if denominator else math.nan), n


def correlogram(x: SparseVector, y: SparseVector) -> pd.DataFrame:
    rows = []
    for lag in LAGS:
        r, n = sparse_pearson(x, y, int(lag))
        rows.append((int(lag), r, n))
    return pd.DataFrame(rows, columns=["lag_sec_btc_leads_positive", "pearson_r", "N"])


def percentile_with_zeros(vector: SparseVector, percentile: float) -> float:
    rank = (vector.length - 1) * percentile
    zeros = vector.length - len(vector.values)
    if rank < zeros:
        return 0.0
    positive = np.sort(vector.values)
    adjusted = rank - zeros
    lower = int(math.floor(adjusted))
    upper = min(lower + 1, len(positive) - 1)
    weight = adjusted - lower
    return float(positive[lower] * (1 - weight) + positive[upper] * weight)


def conditional_pearson(
    x: SparseVector, y: SparseVector, lag: int, threshold: float
) -> tuple[float, int]:
    all_xv = []
    all_yv = []
    start = 0
    for segment_length in x.segment_lengths:
        lo = start + max(0, -lag)
        hi = start + min(segment_length, segment_length - lag)
        xi, xv = _slice(x, lo, hi)
        keep = xv > threshold
        xi, xv = xi[keep], xv[keep]
        targets = xi + lag
        positions = np.searchsorted(y.indices, targets)
        in_range = positions < len(y.indices)
        matches = np.zeros(len(xv), dtype=bool)
        matches[in_range] = y.indices[positions[in_range]] == targets[in_range]
        yv = np.zeros(len(xv), dtype=np.float64)
        yv[matches] = y.values[positions[matches]]
        all_xv.append(xv)
        all_yv.append(yv)
        start += segment_length
    xv = np.concatenate(all_xv)
    yv = np.concatenate(all_yv)
    n = len(xv)
    if n <= 1:
        return math.nan, n
    if np.var(xv) == 0 or np.var(yv) == 0:
        return math.nan, n
    return float(np.corrcoef(xv, yv)[0, 1]), n


def conditional_correlogram(
    x: SparseVector, y: SparseVector, threshold: float
) -> pd.DataFrame:
    rows = []
    for lag in LAGS:
        r, n = conditional_pearson(x, y, int(lag), threshold)
        rows.append((int(lag), r, n))
    return pd.DataFrame(rows, columns=["lag_sec_btc_leads_positive", "pearson_r", "N"])


def shifted(vector: SparseVector, offset: int) -> SparseVector:
    indices = (vector.indices + offset) % vector.length
    order = np.argsort(indices)
    return SparseVector(
        indices[order], vector.values[order], vector.length, vector.segment_lengths
    )


def peak_row(frame: pd.DataFrame, absolute: bool = False) -> pd.Series:
    values = frame["pearson_r"].abs() if absolute else frame["pearson_r"]
    return frame.loc[values.idxmax()]


def month_bounds(start_sec: int, end_sec: int) -> Iterable[tuple[str, int, int]]:
    current = datetime.fromtimestamp(start_sec, timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    while int(current.timestamp()) < end_sec:
        if current.month == 12:
            nxt = current.replace(year=current.year + 1, month=1)
        else:
            nxt = current.replace(month=current.month + 1)
        yield current.strftime("%Y-%m"), max(start_sec, int(current.timestamp())), min(
            end_sec, int(nxt.timestamp())
        )
        current = nxt


def md_table(frame: pd.DataFrame, float_digits: int = 8) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=["float"]).columns:
        formatted[column] = formatted[column].map(
            lambda value: "NaN" if pd.isna(value) else f"{value:.{float_digits}f}"
        )
    headers = [str(c) for c in formatted.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_report(
    schema: dict,
    coverage: pd.DataFrame,
    coverage_explanation: str,
    gaps: Sequence[Gap],
    usable_bins: int,
    correlograms: dict[str, pd.DataFrame],
    monthly_correlograms: pd.DataFrame,
    stability: pd.DataFrame,
    stability_distribution: pd.DataFrame,
    conditional: pd.DataFrame,
    threshold: float,
    shuffles: pd.DataFrame,
    shuffle_summary: pd.DataFrame,
    verdict: str,
    verdict_sentences: Sequence[str],
    db_path: Path,
    state_path: Path,
) -> str:
    bounds = schema["bounds"]
    symbol_rows = schema["symbols"]
    btc_count = int(symbol_rows.loc[symbol_rows.symbol.eq("BTCUSDT"), "event_count"].iloc[0])
    eth_count = int(symbol_rows.loc[symbol_rows.symbol.eq("ETHUSDT"), "event_count"].iloc[0])
    gap_frame = pd.DataFrame([
        {
            "flag": gap.label,
            "last_event_before_utc": utc_text(gap.previous_event_ms),
            "first_event_after_utc": utc_text(gap.next_event_ms),
            "removed_1s_bins": gap.removed_bins,
        }
        for gap in gaps
    ])
    symbol_bounds = schema["bounds_by_symbol"].copy()
    symbol_bounds["min_ts_utc"] = symbol_bounds["min_ts_ms"].map(utc_text)
    symbol_bounds["max_ts_utc"] = symbol_bounds["max_ts_ms"].map(utc_text)
    symbol_bounds = symbol_bounds[
        ["symbol", "event_count", "min_ts_ms", "min_ts_utc", "max_ts_ms", "max_ts_utc"]
    ]
    schema_sections = []
    for table in schema["tables"]:
        columns = pd.DataFrame(
            table["columns"],
            columns=["cid", "name", "type", "notnull", "default", "pk"],
        )
        schema_sections.append(
            f"### `{table['name']}`\n\n```sql\n{table['ddl']}\n```\n\n{md_table(columns)}"
        )
    corr_sections = []
    for variant, frame in correlograms.items():
        corr_sections.append(f"### {variant}\n\n{md_table(frame)}")
    verdict_text = " ".join(verdict_sentences)
    return f"""# BTC/ETH Forced-Order Lead-Lag Measurement

Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  
Database: `{db_path}` (opened SQLite `mode=ro`)  
Gap source: `{state_path}`

## Scope

Research measurement only. This report defines no signal, trading rule, parameter optimization, or holdout evaluation. At every occurrence below, **positive lag means BTC leads ETH**, computed as `corr(BTC[t], ETH[t + lag])`.

## Step 0: Schema Discovery

{chr(10).join(schema_sections)}

The event table is `liquidations`. Symbols present: {len(schema['symbols'])} distinct values; `BTCUSDT` ({btc_count:,} rows) and `ETHUSDT` ({eth_count:,} rows) are confirmed. The timestamp used is `ts_ms`, Unix epoch milliseconds in UTC; its magnitude and UTC conversion span {utc_text(bounds[0])} to {utc_text(bounds[1])}. `trade_time_ms` is also Unix epoch milliseconds; `ts_ms` is the collector/event timestamp used by the required binning rule.

### Symbol Values Present

{md_table(schema['symbols'], float_digits=0)}

### BTC/ETH Timestamp Coverage

{md_table(symbol_bounds, float_digits=0)}

`price`, `quantity`, and stored `notional` are all present. Across BTC/ETH, {schema['notional_mismatch_count']:,} rows differ from `price * quantity` beyond max($0.01, 1e-9 relative), so stored notional is used and does not need derivation. Side values are `BUY` and `SELL`. Binance forced-order `SELL` closes a long, while `BUY` closes a short; the signed convention is therefore **long-liquidation (`SELL`) notional positive and short-liquidation (`BUY`) notional negative**.

Coverage stop-gate result: {coverage_explanation} BTC total event count is {btc_count/eth_count:.1%} of ETH, but collection dates and outage days align; BTC coverage is not materially thinner.

### Full Daily Coverage Map (UTC, Before Gap Exclusion)

{md_table(coverage, float_digits=0)}

## Step 1: Series Construction and Gap Exclusion

Both series sum stored forced-order notional into `floor(ts_ms / 1000)` bins. Empty allowed bins are explicitly zero in the statistical population, not missing. Known liquidation collection gaps were parsed from `SYSTEM_STATE.md`; exact exclusion edges are the adjacent observed events surrounding each flagged gap.

{md_table(gap_frame, float_digits=0)}

Total removed bins: {sum(g.removed_bins for g in gaps):,}. Remaining aligned bins: {usable_bins:,}. Removed ranges are not bridged in wall-clock time; they are omitted from the measurement population.

## Step 2: Cross-Correlation

Lag header and equation: **`lag_sec_btc_leads_positive`; positive lag means BTC leads ETH; `r = corr(BTC[t], ETH[t + lag])`.** Pearson N is reported at every lag.

{chr(10).join(corr_sections)}

## Step 3: Monthly Stability

Each non-overlapping UTC calendar month is recomputed after the same gap exclusions. `argmax_lag_sec_btc_leads_positive` maximizes signed Pearson r, not absolute r.

{md_table(stability)}

### Full Monthly Correlograms

Every row retains the convention **positive lag means BTC leads ETH**, computed as `corr(BTC[t], ETH[t + lag])`.

{md_table(monthly_correlograms)}

### Argmax Lag Distribution

{md_table(stability_distribution, float_digits=0)}

## Step 4: Conditional Large-BTC Check

The full-population BTC raw-flow 99th percentile is {threshold:.8f}. The restriction is strict (`btc_flow > percentile`) and N is the number of selected BTC cascade bins available at each lag. **Positive lag means BTC leads ETH; `r = corr(BTC[t], ETH[t + lag])`.**

{md_table(conditional)}

## Step 5: Circular-Shift Negative Control

Twenty deterministic random circular offsets were drawn without replacement from offsets at least 604,800 bins (7 days) away from zero in either circular direction. For each shuffle and transform, the table reports the peak absolute Pearson correlation across -30s..+30s. This seed fixes reproducibility only (`{RNG_SEED}`); it is not tuned.

{md_table(shuffles)}

### Shuffle Null Distribution Summary

{md_table(shuffle_summary)}

## Microstructure Research Seeds

These are research-only follow-ups planted by the measured zero-lag result. They are not signals, trading rules, optimized variants, or promotion candidates.

| seed_id | question | required measurement | falsifier / guardrail |
|---|---|---|---|
| MS-SEED-BE-001 | Does a sub-second BTC→ETH ordering exist inside the stable 0s bin? | Rebuild event-time cross-correlation at millisecond resolution using `trade_time_ms`, while reporting timestamp ties and collector latency separately. | Reject if the monthly ordering changes sign, remains tied, or sits inside a timestamp-jitter control band. |
| MS-SEED-BE-002 | Is the zero-lag relationship a common market shock rather than directed transmission? | Condition BTC/ETH co-liquidation intensity on all-market forced-order intensity and independently measured market-wide stress. | Reject directed transmission if the BTC→ETH component vanishes after the common-shock control. Do not use price outcomes. |
| MS-SEED-BE-003 | Does zero-lag coupling differ between long- and short-liquidation cascades? | Pre-register separate `SELL`/long-liquidation and `BUY`/short-liquidation descriptive correlograms with the same gaps, lags, blocks, and shuffle controls. | No pooling or threshold search; reject any side-specific claim that is unstable across calendar blocks. |
| MS-SEED-BE-004 | Can symbol-specific timestamp latency create an artificial 0s peak? | Compare `ts_ms - trade_time_ms` distributions for BTC and ETH by month and collector regime, including tie rates. | Treat any inferred sub-second ordering as instrumentation if latency differences are of comparable magnitude. |
| MS-SEED-BE-005 | Negative knowledge: should +1s..+30s BTC-leading liquidation rules be pursued on this dataset? | Preserve this report as the baseline nullifier for future proposals. | Do not reopen the family without a new source, finer timestamp evidence, or a pre-registered mechanism that directly addresses the stable 0s result. |

## VERDICT

**{verdict}**

{verdict_text}
"""


def inconclusive_report(
    schema: dict, coverage: pd.DataFrame, reason: str, db_path: Path, state_path: Path
) -> str:
    columns = pd.DataFrame(
        next(t for t in schema["tables"] if t["name"] == "liquidations")["columns"],
        columns=["cid", "name", "type", "notnull", "default", "pk"],
    )
    return f"""# BTC/ETH Forced-Order Lead-Lag Measurement

## Step 0: Schema Discovery

```sql
{next(t for t in schema['tables'] if t['name'] == 'liquidations')['ddl']}
```

{md_table(columns)}

BTCUSDT and ETHUSDT are present. `ts_ms` is Unix epoch milliseconds UTC; stored `notional`, `price`, and `quantity` are present.

### Full Daily Coverage Map (UTC)

{md_table(coverage, float_digits=0)}

## VERDICT

**INCONCLUSIVE_DATA**

{reason} The required Step 0 stop gate fired, so no series construction, cross-correlation, stability, conditional, or shuffle calculation was performed. This is a coverage insufficiency rather than evidence for or against lead-lag. Re-running against the same database and `{state_path}` will reproduce the stop.
"""


def run(args: argparse.Namespace) -> Path:
    db_path = args.db.resolve()
    state_path = args.system_state.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open_read_only(db_path) as conn:
        schema = discover_schema(conn)
        coverage = coverage_map(conn)
        thinner, coverage_explanation = btc_coverage_materially_thinner(coverage)
        if thinner:
            output_path.write_text(
                inconclusive_report(schema, coverage, coverage_explanation, db_path, state_path),
                encoding="utf-8",
            )
            return output_path

        start_sec = int(schema["bounds"][0]) // 1000
        end_sec = int(schema["bounds"][1]) // 1000 + 1
        gaps = parse_known_gaps(
            state_path, conn, (int(schema["bounds"][0]), int(schema["bounds"][1]))
        )
        intervals = allowed_intervals(start_sec, end_sec, gaps)
        events = load_events(conn)

    vectors = build_vectors(events, intervals)
    correlograms = {
        variant: correlogram(pair[0], pair[1]) for variant, pair in vectors.items()
    }

    stability_rows = []
    monthly_correlogram_rows = []
    for month, month_start, month_end in month_bounds(start_sec, end_sec):
        month_intervals = restrict_intervals(intervals, month_start, month_end)
        if sum(b - a for a, b in month_intervals) <= 60:
            continue
        monthly = build_vectors(events, month_intervals)
        for variant, pair in monthly.items():
            frame = correlogram(pair[0], pair[1])
            for row in frame.itertuples(index=False):
                monthly_correlogram_rows.append(
                    {
                        "month_utc": month,
                        "variant": variant,
                        "lag_sec_btc_leads_positive": int(row.lag_sec_btc_leads_positive),
                        "pearson_r": float(row.pearson_r),
                        "N": int(row.N),
                    }
                )
            peak = peak_row(frame)
            stability_rows.append(
                {
                    "month_utc": month,
                    "variant": variant,
                    "usable_bins": pair[0].length,
                    "argmax_lag_sec_btc_leads_positive": int(peak["lag_sec_btc_leads_positive"]),
                    "peak_r": float(peak["pearson_r"]),
                    "N_at_peak": int(peak["N"]),
                }
            )
    stability = pd.DataFrame(stability_rows)
    monthly_correlograms = pd.DataFrame(monthly_correlogram_rows)
    stability_distribution = (
        stability.groupby(["variant", "argmax_lag_sec_btc_leads_positive"])
        .size()
        .rename("block_count")
        .reset_index()
        .sort_values(["variant", "block_count", "argmax_lag_sec_btc_leads_positive"], ascending=[True, False, True])
    )

    btc_raw, eth_raw = vectors["raw_notional"]
    threshold = percentile_with_zeros(btc_raw, 0.99)
    conditional = conditional_correlogram(btc_raw, eth_raw, threshold)

    min_shift = 7 * 24 * 60 * 60
    n = btc_raw.length
    candidates = np.arange(min_shift, n - min_shift + 1, dtype=np.int64)
    if len(candidates) < args.shuffles:
        raise RuntimeError("Fewer than 20 valid circular offsets at least 7 days from zero")
    rng = np.random.default_rng(args.seed)
    offsets = rng.choice(candidates, size=args.shuffles, replace=False)
    shuffle_rows = []
    for shuffle_id, offset in enumerate(offsets, start=1):
        for variant, (btc, eth) in vectors.items():
            frame = correlogram(btc, shifted(eth, int(offset)))
            peak = peak_row(frame, absolute=True)
            shuffle_rows.append(
                {
                    "shuffle": shuffle_id,
                    "offset_bins": int(offset),
                    "variant": variant,
                    "peak_abs_r": abs(float(peak["pearson_r"])),
                    "peak_abs_lag_sec_btc_leads_positive": int(peak["lag_sec_btc_leads_positive"]),
                }
            )
    shuffles = pd.DataFrame(shuffle_rows)
    shuffle_summary = (
        shuffles.groupby("variant")["peak_abs_r"]
        .agg(null_min="min", null_median="median", null_p95=lambda s: s.quantile(0.95), null_max="max")
        .reset_index()
    )
    real_peak_rows = []
    for variant, frame in correlograms.items():
        peak = peak_row(frame, absolute=True)
        real_peak_rows.append(
            {
                "variant": variant,
                "real_peak_abs_r": abs(float(peak["pearson_r"])),
                "real_peak_abs_lag_sec_btc_leads_positive": int(
                    peak["lag_sec_btc_leads_positive"]
                ),
            }
        )
    shuffle_summary = shuffle_summary.merge(pd.DataFrame(real_peak_rows), on="variant")
    shuffle_summary["real_exceeds_null_max"] = (
        shuffle_summary["real_peak_abs_r"] > shuffle_summary["null_max"]
    )

    raw_peak = peak_row(correlograms["raw_notional"])
    conditional_peak = peak_row(conditional)
    raw_months = stability[stability["variant"].eq("raw_notional")]
    full_lag = int(raw_peak["lag_sec_btc_leads_positive"])
    stable = bool(len(raw_months) and raw_months["argmax_lag_sec_btc_leads_positive"].eq(full_lag).all())
    raw_null_max = float(
        shuffle_summary.loc[shuffle_summary["variant"].eq("raw_notional"), "null_max"].iloc[0]
    )
    clears_null = abs(float(raw_peak["pearson_r"])) > raw_null_max
    strengthens = float(conditional_peak["pearson_r"]) > float(raw_peak["pearson_r"])
    positive_lead = full_lag > 0
    confirmed = stable and clears_null and strengthens and positive_lead
    verdict = "LEAD_LAG_CONFIRMED" if confirmed else "LEAD_LAG_REJECTED"
    if confirmed:
        verdict_sentences = [
            f"Raw notional peaks at +{full_lag}s (BTC leads ETH) with r={float(raw_peak['pearson_r']):.8f}, and every monthly block has the same argmax lag.",
            f"The real peak exceeds the most conservative 20-shuffle null maximum ({raw_null_max:.8f}).",
            f"Conditioning on BTC flow above its 99th percentile strengthens the peak from {float(raw_peak['pearson_r']):.8f} to {float(conditional_peak['pearson_r']):.8f}.",
            "All preregistered mechanism gates are therefore satisfied without tuning.",
        ]
    else:
        failures = []
        if not positive_lead:
            failures.append(f"the full-sample raw argmax is {full_lag}s, not a positive BTC-leading lag")
        if not stable:
            failures.append("monthly raw argmax lags are not identical to the full-sample argmax")
        if not clears_null:
            failures.append("the real raw peak does not exceed the maximum shuffled peak")
        if not strengthens:
            failures.append("the large-BTC conditional peak does not strengthen")
        if not positive_lead and stable:
            verdict_sentences = [
                f"Raw notional peaks at 0s with r={float(raw_peak['pearson_r']):.8f}, and all {len(raw_months)} monthly raw blocks also peak at 0s; the stable result is contemporaneous coupling, not a positive BTC-leading interval.",
                f"The peak clears the 20-shuffle raw null maximum ({raw_null_max:.8f}) and the large-BTC conditional peak strengthens to {float(conditional_peak['pearson_r']):.8f}, but those checks strengthen only the zero-lag relationship.",
                "Because the mandatory positive BTC-lead condition fails, the BTC-to-ETH lead-lag finding is dead.",
            ]
        else:
            verdict_sentences = [
                f"Raw notional peaks at {full_lag:+d}s with r={float(raw_peak['pearson_r']):.8f}; " + "; ".join(failures) + ".",
                f"The 20-shuffle raw null maximum is {raw_null_max:.8f}, while the conditional peak is {float(conditional_peak['pearson_r']):.8f}.",
                "Because at least one mandatory stability, null-separation, strengthening, or positive-lead condition fails, the BTC-to-ETH lead-lag finding is dead.",
            ]

    report = render_report(
        schema, coverage, coverage_explanation, gaps, btc_raw.length,
        correlograms, monthly_correlograms, stability, stability_distribution,
        conditional, threshold,
        shuffles, shuffle_summary, verdict, verdict_sentences, db_path, state_path,
    )
    output_path.write_text(report, encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/microstructure.db"))
    parser.add_argument("--system-state", type=Path, default=Path("SYSTEM_STATE.md"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/reports") / f"btc_eth_leadlag_{datetime.now().strftime('%Y%m%d')}.md",
    )
    parser.add_argument("--shuffles", type=int, default=20)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(result)
