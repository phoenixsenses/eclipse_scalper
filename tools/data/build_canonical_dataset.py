"""
Build canonical, time-aligned trading datasets from live artifacts.

Schema/alignment strategy:
- Base grid: 1-second UTC timestamps for each requested symbol over [start, end].
- Microstructure: resampled to 1s (last price/bid/ask, summed volume).
- Events: sparse state fields merged backward and forward-filled on the 1s grid.
- OHLCV: 1m candles merged backward for volatility (ATR20/close). If missing, fallback
  to rolling std of 1s log returns (window=300).

Output:
  data/canonical/
    canonical_microstructure.parquet
    canonical_events.parquet
    canonical_ohlcv.parquet
    canonical_merged.parquet
    manifest.json
    build_log.txt

Run:
  python -m tools.data.build_canonical_dataset --symbols BTCUSDT,ETHUSDT --start 2026-02-16 --end 2026-02-19 --out data/canonical
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_MERGED_COLUMNS = [
    "timestamp",
    "symbol",
    "price",
    "bid",
    "ask",
    "spread",
    "volume",
    "volatility",
    "signal_state",
    "entry_candidate",
    "entry_executed",
    "exit_candidate",
    "position_state",
]


@dataclass
class BuildContext:
    repo_root: Path
    out_dir: Path
    symbols: list[str]
    start_ts: pd.Timestamp
    end_ts: pd.Timestamp
    discovered_input_files: list[str] = field(default_factory=list)
    missing_source_warnings: list[str] = field(default_factory=list)
    searched_patterns: dict[str, list[str]] = field(default_factory=dict)
    excluded_paths: list[str] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
        self.log_lines.append(line)
        print(msg)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical merged dataset from bot artifacts.")
    p.add_argument("--symbols", required=True, help="Comma separated symbols, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--start", required=True, help="UTC start (YYYY-MM-DD or ISO)")
    p.add_argument("--end", required=True, help="UTC end (YYYY-MM-DD or ISO)")
    p.add_argument("--out", required=True, help="Output directory path")
    p.add_argument("--repo-root", default=".", help="Repo root (tests). Defaults to current directory.")
    return p.parse_args(argv)


def _to_utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    return ts


def _as_utc_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        if s.dt.tz is None:
            return s.dt.tz_localize("UTC")
        return s.dt.tz_convert("UTC")
    # numeric unix timestamps (ms or sec)
    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors="coerce")
        # heuristic: values above year 3000 in seconds imply ms.
        unit = "ms" if numeric.dropna().median() > 10_000_000_000 else "s"
        return pd.to_datetime(numeric, utc=True, unit=unit, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def should_exclude_path(path: Path, out_dir: Path | None = None) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    normalized = resolved.as_posix().lower()
    wrapped = f"/{normalized}/"
    if "/data/canonical/" in wrapped:
        return True
    if "/tmp/" in wrapped:
        return True
    if "/.pytest_cache/" in wrapped:
        return True
    if "pytest-of-" in normalized:
        return True
    if "/__pycache__/" in wrapped:
        return True
    if out_dir is not None:
        try:
            if resolved == out_dir.resolve() or resolved.is_relative_to(out_dir.resolve()):
                return True
        except Exception:
            pass
    return False


def infer_timestamp_column(df: pd.DataFrame, source: str, ctx: BuildContext) -> str:
    candidates = ["timestamp", "ts", "ts_ms", "time", "datetime", "event_time", "trade_time_ms"]
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            chosen = lookup[c.lower()]
            ctx.log(f"[infer] {source}: timestamp column -> {chosen}")
            return chosen
    raise ValueError(f"{source}: no timestamp-like column found among {list(df.columns)}")


def infer_symbol_column(df: pd.DataFrame) -> str | None:
    candidates = ["symbol", "s", "k", "sym"]
    lookup = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    return None


def load_jsonl(path: Path, chunk_size: int = 10_000) -> pd.DataFrame:
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            raw = line.strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
            if len(rows) >= chunk_size:
                frames.append(pd.DataFrame(rows))
                rows = []
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _discover_files(base_dirs: Iterable[Path], patterns: Iterable[str], ctx: BuildContext) -> list[Path]:
    found: list[Path] = []
    for base in base_dirs:
        if not base.exists():
            continue
        if should_exclude_path(base, out_dir=ctx.out_dir):
            resolved_base = str(base.resolve())
            if resolved_base not in ctx.excluded_paths:
                ctx.excluded_paths.append(resolved_base)
                ctx.log(f"[discover] excluded base dir: {resolved_base}")
            continue
        for pat in patterns:
            for p in base.rglob(pat):
                if not p.is_file():
                    continue
                if should_exclude_path(p, out_dir=ctx.out_dir):
                    resolved_path = str(p.resolve())
                    if resolved_path not in ctx.excluded_paths:
                        ctx.excluded_paths.append(resolved_path)
                        ctx.log(f"[discover] excluded path: {resolved_path}")
                    continue
                found.append(p)
    # stable ordering
    uniq: dict[str, Path] = {}
    for p in sorted(found, key=lambda x: str(x)):
        uniq[str(p.resolve())] = p
    return list(uniq.values())


def _discover_ohlcv_sources(ctx: BuildContext) -> list[Path]:
    repo_dirs = [ctx.repo_root / "data", ctx.repo_root / "logs", ctx.repo_root / "state", ctx.repo_root]
    patterns = ["*ohlcv*.csv", "*ohlcv*.json", "*ohlcv*.parquet", "*candles*.csv", "*candles*.json", "*candles*.parquet", "*kline*.csv", "*kline*.json", "*kline*.parquet"]
    ctx.searched_patterns["ohlcv"] = [f"{d}:{p}" for d in repo_dirs for p in patterns]
    files = _discover_files(repo_dirs, patterns, ctx)

    cache_json = Path.home() / ".blade_cosmic_cache.json"
    if cache_json.exists() and not should_exclude_path(cache_json, out_dir=ctx.out_dir):
        files.append(cache_json)
    latest_exam = ctx.repo_root / "state" / "latest_exam.json"
    if latest_exam.exists():
        try:
            exam = json.loads(latest_exam.read_text(encoding="utf-8"))
            input_path = exam.get("input_data_path")
            if input_path:
                candidate = (ctx.repo_root / str(input_path)).resolve()
                if candidate.exists() and not should_exclude_path(candidate, out_dir=ctx.out_dir):
                    files.append(candidate)
        except Exception:
            pass
    uniq: dict[str, Path] = {}
    for p in files:
        uniq[str(p.resolve())] = p
    return list(uniq.values())


def _load_microstructure_sqlite(db_path: Path, ctx: BuildContext) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        symbols = tuple(ctx.symbols)
        start_ms = int(ctx.start_ts.timestamp() * 1000)
        end_ms = int(ctx.end_ts.timestamp() * 1000)
        marks = pd.read_sql_query(
            f"SELECT ts_ms, symbol, mark_price AS price, NULL AS bid, NULL AS ask, 0.0 AS volume FROM mark_prices WHERE symbol IN ({','.join(['?']*len(symbols))}) AND ts_ms BETWEEN ? AND ?",
            conn,
            params=[*symbols, start_ms, end_ms],
        )
        trades = pd.read_sql_query(
            f"SELECT ts_ms, symbol, price, NULL AS bid, NULL AS ask, quantity AS volume FROM agg_trades WHERE symbol IN ({','.join(['?']*len(symbols))}) AND ts_ms BETWEEN ? AND ?",
            conn,
            params=[*symbols, start_ms, end_ms],
        )
        df = pd.concat([marks, trades], ignore_index=True)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True, errors="coerce")
        df["symbol"] = df["symbol"].astype(str)
        for c in ("price", "bid", "ask", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[["timestamp", "symbol", "price", "bid", "ask", "volume"]]
    finally:
        conn.close()


def _normalize_generic_micro(df: pd.DataFrame, source: str, ctx: BuildContext) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "price", "bid", "ask", "volume"])
    ts_col = infer_timestamp_column(df, source, ctx)
    sym_col = infer_symbol_column(df)
    if not sym_col:
        raise ValueError(f"{source}: no symbol-like column found")
    out = pd.DataFrame()
    out["timestamp"] = _as_utc_series(df[ts_col])
    out["symbol"] = df[sym_col].astype(str)
    colmap = {c.lower(): c for c in df.columns}
    out["price"] = pd.to_numeric(df[colmap.get("price", colmap.get("mark_price", ts_col))], errors="coerce") if ("price" in colmap or "mark_price" in colmap) else np.nan
    out["bid"] = pd.to_numeric(df[colmap["bid"]], errors="coerce") if "bid" in colmap else np.nan
    out["ask"] = pd.to_numeric(df[colmap["ask"]], errors="coerce") if "ask" in colmap else np.nan
    vol_col = colmap.get("volume", colmap.get("quantity", ""))
    out["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else np.nan
    return out[["timestamp", "symbol", "price", "bid", "ask", "volume"]]


def build_microstructure(ctx: BuildContext) -> pd.DataFrame:
    data_dirs = [ctx.repo_root / "data", ctx.repo_root / "logs"]
    patterns = ["*microstructure*.db", "*microstructure*.jsonl", "*microstructure*.csv", "*microstructure*.parquet"]
    ctx.searched_patterns["microstructure"] = [f"{d}:{p}" for d in data_dirs for p in patterns]
    files = _discover_files(data_dirs, patterns, ctx)
    if not files:
        raise FileNotFoundError(
            "No microstructure source found. Searched patterns:\n" + "\n".join(ctx.searched_patterns["microstructure"])
        )
    ctx.discovered_input_files.extend([str(p) for p in files])
    frames: list[pd.DataFrame] = []
    for p in files:
        try:
            if p.suffix.lower() == ".db":
                frames.append(_load_microstructure_sqlite(p, ctx))
            elif p.suffix.lower() == ".jsonl":
                frames.append(_normalize_generic_micro(load_jsonl(p), str(p), ctx))
            elif p.suffix.lower() == ".csv":
                frames.append(_normalize_generic_micro(load_csv(p), str(p), ctx))
            elif p.suffix.lower() == ".parquet":
                frames.append(_normalize_generic_micro(pd.read_parquet(p), str(p), ctx))
        except Exception as exc:
            ctx.missing_source_warnings.append(f"microstructure parse failed for {p}: {exc}")
    if not frames:
        raise RuntimeError("Microstructure sources discovered but none could be parsed.")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "price", "bid", "ask", "volume"])
    df = df[df["symbol"].isin(ctx.symbols)]
    df = df[(df["timestamp"] >= ctx.start_ts) & (df["timestamp"] <= ctx.end_ts)]
    df["timestamp"] = df["timestamp"].dt.floor("s")
    out = (
        df.groupby(["symbol", "timestamp"], as_index=False)
        .agg({"price": "last", "bid": "last", "ask": "last", "volume": "sum"})
        .sort_values(["symbol", "timestamp"])
    )
    return out


def _series_or_default(df: pd.DataFrame, col: str, default: object = "") -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _infer_symbol_from_entity(entity_value: str, valid_symbols: set[str]) -> str:
    token = str(entity_value or "").split("-", 2)
    if len(token) < 2:
        return ""
    raw = token[1].strip().upper()
    if raw in valid_symbols:
        return raw
    if raw.endswith("USD") and f"{raw}T" in valid_symbols:
        return f"{raw}T"
    return raw


def _flatten_execution_journal(df: pd.DataFrame, ctx: BuildContext) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "signal_state", "entry_candidate", "entry_executed", "exit_candidate", "position_state"])
    if "data" in df.columns:
        data_norm = pd.json_normalize(df["data"])
        for col in data_norm.columns:
            df[f"data.{col}"] = data_norm[col]

    ts_col: str | None = None
    for candidate in ("ts", "timestamp", "time", "datetime"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = infer_timestamp_column(df, "execution_journal", ctx)

    out = pd.DataFrame()
    out["timestamp"] = _as_utc_series(df[ts_col])
    event = _series_or_default(df, "event", "").fillna("").astype(str)
    entity = _series_or_default(df, "data.entity", "").fillna("").astype(str)
    reason = _series_or_default(df, "data.reason", "").fillna("").astype(str)
    state_to = _series_or_default(df, "data.state_to", "").fillna("").astype(str)
    machine = _series_or_default(df, "data.machine", "").fillna("").astype(str)

    symbols_primary = _series_or_default(df, "data.meta.k", np.nan)
    symbols_secondary = _series_or_default(df, "data.k", np.nan)
    symbols_third = _series_or_default(df, "data.symbol", np.nan)
    symbols_plain = _series_or_default(df, "symbol", np.nan)
    inferred_from_entity = entity.map(lambda x: _infer_symbol_from_entity(x, set(ctx.symbols)))
    sym = (
        symbols_primary.where(symbols_primary.notna(), symbols_secondary)
        .where(lambda s: s.notna(), symbols_third)
        .where(lambda s: s.notna(), symbols_plain)
        .fillna(inferred_from_entity)
        .fillna("")
        .astype(str)
        .str.upper()
    )
    out["symbol"] = sym

    entity_up = entity.str.upper()
    state_to_up = state_to.str.upper()
    reason_low = reason.str.lower()
    event_low = event.str.lower()
    is_entry = entity_up.str.startswith("ENTRY-")
    is_exit = entity_up.str.startswith("EXIT-") | entity_up.str.startswith("SL-")

    out["signal_state"] = (
        "event=" + event.fillna("").astype(str)
        + ";machine=" + machine.fillna("").astype(str)
        + ";reason=" + reason.fillna("").astype(str)
    )
    out["entry_candidate"] = (
        is_entry
        & (
            ((state_to_up == "SUBMITTED") & reason_low.str.contains("router_send", na=False))
            | event_low.str.contains("entry_candidate", na=False)
            | reason_low.str.contains("entry_candidate", na=False)
        )
    ).fillna(False).astype(bool)
    out["entry_executed"] = (
        is_entry
        & (
            state_to_up.isin(["ACKED", "OPEN", "FILLED", "OPEN_CONFIRMED"])
            | reason_low.str.contains("exchange_ack", na=False)
            | reason_low.str.contains("exchange_status", na=False)
            | reason_low.str.contains("dryrun", na=False)
        )
    ).fillna(False).astype(bool)
    out["exit_candidate"] = (
        is_exit
        & (
            ((state_to_up == "SUBMITTED") & reason_low.str.contains("router_send", na=False))
            | reason_low.str.contains("exit", na=False)
            | event_low.str.contains("exit", na=False)
            | entity_up.str.startswith("SL-")
        )
    ).fillna(False).astype(bool)
    out["position_state"] = state_to.fillna("").astype(str)

    missing_symbol_rows = out["symbol"].eq("") | out["symbol"].isna()
    dropped = int(missing_symbol_rows.sum())
    if dropped:
        ctx.missing_source_warnings.append(
            f"execution_journal: dropped {dropped} rows with missing symbol after payload inference"
        )
        out = out[~missing_symbol_rows]
    return out


def _from_signal_stability(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "signal_state", "entry_candidate", "entry_executed", "exit_candidate", "position_state"])
    out = pd.DataFrame()
    ts_col = "ts" if "ts" in df.columns else "timestamp"
    out["timestamp"] = _as_utc_series(df[ts_col])
    out["symbol"] = _series_or_default(df, "symbol", "").fillna("").astype(str)
    signal_type = _series_or_default(df, "signal_type", "").fillna("").astype(str)
    reason = _series_or_default(df, "reason", "").fillna("").astype(str)
    out["signal_state"] = ("type=" + signal_type + ";reason=" + reason).astype(str)
    allowed_col = _series_or_default(df, "allowed", False)
    out["entry_candidate"] = allowed_col.fillna(False).astype(bool)
    out["entry_executed"] = False
    out["exit_candidate"] = False
    out["position_state"] = ""
    return out


def _from_event_diary_csv(df: pd.DataFrame, ctx: BuildContext) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "signal_state", "entry_candidate", "entry_executed", "exit_candidate", "position_state"])
    ts_col = "ts_ms" if "ts_ms" in df.columns else infer_timestamp_column(df, "event_diary.csv", ctx)
    out = pd.DataFrame()
    out["timestamp"] = _as_utc_series(df[ts_col])
    out["symbol"] = _series_or_default(df, "symbol", "").fillna("").astype(str)
    etype = _series_or_default(df, "event_type", "").fillna("").astype(str)
    direction = _series_or_default(df, "direction", "").fillna("").astype(str)
    out["signal_state"] = ("event=" + etype + ";dir=" + direction).astype(str)
    out["entry_candidate"] = False
    out["entry_executed"] = False
    out["exit_candidate"] = False
    out["position_state"] = ""
    return out


def build_events(ctx: BuildContext) -> pd.DataFrame:
    data_dirs = [ctx.repo_root / "data", ctx.repo_root / "logs"]
    patterns = ["event_diary*.csv", "event_diary*.jsonl", "execution_journal*.jsonl", "signal_stability*.jsonl"]
    ctx.searched_patterns["events"] = [f"{d}:{p}" for d in data_dirs for p in patterns]
    files = _discover_files(data_dirs, patterns, ctx)
    if not files:
        raise FileNotFoundError(
            "No event sources found. Searched patterns:\n" + "\n".join(ctx.searched_patterns["events"])
        )
    ctx.discovered_input_files.extend([str(p) for p in files])
    frames: list[pd.DataFrame] = []
    for p in files:
        try:
            name = p.name.lower()
            if p.suffix.lower() == ".jsonl":
                raw = load_jsonl(p)
            elif p.suffix.lower() == ".csv":
                raw = load_csv(p)
            else:
                continue
            if "execution_journal" in name:
                frames.append(_flatten_execution_journal(raw, ctx))
            elif "signal_stability" in name:
                frames.append(_from_signal_stability(raw))
            elif "event_diary" in name and p.suffix.lower() == ".csv":
                frames.append(_from_event_diary_csv(raw, ctx))
            else:
                # best effort generic
                ts_col = infer_timestamp_column(raw, str(p), ctx)
                sym_col = infer_symbol_column(raw) or "symbol"
                out = pd.DataFrame()
                out["timestamp"] = _as_utc_series(raw[ts_col])
                out["symbol"] = raw[sym_col].astype(str) if sym_col in raw.columns else ""
                out["signal_state"] = raw.astype(str).agg("|".join, axis=1)
                out["entry_candidate"] = False
                out["entry_executed"] = False
                out["exit_candidate"] = False
                out["position_state"] = ""
                frames.append(out)
        except Exception as exc:
            ctx.missing_source_warnings.append(f"event parse failed for {p}: {exc}")
    if not frames:
        raise RuntimeError("Event sources discovered but none could be parsed.")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "signal_state", "entry_candidate", "entry_executed", "exit_candidate", "position_state"])
    df = df[df["symbol"].isin(ctx.symbols)]
    df = df[(df["timestamp"] >= ctx.start_ts) & (df["timestamp"] <= ctx.end_ts)]
    df["timestamp"] = df["timestamp"].dt.floor("s")
    df["signal_state"] = df["signal_state"].fillna("").astype(str)
    df["position_state"] = df["position_state"].fillna("").astype(str)
    for b in ("entry_candidate", "entry_executed", "exit_candidate"):
        df[b] = df[b].fillna(False).astype(bool)
    return df.sort_values(["symbol", "timestamp"])


def _load_ohlcv_from_cache_json(path: Path, ctx: BuildContext) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for sym, values in (data.get("ohlcv") or {}).items():
        if sym not in ctx.symbols:
            continue
        for row in values or []:
            if not isinstance(row, list) or len(row) < 6:
                continue
            rows.append(
                {
                    "timestamp": pd.to_datetime(row[0], utc=True, unit="ms", errors="coerce"),
                    "symbol": sym,
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                }
            )
    return pd.DataFrame(rows)


def _normalize_ohlcv_table(df: pd.DataFrame, source: str, ctx: BuildContext) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    ts_col = infer_timestamp_column(df, source, ctx)
    sym_col = infer_symbol_column(df)
    if not sym_col:
        raise ValueError(f"{source}: no symbol column found for OHLCV")
    colmap = {c.lower(): c for c in df.columns}
    needed = {
        "open": colmap.get("open", colmap.get("o")),
        "high": colmap.get("high", colmap.get("h")),
        "low": colmap.get("low", colmap.get("l")),
        "close": colmap.get("close", colmap.get("c")),
        "volume": colmap.get("volume", colmap.get("v")),
    }
    if any(v is None for v in needed.values()):
        raise ValueError(f"{source}: missing OHLCV columns. Found {list(df.columns)}")
    out = pd.DataFrame()
    out["timestamp"] = _as_utc_series(df[ts_col]).dt.floor("min")
    out["symbol"] = df[sym_col].astype(str)
    for k, v in needed.items():
        out[k] = pd.to_numeric(df[v], errors="coerce")
    return out


def build_ohlcv(ctx: BuildContext) -> pd.DataFrame:
    files = _discover_ohlcv_sources(ctx)
    if not files:
        ctx.missing_source_warnings.append(
            "OHLCV source not found; searched patterns:\n" + "\n".join(ctx.searched_patterns.get("ohlcv", []))
        )
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    ctx.discovered_input_files.extend([str(p) for p in files])
    frames: list[pd.DataFrame] = []
    for p in files:
        try:
            name = p.name.lower()
            if name == ".blade_cosmic_cache.json":
                frames.append(_load_ohlcv_from_cache_json(p, ctx))
            elif p.suffix.lower() == ".csv":
                frames.append(_normalize_ohlcv_table(load_csv(p), str(p), ctx))
            elif p.suffix.lower() == ".json":
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    frames.append(_normalize_ohlcv_table(pd.DataFrame(raw), str(p), ctx))
            elif p.suffix.lower() == ".parquet":
                frames.append(_normalize_ohlcv_table(pd.read_parquet(p), str(p), ctx))
        except Exception:
            continue
    if not frames:
        ctx.missing_source_warnings.append(
            "OHLCV discovered but parse failed for all candidates. Fallback volatility will use 1s returns."
        )
        return pd.DataFrame(columns=["timestamp", "symbol", "open", "high", "low", "close", "volume"])
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return df
    df = df[df["symbol"].isin(ctx.symbols)]
    df = df[(df["timestamp"] >= ctx.start_ts.floor("min")) & (df["timestamp"] <= ctx.end_ts.ceil("min"))]
    df = df.dropna(subset=["timestamp", "symbol"]).sort_values(["symbol", "timestamp"])
    df = df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    return df


def _compute_atr_volatility(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    if ohlcv_df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "volatility"])
    parts: list[pd.DataFrame] = []
    for sym, g in ohlcv_df.groupby("symbol", sort=False):
        s = g.sort_values("timestamp").copy()
        prev_close = s["close"].shift(1)
        tr = np.maximum(
            s["high"] - s["low"],
            np.maximum((s["high"] - prev_close).abs(), (s["low"] - prev_close).abs()),
        )
        atr20 = tr.rolling(20, min_periods=1).mean()
        s["volatility"] = (atr20 / s["close"]).replace([np.inf, -np.inf], np.nan)
        parts.append(s[["timestamp", "symbol", "volatility"]])
    return pd.concat(parts, ignore_index=True)


def _make_base_grid(ctx: BuildContext) -> pd.DataFrame:
    idx = pd.date_range(ctx.start_ts.floor("s"), ctx.end_ts.ceil("s"), freq="1s", tz="UTC")
    frames = [pd.DataFrame({"timestamp": idx, "symbol": sym}) for sym in ctx.symbols]
    return pd.concat(frames, ignore_index=True)


def _merge_asof_by_symbol(base: pd.DataFrame, sparse: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if sparse.empty:
        for c in cols:
            base[c] = np.nan
        return base
    out_frames: list[pd.DataFrame] = []
    for sym, g in base.groupby("symbol", sort=False):
        left = g.sort_values("timestamp").copy()
        right = sparse[sparse["symbol"] == sym].sort_values("timestamp")
        if right.empty:
            for c in cols:
                left[c] = np.nan
            out_frames.append(left)
            continue
        merged = pd.merge_asof(left, right[["timestamp", *cols]], on="timestamp", direction="backward")
        out_frames.append(merged)
    return pd.concat(out_frames, ignore_index=True)


def _ensure_required_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_MERGED_COLUMNS:
        if c not in out.columns:
            if c in ("entry_candidate", "entry_executed", "exit_candidate"):
                out[c] = False
            elif c in ("signal_state", "position_state"):
                out[c] = ""
            else:
                out[c] = np.nan
    out["timestamp"] = _as_utc_series(out["timestamp"])
    out["symbol"] = out["symbol"].astype(str)
    for c in ("price", "bid", "ask", "spread", "volume", "volatility"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for b in ("entry_candidate", "entry_executed", "exit_candidate"):
        out[b] = out[b].fillna(False).astype(bool)
    out["signal_state"] = out["signal_state"].fillna("").astype(str)
    out["position_state"] = out["position_state"].fillna("").astype(str)
    return out[REQUIRED_MERGED_COLUMNS].sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def _write_parquet(df: pd.DataFrame, path: Path) -> str:
    engines = ["pyarrow", "fastparquet"]
    last_exc: Exception | None = None
    for eng in engines:
        try:
            df.to_parquet(path, index=False, engine=eng)
            return eng
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(f"Unable to write parquet {path}; tried engines={engines}; last_error={last_exc}")


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except Exception:
        return None
    return None


def _frame_meta(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "min_ts": None, "max_ts": None, "dtypes": {}}
    min_ts = df["timestamp"].min() if "timestamp" in df.columns else None
    max_ts = df["timestamp"].max() if "timestamp" in df.columns else None
    return {
        "rows": int(len(df)),
        "min_ts": str(min_ts) if min_ts is not None else None,
        "max_ts": str(max_ts) if max_ts is not None else None,
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
    }


def build_dataset(ctx: BuildContext) -> dict:
    ctx.log("[build] discovering + loading sources")
    ctx.log(f"[discover] excluded output dir: {ctx.out_dir}")
    micro_df = build_microstructure(ctx)
    events_df = build_events(ctx)
    ohlcv_df = build_ohlcv(ctx)

    ctx.log("[build] creating 1s base grid")
    base = _make_base_grid(ctx)

    ctx.log("[build] aligning microstructure")
    merged = base.merge(micro_df, on=["timestamp", "symbol"], how="left")
    merged["volume"] = merged["volume"].fillna(0.0)
    merged["spread"] = merged["ask"] - merged["bid"]

    ctx.log("[build] aligning events")
    event_cols = ["signal_state", "entry_candidate", "entry_executed", "exit_candidate", "position_state"]
    merged = _merge_asof_by_symbol(merged, events_df, event_cols)
    for c in ("signal_state", "position_state"):
        merged[c] = merged[c].fillna("").astype(str)
    for b in ("entry_candidate", "entry_executed", "exit_candidate"):
        merged[b] = merged[b].fillna(False).astype(bool)

    if not ohlcv_df.empty:
        ctx.log("[build] computing ATR20 volatility from OHLCV")
        vol_df = _compute_atr_volatility(ohlcv_df)
        merged = _merge_asof_by_symbol(merged, vol_df, ["volatility"])
    else:
        ctx.log("[build] OHLCV missing; fallback volatility from 1s returns")
        merged["volatility"] = np.nan
        for sym, g in merged.groupby("symbol", sort=False):
            idx = g.index
            px = g["price"].astype(float).ffill()
            ret = np.log(px).diff()
            vol = ret.rolling(300, min_periods=20).std()
            merged.loc[idx, "volatility"] = vol.values

    merged = _ensure_required_schema(merged)
    # keep ohlcv schema normalized
    if not ohlcv_df.empty:
        ohlcv_df = ohlcv_df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    ctx.log("[build] writing parquet outputs")
    ctx.out_dir.mkdir(parents=True, exist_ok=True)
    parquet_engines = {}
    parquet_engines["microstructure"] = _write_parquet(micro_df, ctx.out_dir / "canonical_microstructure.parquet")
    parquet_engines["events"] = _write_parquet(events_df, ctx.out_dir / "canonical_events.parquet")
    parquet_engines["ohlcv"] = _write_parquet(ohlcv_df, ctx.out_dir / "canonical_ohlcv.parquet")
    parquet_engines["merged"] = _write_parquet(merged, ctx.out_dir / "canonical_merged.parquet")

    manifest = {
        "git_commit": _git_commit(ctx.repo_root),
        "build_time_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": ctx.symbols,
        "start": str(ctx.start_ts),
        "end": str(ctx.end_ts),
        "discovered_input_files": sorted(set(ctx.discovered_input_files)),
        "excluded_paths": sorted(set(ctx.excluded_paths)),
        "row_counts": {
            "canonical_microstructure": int(len(micro_df)),
            "canonical_events": int(len(events_df)),
            "canonical_ohlcv": int(len(ohlcv_df)),
            "canonical_merged": int(len(merged)),
        },
        "timestamp_ranges": {
            "canonical_microstructure": {"min": _frame_meta(micro_df)["min_ts"], "max": _frame_meta(micro_df)["max_ts"]},
            "canonical_events": {"min": _frame_meta(events_df)["min_ts"], "max": _frame_meta(events_df)["max_ts"]},
            "canonical_ohlcv": {"min": _frame_meta(ohlcv_df)["min_ts"], "max": _frame_meta(ohlcv_df)["max_ts"]},
            "canonical_merged": {"min": _frame_meta(merged)["min_ts"], "max": _frame_meta(merged)["max_ts"]},
        },
        "column_dtypes": {
            "canonical_microstructure": _frame_meta(micro_df)["dtypes"],
            "canonical_events": _frame_meta(events_df)["dtypes"],
            "canonical_ohlcv": _frame_meta(ohlcv_df)["dtypes"],
            "canonical_merged": _frame_meta(merged)["dtypes"],
        },
        "missing_source_warnings": ctx.missing_source_warnings,
        "searched_patterns": ctx.searched_patterns,
        "parquet_engines": parquet_engines,
    }
    (ctx.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (ctx.out_dir / "build_log.txt").write_text("\n".join(ctx.log_lines), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided.")
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    start_ts = _to_utc_timestamp(args.start)
    end_ts = _to_utc_timestamp(args.end)
    if end_ts <= start_ts:
        raise SystemExit("--end must be greater than --start")
    ctx = BuildContext(repo_root=repo_root, out_dir=out_dir, symbols=symbols, start_ts=start_ts, end_ts=end_ts)
    manifest = build_dataset(ctx)
    print(f"[done] wrote canonical dataset to {out_dir}")
    print(f"[done] merged rows={manifest['row_counts']['canonical_merged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
