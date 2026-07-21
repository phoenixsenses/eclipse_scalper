"""S34 V Engine v0.2 shadow mirror ledger.

Mirrors the live V Engine v0.2 rule in observation-only mode. This script never
places orders; it rebuilds the same knowable signal set from historical/live DB
data and labels the live-like maker lifecycle for dashboard monitoring.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.research_s34_knowable_anchor_continuation import (
    AnchorSnapshot,
    MarkIndex,
    file_fingerprint,
    iso_ms,
    load_liquidations,
    load_mark_index,
    load_mark_index_range,
    r1,
    r3,
    reconstruct_anchors,
    sha256_text,
)
from tools.research_s34_maker_fade import FadeEvent, anchor_vdepth_bps, summarize
from tools.research_s34_wave_absorption import book_features_at
from tools.s34_v_engine_cancel_replace import simulate_cancel_replace
from tools.s34_v_engine_execution_frontier import anchor_mark_counterfactual, collect_v01_events, prior_return_bps
from tools.s34_v_engine_shadow_observer import (
    ACCEL_WINDOW_SEC,
    BUCKET_SEC,
    FADE_DIRECTION,
    HORIZON_SEC,
    LIQ_SIDE,
    MIN_GAP_SEC,
    PRIOR4H_LT_BPS,
    SYMBOL,
    THRESHOLD_USD,
    VDEPTH_MAX_BPS,
    VDEPTH_MIN_BPS,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
PID_PATH = ROOT / "logs" / "pids" / "s34_v_engine_v02_shadow_mirror.pid"
STATE_PATH = ROOT / "runtime" / "s34_v_engine_v02_shadow_mirror_state.json"
DEFAULT_LEDGER_JSONL = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.jsonl"
DEFAULT_LEDGER_CSV = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_LEDGER.csv"
DEFAULT_BRIEF_JSON = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_BRIEF.json"
DEFAULT_BRIEF_MD = OUT_DIR / "S34_V_ENGINE_V0_2_SHADOW_MIRROR_BRIEF.md"

PROTOCOL_ID = "S34_V_ENGINE_V0_2_ETH_SELL_MAKER_LONG_H2_O20_W300_O5_DEEPBID"
PROTOCOL_STATUS = "EXPLORATORY_FROZEN"
PERMISSION = "EXPLORATORY_V_FADE_V0_2_SHADOW_MIRROR"
DECISION = "OBSERVE_ONLY_NO_ORDER"
LIFECYCLE_ID = "O20_W300_O5_C1"

MIN_BID_DEPTH_USD = 135_423.8
INITIAL_OFFSET_BPS = 20.0
REPLACE_OFFSET_BPS = 5.0
WAIT_SEC = 300
CROSS_MARGIN_BPS = 1.0

# --- Runtime hardening (S34-VENGINE-V02-SHADOW-MIRROR-RUNTIME-HARDENING-V1) ---
# Bounded-memory, incremental, checkpoint-resumable steady-state tick path.
# Does not alter signal/threshold/entry-exit semantics (frozen protocol
# constants above are untouched); it only bounds which historical rows a
# steady-state tick re-reads from mark_prices/liquidations.
CHECKPOINT_PATH = ROOT / "runtime" / "s34_v_engine_v02_shadow_mirror_checkpoint.json"
LOCK_PATH = ROOT / "runtime" / "s34_v_engine_v02_shadow_mirror.lock"
CHECKPOINT_SCHEMA_VERSION = 1

CLOSE_GRACE_SEC = 3600  # a liquidation bucket only becomes "closed"/final once this far behind the newest observed liquidation row (data-time, not wall-clock)
BOOTSTRAP_CHUNK_SEC = 21600  # max span of newly-closed history advanced in a single tick; bounds cold-bootstrap memory/CPU/rows-read
OPEN_WINDOW_MARGIN_SEC = 2 * MIN_GAP_SEC  # small look-ahead past the close boundary so a still-forming bucket/anchor is visible before it closes
MARK_LOOKBACK_MARGIN_SEC = 4 * 3600 + 900  # prior_4h filter needs 4h of mark history behind an anchor; +900s safety margin


class CheckpointCorruptError(RuntimeError):
    """Checkpoint file is unreadable, malformed, or inconsistent with the running protocol config."""


class DuplicateInstanceError(RuntimeError):
    """Another live instance already holds the mirror's lock file."""


def _params_fingerprint() -> str:
    raw = "|".join(
        str(x)
        for x in (
            PROTOCOL_ID,
            SYMBOL,
            LIQ_SIDE,
            THRESHOLD_USD,
            VDEPTH_MIN_BPS,
            VDEPTH_MAX_BPS,
            PRIOR4H_LT_BPS,
            MIN_BID_DEPTH_USD,
            INITIAL_OFFSET_BPS,
            REPLACE_OFFSET_BPS,
            WAIT_SEC,
            CROSS_MARGIN_BPS,
            HORIZON_SEC,
            BUCKET_SEC,
            MIN_GAP_SEC,
            ACCEL_WINDOW_SEC,
        )
    )
    return sha256_text(raw)[:32]


def floor_to_bucket(ts_ms: int, bucket_ms: int) -> int:
    return (int(ts_ms) // int(bucket_ms)) * int(bucket_ms)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptError(
            f"checkpoint at {path} is unreadable/corrupt ({type(exc).__name__}: {exc}); "
            "remediation: move the file aside (do not delete blindly) and restart to force a "
            "fresh bootstrap from the existing ledger, or restore a known-good checkpoint backup."
        ) from exc
    required = {"schema_version", "protocol_id", "params_fingerprint", "closed_before_ts_ms", "last_kept_ts_ms"}
    missing = required - set(data.keys())
    if missing:
        raise CheckpointCorruptError(f"checkpoint at {path} missing required fields: {sorted(missing)}")
    if int(data["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCorruptError(
            f"checkpoint schema_version={data['schema_version']} != expected {CHECKPOINT_SCHEMA_VERSION}; "
            "remediation: this build cannot safely resume an older/newer checkpoint format."
        )
    if str(data["protocol_id"]) != PROTOCOL_ID:
        raise CheckpointCorruptError(
            f"checkpoint protocol_id={data['protocol_id']!r} != running protocol {PROTOCOL_ID!r}"
        )
    if str(data["params_fingerprint"]) != _params_fingerprint():
        raise CheckpointCorruptError(
            "checkpoint params_fingerprint mismatch: frozen protocol constants changed since this "
            "checkpoint was written; remediation: this is a fail-closed guard against silently "
            "reusing stale-config checkpoint state -- move the checkpoint aside to force re-bootstrap."
        )
    return data


def save_checkpoint_atomic(path: Path, data: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=True))


def liq_ts_bounds(conn: sqlite3.Connection, symbol: str, side: str) -> tuple[int | None, int | None]:
    row = conn.execute(
        "SELECT MIN(ts_ms), MAX(ts_ms) FROM liquidations WHERE symbol=? AND side=?",
        (symbol, side),
    ).fetchone()
    if not row or row[0] is None:
        return None, None
    return int(row[0]), int(row[1])


def bootstrap_checkpoint(conn: sqlite3.Connection, existing_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_ms = BUCKET_SEC * 1000
    rewind_ms = max(CLOSE_GRACE_SEC, MIN_GAP_SEC) * 1000
    if existing_rows:
        max_signal_ts = max(int(r["signal_ts_ms"]) for r in existing_rows)
        closed_before = floor_to_bucket(max_signal_ts - rewind_ms, bucket_ms)
        seed_ts = max(
            (int(r["signal_ts_ms"]) for r in existing_rows if int(r["signal_ts_ms"]) <= closed_before),
            default=-(10**18),
        )
    else:
        earliest, _ = liq_ts_bounds(conn, SYMBOL, LIQ_SIDE)
        closed_before = floor_to_bucket(int(earliest), bucket_ms) if earliest is not None else 0
        seed_ts = -(10**18)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "params_fingerprint": _params_fingerprint(),
        "closed_before_ts_ms": int(closed_before),
        "last_kept_ts_ms": int(seed_ts),
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "created_by_pid": os.getpid(),
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, int(pid))
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                existing_pid = int(path.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                existing_pid = -1
            if _pid_alive(existing_pid):
                raise DuplicateInstanceError(
                    f"another s34_v_engine_v02_shadow_mirror instance is already running "
                    f"(pid={existing_pid}); lock={path}"
                )
            try:
                path.unlink()
            except OSError:
                pass
            continue
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            return
    raise DuplicateInstanceError(f"could not acquire lock at {path} (contended by a live process)")


def release_lock(path: Path) -> None:
    try:
        if path.exists() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
            path.unlink()
    except OSError:
        pass


def _own_priority_class() -> str:
    if sys.platform != "win32":
        return "n/a"
    try:
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        cls = ctypes.windll.kernel32.GetPriorityClass(ctypes.c_void_p(handle))
        return {
            0x00000040: "IDLE",
            0x00004000: "BELOW_NORMAL",
            0x00000020: "NORMAL",
            0x00008000: "ABOVE_NORMAL",
            0x00000080: "HIGH",
            0x00000100: "REALTIME",
        }.get(cls, f"UNKNOWN(0x{cls:x})")
    except Exception:
        return "UNKNOWN"


_PROCESS_START_UTC = None  # set once in main() at process start

LEDGER_FIELDS = (
    "observation_id",
    "protocol_id",
    "protocol_status",
    "permission",
    "decision",
    "lifecycle_id",
    "symbol",
    "liq_side",
    "fade_direction",
    "signal_ts_ms",
    "signal_utc",
    "bucket",
    "threshold_usd",
    "vdepth_bps",
    "prior_4h_bps",
    "running_notional",
    "running_liq_count",
    "running_rate_usd_per_sec",
    "running_accel_usd_per_sec",
    "elapsed_since_first_sec",
    "single_liq_dominance_pct",
    "book_ts_ms",
    "book_staleness_ms",
    "bid_depth_usd",
    "ask_depth_usd",
    "book_imbalance",
    "spread_bps",
    "anchor_mark_price",
    "initial_offset_bps",
    "replace_offset_bps",
    "wait_sec",
    "cross_margin_bps",
    "initial_limit_price",
    "replace_limit_price",
    "fill_leg",
    "maker_fill_ts_ms",
    "maker_fill_utc",
    "fill_delay_sec",
    "entry_price",
    "exit_ts_ms",
    "exit_utc",
    "exit_reason",
    "exit_price",
    "gross_bps",
    "fee_bps",
    "net_bps",
    "sim_status",
    "observation_status",
    "counterfactual_anchor_mark_net_bps",
    "momentum_arming_ts_ms",
    "momentum_arming_utc",
    "momentum_arming_delay_sec",
    "fill_minus_arm_sec",
    "entry_vs_arm_bps",
    "retest_depth_bucket",
    "fill_minus_arm_bucket",
    "retest_quality_score",
    "retest_quality_bucket",
    "entry_quality_tags",
    "entry_quality_warnings",
    "notes",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def max_mark_ts(conn: sqlite3.Connection, symbol: str) -> int | None:
    row = conn.execute("SELECT MAX(ts_ms) FROM mark_prices WHERE symbol=?", (symbol,)).fetchone()
    return int(row[0]) if row and row[0] is not None else None


def observation_id(*, signal_ts_ms: int, bucket: int, vdepth_bps: float, bid_depth_usd: float) -> str:
    raw = f"{PROTOCOL_ID}|{SYMBOL}|{LIQ_SIDE}|{bucket}|{signal_ts_ms}|{vdepth_bps:.6f}|{bid_depth_usd:.6f}"
    return sha256_text(raw)[:24]


def observation_status(sim: dict[str, Any], data_end_ms: int | None) -> str:
    if data_end_ms is None:
        return "PENDING"
    if sim.get("status") in {"NO_EXIT_BOOK", "NO_EXIT_FILL"}:
        return "DATA_INCOMPLETE"
    if sim.get("status") == "FILLED" and sim.get("exit_ts_ms") is not None:
        return "CLOSED" if int(sim["exit_ts_ms"]) <= int(data_end_ms) else "PENDING"
    if sim.get("status") == "NO_MAKER_FILL":
        close_ms = int(sim["anchor_ts_ms"]) + HORIZON_SEC * 1000
        return "CLOSED" if close_ms <= int(data_end_ms) else "PENDING"
    return "PENDING"


def closed_filled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if r.get("observation_status") == "CLOSED"
        and r.get("sim_status") == "FILLED"
        and r.get("net_bps") is not None
    ]


def status_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "UNKNOWN")
        out[value] = int(out.get(value, 0)) + 1
    return dict(sorted(out.items()))


def recent_rows(rows: list[dict[str, Any]], days: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    end_ms = max(int(r["signal_ts_ms"]) for r in rows if r.get("signal_ts_ms") is not None)
    start_ms = end_ms - int(days) * 24 * 3600 * 1000
    return [r for r in rows if int(r.get("signal_ts_ms") or 0) >= start_ms]


def mark_at_or_after(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> tuple[int, float] | None:
    row = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? AND ts_ms>=? ORDER BY ts_ms ASC LIMIT 1",
        (symbol, int(ts_ms)),
    ).fetchone()
    return (int(row[0]), float(row[1])) if row else None


def mark_ret_bps(conn: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    a = mark_at_or_after(conn, symbol, int(start_ms))
    b = mark_at_or_after(conn, symbol, int(end_ms))
    if not a or not b or float(a[1]) <= 0:
        return None
    return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0


def trade_flow(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT is_buyer_maker, COALESCE(SUM(notional),0.0), COUNT(*)
        FROM agg_trades
        WHERE symbol=? AND ts_ms>=? AND ts_ms<?
        GROUP BY is_buyer_maker
        """,
        (SYMBOL, int(start_ms), int(end_ms)),
    ).fetchall()
    taker_buy = 0.0
    taker_sell = 0.0
    count = 0
    for is_buyer_maker, notion, c in rows:
        count += int(c or 0)
        if int(is_buyer_maker) == 0:
            taker_buy += float(notion or 0.0)
        else:
            taker_sell += float(notion or 0.0)
    total = taker_buy + taker_sell
    return {
        "taker_buy_usd": taker_buy,
        "taker_sell_usd": taker_sell,
        "taker_imbalance": (taker_buy - taker_sell) / total if total > 0 else None,
        "agg_trade_count": count,
    }


def sell_liq_notional(conn: sqlite3.Connection, start_ms: int, end_ms: int) -> float:
    row = conn.execute(
        """
        SELECT COALESCE(SUM(notional),0.0)
        FROM liquidations
        WHERE symbol=? AND side='SELL' AND ts_ms>=? AND ts_ms<?
        """,
        (SYMBOL, int(start_ms), int(end_ms)),
    ).fetchone()
    return float(row[0] or 0.0) if row else 0.0


def first_momentum_arming_ts(conn: sqlite3.Connection, anchor_ts_ms: int, max_sec: int = 600) -> int | None:
    # FLOW_POSITIVE_ONLY from the research report: first positive taker-flow
    # imbalance after the anchor. This is causal and cheap enough for shadow.
    for sec in range(5, int(max_sec) + 1):
        end_ms = int(anchor_ts_ms) + sec * 1000
        flow = trade_flow(conn, end_ms - 5_000, end_ms)
        imb = flow.get("taker_imbalance")
        if imb is not None and float(imb) > 0.0:
            return end_ms
    return None


def price_ret_bps(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(float(a)) or float(a) <= 0:
        return None
    return (float(b) - float(a)) / float(a) * 10_000.0


def retest_depth_bucket(v: float | None) -> str:
    if v is None:
        return "UNKNOWN"
    x = float(v)
    if x <= -20.0:
        return "DEEP_RETEST_GE20"
    if x <= -10.0:
        return "MID_RETEST_10_20"
    if x <= -2.0:
        return "LIGHT_RETEST_2_10"
    if x <= 0.0:
        return "TOUCH_RETEST_0_2"
    return "CHASE_ABOVE_ARM"


def fill_delay_bucket(v: float | None) -> str:
    if v is None:
        return "UNKNOWN"
    x = float(v)
    if x <= 60.0:
        return "FAST_0_60S"
    if x <= 300.0:
        return "NORMAL_60_300S"
    if x <= 900.0:
        return "SLOW_300_900S"
    return "LATE_GT900S"


def retest_quality_bucket(score: int | None) -> str:
    if score is None:
        return "UNKNOWN"
    if int(score) >= 7:
        return "RETEST_QUALITY_HIGH"
    if int(score) >= 5:
        return "RETEST_QUALITY_MID"
    return "RETEST_QUALITY_LOW"


def entry_quality_for_sim(
    conn: sqlite3.Connection,
    *,
    anchor_ts_ms: int,
    anchor_book: dict[str, Any],
    sim: dict[str, Any],
) -> dict[str, Any]:
    if sim.get("status") != "FILLED" or sim.get("maker_fill_ts_ms") is None or sim.get("entry_price") is None:
        return {
            "momentum_arming_ts_ms": None,
            "momentum_arming_utc": None,
            "momentum_arming_delay_sec": None,
            "fill_minus_arm_sec": None,
            "entry_vs_arm_bps": None,
            "retest_depth_bucket": "NO_FILL",
            "fill_minus_arm_bucket": "NO_FILL",
            "retest_quality_score": None,
            "retest_quality_bucket": "NO_FILL",
            "entry_quality_tags": "",
            "entry_quality_warnings": "",
        }
    fill_ts = int(sim["maker_fill_ts_ms"])
    entry_px = float(sim["entry_price"])
    arm_ts = first_momentum_arming_ts(conn, int(anchor_ts_ms))
    arm_mark = mark_at_or_after(conn, SYMBOL, arm_ts) if arm_ts is not None else None
    arm_px = float(arm_mark[1]) if arm_mark else None
    fill_book = book_features_at(conn, SYMBOL, fill_ts, 10)

    fill_minus_arm_sec = (fill_ts - arm_ts) / 1000.0 if arm_ts is not None else None
    entry_vs_arm = price_ret_bps(arm_px, entry_px) if arm_px is not None else None
    anchor_bid = float(anchor_book.get("bid_depth_usd") or 0.0)
    fill_bid = float((fill_book or {}).get("bid_depth_usd") or 0.0)
    bid_ratio = (fill_bid / anchor_bid) if anchor_bid > 0 else None
    anchor_spread = float(anchor_book.get("spread_bps") or 0.0)
    fill_spread = float((fill_book or {}).get("spread_bps") or 0.0)
    sell_liq_15s = sell_liq_notional(conn, fill_ts - 15_000, fill_ts)

    score = 0
    tags: list[str] = []
    warnings: list[str] = []

    def add(cond: bool, tag: str, points: int = 1) -> None:
        nonlocal score
        if cond:
            score += points
            tags.append(tag)

    def sub(cond: bool, tag: str, points: int = 1) -> None:
        nonlocal score
        if cond:
            score -= points
            warnings.append(tag)

    add(arm_ts is not None and fill_ts >= arm_ts, "POST_ARM_FILL")
    add(entry_vs_arm is not None and entry_vs_arm <= 0.0, "PULLBACK_FILL")
    add(entry_vs_arm is not None and -25.0 <= entry_vs_arm <= -2.0, "RETEST_BAND_2_25")
    add(fill_bid >= MIN_BID_DEPTH_USD, "BID_STILL_THERE")
    add(bid_ratio is not None and bid_ratio >= 0.8, "BID_DEPTH_RETAINED")
    add(fill_spread <= anchor_spread + 0.05, "SPREAD_CLEAN")
    add(fill_minus_arm_sec is not None and fill_minus_arm_sec <= 300.0, "FAST_RETEST_FILL")
    add(sell_liq_15s <= 250_000.0, "NO_LARGE_SELL_LIQ_RESTART")

    sub(bid_ratio is not None and bid_ratio < 0.5, "BID_VANISHED")
    sub(fill_spread > anchor_spread + 0.2, "SPREAD_EXPANDING")
    sub(fill_minus_arm_sec is not None and fill_minus_arm_sec > 900.0, "LATE_RETEST_FILL")
    sub(sell_liq_15s > 250_000.0, "LARGE_SELL_LIQ_RESTART")

    return {
        "momentum_arming_ts_ms": arm_ts,
        "momentum_arming_utc": iso_ms(arm_ts) if arm_ts is not None else None,
        "momentum_arming_delay_sec": r1((arm_ts - int(anchor_ts_ms)) / 1000.0) if arm_ts is not None else None,
        "fill_minus_arm_sec": r1(fill_minus_arm_sec),
        "entry_vs_arm_bps": r1(entry_vs_arm),
        "retest_depth_bucket": retest_depth_bucket(entry_vs_arm),
        "fill_minus_arm_bucket": fill_delay_bucket(fill_minus_arm_sec),
        "retest_quality_score": int(score),
        "retest_quality_bucket": retest_quality_bucket(score),
        "entry_quality_tags": ",".join(tags),
        "entry_quality_warnings": ",".join(warnings),
    }


def weekly_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ts = row.get("signal_ts_ms")
        if ts is None:
            continue
        dt = datetime.fromtimestamp(int(ts) / 1000.0, tz=timezone.utc)
        iso = dt.isocalendar()
        groups.setdefault(f"{iso.year}-W{iso.week:02d}", []).append(row)
    out = []
    for key, items in sorted(groups.items()):
        fills = closed_filled(items)
        vals = [float(r["net_bps"]) for r in fills]
        out.append(
            {
                "week": key,
                "signals": len(items),
                "closed": sum(1 for r in items if r.get("observation_status") == "CLOSED"),
                "pending": sum(1 for r in items if r.get("observation_status") == "PENDING"),
                "filled": len(fills),
                "fill_rate": r3(len(fills) / len(items)) if items else None,
                "summary": summarize(vals),
            }
        )
    return out


def _row_from_event(
    conn: sqlite3.Connection,
    event: Any,
    marks: MarkIndex,
    *,
    data_end: int | None,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
    fee_bps: float,
) -> dict[str, Any] | None:
    """Single source of truth for turning a candidate FadeEvent into a ledger
    row. Shared by the legacy full-recompute path and the incremental path so
    both produce byte-identical rows for the same event (output parity by
    construction, not just by testing).
    """
    ts = int(event.anchor.anchor_ts_ms)
    prior4h = prior_return_bps(marks, ts, 4 * 3600)
    if prior4h is None or not math.isfinite(float(prior4h)) or not (float(prior4h) < PRIOR4H_LT_BPS):
        return None
    book = book_features_at(conn, SYMBOL, ts, int(max_book_staleness_sec))
    if not book or float(book.get("bid_depth_usd") or 0.0) < MIN_BID_DEPTH_USD:
        return None
    sim = simulate_cancel_replace(
        conn,
        event,
        initial_offset_bps=INITIAL_OFFSET_BPS,
        replace_offset_bps=REPLACE_OFFSET_BPS,
        wait_sec=WAIT_SEC,
        cross_margin_bps=CROSS_MARGIN_BPS,
        maker_fee_bps=float(maker_fee_bps),
        taker_fee_bps=float(taker_fee_bps),
        max_book_staleness_sec=int(max_book_staleness_sec),
    )
    entry_quality = entry_quality_for_sim(conn, anchor_ts_ms=ts, anchor_book=book, sim=sim)
    status = observation_status(sim, data_end)
    bid_depth = float(book["bid_depth_usd"])
    oid = observation_id(
        signal_ts_ms=ts,
        bucket=int(event.anchor.bucket),
        vdepth_bps=float(event.vdepth_bps),
        bid_depth_usd=bid_depth,
    )
    return {
        "observation_id": oid,
        "protocol_id": PROTOCOL_ID,
        "protocol_status": PROTOCOL_STATUS,
        "permission": PERMISSION,
        "decision": DECISION,
        "lifecycle_id": LIFECYCLE_ID,
        "symbol": SYMBOL,
        "liq_side": LIQ_SIDE,
        "fade_direction": FADE_DIRECTION,
        "signal_ts_ms": ts,
        "signal_utc": iso_ms(ts),
        "bucket": int(event.anchor.bucket),
        "threshold_usd": THRESHOLD_USD,
        "vdepth_bps": r1(event.vdepth_bps),
        "prior_4h_bps": r1(prior4h),
        "running_notional": r1(event.anchor.running_notional),
        "running_liq_count": int(event.anchor.running_liq_count),
        "running_rate_usd_per_sec": r1(event.anchor.running_rate),
        "running_accel_usd_per_sec": r1(event.anchor.running_accel),
        "elapsed_since_first_sec": r1(event.anchor.elapsed_since_first_sec),
        "single_liq_dominance_pct": r1(event.anchor.running_single_liq_dominance),
        "book_ts_ms": int(book["book_ts_ms"]),
        "book_staleness_ms": int(book["book_staleness_ms"]),
        "bid_depth_usd": r1(book["bid_depth_usd"]),
        "ask_depth_usd": r1(book["ask_depth_usd"]),
        "book_imbalance": r3(book["book_imbalance"]),
        "spread_bps": r1(book["spread_bps"]),
        "anchor_mark_price": sim.get("anchor_mark_price"),
        "initial_offset_bps": INITIAL_OFFSET_BPS,
        "replace_offset_bps": REPLACE_OFFSET_BPS,
        "wait_sec": WAIT_SEC,
        "cross_margin_bps": CROSS_MARGIN_BPS,
        "initial_limit_price": sim.get("initial_limit_price"),
        "replace_limit_price": sim.get("replace_limit_price"),
        "fill_leg": sim.get("fill_leg"),
        "maker_fill_ts_ms": sim.get("maker_fill_ts_ms"),
        "maker_fill_utc": sim.get("maker_fill_utc"),
        "fill_delay_sec": r1(sim.get("fill_delay_sec")),
        "entry_price": sim.get("entry_price"),
        "exit_ts_ms": sim.get("exit_ts_ms"),
        "exit_utc": sim.get("exit_utc"),
        "exit_reason": sim.get("exit_reason"),
        "exit_price": sim.get("exit_price"),
        "gross_bps": r1(sim.get("gross_bps")),
        "fee_bps": r1(sim.get("fee_bps")),
        "net_bps": r1(sim.get("net_bps")),
        "sim_status": sim.get("status"),
        "observation_status": status,
        "counterfactual_anchor_mark_net_bps": r1(anchor_mark_counterfactual(marks, ts, fee_bps=fee_bps)),
        "momentum_arming_ts_ms": entry_quality.get("momentum_arming_ts_ms"),
        "momentum_arming_utc": entry_quality.get("momentum_arming_utc"),
        "momentum_arming_delay_sec": entry_quality.get("momentum_arming_delay_sec"),
        "fill_minus_arm_sec": entry_quality.get("fill_minus_arm_sec"),
        "entry_vs_arm_bps": entry_quality.get("entry_vs_arm_bps"),
        "retest_depth_bucket": entry_quality.get("retest_depth_bucket"),
        "fill_minus_arm_bucket": entry_quality.get("fill_minus_arm_bucket"),
        "retest_quality_score": entry_quality.get("retest_quality_score"),
        "retest_quality_bucket": entry_quality.get("retest_quality_bucket"),
        "entry_quality_tags": entry_quality.get("entry_quality_tags"),
        "entry_quality_warnings": entry_quality.get("entry_quality_warnings"),
        "notes": "paper_shadow_mirror_only_no_order",
    }


def build_rows(
    conn: sqlite3.Connection,
    *,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> list[dict[str, Any]]:
    """Legacy full-history recompute path. O(full mark_prices + liquidations
    history) per call -- kept only for --full-recompute output-parity checks
    against build_rows_incremental; the production loop no longer calls this.
    """
    marks = load_mark_index(conn, SYMBOL)
    data_end = max_mark_ts(conn, SYMBOL)
    fee_bps = float(maker_fee_bps) + float(taker_fee_bps)
    rows: list[dict[str, Any]] = []
    for event in collect_v01_events(conn):
        row = _row_from_event(
            conn,
            event,
            marks,
            data_end=data_end,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            max_book_staleness_sec=max_book_staleness_sec,
            fee_bps=fee_bps,
        )
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: int(r["signal_ts_ms"]))
    return rows


def collect_events_bounded(
    marks: MarkIndex,
    liqs: list[dict[str, Any]],
    *,
    seed_last_kept_ts_ms: int,
) -> tuple[list[FadeEvent], int]:
    """Bounded-window equivalent of collect_v01_events: reconstructs anchors
    only from the caller-supplied liquidation window (not full history) and
    applies the identical vdepth-band + prior-4h filters. seed_last_kept_ts_ms
    resumes the cross-tick min-gap suppression state so bucket-boundary
    behavior matches the full-history reconstruction exactly.

    Returns (events, new_last_kept_ts_ms) so the caller can persist the
    updated gap-suppression watermark in the checkpoint.
    """
    anchors = reconstruct_anchors(
        liqs,
        bucket_sec=BUCKET_SEC,
        min_gap_sec=MIN_GAP_SEC,
        thresholds=(THRESHOLD_USD,),
        accel_window_sec=ACCEL_WINDOW_SEC,
        seed_last_kept={THRESHOLD_USD: seed_last_kept_ts_ms},
    )
    new_last_kept_ts_ms = int(anchors[-1].anchor_ts_ms) if anchors else int(seed_last_kept_ts_ms)

    events: list[FadeEvent] = []
    for anchor in anchors:
        depth = anchor_vdepth_bps(marks, anchor, LIQ_SIDE)
        if depth is None or not (VDEPTH_MIN_BPS <= float(depth) < VDEPTH_MAX_BPS):
            continue
        mark = marks.at_or_after(int(anchor.anchor_ts_ms))
        if not mark:
            continue
        path = tuple(
            (int(ts), float(px))
            for ts, px in marks.slice_range(int(mark[0]), int(mark[0]) + HORIZON_SEC * 1000)
            if int(ts) > int(mark[0])
        )
        if not path:
            continue
        events.append(
            FadeEvent(
                symbol=SYMBOL,
                side=LIQ_SIDE,
                fade_direction=FADE_DIRECTION,
                anchor=anchor,
                anchor_mark_ts_ms=int(mark[0]),
                anchor_mark_price=float(mark[1]),
                vdepth_bps=float(depth),
                path=path,
            )
        )
    events.sort(key=lambda ev: int(ev.anchor.anchor_ts_ms))
    return events, new_last_kept_ts_ms


def _event_from_ledger_row(row: dict[str, Any], marks: MarkIndex) -> FadeEvent | None:
    """Reconstructs a FadeEvent for a still-PENDING ledger row without
    re-deriving it from raw liquidations. All AnchorSnapshot fields needed by
    simulate_cancel_replace/entry_quality_for_sim/row construction are already
    persisted in the ledger row; anchor_mark_ts_ms/path are recomputed fresh
    from the (immutable) historical mark data, which is safe and deterministic.
    """
    anchor_ts = int(row["signal_ts_ms"])
    mark = marks.at_or_after(anchor_ts)
    if not mark:
        return None
    path = tuple(
        (int(ts), float(px))
        for ts, px in marks.slice_range(int(mark[0]), int(mark[0]) + HORIZON_SEC * 1000)
        if int(ts) > int(mark[0])
    )
    if not path:
        return None
    elapsed = float(row.get("elapsed_since_first_sec") or 0.0)
    accel = float(row.get("running_accel_usd_per_sec") or 0.0)
    anchor = AnchorSnapshot(
        event_id=f"{int(row['bucket'])}:{int(THRESHOLD_USD)}",
        bucket=int(row["bucket"]),
        threshold_usd=float(row["threshold_usd"]),
        anchor_ts_ms=anchor_ts,
        first_ts_ms=anchor_ts - int(elapsed * 1000),
        elapsed_since_first_sec=elapsed,
        running_notional=float(row["running_notional"]),
        running_liq_count=int(row["running_liq_count"]),
        running_rate=float(row["running_rate_usd_per_sec"]),
        running_accel=accel,
        running_single_liq_dominance=float(row["single_liq_dominance_pct"]),
        acceleration_bucket="accelerating" if accel > 0.0 else "decelerating",
        max_single_notional=0.0,  # not read by simulate_cancel_replace/entry_quality/_row_from_event
    )
    return FadeEvent(
        symbol=SYMBOL,
        side=LIQ_SIDE,
        fade_direction=FADE_DIRECTION,
        anchor=anchor,
        anchor_mark_ts_ms=int(mark[0]),
        anchor_mark_price=float(mark[1]),
        vdepth_bps=float(row["vdepth_bps"]),
        path=path,
    )


def build_rows_incremental(
    conn: sqlite3.Connection,
    checkpoint: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    *,
    maker_fee_bps: float,
    taker_fee_bps: float,
    max_book_staleness_sec: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Bounded steady-state tick: reads only [checkpoint.closed_before, now]
    liquidations plus a bounded mark_prices window sized to what the new/
    carried-forward candidates actually need -- never full history.
    """
    bucket_ms = BUCKET_SEC * 1000
    data_end = max_mark_ts(conn, SYMBOL)
    _earliest_liq_ts, latest_liq_ts = liq_ts_bounds(conn, SYMBOL, LIQ_SIDE)

    closed_before = int(checkpoint["closed_before_ts_ms"])
    last_kept_ts = int(checkpoint["last_kept_ts_ms"])

    new_closed_before = closed_before
    if latest_liq_ts is not None:
        safe_boundary = latest_liq_ts - CLOSE_GRACE_SEC * 1000
        target_closed_before = floor_to_bucket(safe_boundary, bucket_ms)
        target_closed_before = min(target_closed_before, closed_before + BOOTSTRAP_CHUNK_SEC * 1000)
        new_closed_before = max(closed_before, target_closed_before)

    liq_rows_read = 0
    closing_liqs: list[dict[str, Any]] = []
    if new_closed_before > closed_before:
        closing_liqs = load_liquidations(conn, SYMBOL, LIQ_SIDE, closed_before, new_closed_before - 1)
        liq_rows_read += len(closing_liqs)

    open_range_end = latest_liq_ts if latest_liq_ts is not None else new_closed_before
    open_liqs: list[dict[str, Any]] = []
    if open_range_end is not None and open_range_end >= new_closed_before:
        open_liqs = load_liquidations(
            conn, SYMBOL, LIQ_SIDE, new_closed_before, open_range_end + OPEN_WINDOW_MARGIN_SEC * 1000
        )
        liq_rows_read += len(open_liqs)

    pending_rows = [r for r in existing_rows if r.get("observation_status") == "PENDING"]
    candidate_ts_sources = (
        [int(r["ts_ms"]) for r in closing_liqs]
        + [int(r["ts_ms"]) for r in open_liqs]
        + [int(r["signal_ts_ms"]) for r in pending_rows]
    )
    if candidate_ts_sources:
        min_needed_ts = min(candidate_ts_sources) - MARK_LOOKBACK_MARGIN_SEC * 1000
        marks = load_mark_index_range(conn, SYMBOL, min_needed_ts, None)
    else:
        marks = MarkIndex([])
    mark_rows_read = len(marks.ts)

    closing_events, seed_after_closing = collect_events_bounded(
        marks, closing_liqs, seed_last_kept_ts_ms=last_kept_ts
    )
    open_events, _seed_after_open = collect_events_bounded(
        marks, open_liqs, seed_last_kept_ts_ms=seed_after_closing
    )
    all_events = sorted(closing_events + open_events, key=lambda ev: int(ev.anchor.anchor_ts_ms))

    fee_bps = float(maker_fee_bps) + float(taker_fee_bps)
    new_rows: list[dict[str, Any]] = []
    for event in all_events:
        row = _row_from_event(
            conn,
            event,
            marks,
            data_end=data_end,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            max_book_staleness_sec=max_book_staleness_sec,
            fee_bps=fee_bps,
        )
        if row is not None:
            new_rows.append(row)

    for prow in pending_rows:
        event = _event_from_ledger_row(prow, marks)
        if event is None:
            continue
        row = _row_from_event(
            conn,
            event,
            marks,
            data_end=data_end,
            maker_fee_bps=maker_fee_bps,
            taker_fee_bps=taker_fee_bps,
            max_book_staleness_sec=max_book_staleness_sec,
            fee_bps=fee_bps,
        )
        if row is not None:
            new_rows.append(row)

    new_rows.sort(key=lambda r: int(r["signal_ts_ms"]))

    updated_checkpoint = dict(checkpoint)
    updated_checkpoint["closed_before_ts_ms"] = int(new_closed_before)
    updated_checkpoint["last_kept_ts_ms"] = int(seed_after_closing)
    updated_checkpoint["updated_at_utc"] = utc_now()

    stats = {
        "liq_rows_read": int(liq_rows_read),
        "mark_rows_read": int(mark_rows_read),
        "events_considered": int(len(all_events)),
        "pending_carried_forward": int(len(pending_rows)),
        "closed_advance_ms": int(new_closed_before - closed_before),
        "checkpoint_lag_ms": int(latest_liq_ts - new_closed_before) if latest_liq_ts is not None else None,
    }
    return new_rows, updated_checkpoint, stats


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def merge_rows(existing: list[dict[str, Any]], observed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    by_id = {str(row["observation_id"]): row for row in existing if row.get("observation_id")}
    added = 0
    for row in observed:
        oid = str(row["observation_id"])
        if oid not in by_id:
            added += 1
        by_id[oid] = row
    merged = list(by_id.values())
    merged.sort(key=lambda r: (int(r.get("signal_ts_ms") or 0), str(r.get("observation_id") or "")))
    return merged, added


def build_brief(rows: list[dict[str, Any]], *, brief_days: int, source_db: dict[str, Any], added_n: int) -> dict[str, Any]:
    recent = recent_rows(rows, brief_days)
    all_fills = closed_filled(rows)
    recent_fills = closed_filled(recent)
    no_fill_closed = [r for r in rows if r.get("observation_status") == "CLOSED" and r.get("sim_status") == "NO_MAKER_FILL"]
    no_fill_cf = [
        float(r["counterfactual_anchor_mark_net_bps"])
        for r in no_fill_closed
        if r.get("counterfactual_anchor_mark_net_bps") is not None
    ]
    all_vals = [float(r["net_bps"]) for r in all_fills]
    recent_vals = [float(r["net_bps"]) for r in recent_fills]
    recent_summary = summarize(recent_vals)
    kill_triggered = (
        int(recent_summary["n"] or 0) >= 3
        and float(recent_summary["top3_winner_removed_sum_bps"] or 0.0) < 0.0
    )
    return {
        "generated_at_utc": utc_now(),
        "source_db": source_db,
        "protocol": {
            "id": PROTOCOL_ID,
            "status": PROTOCOL_STATUS,
            "permission": PERMISSION,
            "decision": DECISION,
            "live_rule_match": True,
            "lifecycle": LIFECYCLE_ID,
        },
        "config": {
            "symbol": SYMBOL,
            "liq_side": LIQ_SIDE,
            "fade_direction": FADE_DIRECTION,
            "threshold_usd": THRESHOLD_USD,
            "vdepth_min_bps": VDEPTH_MIN_BPS,
            "vdepth_max_bps": VDEPTH_MAX_BPS,
            "prior4h_lt_bps": PRIOR4H_LT_BPS,
            "min_bid_depth_usd": MIN_BID_DEPTH_USD,
            "initial_offset_bps": INITIAL_OFFSET_BPS,
            "replace_offset_bps": REPLACE_OFFSET_BPS,
            "wait_sec": WAIT_SEC,
            "cross_margin_bps": CROSS_MARGIN_BPS,
            "horizon_sec": HORIZON_SEC,
        },
        "ledger": {
            "rows_total": len(rows),
            "rows_added_this_run": int(added_n),
            "observation_status_counts": status_counts(rows, "observation_status"),
            "sim_status_counts": status_counts(rows, "sim_status"),
        },
        "overall": {
            "signals": len(rows),
            "closed_filled": len(all_fills),
            "fill_rate": r3(len(all_fills) / len(rows)) if rows else None,
            "summary": summarize(all_vals),
            "closed_no_fill_n": len(no_fill_closed),
            "closed_no_fill_counterfactual_summary": summarize(no_fill_cf),
        },
        "recent": {
            "days": int(brief_days),
            "signals": len(recent),
            "closed_filled": len(recent_fills),
            "fill_rate": r3(len(recent_fills) / len(recent)) if recent else None,
            "summary": recent_summary,
            "kill_check": {
                "rule": "60-day forward T3R < 0 after at least 3 closed fills",
                "triggered": bool(kill_triggered),
            },
        },
        "weekly": weekly_groups(rows),
        "latest_observations": rows[-10:],
    }


def summary_cell(summary: dict[str, Any]) -> str:
    return (
        f"N={summary['n']} sum={summary['sum_bps']} med={summary['median_bps']} "
        f"T3R={summary['top3_winner_removed_sum_bps']}"
    )


def render_md(brief: dict[str, Any]) -> str:
    p = brief["protocol"]
    overall = brief["overall"]
    recent = brief["recent"]
    lines = [
        "# S34 V Engine v0.2 Shadow Mirror Brief",
        "",
        f"Generated: `{brief['generated_at_utc']}`",
        "",
        f"Protocol: `{p['id']}`",
        "",
        f"Status: `{p['status']}` / `{p['decision']}`. This is paper/shadow mirror only; it never sends orders.",
        "",
        "## Ledger",
        "",
        f"- rows total: `{brief['ledger']['rows_total']}`",
        f"- rows added this run: `{brief['ledger']['rows_added_this_run']}`",
        f"- observation counts: `{brief['ledger']['observation_status_counts']}`",
        f"- sim counts: `{brief['ledger']['sim_status_counts']}`",
        "",
        "## Performance Labels",
        "",
        f"- overall: signals `{overall['signals']}`, closed fills `{overall['closed_filled']}`, fill rate `{overall['fill_rate']}`, {summary_cell(overall['summary'])}",
        f"- recent {recent['days']}d: signals `{recent['signals']}`, closed fills `{recent['closed_filled']}`, fill rate `{recent['fill_rate']}`, {summary_cell(recent['summary'])}",
        f"- no-fill counterfactual: closed no-fill `{overall['closed_no_fill_n']}`, {summary_cell(overall['closed_no_fill_counterfactual_summary'])}",
        f"- kill check: `{'TRIGGERED' if recent['kill_check']['triggered'] else 'not triggered'}` ({recent['kill_check']['rule']})",
        "",
        "## Latest Observations",
        "",
        "| UTC | Status | Sim | Leg | V-depth | Bid depth | Fill delay | Net | CF mark net |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in brief["latest_observations"]:
        lines.append(
            f"| {row.get('signal_utc')} | {row.get('observation_status')} | {row.get('sim_status')} | "
            f"{row.get('fill_leg')} | {row.get('vdepth_bps')} | {row.get('bid_depth_usd')} | "
            f"{row.get('fill_delay_sec')} | {row.get('net_bps')} | {row.get('counterfactual_anchor_mark_net_bps')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_state(
    path: Path,
    brief: dict[str, Any] | None,
    *,
    loop: bool,
    args: argparse.Namespace | None = None,
    checkpoint: dict[str, Any] | None = None,
    tick_stats: dict[str, Any] | None = None,
    tick_duration_ms: float | None = None,
    error: str | None = None,
    status: str | None = None,
) -> None:
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prev = {}
    payload: dict[str, Any] = {
        "updated_at_utc": utc_now(),
        "pid": os.getpid(),
        "loop": bool(loop),
        "protocol_id": PROTOCOL_ID,
        "process_start_utc": _PROCESS_START_UTC or prev.get("process_start_utc"),
        "priority_class": _own_priority_class(),
        "interval_sec": int(args.interval_sec) if args is not None else prev.get("interval_sec"),
        "db_path": str(args.db) if args is not None else prev.get("db_path"),
        "last_error": error,
    }
    if status is not None:
        payload["status"] = status
    elif brief is not None:
        payload.update(
            {
                "status": "OK",
                "ledger_rows": (brief.get("ledger") or {}).get("rows_total"),
                "rows_added_this_run": (brief.get("ledger") or {}).get("rows_added_this_run"),
                "latest_signal_utc": ((brief.get("latest_observations") or [{}])[-1] or {}).get("signal_utc"),
            }
        )
    if checkpoint is not None:
        closed_before = int(checkpoint.get("closed_before_ts_ms") or 0)
        payload["checkpoint"] = {
            "closed_before_ts_ms": closed_before,
            "closed_before_utc": iso_ms(closed_before),
            "last_kept_ts_ms": checkpoint.get("last_kept_ts_ms"),
        }
    if tick_stats is not None:
        payload["tick_stats"] = tick_stats
    if tick_duration_ms is not None:
        payload["last_tick_duration_ms"] = round(float(tick_duration_ms), 1)
    if error is None and brief is not None:
        payload["last_success_utc"] = utc_now()
        payload["last_processed_ts_ms"] = (checkpoint or {}).get("closed_before_ts_ms")
    else:
        payload["last_success_utc"] = prev.get("last_success_utc")
        payload["last_processed_ts_ms"] = prev.get("last_processed_ts_ms")
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=True))


def update_once(args: argparse.Namespace) -> dict[str, Any]:
    tick_start = time.time()
    checkpoint: dict[str, Any] | None = None
    tick_stats: dict[str, Any] = {}
    with sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30) as conn:
        conn.execute("PRAGMA busy_timeout=30000")
        existing = load_jsonl(args.ledger_jsonl)
        if bool(getattr(args, "full_recompute", False)):
            observed = build_rows(
                conn,
                maker_fee_bps=float(args.maker_fee_bps),
                taker_fee_bps=float(args.taker_fee_bps),
                max_book_staleness_sec=int(args.max_book_staleness_sec),
            )
        else:
            checkpoint = load_checkpoint(args.checkpoint_path)
            if checkpoint is None:
                checkpoint = bootstrap_checkpoint(conn, existing)
            observed, checkpoint, tick_stats = build_rows_incremental(
                conn,
                checkpoint,
                existing,
                maker_fee_bps=float(args.maker_fee_bps),
                taker_fee_bps=float(args.taker_fee_bps),
                max_book_staleness_sec=int(args.max_book_staleness_sec),
            )
    merged, added = merge_rows(existing, observed)
    brief = build_brief(
        merged,
        brief_days=int(args.brief_days),
        source_db=file_fingerprint(args.db),
        added_n=added,
    )
    write_jsonl(args.ledger_jsonl, merged)
    write_csv(args.ledger_csv, merged)
    args.brief_json.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(args.brief_json, json.dumps(brief, indent=2, ensure_ascii=True))
    _atomic_write_text(args.brief_md, render_md(brief))
    # Checkpoint only advances after the ledger/brief writes above have committed,
    # so a crash between them just repeats the same (idempotent) work next tick.
    if checkpoint is not None:
        save_checkpoint_atomic(args.checkpoint_path, checkpoint)
    tick_ms = (time.time() - tick_start) * 1000.0
    write_state(
        args.state_path,
        brief,
        loop=bool(args.loop),
        args=args,
        checkpoint=checkpoint,
        tick_stats=tick_stats,
        tick_duration_ms=tick_ms,
    )
    try:
        from tools.s34_v_engine_sizing_shadow_paper import parse_args as sizing_parse_args
        from tools.s34_v_engine_sizing_shadow_paper import run as sizing_run

        sizing_run(sizing_parse_args([]))
    except Exception as exc:
        brief["sizing_shadow_refresh_error"] = f"{type(exc).__name__}: {exc}"
    return brief


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mirror S34 V Engine v0.2 in shadow/paper observation mode.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ledger-jsonl", type=Path, default=DEFAULT_LEDGER_JSONL)
    p.add_argument("--ledger-csv", type=Path, default=DEFAULT_LEDGER_CSV)
    p.add_argument("--brief-json", type=Path, default=DEFAULT_BRIEF_JSON)
    p.add_argument("--brief-md", type=Path, default=DEFAULT_BRIEF_MD)
    p.add_argument("--state-path", type=Path, default=STATE_PATH)
    p.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    p.add_argument("--lock-path", type=Path, default=LOCK_PATH)
    p.add_argument("--no-lock", action="store_true", help="skip the duplicate-instance lock (tests only)")
    p.add_argument(
        "--full-recompute",
        action="store_true",
        help="legacy full-history recompute path, for output-parity verification only",
    )
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--taker-fee-bps", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=10)
    p.add_argument("--brief-days", type=int, default=60)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval-sec", type=int, default=60)
    p.add_argument("--pid-path", type=Path, default=PID_PATH)
    return p.parse_args(argv)


_SHUTDOWN_REQUESTED = False


def _handle_shutdown_signal(signum: int, frame: Any) -> None:
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True


def main(argv: list[str] | None = None) -> int:
    global _PROCESS_START_UTC
    args = parse_args(argv)
    _PROCESS_START_UTC = utc_now()

    if args.loop and not args.no_lock:
        try:
            acquire_lock(args.lock_path)
        except DuplicateInstanceError as exc:
            print(f"{utc_now()} BLOCKER duplicate_instance: {exc}", file=sys.stderr, flush=True)
            return 2

    try:
        signal.signal(signal.SIGINT, _handle_shutdown_signal)
    except (AttributeError, ValueError):
        pass
    try:
        signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    except (AttributeError, ValueError):
        pass

    try:
        if args.loop:
            args.pid_path.parent.mkdir(parents=True, exist_ok=True)
            args.pid_path.write_text(str(os.getpid()), encoding="utf-8")
            write_state(args.state_path, None, loop=True, args=args, status="STARTING")
            while True:
                tick_start = time.time()
                try:
                    brief = update_once(args)
                except Exception as exc:  # noqa: BLE001 - must never die silently; heartbeat carries exact reason
                    tick_ms = (time.time() - tick_start) * 1000.0
                    err = f"{type(exc).__name__}: {exc}"
                    write_state(args.state_path, None, loop=True, args=args, tick_duration_ms=tick_ms, error=err)
                    print(f"{utc_now()} ERROR {PROTOCOL_ID} {err}", file=sys.stderr, flush=True)
                else:
                    tick_ms = (time.time() - tick_start) * 1000.0
                    print(
                        f"{utc_now()} {PROTOCOL_ID} rows={brief['ledger']['rows_total']} "
                        f"added={brief['ledger']['rows_added_this_run']} tick_ms={tick_ms:.0f}",
                        flush=True,
                    )
                if _SHUTDOWN_REQUESTED:
                    print(f"{utc_now()} SHUTDOWN_REQUESTED, exiting after committed tick", flush=True)
                    break
                interval = max(10, int(args.interval_sec))
                slept = 0.0
                while slept < interval and not _SHUTDOWN_REQUESTED:
                    step = min(1.0, interval - slept)
                    time.sleep(step)
                    slept += step
            return 0
        brief = update_once(args)
        print(render_md(brief))
        return 0
    finally:
        if args.loop and not args.no_lock:
            release_lock(args.lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
