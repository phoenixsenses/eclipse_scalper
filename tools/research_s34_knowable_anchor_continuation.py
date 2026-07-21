from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.s34_feature_availability import (
    FeatureClass,
    FeatureValue,
    assert_feature_set_available,
)


DEFAULT_DB = ROOT / "data" / "microstructure.db"
OUT_DIR = ROOT / "reports" / "research" / "s34"
SURVIVAL_MD = OUT_DIR / "S34_KNOWABLE_ANCHOR_SURVIVAL_CURVE.md"
SURVIVAL_JSON = OUT_DIR / "S34_KNOWABLE_ANCHOR_SURVIVAL_CURVE.json"
HOLDOUT_MD = OUT_DIR / "S34_KNOWABLE_ANCHOR_HOLDOUT.md"
HOLDOUT_JSON = OUT_DIR / "S34_KNOWABLE_ANCHOR_HOLDOUT.json"

THRESHOLDS_USD = (50_000.0, 100_000.0, 150_000.0, 200_000.0, 300_000.0, 500_000.0)
HORIZONS_SEC = (30, 60, 120)
DIRECTIONS = ("LONG", "SHORT")
PARTITIONS = ("all", "accelerating", "decelerating")


@dataclass(frozen=True)
class BookQuote:
    ts_ms: int
    bid: float
    ask: float
    mid: float
    staleness_ms: int


@dataclass(frozen=True)
class AnchorSnapshot:
    event_id: str
    bucket: int
    threshold_usd: float
    anchor_ts_ms: int
    first_ts_ms: int
    elapsed_since_first_sec: float
    running_notional: float
    running_liq_count: int
    running_rate: float
    running_accel: float
    running_single_liq_dominance: float
    acceleration_bucket: str
    max_single_notional: float


class MarkIndex:
    def __init__(self, rows: Iterable[tuple[int, float]]) -> None:
        data = [(int(ts), float(px)) for ts, px in rows]
        self.ts = [r[0] for r in data]
        self.px = [r[1] for r in data]

    def at_or_after(self, ts_ms: int) -> tuple[int, float] | None:
        idx = bisect.bisect_left(self.ts, int(ts_ms))
        if idx >= len(self.ts):
            return None
        return self.ts[idx], self.px[idx]

    def at_or_before(self, ts_ms: int) -> tuple[int, float] | None:
        idx = bisect.bisect_right(self.ts, int(ts_ms)) - 1
        if idx < 0:
            return None
        return self.ts[idx], self.px[idx]

    def slice_range(self, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
        lo = bisect.bisect_left(self.ts, int(start_ms))
        hi = bisect.bisect_right(self.ts, int(end_ms))
        return list(zip(self.ts[lo:hi], self.px[lo:hi]))

    def ret_bps(self, start_ms: int, end_ms: int) -> float | None:
        a = self.at_or_before(int(start_ms))
        b = self.at_or_before(int(end_ms))
        if not a or not b or float(a[1]) <= 0.0:
            return None
        return (float(b[1]) - float(a[1])) / float(a[1]) * 10_000.0


def iso_ms(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat()


def pctile(values: list[float], q: float) -> float | None:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def r1(value: float | None) -> float | None:
    return round(float(value), 1) if value is not None and math.isfinite(float(value)) else None


def r3(value: float | None) -> float | None:
    return round(float(value), 3) if value is not None and math.isfinite(float(value)) else None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def book_at(conn: sqlite3.Connection, symbol: str, ts_ms: int, max_staleness_sec: int) -> BookQuote | None:
    row = conn.execute(
        """
        SELECT ts_ms, bid_price, ask_price, mid_price
        FROM book_ticker
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()
    if not row:
        return None
    stale = int(ts_ms) - int(row[0])
    if stale > int(max_staleness_sec) * 1000:
        return None
    return BookQuote(ts_ms=int(row[0]), bid=float(row[1]), ask=float(row[2]), mid=float(row[3]), staleness_ms=stale)


def load_mark_index(conn: sqlite3.Connection, symbol: str) -> MarkIndex:
    rows = conn.execute(
        "SELECT ts_ms, mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms",
        (symbol,),
    ).fetchall()
    return MarkIndex(rows)


def load_mark_index_range(
    conn: sqlite3.Connection,
    symbol: str,
    start_ms: int | None,
    end_ms: int | None = None,
) -> MarkIndex:
    """Bounded variant of load_mark_index for incremental/steady-state callers.

    Additive: existing load_mark_index (unbounded, full-history) is unchanged
    and remains the default for every other caller.
    """
    clauses = ["symbol=?"]
    params: list[Any] = [symbol]
    if start_ms is not None:
        clauses.append("ts_ms>=?")
        params.append(int(start_ms))
    if end_ms is not None:
        clauses.append("ts_ms<=?")
        params.append(int(end_ms))
    rows = conn.execute(
        f"SELECT ts_ms, mark_price FROM mark_prices WHERE {' AND '.join(clauses)} ORDER BY ts_ms",
        params,
    ).fetchall()
    return MarkIndex(rows)


def load_liquidations(
    conn: sqlite3.Connection,
    symbol: str,
    side: str,
    start_ms: int | None,
    end_ms: int | None,
) -> list[dict[str, Any]]:
    clauses = ["symbol=?", "side=?"]
    params: list[Any] = [symbol, side]
    if start_ms is not None:
        clauses.append("ts_ms>=?")
        params.append(int(start_ms))
    if end_ms is not None:
        clauses.append("ts_ms<=?")
        params.append(int(end_ms))
    rows = conn.execute(
        f"""
        SELECT ts_ms, price, quantity, notional, trade_time_ms
        FROM liquidations
        WHERE {' AND '.join(clauses)}
        ORDER BY ts_ms ASC
        """,
        params,
    ).fetchall()
    return [
        {
            "ts_ms": int(row[0]),
            "price": float(row[1]),
            "quantity": float(row[2]),
            "notional": float(row[3]),
            "trade_time_ms": int(row[4]),
        }
        for row in rows
    ]


def running_sum(rows: list[dict[str, Any]], start_ms: int, end_ms: int) -> float:
    return sum(float(r["notional"]) for r in rows if int(start_ms) <= int(r["ts_ms"]) <= int(end_ms))


def feature_values(snapshot: AnchorSnapshot) -> list[FeatureValue]:
    t = int(snapshot.anchor_ts_ms)
    return [
        FeatureValue("running_notional", snapshot.running_notional, t, FeatureClass.RUNNING_CLUSTER),
        FeatureValue("running_liq_count", snapshot.running_liq_count, t, FeatureClass.RUNNING_CLUSTER),
        FeatureValue("running_rate", snapshot.running_rate, t, FeatureClass.RUNNING_CLUSTER),
        FeatureValue("running_accel", snapshot.running_accel, t, FeatureClass.RUNNING_CLUSTER),
        FeatureValue("running_single_liq_dominance", snapshot.running_single_liq_dominance, t, FeatureClass.RUNNING_CLUSTER),
        FeatureValue("elapsed_since_first", snapshot.elapsed_since_first_sec, t, FeatureClass.RUNNING_CLUSTER),
    ]


def reconstruct_anchors(
    liqs: list[dict[str, Any]],
    *,
    bucket_sec: int,
    min_gap_sec: int,
    thresholds: tuple[float, ...],
    accel_window_sec: int,
    seed_last_kept: dict[float, int] | None = None,
) -> list[AnchorSnapshot]:
    """seed_last_kept lets a caller resume min-gap suppression state across a
    bounded/incremental liquidation window instead of assuming -infinity
    (i.e. "no prior anchor ever"). Additive/optional: omitting it reproduces
    the original full-history behavior exactly for every existing caller.
    """
    bucket_ms = int(bucket_sec) * 1000
    by_bucket: dict[int, list[dict[str, Any]]] = {}
    for row in liqs:
        by_bucket.setdefault(int(row["ts_ms"]) // bucket_ms, []).append(row)

    if seed_last_kept is not None:
        last_kept_by_threshold = {float(x): int(seed_last_kept.get(float(x), -(10**18))) for x in thresholds}
    else:
        last_kept_by_threshold = {float(x): -(10**18) for x in thresholds}
    anchors: list[AnchorSnapshot] = []
    for bucket, rows in sorted(by_bucket.items()):
        rows = sorted(rows, key=lambda r: int(r["ts_ms"]))
        first_ts = int(rows[0]["ts_ms"])
        crossed: set[float] = set()
        total = 0.0
        max_single = 0.0
        for idx, row in enumerate(rows):
            t = int(row["ts_ms"])
            notional = float(row["notional"])
            total += notional
            max_single = max(max_single, notional)
            count = idx + 1
            elapsed = max(0.0, (t - first_ts) / 1000.0)
            rate = total / max(elapsed, 1.0)
            cur_start = t - int(accel_window_sec) * 1000
            prev_start = t - 2 * int(accel_window_sec) * 1000
            prev_end = t - int(accel_window_sec) * 1000
            cur_rate = running_sum(rows[: idx + 1], cur_start, t) / max(1.0, float(accel_window_sec))
            prev_rate = running_sum(rows[: idx + 1], prev_start, prev_end) / max(1.0, float(accel_window_sec))
            accel = cur_rate - prev_rate
            accel_bucket = "accelerating" if accel > 0.0 else "decelerating"
            dominance = max_single / total * 100.0 if total > 0 else 0.0
            for threshold in thresholds:
                threshold = float(threshold)
                if threshold in crossed or total < threshold:
                    continue
                crossed.add(threshold)
                if t - last_kept_by_threshold[threshold] < int(min_gap_sec) * 1000:
                    continue
                snap = AnchorSnapshot(
                    event_id=f"{bucket}:{int(threshold)}",
                    bucket=int(bucket),
                    threshold_usd=threshold,
                    anchor_ts_ms=t,
                    first_ts_ms=first_ts,
                    elapsed_since_first_sec=elapsed,
                    running_notional=total,
                    running_liq_count=count,
                    running_rate=rate,
                    running_accel=accel,
                    running_single_liq_dominance=dominance,
                    acceleration_bucket=accel_bucket,
                    max_single_notional=max_single,
                )
                assert_feature_set_available(feature_values(snap), t, context="knowable_anchor_reconstruct")
                anchors.append(snap)
                last_kept_by_threshold[threshold] = t
    return anchors


def signed_return_bps(direction: str, entry_px: float, exit_px: float) -> float:
    raw = (float(exit_px) - float(entry_px)) / float(entry_px) * 10_000.0
    return raw if direction.upper() == "LONG" else -raw


def evaluate_anchor(
    conn: sqlite3.Connection,
    marks: MarkIndex,
    snapshot: AnchorSnapshot,
    *,
    symbol: str,
    direction: str,
    horizon_sec: int,
    fee_bps_side: float,
    max_book_staleness_sec: int,
) -> dict[str, Any]:
    entry_book = book_at(conn, symbol, snapshot.anchor_ts_ms, max_book_staleness_sec)
    exit_target_ms = int(snapshot.anchor_ts_ms) + int(horizon_sec) * 1000
    exit_book = book_at(conn, symbol, exit_target_ms, max_book_staleness_sec)
    entry_mark = marks.at_or_after(snapshot.anchor_ts_ms)
    exit_mark = marks.at_or_after(exit_target_ms)

    base = {
        "event_id": snapshot.event_id,
        "bucket": snapshot.bucket,
        "threshold_usd": snapshot.threshold_usd,
        "threshold_label": f"{int(snapshot.threshold_usd / 1000)}K",
        "anchor_ts_ms": snapshot.anchor_ts_ms,
        "anchor_utc": iso_ms(snapshot.anchor_ts_ms),
        "first_ts_ms": snapshot.first_ts_ms,
        "elapsed_since_first_sec": snapshot.elapsed_since_first_sec,
        "running_notional": snapshot.running_notional,
        "running_liq_count": snapshot.running_liq_count,
        "running_rate": snapshot.running_rate,
        "running_accel": snapshot.running_accel,
        "acceleration_bucket": snapshot.acceleration_bucket,
        "running_single_liq_dominance": snapshot.running_single_liq_dominance,
        "direction": direction.upper(),
        "horizon_sec": int(horizon_sec),
    }
    if not entry_book:
        base.update({"status": "NO_ENTRY_FILL", "net_bps": None})
    elif not exit_book:
        base.update({"status": "NO_EXIT_FILL", "net_bps": None})
    else:
        if direction.upper() == "LONG":
            entry_px = entry_book.ask
            exit_px = exit_book.bid
        else:
            entry_px = entry_book.bid
            exit_px = exit_book.ask
        gross = signed_return_bps(direction, entry_px, exit_px)
        base.update(
            {
                "status": "FILLED",
                "entry_book_ts_ms": entry_book.ts_ms,
                "exit_book_ts_ms": exit_book.ts_ms,
                "entry_staleness_ms": entry_book.staleness_ms,
                "exit_staleness_ms": exit_book.staleness_ms,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "gross_bps": gross,
                "fee_bps": 2.0 * float(fee_bps_side),
                "net_bps": gross - 2.0 * float(fee_bps_side),
            }
        )

    if entry_mark and exit_mark:
        mark_gross = signed_return_bps(direction, float(entry_mark[1]), float(exit_mark[1]))
        base["counterfactual_mark_gross_bps"] = mark_gross
        base["counterfactual_mark_net_bps"] = mark_gross - 2.0 * float(fee_bps_side)
        base["counterfactual_mark_entry_ts_ms"] = int(entry_mark[0])
        base["counterfactual_mark_exit_ts_ms"] = int(exit_mark[0])
    else:
        base["counterfactual_mark_gross_bps"] = None
        base["counterfactual_mark_net_bps"] = None
    return base


def parse_float_tuple(text: str) -> tuple[float, ...]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("empty threshold list")
    return tuple(out)


def summarize(rows: list[dict[str, Any]], key: str = "net_bps") -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    if not vals:
        return {
            "n": 0,
            "mean_bps": None,
            "median_bps": None,
            "p25_bps": None,
            "p75_bps": None,
            "win_rate": None,
            "cum_bps": 0.0,
            "top3_winner_removed_cum_bps": 0.0,
        }
    return {
        "n": len(vals),
        "mean_bps": r1(mean(vals)),
        "median_bps": r1(pctile(vals, 0.5)),
        "p25_bps": r1(pctile(vals, 0.25)),
        "p75_bps": r1(pctile(vals, 0.75)),
        "win_rate": r3(sum(v > 0 for v in vals) / len(vals)),
        "cum_bps": r1(sum(vals)),
        "top3_winner_removed_cum_bps": r1(sum(sorted(vals, reverse=True)[3:]) if len(vals) > 3 else sum(vals)),
    }


def group_summary(rows: list[dict[str, Any]], *, thresholds: tuple[float, ...], min_n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for threshold in thresholds:
        for direction in DIRECTIONS:
            for horizon in HORIZONS_SEC:
                for partition in PARTITIONS:
                    subset = [
                        r
                        for r in rows
                        if float(r["threshold_usd"]) == float(threshold)
                        and r["direction"] == direction
                        and int(r["horizon_sec"]) == int(horizon)
                        and (partition == "all" or r["acceleration_bucket"] == partition)
                    ]
                    filled = [r for r in subset if r.get("status") == "FILLED"]
                    no_fill = [r for r in subset if r.get("status") != "FILLED"]
                    cf_all = summarize(subset, "counterfactual_mark_net_bps")
                    cf_no_fill = summarize(no_fill, "counterfactual_mark_net_bps")
                    st = summarize(filled)
                    st.update(
                        {
                            "threshold_usd": threshold,
                            "threshold_label": f"{int(threshold / 1000)}K",
                            "direction": direction,
                            "horizon_sec": horizon,
                            "accel_partition": partition,
                            "total_n": len(subset),
                            "filled_n": len(filled),
                            "no_fill_n": len(no_fill),
                            "no_fill_rate": r3(len(no_fill) / len(subset)) if subset else None,
                            "cf_all_n": cf_all["n"],
                            "cf_all_median_bps": cf_all["median_bps"],
                            "cf_all_mean_bps": cf_all["mean_bps"],
                            "cf_all_top3_winner_removed_cum_bps": cf_all["top3_winner_removed_cum_bps"],
                            "cf_no_fill_median_bps": cf_no_fill["median_bps"],
                            "cf_no_fill_mean_bps": cf_no_fill["mean_bps"],
                            "eligible_min_n": len(filled) >= int(min_n),
                        }
                    )
                    out.append(st)
    return out


def split_anchor_ids(anchors: list[AnchorSnapshot], holdout_frac: float) -> tuple[set[str], set[str], dict[str, Any]]:
    base_ids = sorted({str(a.bucket) for a in anchors}, key=lambda x: int(x))
    holdout_n = max(1, int(round(len(base_ids) * float(holdout_frac)))) if base_ids else 0
    holdout_ids = set(base_ids[-holdout_n:]) if holdout_n else set()
    calibration_ids = set(base_ids[:-holdout_n]) if holdout_n else set(base_ids)
    freeze_payload = {
        "split_method": "chronological_bucket_tail_holdout",
        "holdout_frac": float(holdout_frac),
        "calibration_bucket_n": len(calibration_ids),
        "holdout_bucket_n": len(holdout_ids),
        "holdout_bucket_ids": sorted(holdout_ids, key=lambda x: int(x)),
    }
    freeze_payload["holdout_bucket_ids_sha256"] = sha256_text("\n".join(freeze_payload["holdout_bucket_ids"]))
    return calibration_ids, holdout_ids, freeze_payload


def select_primary(calibration_summary: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    eligible = [
        row
        for row in calibration_summary
        if row.get("eligible_min_n")
        and (row.get("mean_bps") is not None and float(row["mean_bps"]) > 0.0)
        and (row.get("median_bps") is not None and float(row["median_bps"]) > 0.0)
        and float(row.get("top3_winner_removed_cum_bps") or 0.0) > 0.0
    ]
    ranked = sorted(
        eligible,
        key=lambda r: (
            float(r.get("median_bps") or -1e9),
            float(r.get("mean_bps") or -1e9),
            float(r.get("filled_n") or 0),
        ),
        reverse=True,
    )
    shortlist = ranked[:3]
    primary = shortlist[0] if shortlist else None
    return primary, shortlist


def config_id(row: dict[str, Any] | None) -> str | None:
    if not row:
        return None
    return (
        f"{row['direction']}_X{int(float(row['threshold_usd']))}_"
        f"{row['accel_partition']}_H{int(row['horizon_sec'])}"
    )


def matching_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if float(r["threshold_usd"]) == float(config["threshold_usd"])
        and r["direction"] == config["direction"]
        and int(r["horizon_sec"]) == int(config["horizon_sec"])
        and (config["accel_partition"] == "all" or r["acceleration_bucket"] == config["accel_partition"])
    ]


def verdict_for_holdout(summary: dict[str, Any], *, min_n: int) -> str:
    if int(summary.get("n") or 0) < int(min_n):
        return "BLOCKED"
    if (
        (summary.get("mean_bps") is not None and float(summary["mean_bps"]) > 0.0)
        and (summary.get("median_bps") is not None and float(summary["median_bps"]) > 0.0)
        and float(summary.get("top3_winner_removed_cum_bps") or 0.0) > 0.0
    ):
        return "PAPER_CANDIDATE"
    return "RESEARCH_ONLY"


def write_survival_report(payload: dict[str, Any], out_md: Path, out_json: Path) -> None:
    rows = sorted(
        payload["calibration_summary"],
        key=lambda r: (
            float(r["threshold_usd"]),
            r["direction"],
            int(r["horizon_sec"]),
            r["accel_partition"],
        ),
    )
    lines = [
        "# S34 Knowable-Anchor Survival Curve",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "RESEARCH_ONLY. Calibration split only; holdout is frozen separately and not used for selection beyond the deterministic rule encoded in the script.",
        "",
        "## Split Freeze",
        "",
        f"- source_db: `{payload['source_db']['path']}`",
        f"- source_db_size_bytes: `{payload['source_db'].get('size_bytes')}`",
        f"- holdout_bucket_ids_sha256: `{payload['split']['holdout_bucket_ids_sha256']}`",
        f"- calibration_buckets: `{payload['split']['calibration_bucket_n']}`",
        f"- holdout_buckets: `{payload['split']['holdout_bucket_n']}`",
        "",
        "## Selected Config",
        "",
        f"- primary_config_id: `{payload['primary_config_id'] or 'NONE'}`",
        f"- shortlist: `{', '.join(payload['shortlist_config_ids']) if payload['shortlist_config_ids'] else 'NONE'}`",
        "",
        "## Calibration Grid",
        "",
        "| X | Dir | H | Accel | Filled N | No-fill % | Median | Mean | WR | Top3W Removed | Mark CF N | Mark CF Med | Mark CF Mean | CF no-fill med |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        wr = "" if row["win_rate"] is None else f"{float(row['win_rate']) * 100:.1f}%"
        nf = "" if row["no_fill_rate"] is None else f"{float(row['no_fill_rate']) * 100:.1f}%"
        lines.append(
            f"| {row['threshold_label']} | {row['direction']} | {row['horizon_sec']} | {row['accel_partition']} "
            f"| {row['filled_n']} | {nf} | {row['median_bps']} | {row['mean_bps']} | {wr} "
            f"| {row['top3_winner_removed_cum_bps']} | {row['cf_all_n']} | {row['cf_all_median_bps']} "
            f"| {row['cf_all_mean_bps']} | {row['cf_no_fill_median_bps']} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- Anchor is first real-time-knowable crossing of running notional X inside the 300s S34 bucket.",
        "- `accelerating` means current trailing liquidation-rate window exceeds the prior trailing window at the anchor timestamp.",
        "- All entry features are passed through the Feature Availability Contract gate before evaluation.",
        "- A positive calibration row is not a live recommendation; only the frozen holdout pass can produce at most `PAPER_CANDIDATE`.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_holdout_report(payload: dict[str, Any], out_md: Path, out_json: Path) -> None:
    lines = [
        "# S34 Knowable-Anchor Holdout",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "RESEARCH_ONLY. This is the single holdout pass for the frozen knowable-anchor configuration selected on calibration.",
        "",
        "## Frozen Config",
        "",
        f"- primary_config_id: `{payload['primary_config_id'] or 'NONE'}`",
        f"- config_sha256: `{payload['primary_config_sha256'] or 'NONE'}`",
        f"- holdout_bucket_ids_sha256: `{payload['split']['holdout_bucket_ids_sha256']}`",
        "",
        "## Holdout Result",
        "",
        f"- verdict: `{payload['verdict']}`",
        f"- n: `{payload['holdout_summary']['n']}`",
        f"- median_net_bps: `{payload['holdout_summary']['median_bps']}`",
        f"- mean_net_bps: `{payload['holdout_summary']['mean_bps']}`",
        f"- win_rate: `{payload['holdout_summary']['win_rate']}`",
        f"- top3_winner_removed_cum_bps: `{payload['holdout_summary']['top3_winner_removed_cum_bps']}`",
        f"- no_fill_n: `{payload['holdout_no_fill_n']}`",
        f"- no_fill_rate: `{payload['holdout_no_fill_rate']}`",
        f"- no_fill_counterfactual_median_bps: `{payload['holdout_no_fill_counterfactual']['median_bps']}`",
        "",
        "## Shortlist Holdout Diagnostics",
        "",
        "| Config | Verdict | N | Median | Mean | WR | Top3W Removed | No-fill % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["shortlist_holdout"]:
        wr = "" if row["summary"]["win_rate"] is None else f"{float(row['summary']['win_rate']) * 100:.1f}%"
        nf = "" if row["no_fill_rate"] is None else f"{float(row['no_fill_rate']) * 100:.1f}%"
        lines.append(
            f"| {row['config_id']} | {row['verdict']} | {row['summary']['n']} | {row['summary']['median_bps']} "
            f"| {row['summary']['mean_bps']} | {wr} | {row['summary']['top3_winner_removed_cum_bps']} | {nf} |"
        )
    lines += [
        "",
        "## Verdict Taxonomy",
        "",
        "- `BLOCKED`: insufficient holdout N or no calibration-selected config.",
        "- `RESEARCH_ONLY`: H0 not rejected on the frozen holdout.",
        "- `PAPER_CANDIDATE`: holdout mean/median net positive and top-3-winner-removed cumulative positive. This authorizes only a fresh paper pre-registration from zero.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S34 knowable-anchor continuation/fade research protocol.")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--liq-side", default="BUY")
    p.add_argument("--thresholds-usd", default=",".join(str(int(x)) for x in THRESHOLDS_USD))
    p.add_argument("--max-single-dominance-pct", type=float, default=None)
    p.add_argument("--bucket-sec", type=int, default=300)
    p.add_argument("--min-gap-sec", type=int, default=900)
    p.add_argument("--accel-window-sec", type=int, default=30)
    p.add_argument("--holdout-frac", type=float, default=0.30)
    p.add_argument("--fee-bps-side", type=float, default=3.05)
    p.add_argument("--max-book-staleness-sec", type=int, default=5)
    p.add_argument("--min-calibration-n", type=int, default=20)
    p.add_argument("--min-holdout-n", type=int, default=10)
    p.add_argument("--out-prefix", default="")
    p.add_argument("--start-ms", type=int, default=None)
    p.add_argument("--end-ms", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_tuple(args.thresholds_usd)
    suffix = str(args.out_prefix or f"{args.symbol.upper()}_{args.liq_side.upper()}").replace("/", "_").replace("\\", "_")
    survival_md = OUT_DIR / f"S34_KNOWABLE_ANCHOR_SURVIVAL_CURVE_{suffix}.md"
    survival_json = OUT_DIR / f"S34_KNOWABLE_ANCHOR_SURVIVAL_CURVE_{suffix}.json"
    holdout_md = OUT_DIR / f"S34_KNOWABLE_ANCHOR_HOLDOUT_{suffix}.md"
    holdout_json = OUT_DIR / f"S34_KNOWABLE_ANCHOR_HOLDOUT_{suffix}.json"
    db_path = Path(str(args.db))
    db_uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        liqs = load_liquidations(conn, args.symbol.upper(), args.liq_side.upper(), args.start_ms, args.end_ms)
        marks = load_mark_index(conn, args.symbol.upper())
        anchors = reconstruct_anchors(
            liqs,
            bucket_sec=int(args.bucket_sec),
            min_gap_sec=int(args.min_gap_sec),
            thresholds=thresholds,
            accel_window_sec=int(args.accel_window_sec),
        )
        if args.max_single_dominance_pct is not None:
            anchors = [
                anchor
                for anchor in anchors
                if float(anchor.running_single_liq_dominance) <= float(args.max_single_dominance_pct)
            ]
        calibration_ids, holdout_ids, split = split_anchor_ids(anchors, float(args.holdout_frac))

        all_rows: list[dict[str, Any]] = []
        for snapshot in anchors:
            for direction in DIRECTIONS:
                for horizon in HORIZONS_SEC:
                    all_rows.append(
                        evaluate_anchor(
                            conn,
                            marks,
                            snapshot,
                            symbol=args.symbol.upper(),
                            direction=direction,
                            horizon_sec=horizon,
                            fee_bps_side=float(args.fee_bps_side),
                            max_book_staleness_sec=int(args.max_book_staleness_sec),
                        )
                    )
    finally:
        conn.close()

    calibration_rows = [r for r in all_rows if str(r["bucket"]) in calibration_ids]
    holdout_rows = [r for r in all_rows if str(r["bucket"]) in holdout_ids]
    calibration_summary = group_summary(calibration_rows, thresholds=thresholds, min_n=int(args.min_calibration_n))
    primary, shortlist = select_primary(calibration_summary)
    primary_id = config_id(primary)
    shortlist_ids = [config_id(row) for row in shortlist if config_id(row)]

    generated_at = datetime.now(timezone.utc).isoformat()
    common = {
        "generated_at_utc": generated_at,
        "research_only": True,
        "source_db": file_fingerprint(db_path),
        "scope": {
            "symbol": args.symbol.upper(),
            "liq_side": args.liq_side.upper(),
            "bucket_sec": int(args.bucket_sec),
            "min_gap_sec": int(args.min_gap_sec),
            "thresholds_usd": list(thresholds),
            "horizons_sec": list(HORIZONS_SEC),
            "directions": list(DIRECTIONS),
            "accel_window_sec": int(args.accel_window_sec),
            "fee_bps_side": float(args.fee_bps_side),
            "max_book_staleness_sec": int(args.max_book_staleness_sec),
            "max_single_dominance_pct": args.max_single_dominance_pct,
        },
        "split": split,
        "anchor_count": len(anchors),
        "evaluated_rows": len(all_rows),
    }
    survival_payload = {
        **common,
        "calibration_rows_n": len(calibration_rows),
        "calibration_summary": calibration_summary,
        "primary_config": primary,
        "primary_config_id": primary_id,
        "shortlist": shortlist,
        "shortlist_config_ids": shortlist_ids,
    }
    write_survival_report(survival_payload, survival_md, survival_json)

    if primary:
        primary_rows = matching_rows(holdout_rows, primary)
        primary_filled = [r for r in primary_rows if r.get("status") == "FILLED"]
        primary_no_fill = [r for r in primary_rows if r.get("status") != "FILLED"]
        holdout_summary = summarize(primary_filled)
        holdout_cf = summarize(primary_no_fill, "counterfactual_mark_net_bps")
        verdict = verdict_for_holdout(holdout_summary, min_n=int(args.min_holdout_n))
        config_hash = sha256_text(json.dumps(primary, sort_keys=True))
    else:
        primary_rows = []
        primary_no_fill = []
        holdout_summary = summarize([])
        holdout_cf = summarize([])
        verdict = "BLOCKED"
        config_hash = None

    shortlist_holdout = []
    for cfg in shortlist:
        rows = matching_rows(holdout_rows, cfg)
        filled = [r for r in rows if r.get("status") == "FILLED"]
        no_fill = [r for r in rows if r.get("status") != "FILLED"]
        st = summarize(filled)
        shortlist_holdout.append(
            {
                "config": cfg,
                "config_id": config_id(cfg),
                "config_sha256": sha256_text(json.dumps(cfg, sort_keys=True)),
                "summary": st,
                "no_fill_n": len(no_fill),
                "no_fill_rate": r3(len(no_fill) / len(rows)) if rows else None,
                "no_fill_counterfactual": summarize(no_fill, "counterfactual_mark_net_bps"),
                "verdict": verdict_for_holdout(st, min_n=int(args.min_holdout_n)),
            }
        )

    holdout_payload = {
        **common,
        "primary_config": primary,
        "primary_config_id": primary_id,
        "primary_config_sha256": config_hash,
        "holdout_rows_n": len(holdout_rows),
        "holdout_summary": holdout_summary,
        "holdout_no_fill_n": len(primary_no_fill),
        "holdout_no_fill_rate": r3(len(primary_no_fill) / len(primary_rows)) if primary_rows else None,
        "holdout_no_fill_counterfactual": holdout_cf,
        "shortlist_holdout": shortlist_holdout,
        "verdict": verdict,
        "note": "A PAPER_CANDIDATE verdict authorizes only fresh paper pre-registration from zero, not live capital.",
    }
    write_holdout_report(holdout_payload, holdout_md, holdout_json)

    print(
        json.dumps(
            {
                "anchors": len(anchors),
                "calibration_rows": len(calibration_rows),
                "holdout_rows": len(holdout_rows),
                "primary_config_id": primary_id,
                "holdout_verdict": verdict,
                "survival_md": str(survival_md),
                "holdout_md": str(holdout_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
