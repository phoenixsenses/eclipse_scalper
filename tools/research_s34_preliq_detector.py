"""ETH pre-liquidation detector dataset and simple score research.

Research only. Builds a labeled dataset:
  positive = book state shortly before a large ETH SELL liquidation cluster
  negative = matched ETH book-pressure states far from SELL liquidation clusters

No runner/config/live state changes.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.storage import production as PR
from ami.storage import research_reader as RR

MICRO_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
SOURCE_DB_PATH = (ROOT / "data" / "microstructure.db").as_posix()
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_PRELIQ_DETECTOR.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_PRELIQ_DETECTOR.md"

RANDOM_SEED = 34034
LEAD_SEC = 5
EXCLUDE_NEAR_LIQ_SEC = 900
CONTROL_MULTIPLE = 5
THRESHOLDS = [500_000.0, 1_000_000.0]


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{float(v):.{digits}f}"


def _pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100:.1f}%"


_BOOK_COLS = ("ts_ms", "bid_price", "bid_qty", "ask_price", "ask_qty", "mid_price",
              "spread_pct", "book_imbalance", "bid_depth_usd")


def _book_at_or_before(con: sqlite3.Connection, ts_ms: int) -> sqlite3.Row | None:
    """Direct-SQL oracle -- kept unchanged as the parity reference for
    `_book_at_or_before_v2` (BATCH-STORAGE-ROTATION-RETENTION-ASOF-LOOKUP-
    CONSUMER-MIGRATION-V6). No longer called by main(); the reader-backed
    path is used instead."""
    return con.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price,
               spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker INDEXED BY idx_bt_symbol_ts
        WHERE symbol='ETHUSDT' AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (int(ts_ms),),
    ).fetchone()


def _book_at_or_before_v2(root, ts_ms: int, source_db_path=None) -> dict[str, float] | None:
    """Reader-backed replacement for `_book_at_or_before`, via
    lookup_latest_at_or_before. Symbol is hardcoded ETHUSDT, exactly as in
    the oracle's SQL (this file never varies it). book_ticker has no
    ETHUSDT archive partition (only SOLUSDT/2026-04 is archived), so real
    production use of this file resolves SQLITE_ONLY -- confirmed, not
    assumed. Returns a dict keyed identically to the oracle Row's selected
    columns, so downstream `row["col"]` access is unchanged (including the
    nullable `bid_depth_usd`, which stays None when absent)."""
    result = RR.lookup_latest_at_or_before(
        root, table="book_ticker", symbol="ETHUSDT", ts_ms=int(ts_ms),
        columns=_BOOK_COLS, source_db_path=source_db_path)
    if not result.found:
        return None
    return dict(zip(_BOOK_COLS, result.row))


def _row_features(root, ts_ms: int, source_db_path=None) -> dict[str, Any] | None:
    now = _book_at_or_before_v2(root, ts_ms, source_db_path=source_db_path)
    if now is None or abs(int(ts_ms) - int(now["ts_ms"])) > 1000:
        return None

    lookbacks = [1, 3, 5, 10, 15, 30]
    past: dict[int, dict[str, float]] = {}
    for sec in lookbacks:
        row = _book_at_or_before_v2(root, ts_ms - sec * 1000, source_db_path=source_db_path)
        if row is None or abs((ts_ms - sec * 1000) - int(row["ts_ms"])) > 1000:
            return None
        past[sec] = row

    mid = float(now["mid_price"])
    bid_qty = float(now["bid_qty"])
    ask_qty = float(now["ask_qty"])
    features: dict[str, Any] = {
        "ts_ms": int(now["ts_ms"]),
        "ts_utc": _iso(int(now["ts_ms"])),
        "mid": mid,
        "spread_bps": float(now["spread_pct"]) * 10_000.0,
        "book_imbalance": float(now["book_imbalance"]),
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        "bid_depth_usd": float(now["bid_depth_usd"] or 0.0),
        "top_qty_usd": (bid_qty + ask_qty) * mid,
    }
    for sec, row in past.items():
        prev_mid = float(row["mid_price"])
        prev_bid_qty = float(row["bid_qty"])
        prev_ask_qty = float(row["ask_qty"])
        prev_imb = float(row["book_imbalance"])
        features[f"mid_down_{sec}s_bps"] = (prev_mid - mid) / prev_mid * 10_000.0
        features[f"imb_delta_{sec}s"] = float(now["book_imbalance"]) - prev_imb
        features[f"bid_qty_delta_{sec}s_pct"] = (bid_qty - prev_bid_qty) / max(prev_bid_qty, 1e-9)
        features[f"ask_qty_delta_{sec}s_pct"] = (ask_qty - prev_ask_qty) / max(prev_ask_qty, 1e-9)
    return features


def _load_event_times(ff: sqlite3.Connection) -> list[int]:
    return [
        int(r[0])
        for r in ff.execute(
            "SELECT event_ts_ms FROM liq_event_features WHERE symbol='ETHUSDT' AND liq_side='SELL' ORDER BY event_ts_ms"
        ).fetchall()
    ]


def _load_positive_samples(root, ff: sqlite3.Connection, threshold: float, source_db_path=None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in ff.execute(
        """
        SELECT event_id, event_ts_ms, cluster_notional
        FROM liq_event_features
        WHERE symbol='ETHUSDT' AND liq_side='SELL' AND cluster_notional>=?
        ORDER BY event_ts_ms
        """,
        (float(threshold),),
    ).fetchall():
        ts = int(row["event_ts_ms"]) - LEAD_SEC * 1000
        feats = _row_features(root, ts, source_db_path=source_db_path)
        if feats is None:
            continue
        feats.update(
            {
                "sample_id": f"pos:{int(threshold)}:{row['event_id']}",
                "label": 1,
                "threshold": threshold,
                "event_ts_ms": int(row["event_ts_ms"]),
                "cluster_notional": float(row["cluster_notional"]),
            }
        )
        out.append(feats)
    return out


def _load_negative_samples(
    root,
    ff: sqlite3.Connection,
    target_n: int,
    min_ts: int,
    max_ts: int,
    source_db_path=None,
) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    event_times = _load_event_times(ff)
    out: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(50_000, target_n * 1000)
    while len(out) < target_n and attempts < max_attempts:
        attempts += 1
        ts = rng.randrange(min_ts + 60_000, max_ts - 60_000)
        if any(abs(ts - ev) <= EXCLUDE_NEAR_LIQ_SEC * 1000 for ev in event_times):
            continue
        feats = _row_features(root, ts, source_db_path=source_db_path)
        if feats is None:
            continue
        # Match the same broad book-pressure family, otherwise the classifier
        # only learns "price wasn't moving".
        if feats["mid_down_10s_bps"] < 5.0 or feats["spread_bps"] > 1.0:
            continue
        feats.update(
            {
                "sample_id": f"neg:{len(out)}:{ts}",
                "label": 0,
                "threshold": None,
                "event_ts_ms": None,
                "cluster_notional": None,
            }
        )
        out.append(feats)
    return out


def _auc(rows: list[dict[str, Any]], key: str) -> float | None:
    pos = [float(r[key]) for r in rows if int(r["label"]) == 1 and r.get(key) is not None]
    neg = [float(r[key]) for r in rows if int(r["label"]) == 0 and r.get(key) is not None]
    if not pos or not neg:
        return None
    wins = ties = total = 0
    for p in pos:
        for n in neg:
            total += 1
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / total if total else None


def _quantile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    xs = sorted(vals)
    idx = (len(xs) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return xs[int(idx)]
    return xs[lo] * (hi - idx) + xs[hi] * (idx - lo)


def _describe(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "mean": statistics.mean(vals),
        "p25": _quantile(vals, 0.25),
        "p75": _quantile(vals, 0.75),
    }


def _rule_score(row: dict[str, Any]) -> float:
    # Simple monotonic score, deliberately transparent. Higher means more
    # "pre-liq-like": stronger short-term down move, more ask-heavy book, tight spread.
    return (
        0.65 * float(row.get("mid_down_10s_bps") or 0.0)
        + 0.35 * float(row.get("mid_down_5s_bps") or 0.0)
        + 4.0 * max(0.0, -float(row.get("book_imbalance") or 0.0))
        - 2.0 * float(row.get("spread_bps") or 0.0)
    )


def _precision_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [{**r, "_score": _rule_score(r)} for r in rows]
    pos_n = sum(1 for r in scored if int(r["label"]) == 1)
    out = []
    for q in [0.50, 0.60, 0.70, 0.80, 0.90]:
        cutoff = _quantile([float(r["_score"]) for r in scored], q)
        if cutoff is None:
            continue
        kept = [r for r in scored if float(r["_score"]) >= cutoff]
        tp = sum(1 for r in kept if int(r["label"]) == 1)
        out.append(
            {
                "score_quantile": q,
                "cutoff": cutoff,
                "kept_n": len(kept),
                "precision": tp / len(kept) if kept else None,
                "recall": tp / pos_n if pos_n else None,
            }
        )
    return out


def main() -> None:
    if not FEATURE_DB.exists():
        raise SystemExit(f"missing {FEATURE_DB}")
    # Every book_ticker point read now goes through the reader (via
    # SOURCE_DB_PATH), so the old direct `micro` connection to the 778GB
    # source DB is fully dead here and is not opened (mirrors the ASOF
    # Batch 2 counter_regime dead-connection drop). No range/forward-scan
    # book_ticker query remains in this file.
    root, _ = PR.resolve_production_root()
    ff = sqlite3.connect(FEATURE_DB)
    ff.row_factory = sqlite3.Row
    min_ts, max_ts = ff.execute(
        "SELECT MIN(event_ts_ms), MAX(event_ts_ms) FROM liq_event_features WHERE symbol='ETHUSDT'"
    ).fetchone()

    positives_by_threshold = {thr: _load_positive_samples(root, ff, thr, source_db_path=SOURCE_DB_PATH) for thr in THRESHOLDS}
    positives_500 = positives_by_threshold[500_000.0]
    neg = _load_negative_samples(root, ff, max(len(positives_500) * CONTROL_MULTIPLE, 200), int(min_ts), int(max_ts), source_db_path=SOURCE_DB_PATH)
    rows_500 = positives_500 + neg
    rows_1m = positives_by_threshold[1_000_000.0] + neg

    feature_keys = [
        "mid_down_1s_bps",
        "mid_down_3s_bps",
        "mid_down_5s_bps",
        "mid_down_10s_bps",
        "mid_down_15s_bps",
        "mid_down_30s_bps",
        "book_imbalance",
        "spread_bps",
        "bid_qty_delta_10s_pct",
        "ask_qty_delta_10s_pct",
        "imb_delta_10s",
        "bid_depth_usd",
        "top_qty_usd",
    ]
    stats = []
    for key in feature_keys:
        stats.append(
            {
                "feature": key,
                "auc_500k": _auc(rows_500, key),
                "auc_1m": _auc(rows_1m, key),
                "pos_500k": _describe(positives_by_threshold[500_000.0], key),
                "pos_1m": _describe(positives_by_threshold[1_000_000.0], key),
                "control": _describe(neg, key),
            }
        )

    score_rows_500 = [{**r, "detector_score": _rule_score(r)} for r in rows_500]
    score_rows_1m = [{**r, "detector_score": _rule_score(r)} for r in rows_1m]
    score_stats = {
        "auc_500k": _auc(score_rows_500, "detector_score"),
        "auc_1m": _auc(score_rows_1m, "detector_score"),
        "precision_500k": _precision_table(rows_500),
        "precision_1m": _precision_table(rows_1m),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lead_sec": LEAD_SEC,
        "positive_counts": {str(int(k)): len(v) for k, v in positives_by_threshold.items()},
        "negative_count": len(neg),
        "feature_stats": stats,
        "score_stats": score_stats,
        "note": "Controls are matched to broad mid_down_10s>=5bps and spread<=1bps, excluding windows near ETH SELL liquidation clusters.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Pre-Liq Detector Research",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "Research only. Builds a labeled detector dataset for ETH SELL pre-liquidation book pressure.",
        "",
        f"Positive counts: 500K={len(positives_by_threshold[500_000.0])}, 1M={len(positives_by_threshold[1_000_000.0])}. Controls={len(neg)}.",
        "",
        "Controls are matched to broad `mid_down_10s >= 5 bps` and `spread <= 1 bps`, excluding +/-900s around ETH SELL liquidation clusters.",
        "",
        "## Feature Separation",
        "",
        "| Feature | AUC 500K | AUC 1M | Pos500 med | Pos1M med | Control med |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(stats, key=lambda x: max(x["auc_500k"] or 0, x["auc_1m"] or 0), reverse=True):
        lines.append(
            f"| {item['feature']} | {_fmt(item['auc_500k'], 3)} | {_fmt(item['auc_1m'], 3)} | "
            f"{_fmt(item['pos_500k'].get('median'), 3)} | {_fmt(item['pos_1m'].get('median'), 3)} | {_fmt(item['control'].get('median'), 3)} |"
        )
    lines += [
        "",
        "## Transparent Detector Score",
        "",
        "`score = 0.65*mid_down_10s + 0.35*mid_down_5s + 4*max(0,-book_imbalance) - 2*spread_bps`",
        "",
        f"- Score AUC 500K: `{_fmt(score_stats['auc_500k'], 3)}`",
        f"- Score AUC 1M: `{_fmt(score_stats['auc_1m'], 3)}`",
        "",
        "### Precision / Recall 500K",
        "",
        "| Score quantile | Cutoff | Kept | Precision | Recall |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in score_stats["precision_500k"]:
        lines.append(f"| {row['score_quantile']:.2f} | {_fmt(row['cutoff'], 3)} | {row['kept_n']} | {_pct(row['precision'])} | {_pct(row['recall'])} |")
    lines += [
        "",
        "### Precision / Recall 1M",
        "",
        "| Score quantile | Cutoff | Kept | Precision | Recall |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in score_stats["precision_1m"]:
        lines.append(f"| {row['score_quantile']:.2f} | {_fmt(row['cutoff'], 3)} | {row['kept_n']} | {_pct(row['precision'])} | {_pct(row['recall'])} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- AUC near 0.5 means the feature cannot separate pre-liq positives from matched controls.",
        "- Useful detector candidates should show AUC materially above 0.65 and precision lift at high score quantiles.",
        "- This is not a trading rule yet; it is a detector research layer.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
