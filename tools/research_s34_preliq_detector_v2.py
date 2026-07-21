"""ETH pre-liquidation detector v2.

Research only. This extends the v1 top-of-book detector with:
  - ETH taker-flow pressure from agg_trades
  - recent mini-liquidation context
  - BTC/SOL cross-symbol returns and liquidation pressure

The goal is not to create a runner rule. It is to answer whether an
"imminent large ETH SELL liquidation" detector can be built from information
available shortly before the liquidation event.
"""

from __future__ import annotations

import json
import math
import random
import sqlite3
import statistics
import sys
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MICRO_DB = f"file:{(ROOT / 'data' / 'microstructure.db').as_posix()}?mode=ro"
FEATURE_DB = ROOT / "data" / "s34_feature_factory.db"
OUT_JSON = ROOT / "reports" / "research" / "s34" / "S34_PRELIQ_DETECTOR_V2.json"
OUT_MD = ROOT / "reports" / "research" / "s34" / "S34_PRELIQ_DETECTOR_V2.md"

RANDOM_SEED = 34035
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


def _book_at_or_before(con: sqlite3.Connection, symbol: str, ts_ms: int) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT ts_ms, bid_price, bid_qty, ask_price, ask_qty, mid_price,
               spread_pct, book_imbalance, bid_depth_usd
        FROM book_ticker INDEXED BY idx_bt_symbol_ts
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()


def _mark_at_or_before(con: sqlite3.Connection, symbol: str, ts_ms: int) -> sqlite3.Row | None:
    return con.execute(
        """
        SELECT ts_ms, mark_price
        FROM mark_prices INDEXED BY idx_mark_symbol_ts
        WHERE symbol=? AND ts_ms<=?
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol, int(ts_ms)),
    ).fetchone()


def _return_bps(con: sqlite3.Connection, symbol: str, ts_ms: int, lookback_sec: int) -> float | None:
    now = _mark_at_or_before(con, symbol, ts_ms)
    past = _mark_at_or_before(con, symbol, ts_ms - lookback_sec * 1000)
    if now is None or past is None:
        return None
    if abs(int(ts_ms) - int(now["ts_ms"])) > 3000:
        return None
    if abs((ts_ms - lookback_sec * 1000) - int(past["ts_ms"])) > 3000:
        return None
    prev = float(past["mark_price"])
    return (float(now["mark_price"]) - prev) / prev * 10_000.0 if prev else None


def _agg_stats(con: sqlite3.Connection, symbol: str, ts_ms: int, window_sec: int) -> dict[str, float]:
    row = con.execute(
        """
        SELECT
          COUNT(*) AS n,
          COALESCE(SUM(notional), 0.0) AS total_notional,
          COALESCE(SUM(CASE WHEN is_buyer_maker=1 THEN notional ELSE 0 END), 0.0) AS sell_taker_notional,
          COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END), 0.0) AS buy_taker_notional
        FROM agg_trades INDEXED BY idx_trade_symbol_ts
        WHERE symbol=? AND ts_ms>? AND ts_ms<=?
        """,
        (symbol, int(ts_ms - window_sec * 1000), int(ts_ms)),
    ).fetchone()
    total = float(row["total_notional"] or 0.0)
    sell = float(row["sell_taker_notional"] or 0.0)
    buy = float(row["buy_taker_notional"] or 0.0)
    return {
        f"{symbol[:3].lower()}_agg_count_{window_sec}s": float(row["n"] or 0),
        f"{symbol[:3].lower()}_agg_notional_{window_sec}s": total,
        f"{symbol[:3].lower()}_sell_taker_notional_{window_sec}s": sell,
        f"{symbol[:3].lower()}_buy_taker_notional_{window_sec}s": buy,
        f"{symbol[:3].lower()}_taker_imbalance_{window_sec}s": (sell - buy) / total if total > 0 else 0.0,
    }


def _liq_stats(con: sqlite3.Connection, symbol: str, ts_ms: int, window_sec: int) -> dict[str, float]:
    rows = con.execute(
        """
        SELECT side, COUNT(*) AS n, COALESCE(SUM(notional), 0.0) AS notional
        FROM liquidations INDEXED BY idx_liq_symbol_ts
        WHERE symbol=? AND ts_ms>? AND ts_ms<=?
        GROUP BY side
        """,
        (symbol, int(ts_ms - window_sec * 1000), int(ts_ms)),
    ).fetchall()
    prefix = symbol[:3].lower()
    out = {
        f"{prefix}_liq_sell_count_{window_sec}s": 0.0,
        f"{prefix}_liq_buy_count_{window_sec}s": 0.0,
        f"{prefix}_liq_sell_notional_{window_sec}s": 0.0,
        f"{prefix}_liq_buy_notional_{window_sec}s": 0.0,
    }
    for r in rows:
        side = str(r["side"]).lower()
        if side in ("sell", "buy"):
            out[f"{prefix}_liq_{side}_count_{window_sec}s"] = float(r["n"] or 0)
            out[f"{prefix}_liq_{side}_notional_{window_sec}s"] = float(r["notional"] or 0.0)
    total = out[f"{prefix}_liq_sell_notional_{window_sec}s"] + out[f"{prefix}_liq_buy_notional_{window_sec}s"]
    out[f"{prefix}_liq_imbalance_{window_sec}s"] = (
        (out[f"{prefix}_liq_sell_notional_{window_sec}s"] - out[f"{prefix}_liq_buy_notional_{window_sec}s"]) / total
        if total > 0
        else 0.0
    )
    return out


def _row_features(con: sqlite3.Connection, ts_ms: int) -> dict[str, Any] | None:
    now = _book_at_or_before(con, "ETHUSDT", ts_ms)
    if now is None or abs(int(ts_ms) - int(now["ts_ms"])) > 1000:
        return None

    lookbacks = [1, 3, 5, 10, 15, 30]
    past: dict[int, sqlite3.Row] = {}
    for sec in lookbacks:
        row = _book_at_or_before(con, "ETHUSDT", ts_ms - sec * 1000)
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

    for sec in [1, 3, 5, 10, 30]:
        features.update(_agg_stats(con, "ETHUSDT", ts_ms, sec))
    for sym in ["ETHUSDT", "BTCUSDT", "SOLUSDT"]:
        for sec in [30, 60, 120]:
            features.update(_liq_stats(con, sym, ts_ms, sec))
    for sym in ["BTCUSDT", "SOLUSDT"]:
        for sec in [5, 10, 30]:
            ret = _return_bps(con, sym, ts_ms, sec)
            if ret is None:
                return None
            features[f"{sym[:3].lower()}_ret_{sec}s_bps"] = ret
    return features


def _load_event_times(ff: sqlite3.Connection) -> list[int]:
    return [
        int(r[0])
        for r in ff.execute(
            "SELECT event_ts_ms FROM liq_event_features WHERE symbol='ETHUSDT' AND liq_side='SELL' ORDER BY event_ts_ms"
        ).fetchall()
    ]


def _load_positive_samples(micro: sqlite3.Connection, ff: sqlite3.Connection, threshold: float) -> list[dict[str, Any]]:
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
        feats = _row_features(micro, ts)
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
    micro: sqlite3.Connection,
    ff: sqlite3.Connection,
    target_n: int,
    min_ts: int,
    max_ts: int,
) -> list[dict[str, Any]]:
    rng = random.Random(RANDOM_SEED)
    event_times = _load_event_times(ff)
    event_times.sort()
    out: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(50_000, target_n * 1200)
    while len(out) < target_n and attempts < max_attempts:
        attempts += 1
        ts = rng.randrange(min_ts + 120_000, max_ts - 120_000)
        pos = bisect_left(event_times, ts)
        too_near = False
        for idx in (pos - 1, pos):
            if 0 <= idx < len(event_times) and abs(ts - event_times[idx]) <= EXCLUDE_NEAR_LIQ_SEC * 1000:
                too_near = True
                break
        if too_near:
            continue
        feats = _row_features(micro, ts)
        if feats is None:
            continue
        # Same family as v1 control: force meaningful down pressure so the
        # detector cannot win by merely observing that price is falling.
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


def _median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def _split_temporal(rows: list[dict[str, Any]], train_frac: float = 0.70) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    xs = sorted(rows, key=lambda r: int(r["ts_ms"]))
    cut = int(len(xs) * train_frac)
    return xs[:cut], xs[cut:]


def _candidate_features(row: dict[str, Any]) -> list[str]:
    skip = {
        "sample_id",
        "label",
        "threshold",
        "event_ts_ms",
        "cluster_notional",
        "ts_ms",
        "ts_utc",
        "mid",
    }
    return [k for k, v in row.items() if k not in skip and isinstance(v, (int, float)) and math.isfinite(float(v))]


def _select_features(train: list[dict[str, Any]], keys: list[str], top_n: int = 8) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for key in keys:
        auc = _auc(train, key)
        if auc is None:
            continue
        best_auc = max(auc, 1.0 - auc)
        selected.append(
            {
                "feature": key,
                "train_auc": auc,
                "train_best_auc": best_auc,
                "direction": 1 if auc >= 0.5 else -1,
                "edge": abs(auc - 0.5),
            }
        )
    selected.sort(key=lambda x: x["edge"], reverse=True)
    return selected[:top_n]


def _fit_scaler(train: list[dict[str, Any]], features: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for item in features:
        key = item["feature"]
        vals = [float(r[key]) for r in train if r.get(key) is not None and math.isfinite(float(r[key]))]
        med = _median(vals) or 0.0
        q25 = _quantile(vals, 0.25)
        q75 = _quantile(vals, 0.75)
        iqr = (q75 - q25) if q25 is not None and q75 is not None else 1.0
        out[key] = {"median": med, "scale": max(abs(iqr), 1e-9)}
    return out


def _score(row: dict[str, Any], features: list[dict[str, Any]], scaler: dict[str, dict[str, float]]) -> float:
    total = 0.0
    for item in features:
        key = item["feature"]
        val = float(row.get(key) or 0.0)
        z = (val - scaler[key]["median"]) / scaler[key]["scale"]
        z = max(-5.0, min(5.0, z))
        total += float(item["direction"]) * float(item["edge"]) * z
    return total


def _precision_table(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    pos_n = sum(1 for r in rows if int(r["label"]) == 1)
    base_rate = pos_n / len(rows) if rows else None
    out = []
    scores = [float(r[score_key]) for r in rows]
    for q in [0.50, 0.60, 0.70, 0.80, 0.90]:
        cutoff = _quantile(scores, q)
        if cutoff is None:
            continue
        kept = [r for r in rows if float(r[score_key]) >= cutoff]
        tp = sum(1 for r in kept if int(r["label"]) == 1)
        precision = tp / len(kept) if kept else None
        out.append(
            {
                "score_quantile": q,
                "cutoff": cutoff,
                "kept_n": len(kept),
                "precision": precision,
                "lift_vs_base": (precision - base_rate) if precision is not None and base_rate is not None else None,
                "recall": tp / pos_n if pos_n else None,
            }
        )
    return out


def _evaluate_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    train, test = _split_temporal(rows)
    keys = _candidate_features(rows[0])
    selected = _select_features(train, keys, top_n=8)
    scaler = _fit_scaler(train, selected)
    train_scored = [{**r, "detector_score": _score(r, selected, scaler)} for r in train]
    test_scored = [{**r, "detector_score": _score(r, selected, scaler)} for r in test]
    return {
        "threshold": threshold,
        "n": len(rows),
        "train_n": len(train),
        "test_n": len(test),
        "train_pos_n": sum(1 for r in train if int(r["label"]) == 1),
        "test_pos_n": sum(1 for r in test if int(r["label"]) == 1),
        "selected_features": selected,
        "train_score_auc": _auc(train_scored, "detector_score"),
        "test_score_auc": _auc(test_scored, "detector_score"),
        "train_precision": _precision_table(train_scored, "detector_score"),
        "test_precision": _precision_table(test_scored, "detector_score"),
    }


def main() -> None:
    if not FEATURE_DB.exists():
        raise SystemExit(f"missing {FEATURE_DB}")
    micro = sqlite3.connect(MICRO_DB, uri=True, timeout=60)
    micro.row_factory = sqlite3.Row
    micro.execute("PRAGMA query_only=1")
    ff = sqlite3.connect(FEATURE_DB)
    ff.row_factory = sqlite3.Row
    min_ts, max_ts = ff.execute(
        "SELECT MIN(event_ts_ms), MAX(event_ts_ms) FROM liq_event_features WHERE symbol='ETHUSDT'"
    ).fetchone()

    positives_by_threshold = {thr: _load_positive_samples(micro, ff, thr) for thr in THRESHOLDS}
    positives_500 = positives_by_threshold[500_000.0]
    neg = _load_negative_samples(micro, ff, max(len(positives_500) * CONTROL_MULTIPLE, 200), int(min_ts), int(max_ts))
    datasets = {thr: positives_by_threshold[thr] + neg for thr in THRESHOLDS}
    results = [_evaluate_threshold(datasets[thr], thr) for thr in THRESHOLDS]

    # Single-feature separation on the 500K dataset, useful for diagnostics.
    keys = _candidate_features(datasets[500_000.0][0])
    feature_stats = []
    for key in keys:
        auc500 = _auc(datasets[500_000.0], key)
        auc1m = _auc(datasets[1_000_000.0], key)
        if auc500 is None or auc1m is None:
            continue
        pos_vals = [float(r[key]) for r in positives_by_threshold[500_000.0] if r.get(key) is not None]
        neg_vals = [float(r[key]) for r in neg if r.get(key) is not None]
        feature_stats.append(
            {
                "feature": key,
                "auc_500k": auc500,
                "best_auc_500k": max(auc500, 1.0 - auc500),
                "auc_1m": auc1m,
                "best_auc_1m": max(auc1m, 1.0 - auc1m),
                "pos500_median": _median(pos_vals),
                "control_median": _median(neg_vals),
            }
        )
    feature_stats.sort(key=lambda x: max(float(x["best_auc_500k"]), float(x["best_auc_1m"])), reverse=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lead_sec": LEAD_SEC,
        "positive_counts": {str(int(k)): len(v) for k, v in positives_by_threshold.items()},
        "negative_count": len(neg),
        "control_definition": "mid_down_10s>=5bps and spread<=1bps, excluding +/-900s around ETH SELL liquidation clusters",
        "feature_stats_top25": feature_stats[:25],
        "threshold_results": results,
        "interpretation": (
            "A usable detector needs temporal-test score AUC > 0.65 and high-quantile precision "
            "materially above the base positive rate. This is detector research only."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Pre-Liq Detector V2",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "Research only. V2 adds ETH taker flow, recent mini-liquidations, and BTC/SOL context to the v1 book-state detector.",
        "",
        f"Positive counts: 500K={len(positives_by_threshold[500_000.0])}, 1M={len(positives_by_threshold[1_000_000.0])}. Controls={len(neg)}.",
        "",
        f"Control definition: `{payload['control_definition']}`.",
        "",
        "## Temporal Split Detector Results",
        "",
        "| Threshold | Train N | Test N | Test pos | Train AUC | Test AUC | Q80 precision | Q80 lift | Q90 precision | Q90 lift |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for res in results:
        q80 = next((r for r in res["test_precision"] if abs(r["score_quantile"] - 0.80) < 1e-9), {})
        q90 = next((r for r in res["test_precision"] if abs(r["score_quantile"] - 0.90) < 1e-9), {})
        lines.append(
            f"| {int(res['threshold']/1000)}K | {res['train_n']} | {res['test_n']} | {res['test_pos_n']} | "
            f"{_fmt(res['train_score_auc'], 3)} | {_fmt(res['test_score_auc'], 3)} | "
            f"{_pct(q80.get('precision'))} | {_pct(q80.get('lift_vs_base'))} | "
            f"{_pct(q90.get('precision'))} | {_pct(q90.get('lift_vs_base'))} |"
        )
    lines += [
        "",
        "## Selected Features Per Threshold",
        "",
    ]
    for res in results:
        lines += [
            f"### {int(res['threshold']/1000)}K",
            "",
            "| Feature | Train AUC | Direction |",
            "| --- | ---: | ---: |",
        ]
        for item in res["selected_features"]:
            direction = "higher" if item["direction"] == 1 else "lower"
            lines.append(f"| {item['feature']} | {_fmt(item['train_auc'], 3)} | {direction} |")
        lines.append("")
    lines += [
        "## Top Single-Feature Separators",
        "",
        "| Feature | Best AUC 500K | Best AUC 1M | Pos500 median | Control median |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in feature_stats[:20]:
        lines.append(
            f"| {item['feature']} | {_fmt(item['best_auc_500k'], 3)} | {_fmt(item['best_auc_1m'], 3)} | "
            f"{_fmt(item['pos500_median'], 3)} | {_fmt(item['control_median'], 3)} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- V1 showed top-of-book alone could not detect imminent large liquidation.",
        "- V2 tests whether taker flow, mini-liquidation context, and cross-symbol pressure add real temporal-test separation.",
        "- This report does not change runner rules, live execution, or config.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
