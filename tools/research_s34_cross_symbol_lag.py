import datetime as dt
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DB = "file:data/microstructure.db?mode=ro"
OUT_JSON = Path("reports/research/s34/S34_CROSS_SYMBOL_LAG_BTC_TO_ETH_SOL.json")
OUT_MD = Path("reports/research/s34/S34_CROSS_SYMBOL_LAG_BTC_TO_ETH_SOL.md")

LEADER = "BTCUSDT"
FOLLOWERS = ["ETHUSDT", "SOLUSDT"]
SIDES = ["BUY", "SELL"]
CLUSTER_THRESHOLD = 200_000.0
BUCKET_SEC = 300
MIN_GAP_SEC = 900
DELAYS_SEC = [0, 60, 300, 900]
HORIZONS_SEC = [60, 300, 900]


def iso(ts_ms: int | None) -> str | None:
    if ts_ms is None:
        return None
    return dt.datetime.fromtimestamp(ts_ms / 1000, dt.timezone.utc).isoformat()


def mark_at(con: sqlite3.Connection, symbol: str, ts_ms: int, before: bool = False):
    op = "<=" if before else ">="
    order = "desc" if before else "asc"
    return con.execute(
        f"""
        select ts_ms, mark_price
        from mark_prices
        where symbol=? and ts_ms {op} ?
        order by ts_ms {order}
        limit 1
        """,
        (symbol, ts_ms),
    ).fetchone()


def ret_bps(con: sqlite3.Connection, symbol: str, start_ms: int, end_ms: int) -> float | None:
    start = mark_at(con, symbol, start_ms, before=False)
    end = mark_at(con, symbol, end_ms, before=False)
    if not start or not end or not start[1]:
        return None
    return (float(end[1]) - float(start[1])) / float(start[1]) * 10000.0


def load_leader_clusters(con: sqlite3.Connection, side: str) -> list[dict]:
    rows = con.execute(
        """
        select cast(ts_ms / ? as integer) as bucket,
               min(ts_ms) as first_ts_ms,
               max(ts_ms) as last_ts_ms,
               count(*) as liq_count,
               sum(notional) as cluster_notional,
               max(notional) as max_notional
        from liquidations
        where symbol=? and side=?
        group by bucket
        having sum(notional)>=?
        order by first_ts_ms asc
        """,
        (BUCKET_SEC * 1000, LEADER, side, CLUSTER_THRESHOLD),
    ).fetchall()
    events = []
    last_signal_ms = -10**18
    prev_kept = None
    for bucket, first_ts, last_ts, count, total, max_notional in rows:
        first_ts = int(first_ts)
        if first_ts - last_signal_ms < MIN_GAP_SEC * 1000:
            continue
        duration = max(1.0, (int(last_ts) - first_ts) / 1000.0)
        events.append(
            {
                "event_id": f"{LEADER}_{side}_{int(bucket)}",
                "leader": LEADER,
                "side": side,
                "bucket": int(bucket),
                "start_ts_ms": first_ts,
                "end_ts_ms": int(last_ts),
                "start_utc": iso(first_ts),
                "duration_sec": duration,
                "count": int(count or 0),
                "notional": float(total or 0.0),
                "max_notional": float(max_notional or 0.0),
                "inter_kept_gap_sec": None if prev_kept is None else (first_ts - prev_kept) / 1000.0,
            }
        )
        last_signal_ms = first_ts
        prev_kept = first_ts
    return events


def follower_liq_nearby(con: sqlite3.Connection, follower: str, side: str, ts_ms: int, window_sec: int = 300):
    row = con.execute(
        """
        select ts_ms, side, notional
        from liquidations
        where symbol=? and ts_ms>=? and ts_ms<=?
        order by abs(ts_ms-?) asc
        limit 1
        """,
        (follower, ts_ms - window_sec * 1000, ts_ms + window_sec * 1000, ts_ms),
    ).fetchone()
    if not row:
        return None
    return {"lag_sec": (int(row[0]) - ts_ms) / 1000.0, "side": row[1], "notional": float(row[2] or 0.0)}


def q(vals: list[float]) -> dict:
    vals = sorted(v for v in vals if v is not None and not math.isnan(v))
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "pos_rate": 0.0}
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "p25": vals[int((len(vals) - 1) * 0.25)],
        "p75": vals[int((len(vals) - 1) * 0.75)],
        "pos_rate": sum(1 for v in vals if v > 0) / len(vals),
    }


def same_direction_value(side: str, raw_bps: float) -> float:
    # In the existing S34 convention, BUY liq is evaluated as LONG continuation;
    # SELL liq is evaluated as SHORT continuation.
    return raw_bps if side == "BUY" else -raw_bps


def main() -> None:
    con = sqlite3.connect(DB, uri=True)
    payload = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {
            "leader": LEADER,
            "followers": FOLLOWERS,
            "sides": SIDES,
            "cluster_threshold": CLUSTER_THRESHOLD,
            "bucket_sec": BUCKET_SEC,
            "min_gap_sec": MIN_GAP_SEC,
            "delays_sec": DELAYS_SEC,
            "horizons_sec": HORIZONS_SEC,
        },
        "sides": {},
    }

    for side in SIDES:
        events = load_leader_clusters(con, side)
        side_payload = {
            "event_count": len(events),
            "first_event_utc": events[0]["start_utc"] if events else None,
            "last_event_utc": events[-1]["start_utc"] if events else None,
            "median_notional": statistics.median([e["notional"] for e in events]) if events else 0.0,
            "median_duration_sec": statistics.median([e["duration_sec"] for e in events]) if events else 0.0,
            "followers": {},
        }
        for follower in FOLLOWERS:
            follower_payload = {"matrix": {}, "nearby_liq": {}}
            for delay in DELAYS_SEC:
                for horizon in HORIZONS_SEC:
                    raw_vals = []
                    same_vals = []
                    for event in events:
                        start = int(event["start_ts_ms"]) + delay * 1000
                        end = start + horizon * 1000
                        raw = ret_bps(con, follower, start, end)
                        if raw is None:
                            continue
                        raw_vals.append(raw)
                        same_vals.append(same_direction_value(side, raw))
                    follower_payload["matrix"][f"delay{delay}_h{horizon}"] = {
                        "raw_return_bps": q(raw_vals),
                        "same_direction_bps": q(same_vals),
                    }
            for window in [60, 300, 900]:
                lags = []
                side_counts = Counter()
                n_near = 0
                for event in events:
                    near = follower_liq_nearby(con, follower, side, int(event["start_ts_ms"]), window)
                    if near:
                        n_near += 1
                        lags.append(float(near["lag_sec"]))
                        side_counts[near["side"]] += 1
                follower_payload["nearby_liq"][str(window)] = {
                    "nearby_count": n_near,
                    "event_count": len(events),
                    "lag_sec": q(lags),
                    "side_counts": dict(side_counts),
                    "before_count": sum(1 for lag in lags if lag < -5),
                    "same_pm5_count": sum(1 for lag in lags if -5 <= lag <= 5),
                    "after_count": sum(1 for lag in lags if lag > 5),
                }
            side_payload["followers"][follower] = follower_payload
        payload["sides"][side] = side_payload

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# S34 Cross-Symbol Lag - BTC Liquidation To ETH/SOL",
        "",
        f"Generated: {payload['generated_at_utc']}",
        "",
        "Scope: BTCUSDT BUY/SELL liquidation clusters >=200K, 300s bucket, 900s minimum gap.",
        "",
        "This is descriptive research only. `same_direction_bps` means BUY-liq -> follower long return, SELL-liq -> follower short return.",
        "",
        "## BTC Leader Cluster Coverage",
        "",
        "| BTC side | Events | First | Last | Median Notional | Median Duration sec |",
        "|---|---:|---|---|---:|---:|",
    ]
    for side, side_payload in payload["sides"].items():
        lines.append(
            f"| {side} | {side_payload['event_count']} | {side_payload['first_event_utc']} | {side_payload['last_event_utc']} | "
            f"{side_payload['median_notional']:,.0f} | {side_payload['median_duration_sec']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Same-Direction Follower Return Matrix",
            "",
            "| BTC side | Follower | Delay | Horizon | N | Mean | Median | Positive Rate | p25/p75 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for side, side_payload in payload["sides"].items():
        for follower, follower_payload in side_payload["followers"].items():
            for delay in DELAYS_SEC:
                for horizon in HORIZONS_SEC:
                    row = follower_payload["matrix"][f"delay{delay}_h{horizon}"]["same_direction_bps"]
                    lines.append(
                        f"| {side} | {follower} | {delay}s | {horizon}s | {row['n']} | {row['mean']:+.2f} | "
                        f"{row['median']:+.2f} | {row['pos_rate']*100:.1f}% | {row['p25']:+.2f}/{row['p75']:+.2f} |"
                    )

    lines.extend(
        [
            "",
            "## Nearby Follower Liquidation Synchrony",
            "",
            "| BTC side | Follower | Window | Nearby | Before | Same +/-5s | After | Median Lag sec | Side Counts |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for side, side_payload in payload["sides"].items():
        for follower, follower_payload in side_payload["followers"].items():
            for window, row in follower_payload["nearby_liq"].items():
                lag = row["lag_sec"]
                lines.append(
                    f"| {side} | {follower} | {window}s | {row['nearby_count']}/{row['event_count']} | "
                    f"{row['before_count']} | {row['same_pm5_count']} | {row['after_count']} | "
                    f"{lag['median']:+.2f} | {row['side_counts']} |"
                )

    lines.extend(
        [
            "",
            "## Observations",
            "",
            "- The table separates raw delayed follower returns from same-direction forced-flow returns.",
            "- This report does not add a cross-symbol rule, choose a delay, or modify S34.",
            "- Apparent BTC->ETH/SOL timing relations are hypothesis material only.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT_MD)
    print(json.dumps(payload["sides"], indent=2)[:5000])
    con.close()


if __name__ == "__main__":
    main()
