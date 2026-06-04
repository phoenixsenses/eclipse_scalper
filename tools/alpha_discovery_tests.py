"""Recurring existing-data alpha discovery tests.

This is deliberately targeted, not a generic indicator sweep. It tests sibling
lanes around the current forced-flow/S34 hypotheses, searches for anti-alpha
rejection filters, checks fold stability and fee survival, and summarizes
forward shadow telemetry if present.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable


def _mark_at(conn: sqlite3.Connection, symbol: str, ts_ms: int, *, before: bool) -> float | None:
    op = "<=" if before else ">="
    order = "DESC" if before else "ASC"
    row = conn.execute(
        f"SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms {op} ? ORDER BY ts_ms {order} LIMIT 1",
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _return_bps(conn: sqlite3.Connection, symbol: str, ts_ms: int, direction: str, horizon_sec: int) -> float | None:
    entry = _mark_at(conn, symbol, ts_ms, before=True)
    exit_px = _mark_at(conn, symbol, ts_ms + int(horizon_sec) * 1000, before=False)
    if entry is None or exit_px is None or entry <= 0:
        return None
    raw = (exit_px - entry) / entry * 1e4
    return -raw if str(direction).upper() == "SHORT" else raw


def _funding_at(conn: sqlite3.Connection, symbol: str, ts_ms: int) -> float | None:
    row = conn.execute(
        """
        SELECT funding_rate FROM mark_prices
        WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (symbol, ts_ms),
    ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _hour(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).hour


def _weekday(ts_ms: int) -> int:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).weekday()


def _session(ts_ms: int) -> str:
    hour = _hour(ts_ms)
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 14:
        return "europe"
    if 14 <= hour < 21:
        return "us"
    return "late_us"


def _stats(vals: list[float]) -> dict[str, Any]:
    if not vals:
        return {"n": 0, "wr": None, "mean_bps": None, "median_bps": None}
    return {
        "n": len(vals),
        "wr": 100.0 * sum(1 for v in vals if v > 0) / len(vals),
        "mean_bps": mean(vals),
        "median_bps": median(vals),
    }


def _folds(vals: list[float], fold_count: int) -> list[dict[str, Any]]:
    if not vals:
        return []
    fold_count = max(1, min(int(fold_count), len(vals)))
    rows = []
    for i in range(fold_count):
        lo = int(i * len(vals) / fold_count)
        hi = int((i + 1) * len(vals) / fold_count)
        sub = vals[lo:hi]
        rows.append({"fold": i + 1, **_stats(sub)})
    return rows


def _load_liq_events(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    side: str,
    threshold: float,
    direction: str,
    horizon_sec: int,
    max_events: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT ts_ms, notional FROM liquidations
        WHERE symbol=? AND side=? AND notional>=?
        ORDER BY ts_ms DESC
        LIMIT ?
        """,
        (symbol, side, float(threshold), int(max_events)),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for ts_ms, notional in reversed(rows):
        ts = int(ts_ms)
        ret = _return_bps(conn, symbol, ts, direction, horizon_sec)
        if ret is None:
            continue
        funding = _funding_at(conn, symbol, ts)
        out.append(
            {
                "ts_ms": ts,
                "symbol": symbol,
                "return_bps": ret,
                "notional": float(notional or 0.0),
                "hour": _hour(ts),
                "weekday": _weekday(ts),
                "session": _session(ts),
                "funding": funding,
                "funding_sign": "negative" if funding is not None and funding < 0 else "positive" if funding is not None and funding > 0 else "missing",
            }
        )
    return out


def _load_s34_events(conn: sqlite3.Connection, *, horizon_sec: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT signal_ts_ms, basis_at_entry, liq_composition, confidence_band,
               session_tag, fingerprint_class, entry_book_state
        FROM detector_signals
        WHERE symbol='ETHUSDT' AND signal_ts_ms IS NOT NULL AND entry_price IS NOT NULL
        ORDER BY signal_ts_ms ASC
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for ts_ms, basis, comp, conf, session_tag, fingerprint, book_state in rows:
        ts = int(ts_ms)
        ret = _return_bps(conn, "ETHUSDT", ts, "SHORT", horizon_sec)
        if ret is None:
            continue
        basis_value = float(basis) if basis is not None else None
        out.append(
            {
                "ts_ms": ts,
                "symbol": "ETHUSDT",
                "return_bps": ret,
                "hour": _hour(ts),
                "weekday": _weekday(ts),
                "session": _session(ts),
                "basis": basis_value,
                "basis_sign": "positive" if basis_value is not None and basis_value > 0 else "nonpositive",
                "liq_composition": str(comp or "unknown"),
                "confidence_band": str(conf or "unknown"),
                "session_tag": str(session_tag or "unknown"),
                "fingerprint_class": str(fingerprint or "unknown"),
                "entry_book_state": str(book_state or "unknown"),
            }
        )
    return out


def _score(
    *,
    candidate: str,
    family: str,
    kind: str,
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    fee_bps: list[float],
    min_n: int,
    min_wr: float,
    min_mean_bps: float,
    folds: int,
) -> dict[str, Any]:
    base_vals = [float(r["return_bps"]) for r in rows]
    filtered_rows = [r for r in rows if predicate(r)]
    vals = [float(r["return_bps"]) for r in filtered_rows]
    base = _stats(base_vals)
    filt = _stats(vals)
    fold_rows = _folds(vals, folds)
    fees: dict[str, Any] = {}
    for fee in fee_bps:
        net = [v - fee for v in vals]
        net_folds = _folds(net, folds)
        fees[str(float(fee))] = {
            **_stats(net),
            "folds_positive": sum(1 for f in net_folds if f["mean_bps"] is not None and float(f["mean_bps"]) > 0),
        }
    uplift = float(filt["mean_bps"] or 0.0) - float(base["mean_bps"] or 0.0)
    folds_positive_gross = sum(1 for f in fold_rows if f["mean_bps"] is not None and float(f["mean_bps"]) > 0)
    net8 = fees.get("8.0", {})
    net8_mean = net8.get("mean_bps")
    net8_folds = int(net8.get("folds_positive", 0) or 0)
    verdict = "REJECT"
    reasons: list[str] = []
    if int(filt["n"]) < int(min_n):
        reasons.append("too_few_events")
    if float(filt["mean_bps"] or 0.0) < float(min_mean_bps):
        reasons.append("mean_below_gate")
    if float(filt["wr"] or 0.0) < float(min_wr):
        reasons.append("wr_below_gate")
    if net8_mean is None or float(net8_mean) <= 0:
        reasons.append("fails_8bps_fee")
    if folds_positive_gross < max(1, folds - 1):
        reasons.append("unstable_gross_folds")
    if net8_folds < max(1, folds - 1):
        reasons.append("unstable_net8_folds")

    if not reasons:
        verdict = "PROMOTE_SHADOW"
    elif int(filt["n"]) >= int(min_n) and float(filt["mean_bps"] or 0.0) > 0 and float((fees.get("4.0") or {}).get("mean_bps") or 0.0) > 0:
        verdict = "WATCH_ONLY"
    if kind == "anti_alpha" and int(filt["n"]) >= int(min_n) and float(filt["mean_bps"] or 0.0) <= 0:
        verdict = "CONFIRMED_REJECTION"
        reasons = ["negative_or_zero_mean"]

    return {
        "candidate": candidate,
        "family": family,
        "kind": kind,
        "verdict": verdict,
        "reasons": reasons,
        "baseline": base,
        "filtered": filt,
        "kept_ratio": (int(filt["n"]) / int(base["n"])) if int(base["n"]) else 0.0,
        "uplift_bps": uplift,
        "folds_positive_gross": folds_positive_gross,
        "folds": fold_rows,
        "fees": fees,
    }


def _liq_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for symbol, thresholds in {
        "ETHUSDT": [100000, 250000, 500000, 1000000],
        "BTCUSDT": [100000, 250000, 500000, 1000000],
        "SOLUSDT": [25000, 50000, 100000],
    }.items():
        for side, direction in (("BUY", "SHORT"), ("SELL", "LONG")):
            for threshold in thresholds:
                specs.append({"symbol": symbol, "side": side, "direction": direction, "threshold": float(threshold)})
    return specs


def _candidate_scores(conn: sqlite3.Connection, args: argparse.Namespace) -> list[dict[str, Any]]:
    fee_bps = [float(x.strip()) for x in str(args.fee_rt_bps).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []

    for spec in _liq_specs():
        family = f"{spec['symbol']}_{spec['side']}{int(spec['threshold'])}_{spec['direction']}_900"
        events = _load_liq_events(
            conn,
            symbol=spec["symbol"],
            side=spec["side"],
            threshold=float(spec["threshold"]),
            direction=spec["direction"],
            horizon_sec=900,
            max_events=int(args.max_events),
        )
        if not events:
            continue
        rows.append(
            _score(
                candidate=f"{family}_BASELINE",
                family=family,
                kind="cross_asset_transfer",
                rows=events,
                predicate=lambda _r: True,
                fee_bps=fee_bps,
                min_n=args.min_n,
                min_wr=args.min_wr,
                min_mean_bps=args.min_mean_bps,
                folds=args.folds,
            )
        )
        for session in ("asia", "europe", "us", "late_us"):
            rows.append(
                _score(
                    candidate=f"{family}_SESSION_{session.upper()}",
                    family=family,
                    kind="sibling_lane",
                    rows=events,
                    predicate=lambda r, s=session: str(r.get("session")) == s,
                    fee_bps=fee_bps,
                    min_n=args.min_n,
                    min_wr=args.min_wr,
                    min_mean_bps=args.min_mean_bps,
                    folds=args.folds,
                )
            )
        for hour in range(24):
            rows.append(
                _score(
                    candidate=f"{family}_UTC{hour:02d}",
                    family=family,
                    kind="sibling_lane",
                    rows=events,
                    predicate=lambda r, h=hour: int(r.get("hour", -1)) == h,
                    fee_bps=fee_bps,
                    min_n=args.min_n,
                    min_wr=args.min_wr,
                    min_mean_bps=args.min_mean_bps,
                    folds=args.folds,
                )
            )
        for sign in ("negative", "positive"):
            rows.append(
                _score(
                    candidate=f"{family}_FUNDING_{sign.upper()}",
                    family=family,
                    kind="sibling_lane",
                    rows=events,
                    predicate=lambda r, s=sign: str(r.get("funding_sign")) == s,
                    fee_bps=fee_bps,
                    min_n=args.min_n,
                    min_wr=args.min_wr,
                    min_mean_bps=args.min_mean_bps,
                    folds=args.folds,
                )
            )

    s34 = _load_s34_events(conn, horizon_sec=900)
    if s34:
        family = "ETHUSDT_S34_SHORT_900"
        s34_tests: list[tuple[str, str, Callable[[dict[str, Any]], bool]]] = [
            ("BASELINE", "s34_quality", lambda _r: True),
            ("BASIS_POSITIVE", "s34_quality", lambda r: str(r.get("basis_sign")) == "positive"),
            ("BASIS_NONPOSITIVE", "anti_alpha", lambda r: str(r.get("basis_sign")) == "nonpositive"),
            ("SINGLE_LARGE", "s34_quality", lambda r: str(r.get("liq_composition")) == "single_large"),
            ("CLUSTERED", "anti_alpha", lambda r: str(r.get("liq_composition")) == "clustered"),
            ("SESSION_US", "s34_quality", lambda r: str(r.get("session")) == "us"),
            ("SESSION_NON_US", "anti_alpha", lambda r: str(r.get("session")) != "us"),
            ("CONFIDENCE_MEDIUM", "s34_quality", lambda r: str(r.get("confidence_band")) == "medium"),
        ]
        for name, kind, pred in s34_tests:
            rows.append(
                _score(
                    candidate=f"{family}_{name}",
                    family=family,
                    kind=kind,
                    rows=s34,
                    predicate=pred,
                    fee_bps=fee_bps,
                    min_n=args.min_n,
                    min_wr=args.min_wr,
                    min_mean_bps=args.min_mean_bps,
                    folds=args.folds,
                )
            )
    return rows


def _load_shadow_telemetry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "rows": 0, "families": []}
    by_family: dict[str, list[float]] = defaultdict(list)
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if str(event.get("event")) != "research.shadow_signal":
            continue
        data = event.get("data") or {}
        family = str(data.get("signal_family") or "")
        labels = data.get("forward_labels") or {}
        value = labels.get("return_bps_900s")
        rows += 1
        if family and value is not None:
            try:
                by_family[family].append(float(value))
            except Exception:
                pass
    return {
        "path": str(path),
        "rows": rows,
        "families": [{"family": fam, **_stats(vals)} for fam, vals in sorted(by_family.items())],
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        rows = _candidate_scores(conn, args)
    finally:
        conn.close()
    rows.sort(
        key=lambda r: (
            r["verdict"] == "PROMOTE_SHADOW",
            r["verdict"] == "WATCH_ONLY",
            float((r["fees"].get("8.0") or {}).get("mean_bps") or -1e9),
            float(r["filtered"].get("mean_bps") or -1e9),
            int(r["filtered"].get("n") or 0),
        ),
        reverse=True,
    )
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["verdict"])] += 1
    return {
        "inputs": vars(args),
        "runtime_sec": round(time.time() - started, 3),
        "candidate_count": len(rows),
        "verdict_counts": dict(sorted(counts.items())),
        "promoted": [r for r in rows if r["verdict"] == "PROMOTE_SHADOW"],
        "watch_only": [r for r in rows if r["verdict"] == "WATCH_ONLY"][:100],
        "confirmed_rejections": [r for r in rows if r["verdict"] == "CONFIRMED_REJECTION"],
        "top_rejected": [r for r in rows if r["verdict"] == "REJECT"][:100],
        "shadow_telemetry": _load_shadow_telemetry(Path(str(args.telemetry_path))),
    }


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.2f}"


def _table(rows: list[dict[str, Any]], limit: int = 30) -> list[str]:
    lines = [
        "| candidate | kind | n | WR | mean | net8 | folds8 | uplift | reasons |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows[:limit]:
        net8 = row["fees"].get("8.0") or {}
        lines.append(
            f"| {row['candidate']} | {row['kind']} | {row['filtered']['n']} | {_fmt(row['filtered']['wr'])}% | "
            f"{_fmt(row['filtered']['mean_bps'])} | {_fmt(net8.get('mean_bps'))} | {int(net8.get('folds_positive') or 0)}/5 | "
            f"{_fmt(row['uplift_bps'])} | {','.join(row.get('reasons') or [])} |"
        )
    return lines


def write_md(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Alpha Discovery Tests",
        "",
        f"- db: `{payload['inputs']['db']}`",
        f"- candidates_tested: `{payload['candidate_count']}`",
        f"- runtime_sec: `{payload['runtime_sec']}`",
        f"- verdict_counts: `{json.dumps(payload['verdict_counts'], sort_keys=True)}`",
        "",
        "## Promote Shadow",
        "",
    ]
    lines.extend(_table(payload["promoted"]))
    lines.extend(["", "## Watch Only", ""])
    lines.extend(_table(payload["watch_only"]))
    lines.extend(["", "## Confirmed Rejections", ""])
    lines.extend(_table(payload["confirmed_rejections"]))
    lines.extend(["", "## Shadow Telemetry", ""])
    shadow = payload["shadow_telemetry"]
    lines.append(f"- path: `{shadow['path']}`")
    lines.append(f"- rows: `{shadow['rows']}`")
    if shadow["families"]:
        lines.extend(["", "| family | n | WR | mean | median |", "|---|---:|---:|---:|---:|"])
        for row in shadow["families"]:
            lines.append(f"| {row['family']} | {row['n']} | {_fmt(row['wr'])}% | {_fmt(row['mean_bps'])} | {_fmt(row['median_bps'])} |")
    else:
        lines.append("- no labeled shadow outcomes yet")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Run recurring existing-data alpha discovery tests.")
    p.add_argument("--db", default="data/microstructure.db")
    p.add_argument("--telemetry-path", default="logs/telemetry.jsonl")
    p.add_argument("--max-events", type=int, default=300)
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--min-wr", type=float, default=60.0)
    p.add_argument("--min-mean-bps", type=float, default=8.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fee-rt-bps", default="2,4,8,10")
    p.add_argument("--out-md", default="reports/ALPHA_DISCOVERY_TESTS.md")
    p.add_argument("--out-json", default="reports/ALPHA_DISCOVERY_TESTS.json")
    args = p.parse_args()
    payload = build_payload(args)
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_md(payload, out_md)
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print(f"promoted={len(payload['promoted'])} watch={len(payload['watch_only'])} rejected={len(payload['confirmed_rejections'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
