#!/usr/bin/env python3
"""
Lightweight telemetry dashboard (JSONL).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from eclipse_scalper.tools.telemetry_codes_by_symbol import summarize_codes
except Exception:
    summarize_codes = None  # type: ignore[assignment]

GUARD_EVENTS = ("entry.partial_fill_escalation", "order.create.retry_alert")

def _load_jsonl(path: Path, limit: int = 5000):
    if not path.exists():
        return []
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
                if len(out) >= limit:
                    break
    except Exception:
        return out
    return out


def _count(events, key: str):
    out = {}
    for ev in events:
        k = ev.get(key)
        if k is None:
            continue
        out[k] = out.get(k, 0) + 1
    return out


def _guard_history_summary(path: Path, limit: int = 4) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    except Exception:
        return []
    if not rows:
        return []
    return rows[-max(1, limit) :]


def _guard_event_counts(events: list[dict[str, object]]) -> tuple[Counter[str], Counter[str]]:
    counts = Counter()
    reasons = Counter()
    for ev in events:
        name = str(ev.get("event") or "").strip()
        if name not in GUARD_EVENTS:
            continue
        counts[name] += 1
        if name == "order.create.retry_alert":
            reason = str(ev.get("data", {}).get("reason") or "unknown").lower()
            reasons[reason] += 1
    return counts, reasons


def _position_closed_summary(events: list[dict[str, object]]) -> dict:
    out = {
        "total": 0,
        "wins": 0,
        "avg_pnl": 0.0,
        "avg_pnl_pct": 0.0,
        "avg_duration": 0.0,
        "by_symbol": Counter(),
        "by_reason": Counter(),
    }
    pnls = []
    pnl_pcts = []
    durations = []
    for ev in events:
        if str(ev.get("event")) != "position.closed":
            continue
        data = ev.get("data") or {}
        symbol = str(ev.get("symbol") or data.get("symbol") or "unknown").upper()
        out["by_symbol"][symbol] += 1
        reason = str(data.get("exit_reason") or data.get("reason") or "unknown")
        out["by_reason"][reason] += 1
        pnl = _safe_float(data.get("pnl_usdt"), 0.0)
        pnl_pct = _safe_float(data.get("pnl_pct"), 0.0)
        duration = _safe_float(data.get("duration_sec"), 0.0)
        pnls.append(pnl)
        pnl_pcts.append(pnl_pct)
        durations.append(duration)
        if pnl > 0:
            out["wins"] += 1
        out["total"] += 1
    if pnls:
        out["avg_pnl"] = sum(pnls) / max(1, len(pnls))
    if pnl_pcts:
        out["avg_pnl_pct"] = sum(pnl_pcts) / max(1, len(pnl_pcts))
    if durations:
        out["avg_duration"] = sum(durations) / max(1, len(durations))
    return out


def _protection_refresh_budget_summary(events: list[dict[str, object]]) -> dict[str, int]:
    keys = (
        "protection_refresh_budget_blocked_count",
        "protection_refresh_budget_force_override_count",
        "protection_refresh_stop_budget_blocked_count",
        "protection_refresh_tp_budget_blocked_count",
        "protection_refresh_stop_force_override_count",
        "protection_refresh_tp_force_override_count",
    )
    out: dict[str, int] = {k: 0 for k in keys}
    for ev in events:
        data = ev.get("data")
        if not isinstance(data, dict):
            continue
        if str(ev.get("event") or "") != "reconcile.summary":
            continue
        for k in keys:
            out[k] = max(int(out.get(k, 0)), int(_safe_float(data.get(k), 0.0)))
    return out


def _correlation_contribution_summary(events: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {
        "blocked": 0,
        "scaled": 0,
        "stress": 0,
        "tightening": 0,
        "top_reasons": [],
        "top_symbols": [],
        "latest_regime": "",
        "latest_pressure": 0.0,
        "latest_reason_tags": "",
    }
    reason_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    latest_state: dict[str, object] = {}
    for ev in events:
        name = str(ev.get("event") or "")
        data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
        if name in ("entry.decision", "entry.blocked"):
            corr_pressure = _safe_float(data.get("corr_pressure"), 0.0)
            corr_regime = str(data.get("corr_regime") or "").strip().upper()
            if corr_pressure <= 0.0 and not corr_regime:
                continue
            if name == "entry.blocked":
                out["blocked"] = int(out.get("blocked", 0) or 0) + 1
            else:
                action = str(data.get("action") or "").strip().upper()
                if action == "SCALE":
                    out["scaled"] = int(out.get("scaled", 0) or 0) + 1
                elif action in ("DENY", "DEFER"):
                    out["blocked"] = int(out.get("blocked", 0) or 0) + 1
            if corr_regime == "STRESS":
                out["stress"] = int(out.get("stress", 0) or 0) + 1
            elif corr_regime == "TIGHTENING":
                out["tightening"] = int(out.get("tightening", 0) or 0) + 1
            reason = str(data.get("corr_reason_tags") or "stable").strip().lower()
            reason_counts[reason or "stable"] += 1
            symbol = str(ev.get("symbol") or data.get("symbol") or "").strip().upper()
            if symbol:
                symbol_counts[symbol] += 1
        elif name == "execution.correlation_state":
            latest_state = data
    if latest_state:
        out["latest_regime"] = str(latest_state.get("corr_regime") or "")
        out["latest_pressure"] = float(_safe_float(latest_state.get("corr_pressure"), 0.0))
        out["latest_reason_tags"] = str(latest_state.get("corr_reason_tags") or "")
    out["top_reasons"] = reason_counts.most_common(3)
    out["top_symbols"] = symbol_counts.most_common(3)
    return out


def _safe_float(value, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _format_guard_history(rows: list[dict[str, str]]) -> list[str]:
    out = []
    for row in rows:
        ts = row.get("timestamp", "n/a")
        hits = row.get("partial_retry_hits", "0")
        guard_reason = row.get("override_reason") or row.get("recent_guard_hit") or row.get("signal_context") or "normal"
        out.append(f"- {ts} | partial hits {hits} | guard {guard_reason}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Telemetry dashboard")
    parser.add_argument("--path", default="logs/telemetry.jsonl", help="Path to telemetry.jsonl")
    parser.add_argument("--limit", type=int, default=100, help="Max events to load from JSONL")
    parser.add_argument("--codes-per-symbol", action="store_true", help="Display top codes per symbol")
    parser.add_argument("--codes-top", type=int, default=3, help="Top codes per symbol when --codes-per-symbol")
    parser.add_argument("--guard-events", action="store_true", help="Summarize guard-related telemetry events")
    parser.add_argument(
        "--guard-history", action="store_true", help="Show the latest guard history rows (CSV snapshot)"
    )
    parser.add_argument("--guard-history-path", default="logs/telemetry_guard_history.csv")
    args = parser.parse_args()

    path = Path(args.path)
    events = _load_jsonl(path, limit=max(1, int(args.limit)))
    print("Telemetry Dashboard (Snapshot)")
    print(f"- File events loaded: {len(events)} from {path}")

    if not events:
        return 0

    by_event = _count(events, "event")
    by_code = _count([ev.get("data", {}) for ev in events], "code")

    print("- Top events:")
    for k, v in sorted(by_event.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {k}: {v}")

    if by_code:
        print("- Top codes:")
        for k, v in sorted(by_code.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  {k}: {v}")

    if args.guard_events:
        counts, reasons = _guard_event_counts(events)
        print("- Guard telemetry events:")
        if counts:
            for name in GUARD_EVENTS:
                print(f"  {name}: {counts.get(name, 0)}")
            if reasons:
                print("  retry reasons:")
                for reason, c in reasons.most_common():
                    print(f"    {reason}: {c}")
        else:
            print("  none detected")

    if args.guard_history:
        history = _guard_history_summary(Path(args.guard_history_path))
        print("- Guard history recent:")
        if history:
            for line in _format_guard_history(history[-3:]):
                print(f"  {line}")
        else:
            print("  missing artifact")

    exit_events = [ev for ev in events if str(ev.get("event", "")).startswith("exit.")]
    if exit_events:
        print("- Exit events summary:")
        exit_by_event = _count(exit_events, "event")
        exit_by_code = _count([ev.get("data", {}) for ev in exit_events], "code")
        if exit_by_event:
            for k, v in sorted(exit_by_event.items(), key=lambda kv: kv[1], reverse=True):
                print(f"  {k}: {v}")
        if exit_by_code:
            print("  exit codes:")
            for k, v in sorted(exit_by_code.items(), key=lambda kv: kv[1], reverse=True):
                print(f"    {k}: {v}")

    pos_summary = _position_closed_summary(events)
    if pos_summary.get("total"):
        total = int(pos_summary["total"])
        wins = int(pos_summary["wins"])
        win_rate = (wins / total) if total else 0.0
        print("- Position closed summary:")
        print(f"  total: {total} | win rate: {win_rate:.0%}")
        print(
            f"  avg pnl: {pos_summary['avg_pnl']:+.2f} | "
            f"avg pnl%: {pos_summary['avg_pnl_pct']:+.2%} | "
            f"avg duration: {pos_summary['avg_duration']:.0f}s"
        )
        if pos_summary["by_symbol"]:
            print("  top symbols:")
            for sym, count in pos_summary["by_symbol"].most_common(5):
                print(f"    {sym}: {count}")
        if pos_summary["by_reason"]:
            print("  top exit reasons:")
            for reason, count in pos_summary["by_reason"].most_common(5):
                print(f"    {reason}: {count}")

    refresh_budget = _protection_refresh_budget_summary(events)
    if any(int(v) > 0 for v in refresh_budget.values()):
        print("- Protection refresh budget:")
        print(
            "  blocked total: "
            f"{int(refresh_budget.get('protection_refresh_budget_blocked_count', 0))} "
            f"(stop={int(refresh_budget.get('protection_refresh_stop_budget_blocked_count', 0))}, "
            f"tp={int(refresh_budget.get('protection_refresh_tp_budget_blocked_count', 0))})"
        )
        print(
            "  force override total: "
            f"{int(refresh_budget.get('protection_refresh_budget_force_override_count', 0))} "
            f"(stop={int(refresh_budget.get('protection_refresh_stop_force_override_count', 0))}, "
            f"tp={int(refresh_budget.get('protection_refresh_tp_force_override_count', 0))})"
        )

    corr_summary = _correlation_contribution_summary(events)
    corr_hits = int(corr_summary.get("blocked", 0) or 0) + int(corr_summary.get("scaled", 0) or 0)
    if corr_hits > 0 or str(corr_summary.get("latest_regime") or ""):
        print("- Correlation contribution:")
        print(
            "  entry impact: "
            f"blocked={int(corr_summary.get('blocked', 0) or 0)} "
            f"scaled={int(corr_summary.get('scaled', 0) or 0)} "
            f"stress={int(corr_summary.get('stress', 0) or 0)} "
            f"tightening={int(corr_summary.get('tightening', 0) or 0)}"
        )
        latest_regime = str(corr_summary.get("latest_regime") or "")
        if latest_regime:
            print(
                "  latest state: "
                f"regime={latest_regime} "
                f"pressure={float(corr_summary.get('latest_pressure', 0.0) or 0.0):.2f} "
                f"tags={str(corr_summary.get('latest_reason_tags') or '')}"
            )
        top_reasons = list(corr_summary.get("top_reasons") or [])
        if top_reasons:
            print("  top reason tags:")
            for reason, count in top_reasons:
                print(f"    {reason}: {count}")
        top_symbols = list(corr_summary.get("top_symbols") or [])
        if top_symbols:
            print("  top symbols:")
            for symbol, count in top_symbols:
                print(f"    {symbol}: {count}")

    recent = events[-20:]
    print("- Recent (last 20):")
    for ev in recent:
        code = (ev.get("data", {}) or {}).get("code")
        print(f"  {ev.get('ts')} {ev.get('event')} {ev.get('symbol')} {code or ''}")

    if args.codes_per_symbol:
        if callable(summarize_codes):
            per_symbol = summarize_codes(events, top=args.codes_top)
            if per_symbol:
                print("- Top codes per symbol:")
                for sym, codes in sorted(per_symbol.items()):
                    print(f"  {sym}: " + ", ".join([f"{k}={v}" for k, v in codes]))
            else:
                print("- Top codes per symbol: none")
        else:
            print("- Top codes per symbol: helper unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
