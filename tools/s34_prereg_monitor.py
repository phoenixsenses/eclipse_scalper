"""
S34 Pre-Registration Monitor — multi-variant
Tracks calibration/holdout progress for all pre-registered and exploratory variants.

Usage:
    python tools/s34_prereg_monitor.py
    python tools/s34_prereg_monitor.py --watch
    python tools/s34_prereg_monitor.py --interval 30
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT     = Path(__file__).resolve().parents[1]
INTEL_DB = ROOT / "data" / "s34_intelligence.db"

# ── ANSI ─────────────────────────────────────────────────────────────────────
R    = "\033[91m"; G  = "\033[92m"; Y  = "\033[93m"
B    = "\033[94m"; M  = "\033[95m"; C  = "\033[96m"; W = "\033[97m"
DIM  = "\033[2m";  BOLD = "\033[1m"; RST = "\033[0m"

# ── Variant definitions ───────────────────────────────────────────────────────
@dataclass
class VariantCfg:
    name:         str
    label:        str
    cutoff:       str
    n_calib:      int
    n_holdout:    int
    kind:         str          # "MAIN" or "SELL_EXP"
    color:        str = C

VARIANTS: list[VariantCfg] = [
    VariantCfg(
        name    = "ETH_BUY_LIQ_LONG_500K_DAYTREND0_TP60_SL40_BE30",
        label   = "ETH 500K BUY→LONG",
        cutoff  = "2026-06-25T09:00:00+00:00",
        n_calib = 40, n_holdout = 60, kind = "MAIN", color = C,
    ),
    VariantCfg(
        name    = "SOL_BUY_LIQ_LONG_200K_TP60_SL40_BE30",
        label   = "SOL 200K BUY→LONG",
        cutoff  = "2026-06-25T09:00:00+00:00",
        n_calib = 40, n_holdout = 60, kind = "MAIN", color = G,
    ),
    VariantCfg(
        name    = "BTC_BUY_LIQ_LONG_1M_DISTRIBUTED_TP60_SL30_BE30",
        label   = "BTC 1M   BUY→LONG",
        cutoff  = "2026-06-25T09:00:00+00:00",
        n_calib = 40, n_holdout = 60, kind = "MAIN", color = Y,
    ),
    VariantCfg(
        name    = "SOL_BUY_LIQ_LONG_100K_TP60_SL40_BE30",
        label   = "SOL 100K BUY->LONG",
        cutoff  = "2026-06-26T00:00:00+00:00",
        n_calib = 40, n_holdout = 60, kind = "MAIN", color = G,
    ),
    VariantCfg(
        name    = "ETH_SELL_LIQ_SHORT_1M_TP80_SL40_BE40",
        label   = "ETH 1M   SELL→SHORT",
        cutoff  = "2026-06-26T00:00:00+00:00",
        n_calib = 30, n_holdout = 0, kind = "SELL_EXP", color = M,
    ),
    VariantCfg(
        name    = "ETH_SELL_LIQ_SHORT_500K_TP60_SL40_BE40",
        label   = "ETH 500K SELL→SHORT",
        cutoff  = "2026-06-26T00:00:00+00:00",
        n_calib = 30, n_holdout = 0, kind = "SELL_EXP", color = M,
    ),
    VariantCfg(
        name    = "SOL_SELL_LIQ_SHORT_200K_TP60_SL30_BE30",
        label   = "SOL 200K SELL→SHORT",
        cutoff  = "2026-06-26T00:00:00+00:00",
        n_calib = 30, n_holdout = 0, kind = "SELL_EXP", color = R,
    ),
    VariantCfg(
        name    = "SOL_SELL_LIQ_SHORT_100K_TP60_SL30_BE40",
        label   = "SOL 100K SELL->SHORT",
        cutoff  = "2026-06-26T00:00:00+00:00",
        n_calib = 30, n_holdout = 0, kind = "SELL_EXP", color = R,
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def bar(n: int, total: int, width: int = 28) -> str:
    filled = min(int(n / max(total, 1) * width), width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def fmt_bps(v: float | None, plus: bool = True) -> str:
    if v is None:
        return f"{DIM}—{RST}"
    sign = "+" if v >= 0 and plus else ""
    col  = G if v > 0 else (R if v < 0 else W)
    return f"{col}{sign}{v:.1f}{RST}"


def fmt_pct(v: float | None) -> str:
    if v is None:
        return f"{DIM}—{RST}"
    col = G if v >= 55 else (R if v < 40 else Y)
    return f"{col}{v:.0f}%{RST}"


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(INTEL_DB), timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


# ── Per-variant data load ─────────────────────────────────────────────────────
def load_trades(conn: sqlite3.Connection, cfg: VariantCfg) -> list[dict[str, Any]]:
    rows = conn.execute("""
        SELECT trade_id, net_bps, exit_reason, opened_at_utc,
               entry_ts_ms, exit_ts_ms, trade_json
        FROM s34_trades
        WHERE rule_name = ?
          AND status    = 'CLOSED'
          AND opened_at_utc >= ?
        ORDER BY entry_ts_ms ASC
    """, (cfg.name, cfg.cutoff)).fetchall()

    trades = []
    for r in rows:
        d: dict = {"trade_id": r["trade_id"], "net_bps": r["net_bps"],
                   "exit_reason": r["exit_reason"], "opened_at_utc": r["opened_at_utc"]}
        try:
            tj = json.loads(r["trade_json"])
            d["gross_bps"]         = tj.get("gross_bps")
            d["entry_adverse_bps"] = tj.get("entry_adverse_bps")
        except Exception:
            pass
        trades.append(d)
    return trades


def signal_rate(conn: sqlite3.Connection, cfg: VariantCfg) -> float:
    rows = conn.execute("""
        SELECT MIN(entry_ts_ms) as t0, MAX(entry_ts_ms) as t1, COUNT(*) as n
        FROM s34_trades
        WHERE rule_name = ? AND status = 'CLOSED' AND opened_at_utc >= ?
    """, (cfg.name, cfg.cutoff)).fetchone()
    n = rows["n"] or 0
    t0, t1 = rows["t0"], rows["t1"]
    if n >= 2 and t0 and t1:
        days = (t1 - t0) / 86_400_000
        return n / max(days, 1)
    now_ms = int(time.time() * 1000)
    two_weeks_ms = now_ms - 14 * 86_400_000
    acc = conn.execute("""
        SELECT COUNT(*) as n, MIN(signal_ts_ms) as t0, MAX(signal_ts_ms) as t1
        FROM s34_decisions WHERE decision = 'ACCEPT' AND signal_ts_ms >= ?
    """, (two_weeks_ms,)).fetchone()
    acc_n = acc["n"] or 0
    if acc_n < 2:
        return 0.0
    days = (acc["t1"] - acc["t0"]) / 86_400_000
    return acc_n / max(days, 1)


def quarantine_rate(conn: sqlite3.Connection, cfg: VariantCfg) -> float | None:
    acc_n = conn.execute("""
        SELECT COUNT(*) as n FROM s34_decisions
        WHERE decision = 'ACCEPT' AND signal_ts_ms >= (
            SELECT MIN(entry_ts_ms) FROM s34_trades
            WHERE rule_name=? AND status='CLOSED' AND opened_at_utc>=?
        )
    """, (cfg.name, cfg.cutoff)).fetchone()["n"]
    trade_n = conn.execute("""
        SELECT COUNT(*) as n FROM s34_trades
        WHERE rule_name=? AND status='CLOSED' AND opened_at_utc>=?
    """, (cfg.name, cfg.cutoff)).fetchone()["n"]
    if not acc_n:
        return None
    effective_acc = acc_n // 2
    if not effective_acc:
        return None
    return max(0, effective_acc - trade_n) / effective_acc


# ── Display ───────────────────────────────────────────────────────────────────
def kill_status(label: str, passed: bool | None) -> str:
    if passed is None:
        return f"  {DIM}{label}  PENDING{RST}"
    col  = G if passed else R
    verb = "PASS" if passed else "FAIL <- KILL"
    return f"  {col}{label}  {verb}{RST}"


def display_variant(cfg: VariantCfg, trades: list[dict], rate: float,
                    q_rate: float | None) -> None:
    col = cfg.color
    n_target = cfg.n_calib + cfg.n_holdout
    n        = len(trades)
    n_prog   = min(n, cfg.n_calib)   # calib portion (or all for SELL_EXP)

    kind_tag = f"{DIM}[MAIN PRE-REG N=100]{RST}" if cfg.kind == "MAIN" else \
               f"{DIM}[SELL EXPLORATORY N=30]{RST}"

    print(f"\n{BOLD}{col}  {cfg.label}{RST}  {kind_tag}")
    print(f"  {DIM}{cfg.name}{RST}")
    print(f"  {DIM}Clock: {cfg.cutoff[:10]}{RST}")

    # Progress bar
    if cfg.kind == "MAIN":
        prog_col = G if n_prog >= cfg.n_calib else Y
        hold_n   = max(0, n - cfg.n_calib)
        hold_col = G if hold_n >= cfg.n_holdout else (Y if hold_n > 0 else DIM)
        print(f"  Calib   {prog_col}{bar(n_prog, cfg.n_calib)}{RST}  {prog_col}{n_prog}/{cfg.n_calib}{RST}"
              + (f"  {G}COMPLETE{RST}" if n_prog >= cfg.n_calib else ""))
        print(f"  Holdout {hold_col}{bar(hold_n, cfg.n_holdout)}{RST}  {hold_col}{hold_n}/{cfg.n_holdout}{RST}"
              + (f"  {G}COMPLETE{RST}" if hold_n >= cfg.n_holdout else ""))
    else:
        prog_col = G if n >= cfg.n_calib else Y
        print(f"  Progress {prog_col}{bar(n, cfg.n_calib)}{RST}  {prog_col}{n}/{cfg.n_calib}{RST}"
              + (f"  {G}COMPLETE{RST}" if n >= cfg.n_calib else ""))

    eta = max(0, (n_target - n) / rate) if rate > 0 else None
    print(f"  N={BOLD}{W}{n}{RST}  rate={Y}{rate:.2f}{RST}/d"
          + (f"  ETA N={n_target}: {Y}~{eta:.0f}d{RST}" if eta and n < n_target else ""))

    # Stats on available trades
    calib_slice = trades[:cfg.n_calib]
    nets  = [t["net_bps"] for t in calib_slice if t["net_bps"] is not None]
    gross = [t.get("gross_bps") for t in calib_slice if t.get("gross_bps") is not None]
    entry_adv = [t.get("entry_adverse_bps") for t in calib_slice
                 if t.get("entry_adverse_bps") is not None]

    if nets:
        wins = sum(1 for v in nets if v > 0)
        print(f"  net  mean={fmt_bps(mean(nets))}  med={fmt_bps(median(nets))}"
              f"  WR={fmt_pct(wins/len(nets)*100)}  ({wins}/{len(nets)})")
        if gross:
            print(f"  gross mean={fmt_bps(mean(gross))}  med={fmt_bps(median(gross))}")
        exits: dict[str, int] = {}
        for t in calib_slice:
            k = t.get("exit_reason") or "?"
            exits[k] = exits.get(k, 0) + 1
        exit_str = "  exits: " + "  ".join(
            f"{k}={v}" for k, v in sorted(exits.items(), key=lambda x: -x[1]))
        print(exit_str)
    else:
        print(f"  {DIM}No closed trades yet.{RST}")

    # Kill / pass criteria
    if cfg.kind == "MAIN":
        k1 = k2 = k3 = None
        if n >= cfg.n_calib and nets:
            k1 = mean(nets) > 0
            if gross and entry_adv:
                k2 = median(entry_adv) < mean(abs(g) for g in gross)
        if q_rate is not None and n >= cfg.n_calib:
            k3 = q_rate < 0.25
        print(kill_status("K1 mean net>0", k1))
        print(kill_status("K2 entry_adv<|gross|", k2))
        print(kill_status("K3 quarantine<25%", k3))
        if q_rate is not None:
            print(f"      {DIM}quarantine proxy: {q_rate*100:.1f}%{RST}")
    else:
        # Lighter criteria for SELL exploratory
        p_med     = None if not nets else median(nets) > 0
        import statistics as _st
        top3_rmv  = (None if len(nets) <= 3
                     else sum(sorted(nets, reverse=True)[3:]) > 0)
        unique_days = len({t["opened_at_utc"][:10] for t in calib_slice}) if calib_slice else 0
        p_days    = None if not calib_slice else (unique_days >= 8)
        print(kill_status("P1 median net>0", p_med))
        print(kill_status("P2 top3-removed cum>0", top3_rmv))
        print(kill_status(f"P3 >= 8 distinct days ({unique_days} so far)", p_days))


def run(watch: bool = False, interval: int = 60) -> None:
    conn = connect()

    while True:
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print("\033[2J\033[H", end="")   # clear screen

        print(f"{BOLD}{C}{'=' * 68}{RST}")
        print(f"{BOLD}  S34 PRE-REGISTRATION MONITOR{RST}              {DIM}{now_utc}{RST}")
        print(f"{BOLD}{C}{'=' * 68}{RST}")

        for cfg in VARIANTS:
            trades = load_trades(conn, cfg)
            rate   = signal_rate(conn, cfg)
            qr     = quarantine_rate(conn, cfg)
            display_variant(cfg, trades, rate, qr)
            print(f"  {DIM}{'-' * 64}{RST}")

        conn.close()

        if not watch:
            break
        print(f"\n  {DIM}Refresh in {interval}s... (Ctrl+C to exit){RST}")
        time.sleep(interval)
        conn = connect()


def main() -> None:
    ap = argparse.ArgumentParser(description="S34 pre-registration monitor (multi-variant)")
    ap.add_argument("--watch",    action="store_true", help="Auto-refresh")
    ap.add_argument("--interval", type=int, default=60, help="Refresh interval seconds")
    args = ap.parse_args()

    if not INTEL_DB.exists():
        print(f"ERROR: {INTEL_DB} not found", file=sys.stderr)
        sys.exit(1)

    run(watch=args.watch, interval=args.interval)


if __name__ == "__main__":
    main()
