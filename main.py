#!/usr/bin/env python3
# main.py — SCALPER ETERNAL — ETERNAL ASCENDANT LAUNCHER — 2026 v4.2 (HARDENED)
# AUTHORITATIVE launcher:
# - main.py decides LIVE vs DRY-RUN (no lingering env ghosts)
# - Explicitly clears SCALPER_DRY_RUN unless requested
# - Windows asyncio policy hardened
# - Clean shutdown on Ctrl+C
# - Zero behavior change required in runner/core/exchange

import argparse
import asyncio
import os
import sys

from bot.runner import run_bot


def _set_windows_asyncio_policy() -> None:
    """
    Windows: ProactorEventLoop can break some socket/subprocess patterns in certain libs.
    Selector policy is usually safer for trading bots using websockets/aiofiles/ccxt async.
    Safe no-op on non-Windows.
    """
    try:
        if sys.platform.startswith("win"):
            # Only available on Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SCALPER ETERNAL — Blade Ascendant Launcher (2026 v4.2)"
    )

    parser.add_argument(
        "--equity",
        type=float,
        help="Override starting equity (e.g. --equity 45)"
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "micro", "production"],
        default="auto",
        help="Config mode: auto | micro | production"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry-run mode (NO REAL ORDERS)"
    )

    args = parser.parse_args()

    # ─────────────────────────────────────────────
    # ENVIRONMENT — MAIN.PY IS GOD
    # ─────────────────────────────────────────────

    # Equity override
    if args.equity is not None:
        os.environ["SCALPER_EQUITY"] = str(args.equity)

    # Mode (auto/micro/production)
    os.environ["SCALPER_MODE"] = args.mode

    # Dry-run authority
    if args.dry_run:
        os.environ["SCALPER_DRY_RUN"] = "1"
    else:
        # CRITICAL: remove any lingering dry-run flag
        os.environ.pop("SCALPER_DRY_RUN", None)

    # Optional: force signal looseness if micro
    if args.mode == "micro":
        os.environ.setdefault("SCALPER_SIGNAL_PROFILE", "micro")

    # ─────────────────────────────────────────────
    # HONEST STARTUP BANNER (NO LIES)
    # ─────────────────────────────────────────────

    effective_dry = os.getenv("SCALPER_DRY_RUN")
    effective_equity = os.getenv("SCALPER_EQUITY")
    effective_signal = os.getenv("SCALPER_SIGNAL_PROFILE")

    print("─" * 80)
    print("SCALPER ETERNAL — LAUNCH CONFIRMATION")
    print(f"Python          : {sys.version.split()[0]}")
    print(f"Mode            : {args.mode}")
    print(f"Equity Override : {effective_equity or 'EXCHANGE BALANCE'}")
    print(f"Dry Run         : {'YES (SIMULATION)' if effective_dry else 'NO (LIVE ORDERS)'}")
    print(f"Signal Profile  : {effective_signal or 'default'}")
    print("─" * 80)

    if not effective_dry:
        print("⚠️  LIVE MODE CONFIRMED — REAL ORDERS MAY BE PLACED")
        print("⚠️  Ensure API KEYS, LEVERAGE, and SYMBOLS are correct")
        print("─" * 80)

    # ─────────────────────────────────────────────
    # RUN BOT
    # ─────────────────────────────────────────────

    _set_windows_asyncio_policy()

    try:
        asyncio.run(run_bot())
        return 0
    except KeyboardInterrupt:
        # Clean exit for Ctrl+C
        print("\n─" * 80)
        print("SCALPER ETERNAL — SHUTDOWN (KeyboardInterrupt)")
        print("─" * 80)
        return 130
    except Exception as e:
        # Crash banner (don’t silently die)
        print("\n" + "─" * 80)
        print("SCALPER ETERNAL — FATAL LAUNCH ERROR")
        print(f"{type(e).__name__}: {e}")
        print("─" * 80)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
