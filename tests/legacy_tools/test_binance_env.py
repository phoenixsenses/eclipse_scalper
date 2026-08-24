#!/usr/bin/env python3
"""
test_binance_env.py
────────────────────────────────────────────────────────
Safe .env / API key verification for CCXT Binance.

What it does:
- Loads .env (if python-dotenv installed; otherwise relies on existing env)
- Detects common env var names (BINANCE_API_KEY / BINANCE_API_SECRET / etc.)
- Instantiates ccxt.binance with those creds
- Calls:
    1) exchange.load_markets()         (proves metadata access)
    2) exchange.fetch_ticker(symbol)   (public, should always work)
    3) exchange.fetch_balance()        (private, proves API keys work)
- Prints helpful errors without leaking secrets.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional, Tuple

def _mask(s: str, keep: int = 4) -> str:
    s = str(s or "")
    if len(s) <= keep:
        return "*" * len(s)
    return "*" * (len(s) - keep) + s[-keep:]


def _get_first_env(*names: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (value, name) of the first env var found from names."""
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip() != "":
            return str(v).strip(), n
    return None, None


def main() -> int:
    # ---- Load .env (optional) ----
    dotenv_loaded = False
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        dotenv_loaded = True
    except Exception:
        dotenv_loaded = False

    print("=== ENV TEST: .env + Binance (ccxt) ===")
    print(f"python: {sys.version.split()[0]}")
    print(f".env loader: {'OK (python-dotenv)' if dotenv_loaded else 'NOT LOADED (install python-dotenv or rely on OS env)'}")
    print()

    # ---- Find credentials (support common naming schemes) ----
    api_key, api_key_name = _get_first_env(
        "BINANCE_API_KEY",
        "BINANCE_KEY",
        "API_KEY",
        "CCXT_BINANCE_API_KEY",
    )
    api_secret, api_secret_name = _get_first_env(
        "BINANCE_API_SECRET",
        "BINANCE_SECRET",
        "API_SECRET",
        "CCXT_BINANCE_API_SECRET",
    )

    # Optional: some setups use passphrase; Binance typically doesn't.
    api_password, api_password_name = _get_first_env(
        "BINANCE_PASSWORD",
        "API_PASSWORD",
        "CCXT_BINANCE_PASSWORD",
    )

    print("Detected credential env vars:")
    print(f"  {api_key_name or 'BINANCE_API_KEY/etc.'}: {_mask(api_key) if api_key else 'MISSING'}")
    print(f"  {api_secret_name or 'BINANCE_API_SECRET/etc.'}: {_mask(api_secret) if api_secret else 'MISSING'}")
    if api_password or api_password_name:
        print(f"  {api_password_name}: {_mask(api_password) if api_password else 'EMPTY'}")
    print()

    if not api_key or not api_secret:
        print("❌ Missing API key/secret in environment.")
        print("Make sure your .env has one of these pairs, for example:")
        print("  BINANCE_API_KEY=... ")
        print("  BINANCE_API_SECRET=... ")
        return 2

    # ---- Import ccxt ----
    try:
        import ccxt  # type: ignore
    except Exception:
        print("❌ ccxt not installed in this environment.")
        print("Install:  pip install ccxt")
        return 3

    # ---- Settings ----
    symbol = os.getenv("TEST_SYMBOL", "BTC/USDT")
    market_mode = os.getenv("TEST_MARKET_MODE", "spot").strip().lower()
    # market_mode options: spot | swap | future
    # For futures/swap, ccxt uses options['defaultType'] = 'future' or 'swap' depending on exchange.
    # For Binance USDT-M perpetuals, ccxt commonly uses 'future' or 'swap' depending on version.
    default_type = None
    if market_mode in ("swap", "future"):
        default_type = market_mode

    print("Test settings:")
    print(f"  TEST_SYMBOL={symbol}")
    print(f"  TEST_MARKET_MODE={market_mode}")
    print()

    # ---- Create exchange ----
    try:
        cfg = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "timeout": 30000,
        }
        if default_type:
            cfg["options"] = {"defaultType": default_type}

        # NOTE: If your account is not futures-enabled, swap/future balance calls can fail.
        ex = ccxt.binance(cfg)

        # ---- 1) Markets ----
        print("1) load_markets() …")
        markets = ex.load_markets()
        print(f"   ✅ markets loaded: {len(markets)}")
        print(f"   ✅ symbol exists? {symbol in markets}")
        print()

        # ---- 2) Public ticker ----
        print("2) fetch_ticker(symbol) …")
        t = ex.fetch_ticker(symbol)
        last = t.get("last")
        print(f"   ✅ ticker ok: last={last}")
        print()

        # ---- 3) Private balance ----
        print("3) fetch_balance() [PRIVATE] …")
        bal = ex.fetch_balance()
        # Print only safe summary
        total = bal.get("total", {}) or {}
        usdt_total = total.get("USDT")
        print("   ✅ private call ok (API keys work).")
        if usdt_total is not None:
            print(f"   USDT total (if available): {usdt_total}")
        else:
            print("   (USDT total not present in this balance response — depends on market mode/account type.)")

        print()
        print("✅ RESULT: API keys + .env are working for CCXT Binance.")
        return 0

    except Exception as e:
        msg = str(e)
        print("❌ ERROR during exchange tests.")
        print(f"   {msg}")

        # Quick diagnosis hints
        if "requires \"apiKey\"" in msg or "apiKey" in msg and "requires" in msg:
            print("\nHint: ccxt is not receiving apiKey/secret. Check env var names and .env loading.")
        if "Invalid API-key" in msg or "invalid api-key" in msg.lower():
            print("\nHint: Key is invalid, restricted by IP, or not enabled for required permissions.")
        if "Permission" in msg or "permission" in msg.lower():
            print("\nHint: API key permissions (Spot/Futures) are not enabled.")
        if "symbol" in msg.lower() and "not found" in msg.lower():
            print("\nHint: Wrong symbol for the selected market mode. Try TEST_SYMBOL=BTC/USDT for spot.")

        # Optional stack trace for deeper debugging
        if os.getenv("TEST_TRACE", "0").strip() in ("1", "true", "yes", "y", "on"):
            print("\n--- TRACE ---")
            traceback.print_exc()

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
