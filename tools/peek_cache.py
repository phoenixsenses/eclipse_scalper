# tools/peek_cache.py — SCALPER ETERNAL — CACHE PEEK — 2026 v1.1 (IMPORT-PROOF)
# Fix: ensures project root is on sys.path so "import execution" works when run as:
#   python tools/peek_cache.py
# Also: prints what keys exist in bot.data.ohlcv, raw_symbol map, and sample row counts.

from __future__ import annotations

import sys
from pathlib import Path

# --- HARD FIX: add project root (folder containing "execution/") to sys.path ---
ROOT = Path(__file__).resolve().parents[1]  # .../eclipse_scalper
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def _symkey(sym: str) -> str:
    s = (sym or "").upper().strip()
    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")
    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _pick_symbols_from_env_or_default() -> List[str]:
    raw = os.getenv("ACTIVE_SYMBOLS", "").strip()
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if parts:
            return parts
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]


async def main() -> None:
    print("CWD:", os.getcwd())
    print("ROOT:", str(ROOT))
    print("sys.path[0]:", sys.path[0])

    # Import AFTER sys.path fix
    from execution.bootstrap import _load_cfg, _init_exchange  # type: ignore

    # Optional: use data_loop helpers if present
    try:
        from execution.data_loop import _resolve_raw_symbol  # type: ignore
    except Exception:
        _resolve_raw_symbol = None

    cfg = _load_cfg()
    ex = await _init_exchange(cfg)

    # minimal bot + data structure
    bot = SimpleNamespace()
    bot.cfg = cfg
    bot.ex = ex

    # If your project has the MiniDataCache class, use it; otherwise fall back to dicts.
    try:
        from execution.data_loop import _MiniDataCache  # type: ignore

        bot.data = _MiniDataCache(
            price={},
            ohlcv={},
            last_poll_ts={},
            raw_symbol={},
            lock=asyncio.Lock(),
            base_timeframe=str(os.getenv("DATA_TIMEFRAME", "1m")),
        )
    except Exception:
        bot.data = SimpleNamespace(
            price={},
            ohlcv={},
            last_poll_ts={},
            raw_symbol={},
            lock=asyncio.Lock(),
            base_timeframe=str(os.getenv("DATA_TIMEFRAME", "1m")),
        )

    # pick symbols
    syms = _pick_symbols_from_env_or_default()
    tf = str(os.getenv("DATA_TIMEFRAME", "1m"))
    limit = int(os.getenv("DATA_OHLCV_LIMIT", "50"))

    print(f"\nSymbols ({len(syms)}): {syms}")
    print(f"TF={tf} LIMIT={limit}\n")

    # fetch a tiny sample and show what keys get created
    fetch_ohlcv = getattr(ex, "fetch_ohlcv", None)
    if not callable(fetch_ohlcv):
        print("exchange.fetch_ohlcv missing (ccxt exchange not initialized correctly?)")
        await ex.close()
        return

    for sym in syms:
        k = _symkey(sym)

        # decide raw symbol
        sym_raw = None
        if callable(_resolve_raw_symbol):
            try:
                sym_raw = _resolve_raw_symbol(ex, bot.data, k)
            except Exception:
                sym_raw = None
        if not sym_raw:
            # heuristic fallback
            if k.endswith("USDT") and len(k) > 4:
                base = k[:-4]
                sym_raw = f"{base}/USDT:USDT"
            else:
                sym_raw = sym

        # fetch
        rows = None
        try:
            rows = await fetch_ohlcv(sym_raw, timeframe=tf, limit=limit)
        except Exception as e:
            print(f"{k:10s} raw={sym_raw:16s} OHLCV FAIL: {type(e).__name__} {e}")
            continue

        # store like your data_loop does (canonical + raw)
        try:
            bot.data.raw_symbol[k] = sym_raw
        except Exception:
            pass

        n = len(rows) if isinstance(rows, list) else 0
        print(f"{k:10s} raw={sym_raw:16s} rows={n}")

        if n:
            # normalize row format like your loop expects
            norm = []
            for r in rows:
                try:
                    if isinstance(r, (list, tuple)) and len(r) >= 6:
                        norm.append([int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])])
                except Exception:
                    continue

            try:
                bot.data.ohlcv[k] = norm
                bot.data.ohlcv[sym_raw] = norm
            except Exception:
                pass

    # show cache keys summary
    ohlcv = getattr(bot.data, "ohlcv", {})
    raw_map = getattr(bot.data, "raw_symbol", {})

    print("\n--- CACHE SUMMARY ---")
    print("ohlcv keys count:", len(ohlcv) if isinstance(ohlcv, dict) else "N/A")
    if isinstance(ohlcv, dict):
        sample_keys = list(ohlcv.keys())[:25]
        print("ohlcv sample keys:", sample_keys)

    if isinstance(raw_map, dict):
        print("raw_symbol map:")
        for sym in syms:
            k = _symkey(sym)
            print(f"  {k:10s} -> {raw_map.get(k)}")

    await ex.close()


if __name__ == "__main__":
    asyncio.run(main())
