from __future__ import annotations

import argparse
import os
from typing import Iterable


DEFAULT_NAMES = [
    "BINANCE_API_KEY",
    "BINANCE_KEY",
    "BINANCE_API_SECRET",
    "BINANCE_SECRET",
    "TELEGRAM_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "SCALPER_DRY_RUN",
    "SKIP_EXCHANGE_AUTH_IN_DRYRUN",
]


def _meta(value: str) -> str:
    s = str(value or "")
    has_quote = ("'" in s or '"' in s)
    trail_ws = (s != s.strip())
    return f"len={len(s)} has_quote={has_quote} trail_ws={trail_ws}"


def _iter_names(raw: str) -> Iterable[str]:
    if not raw:
        return list(DEFAULT_NAMES)
    return [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="Safe env debug helper (never prints secret values).")
    p.add_argument("--names", default=",".join(DEFAULT_NAMES), help="Comma-separated env var names.")
    args = p.parse_args()
    for n in _iter_names(str(args.names)):
        if n in os.environ:
            print(f"{n}: present {_meta(os.environ.get(n, ''))}")
        else:
            print(f"{n}: missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

