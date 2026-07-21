#!/usr/bin/env python
"""Print X/Twitter cookie counts per local Chrome profile."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def main() -> int:
    root = Path(os.environ["LOCALAPPDATA"]) / "Google" / "Chrome" / "User Data"
    for profile in ("Default", "Profile 1", "Profile 2", "Profile 3"):
        cookies = root / profile / "Network" / "Cookies"
        if not cookies.exists():
            continue
        try:
            con = sqlite3.connect(f"file:{cookies}?mode=ro", uri=True)
            count = con.execute(
                """
                select count(*)
                from cookies
                where instr(host_key, 'x.com') > 0
                   or instr(host_key, 'twitter.com') > 0
                """
            ).fetchone()[0]
            con.close()
        except Exception as exc:
            count = f"ERR {type(exc).__name__}"
        print(f"{profile}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
