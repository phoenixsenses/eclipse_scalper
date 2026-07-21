"""Service wrapper for the S34 V Engine v0.2 shadow mirror.

Keeps the observation-only mirror process detached from interactive terminals
and owns stdout/stderr files on Windows.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR / "s34_v_engine_v02_shadow_mirror_service.start.log").write_text("started\n", encoding="utf-8")

sys.stdout = (LOG_DIR / "s34_v_engine_v02_shadow_mirror.out.log").open("a", encoding="utf-8", buffering=1)
sys.stderr = (LOG_DIR / "s34_v_engine_v02_shadow_mirror.err.log").open("a", encoding="utf-8", buffering=1)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import s34_v_engine_v02_shadow_mirror  # noqa: E402


if __name__ == "__main__":
    try:
        (LOG_DIR / "s34_v_engine_v02_shadow_mirror_service.main.log").write_text(
            f"argv={sys.argv!r}\n",
            encoding="utf-8",
        )
        code = s34_v_engine_v02_shadow_mirror.main(["--loop", "--interval-sec", "60"])
        with (LOG_DIR / "s34_v_engine_v02_shadow_mirror_service.main.log").open("a", encoding="utf-8") as fh:
            fh.write(f"returned={code!r}\n")
        raise SystemExit(code)
    except BaseException:
        (LOG_DIR / "s34_v_engine_v02_shadow_mirror_service.fatal.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        raise
