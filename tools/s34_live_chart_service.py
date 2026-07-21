"""Service wrapper for s34_live_chart.

Runs the dashboard with stdout/stderr owned by files so it can survive as a
detached pythonw process on Windows.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
(LOG_DIR / "s34_live_chart_service.start.log").write_text("started\n", encoding="utf-8")

sys.stdout = (LOG_DIR / "s34_live_chart.out.log").open("a", encoding="utf-8", buffering=1)
sys.stderr = (LOG_DIR / "s34_live_chart.err.log").open("a", encoding="utf-8", buffering=1)

if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import s34_live_chart  # noqa: E402


if __name__ == "__main__":
    try:
        (LOG_DIR / "s34_live_chart_service.main.log").write_text(f"argv={sys.argv!r}\n", encoding="utf-8")
        code = s34_live_chart.main()
        with (LOG_DIR / "s34_live_chart_service.main.log").open("a", encoding="utf-8") as fh:
            fh.write(f"returned={code!r}\n")
        raise SystemExit(code)
    except BaseException:
        (LOG_DIR / "s34_live_chart_service.fatal.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
