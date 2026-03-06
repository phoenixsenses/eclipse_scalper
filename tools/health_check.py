from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read logs/health/overall.json and print machine status.")
    p.add_argument("--health", default="logs/health/overall.json")
    p.add_argument("--max-staleness-sec", type=int, default=15)
    return p


def _parse_ts(ts: Any) -> float | None:
    if ts is None:
        return None
    text = str(ts).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def main() -> int:
    args = _build_parser().parse_args()
    p = Path(str(args.health))
    obj = _load_json(p)
    if obj is None:
        print(f"health_check halted missing_or_invalid={p}")
        return 2
    ts = _parse_ts(obj.get("ts_utc"))
    if ts is not None:
        age = max(0.0, time.time() - float(ts))
        if age > float(max(0, int(args.max_staleness_sec))):
            print(
                f"health_check halted reason=health_stale age_sec={int(age)} "
                f"max_staleness_sec={int(args.max_staleness_sec)}"
            )
            return 2
    state = str(obj.get("state") or "halted").lower().strip()
    reason = str(obj.get("reason") or "")
    print(f"health_check state={state} reason={reason}")
    comps = obj.get("components") if isinstance(obj.get("components"), dict) else {}
    for name in sorted(comps):
        c = comps.get(name) if isinstance(comps.get(name), dict) else {}
        print(
            f"- {name}: status={c.get('status')} connected={c.get('connected')} "
            f"lag={c.get('progress_lag_sec')} reconnects_5m={c.get('reconnects_last_5m')} errors_5m={c.get('errors_last_5m')}"
        )
    if state == "ok":
        return 0
    if state == "degraded":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
