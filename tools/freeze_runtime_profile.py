from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from tools.run_summary import build_run_summary

PROFILE_KEYS = [
    "SCALPER_DRY_RUN",
    "ACTIVE_SYMBOLS",
    "ENTRY_LOOP_MODE",
    "ENTRY_MIN_CONFIDENCE",
    "ENTRY_ADAPTIVE_GUARD_ENABLED",
    "ENTRY_REGIME",
    "ENTRY_ALLOW_UNKNOWN_REGIME",
    "ENTRY_REGIME_WARMUP_SEC",
    "ENTRY_REGIME_RISK_ENABLED",
    "ENTRY_MICRO_SIGNAL_ENABLED",
    "MICRO_SIGNAL_SYMBOL",
    "MICRO_SIGNAL_DB",
    "MICRO_SIGNAL_LOOKBACK_SEC",
    "MICRO_SIGNAL_BUCKET_SEC",
    "MICRO_SIGNAL_REQUIRE_REGIME",
    "MICRO_SIGNAL_ALLOW_UNKNOWN_REGIME",
    "MICRO_SIGNAL_REGIME_WARMUP_SEC",
    "EXCHANGE",
    "DEFAULT_TYPE",
    "BINANCE_RECV_WINDOW",
    "NOTIFY_ENABLED",
    "NOTIFY_HEARTBEAT",
    "NOTIFY_SEND_TIMEOUT_SEC",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _collect_profile() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in PROFILE_KEYS:
        out[k] = str(os.getenv(k, "")).strip()
    return out


def _profile_hash(profile: Dict[str, str]) -> str:
    raw = json.dumps(profile, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze and enforce runtime profile lock.")
    p.add_argument("--out-json", default="reports/RUNTIME_PROFILE_LOCK.json")
    p.add_argument("--out-md", default="reports/RUNTIME_PROFILE_LOCK.md")
    p.add_argument("--enforce", action="store_true", help="Fail if current profile hash mismatches existing lock file.")
    p.add_argument("--write-lock", action="store_true", help="Write/refresh lock artifacts.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_json = Path(str(args.out_json))
    out_md = Path(str(args.out_md))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    profile = _collect_profile()
    prof_hash = _profile_hash(profile)
    payload: Dict[str, Any] = {
        "ts_utc": _utc_now(),
        "hash": prof_hash,
        "profile": profile,
    }

    old_hash = ""
    if out_json.exists():
        try:
            old = json.loads(out_json.read_text(encoding="utf-8"))
            old_hash = str(old.get("hash", "")).strip()
        except Exception:
            old_hash = ""

    if bool(args.enforce) and old_hash and old_hash != prof_hash:
        print(f"runtime_profile_lock: mismatch old={old_hash} new={prof_hash}")
        return 1

    payload["run_summary"] = build_run_summary(
        run_type="freeze_runtime_profile",
        inputs={"enforce": bool(args.enforce), "write_lock": bool(args.write_lock)},
        metrics={"profile_key_count": len(profile), "hash_changed": bool(old_hash and old_hash != prof_hash)},
        artifacts={"json": str(out_json), "md": str(out_md)},
    )

    if bool(args.write_lock) or (not out_json.exists()):
        out_json.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Runtime Profile Lock",
        "",
        f"- generated_utc: {payload['ts_utc']}",
        f"- profile_hash: `{prof_hash}`",
        f"- enforce_mode: `{int(bool(args.enforce))}`",
        "",
        "## Keys",
    ]
    for k in PROFILE_KEYS:
        md.append(f"- `{k}` = `{profile.get(k, '')}`")
    md.extend(["", "## Run Summary", f"- `{payload['run_summary']}`"])
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"runtime_profile_lock: hash={prof_hash}")
    print(f"runtime_profile_lock: wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
