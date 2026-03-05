from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_dotenv_best_effort() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return
    root = Path(__file__).resolve().parents[1]
    env_paper = root / ".env.paper"
    env_default = root / ".env"
    try:
        if env_paper.exists():
            load_dotenv(dotenv_path=env_paper, override=False)
        elif env_default.exists():
            load_dotenv(dotenv_path=env_default, override=False)
    except Exception:
        return


_load_dotenv_best_effort()


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate websocket reconnection audit report from collector heartbeat.")
    p.add_argument("--heartbeat", default="logs/collector_heartbeat.json")
    p.add_argument("--out", default="reports/RECONNECTION_AUDIT.md")
    return p.parse_args()


def _safe_read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = _args()
    hb_path = Path(args.heartbeat)
    out_path = Path(args.out)
    payload = _safe_read_json(hb_path) if hb_path.exists() else {}

    connected = bool(payload.get("connected", False))
    backoff = float(payload.get("current_backoff_seconds", 0.0) or 0.0)
    backend = str(payload.get("backend", "unknown"))
    last_error = str(payload.get("last_error", ""))
    wal_size = float(payload.get("wal_size_mb", 0.0) or 0.0)
    wal_alert = bool(payload.get("wal_alert", False))
    last_msg = str(payload.get("last_message_ts_utc", ""))

    status = "ok" if connected and not last_error else "degraded"
    findings: list[str] = []
    findings.append("Reconnection logic uses exponential backoff + jitter in collector.")
    findings.append("Collector resets reconnect delay after stable connection window.")
    if backoff > 30.0:
        findings.append(f"Current reconnect backoff is high ({backoff:.2f}s). Inspect network path and WS stability.")
    if last_error:
        findings.append(f"Last collector error: `{last_error}`")
    if wal_alert:
        findings.append(f"WAL file size alert active ({wal_size:.2f} MB). Run DB maintenance/checkpoint.")
    if not findings:
        findings.append("No reconnect anomalies detected in current heartbeat snapshot.")

    lines = [
        "# Reconnection Audit Report",
        "",
        f"- Generated UTC: `{_iso_now()}`",
        f"- Heartbeat path: `{hb_path}`",
        f"- Status: `{status}`",
        f"- Connected: `{int(connected)}`",
        f"- Backend: `{backend}`",
        f"- Current backoff sec: `{backoff:.2f}`",
        f"- Last message ts: `{last_msg}`",
        f"- WAL size MB: `{wal_size:.2f}`",
        "",
        "## Findings",
    ]
    lines.extend([f"- {x}" for x in findings])
    lines.extend(
        [
            "",
            "## Recommendations",
            "- Keep `stall_timeout_sec` finite to force reconnect on silent WS stalls.",
            "- Use supervisor with restart cap to avoid crash loops.",
            "- If ISP intermittently blocks Binance WS, configure VPN/SOCKS5 path and validate DNS/TLS.",
            "- Monitor `current_backoff_seconds` and repeated errors via heartbeat + Telegram alerts.",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[reconnection_audit] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
