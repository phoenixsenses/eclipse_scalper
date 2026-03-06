from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast ops smoke checks for paper run.")
    p.add_argument("--env", default=".env.paper")
    p.add_argument("--base", default="http://127.0.0.1:8765", help="Dashboard backend base URL")
    p.add_argument("--skip-local", action="store_true", help="Skip validate_env and push_status checks")
    return p


def _run(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return int(p.returncode), (p.stdout + p.stderr)


def _sanitize_output(text: str) -> str:
    # never leak env secrets through smoke output
    for key in ("TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_SECRET", "BINANCE_KEY"):
        val = os.getenv(key)
        if val:
            text = text.replace(val, "<redacted>")
    return text


def _http_ok(url: str, timeout: float = 4.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return (200 <= int(resp.status) < 300), f"{resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"http_{e.code}"
    except Exception as e:
        return False, f"err:{type(e).__name__}"


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    for d in (Path("logs"), Path("logs/pids"), Path("logs/health")):
        d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SKIP_EXCHANGE_AUTH_IN_DRYRUN"] = "1"

    base = args.base.rstrip("/")
    checks = [
        "/api/health",
        "/api/runtime",
        "/api/ops/health",
        "/api/ops/supervisor",
        "/api/debug/security-audit?limit=5",
    ]
    print(f"ops_smoke dashboard_base={base}")
    all_ok = True
    for path in checks:
        ok, detail = _http_ok(base + path)
        print(f"ops_smoke endpoint {path} ok={int(ok)} detail={detail}")
        all_ok = all_ok and ok

    if not all_ok:
        return 1

    if not args.skip_local:
        rc_v, out_v = _run([sys.executable, "-m", "tools.validate_env", "--env", str(args.env)], env)
        print(f"ops_smoke validate_env_rc={rc_v}")
        print(_sanitize_output(out_v))
        if rc_v not in (0, 2):  # 2 may happen for non-exchange env issues
            return 1
        rc_p, out_p = _run([sys.executable, "-m", "tools.push_status"], env)
        print(f"ops_smoke push_status_rc={rc_p}")
        print(_sanitize_output(out_p))
        # missing telegram config is acceptable in smoke; send failure is acceptable but surfaced
        if rc_p not in (0, 2, 3):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
