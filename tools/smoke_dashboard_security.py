from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def _req(method: str, url: str, payload: dict | None = None, headers: dict | None = None) -> tuple[int, str]:
    body = None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=req_headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return e.code, raw


def _check(cond: bool, msg: str) -> None:
    if cond:
        print(f"[PASS] {msg}")
        return
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dashboard security smoke checks")
    ap.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--api-key", default="", help="X-Api-Key value")
    ap.add_argument("--operator", default="smoke", help="X-Operator value")
    args = ap.parse_args()

    base = args.base.rstrip("/")
    h_ok = {"X-Api-Key": args.api_key, "X-Operator": args.operator, "X-Role": "admin"}
    h_bad = {"X-Api-Key": "bad-key", "X-Operator": args.operator, "X-Role": "admin"}

    print("== Dashboard Security Smoke ==")
    print(f"base={base}")

    # 1) API key gate (if key is configured server-side this should be 401 with bad key).
    st_bad, _ = _req("POST", f"{base}/api/debug/run", {"action": "validate_env"}, headers=h_bad)
    st_good, _ = _req("POST", f"{base}/api/debug/run", {"action": "validate_env"}, headers=h_ok)
    if args.api_key:
        _check(st_bad == 401, "invalid X-Api-Key rejected")
        _check(st_good in (200, 400, 403), "valid X-Api-Key accepted by auth layer")
    else:
        _check(st_bad in (200, 400, 403), "API key not enforced when --api-key empty")

    # 2) Role gate
    st_view, _ = _req(
        "POST",
        f"{base}/api/debug/run",
        {"action": "validate_env"},
        headers={"X-Api-Key": args.api_key, "X-Operator": args.operator, "X-Role": "viewer"},
    )
    _check(st_view == 403, "viewer role blocked on write endpoint")

    # 3) Idempotency replay (auto-run endpoint, safe)
    idem = f"smoke-{int(time.time())}"
    h_idem = {"X-Api-Key": args.api_key, "X-Operator": args.operator, "X-Role": "admin", "X-Idempotency-Key": idem}
    st1, b1 = _req("POST", f"{base}/api/debug/incidents/auto-run", {}, headers=h_idem)
    st2, b2 = _req("POST", f"{base}/api/debug/incidents/auto-run", {}, headers=h_idem)
    _check(st1 == 200 and st2 == 200, "idempotency endpoint returned 200")
    _check(b1 == b2, "idempotency replay returned same response")

    # 4) Rate limit surface (best-effort: ensure endpoint still responds and no crash)
    sts = []
    for _ in range(3):
        st, _ = _req("POST", f"{base}/api/debug/incidents/auto-run", {}, headers=h_ok)
        sts.append(st)
    _check(all(s in (200, 403, 429) for s in sts), "rate-limited endpoint responses are valid")

    # 5) Security audit endpoint alive
    st_a, b_a = _req("GET", f"{base}/api/debug/security-audit?limit=20", headers=h_ok)
    _check(st_a == 200, "security-audit endpoint reachable")
    try:
        obj = json.loads(b_a)
        _check(isinstance(obj, list), "security-audit payload is list")
    except Exception:
        _check(False, "security-audit payload is valid JSON")

    print("Smoke checks completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

