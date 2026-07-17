"""GET/HEAD-only read-only HTTP server for the canonical S34 operator dashboard.

Hard safety properties (all covered by tests/dashboard):
- binds to 127.0.0.1 only (loopback) on a bounded, configurable port
- serves ONLY GET and HEAD; POST/PUT/PATCH/DELETE -> 405; unknown route -> 404
- never spawns a subprocess, signals a process, mutates env, writes source/runtime
  artifacts, or writes to any database
- exposes no command/control endpoint
- responses come from the read-only aggregator; DB access (in adapters) is mode=ro

The server is stdlib-only (``http.server``) and must NOT import the legacy,
mutation-capable ``dashboard.backend.app`` / ``control_actions`` subsystem.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from dashboard.backend.aggregator import build_overview, build_panel
from dashboard.backend.sources import REPO_ROOT, DashboardContext

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
CACHE_TTL_SEC = 3.0

_ALLOWED_METHODS = ("GET", "HEAD")
_STATIC_CONTENT_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


class _OverviewCache:
    """Tiny short-TTL cache. A cache miss/failure is never presented as fresh
    data — callers see the freshly-built payload or a typed error."""

    def __init__(self, ttl: float = CACHE_TTL_SEC) -> None:
        self._ttl = ttl
        self._lock = threading.Lock()
        self._value: dict[str, Any] | None = None
        self._built_at = 0.0

    def get(self, ctx: DashboardContext) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            if self._value is not None and (now - self._built_at) <= self._ttl:
                cached = dict(self._value)
                cached["_cache"] = {"hit": True, "age_seconds": round(now - self._built_at, 3)}
                return cached
        payload = build_overview(ctx)
        with self._lock:
            self._value = payload
            self._built_at = now
        out = dict(payload)
        out["_cache"] = {"hit": False, "age_seconds": 0.0}
        return out


def make_handler(ctx: DashboardContext, cache: _OverviewCache):
    class Handler(BaseHTTPRequestHandler):
        server_version = "S34OperatorDashboard/1.0"
        protocol_version = "HTTP/1.1"

        # ---- helpers ----
        def _send_json(self, obj: Any, code: int = 200, head_only: bool = False) -> None:
            body = json.dumps(obj, ensure_ascii=True, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Dashboard-Contract", "read-only; GET/HEAD only")
            self.end_headers()
            if not head_only:
                self.wfile.write(body)

        def _send_bytes(self, data: bytes, content_type: str, code: int = 200, head_only: bool = False) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if not head_only:
                self.wfile.write(data)

        def _serve_static(self, rel: str, head_only: bool) -> None:
            # Path traversal guard: resolve under STATIC_DIR only.
            target = (STATIC_DIR / rel).resolve()
            try:
                target.relative_to(STATIC_DIR.resolve())
            except ValueError:
                self._send_json({"error": "not_found"}, 404, head_only)
                return
            if not target.is_file():
                self._send_json({"error": "not_found"}, 404, head_only)
                return
            ct = _STATIC_CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            self._send_bytes(target.read_bytes(), ct, 200, head_only)

        def _route(self, head_only: bool) -> None:
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/" or path == "/index.html":
                idx = TEMPLATES_DIR / "index.html"
                if idx.is_file():
                    self._send_bytes(idx.read_bytes(), "text/html; charset=utf-8", 200, head_only)
                else:
                    self._send_json({"error": "template_missing"}, 500, head_only)
                return
            if path == "/api/overview":
                self._send_json(cache.get(ctx), 200, head_only)
                return
            if path == "/api/health":
                self._send_json({"ok": True, "read_only": True, "ts": time.time()}, 200, head_only)
                return
            if path.startswith("/api/panel/"):
                key = path[len("/api/panel/"):]
                panel = build_panel(key, ctx)
                if panel is None:
                    self._send_json({"error": "unknown_panel", "key": key}, 404, head_only)
                else:
                    self._send_json(panel, 200, head_only)
                return
            if path.startswith("/static/"):
                self._serve_static(path[len("/static/"):], head_only)
                return
            self._send_json({"error": "not_found", "path": path}, 404, head_only)

        # ---- methods ----
        def do_GET(self) -> None:  # noqa: N802
            self._route(head_only=False)

        def do_HEAD(self) -> None:  # noqa: N802
            self._route(head_only=True)

        def _reject(self) -> None:
            self.send_response(405)
            self.send_header("Allow", ", ".join(_ALLOWED_METHODS))
            self.send_header("Content-Length", "0")
            self.end_headers()

        # Every mutating method is explicitly rejected with 405.
        do_POST = _reject     # noqa: N815
        do_PUT = _reject      # noqa: N815
        do_PATCH = _reject    # noqa: N815
        do_DELETE = _reject   # noqa: N815
        do_OPTIONS = _reject  # noqa: N815

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            return

    return Handler


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                  ctx: DashboardContext | None = None) -> ThreadingHTTPServer:
    """Create (but do not serve) the read-only server bound to loopback."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(f"read-only dashboard only binds loopback, got host={host!r}")
    ctx = ctx or DashboardContext(repo_root=REPO_ROOT)
    cache = _OverviewCache()
    handler = make_handler(ctx, cache)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    return httpd


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          ctx: DashboardContext | None = None) -> int:
    """Blocking serve loop with graceful shutdown on Ctrl-C / SIGTERM."""
    httpd = create_server(host, port, ctx)
    print(f"S34 canonical operator dashboard (READ-ONLY) serving on http://{host}:{port}")
    print("Contract: GET/HEAD only; no trade/order/executor/scheduler/process control; DB mode=ro.")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("\nGraceful shutdown requested (KeyboardInterrupt).")
    finally:
        httpd.shutdown()
        httpd.server_close()
        print("Server stopped.")
    return 0
