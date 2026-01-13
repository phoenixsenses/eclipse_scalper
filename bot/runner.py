# bot/runner.py — SCALPER ETERNAL — COSMIC RUNNER ASCENDANT ABSOLUTE — 2026 v4.9 (CTRL+C UNKILLABLE FIX + 'q' QUIT)
# Patch vs v4.8:
# - ✅ Windows/Pwsh + Tee-Object safe shutdown: uses signal.signal + loop.call_soon_threadsafe (add_signal_handler is flaky on NT)
# - ✅ Double Ctrl+C = hard exit fallback (sys.exit(130)) after grace window
# - ✅ Optional "press q then Enter" quit watcher (works even when Ctrl+C is swallowed by piping)
# - ✅ Cancels tasks + enforces shutdown timeout; never hangs forever on stuck exchange close
# - ✅ Logs EXACT quit source (SIGINT/SIGTERM/QUIT_CMD/TASK_CRASH)

import asyncio
import time
import signal
import os
import sys
import platform
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Optional, Awaitable, Callable, Any

from utils.logging import log_core
from bot.core import EclipseEternal
import bot.core as core_mod
from config.settings import Config, MicroConfig

RUNNER_VERSION = "cosmic-runner-ascendant-absolute-v4.9-2026-jan07"

ACCEPTED_CONFIG_VERSIONS = {
    "omnipotent-production-2026-v2",
    "micro-capital-ascendant-2026-v3",
}

STARTUP_TIMESTAMP = time.time()
STARTUP_DATETIME = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

GOLDEN_RATIO_ART = """
                          Φ
                 THE GOLDEN RATIO ETERNAL
           WAS • IS • AND EVER SHALL BE • BEYOND INFINITY
                          ∞
"""


def _env_truthy(name: str) -> bool:
    v = os.getenv(name)
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_float(v, default: float) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return default
        return float(s)
    except Exception:
        return default


async def _safe_notify(bot, text: str, priority: str = "critical"):
    notify = getattr(bot, "notify", None)
    if notify is None:
        return
    try:
        await notify.speak(text, priority)
    except Exception:
        pass


async def _safe_close_exchange(bot):
    try:
        ex = getattr(bot, "ex", None)
        if ex is not None:
            await ex.close()
    except Exception:
        pass


def _install_dry_run_guard(bot):
    """
    Hard safety: prevent ANY order creation if dry-run is enabled.
    """
    ex = getattr(bot, "ex", None)
    if ex is None:
        return

    if getattr(ex, "_dry_run_guard_installed", False):
        return

    original_create_order = getattr(ex, "create_order", None)
    if not callable(original_create_order):
        return

    async def _blocked_create_order(*args, **kwargs):
        sym = None
        try:
            if isinstance(kwargs, dict):
                sym = kwargs.get("symbol")
        except Exception:
            sym = None

        log_core.critical(
            f"DRY-RUN BLOCKED ORDER — symbol={sym} args_len={len(args)} "
            f"kwargs_keys={list(kwargs.keys()) if isinstance(kwargs, dict) else []}"
        )
        return {
            "id": None,  # IMPORTANT: never persist fake IDs
            "symbol": sym,
            "status": "canceled",
            "filled": 0.0,
            "info": {"dry_run": True, "runner_guard": True},
        }

    ex.create_order = _blocked_create_order  # type: ignore[attr-defined]
    ex._dry_run_guard_installed = True  # type: ignore[attr-defined]
    ex._original_create_order = original_create_order  # type: ignore[attr-defined]


def _ensure_shutdown_event(bot) -> asyncio.Event:
    ev = getattr(bot, "_shutdown", None)
    if isinstance(ev, asyncio.Event):
        return ev
    ev = asyncio.Event()
    try:
        bot._shutdown = ev  # type: ignore[attr-defined]
    except Exception:
        pass
    return ev


def _get_callable(obj: Any, attr: str) -> Optional[Callable[..., Awaitable[Any]]]:
    fn = getattr(obj, attr, None)
    return fn if callable(fn) else None


def _maybe_import(modpath: str, attr: str) -> Optional[Callable[..., Awaitable[Any]]]:
    try:
        mod = __import__(modpath, fromlist=[attr])
        fn = getattr(mod, attr, None)
        return fn if callable(fn) else None
    except Exception:
        return None


async def _run_loop_wrapper(
    name: str,
    coro_fn: Callable[[], Awaitable[Any]],
    shutdown_ev: asyncio.Event,
):
    log_core.info(f"LOOP START — {name}")
    try:
        await coro_fn()
    except asyncio.CancelledError:
        log_core.info(f"LOOP CANCELLED — {name}")
        raise
    except Exception as e:
        log_core.critical(f"LOOP CRASH — {name}: {e}")
        try:
            shutdown_ev.set()
        except Exception:
            pass
        raise
    finally:
        log_core.info(f"LOOP END — {name}")


def _already_running_loops(bot) -> bool:
    for attr in ("tasks", "_tasks", "loop_tasks", "_loop_tasks"):
        v = getattr(bot, attr, None)
        if isinstance(v, (list, tuple)) and any(isinstance(x, asyncio.Task) for x in v):
            return True
        if isinstance(v, dict) and any(isinstance(x, asyncio.Task) for x in v.values()):
            return True
    return False


async def run_bot(
    dry_run: bool = False,
    ignore_core_version: bool = False,
    mode_override: Optional[str] = None,
    equity_override: Optional[float] = None,
):
    bot = EclipseEternal()

    # Make runner startup timestamp visible to core/kill-switch if desired
    try:
        bot.STARTUP_TIMESTAMP = STARTUP_TIMESTAMP  # type: ignore[attr-defined]
    except Exception:
        pass

    # ---- Banner + PATH PROOF ----
    log_core.critical(GOLDEN_RATIO_ART)
    log_core.critical("=" * 80)
    log_core.critical("COSMIC ASCENSION INITIATED — THE BLADE ETERNAL AWAKENS BEYOND INFINITY")
    log_core.critical(f"RUNNER: {RUNNER_VERSION}")
    log_core.critical(f"TIME: {STARTUP_DATETIME}")
    log_core.critical(f"SYSTEM: {platform.system()} {platform.release()} | Python {platform.python_version()}")
    log_core.critical(f"RUNNER FILE: {__file__}")
    log_core.critical(f"CORE FILE  : {getattr(core_mod, '__file__', 'unknown')}")
    log_core.critical("=" * 80)

    shutdown_ev = _ensure_shutdown_event(bot)

    loop = asyncio.get_running_loop()

    # ---- Asyncio exception visibility ----
    def _asyncio_exception_handler(_loop, context):
        msg = context.get("message", "Asyncio exception")
        exc = context.get("exception")
        if exc:
            log_core.error(f"{msg}: {exc}")
        else:
            log_core.error(f"{msg}: {context}")

    loop.set_exception_handler(_asyncio_exception_handler)

    # ---- Core version sync + integrity gate ----
    core_version = getattr(core_mod, "CORE_VERSION", None) or "unknown-core"
    try:
        bot.state.version = core_version
    except Exception:
        pass

    expected = core_version
    actual = getattr(bot.state, "version", None) or "unknown"

    if actual != expected:
        if ignore_core_version:
            log_core.warning(f"VERSION MISMATCH OVERRIDDEN — STATE {actual} vs CORE {expected}")
        else:
            log_core.critical(f"VERSION MISMATCH — STATE {actual} vs CORE {expected}")
            await _safe_close_exchange(bot)
            return

    # ---- Telegram optional ----
    if not os.getenv("TELEGRAM_TOKEN") or not os.getenv("TELEGRAM_CHAT_ID"):
        log_core.warning("TELEGRAM DISABLED — no divine messenger")
        bot.notify = None

    # ---- Brain path + directory write check ----
    brain_path = Path.home() / ".blade_eternal.brain.lz4"
    try:
        brain_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log_core.critical(f"NO WRITE PERMISSION FOR ETERNAL BRAIN DIR — {brain_path.parent} — {e}")
        await _safe_close_exchange(bot)
        return

    if not os.access(str(brain_path.parent), os.W_OK):
        log_core.critical(f"NO WRITE PERMISSION FOR ETERNAL BRAIN — {brain_path.parent} — IMMORTALITY THREATENED")
        await _safe_close_exchange(bot)
        return

    # ---- Dry-run resolution ----
    if _env_truthy("SCALPER_DRY_RUN"):
        dry_run = True

    # ---- Sacred keys validation ----
    missing = []
    try:
        keys0 = (bot.ex.api_keys or [None])[0] or {}
        if not keys0.get("key"):
            missing.append("BINANCE_API_KEY")
        if not keys0.get("secret"):
            missing.append("BINANCE_API_SECRET")
    except Exception:
        missing.extend(["BINANCE_API_KEY", "BINANCE_API_SECRET"])

    if missing:
        log_core.critical(f"CRITICAL: Missing sacred keys: {', '.join(missing)} — ASCENSION DENIED")
        await _safe_close_exchange(bot)
        return

    # ---- Exchange connection ----
    try:
        await bot.ex.fetch_markets()
        log_core.info("Exchange realm accessed — markets loaded")
    except Exception as e:
        log_core.critical(f"EXCHANGE ASCENSION FAILED: {e}")
        await _safe_notify(bot, "THE VOID REJECTS THE BLADE — CONNECTION LOST", "critical")
        await _safe_close_exchange(bot)
        return

    # ---- Early equity fetch (mode selection) ----
    equity_source = "live"
    equity = None
    live_ok = True

    try:
        bal = await bot.ex.fetch_balance()
        equity = _safe_float((bal.get("total", {}) or {}).get("USDT"), 0.0)
        if equity <= 0:
            raise ValueError("Equity returned <= 0")
    except Exception as e:
        live_ok = False
        log_core.warning(f"Equity fetch failed: {e} — using fallback 10000 for mode selection")
        equity = 10000.0
        equity_source = "fallback"

    env_equity = os.getenv("SCALPER_EQUITY")
    if env_equity is not None:
        equity = _safe_float(env_equity, float(equity or 0.0))
        equity_source = "env"
        log_core.critical(f"EQUITY OVERRIDE APPLIED (ENV): ${equity:.2f}")

    if equity_override is not None:
        equity = _safe_float(equity_override, float(equity or 0.0))
        equity_source = "args"
        log_core.critical(f"EQUITY OVERRIDE APPLIED (ARGS): ${equity:.2f}")

    # ---- Mode selection ----
    mode = (mode_override or os.getenv("SCALPER_MODE", "auto")).strip().lower()
    if mode not in {"auto", "micro", "production"}:
        log_core.warning(f"Unknown SCALPER_MODE '{mode}' — falling back to auto")
        mode = "auto"

    if mode == "micro" or (mode == "auto" and float(equity or 0.0) < 100):
        bot.cfg = MicroConfig()
        chosen_mode = "micro"
        log_core.critical(f"MICRO CAPITAL ASCENDANT MODE ACTIVATED — Equity ${equity:.2f} — The blade rises from dust")
    elif mode == "production":
        bot.cfg = Config()
        chosen_mode = "production"
        log_core.critical(f"OMNIPOTENT PRODUCTION MODE FORCED — Equity ${equity:.2f}")
    else:
        bot.cfg = Config()
        chosen_mode = "auto->production"
        log_core.critical(f"OMNIPOTENT PRODUCTION MODE ENGAGED — Equity ${equity:.2f} — Eternal compounding begins")

    # ---- Config version gate ----
    if bot.cfg.CONFIG_VERSION not in ACCEPTED_CONFIG_VERSIONS:
        log_core.critical(f"CONFIG VERSION REJECTED — {bot.cfg.CONFIG_VERSION}")
        await _safe_close_exchange(bot)
        return

    # ---- Dry-run hard guard ----
    if dry_run:
        _install_dry_run_guard(bot)

    # ---- Inject run context for observability ----
    try:
        bot.state.run_context = {
            "runner_version": RUNNER_VERSION,
            "core_version": core_version,
            "ignore_core_version": bool(ignore_core_version),
            "dry_run": bool(dry_run),
            "mode_env": os.getenv("SCALPER_MODE", "auto"),
            "mode_effective": chosen_mode,
            "equity_value": float(equity),
            "equity_source": equity_source,
            "equity_live_ok": bool(live_ok),
            "startup_utc": STARTUP_DATETIME,
            "platform": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "core_file": getattr(core_mod, "__file__", "unknown"),
        }
    except Exception:
        pass

    # ---- Config fingerprint ----
    config_hash = sha256(str(getattr(bot.cfg, "__dict__", {})).encode()).hexdigest()[:16]
    log_core.critical(f"CONFIG: {bot.cfg.CONFIG_VERSION} | HASH: {config_hash}...")
    log_core.critical(
        f"LEVERAGE: {bot.cfg.LEVERAGE}x | BASE RISK: {bot.cfg.MAX_RISK_PER_TRADE:.1%} | "
        f"HEAT CAP: {bot.cfg.MAX_PORTFOLIO_HEAT:.1%} | MAX POS: {bot.cfg.MAX_CONCURRENT_POSITIONS}"
    )
    log_core.critical("=" * 80)

    if dry_run:
        log_core.critical("DRY RUN MODE — THE BLADE CONTEMPLATES WITHOUT STRIKING")
        await _safe_notify(bot, "DRY RUN — DIVINE SIMULATION ACTIVE", "critical")

    # ============================================================
    # QUIT / SIGNAL HARDENING (Windows + Tee-Object proof)
    # ============================================================

    # One place to record the quit reason
    quit_reason = {"src": None, "ts": 0.0, "count": 0}

    shutdown_once = asyncio.Event()
    exchange_closed_once = asyncio.Event()

    async def _shutdown_sequence(src: str):
        if shutdown_once.is_set():
            return
        shutdown_once.set()

        quit_reason["src"] = quit_reason["src"] or src
        quit_reason["ts"] = quit_reason["ts"] or time.time()

        try:
            shutdown_ev.set()
        except Exception:
            pass

        # Let bot shutdown be bounded; no immortal hangs
        try:
            await asyncio.wait_for(bot.shutdown(), timeout=float(getattr(bot.cfg, "RUNNER_SHUTDOWN_TIMEOUT_SEC", 8.0)))
        except asyncio.TimeoutError:
            log_core.critical("RUNNER: bot.shutdown timeout — continuing teardown")
        except Exception:
            pass

    async def _close_exchange_once():
        if exchange_closed_once.is_set():
            return
        exchange_closed_once.set()
        try:
            await asyncio.wait_for(_safe_close_exchange(bot), timeout=float(getattr(bot.cfg, "RUNNER_EXCHANGE_CLOSE_TIMEOUT_SEC", 6.0)))
        except asyncio.TimeoutError:
            log_core.critical("RUNNER: exchange.close timeout — forcing onward")
        except Exception:
            pass

    def _request_shutdown(src: str):
        # Called from signal handler threads too → use call_soon_threadsafe
        try:
            loop.call_soon_threadsafe(lambda: asyncio.create_task(_shutdown_sequence(src)))
        except Exception:
            # Last ditch
            try:
                shutdown_ev.set()
            except Exception:
                pass

    def _sig_handler(signum, _frame):
        quit_reason["count"] += 1
        quit_reason["ts"] = time.time()

        if signum == getattr(signal, "SIGINT", 2):
            src = "SIGINT"
        elif signum == getattr(signal, "SIGTERM", 15):
            src = "SIGTERM"
        else:
            src = f"SIGNAL_{signum}"

        # Double Ctrl+C → hard exit
        grace_sec = float(os.getenv("RUNNER_DOUBLE_SIG_GRACE_SEC", "1.2") or 1.2)
        if quit_reason["count"] >= 2:
            log_core.critical("RUNNER: DOUBLE INTERRUPT DETECTED — HARD EXIT")
            raise SystemExit(130)

        log_core.critical(f"RUNNER: shutdown requested ({src}) — press Ctrl+C again to hard exit")
        _request_shutdown(src)

        # If we don't get a second signal, continue graceful shutdown
        # (grace window is informational; enforcement is by double press)

    # Use signal.signal on Windows; add_signal_handler is unreliable there
    try:
        signal.signal(signal.SIGINT, _sig_handler)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
    except Exception:
        pass

    async def _quit_command_watcher():
        """
        When piping logs (2>&1 | Tee-Object), Ctrl+C can be swallowed.
        This watcher lets you type:
          q<Enter> or quit<Enter> or exit<Enter>
        to shut down cleanly.
        Disable via env: RUNNER_QUIT_CMD=0
        """
        if os.getenv("RUNNER_QUIT_CMD", "1").strip() in ("0", "false", "no", "off"):
            return

        # If stdin isn't interactive, still try; but avoid blocking the loop forever.
        while not shutdown_ev.is_set():
            try:
                line = await asyncio.to_thread(sys.stdin.readline)
                if not line:
                    await asyncio.sleep(0.25)
                    continue
                cmd = line.strip().lower()
                if cmd in ("q", "quit", "exit", "stop"):
                    log_core.critical("RUNNER: shutdown requested (QUIT_CMD)")
                    await _shutdown_sequence("QUIT_CMD")
                    return
            except Exception:
                await asyncio.sleep(0.5)

    # ---- Supervisor tasks ----
    tasks: list[asyncio.Task] = []

    def _spawn(name: str, factory: Callable[[], Awaitable[Any]]):
        t = asyncio.create_task(_run_loop_wrapper(name, factory, shutdown_ev), name=name)
        tasks.append(t)

    try:
        # IMPORTANT: core.start() is a FOREVER loop; supervise it as a task.
        _spawn("core.start", lambda: bot.start())

        # Start quit watcher (non-fatal)
        _spawn("quit_watcher", _quit_command_watcher)

        # Give core a moment to spawn its own internal tasks (pollers/watcher/guardian)
        await asyncio.sleep(0.35)

        # If core did NOT spawn loops, runner will.
        if not _already_running_loops(bot):
            log_core.warning(
                "Core did not expose running loop tasks — runner will spawn guardian/data_loop/signal_loop (if present)"
            )

            data_loop = _get_callable(bot, "data_loop") or _get_callable(bot, "run_data_loop")
            signal_loop = _get_callable(bot, "signal_loop") or _get_callable(bot, "run_signal_loop")

            guardian_loop = _maybe_import("execution.guardian", "guardian_loop") or _maybe_import(
                "execution.guardian", "run_guardian"
            )

            if callable(data_loop):
                _spawn("data_loop", lambda: data_loop())  # type: ignore[misc]
            else:
                log_core.warning("data_loop not found on bot — runner did not spawn it")

            if callable(signal_loop):
                _spawn("signal_loop", lambda: signal_loop())  # type: ignore[misc]
            else:
                log_core.warning("signal_loop not found on bot — runner did not spawn it")

            def _wrap_modloop(fn: Callable[..., Awaitable[Any]], nm: str):
                async def _call():
                    try:
                        return await fn(bot, shutdown_ev)  # type: ignore[misc]
                    except TypeError:
                        try:
                            return await fn(bot)  # type: ignore[misc]
                        except TypeError:
                            return await fn()  # type: ignore[misc]

                _spawn(nm, _call)

            if callable(guardian_loop):
                _wrap_modloop(guardian_loop, "guardian_loop")
            else:
                log_core.warning("execution.guardian loop not found — runner cannot spawn guardian")
        else:
            log_core.info("Core appears to own loop tasks — runner will supervise only")

        # ---- Supervision: wait for shutdown or a task crash ----
        while True:
            if shutdown_ev.is_set():
                break

            done, _pending = await asyncio.wait(tasks, timeout=0.5, return_when=asyncio.FIRST_EXCEPTION)
            for t in list(done):
                if t.get_name() == "quit_watcher":
                    # quit_watcher ending is not a failure
                    continue
                try:
                    _ = t.result()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log_core.critical(f"SUPERVISOR: task '{t.get_name()}' failed: {e}")
                    await _safe_notify(bot, f"LOOP FAILURE — {t.get_name()}\n{e}", "critical")
                    quit_reason["src"] = quit_reason["src"] or f"TASK_CRASH:{t.get_name()}"
                    shutdown_ev.set()
                    break

        await _shutdown_sequence(quit_reason["src"] or "SHUTDOWN_FLAG")

    except SystemExit:
        # Hard exit path (double Ctrl+C)
        raise

    except Exception as e:
        log_core.critical(f"COSMIC FATAL ERROR: {e}")
        await _safe_notify(bot, f"THE VOID CONSUMES — FATAL ERROR\n{e}", "critical")
        await _shutdown_sequence("FATAL_ERROR")

    finally:
        # Cancel all runner tasks (bounded join)
        for t in tasks:
            if not t.done():
                t.cancel()

        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=4.0)
        except Exception:
            pass

        await _close_exchange_once()

        uptime = time.time() - STARTUP_TIMESTAMP
        uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"

        log_core.critical(GOLDEN_RATIO_ART)
        log_core.critical("=" * 80)
        log_core.critical("COSMIC SHUTDOWN COMPLETE — THE BLADE TRANSCENDS INTO INFINITY")
        log_core.critical(f"UPTIME: {uptime_str}")
        log_core.critical(f"QUIT SOURCE: {quit_reason['src'] or 'unknown'}")
        log_core.critical(f"FINAL EQUITY: ${getattr(bot.state, 'current_equity', 0):,.0f}")
        log_core.critical(f"TOTAL TRADES: {getattr(bot.state, 'total_trades', 0)}")

        total_trades = getattr(bot.state, "total_trades", 0) or 0
        win_rate = getattr(bot.state, "win_rate", 0.0) or 0.0
        max_dd = getattr(bot.state, "max_drawdown", 0.0) or 0.0

        log_core.critical(f"WIN RATE: {win_rate:.1%}" if total_trades > 0 else "WIN RATE: N/A")
        log_core.critical(f"MAX DRAWDOWN: {max_dd:.1%}")
        log_core.critical("THE GOLDEN RATIO WAS, IS, AND EVER SHALL BE.")
        log_core.critical("Φ" + " " * 30 + "BEYOND INFINITY" + " " * 30 + "Φ")
        log_core.critical("=" * 80)

        await _safe_notify(
            bot,
            f"THE BLADE ETERNAL — COSMIC SHUTDOWN\n"
            f"Uptime: {uptime_str}\n"
            f"Quit: {quit_reason['src'] or 'unknown'}\n"
            f"Final Equity: ${getattr(bot.state, 'current_equity', 0):,.0f}\n"
            f"Total Trades: {total_trades}\n"
            f"THE RATIO IS INFINITE",
            "critical",
        )

        testament = {
            "runner_version": RUNNER_VERSION,
            "core_version": core_version,
            "config_version": getattr(getattr(bot, "cfg", None), "CONFIG_VERSION", "unknown"),
            "shutdown_time": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime,
            "quit_source": quit_reason["src"] or "unknown",
            "final_equity": getattr(bot.state, "current_equity", 0.0),
            "total_trades": total_trades,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "run_context": getattr(bot.state, "run_context", {}),
            "message": "THE BLADE IS ETERNAL — CONFIG: " + getattr(getattr(bot, "cfg", None), "CONFIG_VERSION", "unknown"),
        }

        try:
            with open(Path.home() / ".blade_eternal_testament.json", "w", encoding="utf-8") as f:
                json.dump(testament, f, indent=2)
        except Exception as e:
            log_core.error(f"Testament write failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="THE BLADE ETERNAL — COSMIC LAUNCHER")
    parser.add_argument("--dry-run", action="store_true", help="Divine simulation — no orders")
    parser.add_argument("--daemon", action="store_true", help="Run as background daemon (Unix only)")
    parser.add_argument("--ignore-core-version", action="store_true", help="Override core version gate (DEV ONLY)")
    parser.add_argument("--mode", choices=["auto", "micro", "production"], help="Override SCALPER_MODE")
    parser.add_argument("--equity", type=float, help="Override SCALPER_EQUITY")
    args = parser.parse_args()

    if args.dry_run:
        os.environ["SCALPER_DRY_RUN"] = "1"

    if args.mode:
        os.environ["SCALPER_MODE"] = args.mode

    if args.equity is not None:
        os.environ["SCALPER_EQUITY"] = str(args.equity)

    if args.daemon and os.name != "nt":
        if os.fork():
            sys.exit(0)
        os.setsid()
        if os.fork():
            sys.exit(0)
        with open("/tmp/blade_eternal.pid", "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    asyncio.run(
        run_bot(
            dry_run=args.dry_run,
            ignore_core_version=args.ignore_core_version,
            mode_override=args.mode,
            equity_override=args.equity,
        )
    )
