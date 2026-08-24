"""OI + Spot Poller — Faz 5 (mekanizma arastirmasi veri altyapisi).

Olu tablolari canlandirir:
- open_interest  (producer 2026-04-18'de olmus)  <- fapi/v1/openInterest
- spot_prices    (producer 2026-06-05'te olmus)  <- api/v3/ticker/price

60 saniyede bir 3 sembol icin poll eder, ana DB'ye yazar (WAL, busy_timeout).
Public endpoint — API key gerekmez. Hafif: dakikada 4 HTTP istegi.

Kullanim:
  python -m data.oi_spot_poller --once      # tek tur (test)
  python -m data.oi_spot_poller             # surekli (60s)
"""
from __future__ import annotations
import argparse, json, sqlite3, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from ami.storage.union_reader import resolve_writer_db_path

DB = ROOT / "data" / "microstructure.db"
PID_PATH = ROOT / "logs" / "pids" / "oi_spot_poller.pid"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL_SEC = 60
FAPI_OI = "https://fapi.binance.com/fapi/v1/openInterest?symbol={sym}"
SPOT_PX = 'https://api.binance.com/api/v3/ticker/price?symbols=%5B%22BTCUSDT%22,%22ETHUSDT%22,%22SOLUSDT%22%5D'


def http_json(url: str, timeout: int = 10):
    req = urllib.request.Request(url, headers={"User-Agent": "eclipse-poller/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def poll_once(conn: sqlite3.Connection) -> tuple[int, int]:
    ts = int(time.time() * 1000)
    n_oi = n_sp = 0
    # spot (tek istek, 3 sembol)
    try:
        for row in http_json(SPOT_PX):
            sym = row.get("symbol"); px = row.get("price")
            if sym in SYMBOLS and px is not None:
                conn.execute("INSERT INTO spot_prices (ts_ms, symbol, spot_price) VALUES (?,?,?)",
                             (ts, sym, float(px)))
                n_sp += 1
    except Exception as e:
        print(f"{utc_now()} SPOT_ERR {e}", flush=True)
    # OI (sembol basi istek) — USD = kontrat * son mark
    for sym in SYMBOLS:
        try:
            d = http_json(FAPI_OI.format(sym=sym))
            oi = float(d.get("openInterest") or 0)
            mk = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? ORDER BY ts_ms DESC LIMIT 1", (sym,)).fetchone()
            usd = oi * float(mk[0]) if mk and mk[0] else None
            if usd:
                conn.execute("INSERT INTO open_interest (ts_ms, symbol, open_interest_usd) VALUES (?,?,?)",
                             (ts, sym, usd))
                n_oi += 1
        except Exception as e:
            print(f"{utc_now()} OI_ERR {sym} {e}", flush=True)
    conn.commit()
    return n_oi, n_sp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()
    # DB above is the PRE-ROTATION path, deleted in Phase-4; opening it would
    # silently create an empty database and write a feed nobody reads.
    args.db = str(resolve_writer_db_path(args.db))
    try:
        PID_PATH.parent.mkdir(parents=True, exist_ok=True)
        PID_PATH.write_text(str(__import__("os").getpid()), encoding="ascii")
    except Exception:
        pass
    conn = sqlite3.connect(args.db, timeout=30)
    conn.execute("PRAGMA busy_timeout=30000")
    print(f"{utc_now()} OI+Spot poller started  db={args.db}  symbols={SYMBOLS}", flush=True)
    while True:
        t0 = time.time()
        n_oi, n_sp = poll_once(conn)
        print(f"{utc_now()} tick oi={n_oi} spot={n_sp}", flush=True)
        if args.once:
            break
        time.sleep(max(1.0, INTERVAL_SEC - (time.time() - t0)))
    conn.close()


if __name__ == "__main__":
    main()
