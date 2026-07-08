"""Trade Lifecycle Engine — Phase 3 (whitepaper §93, Part VIII §42-44).

- classify_lifecycle_path: 1m pnl path -> lifecycle state sequence
- LifecycleEngine.replay_shadow_ledger: real closed shadow trades -> lifecycle
  dataset + MFE milestone outcome distribution (MFE State Classifier v0 data)
"""
from __future__ import annotations
import json, sqlite3
from pathlib import Path

from ami.enums import TradeLifecycleState as TLS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "reports" / "shadow" / "s34_state_machine_shadow.jsonl"
DEFAULT_DB = ROOT / "data" / "microstructure.db"


def classify_lifecycle_path(pnl_1m: list[float]) -> list[tuple[int, str]]:
    """1m net-bps path (from entry) -> [(minute, state)]. Rule taxonomy v0."""
    out: list[tuple[int, str]] = [(0, TLS.OPEN.value)]
    peak = 0.0
    for k in range(1, len(pnl_1m)):
        p = pnl_1m[k]; peak = max(peak, p)
        prev = pnl_1m[max(0, k - 15)]
        slope15 = p - prev
        if p <= -100:
            st = TLS.INVALIDATED
        elif peak >= 50 and p <= peak - 0.6 * max(peak, 1) and p < 25:
            st = TLS.REVERSING if p < 0 else TLS.WEAKENING
        elif slope15 > 25 and p > 0:
            st = TLS.ACCELERATING
        elif p >= 25 and abs(slope15) <= 10:
            st = TLS.STALLING if peak - p > 25 else TLS.HEALTHY
        elif p < 0 and slope15 > 10:
            st = TLS.RECOVERING
        elif p < -25 and slope15 < 0:
            st = TLS.WEAKENING
        elif p >= 0:
            st = TLS.HEALTHY
        else:
            st = TLS.EXHAUSTED if p < -50 else TLS.WEAKENING
        if st.value != out[-1][1]:
            out.append((k, st.value))
    out.append((len(pnl_1m) - 1, TLS.CLOSED.value))
    return out


class LifecycleEngine:
    def __init__(self, ledger_path: str | Path = DEFAULT_LEDGER, db_path: str | Path = DEFAULT_DB):
        self.ledger_path = Path(ledger_path)
        self.db_path = Path(db_path)

    def _mark_path_1m(self, conn, entry_ts: int, entry_px: float, minutes: int) -> list[float] | None:
        if entry_px <= 0:
            return None
        out = []
        for k in range(minutes + 1):
            r = conn.execute("SELECT mark_price FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                             (entry_ts + k * 60_000,)).fetchone()
            if not r:
                return None
            out.append((float(r[0]) - entry_px) / entry_px * 1e4)
        return out

    def replay_shadow_ledger(self, signals: tuple[str, ...] = ("LONG_HOUR17_HOLD6H", "LONG_HOUR17_COMPOSITE",
                                                               "LONG_HOUR17_100K_COMPOSITE", "LONG_SILENCE"),
                             minutes: int = 360, limit: int = 300) -> dict:
        """Closed shadow trades -> lifecycle sequences + MFE milestone stats."""
        rows = []
        if self.ledger_path.exists():
            with self.ledger_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("event") == "CLOSE" and r.get("signal") in signals \
                            and r.get("entry_ts_ms") and r.get("entry_price") and r.get("direction") == "LONG":
                        rows.append(r)
        rows = rows[-limit:]
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        seqs = []; milestone = {50: {"A_continue": 0, "B_breakeven": 0, "C_negative": 0, "D_time_pos": 0}}
        state_minutes: dict[str, int] = {}
        for r in rows:
            path = self._mark_path_1m(conn, int(r["entry_ts_ms"]), float(r["entry_price"]), minutes)
            if not path:
                continue
            seq = classify_lifecycle_path(path)
            seqs.append({"id": r.get("id"), "signal": r.get("signal"),
                         "net_bps": r.get("net_bps"), "sequence": seq})
            for i, (m, st) in enumerate(seq):
                nxt = seq[i + 1][0] if i + 1 < len(seq) else minutes
                state_minutes[st] = state_minutes.get(st, 0) + (nxt - m)
            # MFE milestone: reached +50?
            hit = next((k for k, p in enumerate(path) if p >= 50), None)
            if hit is not None:
                rest = path[hit:]
                final = path[-1]
                if max(rest) >= 100:
                    milestone[50]["A_continue"] += 1
                elif final <= 5 and final > -5:
                    milestone[50]["B_breakeven"] += 1
                elif final < -5:
                    milestone[50]["C_negative"] += 1
                else:
                    milestone[50]["D_time_pos"] += 1
        conn.close()
        n50 = sum(milestone[50].values())
        return {"n_trades": len(seqs), "state_minutes": dict(sorted(state_minutes.items(), key=lambda kv: -kv[1])),
                "mfe50_outcomes": milestone[50], "mfe50_n": n50,
                "sequences_sample": seqs[:5]}
