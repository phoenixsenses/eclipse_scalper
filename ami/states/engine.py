"""StateEngine — multi-timeframe state construction from microstructure.db (read-only).

Phase 1 (whitepaper §91): unified state objects, 1m..1W, data-health propagation,
hand-defined taxonomy v0. Rule-based; ML latent states come in Phase 6.
"""
from __future__ import annotations
import math, sqlite3, time
from pathlib import Path

from ami.enums import CascadePhase, DataQuality, StateFamily, StructurePhase, TIMEFRAMES, TIMEFRAME_MS
from ami.states.objects import StateBundle, StateObject

DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "microstructure.db"

# feed -> max healthy age (min)
FEED_LIMITS = {"mark_prices": 5.0, "liquidations": 120.0, "book_ticker": 5.0,
               "agg_trades": 5.0, "open_interest": 10.0, "spot_prices": 10.0, "vol_state": 10.0}


class StateEngine:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    # ---- helpers ----
    def _px(self, sym: str, ts: int) -> float | None:
        r = self.conn.execute("SELECT mark_price FROM mark_prices WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                              (sym, ts)).fetchone()
        return float(r[0]) if r else None

    def _ret_bps(self, sym: str, ts: int, lb_ms: int) -> float | None:
        a = self._px(sym, ts - lb_ms); b = self._px(sym, ts)
        return (b - a) / a * 1e4 if a and b and a > 0 else None

    def _rv(self, sym: str, ts: int, bar_ms: int, nbars: int = 12) -> float | None:
        px = []
        for k in range(nbars, -1, -1):
            p = self._px(sym, ts - k * bar_ms)
            if p is None or p <= 0:
                return None
            px.append(p)
        rets = [math.log(px[i + 1] / px[i]) for i in range(nbars)]
        return math.sqrt(sum(x * x for x in rets) / nbars)

    def feed_ages_min(self, now_ms: int | None = None) -> dict[str, float | None]:
        now = now_ms or int(time.time() * 1000)
        out: dict[str, float | None] = {}
        for t in FEED_LIMITS:
            try:
                r = self.conn.execute(f"SELECT MAX(ts_ms) FROM {t} WHERE symbol='ETHUSDT'").fetchone()
                out[t] = round((now - int(r[0])) / 60_000, 1) if r and r[0] else None
            except sqlite3.OperationalError:
                out[t] = None
        return out

    def data_quality(self, ages: dict[str, float | None]) -> dict[str, DataQuality]:
        out = {}
        for feed, age in ages.items():
            lim = FEED_LIMITS[feed]
            if age is None:
                out[feed] = DataQuality.UNAVAILABLE
            elif age <= lim:
                out[feed] = DataQuality.HEALTHY
            elif age <= 10 * lim:
                out[feed] = DataQuality.DEGRADED
            else:
                out[feed] = DataQuality.STALE
        return out

    # ---- structure phase v0 (rule-based, per timeframe) ----
    def _structure(self, sym: str, ts: int, tf: str) -> tuple[StructurePhase, str, float]:
        bar = TIMEFRAME_MS[tf]
        ret1 = self._ret_bps(sym, ts, bar) or 0.0
        ret3 = self._ret_bps(sym, ts, 3 * bar) or 0.0
        ret12 = self._ret_bps(sym, ts, 12 * bar) or 0.0
        rv_now = self._rv(sym, ts, bar, 6) or 0.0
        rv_prev = self._rv(sym, ts - 6 * bar, bar, 6) or 0.0
        vol_exp = rv_prev > 0 and rv_now / rv_prev > 1.5
        vol_comp = rv_prev > 0 and rv_now / rv_prev < 0.6
        direction = "UP" if ret3 > abs(ret12) * 0.15 and ret3 > 0 else \
                    ("DOWN" if ret3 < -abs(ret12) * 0.15 and ret3 < 0 else "FLAT")
        if vol_comp and abs(ret3) < 30:
            ph = StructurePhase.COMPRESSION
        elif vol_exp and ret1 < -50:
            ph = StructurePhase.BREAKDOWN
        elif vol_exp and ret1 > 50:
            ph = StructurePhase.EARLY_EXPANSION
        elif ret12 < -150 and ret3 > 0:
            ph = StructurePhase.RECOVERY
        elif ret12 > 150 and ret3 < 0:
            ph = StructurePhase.DISTRIBUTION
        elif abs(ret12) < 50 and abs(ret3) < 30:
            ph = StructurePhase.RANGE
        elif ret12 > 150 and ret3 > 0:
            ph = StructurePhase.MATURE_TREND
        elif ret12 < -150 and ret3 < 0:
            ph = StructurePhase.EXPANSION  # bearish expansion
        else:
            ph = StructurePhase.RANGE
        conf = min(0.9, 0.4 + abs(ret3) / 400 + (0.15 if vol_exp or vol_comp else 0))
        return ph, direction, round(conf, 2)

    # ---- cascade state ----
    def _cascade(self, ts: int) -> tuple[CascadePhase, dict]:
        s5 = self.conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<?",
                               (ts - 300_000, ts)).fetchone()[0] or 0.0
        s30 = self.conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<?",
                                (ts - 1_800_000, ts)).fetchone()[0] or 0.0
        s2h = self.conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations WHERE symbol='ETHUSDT' AND side='SELL' AND ts_ms>=? AND ts_ms<?",
                                (ts - 7_200_000, ts)).fetchone()[0] or 0.0
        meta = {"sell_5m": round(s5, 0), "sell_30m": round(s30, 0), "sell_2h": round(s2h, 0)}
        if s5 >= 200_000:
            return (CascadePhase.ACCELERATION if s5 > s30 - s5 else CascadePhase.EARLY_CASCADE), meta
        if s30 >= 200_000:
            return CascadePhase.EXHAUSTION, meta
        if s2h >= 200_000:
            return CascadePhase.RECOVERY, meta
        return CascadePhase.NONE, meta

    # ---- leverage/funding state ----
    def _leverage(self, ts: int) -> tuple[str, dict]:
        r = self.conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",
                              (ts,)).fetchone()
        f0 = float(r[0]) if r and r[0] is not None else None
        r1 = self.conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",
                               (ts - 3_600_000,)).fetchone()
        f1 = float(r1[0]) if r1 and r1[0] is not None else None
        vel = (f0 - f1) if (f0 is not None and f1 is not None) else None
        oi = self.conn.execute("SELECT open_interest_usd, ts_ms FROM open_interest WHERE symbol='ETHUSDT' ORDER BY ts_ms DESC LIMIT 1").fetchone()
        oi_usd = float(oi[0]) if oi else None
        oi_fresh = oi and (ts - int(oi[1])) < 30 * 60_000
        label = "NEUTRAL"
        if f0 is not None:
            if f0 < 0 and (vel or 0) <= 0:
                label = "SHORT_CROWDED_BUILDING"
            elif f0 < 0:
                label = "SHORT_CROWDED_EASING"
            elif f0 > 0.0001:
                label = "LONG_CROWDED"
        return label, {"funding": f0, "funding_vel_1h": vel,
                       "oi_usd": oi_usd if oi_fresh else None, "oi_fresh": bool(oi_fresh)}

    # ---- book state ----
    def _book(self, ts: int) -> tuple[str, dict]:
        r = self.conn.execute("SELECT bid_qty, ask_qty, spread_pct, book_imbalance, ts_ms FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                              (ts,)).fetchone()
        if not r or (ts - int(r[4])) > 5 * 60_000:
            return "UNKNOWN", {}
        pre = self.conn.execute("SELECT AVG(bid_qty) FROM book_ticker WHERE symbol='ETHUSDT' AND ts_ms>=? AND ts_ms<?",
                                (ts - 600_000, ts - 60_000)).fetchone()
        base = float(pre[0]) if pre and pre[0] else None
        bidq = float(r[0])
        ratio = bidq / base if base and base > 0 else 1.0
        label = "BOOK_PULLS" if ratio < 0.5 else ("BOOK_THICKENS" if ratio > 1.6 else "BOOK_STABLE")
        return label, {"bid_qty": bidq, "bid_vs_10m": round(ratio, 2),
                       "spread_pct": float(r[2]) if r[2] is not None else None,
                       "imbalance": float(r[3]) if r[3] is not None else None}

    # ---- main ----
    def build_bundle(self, sym: str = "ETHUSDT", ts_ms: int | None = None) -> StateBundle:
        ts = ts_ms or int(time.time() * 1000)
        ages = self.feed_ages_min(ts)
        dq = self.data_quality(ages)
        mark_q = dq.get("mark_prices", DataQuality.UNAVAILABLE)
        states: list[StateObject] = []
        # structure per timeframe
        for tf in TIMEFRAMES:
            ph, direction, conf = self._structure(sym, ts, tf)
            states.append(StateObject(
                state_id=f"ST:{sym}:{tf}:{ts}", entity=sym, timeframe=tf,
                family=StateFamily.STRUCTURE_STATE, label=ph.value, start_ms=ts,
                confidence=conf if mark_q == DataQuality.HEALTHY else conf * 0.5,
                data_quality=mark_q, meta={"direction": direction}))
        # cascade
        cph, cmeta = self._cascade(ts)
        states.append(StateObject(
            state_id=f"CS:{sym}:{ts}", entity=sym, timeframe="event",
            family=StateFamily.CASCADE_STATE, label=cph.value, start_ms=ts,
            confidence=0.8 if dq.get("liquidations") == DataQuality.HEALTHY else 0.4,
            data_quality=dq.get("liquidations", DataQuality.UNAVAILABLE), meta=cmeta))
        # leverage
        lab, lmeta = self._leverage(ts)
        states.append(StateObject(
            state_id=f"LV:{sym}:{ts}", entity=sym, timeframe="8h",
            family=StateFamily.LEVERAGE_STATE, label=lab, start_ms=ts,
            confidence=0.7, data_quality=dq.get("open_interest", DataQuality.UNAVAILABLE)
            if lmeta.get("oi_fresh") else DataQuality.DEGRADED, meta=lmeta))
        # book
        blab, bmeta = self._book(ts)
        states.append(StateObject(
            state_id=f"BK:{sym}:{ts}", entity=sym, timeframe="1m",
            family=StateFamily.BOOK_STATE, label=blab, start_ms=ts,
            confidence=0.6, data_quality=dq.get("book_ticker", DataQuality.UNAVAILABLE), meta=bmeta))
        # knowledge/data-health state
        states.append(StateObject(
            state_id=f"DH:{ts}", entity="SYSTEM", timeframe="event",
            family=StateFamily.KNOWLEDGE_STATE, label="DATA_HEALTH", start_ms=ts,
            confidence=1.0, data_quality=DataQuality.HEALTHY,
            meta={"feed_ages_min": ages, "quality": {k: v.value for k, v in dq.items()}}))
        return StateBundle(ts_ms=ts, states=states)

    def close(self) -> None:
        self.conn.close()
