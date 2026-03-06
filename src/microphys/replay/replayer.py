from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x != x:  # NaN
            return float(default)
        return x
    except Exception:
        return float(default)


def _to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _norm_ts_seconds(v: Any) -> float:
    ts = _to_float(v, 0.0)
    if ts <= 0.0:
        return 0.0
    # Accept ms/us/ns and normalize to seconds.
    if ts > 1e18:
        return ts / 1e9
    if ts > 1e15:
        return ts / 1e6
    if ts > 1e12:
        return ts / 1e3
    return ts


def _pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return default


@dataclass(frozen=True)
class ReplayMatch:
    symbol: str
    side: str
    sim_event_id: str
    sim_ts: float
    live_ts: float
    dt_sec: float
    sim_filled: bool
    live_filled: bool
    sim_fill_delay_sec: float
    live_fill_delay_sec: float
    fill_delay_delta_sec: float
    sim_pnl_bps: float
    live_pnl_bps: float
    pnl_bps_delta: float
    sim_max_adverse_bps: float
    live_max_adverse_bps: float
    adverse_bps_delta: float


@dataclass(frozen=True)
class ReplayParityResult:
    sim_count: int
    live_count: int
    matched_count: int
    unmatched_sim_count: int
    unmatched_live_count: int
    match_rate_vs_sim: float
    sim_fill_rate: float
    live_fill_rate: float
    fill_rate_delta: float
    mean_dt_sec: float
    mean_abs_dt_sec: float
    mean_fill_delay_delta_sec: float
    mean_pnl_bps_delta: float
    mean_adverse_bps_delta: float
    matches: List[ReplayMatch]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sim_count": int(self.sim_count),
            "live_count": int(self.live_count),
            "matched_count": int(self.matched_count),
            "unmatched_sim_count": int(self.unmatched_sim_count),
            "unmatched_live_count": int(self.unmatched_live_count),
            "match_rate_vs_sim": float(self.match_rate_vs_sim),
            "sim_fill_rate": float(self.sim_fill_rate),
            "live_fill_rate": float(self.live_fill_rate),
            "fill_rate_delta": float(self.fill_rate_delta),
            "mean_dt_sec": float(self.mean_dt_sec),
            "mean_abs_dt_sec": float(self.mean_abs_dt_sec),
            "mean_fill_delay_delta_sec": float(self.mean_fill_delay_delta_sec),
            "mean_pnl_bps_delta": float(self.mean_pnl_bps_delta),
            "mean_adverse_bps_delta": float(self.mean_adverse_bps_delta),
            "matches": [asdict(m) for m in self.matches],
        }


def _normalize_sim_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(_pick(raw, "symbol", default="")).upper()
    side = str(_pick(raw, "side", "resolved_side", default="")).upper()
    event_id = str(_pick(raw, "event_id", "id", "trade_id", "order_id", default=""))
    ts = _norm_ts_seconds(_pick(raw, "entry_time", "entry_ts", "entry_ts_utc", "ts", "ts_ms", "event_ts"))
    filled = _to_bool(_pick(raw, "filled", default=True), default=True)
    fill_delay_sec = _to_float(_pick(raw, "fill_delay_sec"), 0.0)
    if fill_delay_sec <= 0.0:
        fill_delay_bars = _to_float(_pick(raw, "fill_delay_bars"), 0.0)
        bucket_sec = _to_float(_pick(raw, "bucket_sec"), 1.0)
        fill_delay_sec = max(0.0, fill_delay_bars * max(1e-9, bucket_sec))
    return {
        "event_id": event_id,
        "symbol": symbol,
        "side": side,
        "ts": float(ts),
        "filled": bool(filled),
        "fill_delay_sec": float(fill_delay_sec),
        "pnl_bps": _to_float(_pick(raw, "pnl_bps", "pnl_net_bps", "net_return_bps"), 0.0),
        "max_adverse_bps": _to_float(_pick(raw, "max_adverse_bps", "adverse_bps"), 0.0),
    }


def _normalize_live_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    symbol = str(_pick(raw, "symbol", default="")).upper()
    side = str(_pick(raw, "side", default="")).upper()
    event_id = str(_pick(raw, "event_id", "id", "trade_id", "order_id", default=""))
    entry_ts = _norm_ts_seconds(_pick(raw, "entry_time", "entry_ts", "ts", "ts_ms"))
    exit_ts = _norm_ts_seconds(_pick(raw, "exit_time", "exit_ts", default=0.0))
    elapsed = _to_float(_pick(raw, "elapsed_sec"), 0.0)
    if elapsed <= 0.0 and entry_ts > 0.0 and exit_ts > entry_ts:
        elapsed = exit_ts - entry_ts
    return {
        "event_id": event_id,
        "symbol": symbol,
        "side": side,
        "ts": float(entry_ts),
        "filled": True,  # Paper trade rows are actual completed trades.
        "fill_delay_sec": max(0.0, float(elapsed)),
        "pnl_bps": _to_float(_pick(raw, "pnl_bps", "net_return_bps"), 0.0),
        "max_adverse_bps": _to_float(_pick(raw, "max_adverse_bps", "adverse_bps"), 0.0),
    }


def _stable_sort(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        list(rows),
        key=lambda r: (
            str(r.get("symbol", "")),
            str(r.get("side", "")),
            float(r.get("ts", 0.0)),
            str(r.get("event_id", "")),
        ),
    )


def _coerce_rows(rows: List[Dict[str, Any]], *, kind: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if "ts" in row and "symbol" in row:
            out.append(
                {
                    "event_id": str(row.get("event_id", "")),
                    "symbol": str(row.get("symbol", "")).upper(),
                    "side": str(row.get("side", "")).upper(),
                    "ts": _norm_ts_seconds(row.get("ts")),
                    "filled": _to_bool(row.get("filled"), default=(kind == "live")),
                    "fill_delay_sec": _to_float(row.get("fill_delay_sec"), 0.0),
                    "pnl_bps": _to_float(row.get("pnl_bps"), 0.0),
                    "max_adverse_bps": _to_float(row.get("max_adverse_bps"), 0.0),
                }
            )
            continue
        if kind == "live":
            n = _normalize_live_row(row)
        else:
            n = _normalize_sim_row(row)
        if n["ts"] > 0.0:
            out.append(n)
    return _stable_sort(out)


def load_simulated_fill_rows(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    if p.suffix.lower() in {".jsonl", ".jl", ".log"}:
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                n = _normalize_sim_row(obj)
                if n["ts"] > 0.0:
                    out.append(n)
    else:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            obj = None
        if isinstance(obj, list):
            for row in obj:
                if not isinstance(row, dict):
                    continue
                n = _normalize_sim_row(row)
                if n["ts"] > 0.0:
                    out.append(n)
        elif isinstance(obj, dict):
            rows = obj.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    n = _normalize_sim_row(row)
                    if n["ts"] > 0.0:
                        out.append(n)
    return _stable_sort(out)


def load_live_fill_rows(path: str | Path, *, table: str = "trades") -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT symbol, side, entry_time, exit_time, elapsed_sec, pnl_bps, max_adverse_bps FROM {table} ORDER BY entry_time ASC"
        ).fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()
    for row in rows:
        d = {k: row[k] for k in row.keys()}
        n = _normalize_live_row(d)
        if n["ts"] > 0.0:
            out.append(n)
    return _stable_sort(out)


def _mean(vals: List[float]) -> float:
    return (sum(vals) / len(vals)) if vals else 0.0


def compute_replay_parity(
    simulated_rows: List[Dict[str, Any]],
    live_rows: List[Dict[str, Any]],
    *,
    match_window_sec: float = 30.0,
) -> ReplayParityResult:
    sim = _coerce_rows(simulated_rows, kind="sim")
    live = _coerce_rows(live_rows, kind="live")
    used_live: set[int] = set()
    matches: List[ReplayMatch] = []
    w = max(0.0, float(match_window_sec))

    for s in sim:
        s_sym = str(s.get("symbol", ""))
        s_side = str(s.get("side", ""))
        s_ts = float(s.get("ts", 0.0))
        best_idx: Optional[int] = None
        best_dt: Optional[float] = None
        for idx, l in enumerate(live):
            if idx in used_live:
                continue
            if str(l.get("symbol", "")) != s_sym:
                continue
            if s_side and str(l.get("side", "")) and str(l.get("side", "")) != s_side:
                continue
            dt = abs(float(l.get("ts", 0.0)) - s_ts)
            if dt > w:
                continue
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_idx = idx
        if best_idx is None:
            continue
        used_live.add(best_idx)
        l = live[best_idx]
        sim_delay = float(s.get("fill_delay_sec", 0.0))
        live_delay = float(l.get("fill_delay_sec", 0.0))
        sim_pnl = float(s.get("pnl_bps", 0.0))
        live_pnl = float(l.get("pnl_bps", 0.0))
        sim_adv = float(s.get("max_adverse_bps", 0.0))
        live_adv = float(l.get("max_adverse_bps", 0.0))
        dt_signed = float(l.get("ts", 0.0)) - s_ts
        matches.append(
            ReplayMatch(
                symbol=s_sym,
                side=s_side,
                sim_event_id=str(s.get("event_id", "")),
                sim_ts=s_ts,
                live_ts=float(l.get("ts", 0.0)),
                dt_sec=dt_signed,
                sim_filled=bool(s.get("filled", False)),
                live_filled=bool(l.get("filled", False)),
                sim_fill_delay_sec=sim_delay,
                live_fill_delay_sec=live_delay,
                fill_delay_delta_sec=live_delay - sim_delay,
                sim_pnl_bps=sim_pnl,
                live_pnl_bps=live_pnl,
                pnl_bps_delta=live_pnl - sim_pnl,
                sim_max_adverse_bps=sim_adv,
                live_max_adverse_bps=live_adv,
                adverse_bps_delta=live_adv - sim_adv,
            )
        )

    sim_fill_rate = _mean([1.0 if bool(r.get("filled", False)) else 0.0 for r in sim])
    live_fill_rate = _mean([1.0 if bool(r.get("filled", False)) else 0.0 for r in live])
    dt_vals = [m.dt_sec for m in matches]
    abs_dt_vals = [abs(v) for v in dt_vals]
    delay_delta = [m.fill_delay_delta_sec for m in matches]
    pnl_delta = [m.pnl_bps_delta for m in matches]
    adv_delta = [m.adverse_bps_delta for m in matches]
    matched = len(matches)
    return ReplayParityResult(
        sim_count=len(sim),
        live_count=len(live),
        matched_count=matched,
        unmatched_sim_count=max(0, len(sim) - matched),
        unmatched_live_count=max(0, len(live) - matched),
        match_rate_vs_sim=(matched / len(sim)) if sim else 0.0,
        sim_fill_rate=sim_fill_rate,
        live_fill_rate=live_fill_rate,
        fill_rate_delta=live_fill_rate - sim_fill_rate,
        mean_dt_sec=_mean(dt_vals),
        mean_abs_dt_sec=_mean(abs_dt_vals),
        mean_fill_delay_delta_sec=_mean(delay_delta),
        mean_pnl_bps_delta=_mean(pnl_delta),
        mean_adverse_bps_delta=_mean(adv_delta),
        matches=matches,
    )
