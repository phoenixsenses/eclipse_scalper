# brain/state.py — PSYCHE STATE GOD-EMPEROR INFINITE FINAL (CONSCIOUSNESS ABSOLUTE)
# 2026 v4.6 (FULL PATCH: entry_watches canon + safer merges + hard caps)
#
# Patch vs v4.5:
# - Canon-law parity: _symkey handles '/USDT:USDT' + ':USDT' + '/USDT'
# - Adds entry_watches to PsycheState + migration/validation canonicalization
# - Fix: blacklist_reason merge uses "prefer new" (strings), not dict-merge
# - Adds last_trail_ts default in symbol_performance (exit.py expects it)
# - Caps: known_exit_order_ids, entry_confidence_history, trailing_order_ids, entry_watches

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Set, Optional, List, Tuple, Any, Callable
from datetime import date, datetime
import time

DEFAULT_CORE_VERSION = "god-emperor-ascendant-absolute-v4.6-2026-jan06"
STATE_SCHEMA_VERSION = 3

KNOWN_EXIT_IDS_CAP = 50_000
ENTRY_CONF_HISTORY_CAP = 200
TRAILING_IDS_CAP = 20
ENTRY_WATCHES_CAP = 500


def _now() -> float:
    return time.time()


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        return v
    except Exception:
        return default


def _safe_int(x, default: int = 0) -> int:
    try:
        v = int(x)
        return v if v >= 0 else default
    except Exception:
        return default


def _symkey(sym: Any) -> str:
    """
    Canonicalize symbol keys for state maps.
    Examples:
      'BTC/USDT:USDT' -> 'BTCUSDT'
      'BTC/USDT'      -> 'BTCUSDT'
      'btcusdt'       -> 'BTCUSDT'
      None            -> ''
    """
    if sym is None:
        return ""
    try:
        s = str(sym).strip().upper()
    except Exception:
        return ""
    if not s:
        return ""

    s = s.replace("/USDT:USDT", "USDT").replace("/USDT", "USDT")
    s = s.replace(":USDT", "USDT").replace(":", "")
    s = s.replace("/", "")

    if s.endswith("USDTUSDT"):
        s = s[:-4]
    return s


def _merge_max(a: Any, b: Any) -> Any:
    """Merge numeric-ish values by max (used for expiries/timestamps/counters)."""
    try:
        fa = _safe_float(a, 0.0)
        fb = _safe_float(b, 0.0)
        return a if fa >= fb else b
    except Exception:
        return a if a is not None else b


def _merge_dict_shallow(a: Any, b: Any) -> Any:
    """Shallow merge dicts; b wins on key collisions."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        out.update(b)
        return out
    return b if b is not None else a


def _merge_pick_b(a: Any, b: Any) -> Any:
    """Prefer new value on collisions (good for str maps like blacklist_reason)."""
    return b if b is not None else a


def _canon_map_keys(m: Any, *, merge_fn: Optional[Callable[[Any, Any], Any]] = None) -> Dict[str, Any]:
    """
    Canonicalize dict keys; drop empty keys; never throws.
    If merge_fn is provided, collisions are merged instead of overwritten.
    """
    out: Dict[str, Any] = {}
    if not isinstance(m, dict):
        return out

    for k, v in m.items():
        ck = _symkey(k)
        if not ck:
            continue

        if ck in out and merge_fn is not None:
            try:
                out[ck] = merge_fn(out[ck], v)
            except Exception:
                out[ck] = v
        else:
            out[ck] = v

    return out


def _merge_entry_watches_canon(m: Any) -> Dict[str, Dict[str, Any]]:
    """
    Canonicalize entry_watches keys and keep the newest created_ts per symbol.
    Schema is defined by execution/entry_watch.py.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(m, dict):
        return out

    for sym, w in m.items():
        k = _symkey(sym)
        if not k or not isinstance(w, dict):
            continue

        w2 = dict(w)
        w2["k"] = k

        try:
            w2["symbol_any"] = str(w2.get("symbol_any") or k)
        except Exception:
            w2["symbol_any"] = k

        ct = _safe_float(w2.get("created_ts", 0.0), 0.0)

        if k not in out:
            out[k] = w2
        else:
            prev_ct = _safe_float(out[k].get("created_ts", 0.0), 0.0)
            if ct >= prev_ct:
                out[k] = w2

    # cap deterministically (keep newest)
    try:
        if len(out) > ENTRY_WATCHES_CAP:
            items = sorted(out.items(), key=lambda kv: _safe_float(kv[1].get("created_ts", 0.0), 0.0))
            out = dict(items[-ENTRY_WATCHES_CAP:])
    except Exception:
        pass

    return out


def _merge_positions_canon(pos_map: Any) -> Dict[str, "Position"]:
    """
    Canonicalize position keys and repair values into Position.
    Collision policy: keep newer entry_ts.
    """
    out: Dict[str, Position] = {}
    if not isinstance(pos_map, dict):
        return out

    for sym, pv in pos_map.items():
        k = _symkey(sym)
        if not k:
            continue

        p: Optional[Position] = None
        if isinstance(pv, Position):
            p = pv
        elif isinstance(pv, dict):
            try:
                p = Position(**pv)
            except Exception:
                p = Position(
                    side=str(pv.get("side", "long")),
                    size=_safe_float(pv.get("size", 0.0), 0.0),
                    entry_price=_safe_float(pv.get("entry_price", 0.0), 0.0),
                    atr=_safe_float(pv.get("atr", 0.0), 0.0),
                    leverage=_safe_int(pv.get("leverage", 0), 0),
                    entry_ts=_safe_float(pv.get("entry_ts", _now()), _now()),
                    hard_stop_order_id=pv.get("hard_stop_order_id"),
                    trailing_active=bool(pv.get("trailing_active", False)),
                    breakeven_moved=bool(pv.get("breakeven_moved", False)),
                    confidence=_safe_float(pv.get("confidence", 0.0), 0.0),
                    last_breakeven_move=_safe_float(pv.get("last_breakeven_move", 0.0), 0.0),
                    symbol=str(pv.get("symbol")) if pv.get("symbol") is not None else None,
                )
        else:
            continue

        p.validate()
        p.symbol = k

        if k not in out:
            out[k] = p
        else:
            try:
                if float(p.entry_ts) >= float(out[k].entry_ts):
                    out[k] = p
            except Exception:
                pass

    return out


@dataclass
class Position:
    side: str  # 'long' or 'short'
    size: float
    entry_price: float

    atr: float = 0.0
    leverage: int = 0
    entry_ts: float = field(default_factory=_now)

    hard_stop_order_id: Optional[str] = None
    trailing_active: bool = False
    breakeven_moved: bool = False
    confidence: float = 0.0
    last_breakeven_move: float = 0.0

    symbol: Optional[str] = None  # canonical

    def validate(self) -> None:
        self.side = (self.side or "").lower().strip()
        if self.side not in ("long", "short"):
            self.side = "long"

        # Canon law: size is absolute
        self.size = abs(_safe_float(self.size, 0.0))

        self.entry_price = _safe_float(self.entry_price, 0.0)
        self.atr = _safe_float(self.atr, 0.0)
        self.leverage = _safe_int(self.leverage, 0)
        self.entry_ts = _safe_float(self.entry_ts, _now())
        self.confidence = _safe_float(self.confidence, 0.0)
        self.last_breakeven_move = _safe_float(self.last_breakeven_move, 0.0)

        self.trailing_active = bool(self.trailing_active)
        self.breakeven_moved = bool(self.breakeven_moved)

        if self.hard_stop_order_id is not None:
            self.hard_stop_order_id = str(self.hard_stop_order_id)

        if self.symbol is not None:
            self.symbol = _symkey(self.symbol) or None


@dataclass
class PsycheState:
    schema_version: int = STATE_SCHEMA_VERSION
    version: str = DEFAULT_CORE_VERSION
    run_context: Dict[str, Any] = field(default_factory=dict)

    current_equity: float = 0.0
    peak_equity: float = 0.0
    peak_equity_timestamp: float = field(default_factory=_now)
    current_drawdown_pct: float = 0.0

    daily_pnl: float = 0.0
    start_of_day_equity: float = 0.0
    current_day: Optional[date] = None

    total_trades: int = 0
    total_wins: int = 0
    win_streak: int = 0

    positions: Dict[str, Position] = field(default_factory=dict)

    blacklist: Dict[str, float] = field(default_factory=dict)
    blacklist_reason: Dict[str, str] = field(default_factory=dict)
    consecutive_losses: Dict[str, int] = field(default_factory=dict)
    last_exit_time: Dict[str, float] = field(default_factory=dict)
    known_exit_order_ids: Set[str] = field(default_factory=set)

    symbol_performance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    entry_confidence_history: Dict[str, List[float]] = field(default_factory=dict)

    streak_history: List[Tuple[date, int, float]] = field(default_factory=list)
    adaptive_risk_multiplier: float = 1.0

    funding_paid: float = 0.0
    funding_rate_snapshot: Dict[str, float] = field(default_factory=dict)

    # entry_watch persistence
    entry_watches: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    session_start_timestamp: float = field(default_factory=_now)
    uptime_seconds: float = 0.0

    win_rate: float = 0.0
    max_drawdown: float = 0.0

    # ---------------------------
    # Migration / construction
    # ---------------------------
    @classmethod
    def from_loaded(cls, obj: Any) -> "PsycheState":
        try:
            if isinstance(obj, cls):
                st = obj
            elif isinstance(obj, dict):
                st = cls()
                for k, v in obj.items():
                    if hasattr(st, k):
                        setattr(st, k, v)
            else:
                st = cls()

            st._migrate_in_place()
            st.validate()
            st.recompute_derived()
            return st
        except Exception:
            st = cls()
            try:
                st.validate()
                st.recompute_derived()
            except Exception:
                pass
            return st

    def _migrate_in_place(self) -> None:
        sv = _safe_int(getattr(self, "schema_version", 0), 0)

        if sv < 3:
            if getattr(self, "run_context", None) is None or not isinstance(self.run_context, dict):
                self.run_context = {}

            kei = getattr(self, "known_exit_order_ids", None)
            if isinstance(kei, (list, tuple)):
                self.known_exit_order_ids = set(str(x) for x in kei)
            elif not isinstance(kei, set):
                self.known_exit_order_ids = set()

            self.positions = _merge_positions_canon(getattr(self, "positions", {}))

            self.blacklist = _canon_map_keys(getattr(self, "blacklist", {}), merge_fn=_merge_max)
            self.last_exit_time = _canon_map_keys(getattr(self, "last_exit_time", {}), merge_fn=_merge_max)
            self.consecutive_losses = _canon_map_keys(getattr(self, "consecutive_losses", {}), merge_fn=_merge_max)

            self.blacklist_reason = _canon_map_keys(getattr(self, "blacklist_reason", {}), merge_fn=_merge_pick_b)

            self.symbol_performance = _canon_map_keys(getattr(self, "symbol_performance", {}), merge_fn=_merge_dict_shallow)
            self.entry_confidence_history = _canon_map_keys(getattr(self, "entry_confidence_history", {}), merge_fn=_merge_dict_shallow)
            self.funding_rate_snapshot = _canon_map_keys(getattr(self, "funding_rate_snapshot", {}), merge_fn=_merge_dict_shallow)

            self.entry_watches = _merge_entry_watches_canon(getattr(self, "entry_watches", {}))

            self.schema_version = STATE_SCHEMA_VERSION

    # ---------------------------
    # Validation / repair
    # ---------------------------
    def validate(self) -> None:
        if self.run_context is None or not isinstance(self.run_context, dict):
            self.run_context = {}

        self.positions = _merge_positions_canon(self.positions)

        self.blacklist = _canon_map_keys(self.blacklist, merge_fn=_merge_max)
        self.last_exit_time = _canon_map_keys(self.last_exit_time, merge_fn=_merge_max)
        self.consecutive_losses = _canon_map_keys(self.consecutive_losses, merge_fn=_merge_max)

        self.blacklist_reason = _canon_map_keys(self.blacklist_reason, merge_fn=_merge_pick_b)
        self.symbol_performance = _canon_map_keys(self.symbol_performance, merge_fn=_merge_dict_shallow)
        self.entry_confidence_history = _canon_map_keys(self.entry_confidence_history, merge_fn=_merge_dict_shallow)
        self.funding_rate_snapshot = _canon_map_keys(self.funding_rate_snapshot, merge_fn=_merge_dict_shallow)

        self.entry_watches = _merge_entry_watches_canon(getattr(self, "entry_watches", {}))

        # known_exit_order_ids: ensure set[str]
        if self.known_exit_order_ids is None:
            self.known_exit_order_ids = set()
        elif isinstance(self.known_exit_order_ids, (list, tuple)):
            self.known_exit_order_ids = set(str(x) for x in self.known_exit_order_ids)
        elif not isinstance(self.known_exit_order_ids, set):
            self.known_exit_order_ids = set()

        try:
            if len(self.known_exit_order_ids) > KNOWN_EXIT_IDS_CAP:
                self.known_exit_order_ids = set(list(self.known_exit_order_ids)[-KNOWN_EXIT_IDS_CAP:])
        except Exception:
            pass

        self.current_equity = max(0.0, _safe_float(self.current_equity, 0.0))
        self.peak_equity = max(0.0, _safe_float(self.peak_equity, 0.0))
        self.start_of_day_equity = max(0.0, _safe_float(self.start_of_day_equity, 0.0))
        self.daily_pnl = _safe_float(self.daily_pnl, 0.0)

        self.total_trades = max(0, _safe_int(self.total_trades, 0))
        self.total_wins = max(0, _safe_int(self.total_wins, 0))
        self.win_streak = max(0, _safe_int(self.win_streak, 0))

        self.peak_equity_timestamp = _safe_float(self.peak_equity_timestamp, _now())
        self.current_drawdown_pct = max(0.0, _safe_float(self.current_drawdown_pct, 0.0))
        self.max_drawdown = max(0.0, _safe_float(self.max_drawdown, 0.0))
        self.win_rate = max(0.0, _safe_float(self.win_rate, 0.0))

        self.funding_paid = _safe_float(self.funding_paid, 0.0)
        self.session_start_timestamp = _safe_float(self.session_start_timestamp, _now())
        self.uptime_seconds = max(0.0, _safe_float(self.uptime_seconds, 0.0))
        self.adaptive_risk_multiplier = max(0.0, _safe_float(self.adaptive_risk_multiplier, 1.0))

        # current_day normalization
        if self.current_day is not None and not isinstance(self.current_day, date):
            try:
                if isinstance(self.current_day, str):
                    self.current_day = date.fromisoformat(self.current_day)
                elif isinstance(self.current_day, (int, float)):
                    self.current_day = datetime.utcfromtimestamp(float(self.current_day)).date()
                else:
                    self.current_day = None
            except Exception:
                self.current_day = None

        if self.current_equity > 0 and self.peak_equity < self.current_equity:
            self.peak_equity = self.current_equity
            self.peak_equity_timestamp = _now()

        if self.total_wins > self.total_trades:
            self.total_wins = self.total_trades

        # validate positions
        for k, p in list(self.positions.items()):
            try:
                p.validate()
                p.symbol = k
            except Exception:
                try:
                    del self.positions[k]
                except Exception:
                    pass

        # symbol_performance hygiene
        for sym, perf in list(self.symbol_performance.items()):
            k = _symkey(sym)
            if not k:
                try:
                    del self.symbol_performance[sym]
                except Exception:
                    pass
                continue

            if not isinstance(perf, dict):
                self.symbol_performance[k] = {"pnl": 0.0, "wins": 0, "losses": 0, "last_win": 0.0}
                perf = self.symbol_performance[k]
            else:
                if k != sym:
                    self.symbol_performance[k] = perf
                    try:
                        del self.symbol_performance[sym]
                    except Exception:
                        pass
                    perf = self.symbol_performance[k]

            perf.setdefault("pnl", 0.0)
            perf.setdefault("wins", 0)
            perf.setdefault("losses", 0)
            perf.setdefault("last_win", 0.0)
            perf.setdefault("pos_realized_pnl", 0.0)
            perf.setdefault("entry_size_abs", 0.0)
            perf.setdefault("mfe_pct", 0.0)
            perf.setdefault("trailing_order_ids", [])
            perf.setdefault("last_trail_ts", 0.0)

            if not isinstance(perf.get("trailing_order_ids"), list):
                perf["trailing_order_ids"] = list(perf.get("trailing_order_ids") or [])
            if len(perf["trailing_order_ids"]) > TRAILING_IDS_CAP:
                perf["trailing_order_ids"] = perf["trailing_order_ids"][-TRAILING_IDS_CAP:]

        # confidence history hygiene
        for sym, hist in list(self.entry_confidence_history.items()):
            k = _symkey(sym)
            if not k:
                try:
                    del self.entry_confidence_history[sym]
                except Exception:
                    pass
                continue

            hist2 = hist if isinstance(hist, list) else (list(hist) if hist is not None else [])
            cleaned: List[float] = []
            for x in hist2[-ENTRY_CONF_HISTORY_CAP:]:
                cleaned.append(_safe_float(x, 0.0))

            if k != sym:
                try:
                    del self.entry_confidence_history[sym]
                except Exception:
                    pass
            self.entry_confidence_history[k] = cleaned

        # funding snapshot floats
        for sym, v in list(self.funding_rate_snapshot.items()):
            k = _symkey(sym)
            if not k:
                try:
                    del self.funding_rate_snapshot[sym]
                except Exception:
                    pass
                continue
            fv = _safe_float(v, 0.0)
            if k != sym:
                try:
                    del self.funding_rate_snapshot[sym]
                except Exception:
                    pass
            self.funding_rate_snapshot[k] = fv

    # ---------------------------
    # Derived metrics
    # ---------------------------
    def recompute_derived(self) -> None:
        self.win_rate = (self.total_wins / self.total_trades) if self.total_trades > 0 else 0.0

        if self.peak_equity > 0 and self.current_equity >= 0:
            dd = (self.peak_equity - self.current_equity) / self.peak_equity
            self.current_drawdown_pct = max(0.0, dd)
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown_pct)
        else:
            self.current_drawdown_pct = 0.0

    def update_equity(self, equity: float, *, ts: Optional[float] = None) -> None:
        ts = _safe_float(ts, _now())
        eq = max(0.0, _safe_float(equity, 0.0))
        self.current_equity = eq

        if self.peak_equity <= 0 or eq > self.peak_equity:
            self.peak_equity = eq
            self.peak_equity_timestamp = ts

        self.recompute_derived()

    # ---------------------------
    # Hygiene helpers
    # ---------------------------
    def cleanup_expired_blacklist(self, now_ts: Optional[float] = None) -> None:
        now_ts = _safe_float(now_ts, _now())
        expired = []
        for sym, exp in (self.blacklist or {}).items():
            if _safe_float(exp, 0.0) <= now_ts:
                expired.append(sym)
        for sym in expired:
            try:
                del self.blacklist[sym]
            except Exception:
                pass
            try:
                if sym in self.blacklist_reason:
                    del self.blacklist_reason[sym]
            except Exception:
                pass

    # ---------------------------
    # Disk-safe serialization
    # ---------------------------
    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        self.recompute_derived()

        raw = asdict(self)

        # sets -> lists (cap)
        kei = raw.get("known_exit_order_ids")
        if isinstance(kei, set):
            kei_list = sorted(str(x) for x in kei)
        elif isinstance(kei, list):
            kei_list = [str(x) for x in kei]
        else:
            kei_list = []
        if len(kei_list) > KNOWN_EXIT_IDS_CAP:
            kei_list = kei_list[-KNOWN_EXIT_IDS_CAP:]
        raw["known_exit_order_ids"] = kei_list

        # date -> iso
        cd = raw.get("current_day")
        if isinstance(cd, date):
            raw["current_day"] = cd.isoformat()
        elif cd is None:
            raw["current_day"] = None
        else:
            try:
                raw["current_day"] = str(cd)
            except Exception:
                raw["current_day"] = None

        # streak_history: (date, int, float) -> (iso, int, float)
        sh = raw.get("streak_history")
        if isinstance(sh, list):
            cleaned = []
            for row in sh:
                try:
                    d, n, v = row
                    d_iso = d.isoformat() if isinstance(d, date) else str(d)
                    cleaned.append((d_iso, _safe_int(n, 0), _safe_float(v, 0.0)))
                except Exception:
                    continue
            raw["streak_history"] = cleaned
        else:
            raw["streak_history"] = []

        # entry_watches: keep dict
        ew = raw.get("entry_watches")
        raw["entry_watches"] = ew if isinstance(ew, dict) else {}

        return raw
