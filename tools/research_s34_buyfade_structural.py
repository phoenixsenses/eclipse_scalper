"""C-BUY-FADE Structural Path, Timing and Multi-Timeframe Research (preregistered).

Route AYNEN korunur (tools/s34_realtime_shadow_runner.py BUY_FADE_SHORT_H45_SL75):
  event: ETH BUY liquidation cascade >=200K (bucket 300s, min_gap 900s, accel 60s)
  veto : session==EUROPE  VEYA  bear-squeeze (eth1h<-20 AND btc4h<-50)
  entry: T0 mark at_or_after, SHORT; hold 45m; SL 75bps adverse; FEE 5bps
  silence v1: [T0+60s, T0+30m] icinde ETH BUY follow-on >=50K YOK
              (T+30m'DE BILINIR — T0 entry feature'i DEGILDIR, yapisal guard var)

Bolumler: A pre-event LONG genesis/maturity · B delayed entry · C horizon/continuation ·
D multi-TF context matrix · silence decomposition + matched controls · management.

Mevcut olmayan feature'lar NOT_AVAILABLE raporlanir (proxy uydurulmaz):
  OI slope/accel, price-OI divergence  -> open_interest tarihsel bos (poller 2026-07-02)
  spot/perp participation              -> spot_prices hacim yok + kapsama bosluklu
  basis                                -> spot kapsama <%60 ise NOT_AVAILABLE
  depth beyond L1 / impact-per-dollar  -> book_ticker L1-only

p=0.010 notu: dashboard mc_p degerleri HARDCODED metadata; 100-permutasyon tabani
(1/101=0.0099) ile tutarli. Bu calismada 20.000 permutasyon (cozunurluk 5e-5).

Run: python tools/research_s34_buyfade_structural.py
Cikti: reports/research/s34/BUYFADE_STRUCTURAL.{json,md}
"""
from __future__ import annotations
import json, math, sqlite3, sys
from bisect import bisect_left, bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.constitution import ConstitutionViolation
from ami.research.registry import ExperimentSpec, ResearchQuestion, ResearchRegistry
from tools.research_s34_knowable_anchor_continuation import (
    load_liquidations, load_mark_index, reconstruct_anchors)
from tools.research_s34_wave_absorption import book_features_at

OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "BUYFADE_STRUCTURAL.json"; OM = OUT / "BUYFADE_STRUCTURAL.md"
DB = ROOT / "data" / "microstructure.db"

# ── ROUTE SABITLERI (shadow runner ile birebir; DEGISTIRILEMEZ) ───────────────
ETH_THRESH = 200_000.0; PROP_THRESH = 50_000.0
BUCKET_SEC = 300; MIN_GAP_SEC = 900; ACCEL_WIN_SEC = 60
HOLD_MS = 45 * 60_000; SL_BPS = 75.0; FEE = 5.0
SIL_LO_MS = 60_000; SIL_HI_MS = 30 * 60_000
SILENCE_VERSION = "buyfade_silence_v1: no ETH BUY >=50K in [T0+60s, T0+30m] (knowable T+30m)"

# ── FROZEN ARASTIRMA GRIDLERI ─────────────────────────────────────────────────
SEED = 11
N_PERM = 20_000
MIN_CELL = 12
MIN_TOTAL = 30
ECON_MIN_NET = 3.0            # bps/trade ekonomik anlam esigi (fee sonrasi)
PURGE_MS = 24 * 3600_000      # split sinirlarinda purge (ayni cycle iki tarafta olamaz)
SPLITS = {"train": (0.00, 0.60), "val": (0.60, 0.85), "untouched": (0.85, 1.00)}
PRE_WINDOWS_MIN = [15, 30, 60, 120, 240, 360, 720, 1440, 4320, 10080]
HORIZONS_MIN = [5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 1440, 2880, 4320, 10080]
FIXED_DELAYS_S = [0, 30, 60, 120, 180, 300, 600, 900, 1800]
PULLBACK_BPS = [10.0, 20.0, 30.0]
ENTRY_WINDOW_MS = 30 * 60_000     # gecikmeli girisler icin tetik penceresi
TF_DEFS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
           "4h": 14_400_000, "1D": 86_400_000, "1W": 604_800_000}
TF_THR_BPS = {"1m": 5, "5m": 10, "15m": 15, "1h": 30, "4h": 80, "1D": 200, "1W": 400}
TF_BARS = 20                      # SMA/state icin tamamlanmis bar sayisi
MATURITY_ORDER = ["C_EXHAUSTION", "E_HTF_DOWN_BOUNCE", "D_HTF_UP_PULLBACK",
                  "A_NEW_EXPANSION", "B_MATURE_TREND", "F_RANGE"]
LOCK_CANDIDATES = [(20.0, 0.0), (20.0, 5.0), (30.0, 10.0), (50.0, 20.0), (50.0, 25.0)]
TIME_STOP_CHECKS_MIN = [5, 10, 15, 30]
FIXED_HOLDS_MIN = [15, 30, 45, 60, 90, 120, 240, 360, 720, 1440]
COMBOS = ["rejection+silence3m", "lowerhigh+ofiflip", "delay5m+no_new_high", "silence3m+btc_weak"]

FORBIDDEN_T0_KEYS = ("silence_30m", "silence_full", "outcome", "fwd_", "mfe", "mae",
                     "path_class", "reclaim", "buy_state")


# ── GUARD'LAR (mutation suite test eder) ─────────────────────────────────────
def g_backward_only(t0: int, data_end: int) -> None:
    if data_end > t0:
        raise ConstitutionViolation(f"FUTURE LEAKAGE: feature verisi {data_end} > T0 {t0}")


def g_completed_bar(t0: int, bar_close_ts: int) -> None:
    if bar_close_ts > t0:
        raise ConstitutionViolation(f"UNFINISHED CANDLE: bar close {bar_close_ts} > T0 {t0}")


def g_no_t0_silence(feature_keys) -> None:
    bad = [k for k in feature_keys for f in FORBIDDEN_T0_KEYS if f in k.lower()]
    if bad:
        raise ConstitutionViolation(f"T0 feature setinde yasak anahtar: {bad}")


def g_event_high_causal(trigger_ts: int, high_window_end: int) -> None:
    if high_window_end > trigger_ts:
        raise ConstitutionViolation("Delayed entry: event high gelecek pencereden hesaplanmis")


def g_split_purge(t_train_max: int, t_val_min: int, purge_ms: int = PURGE_MS) -> None:
    if t_val_min - t_train_max < purge_ms:
        raise ConstitutionViolation("Split purge ihlali: ayni cycle train+val'e sizabilir")


def g_train_only_selection(split_tag: str) -> None:
    if split_tag != "train":
        raise ConstitutionViolation(f"Esik/horizon secimi yalniz TRAIN'de: '{split_tag}' yasak")


def g_tiny_cell(n: int, label: str) -> str:
    return label if n >= MIN_CELL else "INSUFFICIENT_SAMPLE"


def g_unknown_not_neutral(state: str) -> str:
    if state not in ("UP", "DOWN", "RANGE", "UNKNOWN"):
        raise ConstitutionViolation(f"Gecersiz state: {state}")
    return state


def g_executable(fill_ts: int | None, trigger_ts: int, staleness_ms: int = 120_000) -> bool:
    return fill_ts is not None and 0 <= fill_ts - trigger_ts <= staleness_ms


def candidate_pass(*, holdout_net: float, holdout_n: int, top3_removed_net: float,
                   incremental_vs_baseline: bool, econ_net: float, folds_same_dir: int,
                   family_p: float, tiny: bool) -> tuple[bool, list[str]]:
    fails = []
    if holdout_net <= 0: fails.append("holdout_not_positive")
    if holdout_n < MIN_CELL: fails.append("holdout_tiny")
    if top3_removed_net <= 0: fails.append("topwinner_dependent")
    if not incremental_vs_baseline: fails.append("not_incremental")
    if econ_net < ECON_MIN_NET: fails.append("not_economic")
    if folds_same_dir < 2: fails.append("fold_direction")
    if family_p >= 0.05: fails.append("family_p")
    if tiny: fails.append("tiny_cell")
    return (not fails), fails


# ── PREREG ───────────────────────────────────────────────────────────────────
def freeze_spec(reg: ResearchRegistry) -> ExperimentSpec:
    reg.add_question(ResearchQuestion(
        question_id="Q-BUYFADE-STRUCT-001",
        question="BUY-fade SHORT'un yapisal yolu: LONG genesis/maturity, dogru giris "
                 "zamani, continuation suresi, multi-TF baglam ve silence'in zamansal yapisi",
        origin_observation="shadow N=26 WR54 +2.8 / silence N=14 WR86 +21.7 (HIPOTEZ; "
                           "mc_p=0.010 hardcoded, 100-perm tabaniyla tutarli). "
                           "NOT: failure archive'da eski 'ETH BUY->SHORT mc 0.988' kaydi var; "
                           "retry gerekce = yeni route tanimi (45m/SL75/veto'lar) + shadow forward.",
        economic_value=0.6, risk_reduction_value=0.5, falsifiability=0.9,
        required_sample=MIN_TOTAL))
    spec = ExperimentSpec(
        experiment_id="E-BUYFADE-STRUCT-001", question_id="Q-BUYFADE-STRUCT-001",
        population=("ETH BUY cascade >=200K anchor'lari (bucket 300s, min_gap 900s, accel 60s), "
                    "route veto'lari AYNEN: EUROPE session skip + bear-squeeze skip "
                    "(eth1h<-20 AND btc4h<-50). Event universe sonuca gore DEGISTIRILEMEZ. "
                    "Veri: 2026-02-15..build-ani (mark_prices siniri); son 7D horizon icin kirpilir."),
        target=("A: pre-event LONG genesis/maturity siniflari (frozen kural, precedence "
                f"{MATURITY_ORDER}); B: delayed/price/flow/combo girisler; C: {len(HORIZONS_MIN)} "
                "horizon expectancy curve + continuation; D: 7-TF context matrix; silence "
                "zaman-decompose + matched controls; management (fixed/time-stop/lock/structural)."),
        features=["price_structure(mark index, backward-only)", "flow(agg_trades OFI/rate/large-share)",
                  "liq(build-up ETH/BTC BUY-SELL)", "funding(level+slope8h)",
                  "book_L1(spread/imbalance/depth/pull-refill)", "btc_sync",
                  "NOT_AVAILABLE: OI-slope/accel, price-OI-div, spot-perp-participation, "
                  "basis(kapsama<0.6 ise), depth>L1, impact-per-dollar"],
        threshold_method=(f"tum gridler frozen: pre-windows {PRE_WINDOWS_MIN}m; delays {FIXED_DELAYS_S}s; "
                          f"pullback {PULLBACK_BPS}bps; horizons {HORIZONS_MIN}m; TF thr {TF_THR_BPS}; "
                          f"combo listesi {COMBOS} (4 adet, fishing yok); lock {LOCK_CANDIDATES}; "
                          f"time-stop MFE esigi = TRAIN median MFE@checkpoint (frozen kural); "
                          f"min_cell {MIN_CELL}; econ esik {ECON_MIN_NET}bps"),
        chronological_split=(f"event-bazli kronolojik {SPLITS} + split sinirlarinda {PURGE_MS//3600_000}h "
                             "purge (ayni cascade cycle iki tarafta olamaz); esik/secim TRAIN'de, "
                             "rapor VAL'de, final UNTOUCHED'ta"),
        untouched_data=("son %15 (event bazli) — YENI sorular (genesis/timing/horizon/management) icin "
                        "untouched; route'un KENDISI (T0/45m/SL75/silence) shadow'da secildigi icin "
                        "route-level claim'ler icin untouched sayilmaz (durustluk beyani)"),
        negative_control=("random-delay control (ayni event, uniform[0,30m] rastgele giris, 2000 cekim); "
                          "random-veto; hour/session-only kontrol; HTF-regime-only kontrol; "
                          "silence matched-controls (hour+size+rv+spread+trend+btc+density 1-NN)"),
        min_sample=MIN_TOTAL,
        effect_size_required_bps=ECON_MIN_NET,
        multiple_testing_control=f"aile ici max-stat permutation, N_PERM={N_PERM} (cozunurluk 5e-5); "
                                 "subset-aileler: ayni-N rastgele altkume; timing-aileler: paired block bootstrap",
        execution_model="mark_fill_fee5bps (canonical; executor market-order ile tutarli); "
                        "delayed fill staleness<=120s yoksa MISSED; L1 spread sanity subsample",
        decision_criteria=("her bolum ayri verdict: PASS icin candidate_pass() 8 kosulu "
                           "(holdout+, top3-removed+, incremental, econ>=3bps, >=2 fold ayni yon, "
                           f"family-p<0.05, cell>={MIN_CELL}, leakage yok). "
                           "Verdict sozlugu: TIMING_INCREMENTAL/NON, PRE_EVENT_STRUCTURE_PREDICTIVE/NON, "
                           "SCALP_ONLY/MULTI_HOUR/DAILY_CONTINUATION, MANAGEMENT_INCREMENTAL/NON, "
                           "SILENCE_INCREMENTAL/PROXY_ONLY, INSUFFICIENT_SAMPLE, REJECTED"),
        falsification_rule=("null gecerli sonuctur; kriter gevsetme YASAK. Route sonuclarina bakarak "
                            "delay/hold/threshold listesi degistirilemez (frozen gridler). "
                            "Compute budget: tek surec, tek koşum + duzeltme koşumu; "
                            "hyperparam taramasi gridlerle sinirli."))
    spec.freeze()
    reg.register_experiment(spec)
    return spec


# ── VERI KATMANI ─────────────────────────────────────────────────────────────
class Bars:
    """Mark index'ten tamamlanmis-bar OHLC; state hesaplari yalniz close<=T0 barlarla."""
    def __init__(self, marks, tf_ms: int, t_start: int, t_end: int):
        self.tf = tf_ms
        self.t0s, self.closes, self.highs, self.lows = [], [], [], []
        t = (t_start // tf_ms) * tf_ms
        while t + tf_ms <= t_end:
            seg = marks.slice_range(t, t + tf_ms - 1)
            if seg:
                px = [p for _, p in seg]
                self.t0s.append(t); self.closes.append(px[-1])
                self.highs.append(max(px)); self.lows.append(min(px))
            t += tf_ms
        self.close_ts = [t + tf_ms for t in self.t0s]   # bar kapanis ani

    def state_at(self, t0: int, thr_bps: float, n_bars: int = TF_BARS) -> str:
        i = bisect_right(self.close_ts, t0)             # yalniz close<=t0
        if i < n_bars:
            return "UNKNOWN"
        g_completed_bar(t0, self.close_ts[i - 1])
        window = self.closes[i - n_bars:i]
        sma = sum(window) / len(window)
        c = window[-1]; c5 = window[-6] if len(window) >= 6 else window[0]
        ret5 = (c - c5) / c5 * 1e4
        if c > sma and ret5 > thr_bps:
            return "UP"
        if c < sma and ret5 < -thr_bps:
            return "DOWN"
        return "RANGE"


def px_at(marks, ts):
    r = marks.at_or_before(int(ts))
    return float(r[1]) if r else None


def ret_bps(marks, t_from, t_to):
    a, b = px_at(marks, t_from), px_at(marks, t_to)
    return (b - a) / a * 1e4 if a and b and a > 0 else None


def agg_flow(conn, sym, lo, hi):
    r = conn.execute(
        "SELECT COALESCE(SUM(CASE WHEN is_buyer_maker=0 THEN notional ELSE 0 END),0),"
        "COALESCE(SUM(notional),0), COUNT(*),"
        "COALESCE(SUM(CASE WHEN notional>=50000 THEN notional ELSE 0 END),0)"
        " FROM agg_trades WHERE symbol=? AND ts_ms>=? AND ts_ms<?",
        (sym, int(lo), int(hi))).fetchone()
    buy, tot, cnt, large = (float(r[0]), float(r[1]), int(r[2]), float(r[3]))
    return {"ofi": (2 * buy - tot) / tot if tot > 0 else None, "tot": tot,
            "cnt": cnt, "large_share": large / tot if tot > 0 else None,
            "buy": buy}


def liq_sum(conn, sym, side, lo, hi):
    r = conn.execute("SELECT COALESCE(SUM(notional),0) FROM liquidations "
                     "WHERE symbol=? AND side=? AND ts_ms>=? AND ts_ms<?",
                     (sym, side, int(lo), int(hi))).fetchone()
    return float(r[0] or 0.0)


def liq_first_ts(conn, sym, side, lo, hi, thr):
    r = conn.execute("SELECT MIN(ts_ms) FROM liquidations WHERE symbol=? AND side=? "
                     "AND ts_ms>=? AND ts_ms<? AND notional>=?",
                     (sym, side, int(lo), int(hi), float(thr))).fetchone()
    return int(r[0]) if r and r[0] is not None else None


def funding_at(conn, ts):
    r = conn.execute("SELECT funding_rate FROM mark_prices WHERE symbol='ETHUSDT' AND ts_ms<=? "
                     "AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1", (int(ts),)).fetchone()
    return float(r[0]) if r and r[0] is not None else None


def session_name(ts_ms):
    h = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).hour
    return "EUROPE" if 7 <= h < 13 else ("US" if 13 <= h < 21 else "OFF")


# ── EVENT INSASI ─────────────────────────────────────────────────────────────
def build_events(conn, marks_eth, marks_btc):
    t_lo = marks_eth.ts[0]; t_hi = marks_eth.ts[-1]
    liqs = load_liquidations(conn, "ETHUSDT", "BUY", t_lo, t_hi)
    anchors = reconstruct_anchors(liqs, bucket_sec=BUCKET_SEC, min_gap_sec=MIN_GAP_SEC,
                                  thresholds=(ETH_THRESH,), accel_window_sec=ACCEL_WIN_SEC)
    events, vetoed = [], {"EUROPE": 0, "bear_squeeze": 0, "no_mark": 0, "tail_7d": 0}
    for a in anchors:
        ts = int(a.anchor_ts_ms)
        if ts + HORIZONS_MIN[-1] * 60_000 > t_hi:
            vetoed["tail_7d"] += 1; continue
        sess = session_name(ts)
        if sess == "EUROPE":
            vetoed["EUROPE"] += 1; continue
        eth1h = ret_bps(marks_eth, ts - 3_600_000, ts)
        btc4h = ret_bps(marks_btc, ts - 4 * 3_600_000, ts)
        if eth1h is not None and btc4h is not None and eth1h < -20.0 and btc4h < -50.0:
            vetoed["bear_squeeze"] += 1; continue
        m = marks_eth.at_or_after(ts)
        if not m:
            vetoed["no_mark"] += 1; continue
        events.append({"ts": ts, "px0": float(m[1]), "notional": float(a.running_notional),
                       "liq_count": int(a.running_liq_count), "accel": a.acceleration_bucket,
                       "dominance": float(a.running_single_liq_dominance),
                       "session": sess,
                       "hour": datetime.fromtimestamp(ts/1000, tz=timezone.utc).hour,
                       "dow": datetime.fromtimestamp(ts/1000, tz=timezone.utc).weekday()})
    return events, vetoed


def compute_paths(marks, ev):
    """1m path 0..2h + 5m path 0..7D (SHORT perspektifi: adverse=+ret)."""
    t0, p0 = ev["ts"], ev["px0"]
    p1 = []
    for j in range(1, 121):
        p = px_at(marks, t0 + j * 60_000)
        p1.append((p - p0) / p0 * 1e4 if p else None)
    p5 = []
    for j in range(1, 2017):
        p = px_at(marks, t0 + j * 300_000)
        p5.append((p - p0) / p0 * 1e4 if p else None)
    ev["path1m"], ev["path5m"] = p1, p5
    # event-window high (T0'dan itibaren gorulen en yuksek — causal seri)
    ev["ehigh_bps_1m"] = np.maximum.accumulate(
        np.array([x if x is not None else -1e9 for x in p1]))


def short_outcome(ev, horizon_min, sl_bps=None, entry_delay_min=0.0, entry_bps_adj=0.0):
    """SHORT sonucu: net = -(ret) - FEE; sl_bps verilirse adverse>=sl'de cikis.
    entry_bps_adj: girisin T0 fiyatina gore bps farki (delayed/pullback girisler)."""
    path = ev["path1m"] if horizon_min <= 120 else ev["path5m"]
    step = 1 if horizon_min <= 120 else 5
    i_ent = int(entry_delay_min / step)
    i_exit = int(horizon_min / step)
    if i_exit > len(path):
        return None
    base = entry_bps_adj
    seg = path[i_ent:i_exit]
    if not seg or any(v is None for v in seg[:1]):
        return None
    seg = [v for v in seg if v is not None]
    if not seg:
        return None
    rel = [v - base for v in seg]                       # giris fiyatina gore
    adverse = np.maximum.accumulate(rel)
    mfe = float(-np.minimum.accumulate(rel).min())      # short lehine max
    mae = float(-max(0.0, adverse.max()))               # short aleyhine (negatif)
    t_mfe = int(np.argmin(np.minimum.accumulate(rel)) + 1) * step
    stop_hit = None
    if sl_bps is not None:
        hit = np.where(adverse >= sl_bps)[0]
        if len(hit):
            stop_hit = int(hit[0] + 1) * step
            net = -(rel[hit[0]]) - FEE                  # yaklastirma: bar degerinde cikis
            return {"net": net, "mfe": mfe, "mae": mae, "t_mfe": t_mfe,
                    "stop_hit_min": stop_hit, "gross": -(rel[hit[0]])}
    net = -(rel[-1]) - FEE
    return {"net": net, "mfe": mfe, "mae": mae, "t_mfe": t_mfe,
            "stop_hit_min": None, "gross": -(rel[-1])}


# ── T0 FEATURE'LARI (backward-only) ──────────────────────────────────────────
def t0_features(conn, marks_eth, marks_btc, ev):
    t0 = ev["ts"]; f = {}
    for w in PRE_WINDOWS_MIN:
        f[f"ret_{w}m"] = ret_bps(marks_eth, t0 - w * 60_000, t0)
    # rv30m
    pxs = [px_at(marks_eth, t0 - k * 300_000) for k in range(6, -1, -1)]
    f["rv30m"] = (math.sqrt(sum(math.log(pxs[i+1]/pxs[i])**2 for i in range(6)))
                  if all(p and p > 0 for p in pxs) else None)
    pxs24 = [px_at(marks_eth, t0 - k * 3_600_000) for k in range(24, -1, -1)]
    f["rv24h"] = (math.sqrt(sum(math.log(pxs24[i+1]/pxs24[i])**2 for i in range(24)))
                  if all(p and p > 0 for p in pxs24) else None)
    f["compression"] = (f["rv30m"] / f["rv24h"] * math.sqrt(48)
                        if f["rv30m"] and f["rv24h"] and f["rv24h"] > 0 else None)
    # rolling high/low mesafeleri (backward pencereler)
    p0 = ev["px0"]
    for wname, wmin in (("4h", 240), ("24h", 1440), ("7d", 10080)):
        seg = marks_eth.slice_range(t0 - wmin * 60_000, t0)
        if seg:
            hi = max(p for _, p in seg); lo = min(p for _, p in seg)
            f[f"dist_high_{wname}"] = (p0 - hi) / hi * 1e4
            f[f"dist_low_{wname}"] = (p0 - lo) / lo * 1e4
            f[f"range_pos_{wname}"] = (p0 - lo) / (hi - lo) if hi > lo else None
    # flow
    for w in (5, 15, 30, 60, 120):
        a = agg_flow(conn, "ETHUSDT", t0 - w * 60_000, t0)
        f[f"ofi_{w}m"] = a["ofi"]; f[f"rate_{w}m"] = a["cnt"] / w
        if w in (30, 120):
            f[f"large_share_{w}m"] = a["large_share"]
    f["cvd_slope"] = ((f["ofi_5m"] - f["ofi_30m"])
                      if f["ofi_5m"] is not None and f["ofi_30m"] is not None else None)
    f["aggr_decel"] = None
    a1 = agg_flow(conn, "ETHUSDT", t0 - 60_000, t0)
    a5 = agg_flow(conn, "ETHUSDT", t0 - 300_000, t0 - 60_000)
    if a5["buy"] > 0:
        f["aggr_decel"] = a1["buy"] / (a5["buy"] / 4.0)   # <1 = son 1m'de yavaslama
    # liq build-up
    for w in (30, 120, 1440):
        f[f"liq_buy_{w}m"] = liq_sum(conn, "ETHUSDT", "BUY", t0 - w * 60_000, t0 - 1000)
        f[f"liq_sell_{w}m"] = liq_sum(conn, "ETHUSDT", "SELL", t0 - w * 60_000, t0)
    f["btc_sync_10m"] = liq_sum(conn, "BTCUSDT", "BUY", t0 - 600_000, t0)
    f["btc_ret_1h"] = ret_bps(marks_btc, t0 - 3_600_000, t0)
    f["btc_ret_4h"] = ret_bps(marks_btc, t0 - 4 * 3_600_000, t0)
    f["btc_ret_7d"] = ret_bps(marks_btc, t0 - 7 * 86_400_000, t0)
    # leverage (mevcut olanlar)
    fr0 = funding_at(conn, t0); fr8 = funding_at(conn, t0 - 8 * 3_600_000)
    f["funding"] = fr0
    f["funding_slope_8h"] = (fr0 - fr8) if fr0 is not None and fr8 is not None else None
    # book L1
    b = book_features_at(conn, "ETHUSDT", t0, 30)
    if b:
        f["spread_bps"] = float(b.get("spread_pct") or 0) * 1e4 if b.get("spread_pct") is not None else None
        f["imbalance"] = b.get("book_imbalance")
        f["bid_depth"] = b.get("bid_depth_usd"); f["ask_depth"] = b.get("ask_depth_usd")
        b5 = book_features_at(conn, "ETHUSDT", t0 - 300_000, 30)
        if b5 and b5.get("bid_depth_usd"):
            f["bid_pull_5m"] = 1.0 - float(b.get("bid_depth_usd") or 0) / float(b5["bid_depth_usd"])
    # pre-event silence (T0'da BILINIR)
    f["pre_silence_10m"] = liq_first_ts(conn, "ETHUSDT", "BUY", t0 - 600_000, t0 - 1000, PROP_THRESH) is None
    g_backward_only(t0, t0)
    g_no_t0_silence(f.keys())
    ev["f"] = f


def silence_decompose(conn, ev):
    """Zamansal silence: her biri KENDI aninda bilinir; T0 feature'i DEGIL."""
    t0 = ev["ts"]
    first = liq_first_ts(conn, "ETHUSDT", "BUY", t0 + SIL_LO_MS, t0 + SIL_HI_MS, PROP_THRESH)
    ev["sil"] = {
        "pre_10m": ev["f"]["pre_silence_10m"],
        "s30s": liq_first_ts(conn, "ETHUSDT", "BUY", t0 + 1000, t0 + 30_000, PROP_THRESH) is None,
        "s1m": liq_first_ts(conn, "ETHUSDT", "BUY", t0 + 1000, t0 + 60_000, PROP_THRESH) is None,
        "s3m": first is None or first > t0 + 180_000,
        "s5m": first is None or first > t0 + 300_000,
        "s10m": first is None or first > t0 + 600_000,
        "full30m": first is None,                       # route silence v1 (T+30m'de bilinir)
        "breakdown_min": (first - t0) / 60_000 if first else None,
    }


# ── BOLUM A: LONG GENESIS ────────────────────────────────────────────────────
def genesis(marks, ev):
    t0, p0 = ev["ts"], ev["px0"]
    g_backward_only(t0, t0)
    seg = marks.slice_range(t0 - 48 * 3_600_000, t0)
    out = {"anchor_rule": "local_low_48h(primary,frozen)"}
    if len(seg) < 10:
        out["available"] = False; ev["gen"] = out; return
    ts_a = np.array([t for t, _ in seg]); px_a = np.array([p for _, p in seg])
    i_lo = int(px_a.argmin())
    start_ts, start_px = int(ts_a[i_lo]), float(px_a[i_lo])
    strength = (p0 - start_px) / start_px * 1e4
    age_h = (t0 - start_ts) / 3_600_000
    out.update({"available": True, "long_start_ts": start_ts,
                "long_age_h": round(age_h, 2), "long_strength_bps": round(strength, 1)})
    # accel: swing'in son %25'inin katkisi
    t_75 = start_ts + 0.75 * (t0 - start_ts)
    p75 = px_at(marks, t_75)
    if p75 and strength != 0:
        late = (p0 - p75) / start_px * 1e4
        out["long_accel"] = round(late / strength, 2) if strength > 0 else None
    else:
        out["long_accel"] = None
    ev["gen"] = out


def maturity_class(ev):
    f, g = ev["f"], ev["gen"]
    tf = ev["tf"]
    exh = 0
    if f.get("ret_60m") is not None and f["ret_60m"] <= 0: exh += 1
    if f.get("ofi_30m") is not None and f["ofi_30m"] <= 0: exh += 1
    if f.get("aggr_decel") is not None and f["aggr_decel"] < 1.0: exh += 1
    if g.get("long_accel") is not None and g["long_accel"] < 0.25: exh += 1
    ev["exhaustion_score"] = exh
    age = g.get("long_age_h") or 0; s = g.get("long_strength_bps") or 0
    d1 = tf.get("1D", "UNKNOWN")
    r4 = f.get("ret_240m")
    checks = {
        "C_EXHAUSTION": g.get("available") and age > 4 and exh >= 2,
        "E_HTF_DOWN_BOUNCE": d1 == "DOWN" and r4 is not None and r4 > 0,
        "D_HTF_UP_PULLBACK": d1 == "UP" and r4 is not None and r4 < 0,
        "A_NEW_EXPANSION": g.get("available") and age <= 4 and s > 100,
        "B_MATURE_TREND": g.get("available") and 4 < age <= 48 and s > 200,
        "F_RANGE": f.get("ret_1440m") is not None and abs(f["ret_1440m"]) < 100,
    }
    for c in MATURITY_ORDER:
        if checks.get(c):
            ev["maturity"] = c; return
    ev["maturity"] = "F_RANGE" if checks.get("F_RANGE") is not None else "UNKNOWN"


# ── BOLUM B: GIRIS VARYANTLARI ───────────────────────────────────────────────
def entry_variants(conn, marks_btc, ev):
    """Her varyant: (fill_delay_min, entry_bps_adj) veya None=missed. Yalniz o ana
    kadar bilinen bilgiler (event high causal seri, delayed flow pencereleri)."""
    t0 = ev["ts"]; p1 = ev["path1m"]; eh = ev["ehigh_bps_1m"]
    V = {}
    for d in FIXED_DELAYS_S:
        dm = d / 60.0
        idx = max(0, int(dm) - 1)
        adj = 0.0 if d == 0 else (p1[idx] if idx < len(p1) and p1[idx] is not None else None)
        V[f"delay_{d}s"] = None if adj is None else (dm, adj)
    # price-based: event-high'dan pullback (causal high, trigger<=30m)
    for pb in PULLBACK_BPS:
        fill = None
        for j in range(1, 31):
            if p1[j-1] is None: continue
            g_event_high_causal(t0 + j * 60_000, t0 + j * 60_000)
            if eh[j-1] - p1[j-1] >= pb:                 # high'dan pb geri cekilme
                fill = (j, p1[j-1]); break
        V[f"pullback_{int(pb)}bps"] = fill
    # 1m lower-high confirmation: bar high'i onceki bar high'indan dusuk ilk bar
    fill = None
    for j in range(2, 31):
        if p1[j-1] is None or p1[j-2] is None: continue
        if p1[j-1] < p1[j-2] and eh[j-1] == eh[j-2]:    # yeni high yok + dusus
            fill = (j, p1[j-1]); break
    V["lower_high_1m"] = fill
    # local low break: T0 sonrasi ilk 5m'in low'unu kiran ilk bar
    fill = None
    lows5 = [v for v in p1[:5] if v is not None]
    if lows5:
        lo5 = min(lows5)
        for j in range(6, 31):
            if p1[j-1] is not None and p1[j-1] < lo5:
                fill = (j, p1[j-1]); break
    V["low_break"] = fill
    # rejection close: high'a +5bps yaklasip altinda kapanan ilk 1m bar
    fill = None
    for j in range(2, 31):
        if p1[j-1] is None: continue
        if eh[j-1] - p1[j-1] >= 5.0 and p1[j-1] < (p1[j-2] if p1[j-2] is not None else 1e9):
            fill = (j, p1[j-1]); break
    V["rejection_close"] = fill
    # flow-based (her tetik kendi aninda olculur)
    sil = ev["sil"]
    a_pre = agg_flow(conn, "ETHUSDT", t0 - 300_000, t0)
    trig = None
    for j in (1, 2, 3, 5):
        a = agg_flow(conn, "ETHUSDT", t0 + (j-1) * 60_000, t0 + j * 60_000)
        if a_pre["buy"] > 0 and a["buy"] < a_pre["buy"] / 5.0 * 0.5:
            trig = j; break
    V["buy_decel"] = (trig, p1[trig-1]) if trig and p1[trig-1] is not None else None
    trig = None
    for j in (1, 2, 3, 5, 10):
        a = agg_flow(conn, "ETHUSDT", t0, t0 + j * 60_000)
        if a["ofi"] is not None and a["ofi"] < 0:
            trig = j; break
    V["ofi_flip"] = (trig, p1[trig-1]) if trig and p1[trig-1] is not None else None
    V["silence_3m_entry"] = ((3, p1[2]) if sil["s3m"] and p1[2] is not None else None)
    b0 = ev["f"].get("spread_bps")
    trig = None
    if b0 is not None:
        for j in (2, 5, 10, 15):
            b = book_features_at(conn, "ETHUSDT", t0 + j * 60_000, 60)
            if b and b.get("spread_pct") is not None and float(b["spread_pct"]) * 1e4 <= b0:
                trig = j; break
    V["spread_norm"] = (trig, p1[trig-1]) if trig and p1[trig-1] is not None else None
    btc15 = ret_bps(marks_btc, t0 - 600_000, t0 + 300_000)
    V["btc_weak_5m"] = ((5, p1[4]) if btc15 is not None and btc15 < 0 and p1[4] is not None else None)
    # frozen kombinasyonlar (4)
    V["rejection+silence3m"] = V["rejection_close"] if (V["rejection_close"] and sil["s3m"]) else None
    V["lowerhigh+ofiflip"] = V["lower_high_1m"] if (V["lower_high_1m"] and V["ofi_flip"]) else None
    no_new_high = eh[4] == eh[0] if len(eh) > 4 else False
    V["delay5m+no_new_high"] = V["delay_300s"] if (V["delay_300s"] and no_new_high) else None
    V["silence3m+btc_weak"] = V["silence_3m_entry"] if (V["silence_3m_entry"] and V["btc_weak_5m"]) else None
    ev["entries"] = V


# ── METRIKLER ────────────────────────────────────────────────────────────────
def stat_block(rows, span_days):
    """rows: list of outcome dicts (net/mfe/mae/t_mfe/stop_hit_min)."""
    if not rows:
        return {"n": 0}
    nets = np.array([r["net"] for r in rows])
    srt = np.sort(nets)[::-1]
    eq = np.cumsum(nets); dd = eq - np.maximum.accumulate(eq)
    g = nets[nets > 0].sum(); l = -nets[nets <= 0].sum()
    maes = np.array([r["mae"] for r in rows]); mfes = np.array([r["mfe"] for r in rows])
    return {"n": len(nets), "trades_per_day": round(len(nets) / span_days, 2) if span_days else None,
            "wr": round(float((nets > 0).mean()) * 100, 1),
            "median": round(float(np.median(nets)), 1), "mean": round(float(nets.mean()), 2),
            "cum": round(float(nets.sum()), 0),
            "top1_removed": round(float(srt[1:].sum()), 0) if len(nets) > 1 else None,
            "top3_removed": round(float(srt[3:].sum()), 0) if len(nets) > 3 else None,
            "top5_removed": round(float(srt[5:].sum()), 0) if len(nets) > 5 else None,
            "pf": round(float(g / l), 2) if l > 0 else None,
            "mdd": round(float(dd.min()), 0), "avg_dd": round(float(dd.mean()), 1),
            "worst": round(float(nets.min()), 1),
            "bottom3_cum": round(float(np.sort(nets)[:3].sum()), 1),
            "mfe_med": round(float(np.median(mfes)), 1), "mae_med": round(float(np.median(maes)), 1),
            "mae_p10": round(float(np.percentile(maes, 10)), 1),
            "t_mfe_med": round(float(np.median([r["t_mfe"] for r in rows])), 1),
            "stop_rate": round(float(np.mean([1 if r["stop_hit_min"] else 0 for r in rows])), 3)}


def no_overlap(evs, horizon_min, delay_min=0.0):
    out, busy = [], -1
    for e in evs:
        t_ent = e["ts"] + delay_min * 60_000
        if t_ent < busy:
            continue
        out.append(e); busy = t_ent + horizon_min * 60_000
    return out


def subset_perm_p(all_nets: np.ndarray, sub_mean: float, k: int, rng) -> float:
    if k < 2 or k >= len(all_nets):
        return float("nan")
    draws = np.array([all_nets[rng.choice(len(all_nets), k, replace=False)].mean()
                      for _ in range(N_PERM)])
    return float((draws >= sub_mean).mean())


# ── ANA AKIS ─────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rng = np.random.RandomState(SEED)
    print("=== C-BUY-FADE Structural Research ===")
    reg = ResearchRegistry()
    spec = freeze_spec(reg)
    print(f"  prereg FROZEN: {spec.experiment_id} hash={spec.frozen_hash}")

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000")
    print("  mark index yukleniyor (ETH+BTC)...")
    marks_eth = load_mark_index(conn, "ETHUSDT")
    marks_btc = load_mark_index(conn, "BTCUSDT")

    # NOT_AVAILABLE denetimi
    oi_n = conn.execute("SELECT COUNT(*) FROM open_interest WHERE ts_ms < ?",
                        (marks_eth.ts[-1] - 86_400_000,)).fetchone()[0]
    spot = conn.execute("SELECT MIN(ts_ms), MAX(ts_ms), COUNT(*) FROM spot_prices").fetchone()
    span = marks_eth.ts[-1] - marks_eth.ts[0]
    spot_cov = ((spot[1] - spot[0]) / span) if spot and spot[0] else 0.0
    NA = {"oi_slope_accel": f"NOT_AVAILABLE (historical rows={oi_n}; poller 2026-07-02'de basladi)",
          "price_oi_divergence": "NOT_AVAILABLE (ayni neden)",
          "spot_perp_participation": "NOT_AVAILABLE (spot_prices hacim icermiyor)",
          "basis": ("PARTIAL" if spot_cov >= 0.6 else "NOT_AVAILABLE") + f" (kapsama~{spot_cov:.0%}, Haz-05..Tem-02 bosluk)",
          "depth_beyond_L1": "NOT_AVAILABLE (book_ticker L1-only)",
          "impact_per_aggressive_dollar": "NOT_AVAILABLE (L1-only; L2 gerekli)"}
    print(f"  NOT_AVAILABLE: {list(NA.keys())}")

    events, vetoed = build_events(conn, marks_eth, marks_btc)
    print(f"  events: {len(events)}  vetoed={vetoed}")
    if len(events) < MIN_TOTAL:
        print("  INSUFFICIENT_SAMPLE — rapor yaziliyor, kriter gevsetilmiyor")

    for i, ev in enumerate(events):
        compute_paths(marks_eth, ev)
        t0_features(conn, marks_eth, marks_btc, ev)
        silence_decompose(conn, ev)
        genesis(marks_eth, ev)
        ev["tf"] = {tf: "PENDING" for tf in TF_DEFS}
        if (i + 1) % 25 == 0:
            print(f"    features {i+1}/{len(events)}", flush=True)

    # TF state'leri (bar cache'leri bir kez kur)
    print("  TF bars/states...")
    t_lo, t_hi = marks_eth.ts[0], marks_eth.ts[-1]
    bars = {tf: Bars(marks_eth, ms, t_lo, t_hi) for tf, ms in TF_DEFS.items()}
    for ev in events:
        ev["tf"] = {tf: g_unknown_not_neutral(bars[tf].state_at(ev["ts"], TF_THR_BPS[tf]))
                    for tf in TF_DEFS}
        maturity_class(ev)

    for ev in events:
        entry_variants(conn, marks_btc, ev)

    # splitler (purge'lu)
    events.sort(key=lambda e: e["ts"])
    n = len(events)
    split_of = {}
    bounds = {k: (int(n * a), int(n * b)) for k, (a, b) in SPLITS.items()}
    for k, (a, b) in bounds.items():
        for e in events[a:b]:
            split_of[e["ts"]] = k
    # purge: val/untouched basindaki, onceki split son event'ine 24h'den yakin event'ler dusulur
    purged = 0
    for k in ("val", "untouched"):
        a, _ = bounds[k]
        if a == 0: continue
        prev_last = events[a-1]["ts"]
        for e in events[a:]:
            if split_of[e["ts"]] != k: break
            if e["ts"] - prev_last < PURGE_MS:
                split_of[e["ts"]] = "PURGED"; purged += 1
            else:
                break
    ev_split = {k: [e for e in events if split_of[e["ts"]] == k] for k in SPLITS}
    span_days = {k: ((v[-1]["ts"] - v[0]["ts"]) / 86_400_000 if len(v) > 1 else 1.0)
                 for k, v in ev_split.items()}
    print(f"  splits: " + " ".join(f"{k}={len(v)}" for k, v in ev_split.items()) + f" purged={purged}")

    R = {"spec_hash": spec.frozen_hash, "n_events": n, "vetoed": vetoed,
         "not_available": NA, "purged": purged,
         "splits": {k: len(v) for k, v in ev_split.items()},
         "p_note": "dashboard mc_p=0.010 HARDCODED; 100-perm tabaniyla tutarli; burada N_PERM=20000",
         "silence_version": SILENCE_VERSION}

    # ── baseline replay (route: T0/45m/SL75) ─────────────────────────────────
    def replay(evs, silence_only=None, hold=45, sl=SL_BPS, delay=0.0):
        rows = []
        for e in no_overlap(evs, hold, delay):
            if silence_only is True and not e["sil"]["full30m"]: continue
            if silence_only is False and e["sil"]["full30m"]: continue
            o = short_outcome(e, hold, sl_bps=sl, entry_delay_min=delay)
            if o: rows.append(o)
        return rows

    base = {}
    for k, evs in ev_split.items():
        base[k] = {"ALL_T0_45m_SL75": stat_block(replay(evs), span_days[k]),
                   "SILENCE_T0_45m_SL75": stat_block(replay(evs, silence_only=True), span_days[k]),
                   "NOISY_T0_45m_SL75": stat_block(replay(evs, silence_only=False), span_days[k])}
    R["baseline"] = base
    print(f"  baseline train ALL: {base['train']['ALL_T0_45m_SL75']}")

    # ── BOLUM A: genesis/maturity ────────────────────────────────────────────
    secA = {}
    for k, evs in ev_split.items():
        by = {}
        for m in MATURITY_ORDER + ["UNKNOWN"]:
            sub = [e for e in evs if e["maturity"] == m]
            lbl = g_tiny_cell(len(sub), m)
            rows = replay(sub)
            by[m] = {"n_events": len(sub), "cell": lbl, "stats": stat_block(rows, span_days[k])}
        secA[k] = by
    # ana soru: silence-basari olgun/yorgun LONG'dan mi? (train'de test, val'de rapor)
    for k in ("train", "val", "untouched"):
        evs = ev_split[k]
        sil_ev = [e for e in evs if e["sil"]["full30m"]]
        mature = [e for e in sil_ev if e["maturity"] in ("C_EXHAUSTION", "B_MATURE_TREND")]
        young = [e for e in sil_ev if e["maturity"] == "A_NEW_EXPANSION"]
        secA[f"{k}_silence_maturity"] = {
            "silence_n": len(sil_ev),
            "mature_exhausted": stat_block(replay(mature), span_days[k]),
            "new_expansion": stat_block(replay(young), span_days[k])}
    # genesis alan raporu
    secA["genesis_fields"] = {"reported": ["estimated_long_start_ts", "time_from_long_start_to_event",
                              "long_age", "long_strength_at_event", "long_acceleration_at_event",
                              "long_exhaustion_score", "higher_tf_alignment"],
                              "long_age_med_h": round(float(np.median([e["gen"].get("long_age_h") or 0
                                                                        for e in events])), 1),
                              "exhaustion_dist": {str(s): sum(1 for e in events if e["exhaustion_score"] == s)
                                                  for s in range(5)}}
    R["A_genesis"] = secA

    # ── BOLUM B: giris varyantlari ───────────────────────────────────────────
    all_variants = ([f"delay_{d}s" for d in FIXED_DELAYS_S] +
                    [f"pullback_{int(p)}bps" for p in PULLBACK_BPS] +
                    ["lower_high_1m", "low_break", "rejection_close",
                     "buy_decel", "ofi_flip", "silence_3m_entry", "spread_norm", "btc_weak_5m"] +
                    COMBOS)
    secB = {}
    t0_nets = {}
    for k, evs in ev_split.items():
        vb = {}
        for vname in all_variants:
            rows = []; missed = 0
            for e in no_overlap(evs, 45):
                ent = e["entries"].get(vname)
                if ent is None:
                    missed += 1; continue
                dmin, adj = ent
                o = short_outcome(e, 45 + dmin if False else 45, sl_bps=SL_BPS,
                                  entry_delay_min=dmin, entry_bps_adj=adj)
                if o: rows.append(o)
            st = stat_block(rows, span_days[k])
            tot = len(no_overlap(evs, 45))
            st["fill_rate"] = round(len(rows) / tot, 2) if tot else None
            st["missed_rate"] = round(missed / tot, 2) if tot else None
            vb[vname] = st
            if k == "train" and vname == "delay_0s":
                t0_nets[k] = rows
        secB[k] = vb
    # random-delay control (train)
    ctrl = []
    for _ in range(2000):
        rows = []
        for e in no_overlap(ev_split["train"], 45):
            d = rng.uniform(0, 30)
            idx = max(0, int(d) - 1)
            adj = e["path1m"][idx] if idx < 120 and e["path1m"][idx] is not None else None
            if adj is None: continue
            o = short_outcome(e, 45, sl_bps=SL_BPS, entry_delay_min=d, entry_bps_adj=adj)
            if o: rows.append(o["net"])
        if rows: ctrl.append(float(np.mean(rows)))
    secB["random_delay_control_train"] = {"mean_of_means": round(float(np.mean(ctrl)), 2),
                                          "p95": round(float(np.percentile(ctrl, 95)), 2)}
    # timing family-p (train'de en iyi varyanti sec, val'de dogrula) — max-stat
    tr_means = {v: secB["train"][v].get("mean") for v in all_variants
                if secB["train"][v].get("n", 0) >= MIN_CELL}
    best_timing = max(tr_means, key=lambda v: tr_means[v]) if tr_means else None
    fam_p = None
    if best_timing:
        g_train_only_selection("train")
        base_nets = np.array([o["net"] for o in replay(ev_split["train"])])
        obs = tr_means[best_timing] - float(base_nets.mean())
        null = []
        for _ in range(N_PERM):
            bi = rng.choice(len(base_nets), len(base_nets), replace=True)
            null.append(float(base_nets[bi].mean() - base_nets.mean()))
        fam_p = float((np.array(null) >= obs).mean())
    secB["best_timing_train"] = {"variant": best_timing, "family_p_vs_T0": fam_p}
    R["B_entries"] = secB
    print(f"  B: best_timing_train={best_timing} family_p={fam_p}")

    # ── BOLUM C: horizon expectancy curves ───────────────────────────────────
    secC = {}
    for k, evs in ev_split.items():
        curve = {}
        for h in HORIZONS_MIN:
            rows = [short_outcome(e, h) for e in no_overlap(evs, h)]
            rows = [r for r in rows if r]
            recl = [1 if max([v for v in (e["path1m"] if h <= 120 else e["path5m"])[:max(1,int(h/(1 if h<=120 else 5)))] if v is not None] or [0]) > 0 else 0
                    for e in no_overlap(evs, h)]
            st = stat_block(rows, span_days[k])
            st["event_high_reclaim_rate"] = round(float(np.mean(recl)), 2) if recl else None
            curve[f"h{h}m"] = st
        secC[k] = curve
    # outcome path siniflari (evaluation-only; model girdisi DEGIL)
    def path_class(e):
        o45 = short_outcome(e, 45); o240 = short_outcome(e, 240); o1440 = short_outcome(e, 1440)
        if not o45 or not o240: return "NO_DATA"
        if o45["net"] > 20 and o45["t_mfe"] <= 15: c = "IMMEDIATE_FADE"
        elif o45["net"] > 20: c = "DELAYED_FADE"
        elif o45["net"] > 0 and o240["net"] <= 0: c = "SCALP_ONLY"
        elif o240["net"] > 30: c = "MULTI_HOUR_SHORT"
        elif o1440 and o1440["net"] > 50: c = "DAILY_SHORT_CONTINUATION"
        elif o45["mae"] < -SL_BPS: c = "FAILED_FADE"
        elif o45["mfe"] > 20 and o45["net"] < -10: c = "WHIPSAW"
        elif o1440 and o1440["net"] < -100: c = "SHORT_TO_LONG_REVERSAL"
        else: c = "NO_DIRECTION"
        return c
    for k, evs in ev_split.items():
        cls = {}
        for e in evs:
            c = path_class(e)
            cls[c] = cls.get(c, 0) + 1
        secC[f"{k}_path_classes"] = cls
    # H1-H5 hipotezleri (train'de kural, val/untouched'ta rapor)
    def hyp_eval(evs, k):
        out = {}
        h1 = [e for e in evs if e["tf"].get("1D") == "UP" and e["exhaustion_score"] >= 2]
        out["H1_1DUP_exh_scalp"] = {"n": len(h1),
            "h45": stat_block([o for o in (short_outcome(e, 45, sl_bps=SL_BPS) for e in no_overlap(h1, 45)) if o], span_days[k]),
            "h240": stat_block([o for o in (short_outcome(e, 240) for e in no_overlap(h1, 240)) if o], span_days[k])}
        h2 = [e for e in evs if e["tf"].get("4h") == "DOWN" and e["sil"]["full30m"]]
        out["H2_4hDOWN_silence_multihour"] = {"n": len(h2),
            "h45": stat_block([o for o in (short_outcome(e, 45, sl_bps=SL_BPS) for e in no_overlap(h2, 45)) if o], span_days[k]),
            "h240": stat_block([o for o in (short_outcome(e, 240) for e in no_overlap(h2, 240)) if o], span_days[k]),
            "h720": stat_block([o for o in (short_outcome(e, 720) for e in no_overlap(h2, 720)) if o], span_days[k])}
        h3 = [e for e in evs if (e["gen"].get("long_age_h") or 0) > 12
              and (e["f"].get("aggr_decel") or 9) < 1.0 and e["sil"]["s3m"]]
        out["H3_oldlong_decel_delayed"] = {"n": len(h3),
            "T0": stat_block([o for o in (short_outcome(e, 45, sl_bps=SL_BPS) for e in no_overlap(h3, 45)) if o], span_days[k]),
            "delay5m": stat_block([o for o in (short_outcome(e, 45, sl_bps=SL_BPS, entry_delay_min=5,
                                   entry_bps_adj=(e["path1m"][4] or 0)) for e in no_overlap(h3, 45)
                                   if e["path1m"][4] is not None) if o], span_days[k])}
        h4 = [e for e in evs if e["maturity"] == "A_NEW_EXPANSION" and e["tf"].get("1D") == "UP"]
        out["H4_newlong_htfup_fail"] = {"n": len(h4),
            "h45": stat_block([o for o in (short_outcome(e, 45, sl_bps=SL_BPS) for e in no_overlap(h4, 45)) if o], span_days[k])}
        h5 = []
        for e in evs:
            eh30 = e["ehigh_bps_1m"][29] if len(e["ehigh_bps_1m"]) > 29 else None
            p30 = e["path1m"][29] if len(e["path1m"]) > 29 else None
            a = agg_flow(conn, "ETHUSDT", e["ts"] + 600_000, e["ts"] + 1_800_000)
            btc = e["f"].get("btc_ret_1h")
            if eh30 is not None and p30 is not None and p30 < eh30 - 10 and \
               (a["ofi"] is None or a["ofi"] < 0.1) and btc is not None and btc < 0:
                h5.append(e)
        out["H5_noreclaim_nobuy_btcweak_2h12h"] = {"n": len(h5),
            "h120": stat_block([o for o in (short_outcome(e, 120, entry_delay_min=30,
                                entry_bps_adj=(e["path1m"][29] or 0)) for e in no_overlap(h5, 120)
                                if e["path1m"][29] is not None) if o], span_days[k]),
            "h720": stat_block([o for o in (short_outcome(e, 720, entry_delay_min=30,
                                entry_bps_adj=(e["path1m"][29] or 0)) for e in no_overlap(h5, 720)
                                if e["path1m"][29] is not None) if o], span_days[k])}
        return out
    secC["hypotheses"] = {k: hyp_eval(ev_split[k], k) for k in ("train", "val", "untouched")}
    R["C_horizons"] = secC

    # ── BOLUM D: multi-TF context matrix ─────────────────────────────────────
    def combo_of(e):
        t = e["tf"]
        if any(t[x] == "UNKNOWN" for x in ("1m", "5m", "15m", "1h", "4h", "1D")):
            return "7_UNKNOWN"
        if all(t[x] == "DOWN" for x in ("1m", "5m", "15m", "1h", "4h", "1D")): return "1_all_down"
        if all(t[x] == "DOWN" for x in ("1m", "5m", "15m", "1h")) and \
           t["4h"] == "UP" and t["1D"] == "UP": return "2_ltf_down_htf_up"
        if all(t[x] == "UP" for x in ("1m", "5m", "15m")) and t["1h"] in ("RANGE", "DOWN"): return "3_ltf_up_mid_weak"
        if t["1h"] == "DOWN" and t["4h"] == "RANGE" and t["1D"] == "UP": return "4_1hD_4hR_1DU"
        if t["4h"] == "DOWN" and t["1D"] == "DOWN": return "5_4h1D_down"
        if t["1D"] == "UP" and t["1W"] == "UP": return "6_1D1W_up"
        return "7_conflict"
    secD = {"tf_state_dist": {tf: {s: sum(1 for e in events if e["tf"][tf] == s)
                                   for s in ("UP", "DOWN", "RANGE", "UNKNOWN")} for tf in TF_DEFS}}
    for k, evs in ev_split.items():
        cells = {}
        for e in evs:
            cells.setdefault(combo_of(e), []).append(e)
        out = {}
        for cname, ces in sorted(cells.items()):
            lbl = g_tiny_cell(len(ces), cname)
            if lbl == "INSUFFICIENT_SAMPLE":
                out[cname] = {"n": len(ces), "cell": "INSUFFICIENT_SAMPLE"}
                continue
            curve = {f"h{h}m": stat_block([o for o in (short_outcome(e, h) for e in no_overlap(ces, h)) if o],
                                          span_days[k]) for h in (45, 240, 720, 1440)}
            best_h = max(curve, key=lambda h: curve[h].get("mean") or -1e9)
            out[cname] = {"n": len(ces), "curve": curve, "best_h_by_mean": best_h,
                          "stop75_rate_45m": stat_block([o for o in (short_outcome(e, 45, sl_bps=SL_BPS)
                                                        for e in no_overlap(ces, 45)) if o], span_days[k]).get("stop_rate")}
        secD[k] = out
    R["D_context_matrix"] = secD

    # ── SILENCE DECOMPOSITION + matched controls ─────────────────────────────
    secS = {}
    for k, evs in ev_split.items():
        out = {}
        for key in ("pre_10m", "s30s", "s1m", "s3m", "s5m", "s10m", "full30m"):
            sub = [e for e in evs if e["sil"][key]]
            rows = replay(sub)
            out[key] = {"n_events": len(sub), "stats": stat_block(rows, span_days[k]),
                        "knowable_at": {"pre_10m": "T0", "s30s": "T+30s", "s1m": "T+1m",
                                        "s3m": "T+3m", "s5m": "T+5m", "s10m": "T+10m",
                                        "full30m": "T+30m (route v1 — T0 girisi icin KULLANILAMAZ)"}[key]}
        bmins = [e["sil"]["breakdown_min"] for e in evs if e["sil"]["breakdown_min"] is not None]
        out["breakdown_min_dist"] = {"n": len(bmins),
                                     "median": round(float(np.median(bmins)), 1) if bmins else None,
                                     "p25_p75": [round(float(np.percentile(bmins, q)), 1) for q in (25, 75)] if bmins else None}
        # silence bozulunca edge bitiyor mu: breakdown sonrasi kalan 45m PnL
        secS[k] = out
    # matched controls (train+val birlesik, 1-NN, silence vs non-silence)
    def mc_features(e):
        return np.array([e["hour"], math.log(max(e["notional"], 1)),
                         (e["f"].get("rv30m") or 0) * 1e4, e["f"].get("spread_bps") or 0,
                         (e["f"].get("ret_240m") or 0) / 100, (e["f"].get("btc_ret_4h") or 0) / 100,
                         e["f"].get("liq_buy_1440m", 0) / 1e6])
    pool_ev = ev_split["train"] + ev_split["val"]
    sil_ev = [e for e in pool_ev if e["sil"]["full30m"]]
    non_ev = [e for e in pool_ev if not e["sil"]["full30m"]]
    if len(sil_ev) >= 5 and len(non_ev) >= len(sil_ev):
        F_non = np.array([mc_features(e) for e in non_ev])
        mu, sd = F_non.mean(0), F_non.std(0) + 1e-9
        used = set(); pairs = []
        for e in sil_ev:
            d = np.linalg.norm((F_non - mc_features(e)) / sd, axis=1)
            for j in np.argsort(d):
                if j not in used:
                    used.add(int(j)); pairs.append((e, non_ev[int(j)])); break
        sil_rows = [short_outcome(e, 45, sl_bps=SL_BPS) for e, _ in pairs]
        ctl_rows = [short_outcome(c, 45, sl_bps=SL_BPS) for _, c in pairs]
        sil_rows = [r for r in sil_rows if r]; ctl_rows = [r for r in ctl_rows if r]
        d_mean = (np.mean([r["net"] for r in sil_rows]) - np.mean([r["net"] for r in ctl_rows])
                  if sil_rows and ctl_rows else None)
        # hour-only kontrol: silence yerine hour-esiti rastgele altkume
        all45 = [o for o in (short_outcome(e, 45, sl_bps=SL_BPS) for e in no_overlap(pool_ev, 45)) if o]
        all_nets = np.array([r["net"] for r in all45])
        p_sub = subset_perm_p(all_nets, float(np.mean([r["net"] for r in sil_rows])), len(sil_rows), rng) if sil_rows else None
        secS["matched_controls"] = {"n_pairs": len(pairs),
                                    "silence_mean": round(float(np.mean([r['net'] for r in sil_rows])), 2) if sil_rows else None,
                                    "matched_nonsilence_mean": round(float(np.mean([r['net'] for r in ctl_rows])), 2) if ctl_rows else None,
                                    "diff": round(float(d_mean), 2) if d_mean is not None else None,
                                    "subset_perm_p_vs_all": p_sub,
                                    "match_dims": ["hour", "log_size", "rv30m", "spread", "ret4h", "btc4h", "density24h"]}
    R["silence"] = secS

    # ── MANAGEMENT ───────────────────────────────────────────────────────────
    secM = {}
    # time-stop esikleri TRAIN'de frozen kural: checkpoint'te train median MFE
    g_train_only_selection("train")
    tr45 = [short_outcome(e, 45, sl_bps=SL_BPS) for e in no_overlap(ev_split["train"], 45)]
    tr45 = [r for r in tr45 if r]
    ts_thr = {}
    for cp in TIME_STOP_CHECKS_MIN:
        mfes_cp = []
        for e in no_overlap(ev_split["train"], 45):
            seg = [v for v in e["path1m"][:cp] if v is not None]
            if seg: mfes_cp.append(-min(seg))
        ts_thr[cp] = round(float(np.median(mfes_cp)), 1) if mfes_cp else 0.0
    secM["time_stop_thresholds_train_frozen"] = ts_thr

    def managed(evs, k, hold=45, lock=None, tstop=None, struct=None):
        rows = []
        for e in no_overlap(evs, hold):
            o = short_outcome(e, hold, sl_bps=SL_BPS)
            if not o: continue
            path = e["path1m"][:hold] if hold <= 120 else e["path5m"][:int(hold/5)]
            step = 1 if hold <= 120 else 5
            net = o["net"]; exited = False
            fav = [-v if v is not None else None for v in path]   # short lehine
            if tstop:
                cp, thr = tstop
                seg = [v for v in fav[:int(cp/step)] if v is not None]
                if seg and max(seg) < thr:
                    v = fav[int(cp/step)-1]
                    if v is not None: net = v - FEE; exited = True
            if lock and not exited:
                trig, lk = lock
                hitp = [i for i, v in enumerate(fav) if v is not None and v >= trig]
                if hitp:
                    after = fav[hitp[0]:]
                    give = [i for i, v in enumerate(after) if v is not None and v <= lk]
                    if give: net = lk - FEE; exited = True
            if struct == "ehigh_reclaim" and not exited:
                eh = e["ehigh_bps_1m"]
                rec = [i for i in range(1, min(hold, 120)) if e["path1m"][i] is not None
                       and e["path1m"][i] > max(0.0, eh[0])]
                if rec:
                    v = fav[rec[0]]
                    if v is not None: net = v - FEE; exited = True
            if struct == "silence_breakdown" and not exited:
                bm = e["sil"]["breakdown_min"]
                if bm is not None and bm < hold:
                    v = fav[int(bm/step)] if int(bm/step) < len(fav) else None
                    if v is not None: net = v - FEE; exited = True
            rows.append({**o, "net": net})
        return stat_block(rows, span_days[k])

    for k in ("train", "val", "untouched"):
        evs = ev_split[k]; out = {}
        for h in FIXED_HOLDS_MIN:
            out[f"fixed_{h}m"] = stat_block([o for o in (short_outcome(e, h, sl_bps=SL_BPS)
                                            for e in no_overlap(evs, h)) if o], span_days[k])
        for cp in TIME_STOP_CHECKS_MIN:
            out[f"tstop_{cp}m_thr{ts_thr[cp]}"] = managed(evs, k, tstop=(cp, ts_thr[cp]))
        for trig, lk in LOCK_CANDIDATES:
            out[f"lock_{int(trig)}_{int(lk)}"] = managed(evs, k, lock=(trig, lk))
        out["struct_ehigh_reclaim"] = managed(evs, k, struct="ehigh_reclaim")
        out["struct_silence_breakdown"] = managed(evs, k, struct="silence_breakdown")
        secM[k] = out
    # SHORT->LONG transition (ayri iddia, yuksek standart — yalniz rapor)
    trans = {}
    for k in ("train", "val"):
        evs = ev_split[k]; rows_exit, rows_long = [], []
        for e in no_overlap(evs, 240):
            # kosul: 45m icinde yeni dip yok + event high reclaim
            p45 = [v for v in e["path1m"][:45] if v is not None]
            if not p45: continue
            reclaimed = max(p45) > 0 and min(p45) > -30
            if reclaimed:
                o_l = short_outcome(e, 240, entry_delay_min=45,
                                    entry_bps_adj=(e["path1m"][44] or 0))
                if o_l:
                    rows_long.append({**o_l, "net": -o_l["net"] - 2 * FEE})  # LONG = short'un tersi, cift fee
        trans[k] = {"ENTER_LONG_after_reclaim_4h": stat_block(rows_long, span_days[k]),
                    "note": "reverse iddiasi icin kanit standardi yuksek; yalniz rapor"}
    secM["short_to_long"] = trans
    R["management"] = secM

    # ── kontrol setleri ozeti + verdicts ─────────────────────────────────────
    verdicts = {}
    va = ev_split["val"]; ua = ev_split["untouched"]
    # A: pre-event structure predictive?
    tr_mat = {m: secA["train"][m]["stats"].get("mean") for m in MATURITY_ORDER
              if secA["train"][m]["stats"].get("n", 0) >= MIN_CELL}
    if not tr_mat:
        verdicts["A"] = "INSUFFICIENT_SAMPLE"
    else:
        best_m = max(tr_mat, key=lambda m: tr_mat[m])
        vst = secA["val"][best_m]["stats"]; ust = secA["untouched"][best_m]["stats"]
        ok = (vst.get("n", 0) >= MIN_CELL and (vst.get("mean") or -9) > (base["val"]["ALL_T0_45m_SL75"].get("mean") or 0)
              and (ust.get("mean") or -9) > 0)
        verdicts["A"] = ("PRE_EVENT_STRUCTURE_PREDICTIVE:" + best_m) if ok else "PRE_EVENT_STRUCTURE_NON_PREDICTIVE"
    # B: timing
    if best_timing:
        vt = secB["val"].get(best_timing, {}); ut = secB["untouched"].get(best_timing, {})
        vb = base["val"]["ALL_T0_45m_SL75"].get("mean") or 0
        ok = ((vt.get("mean") or -9) > vb and (vt.get("n") or 0) >= MIN_CELL
              and (fam_p is not None and fam_p < 0.05) and (ut.get("mean") or -9) > 0)
        verdicts["B"] = ("TIMING_INCREMENTAL:" + best_timing) if ok else "TIMING_NON_INCREMENTAL"
    else:
        verdicts["B"] = "INSUFFICIENT_SAMPLE"
    # C: horizon rejimi (train best-h val'de tutuyor mu; scalp vs continuation)
    trc = {h: secC["train"][h].get("mean") for h in secC["train"]
           if secC["train"][h].get("n", 0) >= MIN_CELL}
    best_h = max(trc, key=lambda h: trc[h]) if trc else None
    if best_h:
        vh = secC["val"].get(best_h, {})
        hm = int(best_h[1:-1])
        cls = ("SCALP_ONLY" if hm <= 60 else "MULTI_HOUR_CONTINUATION" if hm <= 720 else "DAILY_CONTINUATION")
        verdicts["C"] = (cls if (vh.get("mean") or -9) > 0 else "SCALP_ONLY(default; val'de tasima yok)")
        verdicts["C_best_h_train"] = best_h
    else:
        verdicts["C"] = "INSUFFICIENT_SAMPLE"
    # silence
    mcs = R["silence"].get("matched_controls", {})
    if mcs.get("n_pairs", 0) >= MIN_CELL and mcs.get("diff") is not None:
        p_sub_v = mcs.get("subset_perm_p_vs_all")
        verdicts["SILENCE"] = ("SILENCE_INCREMENTAL_INFO_T30M" if (mcs["diff"] > 0 and
                               p_sub_v is not None and p_sub_v < 0.05)
                               else "SILENCE_PROXY_ONLY_OR_WEAK")
    else:
        verdicts["SILENCE"] = "INSUFFICIENT_SAMPLE"
    # management
    trm = {v: secM["train"][v].get("mean") for v in secM["train"]
           if isinstance(secM["train"][v], dict) and secM["train"][v].get("n", 0) >= MIN_CELL}
    base_tr = secM["train"].get("fixed_45m", {}).get("mean")
    best_mgmt = max(trm, key=lambda v: trm[v]) if trm else None
    if best_mgmt and base_tr is not None:
        vm = secM["val"].get(best_mgmt, {}); vb45 = secM["val"].get("fixed_45m", {}).get("mean") or 0
        ok = (trm[best_mgmt] > base_tr and (vm.get("mean") or -9) > vb45 and (vm.get("n") or 0) >= MIN_CELL)
        verdicts["MANAGEMENT"] = ("MANAGEMENT_INCREMENTAL:" + best_mgmt) if ok else "MANAGEMENT_NON_INCREMENTAL"
    else:
        verdicts["MANAGEMENT"] = "INSUFFICIENT_SAMPLE"
    R["verdicts"] = verdicts
    print(f"  VERDICTS: {json.dumps(verdicts, ensure_ascii=False)}")

    OJ.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    _write_md(R, spec)
    conn.close(); reg.close()
    return R


def _write_md(R, spec):
    lines = ["# C-BUY-FADE Structural Path Research", "",
             f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — prereg `{spec.experiment_id}` "
             f"hash `{spec.frozen_hash}` · route AYNEN korundu · {R['n_events']} event "
             f"(veto: {R['vetoed']}) · splits {R['splits']} (purged {R['purged']})", "",
             f"**Silence version:** {R['silence_version']}", "",
             f"**p=0.010 denetimi:** {R['p_note']}", "",
             "## NOT_AVAILABLE features", "```json", json.dumps(R["not_available"], indent=1), "```", "",
             "## Verdicts", "```json", json.dumps(R["verdicts"], indent=1), "```", ""]
    for sec in ("baseline", "A_genesis", "B_entries", "C_horizons", "D_context_matrix",
                "silence", "management"):
        lines += [f"## {sec}", "```json", json.dumps(R[sec], indent=1, default=str), "```", ""]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD: {OM}")


if __name__ == "__main__":
    main()
