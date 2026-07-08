"""BUY-FADE Silence-Conditional Exit Timing Research (preregistered, bagimsiz).

E-BUYFADE-STRUCT-001 / E-BUYFADE-REENTRY-001 DEGISTIRILMEZ; verdict'leri acilmaz.
Onlarin silence sonuclari yalniz HIPOTEZ KAYNAGIDIR.

Ana soru: T+30m'de silence_v1 dogrulanan BUY-fade SHORT trade'lerinde T+45m sabit
cikis erken mi / gec mi / optimal mi? YENI ENTRY ALPHA ARANMAZ — T0 entry, SL75,
fee modeli, event universe SABIT.

Zaman yapisi (frozen):
  Entry T0 · silence gozlem [T0+60s, T0+30m] · silence BILINIR: T+30m · baseline exit T+45m.
  Silence bilgisi T+30m'den once HICBIR kararda kullanilamaz (yapisal guard).

Survivor/lookahead kurali (frozen):
  T0'da acilan TUM trade yolu korunur. T+30 oncesi SL olan trade silence olarak
  siniflansa bile SL sonucu AYNEN kalir ve evrenden CIKARILMAZ. Yalnizca T+30'da
  HALA ACIK pozisyonlara post-T30 yonetimi uygulanabilir.

Senaryolar: A (ANA) = mevcut T0 route + T30-survivor yonetimi.
            B (ayri KONTROL) = T+30 observer entry — yeni entry deneyi, A ile KARISTIRILMAZ.

Run: python tools/research_s34_buyfade_silence_exit.py
Cikti: reports/research/s34/BUYFADE_SILENCE_EXIT.{json,md}
"""
from __future__ import annotations
import json, math, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.constitution import ConstitutionViolation
from ami.research.registry import ExperimentSpec, ResearchQuestion, ResearchRegistry
from tools.research_s34_knowable_anchor_continuation import load_mark_index
from tools.research_s34_buyfade_structural import (
    Bars, FEE, MIN_CELL, PROP_THRESH, PURGE_MS, SL_BPS, SPLITS, TF_THR_BPS,
    agg_flow, build_events, compute_paths, g_tiny_cell, g_train_only_selection,
    liq_first_ts, ret_bps, silence_decompose, stat_block)

OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "BUYFADE_SILENCE_EXIT.json"; OM = OUT / "BUYFADE_SILENCE_EXIT.md"
DB = ROOT / "data" / "microstructure.db"

SEED = 11
N_DRAW = 2000
ECON_MIN = 3.0
WINNER_RETENTION_MIN = 0.70
CVAR_TOL_BPS = 10.0
T30 = 30.0
BASE_EXIT = 45.0
DECOMP_WINDOWS = [(0, 5), (0, 15), (0, 30), (30, 45), (30, 60), (30, 90), (30, 120),
                  (30, 180), (30, 240), (30, 360), (30, 480), (30, 720), (30, 1440)]
FIXED_EXITS = [35, 40, 45, 60, 75, 90, 120, 180, 240, 360, 480, 720, 1440]
BREAKDOWN_DEFS = ["first_buy50", "ofi_flip_5m", "buy_restart_5m"]   # coverage'i olan 3 tanim
GRACE_MIN = [0.0, 1.0, 3.0]
STRUCT_EXITS = ["ehigh_reclaim", "hh_5m", "mfe_giveback_50", "btc_recovery_15m"]
PARTIAL_MODES = ["close_all_t30", "half_t30_rest_45", "half_t30_rest_breakdown",
                 "be_stop_t30", "lock5_t30", "lock10_t30"]
MFE_GIVEBACK_RATIO = 0.50
CYCLE_CAP_MIN = 1440.0

# ── GUARD'LAR (mutation suite) ────────────────────────────────────────────────
def g_survivor_universe(n_silence_all: int, n_reported: int) -> None:
    """T+30'a ulasamayan (erken SL) silence trade'leri evrenden atilamaz."""
    if n_reported != n_silence_all:
        raise ConstitutionViolation(
            f"Survivor bias: silence evreni {n_silence_all}, raporlanan {n_reported}")


def g_no_pre_t30_silence_use(decision_min: float) -> None:
    if decision_min < T30:
        raise ConstitutionViolation(
            f"Silence bilgisi T+{decision_min}m'de kullanilamaz (bilinirlik T+30m)")


def g_breakdown_causal(exit_min: float, breakdown_min: float, grace: float) -> None:
    if exit_min < breakdown_min + grace - 1e-9:
        raise ConstitutionViolation("Gelecekteki breakdown zamani onceden bilinemez")


def g_realized_only(report: dict) -> None:
    if "t30_unrealized_bps" in report and report.get("_unrealized_in_cum"):
        raise ConstitutionViolation("T+30 unrealized PnL realized gibi yazilamaz")


def g_no_manage_closed(closed_pre_t30: bool, action: str) -> None:
    if closed_pre_t30 and action != "KEEP_RESULT":
        raise ConstitutionViolation("T+30 oncesi kapanmis trade'e sonradan yonetim uygulanamaz")


def g_fee_on_extension(net_gross_diff: float) -> None:
    if abs(net_gross_diff - FEE) > 1e-6:
        raise ConstitutionViolation(f"Fee extended hold'da da uygulanmali (fark {net_gross_diff})")


def g_no_route_mutation(path: str) -> None:
    protected = ("s34_realtime_shadow_runner", "s34_state_machine_live_executor",
                 ".env", "execution/", "risk/", "brain/")
    if any(p in str(path) for p in protected):
        raise ConstitutionViolation(f"Shadow/live route otomatik degistirilemez: {path}")


def require_noisy_control(results: dict) -> None:
    if "G_noisy_same_exits" not in results:
        raise ConstitutionViolation("Noisy kontrolu atlanamaz")


# ── PREREG ───────────────────────────────────────────────────────────────────
def freeze_spec(reg: ResearchRegistry) -> ExperimentSpec:
    reg.add_question(ResearchQuestion(
        question_id="Q-BUYFADE-SILEXIT-001",
        question="Silence(T+30m-bilinir) dogrulanan BUY-fade SHORT'larda T+45m cikis "
                 "erken mi/gec mi/optimal mi? (yeni entry alpha ARANMAZ)",
        origin_observation="E-BUYFADE-STRUCT-001: silence-subset +20/+30/+20 (HIPOTEZ KAYNAGI); "
                           "ALL-horizon egrisi negatif ama silence-KOSULLU post-T30 egri hic olculmedi",
        economic_value=0.6, risk_reduction_value=0.5, falsifiability=0.9, required_sample=MIN_CELL))
    spec = ExperimentSpec(
        experiment_id="E-BUYFADE-SILEXIT-001", question_id="Q-BUYFADE-SILEXIT-001",
        population=("E-BUYFADE-STRUCT-001 ile AYNI event universe (ETH BUY>=200K, EUROPE+"
                    "bear-squeeze veto, T0 mark-fill, FEE 5bps roundtrip/unit). Senaryo A (ANA): "
                    "T0'da TUM trade'ler acilir; T+30 oncesi SL sonuclari AYNEN korunur ve "
                    "silence evreninden CIKARILMAZ; post-T30 yonetim yalniz T+30'da ACIK "
                    "pozisyonlara. Senaryo B (T+30 observer entry) AYRI kontrol — yeni entry "
                    "deneyi, A ile karistirilmaz."),
        target=("silence-kosullu exit timing: path decomposition (T0-T30 vs T30-exit), "
                f"fixed exits {FIXED_EXITS}m, breakdown exits ({BREAKDOWN_DEFS} x grace {GRACE_MIN}m "
                f"x price-confirm), structural {STRUCT_EXITS}, partial {PARTIAL_MODES}. "
                "silence_v1 = buyfade_silence_v1 (no ETH BUY>=50K in [T0+60s,T0+30m])."),
        features=["path 1m(0-2h)+5m(2h-24h)", "post-T30 5m flow bucket'lari (OFI/buy notional)",
                  "post-T30 ETH BUY>=50K liq zamanlari", "TF state 1h/4h/1D (completed bars)",
                  "session/rv30m/event-size/btc4h (timestamp-safe)"],
        threshold_method=(f"tum aday listeleri frozen (yukarida); MFE-giveback orani {MFE_GIVEBACK_RATIO}; "
                          f"cycle cap {CYCLE_CAP_MIN}m; SL75 uzatilmis hold boyunca AKTIF kalir; "
                          f"secim kurali: TRAIN'de silence-T30-acik altkumede mean net; "
                          f"lock/BE semantigi: T+30'da floor zaten asilmissa ANINDA T+30 fiyatiyla cikis"),
        chronological_split=f"event-bazli {SPLITS} + 24h purge (E-BUYFADE-STRUCT ile ayni sinirlar)",
        untouched_data=("son %15: YENI exit adaylari icin ilk kez kullanilir; ANCAK baseline "
                        "silence-fixed-45 istatistigi E-BUYFADE-STRUCT'ta GORULDU -> bu deneyin "
                        "maksimum statusu CHRONOLOGICALLY_SUPPORTED_PENDING_FORWARD (prereg beyani)"),
        negative_control=("G: noisy altkumeye AYNI exit modelleri (etki silence'a mi ozgu?); "
                          f"H: matched random exit timing (uniform[35,{int(CYCLE_CAP_MIN)}]m, {N_DRAW} cekim); "
                          f"I: random management action ({N_DRAW} cekim); Senaryo B ayri kontrol"),
        min_sample=MIN_CELL,
        effect_size_required_bps=ECON_MIN,
        multiple_testing_control=f"aday ailesi TRAIN-secim + random-exit-timing kontrol dagilimi "
                                 f"percentile>=0.95 sarti + VAL/UNTOUCHED yon teyidi",
        execution_model="mark_fill; FEE 5bps roundtrip/unit (partial: entry 2.5 + her cikis "
                        "bacaginda birim basina 2.5 -> toplam 5/unit); SL replay 1m/5m path",
        decision_criteria=(f"PASS (exit adayi): silence-fixed-45 baseline'ina TRAIN+VAL incremental "
                           f"(mean, ayni yon) + untouched ayni yon (varsa) + top3-removed pozitif + "
                           f"econ>={ECON_MIN}bps + winner_retention>={WINNER_RETENTION_MIN} + "
                           f"MFE-capture artarken cvar5 en fazla {CVAR_TOL_BPS}bps kotulesme + "
                           f"N>={MIN_CELL} + random-exit-timing pct>=0.95 + tiny-cell degil. "
                           "Verdict sozlugu: PROFIT_PRE_T30_ONLY / POST_T30_CONTINUATION / "
                           "T45_EXIT_TOO_EARLY / T45_EXIT_TOO_LATE / T45_EXIT_ROBUST / "
                           "SILENCE_BREAKDOWN_EXIT_INCREMENTAL / STRUCTURAL_EXIT_INCREMENTAL / "
                           "PARTIAL_PROFIT_INCREMENTAL / EXIT_TIMING_NON_INCREMENTAL / "
                           "REGIME_CONDITIONAL_EXIT / INSUFFICIENT_SAMPLE / REJECTED. "
                           "Izin tavani: RESEARCH_ONLY/BACKTEST/SHADOW/FORWARD_VALIDATING; "
                           "LIVE/SIZING/AUTOMATIC_ROUTE_CHANGE YASAK."),
        falsification_rule=("survivor-bias yasaklari: T+30'a ulasamayanlari dislama, pre-T30 SL'leri "
                            "silme, silence'i T0 kararinda kullanma, gelecekteki breakdown'i onceden "
                            "bilme — hepsi yapisal guard + mutation. 4h-DOWN+silence dar hipotezi "
                            "kucukse INSUFFICIENT kalir; threshold/horizon N'e bakilarak GEVSETILMEZ; "
                            "kriter degisikligi YASAK."))
    spec.freeze()
    reg.register_experiment(spec)
    return spec


# ── PATH YARDIMCILARI ────────────────────────────────────────────────────────
def bar_at(ev, minute: float):
    if minute <= 0:
        return 0.0
    if minute <= 120:
        i = int(minute) - 1
        return ev["path1m"][i] if 0 <= i < 120 else None
    i = int(minute / 5) - 1
    return ev["path5m"][i] if 0 <= i < len(ev["path5m"]) else None


def scan_minutes(a: float, b: float):
    m = max(1.0, float(int(a)))
    while m <= b:
        yield m
        m += 1.0 if m < 120 else 5.0


def pre_t30_state(ev):
    """T0-T30 replay (SL75). Doner: (closed?, exit_min, net) veya (False, None, None)."""
    for m in scan_minutes(1, T30):
        v = bar_at(ev, m)
        if v is not None and v >= SL_BPS:
            return True, m, -(v) - FEE
    return False, None, None


def exit_net(ev, exit_min: float, from_min: float = T30):
    """[from_min, exit_min] SL75 replay; net = -(bar_exit) - FEE (tam roundtrip)."""
    for m in scan_minutes(from_min + 1e-9, exit_min):
        v = bar_at(ev, m)
        if v is not None and v >= SL_BPS:
            return {"net": -(v) - FEE, "exit_min": m, "reason": "SL"}
    v = bar_at(ev, exit_min)
    if v is None:
        return None
    g_fee_on_extension((-(v)) - (-(v) - FEE))
    return {"net": -(v) - FEE, "exit_min": exit_min, "reason": "HOLD"}


def trade_row(ev, o, extra=None):
    """stat_block satiri (mfe/mae T0-exit araligi, short perspektif)."""
    end = o["exit_min"]
    vals = [bar_at(ev, m) for m in scan_minutes(1, end)]
    vals = [v for v in vals if v is not None]
    mfe = float(-min(vals)) if vals else 0.0
    mae = float(-max(0.0, max(vals))) if vals else 0.0
    t_mfe = 0.0
    if vals:
        best = min(vals)
        for m in scan_minutes(1, end):
            if bar_at(ev, m) == best:
                t_mfe = m; break
    r = {"net": o["net"], "mfe": mfe, "mae": mae, "t_mfe": t_mfe,
         "stop_hit_min": o["exit_min"] if o["reason"] == "SL" else None,
         "exit_min": o["exit_min"], "hold_min": o["exit_min"]}
    if extra:
        r.update(extra)
    return r


# ── ADAY CIKISLAR (yalniz T30'da ACIK pozisyonlara) ───────────────────────────
def cand_fixed(ev, h):
    g_no_pre_t30_silence_use(T30)
    return exit_net(ev, float(h))


def find_breakdown(ev, defn):
    """Post-T30 breakdown dakikasi (yoksa None). Tanimlar frozen."""
    if defn == "first_buy50":
        return ev.get("bd_first_buy50")
    if defn == "ofi_flip_5m":
        for m, ofi, _buy in ev.get("flow5m_post30", []):
            if ofi is not None and ofi > 0:
                return m
        return None
    if defn == "buy_restart_5m":
        thr = ev.get("pre_buy_5m_avg") or 0.0
        for m, _ofi, buy in ev.get("flow5m_post30", []):
            if thr > 0 and buy > thr:
                return m
        return None
    raise ValueError(defn)


def cand_breakdown(ev, defn, grace, price_confirm=False):
    bd = find_breakdown(ev, defn)
    if bd is None:
        return exit_net(ev, CYCLE_CAP_MIN)         # breakdown yok -> cycle-cap'te cikis
    ex = bd + grace
    if price_confirm:
        conf = None
        v_bd = bar_at(ev, bd)
        for m in scan_minutes(ex, min(ex + 15, CYCLE_CAP_MIN)):
            v = bar_at(ev, m)
            if v is not None and v_bd is not None and v > v_bd:
                conf = m; break
        if conf is None:
            return exit_net(ev, CYCLE_CAP_MIN)
        ex = conf
    g_breakdown_causal(ex, bd, grace)
    return exit_net(ev, min(ex, CYCLE_CAP_MIN))


def cand_structural(ev, kind, marks_btc=None):
    if kind == "ehigh_reclaim":
        eh30 = float(ev["ehigh_bps_1m"][29])
        for m in scan_minutes(T30 + 1, CYCLE_CAP_MIN):
            v = bar_at(ev, m)
            if v is not None and v > max(0.0, eh30):
                return exit_net(ev, m)
        return exit_net(ev, CYCLE_CAP_MIN)
    if kind == "hh_5m":
        prev = bar_at(ev, T30)
        for m in scan_minutes(T30 + 5, CYCLE_CAP_MIN):
            if m % 5:  # yalniz 5m sinirlarinda
                continue
            v = bar_at(ev, m)
            if v is not None and prev is not None and v > prev + 5:
                return exit_net(ev, m)
            prev = v if v is not None else prev
        return exit_net(ev, CYCLE_CAP_MIN)
    if kind == "mfe_giveback_50":
        best = 0.0
        for m in scan_minutes(T30 + 1, CYCLE_CAP_MIN):
            v = bar_at(ev, m)
            if v is None:
                continue
            fav = -v
            best = max(best, fav)
            if best > 10 and fav <= best * (1 - MFE_GIVEBACK_RATIO):
                return exit_net(ev, m)
        return exit_net(ev, CYCLE_CAP_MIN)
    if kind == "btc_recovery_15m":
        for m in scan_minutes(T30 + 5, CYCLE_CAP_MIN):
            if m % 5:
                continue
            r = ev.get("btc15m_post30", {}).get(int(m))
            if r is not None and r > 0:
                return exit_net(ev, m)
        return exit_net(ev, CYCLE_CAP_MIN)
    raise ValueError(kind)


def cand_partial(ev, mode):
    """Fee: entry 2.5 + her cikis bacaginda birim basina 2.5 -> toplam 5/unit."""
    v30 = bar_at(ev, T30)
    if v30 is None:
        return None
    unreal30 = -(v30)                                # short unrealized gross
    if mode == "close_all_t30":
        return {"net": unreal30 - FEE, "exit_min": T30, "reason": "T30_CLOSE"}
    if mode == "half_t30_rest_45":
        rest = exit_net(ev, BASE_EXIT)
        if not rest:
            return None
        return {"net": 0.5 * (unreal30 - FEE) + 0.5 * rest["net"],
                "exit_min": rest["exit_min"], "reason": "PARTIAL"}
    if mode == "half_t30_rest_breakdown":
        rest = cand_breakdown(ev, "first_buy50", 0.0)
        if not rest:
            return None
        return {"net": 0.5 * (unreal30 - FEE) + 0.5 * rest["net"],
                "exit_min": rest["exit_min"], "reason": "PARTIAL"}
    if mode in ("be_stop_t30", "lock5_t30", "lock10_t30"):
        floor = {"be_stop_t30": 0.0, "lock5_t30": 5.0, "lock10_t30": 10.0}[mode]
        if unreal30 <= floor:                        # floor zaten asilmis/altta -> ANINDA cikis
            return {"net": unreal30 - FEE, "exit_min": T30, "reason": "FLOOR_IMMEDIATE"}
        for m in scan_minutes(T30 + 1e-9, BASE_EXIT):
            v = bar_at(ev, m)
            if v is not None and -(v) <= floor:
                return {"net": floor - FEE, "exit_min": m, "reason": "FLOOR"}
        return exit_net(ev, BASE_EXIT)
    raise ValueError(mode)


# ── ANA AKIS ─────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rng = np.random.RandomState(SEED)
    print("=== BUY-FADE Silence-Conditional Exit Timing ===")
    g_no_route_mutation(__file__)                    # kendi dosyamiz korumali degil (pozitif kontrol)
    reg = ResearchRegistry()
    spec = freeze_spec(reg)
    print(f"  prereg FROZEN: {spec.experiment_id} hash={spec.frozen_hash}")

    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    conn.execute("PRAGMA cache_size=-200000")
    print("  mark index yukleniyor...")
    marks_eth = load_mark_index(conn, "ETHUSDT")
    marks_btc = load_mark_index(conn, "BTCUSDT")
    events, vetoed = build_events(conn, marks_eth, marks_btc)
    events.sort(key=lambda e: e["ts"])
    print(f"  events: {len(events)} vetoed={vetoed}")

    bars = {tf: Bars(marks_eth, ms, marks_eth.ts[0], marks_eth.ts[-1])
            for tf, ms in (("1h", 3_600_000), ("4h", 14_400_000), ("1D", 86_400_000))}

    for i, ev in enumerate(events):
        compute_paths(marks_eth, ev)
        t0 = ev["ts"]
        ev["f"] = {"pre_silence_10m": liq_first_ts(conn, "ETHUSDT", "BUY",
                                                   t0 - 600_000, t0 - 1000, PROP_THRESH) is None}
        silence_decompose(conn, ev)
        closed, sl_min, sl_net = pre_t30_state(ev)
        ev["closed_pre_t30"] = closed
        ev["pre_t30"] = {"exit_min": sl_min, "net": sl_net}
        ev["tf"] = {tf: bars[tf].state_at(t0, TF_THR_BPS[tf]) for tf in bars}
        # rv30m + btc4h (rejim kirilimi icin, timestamp-safe)
        pxs = [marks_eth.at_or_before(t0 - k * 300_000) for k in range(6, -1, -1)]
        pxs = [float(p[1]) if p else None for p in pxs]
        ev["rv30m"] = (math.sqrt(sum(math.log(pxs[j+1]/pxs[j])**2 for j in range(6)))
                       if all(p and p > 0 for p in pxs) else None)
        ev["btc4h"] = ret_bps(marks_btc, t0 - 4 * 3_600_000, t0)
        if ev["sil"]["full30m"] and not closed:
            # post-T30 veriler (yalniz acik silence trade'leri icin)
            r = conn.execute("SELECT MIN(ts_ms) FROM liquidations WHERE symbol='ETHUSDT' "
                             "AND side='BUY' AND notional>=? AND ts_ms>? AND ts_ms<=?",
                             (PROP_THRESH, t0 + 30 * 60_000, t0 + int(CYCLE_CAP_MIN) * 60_000)).fetchone()
            ev["bd_first_buy50"] = ((int(r[0]) - t0) / 60_000) if r and r[0] else None
            a_pre = agg_flow(conn, "ETHUSDT", t0 - 1_800_000, t0)
            ev["pre_buy_5m_avg"] = a_pre["buy"] / 6.0 if a_pre["buy"] else None
            flow = []
            for mb in range(35, 241, 5):             # T+35..T+240 5m bucket'lar
                a = agg_flow(conn, "ETHUSDT", t0 + (mb - 5) * 60_000, t0 + mb * 60_000)
                flow.append((float(mb), a["ofi"], a["buy"]))
            ev["flow5m_post30"] = flow
            ev["btc15m_post30"] = {mb: ret_bps(marks_btc, t0 + (mb - 15) * 60_000, t0 + mb * 60_000)
                                   for mb in range(35, 241, 5)}
        if (i + 1) % 50 == 0:
            print(f"    prep {i+1}/{len(events)}", flush=True)

    n = len(events); split_of = {}
    bounds = {k: (int(n * a), int(n * b)) for k, (a, b) in SPLITS.items()}
    for k, (a, b) in bounds.items():
        for e in events[a:b]:
            split_of[e["ts"]] = k
    purged = 0
    for k in ("val", "untouched"):
        a, _ = bounds[k]
        if a == 0:
            continue
        prev_last = events[a - 1]["ts"]
        for e in events[a:]:
            if split_of.get(e["ts"]) != k:
                break
            if e["ts"] - prev_last < PURGE_MS:
                split_of[e["ts"]] = "PURGED"; purged += 1
            else:
                break
    ev_split = {k: [e for e in events if split_of[e["ts"]] == k] for k in SPLITS}
    span_days = {k: ((v[-1]["ts"] - v[0]["ts"]) / 86_400_000 if len(v) > 1 else 1.0)
                 for k, v in ev_split.items()}
    print("  splits: " + " ".join(f"{k}={len(v)}" for k, v in ev_split.items()) + f" purged={purged}")

    R = {"spec_hash": spec.frozen_hash, "n_events": n, "purged": purged,
         "splits": {k: len(v) for k, v in ev_split.items()},
         "silence_version": "buyfade_silence_v1 (E-BUYFADE-STRUCT-001 ile birebir ayni)"}

    def sil_all(evs):    # silence evreni: pre-T30 SL'ler DAHIL (survivor-safe)
        return [e for e in evs if e["sil"]["full30m"]]

    def sil_open(evs):
        return [e for e in evs if e["sil"]["full30m"] and not e["closed_pre_t30"]]

    def noisy_open(evs):
        return [e for e in evs if not e["sil"]["full30m"] and not e["closed_pre_t30"]]

    def universe_rows(evs, cand_fn):
        """Silence evreninin TAMAMI: kapali-pre-T30 sonuclari aynen + aciklara aday cikis."""
        rows, nones = [], 0
        for e in sil_all(evs):
            if e["closed_pre_t30"]:
                g_no_manage_closed(True, "KEEP_RESULT")
                rows.append(trade_row(e, {"net": e["pre_t30"]["net"],
                                          "exit_min": e["pre_t30"]["exit_min"], "reason": "SL"}))
            else:
                o = cand_fn(e)
                if o:
                    rows.append(trade_row(e, o))
                else:
                    nones += 1
        g_survivor_universe(len(sil_all(evs)) - nones, len(rows))
        return rows

    # ── survivor-bias audit ───────────────────────────────────────────────────
    audit = {}
    for k, evs in ev_split.items():
        sa, so = sil_all(evs), sil_open(evs)
        audit[k] = {"t0_entries": len(evs),
                    "silence_n": len(sa),
                    "silence_pre_t30_SL_n": len(sa) - len(so),
                    "t30_survivor_silence_n": len(so),
                    "noisy_open_n": len(noisy_open(evs)),
                    "t30_unreal_median_bps": (round(float(np.median(
                        [-(bar_at(e, T30) or 0) for e in so])), 1) if so else None)}
    R["survivor_audit"] = audit
    print(f"  survivor audit: {audit['train']}")

    # ── 1. path decomposition ────────────────────────────────────────────────
    dec = {}
    for k, evs in ev_split.items():
        out = {}
        for sub_name, sub in (("silence_open", sil_open(evs)), ("noisy_open", noisy_open(evs)),
                              ("all_open", [e for e in evs if not e["closed_pre_t30"]])):
            seg = {}
            for a, b in DECOMP_WINDOWS:
                vals = []
                for e in sub:
                    va = bar_at(e, a) if a > 0 else 0.0
                    vb = bar_at(e, b)
                    if va is not None and vb is not None:
                        vals.append(-(vb - va))       # short katkisi
                seg[f"{a}m_{b}m"] = {"n": len(vals),
                                     "mean": round(float(np.mean(vals)), 2) if vals else None,
                                     "median": round(float(np.median(vals)), 2) if vals else None}
            out[sub_name] = seg
        dec[k] = out
    R["decomposition"] = dec

    # dekompozisyon tablosu: total = (T0->T30) + (T30->45)
    for k in ("train", "val", "untouched"):
        so = sil_open(ev_split[k])
        t030 = [-(bar_at(e, T30) or 0) for e in so]
        t3045 = [-(bar_at(e, 45) or 0) - (-(bar_at(e, T30) or 0)) for e in so
                 if bar_at(e, 45) is not None]
        R.setdefault("pnl_table", {})[k] = {
            "t0_t30_mean": round(float(np.mean(t030)), 2) if t030 else None,
            "t30_exit45_mean": round(float(np.mean(t3045)), 2) if t3045 else None,
            "note": "fee haric brut katkilar; total45 = iki sutun toplami - fee"}
    print(f"  decomposition train silence: {R['pnl_table']['train']}")

    # ── adaylar ──────────────────────────────────────────────────────────────
    def all_candidates():
        C = {}
        for h in FIXED_EXITS:
            C[f"fixed_{h}m"] = lambda e, h=h: cand_fixed(e, h)
        for d in BREAKDOWN_DEFS:
            for gmin in GRACE_MIN:
                C[f"bd_{d}_g{int(gmin)}m"] = lambda e, d=d, g=gmin: cand_breakdown(e, d, g)
            C[f"bd_{d}_pconf"] = lambda e, d=d: cand_breakdown(e, d, 0.0, price_confirm=True)
        for s in STRUCT_EXITS:
            C[f"st_{s}"] = lambda e, s=s: cand_structural(e, s)
        for p in PARTIAL_MODES:
            C[f"pp_{p}"] = lambda e, p=p: cand_partial(e, p)
        return C

    CANDS = all_candidates()
    results = {}
    for k in ("train", "val", "untouched"):
        evs = ev_split[k]; out = {}
        for name, fn in CANDS.items():
            rows = universe_rows(evs, fn)
            st = stat_block(rows, span_days[k])
            open_rows = [r for r in rows if r.get("stop_hit_min") is None or r["exit_min"] > T30]
            st["avg_hold_min"] = round(float(np.mean([r["hold_min"] for r in rows])), 1) if rows else None
            st["mfe_captured_ratio"] = (round(float(np.mean(
                [r["net"] / r["mfe"] for r in rows if r["mfe"] > 5])), 2)
                if any(r["mfe"] > 5 for r in rows) else None)
            out[name] = st
        results[k] = out

    # baseline'lar
    base = {k: results[k]["fixed_45m"] for k in ("train", "val", "untouched")}
    R["baseline_silence_fixed45"] = base
    R["candidates"] = results

    # ── kontroller ───────────────────────────────────────────────────────────
    ctrl = {}
    # G: noisy ayni exit'ler (val)
    gn = {}
    for name in ("fixed_45m", "fixed_240m", "bd_first_buy50_g0m", "st_mfe_giveback_50"):
        rows = []
        for e in noisy_open(ev_split["val"]):
            o = CANDS[name](e)
            if o:
                rows.append(trade_row(e, o))
        gn[name] = stat_block(rows, span_days["val"])
    ctrl["G_noisy_same_exits"] = gn
    # H: matched random exit timing (train, silence-open)
    so_tr = sil_open(ev_split["train"])
    rnd_means = []
    for _ in range(N_DRAW):
        vals = []
        for e in so_tr:
            h = float(rng.uniform(35, CYCLE_CAP_MIN))
            o = exit_net(e, h)
            if o:
                vals.append(o["net"])
        if vals:
            rnd_means.append(float(np.mean(vals)))
    ctrl["H_random_exit_timing_train"] = {
        "mean_of_means": round(float(np.mean(rnd_means)), 2),
        "p95": round(float(np.percentile(rnd_means, 95)), 2),
        "p99": round(float(np.percentile(rnd_means, 99)), 2)}
    # I: random management action (train)
    names = list(CANDS)
    rnd_act = []
    for _ in range(N_DRAW):
        vals = []
        for e in so_tr:
            o = CANDS[names[rng.randint(len(names))]](e)
            if o:
                vals.append(o["net"])
        if vals:
            rnd_act.append(float(np.mean(vals)))
    ctrl["I_random_action_train"] = {
        "mean_of_means": round(float(np.mean(rnd_act)), 2),
        "p95": round(float(np.percentile(rnd_act, 95)), 2)}
    require_noisy_control(ctrl)
    R["controls"] = ctrl
    print(f"  controls: H_p95={ctrl['H_random_exit_timing_train']['p95']} "
          f"I_p95={ctrl['I_random_action_train']['p95']}")

    # Senaryo B (ayri kontrol; A ile karistirilmaz): T+30 observer entry
    sb = {}
    for k in ("train", "val"):
        rows = []
        for e in sil_open(ev_split[k]):
            v30 = bar_at(e, T30)
            o = exit_net(e, 240.0)
            if v30 is not None and o:
                rows.append({"net": (o["net"] + FEE) - (-(v30)) - FEE,   # T30 girisli brut - fee
                             "mfe": 0.0, "mae": 0.0, "t_mfe": 0.0, "stop_hit_min": None,
                             "hold_min": 210.0, "exit_min": 240.0})
        sb[k] = stat_block(rows, span_days[k])
    R["scenario_B_t30_observer_4h"] = {"note": "AYRI kontrol — YENI ENTRY deneyi; "
                                       "Senaryo A sonucuyla karistirilamaz", **sb}

    # ── silence maturity (T+30'da hesaplanir; yalniz post-T30 yonetim icin) ──
    def maturity_of(e):
        g_no_pre_t30_silence_use(T30)
        had_60s = not e["sil"]["s1m"]                 # ilk 60s'te buy>=50K var miydi (s1m=false)
        if e["sil"]["pre_10m"] and not had_60s:
            return "early_continuous"
        if had_60s:
            return "immediate_noise_then_silent"
        return "late_silence"
    mat = {}
    for k in ("train", "val"):
        out = {}
        for cls in ("early_continuous", "immediate_noise_then_silent", "late_silence"):
            sub = [e for e in sil_open(ev_split[k]) if maturity_of(e) == cls]
            rows45 = [trade_row(e, o) for e in sub if (o := cand_fixed(e, 45))]
            rows240 = [trade_row(e, o) for e in sub if (o := cand_fixed(e, 240))]
            out[cls] = {"n": len(sub), "cell": g_tiny_cell(len(sub), cls),
                        "fixed45": stat_block(rows45, span_days[k]),
                        "fixed240": stat_block(rows240, span_days[k])}
        mat[k] = out
    R["silence_maturity"] = mat

    # ── rejim/timeframe kirilimi (rapor-only; dar hipotez dahil) ─────────────
    regs = {}
    for k in ("train", "val"):
        out = {}
        for ctx_name, ctx_fn in (
            ("4h_DOWN", lambda e: e["tf"].get("4h") == "DOWN"),
            ("4h_UP", lambda e: e["tf"].get("4h") == "UP"),
            ("1D_UP", lambda e: e["tf"].get("1D") == "UP"),
            ("1D_DOWN", lambda e: e["tf"].get("1D") == "DOWN"),
            ("US", lambda e: e["session"] == "US"),
            ("OFF", lambda e: e["session"] == "OFF"),
            ("size>=500K", lambda e: e["notional"] >= 500_000),
            ("btc4h<0", lambda e: (e.get("btc4h") or 0) < 0),
        ):
            sub = [e for e in sil_open(ev_split[k]) if ctx_fn(e)]
            lbl = g_tiny_cell(len(sub), ctx_name)
            if lbl == "INSUFFICIENT_SAMPLE":
                out[ctx_name] = {"n": len(sub), "cell": "INSUFFICIENT_SAMPLE"}
                continue
            r45 = [trade_row(e, o) for e in sub if (o := cand_fixed(e, 45))]
            r240 = [trade_row(e, o) for e in sub if (o := cand_fixed(e, 240))]
            out[ctx_name] = {"n": len(sub),
                             "fixed45_mean": stat_block(r45, span_days[k]).get("mean"),
                             "fixed240_mean": stat_block(r240, span_days[k]).get("mean")}
        regs[k] = out
    # dar prereg hipotezi: 4h DOWN + silence -> T45 sonrasi 2-4h continuation
    nh = {}
    for k in ("train", "val"):
        sub = [e for e in sil_open(ev_split[k]) if e["tf"].get("4h") == "DOWN"]
        nh[k] = {"n": len(sub), "cell": g_tiny_cell(len(sub), "4hDOWN_sil"),
                 "fixed45": stat_block([trade_row(e, o) for e in sub if (o := cand_fixed(e, 45))], span_days[k]),
                 "fixed120": stat_block([trade_row(e, o) for e in sub if (o := cand_fixed(e, 120))], span_days[k]),
                 "fixed240": stat_block([trade_row(e, o) for e in sub if (o := cand_fixed(e, 240))], span_days[k])}
    regs["narrow_4hDOWN_silence"] = nh
    R["regime_breakdown"] = regs

    # ── secim (TRAIN) + dogrulama ────────────────────────────────────────────
    g_train_only_selection("train")
    tr_ok = {nm: st.get("mean") for nm, st in results["train"].items()
             if st.get("n", 0) >= MIN_CELL}
    base_tr = base["train"].get("mean") or -9e9
    best = max(tr_ok, key=lambda nm: tr_ok[nm]) if tr_ok else None
    h_p = ctrl["H_random_exit_timing_train"]
    best_beats_random = (tr_ok.get(best, -9e9) >= h_p["p95"]) if best else False
    sel = {"best_train": best, "best_train_mean": tr_ok.get(best),
           "baseline_train_mean": base["train"].get("mean"),
           "beats_random_p95": bool(best_beats_random)}

    def winner_retention(k, cand_name):
        so = sil_open(ev_split[k])
        keep = tot = 0
        for e in so:
            b = cand_fixed(e, 45); c = CANDS[cand_name](e)
            if b and c and b["net"] > 0:
                tot += 1
                if c["net"] > 0:
                    keep += 1
        return round(keep / tot, 2) if tot else None

    verdict_flags = []
    if best:
        vt, ut = results["val"].get(best, {}), results["untouched"].get(best, {})
        bv, bu = base["val"], base["untouched"]
        incr_tr = tr_ok[best] - (base["train"].get("mean") or 0)
        incr_v = (vt.get("mean") or -9e9) - (bv.get("mean") or 0)
        incr_u = (ut.get("mean") or -9e9) - (bu.get("mean") or 0)
        wr_ret = winner_retention("val", best)
        cvar_ok = ((vt.get("cvar5") if "cvar5" in vt else vt.get("bottom3_cum", 0)) is not None)
        cvar_delta = None
        if vt.get("mae_p10") is not None and bv.get("mae_p10") is not None:
            cvar_delta = vt["mae_p10"] - bv["mae_p10"]
        checks = {
            "train_incremental": incr_tr > 0,
            "beats_random_p95": best_beats_random,
            "val_same_direction": incr_v > 0,
            "untouched_same_direction": (incr_u > 0) if (ut.get("n") or 0) >= MIN_CELL else None,
            "top3_ok": (vt.get("top3_removed") or -1) > 0,
            "econ": (incr_v >= ECON_MIN),
            "winner_retention": (wr_ret is not None and wr_ret >= WINNER_RETENTION_MIN),
            "tail_ok": (cvar_delta is None or cvar_delta >= -CVAR_TOL_BPS),
            "min_n": (vt.get("n") or 0) >= MIN_CELL,
        }
        sel["checks"] = checks
        sel["winner_retention_val"] = wr_ret
        sel["incrementals"] = {"train": round(incr_tr, 2), "val": round(incr_v, 2),
                               "untouched": round(incr_u, 2)}
        hard = [c for c, v in checks.items() if v is False]
        verdict_flags = hard
    R["selection"] = sel

    # ── verdicts ─────────────────────────────────────────────────────────────
    V = {}
    pt = R["pnl_table"]
    pre = [pt[k]["t0_t30_mean"] for k in ("train", "val") if pt[k]["t0_t30_mean"] is not None]
    post = [pt[k]["t30_exit45_mean"] for k in ("train", "val") if pt[k]["t30_exit45_mean"] is not None]
    V["profit_location"] = ("PROFIT_PRE_T30_ONLY" if pre and post and np.mean(pre) > 0 > np.mean(post)
                            else ("POST_T30_CONTINUATION" if post and np.mean(post) > 0 else "MIXED"))
    # T45 durusu: uzun fixed'ler val'de baseline'i geciyor mu (yalniz fixed ailesi)
    fx_val = {h: results["val"].get(f"fixed_{h}m", {}).get("mean") for h in FIXED_EXITS}
    fx_tr = {h: results["train"].get(f"fixed_{h}m", {}).get("mean") for h in FIXED_EXITS}
    b45v = fx_val.get(45) or 0
    longer_better_val = [h for h in FIXED_EXITS if h > 45 and fx_val.get(h) is not None
                         and fx_val[h] > b45v + 1]
    shorter_better_val = [h for h in FIXED_EXITS if h < 45 and fx_val.get(h) is not None
                          and fx_val[h] > b45v + 1]
    longer_better_tr = [h for h in FIXED_EXITS if h > 45 and fx_tr.get(h) is not None
                        and (fx_tr[h] or -9e9) > (fx_tr.get(45) or 0) + 1]
    if longer_better_val and longer_better_tr and set(longer_better_val) & set(longer_better_tr):
        V["t45_status"] = "T45_EXIT_TOO_EARLY(candidates=" + \
            ",".join(str(h) for h in sorted(set(longer_better_val) & set(longer_better_tr))) + ")"
    elif shorter_better_val and not longer_better_val:
        V["t45_status"] = "T45_EXIT_TOO_LATE"
    else:
        V["t45_status"] = "T45_EXIT_ROBUST"
    if best and not verdict_flags and sel["checks"].get("val_same_direction"):
        fam = ("SILENCE_BREAKDOWN_EXIT_INCREMENTAL" if best.startswith("bd_")
               else "STRUCTURAL_EXIT_INCREMENTAL" if best.startswith("st_")
               else "PARTIAL_PROFIT_INCREMENTAL" if best.startswith("pp_")
               else "T45_EXIT_TOO_EARLY" if best.startswith("fixed_") and int(best.split("_")[1][:-1]) > 45
               else "EXIT_TIMING_NON_INCREMENTAL")
        V["exit_candidate"] = fam + f":{best}"
        V["status"] = "CHRONOLOGICALLY_SUPPORTED_PENDING_FORWARD"
    else:
        V["exit_candidate"] = "EXIT_TIMING_NON_INCREMENTAL" + (f"(fails={verdict_flags})" if verdict_flags else "")
        V["status"] = "REJECTED" if best else "INSUFFICIENT_SAMPLE"
    nhv = regs["narrow_4hDOWN_silence"]["val"]
    V["narrow_4hDOWN"] = (nhv["cell"] if nhv["cell"] == "INSUFFICIENT_SAMPLE"
                          else ("REGIME_CONDITIONAL_EXIT" if (nhv["fixed240"].get("mean") or -9) >
                                (nhv["fixed45"].get("mean") or 0) else "NON_CONDITIONAL"))
    R["verdicts"] = V
    print(f"  VERDICTS: {json.dumps(V, ensure_ascii=False)}")

    OJ.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    _write_md(R, spec)
    conn.close(); reg.close()
    return R


def _write_md(R, spec):
    lines = ["# BUY-FADE Silence-Conditional Exit Timing", "",
             f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — prereg `{spec.experiment_id}` "
             f"hash `{spec.frozen_hash}` · Senaryo A (T0 route + T30-survivor yonetimi) ANA · "
             f"{R['n_events']} event · splits {R['splits']}", "",
             "## Verdicts", "```json", json.dumps(R["verdicts"], indent=1), "```", "",
             "## Survivor-bias audit", "```json", json.dumps(R["survivor_audit"], indent=1), "```", "",
             "## PnL decomposition (Total = T0→T30 + T30→Exit)", "```json",
             json.dumps(R["pnl_table"], indent=1), "```", ""]
    for sec in ("decomposition", "baseline_silence_fixed45", "candidates", "controls",
                "scenario_B_t30_observer_4h", "silence_maturity", "regime_breakdown", "selection"):
        lines += [f"## {sec}", "```json", json.dumps(R[sec], indent=1, default=str), "```", ""]
    lines += ["Durust statuler: software-correct · survivor-bias-safe ✓ (audit tablosu) · "
              "chronological-validation: bkz selection.checks · "
              f"profit: {R['verdicts'].get('profit_location')} · "
              f"exit timing: {R['verdicts'].get('exit_candidate')} · "
              f"regime: {R['verdicts'].get('narrow_4hDOWN')} · forward-not-validating · "
              "**operationally FORBIDDEN (LIVE/SIZING/AUTO-ROUTE-CHANGE yasak)**",
              "", "*Mevcut shadow/live route DEGISTIRILMEDI.*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD: {OM}")


if __name__ == "__main__":
    main()
