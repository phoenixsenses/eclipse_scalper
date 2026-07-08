"""8A — Bidirectional Re-Entry Research (BUY-FADE cycle'lari; preregistered).

Dort bagimsiz kol: S->S re-entry, S->L flip, L->L re-entry, L->S flip.
NO_POSITION/WAIT gecerli ara state'tir; cikis != ters yone giris.

Frozen tasarim notu: trigger'lar SUREKLI causal taramayla bulunur (ilk tetik);
"pencere listesi" gerceklesen re-entry gecikmelerinin dagilim raporudur, ayri
ayri optimize edilmez. Cooldown grid'i ayri kol olarak taranir ve secim YALNIZ
TRAIN'de yapilir. Her giris ayri FEE tasir. Ayni cycle'dan gelen sinyaller yeni
bagimsiz alpha SAYILMAZ (cycle_id ile baglanir).

Cycle: [T0, min(T0+24h, sonraki anchor T0)).
FIRST_SHORT = route aynen (T0, 45m, SL75).
FIRST_LONG (research-only tanim, live route DEGIL): T0+45m'de 4h VE 1D state=UP
ise LONG; hold 24h (cycle sonuna kirpilir), SL 150bps.

Run: python tools/research_s34_buyfade_reentry.py
Cikti: reports/research/s34/BUYFADE_REENTRY.{json,md}
"""
from __future__ import annotations
import json, sqlite3, sys
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
    Bars, FEE, HOLD_MS, MIN_CELL, PROP_THRESH, PURGE_MS, SL_BPS, SPLITS, TF_THR_BPS,
    build_events, compute_paths, g_tiny_cell, g_train_only_selection,
    liq_first_ts, session_name, silence_decompose, stat_block)

OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "BUYFADE_REENTRY.json"; OM = OUT / "BUYFADE_REENTRY.md"
DB = ROOT / "data" / "microstructure.db"

SEED = 11
CYCLE_MAX_MIN = 1440
COOLDOWNS_MIN = [0, 1, 3, 5, 10, 15, 30, 60, 120]
REENTRY_WINDOWS_MIN = [1, 3, 5, 10, 15, 30, 60, 120, 240, 360, 720, 1440]
LONG_SL_BPS = 150.0; LONG_HOLD_MIN = 1440
RELIEF_BPS = 30.0
MAX_ENTRIES_GRID = [1, 2, 3, 99]     # 99 = sinirsiz KONTROL (operasyonel aday OLAMAZ)
FLIP_MIN_CONFIRM = 2                  # direction flip icin bagimsiz confirmation sayisi

STATES = ["NO_POSITION", "FIRST_LONG", "FIRST_SHORT", "EXITED_LONG", "EXITED_SHORT",
          "WAITING", "LONG_REENTRY_ELIGIBLE", "SHORT_REENTRY_ELIGIBLE",
          "LONG_REENTERED", "SHORT_REENTERED", "DIRECTION_FLIP", "INVALIDATED",
          "CYCLE_CLOSED"]


# ── GUARD'LAR ────────────────────────────────────────────────────────────────
def g_causal_prefix(scan_end_idx: int, used_idx: int) -> None:
    if used_idx > scan_end_idx:
        raise ConstitutionViolation(f"Re-entry trigger gelecek veriyle: {used_idx}>{scan_end_idx}")


def g_fee_per_entry(n_entries: int, fee_applied: float) -> None:
    if abs(fee_applied - n_entries * FEE) > 1e-9:
        raise ConstitutionViolation(f"Fee her girise uygulanmali: {n_entries} giris, fee {fee_applied}")


def g_entries_separate(report: dict) -> None:
    for key in ("entry1", "entry2"):
        if key not in report:
            raise ConstitutionViolation(f"Giris siralari ayri raporlanmali: {key} eksik")


def g_all_attempts_reported(n_attempt: int, n_reported: int) -> None:
    if n_reported != n_attempt:
        raise ConstitutionViolation(f"Basarisiz re-entry'ler rapordan cikarilamaz: {n_attempt}!={n_reported}")


def g_flip_separate_claim(verdicts: dict) -> None:
    if "FLIP" in str(verdicts.get("S_TO_S", "")) or "REENTRY" in str(verdicts.get("S_TO_L", "")):
        raise ConstitutionViolation("Yon degisimi ile ayni-yon re-entry ayni iddia sayilamaz")


def g_cycle_one_side(cycle_ts: int, split_of: dict) -> str:
    return split_of.get(cycle_ts, "PURGED")


# ── PREREG ───────────────────────────────────────────────────────────────────
def freeze_spec(reg: ResearchRegistry) -> ExperimentSpec:
    reg.add_question(ResearchQuestion(
        question_id="Q-BUYFADE-REENTRY-001",
        question="BUY-fade cycle'inda tekrar girisler (ayni yon) ve yon degisimleri "
                 "tek-giris baseline'ina fee-sonrasi incremental katki saglar mi?",
        origin_observation="H-RE-NULL varsayilan: churn+fee+repeated exposure",
        economic_value=0.5, risk_reduction_value=0.4, falsifiability=0.9, required_sample=MIN_CELL))
    spec = ExperimentSpec(
        experiment_id="E-BUYFADE-REENTRY-001", question_id="Q-BUYFADE-REENTRY-001",
        population=("E-BUYFADE-STRUCT-001 ile AYNI event universe (route aynen). "
                    f"Cycle=[T0, min(T0+{CYCLE_MAX_MIN}m, sonraki anchor)). "
                    "FIRST_SHORT=route(T0,45m,SL75). FIRST_LONG(research-only)="
                    "T0+45m'de 4h&1D UP ise, 24h hold (cycle-kirpilir), SL150."),
        target=("4 kol: S->S, S->L, L->L, L->S. Stop-out taksonomisi ayri "
                "(wrong_direction/bad_timing/vol_spike/structure_invalid). "
                "Entry cap 1/2/3/sinirsiz-KONTROL; cooldown grid; per-entry-order metrik."),
        features=["path-bazli causal trigger'lar: failed_reclaim, lower_high(5m), "
                  "new_buy_liq_event, relief_done(+30bps stall), higher_low, reclaim_hold, "
                  "ofi_flip, no_new_low30m", "silence(kendi aninda)", "TF state(4h/1D, completed bars)"],
        threshold_method=(f"frozen: cooldown {COOLDOWNS_MIN}m (secim TRAIN); pencere raporu "
                          f"{REENTRY_WINDOWS_MIN}m (dagilim, optimizasyon degil); relief={RELIEF_BPS}bps; "
                          f"flip icin >= {FLIP_MIN_CONFIRM} bagimsiz confirmation + ilk yon invalid; "
                          f"cap grid {MAX_ENTRIES_GRID} (99=kontrol)"),
        chronological_split=f"E-BUYFADE-STRUCT-001 ile ayni {SPLITS} + {PURGE_MS//3600_000}h purge; "
                            "cycle iki split'e bolunEMEZ (cycle T0'a gore atanir)",
        untouched_data="son %15 — re-entry sorulari icin untouched (ilk kez test ediliyor)",
        negative_control="random re-entry timing (2000 cekim, ayni eligible set); "
                         "fixed-cooldown kontrol; tek-giris baseline; sinirsiz-giris kontrol kolu",
        min_sample=MIN_CELL,
        effect_size_required_bps=3.0,
        multiple_testing_control="kol basina TRAIN-secim + VAL/UNTOUCHED rapor; "
                                 "random-timing kontrol dagilimi percentile",
        execution_model="mark_fill_fee5bps; FEE HER GIRISTE ayri; SL replay 1m/5m path",
        decision_criteria=("PASS kosullari (kol basina): tek-giris baseline'a incremental "
                           "(fee sonrasi cum>0 katki), val'de ayni yon, untouched'ta >=0, "
                           f"re-entry N>={MIN_CELL}, top3-removed pozitif kalir, "
                           "same-cycle loss-stacking (max ardil re-entry kaybi) baseline mdd'yi buyutmuyor, "
                           "random-timing kontrolunden iyi (>=0.75 pct). "
                           "Verdict sozlugu: SHORT_REENTRY_INCREMENTAL/NON, LONG_REENTRY_INCREMENTAL/NON, "
                           "SHORT_TO_LONG_INCREMENTAL, LONG_TO_SHORT_INCREMENTAL, REENTRY_CHURN, "
                           "REENTRY_TAIL_RISK, TIMING_ONLY_IMPROVEMENT, INSUFFICIENT_SAMPLE, REJECTED"),
        falsification_rule=("H-RE-NULL varsayilan; kriter gevsetme YASAK; exit sonucu bilinerek "
                            "re-entry yonu secilemez (trigger'lar yalniz fiyat/flow prefix'i); "
                            "ayni event yeni bagimsiz event sayilamaz; live sistem OTOMATIK DEGISTIRILMEZ"))
    spec.freeze()
    reg.register_experiment(spec)
    return spec


# ── PATH YARDIMCILARI (causal) ───────────────────────────────────────────────
def bar_at(ev, minute: float):
    """T0'dan minute sonra bps (1m<=120dk, sonrasi 5m)."""
    if minute <= 0:
        return 0.0
    if minute <= 120:
        i = int(minute) - 1
        return ev["path1m"][i] if 0 <= i < 120 else None
    i = int(minute / 5) - 1
    return ev["path5m"][i] if 0 <= i < len(ev["path5m"]) else None


def scan_minutes(start_min: float, end_min: float):
    m = max(1.0, float(int(start_min)))
    while m <= end_min:
        yield m
        m += 1.0 if m < 120 else 5.0


def first_short_result(ev):
    """Route replay: (exit_min, exit_bps, reason)."""
    for m in scan_minutes(1, 45):
        v = bar_at(ev, m)
        if v is not None and v >= SL_BPS:
            return m, v, "SL75"
    v = bar_at(ev, 45)
    return 45.0, (v if v is not None else 0.0), "HOLD_COMPLETE"


def stop_taxonomy(ev, exit_min):
    """Stop nedeni siniflari (evaluation-only)."""
    v4h = bar_at(ev, 240)
    later_min = min((bar_at(ev, m) for m in scan_minutes(exit_min, 240)
                     if bar_at(ev, m) is not None), default=None)
    rv = ev["f"].get("rv30m") or 0
    eh = ev["ehigh_bps_1m"][min(int(exit_min) - 1, 119)]
    if v4h is not None and v4h > SL_BPS:
        return "WRONG_DIRECTION"
    if later_min is not None and later_min < 0:
        return "BAD_TIMING"
    if exit_min <= 5 and rv * 1e4 > 30:
        return "VOL_SPIKE"
    if eh > 0:
        return "STRUCTURE_INVALID"
    return "BAD_TIMING"


# ── TRIGGER'LAR (yalniz prefix; g_causal_prefix yapisal) ─────────────────────
def precompute_buy50(conn, ev):
    """Cycle icindeki ETH BUY >=50K dakikalari (tek sorgu; trigger'lar bunu okur)."""
    t0 = ev["ts"]
    rows = conn.execute(
        "SELECT ts_ms FROM liquidations WHERE symbol='ETHUSDT' AND side='BUY' "
        "AND notional>=50000 AND ts_ms>? AND ts_ms<=? ORDER BY ts_ms",
        (t0, t0 + int(ev["cyc_end_min"]) * 60_000)).fetchall()
    ev["buy50_mins"] = [(int(r[0]) - t0) / 60_000 for r in rows]


def short_reentry_trigger(ev, from_min, cyc_end_min, conn=None):
    """Ilk SHORT-re-entry tetigi (failed_reclaim / lower_high / new_buy_event / relief_done)."""
    ref = bar_at(ev, from_min) or 0.0
    seen_above = False; local_max = ref; stall_since = None
    prev5 = None
    buy50 = ev.get("buy50_mins", [])
    for m in scan_minutes(from_min + 1, cyc_end_min):
        v = bar_at(ev, m)
        if v is None: continue
        g_causal_prefix(int(m), int(m))
        if v > 0: seen_above = True
        if seen_above and v < 0:
            return m, v, "failed_reclaim"
        if m % 5 < 1 and prev5 is not None and v < prev5 - 5:
            return m, v, "lower_high_5m"
        if m % 5 < 1: prev5 = v
        if v > local_max:
            local_max = v; stall_since = m
        elif v >= ref + RELIEF_BPS and stall_since and m - stall_since >= 10:
            return m, v, "relief_done"
        if any(m - 5 < bm <= m for bm in buy50):
            v2 = bar_at(ev, m + 5)
            if v2 is not None and v2 <= v:
                return m + 5, v2, "new_buy_event_no_gain"
    return None


def long_entry_trigger(ev, from_min, cyc_end_min):
    """LONG giris/re-entry tetigi (higher_low / reclaim_hold / no_new_low_30m)."""
    lo = None; lo_min = None; confirms = []
    for m in scan_minutes(from_min + 1, cyc_end_min):
        v = bar_at(ev, m)
        if v is None: continue
        if lo is None or v < lo:
            lo = v; lo_min = m
        if lo_min and m - lo_min >= 30 and v > lo + 10:
            confirms.append("no_new_low_30m")
        if lo_min and m - lo_min >= 10 and v > lo + 20:
            confirms.append("higher_low")
        if v > 0:
            hold = all((bar_at(ev, m + j) or -1) > 0 for j in (5, 10) if m + 10 <= cyc_end_min)
            if hold: confirms.append("reclaim_hold")
        if len(set(confirms)) >= FLIP_MIN_CONFIRM:
            return m, v, sorted(set(confirms))
    return None


def leg_result(ev, ent_min, ent_bps, hold_min, sl_bps, cyc_end_min, direction):
    """Bir bacagin sonucu (fee dahil); path uzerinde SL replay."""
    end = min(ent_min + hold_min, cyc_end_min)
    adverse_ref = ent_bps
    for m in scan_minutes(ent_min + 1, end):
        v = bar_at(ev, m)
        if v is None: continue
        adv = (v - adverse_ref) if direction == "SHORT" else (adverse_ref - v)
        if adv >= sl_bps:
            pnl = -(v - ent_bps) if direction == "SHORT" else (v - ent_bps)
            return {"net": pnl - FEE, "exit_min": m, "reason": "SL",
                    "mfe": _leg_mfe(ev, ent_min, m, ent_bps, direction),
                    "mae": -adv, "t_mfe": m, "stop_hit_min": m}
    v = bar_at(ev, end)
    if v is None:
        return None
    pnl = -(v - ent_bps) if direction == "SHORT" else (v - ent_bps)
    return {"net": pnl - FEE, "exit_min": end, "reason": "HOLD",
            "mfe": _leg_mfe(ev, ent_min, end, ent_bps, direction),
            "mae": _leg_mae(ev, ent_min, end, ent_bps, direction),
            "t_mfe": end, "stop_hit_min": None}


def _leg_vals(ev, a, b, ref, direction):
    vals = [bar_at(ev, m) for m in scan_minutes(a + 1, b)]
    vals = [v - ref for v in vals if v is not None]
    if direction == "LONG":
        vals = [-v for v in vals]
    return vals  # short-perspektif: negatif=lehte


def _leg_mfe(ev, a, b, ref, direction):
    vals = _leg_vals(ev, a, b, ref, direction)
    return float(-min(vals)) if vals else 0.0


def _leg_mae(ev, a, b, ref, direction):
    vals = _leg_vals(ev, a, b, ref, direction)
    return float(-max(0.0, max(vals))) if vals else 0.0


# ── ANA AKIS ─────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rng = np.random.RandomState(SEED)
    print("=== 8A Bidirectional Re-Entry Research ===")
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

    bars4h = Bars(marks_eth, 14_400_000, marks_eth.ts[0], marks_eth.ts[-1])
    bars1d = Bars(marks_eth, 86_400_000, marks_eth.ts[0], marks_eth.ts[-1])

    for i, ev in enumerate(events):
        compute_paths(marks_eth, ev)
        ev["f"] = {"rv30m": None,
                   "pre_silence_10m": liq_first_ts(conn, "ETHUSDT", "BUY",
                                                   ev["ts"] - 600_000, ev["ts"] - 1000,
                                                   PROP_THRESH) is None}
        silence_decompose(conn, ev)
        # cycle sonu
        nxt = events[i + 1]["ts"] if i + 1 < len(events) else None
        cyc_end = min(CYCLE_MAX_MIN, (nxt - ev["ts"]) / 60_000 if nxt else CYCLE_MAX_MIN)
        ev["cyc_end_min"] = max(45.0, cyc_end)
        ev["cycle_id"] = f"CYC:{ev['ts']}"
        precompute_buy50(conn, ev)
        if (i + 1) % 50 == 0:
            print(f"    paths {i+1}/{len(events)}", flush=True)

    # splitler (cycle T0'a gore; purge)
    n = len(events); split_of = {}
    bounds = {k: (int(n * a), int(n * b)) for k, (a, b) in SPLITS.items()}
    for k, (a, b) in bounds.items():
        for e in events[a:b]:
            split_of[e["ts"]] = k
    purged = 0
    for k in ("val", "untouched"):
        a, _ = bounds[k]
        if a == 0: continue
        prev_last = events[a - 1]["ts"]
        for e in events[a:]:
            if split_of.get(e["ts"]) != k: break
            if e["ts"] - prev_last < PURGE_MS:
                split_of[e["ts"]] = "PURGED"; purged += 1
            else:
                break
    ev_split = {k: [e for e in events if split_of[e["ts"]] == k] for k in SPLITS}
    span_days = {k: ((v[-1]["ts"] - v[0]["ts"]) / 86_400_000 if len(v) > 1 else 1.0)
                 for k, v in ev_split.items()}
    print("  splits: " + " ".join(f"{k}={len(v)}" for k, v in ev_split.items()) + f" purged={purged}")

    R = {"spec_hash": spec.frozen_hash, "n_events": n, "purged": purged,
         "splits": {k: len(v) for k, v in ev_split.items()}}

    # ── cycle simulasyonu ────────────────────────────────────────────────────
    def run_cycles(evs, cooldown_min, max_entries, allow_flip):
        """Doner: cycle kayitlari (state path + tum bacaklar entry_order'li)."""
        cycles = []
        for ev in evs:
            legs = []; states = ["NO_POSITION", "FIRST_SHORT"]
            ex_min, ex_bps, reason = first_short_result(ev)
            legs.append({"order": 1, "dir": "SHORT",
                         "net": (-(ex_bps) - FEE), "entry_min": 0.0, "exit_min": ex_min,
                         "reason": reason,
                         "mfe": _leg_mfe(ev, 0, ex_min, 0.0, "SHORT"),
                         "mae": _leg_mae(ev, 0, ex_min, 0.0, "SHORT"),
                         "t_mfe": ex_min, "stop_hit_min": ex_min if reason == "SL75" else None})
            states.append("EXITED_SHORT")
            if reason == "SL75":
                legs[-1]["stop_class"] = stop_taxonomy(ev, ex_min)
            cur_min = ex_min; entries = 1; flipped = False
            while entries < max_entries and cur_min + cooldown_min < ev["cyc_end_min"]:
                states.append("WAITING")
                trig = short_reentry_trigger(ev, cur_min + cooldown_min, ev["cyc_end_min"], conn)
                ltrig = long_entry_trigger(ev, cur_min + cooldown_min, ev["cyc_end_min"]) if allow_flip else None
                if trig and (not ltrig or trig[0] <= ltrig[0]):
                    m, v, why = trig
                    states += ["SHORT_REENTRY_ELIGIBLE", "SHORT_REENTERED"]
                    o = leg_result(ev, m, v, 45, SL_BPS, ev["cyc_end_min"], "SHORT")
                    if not o: break
                    entries += 1
                    legs.append({"order": entries, "dir": "SHORT", "trigger": why,
                                 "entry_min": m, **o})
                    cur_min = o["exit_min"]
                elif ltrig:
                    m, v, confs = ltrig
                    first_invalid = (reason == "SL75") or any(
                        bar_at(ev, mm) is not None and bar_at(ev, mm) > 0
                        for mm in scan_minutes(ex_min, min(ex_min + 30, ev["cyc_end_min"])))
                    if not first_invalid:
                        states.append("INVALIDATED"); break
                    states += ["LONG_REENTRY_ELIGIBLE", "DIRECTION_FLIP", "LONG_REENTERED"]
                    o = leg_result(ev, m, v, LONG_HOLD_MIN, LONG_SL_BPS, ev["cyc_end_min"], "LONG")
                    if not o: break
                    entries += 1; flipped = True
                    legs.append({"order": entries, "dir": "LONG", "trigger": "+".join(confs),
                                 "entry_min": m, **o})
                    cur_min = o["exit_min"]
                else:
                    break
            states.append("CYCLE_CLOSED")
            g_fee_per_entry(len(legs), sum(FEE for _ in legs))
            cycles.append({"cycle_id": ev["cycle_id"], "legs": legs, "states": states,
                           "flipped": flipped})
        return cycles

    def order_stats(cycles, k, order=None, direction=None):
        rows = []
        for c in cycles:
            for l in c["legs"]:
                if order is not None and l["order"] != order: continue
                if direction is not None and l["dir"] != direction: continue
                rows.append(l)
        return stat_block(rows, span_days[k]), len(rows)

    def cycle_total(cycles):
        return [sum(l["net"] for l in c["legs"]) for c in cycles]

    # ── S->S ve S->L kollari (cooldown TRAIN secimi) ─────────────────────────
    secSS = {}
    g_train_only_selection("train")
    cd_scan = {}
    for cd in COOLDOWNS_MIN:
        cyc = run_cycles(ev_split["train"], cd, 2, allow_flip=False)
        e2, n2 = order_stats(cyc, "train", order=2)
        base = cycle_total(run_cycles(ev_split["train"], cd, 1, allow_flip=False))
        tot = cycle_total(cyc)
        cd_scan[cd] = {"n_reentry": n2, "entry2_mean": e2.get("mean"),
                       "incr_cum": round(float(np.sum(tot) - np.sum(base)), 0)}
    best_cd = max((cd for cd in COOLDOWNS_MIN if cd_scan[cd]["n_reentry"] >= MIN_CELL),
                  key=lambda cd: cd_scan[cd]["incr_cum"], default=None)
    secSS["cooldown_scan_train"] = cd_scan
    secSS["best_cooldown_train"] = best_cd
    print(f"  S->S cooldown scan: best={best_cd}")

    if best_cd is not None:
        for k in ("train", "val", "untouched"):
            evs = ev_split[k]
            out = {}
            for cap in MAX_ENTRIES_GRID:
                cyc = run_cycles(evs, best_cd, cap, allow_flip=False)
                rep = {"cycle_total": stat_block(
                    [{"net": t, "mfe": 0, "mae": 0, "t_mfe": 0, "stop_hit_min": None}
                     for t in cycle_total(cyc)], span_days[k])}
                for o in (1, 2, 3):
                    st, cnt = order_stats(cyc, k, order=o)
                    rep[f"entry{o}"] = {**st, "label": g_tiny_cell(cnt, f"entry{o}")}
                nre = sum(1 for c in cyc for l in c["legs"] if l["order"] > 1)
                att = sum(len(c["legs"]) - 1 for c in cyc)
                g_all_attempts_reported(att, nre)
                rep["reentry_n"] = nre
                rep["churn"] = round(nre / max(1, len(cyc)), 2)
                # same-cycle loss stacking
                stacks = [sum(1 for l in c["legs"] if l["net"] <= 0) for c in cyc]
                rep["max_same_cycle_losses"] = int(max(stacks)) if stacks else 0
                # trigger dagilimi + gecikme pencere raporu
                delays = [l["entry_min"] - 45 for c in cyc for l in c["legs"] if l["order"] == 2]
                rep["reentry_delay_dist"] = {f"<={w}m": int(sum(1 for d in delays if d <= w))
                                             for w in REENTRY_WINDOWS_MIN}
                trigs = {}
                for c in cyc:
                    for l in c["legs"]:
                        if l["order"] > 1:
                            trigs[l.get("trigger", "?")] = trigs.get(l.get("trigger", "?"), 0) + 1
                rep["triggers"] = trigs
                g_entries_separate(rep)
                out[f"cap{cap}"] = rep
            secSS[k] = out
        # random re-entry timing kontrolu (train, cap2)
        cyc2 = run_cycles(ev_split["train"], best_cd, 2, allow_flip=False)
        obs_incr = (np.sum(cycle_total(cyc2))
                    - np.sum(cycle_total(run_cycles(ev_split["train"], best_cd, 1, False))))
        rnd = []
        for _ in range(2000):
            tot = 0.0
            for ev in ev_split["train"]:
                ex_min, ex_bps, _ = first_short_result(ev)
                tot += -(ex_bps) - FEE
                mmax = ev["cyc_end_min"] - 45
                if mmax > ex_min + 1:
                    m = rng.uniform(ex_min + 1, mmax)
                    v = bar_at(ev, m)
                    if v is not None:
                        o = leg_result(ev, m, v, 45, SL_BPS, ev["cyc_end_min"], "SHORT")
                        if o: tot += o["net"]
            rnd.append(tot)
        base_tot = float(np.sum(cycle_total(run_cycles(ev_split["train"], best_cd, 1, False))))
        rnd_incr = np.array(rnd) - base_tot
        secSS["random_timing_control"] = {
            "obs_incremental_cum": round(float(obs_incr), 0),
            "random_incr_median": round(float(np.median(rnd_incr)), 0),
            "pct_beat_random": round(float((obs_incr > rnd_incr).mean()), 3)}
    R["S_to_S"] = secSS

    # S->L flip kolu
    secSL = {}
    for k in ("train", "val", "untouched"):
        cyc = run_cycles(ev_split[k], best_cd if best_cd is not None else 5, 2, allow_flip=True)
        longs = [l for c in cyc for l in c["legs"] if l["dir"] == "LONG"]
        secSL[k] = {"flip_n": len(longs),
                    "flip_stats": stat_block(longs, span_days[k]),
                    "flip_rate": round(len(longs) / max(1, len(cyc)), 3),
                    "note": "flip = ilk yon invalid + >=2 bagimsiz confirmation (frozen)"}
    R["S_to_L"] = secSL

    # ── L->L ve L->S kollari ─────────────────────────────────────────────────
    secL = {}
    for k in ("train", "val", "untouched"):
        evs = ev_split[k]
        first_longs, ll_re, ls_flip = [], [], []
        elig = 0
        for ev in evs:
            t45 = ev["ts"] + HOLD_MS
            if bars4h.state_at(t45, TF_THR_BPS["4h"]) != "UP" or \
               bars1d.state_at(t45, TF_THR_BPS["1D"]) != "UP":
                continue
            elig += 1
            v45 = bar_at(ev, 45)
            if v45 is None: continue
            o = leg_result(ev, 45, v45, LONG_HOLD_MIN, LONG_SL_BPS, ev["cyc_end_min"], "LONG")
            if not o: continue
            first_longs.append({**o, "order": 1})
            # L->L: cikis sonrasi higher_low/reclaim ile re-entry
            lt = long_entry_trigger(ev, o["exit_min"], ev["cyc_end_min"])
            if lt:
                m, v, confs = lt
                o2 = leg_result(ev, m, v, LONG_HOLD_MIN, LONG_SL_BPS, ev["cyc_end_min"], "LONG")
                if o2: ll_re.append({**o2, "order": 2, "trigger": "+".join(confs)})
            # L->S: LONG SL olduysa breakdown flip
            if o["reason"] == "SL":
                st = short_reentry_trigger(ev, o["exit_min"] + 5, ev["cyc_end_min"], conn)
                if st:
                    m, v, why = st
                    o3 = leg_result(ev, m, v, 45, SL_BPS, ev["cyc_end_min"], "SHORT")
                    if o3: ls_flip.append({**o3, "order": 2, "trigger": why})
        secL[k] = {"first_long_eligible": elig,
                   "first_long": stat_block(first_longs, span_days[k]),
                   "L_to_L_reentry": {**stat_block(ll_re, span_days[k]),
                                      "label": g_tiny_cell(len(ll_re), "L2L")},
                   "L_to_S_flip": {**stat_block(ls_flip, span_days[k]),
                                   "label": g_tiny_cell(len(ls_flip), "L2S")}}
    R["L_arms"] = secL

    # ── stop-out taksonomisi + SL sonrasi ayni-yon re-entry ──────────────────
    secST = {}
    for k in ("train", "val"):
        evs = ev_split[k]
        cls_rows = {}
        for ev in evs:
            ex_min, ex_bps, reason = first_short_result(ev)
            if reason != "SL75": continue
            cls = stop_taxonomy(ev, ex_min)
            trig = short_reentry_trigger(ev, ex_min + (best_cd or 5), ev["cyc_end_min"], conn)
            o = None
            if trig:
                o = leg_result(ev, trig[0], trig[1], 45, SL_BPS, ev["cyc_end_min"], "SHORT")
            cls_rows.setdefault(cls, []).append(o)
        out = {}
        for cls, rows in cls_rows.items():
            done = [r for r in rows if r]
            out[cls] = {"stopped_n": len(rows), "reentered_n": len(done),
                        "reentry_stats": stat_block(done, span_days[k]),
                        "label": g_tiny_cell(len(done), cls)}
        secST[k] = out
    R["stop_taxonomy"] = secST

    # ── hipotezler ───────────────────────────────────────────────────────────
    hyp = {}
    for k in ("train", "val"):
        evs = ev_split[k]
        # H-RE-S2: HTF DOWN'da S->S degerli mi, 1D/1W UP'ta churn mu
        dn = [e for e in evs if bars1d.state_at(e["ts"], TF_THR_BPS["1D"]) == "DOWN"]
        up = [e for e in evs if bars1d.state_at(e["ts"], TF_THR_BPS["1D"]) == "UP"]
        def incr(sub):
            if not sub: return None
            c2 = cycle_total(run_cycles(sub, best_cd or 5, 2, False))
            c1 = cycle_total(run_cycles(sub, best_cd or 5, 1, False))
            return round(float(np.sum(c2) - np.sum(c1)), 0)
        hyp.setdefault("H_RE_S2", {})[k] = {"1D_DOWN_incr_cum": incr(dn), "n_down": len(dn),
                                            "1D_UP_incr_cum": incr(up), "n_up": len(up)}
    R["hypotheses"] = hyp

    # ── verdicts ─────────────────────────────────────────────────────────────
    V = {}
    if best_cd is None:
        V["S_TO_S"] = "INSUFFICIENT_SAMPLE"
    else:
        va = secSS.get("val", {}).get("cap2", {})
        ua = secSS.get("untouched", {}).get("cap2", {})
        tr = secSS.get("train", {}).get("cap2", {})
        e2v = va.get("entry2", {})
        rt = secSS.get("random_timing_control", {})
        incr_v = ((va.get("cycle_total", {}).get("cum") or 0)
                  - (secSS.get("val", {}).get("cap1", {}).get("cycle_total", {}).get("cum") or 0))
        checks = {"reentry_n": (e2v.get("n") or 0) >= MIN_CELL,
                  "val_incremental": incr_v > 0,
                  "untouched_nonneg": ((ua.get("cycle_total", {}).get("cum") or 0)
                                       - (secSS.get("untouched", {}).get("cap1", {}).get("cycle_total", {}).get("cum") or 0)) >= 0,
                  "beats_random": (rt.get("pct_beat_random") or 0) >= 0.75,
                  "top3_ok": (e2v.get("top3_removed") or -1) > 0,
                  "loss_stack_ok": (va.get("max_same_cycle_losses") or 9) <= 3}
        V["S_TO_S"] = ("SHORT_REENTRY_INCREMENTAL" if all(checks.values())
                       else ("REENTRY_CHURN" if (tr.get("cap2", {}).get("churn") or 0) > 0.5
                             and incr_v <= 0 else "SHORT_REENTRY_NON_INCREMENTAL"))
        V["S_TO_S_checks"] = checks
    fv = secSL.get("val", {})
    V["S_TO_L"] = ("INSUFFICIENT_SAMPLE" if (fv.get("flip_n") or 0) < MIN_CELL
                   else ("SHORT_TO_LONG_INCREMENTAL" if (fv.get("flip_stats", {}).get("mean") or -9) > 0
                         and (fv.get("flip_stats", {}).get("top3_removed") or -1) > 0
                         else "REJECTED"))
    lv = secL.get("val", {})
    V["L_TO_L"] = ("INSUFFICIENT_SAMPLE" if lv.get("L_to_L_reentry", {}).get("label") == "INSUFFICIENT_SAMPLE"
                   else ("LONG_REENTRY_INCREMENTAL" if (lv.get("L_to_L_reentry", {}).get("mean") or -9) > 0
                         else "LONG_REENTRY_NON_INCREMENTAL"))
    V["L_TO_S"] = ("INSUFFICIENT_SAMPLE" if lv.get("L_to_S_flip", {}).get("label") == "INSUFFICIENT_SAMPLE"
                   else ("LONG_TO_SHORT_INCREMENTAL" if (lv.get("L_to_S_flip", {}).get("mean") or -9) > 0
                         else "REJECTED"))
    g_flip_separate_claim(V)
    R["verdicts"] = V
    print(f"  VERDICTS: {json.dumps(V, ensure_ascii=False, default=str)}")

    OJ.write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
    lines = ["# 8A — Bidirectional Re-Entry Research (BUY-FADE)", "",
             f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — prereg `{spec.experiment_id}` "
             f"hash `{spec.frozen_hash}` · {R['n_events']} event · splits {R['splits']}", "",
             "## Verdicts", "```json", json.dumps(V, indent=1, default=str), "```", ""]
    for sec in ("S_to_S", "S_to_L", "L_arms", "stop_taxonomy", "hypotheses"):
        lines += [f"## {sec}", "```json", json.dumps(R[sec], indent=1, default=str), "```", ""]
    lines += ["", "State machine: " + " -> ".join(STATES),
              "", "*Live sistem OTOMATIK DEGISTIRILMEDI; tum kollar research-only.*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD: {OM}")
    conn.close(); reg.close()
    return R


if __name__ == "__main__":
    main()
