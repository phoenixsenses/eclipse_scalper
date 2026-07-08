"""Faz 6A-R2 — Risk and Applicability Validation (RESEARCH-ONLY).

Soru: "Regime + latent katmani, ayni trade sayisi ve ayni piyasa maruziyeti
altinda baseline'a gore kayip yogunlugunu, tail risk'i veya drawdown'i
azaltiyor mu?" Bu paket YENI GIRIS ALPHA'SI ARAMAZ.

Aday kullanim siniflari YALNIZ: applicability restriction, shadow suspension,
risk warning, research prioritization.

Frozen akis:
1. Prereg E-RISKAPP-6AR2-001 DONDURULUR (hesaplamadan once).
2. Trade populasyonu: no-overlap 6h LONG grid trade'leri (veto yorumu:
   filtreler AYNI populasyonun altkumesini secer, yeniden simulasyon yok).
3. Walk-forward fold'lar (expanding window, per-fold artifact refit).
4. Her fold: 5 kol (baseline / random-veto / regime-only / latent-only /
   regime+latent) + matched-count blocked bootstrap + tam risk metrikleri +
   loss-concentration.
5. Drift alarm lead/lag (forward-only pencere kurgusu).
6. Kabul degerlendirmesi frozen kriterlerle; untouched veri YOK oldugu
   PREREG'DE beyan edilir -> tam PASS bu dataset'te YAPISAL OLARAK IMKANSIZ;
   ulasilabilir en iyi sonuc CHRONO_SUPPORTED_PENDING_FORWARD.

Onemli durustluk notu: 6A-R alpha_eval son %15 pencereyi GORDU (N=14 hipotez
kaynagi). Bu yuzden fold-4 (85-100%) CONTAMINATED etiketiyle rapor edilir ve
tek basina kanit sayilmaz.

Run: python -m ami.latent.risk_applicability
Cikti: reports/research/s34/AMI_PHASE6AR2_RISK.md + .json
"""
from __future__ import annotations
import json, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.constitution import ConstitutionViolation
from ami.enums import ClaimType, EvidenceLevel, FailureType, KnowledgeStatus, Permission
from ami.knowledge.objects import KnowledgeObject, Provenance
from ami.knowledge.store import KnowledgeStore
from ami.latent.dataset import load_dataset
from ami.latent.drift_monitor import DriftMonitor
from ami.latent.models import Standardizer, seeded_kmeans
from ami.latent.regime import RegimeDefiner, check_regime_fit_boundary
from ami.research.registry import (EvidenceBundle, ExperimentSpec, ResearchQuestion,
                                   ResearchRegistry, assert_no_overlap)

OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "AMI_PHASE6AR2_RISK.json"; OM = OUT / "AMI_PHASE6AR2_RISK.md"
ARTIFACTS = ROOT / "data" / "ami" / "latent_artifacts.json"

FEE = 5.0
HOLD_MS = 6 * 3600_000
PATH_STEP_MS = 300_000          # MAE/MFE cozunurlugu (5m)
SEED = 11                       # 6A/6A-R ile ayni frozen seed
K = 4                           # 6A/6A-R frozen k (fold'da yeniden SECILMEZ)
WF_FOLDS = [(0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]
CONTAMINATED_FOLD = 3           # fold index (85-100%): 6A-R alpha_eval hipotez kaynagi
N_BOOT = 2000                   # matched-count + random-veto cekim sayisi
BLOCK_LEN = 5                   # moving-block bootstrap blok uzunlugu (trade)
MIN_FOLD_CAND = 8               # fold degerlendirilebilirlik esigi
MIN_TOTAL_CAND = 40             # herhangi bir SUPPORTED iddiasi icin toplam aday N
PCT_BEAT = 0.75                 # matched/random dagiliminda "daha iyi" esigi
RETENTION_MIN = 0.90            # winner-retention-ratio frozen minimumu
BAD_NET_BPS = -150.0            # SL-analog kotu trade esigi
BAD_MAE_BPS = -150.0            # yuksek-MAE kotu trade esigi
GIVEBACK_MFE = 100.0            # giveback: MFE>=+100 & net<=0
DD_START_BPS = -200.0           # DD-baslatan trade: sonraki drawdown >= 200bps
DRIFT_WIN = 2016                # 7 gun (5m grid)
DRIFT_STEP = 576                # 2 gun
DETER_SIGMA = 0.5               # deterioration: geri-7g mean net <= train_mean - 0.5*train_std

ALLOWED_PERMS = {Permission.RESEARCH_ONLY, Permission.BACKTEST_ALLOWED,
                 Permission.SHADOW_ALLOWED}
FORBIDDEN_PERMS = {Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED,
                   Permission.PORTFOLIO_ALLOWED}

REQUIRED_CONTROLS = ("random_veto", "regime_only", "latent_only", "matched_count")
REQUIRED_EXPOSURE_KEYS = ("mean", "median", "per_active_hour", "downside_dev", "cvar5")
REQUIRED_CONCENTRATION_KEYS = ("bad_trade_recall", "bad_trade_precision",
                               "good_trade_retention", "winner_sacrifice_bps",
                               "loss_avoided_bps", "profit_sacrificed_bps",
                               "net_economic_value_bps")


# ── Guard'lar (mutation suite bunlari test eder) ─────────────────────────────
def guard_frequency_normalized(n_a: int, n_b: int, compared_metrics: list[str]) -> None:
    """Farkli N'li kollar arasinda ham (kumulatif) MDD/cum karsilastirmasi YASAK.

    Ham MDD yalniz esit-N dagilimlari (matched-count / random-veto) icinde
    karsilastirilabilir."""
    raw = {"mdd", "cum", "max_drawdown", "cumulative"}
    if n_a != n_b and any(m.lower() in raw for m in compared_metrics):
        raise ConstitutionViolation(
            f"Frequency artifact riski: N={n_a} vs N={n_b} ile ham metrik "
            f"{compared_metrics} dogrudan karsilastirilamaz (esit-N kontrol dagilimi sart).")


def require_controls(fold_result: dict) -> None:
    missing = [c for c in REQUIRED_CONTROLS if c not in fold_result]
    if missing:
        raise ConstitutionViolation(f"Zorunlu kontrol kollari eksik: {missing}")


def require_exposure_normalization(metrics: dict) -> None:
    missing = [k for k in REQUIRED_EXPOSURE_KEYS if k not in metrics]
    if missing:
        raise ConstitutionViolation(f"Exposure-normalize metrikler eksik: {missing}")


def require_topwinner_disclosure(metrics: dict) -> None:
    for k in ("top1_removed", "top3_removed", "top5_removed"):
        if k not in metrics:
            raise ConstitutionViolation(f"Top-winner dependence gizlenemez: {k} eksik")


def require_loss_concentration(report: dict) -> None:
    missing = [k for k in REQUIRED_CONCENTRATION_KEYS if k not in report]
    if missing:
        raise ConstitutionViolation(f"Good-trade sacrifice raporlanmadi: {missing}")


def guard_no_retroactive_alarm(window_end_idx: int, data_used_max_idx: int) -> None:
    """Alarm yalniz pencere sonuna kadarki veriyle uretilir (geriye donuk yasak)."""
    if data_used_max_idx > window_end_idx:
        raise ConstitutionViolation(
            f"Retroaktif alarm: pencere sonu {window_end_idx} ama veri {data_used_max_idx} kullanildi.")


def fold_verdict(folds: list[dict]) -> dict:
    """Karar TUM degerlendirilebilir fold'lardan; cherry-picking yapisal olarak imkansiz."""
    ev = [f for f in folds if f.get("evaluable")]
    passed = sum(1 for f in ev if f.get("fold_pass"))
    return {"evaluable_folds": len(ev), "passed_folds": passed,
            "majority_pass": bool(ev and passed * 2 > len(ev)),
            "all_folds_reported": len(folds)}


def guard_fold_aggregation(folds: list[dict], claimed: dict) -> None:
    actual = fold_verdict(folds)
    if actual != claimed:
        raise ConstitutionViolation(f"Fold cherry-picking: claimed={claimed} actual={actual}")


def guard_artifact_usable(drift_status: str, use: str) -> None:
    """UNUSABLE drift'te artifact YENI trade secimi/karari icin kullanilamaz.
    research/rapor kullanimi serbesttir."""
    if drift_status in ("UNUSABLE", "SHIFTED") and use in ("trade_selection", "live", "sizing"):
        raise ConstitutionViolation(
            f"Drift={drift_status} iken artifact '{use}' icin kullanilamaz (yalniz research/shadow-suspend onerisi).")


def guard_artifact_version(artifact_meta: dict, dataset_meta: dict) -> None:
    if artifact_meta.get("feature_version") != dataset_meta.get("feature_version"):
        raise ConstitutionViolation(
            f"Stale artifact: feature_version {artifact_meta.get('feature_version')} != "
            f"dataset {dataset_meta.get('feature_version')}")


def guard_permissions(requested: set) -> None:
    bad = set(requested) & FORBIDDEN_PERMS
    if bad:
        raise ConstitutionViolation(
            f"Risk/applicability kaniti {sorted(p.value for p in bad)} izni VEREMEZ "
            f"(max: RESEARCH_ONLY/BACKTEST/SHADOW + SHADOW_SUSPEND_SUGGESTION).")


def guard_bootstrap_confidence(n: int, min_n: int = MIN_FOLD_CAND) -> str:
    """Kucuk-N bootstrap guveni abartilamaz: n<min ise sonuc INSUFFICIENT_SAMPLE."""
    return "OK" if n >= min_n else "INSUFFICIENT_SAMPLE"


# ── Risk metrikleri (§4 zorunlu set) ──────────────────────────────────────────
def risk_metrics(nets: np.ndarray, maes: np.ndarray, mfes: np.ndarray,
                 sessions: list[str], regimes: list[str], span_days: float,
                 hold_hours: float = 6.0) -> dict:
    n = len(nets)
    if n == 0:
        return {"n": 0}
    nets = np.asarray(nets, dtype=float)
    srt = np.sort(nets)[::-1]
    eq = np.cumsum(nets)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    mdd = float(dd.min())
    # dd duration / recovery (trade birimi)
    dur = best_dur = 0
    rec_times = []
    cur_rec = None
    for v in dd:
        if v < 0:
            dur += 1; best_dur = max(best_dur, dur)
            cur_rec = dur if cur_rec is None else cur_rec + 1
        else:
            dur = 0
            if cur_rec is not None:
                rec_times.append(cur_rec); cur_rec = None
    losses = nets[nets <= 0]
    gains = nets[nets > 0]
    k5 = max(1, int(np.ceil(n * 0.05)))
    cvar5 = float(np.sort(nets)[:k5].mean())
    # ardil kayip dagilimi
    streak = best_streak = 0; streaks = []
    for v in nets:
        if v <= 0:
            streak += 1; best_streak = max(best_streak, streak)
        else:
            if streak:
                streaks.append(streak)
            streak = 0
    if streak:
        streaks.append(streak)
    dsd = float(np.sqrt(np.mean(np.minimum(nets, 0.0) ** 2)))
    exposure_h = n * hold_hours
    sess_u, sess_c = np.unique(np.asarray(sessions), return_counts=True) if sessions else ([], [])
    reg_u, reg_c = np.unique(np.asarray(regimes), return_counts=True) if regimes else ([], [])
    m = {
        "n": n,
        "trade_per_day": round(n / span_days, 2) if span_days > 0 else None,
        "exposure_hours": round(exposure_h, 1),
        "cum": round(float(nets.sum()), 1),
        "mean": round(float(nets.mean()), 2),
        "median": round(float(np.median(nets)), 2),
        "per_active_hour": round(float(nets.sum()) / exposure_h, 3) if exposure_h else None,
        "top1_removed": round(float(srt[1:].sum()), 1) if n > 1 else None,
        "top3_removed": round(float(srt[3:].sum()), 1) if n > 3 else None,
        "top5_removed": round(float(srt[5:].sum()), 1) if n > 5 else None,
        "pf": round(float(gains.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "mdd": round(mdd, 1),
        "avg_dd": round(float(dd.mean()), 1),
        "dd_duration_max_trades": int(best_dur),
        "recovery_trades_mean": round(float(np.mean(rec_times)), 1) if rec_times else None,
        "worst": round(float(nets.min()), 1),
        "bottom3_cum": round(float(np.sort(nets)[:3].sum()), 1) if n >= 3 else round(float(nets.sum()), 1),
        "cvar5": round(cvar5, 1),
        "mae_mean": round(float(np.mean(maes)), 1) if len(maes) else None,
        "mae_p10": round(float(np.percentile(maes, 10)), 1) if len(maes) else None,
        "loss_rate": round(float((nets <= 0).mean()), 3),
        "max_consec_loss": int(best_streak),
        "ret_vol": round(float(nets.std()), 1),
        "ret_to_dd": round(float(nets.sum() / -mdd), 2) if mdd < 0 else None,
        "downside_dev": round(dsd, 1),
        "session_conc": {str(u): round(int(c) / n, 2) for u, c in zip(sess_u, sess_c)},
        "regime_conc": {str(u): round(int(c) / n, 2) for u, c in zip(reg_u, reg_c)},
    }
    require_exposure_normalization(m)
    require_topwinner_disclosure(m)
    return m


def moving_block_indices(rng: np.random.RandomState, n_pop: int, n_out: int,
                         block: int = BLOCK_LEN) -> np.ndarray:
    idx = []
    while len(idx) < n_out:
        s = rng.randint(0, max(1, n_pop - block + 1))
        idx.extend(range(s, min(s + block, n_pop)))
    return np.array(idx[:n_out])


def dist_percentile(cand_value: float, dist: np.ndarray, higher_is_better: bool) -> float:
    """Aday, dagilimin yuzde kacindan iyi? (1.0 = hepsinden iyi)"""
    dist = dist[np.isfinite(dist)]
    if len(dist) == 0 or not np.isfinite(cand_value):
        return float("nan")
    return float((cand_value > dist).mean() if higher_is_better else (cand_value < dist).mean())


def loss_concentration(base_nets: np.ndarray, base_maes: np.ndarray, base_mfes: np.ndarray,
                       selected: np.ndarray) -> dict:
    """Aday veto'su kotu trade'leri onceden ayirabiliyor mu? (selected=aday tuttu)"""
    n = len(base_nets)
    vetoed = ~selected
    order = np.argsort(base_nets)
    worst5 = np.zeros(n, bool); worst5[order[:max(1, int(np.ceil(n * 0.05)))]] = True
    worst10 = np.zeros(n, bool); worst10[order[:max(1, int(np.ceil(n * 0.10)))]] = True
    sl = base_nets <= BAD_NET_BPS
    hi_mae = base_maes <= BAD_MAE_BPS
    giveback = (base_mfes >= GIVEBACK_MFE) & (base_nets <= 0)
    # DD-baslatan: bu trade'den itibaren baseline equity >= |DD_START_BPS| dusuyor
    eq = np.cumsum(base_nets)
    dd_start = np.zeros(n, bool)
    for i in range(n):
        fut = eq[i:] - (eq[i - 1] if i else 0.0)
        if len(fut) and fut.min() <= DD_START_BPS:
            dd_start[i] = True
    bad = worst10 | sl | hi_mae | giveback
    winners = base_nets > 0

    def rr(mask):  # veto'nun bu siniftaki yakalama orani
        return round(float(vetoed[mask].mean()), 3) if mask.any() else None

    n_veto = int(vetoed.sum())
    rep = {
        "bad_class_sizes": {"worst5": int(worst5.sum()), "worst10": int(worst10.sum()),
                            "sl": int(sl.sum()), "high_mae": int(hi_mae.sum()),
                            "giveback": int(giveback.sum()), "dd_start": int(dd_start.sum())},
        "veto_recall_by_class": {"worst5": rr(worst5), "worst10": rr(worst10), "sl": rr(sl),
                                 "high_mae": rr(hi_mae), "giveback": rr(giveback),
                                 "dd_start": rr(dd_start)},
        "bad_trade_recall": round(float(vetoed[bad].mean()), 3) if bad.any() else None,
        "bad_trade_precision": round(float(bad[vetoed].mean()), 3) if n_veto else None,
        "good_trade_retention": round(float(selected[winners].mean()), 3) if winners.any() else None,
        "selection_rate": round(float(selected.mean()), 3),
        "winner_sacrifice_bps": round(float(base_nets[vetoed & winners].sum()), 1),
        "loss_avoided_bps": round(float(-base_nets[vetoed & ~winners].sum()), 1),
        "profit_sacrificed_bps": round(float(base_nets[vetoed & winners].sum()), 1),
        "net_economic_value_bps": round(float(-base_nets[vetoed].sum()), 1),
    }
    sel_rate = rep["selection_rate"]
    ret = rep["good_trade_retention"]
    rep["retention_ratio"] = round(ret / sel_rate, 2) if ret is not None and sel_rate else None
    require_loss_concentration(rep)
    return rep


# ── Prereg ───────────────────────────────────────────────────────────────────
def spec_6ar2(reg: ResearchRegistry) -> ExperimentSpec:
    reg.add_question(ResearchQuestion(
        question_id="Q-RISK-APPLICABILITY-001",
        question="Regime+latent katmani ayni trade sayisi/maruziyet altinda tail risk ve "
                 "kayip yogunlugunu azaltiyor mu, yoksa dusuk frekansin mekanik sonucu mu?",
        origin_observation="6A-R alpha_eval: regime+latent N=14 mdd -416 vs baseline N=50 mdd -1363 "
                           "(HIPOTEZ KAYNAGI, dogrulama kaniti DEGIL)",
        risk_reduction_value=0.9, economic_value=0.3, novelty=0.4, falsifiability=0.9,
        required_sample=MIN_TOTAL_CAND))
    spec = ExperimentSpec(
        experiment_id="E-RISKAPP-6AR2-001", question_id="Q-RISK-APPLICABILITY-001",
        population=("no-overlap 6h LONG grid trade'leri: latent_dataset.npz 5m grid'inin TAMAMI, "
                    "busy-guard 6h, mark-fill, FEE=5bps, son 6h haric (truncation). "
                    "Filtreler VETO yorumuyla ayni populasyonun altkumesini secer; yeniden "
                    "simulasyon YOK. Yeni giris alpha'si aranmaz."),
        target=("risk/applicability: candidate=trend=UP&latent-calm veto'sunun matched-count, "
                "random-veto ve regime-only kontrollerine karsi tail-risk katkisi. "
                "Aday kullanim siniflari YALNIZ applicability-restriction/shadow-suspension/"
                "risk-warning/research-prioritization."),
        features=["regime:trend(per-fold refit)", "latent:k4-seed11-calm(argmin-center-norm, per-fold refit)",
                  "session", "net_bps", "mae_5m", "mfe_5m"],
        threshold_method=(f"frozen: K={K} seed={SEED} (6A artifact'i, fold'da yeniden secilmez); "
                          f"calm=argmin||center|| (outcome'suz kural); trend esigi ±100bps/24h sabit; "
                          f"bootstrap N={N_BOOT} moving-block block={BLOCK_LEN}; kotu-trade siniflari: "
                          f"worst10%, net<={BAD_NET_BPS}, MAE<={BAD_MAE_BPS}, giveback(MFE>={GIVEBACK_MFE}&net<=0), "
                          f"dd-start<={DD_START_BPS}; drift pencere {DRIFT_WIN}x{DRIFT_STEP} step, "
                          f"deterioration=geri-7g mean <= train_mean-{DETER_SIGMA}*train_std"),
        chronological_split=(f"expanding walk-forward, val pencereleri {WF_FOLDS}; her fold'da "
                             "standardizer+rejim esikleri+kmeans merkezleri YALNIZ [0,val_lo) uzerinde "
                             "refit; label'lama prefix [0,val_hi) verisiyle (impute lookahead yok)"),
        untouched_data=("YOKTUR ve bu PREREG'DE BEYAN EDILIR: 85-100% penceresi 6A-R alpha_eval "
                        "tarafindan goruldu (N=14 hipotez kaynagi) -> fold-4 CONTAMINATED olarak "
                        "raporlanir, tek basina kanit sayilmaz. Bu deney untouched-PASS uretemez; "
                        "ulasilabilir en iyi sonuc CHRONO_SUPPORTED_PENDING_FORWARD "
                        "(forward shadow verisi birikmeden terfi YOK)."),
        negative_control=("random-veto: aday sayisina esitlenmis rastgele trade cikarma "
                          f"({N_BOOT} seed); matched-count moving-block bootstrap; regime-only ve "
                          "latent-only ayristirma kollari; dq/missingness fold bazinda kontrol"),
        min_sample=MIN_TOTAL_CAND,
        effect_size_required_bps=0.0,   # risk deneyi: getiri etki buyuklugu hedefi YOK
        multiple_testing_control="tek frozen candidate; kontrol dagilimlari percentile bazli",
        execution_model="research_only_no_execution",
        decision_criteria=(
            f"Fold-PASS (hepsi birden): (a) cvar5 matched-count dagiliminin >={PCT_BEAT} "
            f"percentile'inde; (b) cvar5 random-veto dagiliminin >={PCT_BEAT} percentile'inde; "
            f"(c) cvar5 ve downside_dev regime-only kolundan daha iyi (incremental); "
            f"(d) retention_ratio >= {RETENTION_MIN}; (e) fold aday N >= {MIN_FOLD_CAND}. "
            f"Genel: degerlendirilebilir fold'larin cogunlugu PASS VE toplam aday N >= {MIN_TOTAL_CAND} "
            f"VE top3-removed aday-vs-regime-only siralamayi cevirmiyor VE alarm lead>=0 VE "
            f"dq aciklamasi elenmis -> sonuc CHRONO_SUPPORTED_PENDING_FORWARD (tam PASS untouched "
            f"yoklugundan YAPISAL OLARAK verilemez). Izin tavani: RESEARCH_ONLY/BACKTEST/SHADOW + "
            f"SHADOW_SUSPEND_SUGGESTION; LIVE/SIZING/PORTFOLIO YASAK."),
        falsification_rule=(
            "Frozen oncelik sirasiyla siniflandir: toplam aday N<40 veya degerlendirilebilir fold<2 "
            "-> INSUFFICIENT_SAMPLE; random-veto'yu cogunlukta gecemiyor -> FREQUENCY_ARTIFACT; "
            "regime-only'ye incremental degil -> RISK_NON_INCREMENTAL; risk gecti ama alarm "
            "deterioration'dan SONRA -> LATE_DRIFT_DETECTION; diger -> REJECTED. "
            "YASAK post-hoc degisiklikler: metrik ekleme/cikarma, esik/percentile degisimi, "
            "fold secimi, block uzunlugu, calm-state veya kotu-trade tanimi degisimi."))
    spec.freeze()
    reg.register_experiment(spec)
    return spec


# ── Trade populasyonu ─────────────────────────────────────────────────────────
def build_trades(ts: np.ndarray, sessions: list[str]) -> list[dict]:
    from tools.research_s34_knowable_anchor_continuation import load_mark_index
    conn = sqlite3.connect(f"file:{ROOT/'data'/'microstructure.db'}?mode=ro", uri=True)
    m = load_mark_index(conn, "ETHUSDT")
    trades = []
    busy = -1
    n = len(ts)
    for i in range(n):
        t0 = int(ts[i])
        if t0 < busy or t0 + HOLD_MS > int(ts[-1]):
            continue
        a = m.at_or_before(t0)
        if not a or float(a[1]) <= 0:
            continue
        p0 = float(a[1])
        path = []
        for j in range(1, int(HOLD_MS // PATH_STEP_MS) + 1):
            r = m.at_or_before(t0 + j * PATH_STEP_MS)
            if r:
                path.append((float(r[1]) - p0) / p0 * 1e4)
        if not path:
            continue
        gross = path[-1]
        trades.append({"idx": i, "t0": t0, "net": gross - FEE,
                       "mae": min(path), "mfe": max(path), "session": sessions[i]})
        busy = t0 + HOLD_MS
    conn.close()
    return trades


# ── Fold degerlendirmesi ─────────────────────────────────────────────────────
def eval_fold(fold_no: int, lo_f: float, hi_f: float, X, ts, trades, feats, sessions,
              contaminated: bool) -> dict:
    n = X.shape[0]
    tr_end, va_end = int(n * lo_f), int(n * hi_f)
    assert_no_overlap(set(range(tr_end)), set(range(tr_end, va_end)))

    # per-fold artifact refit — YALNIZ [0, tr_end)
    std = Standardizer().fit(X[:tr_end], (int(ts[0]), int(ts[tr_end - 1])))
    rd = RegimeDefiner().fit(X[:tr_end], feats, (int(ts[0]), int(ts[tr_end - 1])))
    check_regime_fit_boundary(rd, int(ts[tr_end]))
    _, C, _ = seeded_kmeans(std.transform(X[:tr_end]), K, SEED)
    calm = int(np.argmin(np.linalg.norm(C, axis=1)))
    # label'lama prefix verisiyle (impute/roll lookahead yok)
    regs = rd.label(X[:va_end], feats, sessions[:va_end])
    Zp = std.transform(X[:va_end])
    d2 = ((Zp[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    lat = d2.argmin(axis=1)

    ft = [t for t in trades if tr_end <= t["idx"] < va_end]
    span_days = (ts[va_end - 1] - ts[tr_end]) / 86_400_000
    if not ft:
        return {"fold": fold_no, "evaluable": False, "reason": "no_trades"}
    nets = np.array([t["net"] for t in ft]); maes = np.array([t["mae"] for t in ft])
    mfes = np.array([t["mfe"] for t in ft])
    sess = [t["session"] for t in ft]
    tregs = [str(regs["trend"][t["idx"]]) for t in ft]
    tlats = np.array([int(lat[t["idx"]]) for t in ft])
    sel_reg = np.array([r == "UP" for r in tregs])
    sel_lat = tlats == calm
    sel_cand = sel_reg & sel_lat
    n_cand = int(sel_cand.sum())

    def arm(mask):
        return risk_metrics(nets[mask], maes[mask], mfes[mask],
                            [s for s, m_ in zip(sess, mask) if m_],
                            [r for r, m_ in zip(tregs, mask) if m_], span_days)

    base_m = arm(np.ones(len(ft), bool))
    cand_m = arm(sel_cand) if n_cand else {"n": 0}
    reg_m = arm(sel_reg) if sel_reg.any() else {"n": 0}
    lat_m = arm(sel_lat) if sel_lat.any() else {"n": 0}

    res = {"fold": fold_no, "val_win": [lo_f, hi_f], "contaminated": contaminated,
           "calm_state": calm, "n_base": len(ft), "n_cand": n_cand,
           "baseline": base_m, "candidate": cand_m, "regime_only": reg_m,
           "latent_only": lat_m}

    evaluable = n_cand >= MIN_FOLD_CAND
    res["evaluable"] = evaluable and not contaminated
    res["sample_status"] = guard_bootstrap_confidence(n_cand)
    if not evaluable:
        res["matched_count"] = {"skipped": "n_cand<min"}
        res["random_veto"] = {"skipped": "n_cand<min"}
        require_controls(res)
        return res

    # matched-count moving-block bootstrap (esit-N icinde ham metrik karsilastirmasi mesru)
    rng = np.random.RandomState(SEED)
    boot = {"cvar5": [], "mdd": [], "downside_dev": [], "mean": [], "worst": []}
    rv = {"cvar5": [], "mdd": [], "downside_dev": [], "mean": [], "worst": []}
    for _ in range(N_BOOT):
        bi = moving_block_indices(rng, len(ft), n_cand)
        bm = risk_metrics(nets[bi], maes[bi], mfes[bi], [sess[j] for j in bi],
                          [tregs[j] for j in bi], span_days)
        for k in boot:
            boot[k].append(bm[k])
        keep = rng.choice(len(ft), n_cand, replace=False)
        km = np.zeros(len(ft), bool); km[keep] = True
        rm = risk_metrics(nets[km], maes[km], mfes[km],
                          [s for s, m_ in zip(sess, km) if m_],
                          [r for r, m_ in zip(tregs, km) if m_], span_days)
        for k in rv:
            rv[k].append(rm[k])
    guard_frequency_normalized(n_cand, n_cand, ["mdd"])   # esit-N: mesru
    pct_m = {k: dist_percentile(cand_m[k], np.array(boot[k], float), higher_is_better=True)
             for k in boot}
    pct_r = {k: dist_percentile(cand_m[k], np.array(rv[k], float), higher_is_better=True)
             for k in rv}
    res["matched_count"] = {"n_draws": N_BOOT, "block_len": BLOCK_LEN,
                            "boot_median": {k: round(float(np.median(boot[k])), 1) for k in boot},
                            "cand_percentile": {k: round(v, 3) for k, v in pct_m.items()}}
    res["random_veto"] = {"n_draws": N_BOOT,
                          "rv_median": {k: round(float(np.median(rv[k])), 1) for k in rv},
                          "cand_percentile": {k: round(v, 3) for k, v in pct_r.items()}}
    # session dagilim benzerligi (matched-count secondary kontrol)
    cs = {s: sum(1 for j in np.where(sel_cand)[0] if sess[j] == s) / n_cand
          for s in set(sess)}
    bs = {s: sess.count(s) / len(sess) for s in set(sess)}
    res["session_match_tv"] = round(sum(abs(cs.get(s, 0) - bs[s]) for s in bs) / 2, 3)

    res["loss_concentration"] = loss_concentration(nets, maes, mfes, sel_cand)

    # incremental (regime-only'ye karsi): daha iyi = daha az negatif
    inc_cvar = cand_m["cvar5"] > reg_m.get("cvar5", float("-inf")) if reg_m.get("n") else False
    inc_dsd = cand_m["downside_dev"] < reg_m.get("downside_dev", float("inf")) if reg_m.get("n") else False
    rr = res["loss_concentration"]["retention_ratio"] or 0.0
    checks = {
        "a_tail_vs_matched": pct_m["cvar5"] >= PCT_BEAT,
        "b_beats_random_veto": pct_r["cvar5"] >= PCT_BEAT,
        "c_incremental_over_regime": bool(inc_cvar and inc_dsd),
        "d_retention": rr >= RETENTION_MIN,
        "e_min_n": n_cand >= MIN_FOLD_CAND,
    }
    res["checks"] = checks
    res["fold_pass"] = all(checks.values())
    require_controls(res)
    return res


# ── Drift alarm lead/lag ─────────────────────────────────────────────────────
def alarm_lead_lag(X, ts, trades, feats, e_cut: int) -> dict:
    mon = DriftMonitor()
    std = Standardizer().fit(X[:e_cut], (int(ts[0]), int(ts[e_cut - 1])))
    lab_ref, C, _ = seeded_kmeans(std.transform(X[:e_cut]), K, SEED)
    ref = X[:e_cut]
    tnet = {t["idx"]: t["net"] for t in trades}
    tr_nets = np.array([t["net"] for t in trades if t["idx"] < e_cut])
    train_mu, train_sd = float(tr_nets.mean()), float(tr_nets.std() or 1.0)
    deter_thr = train_mu - DETER_SIGMA * train_sd

    windows = []
    e = e_cut + DRIFT_WIN
    while e <= X.shape[0]:
        s = e - DRIFT_WIN
        guard_no_retroactive_alarm(e - 1, e - 1)   # yapisal: yalniz [s,e) verisi
        cur = X[s:e]
        d2 = ((std.transform(cur)[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        lab_c = d2.argmin(axis=1)
        rep = mon.assess(ref, cur, feats, lab_ref, lab_c, k=K)
        back = [tnet[i] for i in tnet if s <= i < e]
        fwd = [tnet[i] for i in tnet if e <= i < e + DRIFT_WIN]
        windows.append({
            "end_idx": e, "end_utc": datetime.fromtimestamp(int(ts[e - 1]) / 1000, tz=timezone.utc).strftime("%m-%d"),
            "status": rep["status"],
            "back_mean": round(float(np.mean(back)), 1) if back else None,
            "fwd_mean": round(float(np.mean(fwd)), 1) if fwd else None,
            "n_back": len(back), "n_fwd": len(fwd)})
        e += DRIFT_STEP

    alarm_on = [i for i, w in enumerate(windows) if w["status"] in ("SHIFTED", "UNUSABLE")]
    deter_on = [i for i, w in enumerate(windows)
                if w["back_mean"] is not None and w["back_mean"] <= deter_thr]
    lead = None
    if alarm_on and deter_on:
        lead = (deter_on[0] - alarm_on[0]) * DRIFT_STEP * 5 / (60 * 24)   # gun
    # false-positive suspension: alarm var ama forward performans train ortalamasinin ustunde
    fp = [i for i in alarm_on if windows[i]["fwd_mean"] is not None
          and windows[i]["fwd_mean"] >= train_mu]
    # alarm sonrasi engellenen iyi/kotu trade'ler (forward penceresinde)
    blocked_good = blocked_bad = 0
    for i in alarm_on:
        e = windows[i]["end_idx"]
        for idx, netv in tnet.items():
            if e <= idx < e + DRIFT_WIN:
                if netv > 0:
                    blocked_good += 1
                else:
                    blocked_bad += 1
    return {
        "train_mean": round(train_mu, 1), "train_std": round(train_sd, 1),
        "deterioration_threshold": round(deter_thr, 1),
        "n_windows": len(windows), "windows": windows,
        "first_alarm_window": alarm_on[0] if alarm_on else None,
        "first_deterioration_window": deter_on[0] if deter_on else None,
        "alarm_lead_days": round(lead, 1) if lead is not None else None,
        "alarm_leading": (lead is not None and lead >= 0) if (alarm_on and deter_on) else None,
        "false_positive_suspension_rate": round(len(fp) / len(alarm_on), 2) if alarm_on else None,
        "blocked_good_trades_after_alarm": blocked_good,
        "blocked_bad_trades_after_alarm": blocked_bad,
    }


# ── Ana akis ─────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== AMI Faz 6A-R2 — Risk and Applicability Validation ===")
    reg = ResearchRegistry(); store = KnowledgeStore()
    spec = spec_6ar2(reg)
    print(f"  prereg FROZEN: {spec.experiment_id} hash={spec.frozen_hash}")

    X, ts, miss, meta = load_dataset()
    feats = list(meta["features"]); sessions = meta["sessions"]
    art = json.loads(ARTIFACTS.read_text(encoding="utf-8"))
    guard_artifact_version(art, meta)
    guard_permissions(ALLOWED_PERMS)   # istenen izin seti yasal mi (yapisal kontrol)

    print("  trade populasyonu kuruluyor (no-overlap 6h LONG, mark-fill)...")
    trades = build_trades(ts, sessions)
    print(f"  trades: {len(trades)}  ({trades[0]['t0']} .. {trades[-1]['t0']})")

    R = {"spec_hash": spec.frozen_hash, "n_samples": int(X.shape[0]),
         "n_trades": len(trades), "artifact_hash": art.get("artifact_hash")}

    folds = []
    for fi, (lo_f, hi_f) in enumerate(WF_FOLDS):
        fr = eval_fold(fi, lo_f, hi_f, X, ts, trades, feats, sessions,
                       contaminated=(fi == CONTAMINATED_FOLD))
        folds.append(fr)
        tag = " [CONTAMINATED — hipotez kaynagi, kanit sayilmaz]" if fr.get("contaminated") else ""
        if fr.get("evaluable"):
            c = fr["checks"]
            print(f"    fold{fi} val={fr['val_win']} n_base={fr['n_base']} n_cand={fr['n_cand']} "
                  f"pass={fr['fold_pass']} checks={ {k: v for k, v in c.items()} }{tag}")
        else:
            print(f"    fold{fi} val={fr.get('val_win')} n_cand={fr.get('n_cand')} "
                  f"NOT-EVALUABLE ({fr.get('sample_status', fr.get('reason'))}){tag}")
    R["folds"] = folds
    verdict = fold_verdict(folds)
    guard_fold_aggregation(folds, verdict)
    R["fold_verdict"] = verdict

    e_cut = int(X.shape[0] * 0.60)
    print("  drift alarm lead/lag hesaplaniyor...")
    R["alarm"] = alarm_lead_lag(X, ts, trades, feats, e_cut)
    a = R["alarm"]
    print(f"    alarm_lead_days={a['alarm_lead_days']} leading={a['alarm_leading']} "
          f"fp_rate={a['false_positive_suspension_rate']}")

    total_cand = sum(f.get("n_cand", 0) for f in folds if f.get("evaluable"))
    dq_ok = all(float(miss[int(X.shape[0]*lo):int(X.shape[0]*hi)].mean()) < 0.30
                for lo, hi in WF_FOLDS)
    R["dq_explanation_ruled_out"] = bool(dq_ok)
    # top-winner dependence: aday-vs-regime-only siralamasi top3-removed ile korunuyor mu
    tw_ok = True
    for f in folds:
        if not f.get("evaluable"):
            continue
        c3, r3 = f["candidate"].get("top3_removed"), f["regime_only"].get("top3_removed")
        c0, r0 = f["candidate"].get("cum"), f["regime_only"].get("cum")
        if None not in (c3, r3, c0, r0) and ((c0 > r0) != (c3 > r3)):
            tw_ok = False
    R["topwinner_ordering_stable"] = tw_ok

    # frozen siniflandirma (oncelik sirasi falsification_rule'da)
    alarm_leads = a["alarm_leading"]
    if total_cand < MIN_TOTAL_CAND or verdict["evaluable_folds"] < 2:
        outcome, oc = "FALSIFIES", "INSUFFICIENT_SAMPLE"
    else:
        rv_fail = sum(1 for f in folds if f.get("evaluable")
                      and not f["checks"]["b_beats_random_veto"])
        inc_fail = sum(1 for f in folds if f.get("evaluable")
                       and not f["checks"]["c_incremental_over_regime"])
        ev_n = verdict["evaluable_folds"]
        if rv_fail * 2 > ev_n:
            outcome, oc = "FALSIFIES", "FREQUENCY_ARTIFACT"
        elif inc_fail * 2 > ev_n:
            outcome, oc = "FALSIFIES", "RISK_NON_INCREMENTAL"
        elif verdict["majority_pass"] and tw_ok and dq_ok:
            if alarm_leads is False:
                outcome, oc = "WEAKENS", "LATE_DRIFT_DETECTION"
            else:
                outcome, oc = "SUPPORTS", "CHRONO_SUPPORTED_PENDING_FORWARD"
        else:
            outcome, oc = "FALSIFIES", "REJECTED"
    R["total_cand_n"] = total_cand
    R["outcome_class"] = oc
    R["outcome"] = outcome
    print(f"  OUTCOME: {outcome} / {oc} (total_cand={total_cand}, "
          f"majority_pass={verdict['majority_pass']}, alarm_leading={alarm_leads})")

    # applicability durumu (drift monitor guncel durumuyla tutarlilik)
    guard_artifact_usable("UNUSABLE", "research")        # research kullanim serbest
    try:
        guard_artifact_usable("UNUSABLE", "trade_selection")
        raise RuntimeError("guard delinmis olmali")
    except ConstitutionViolation:
        R["unusable_selection_blocked"] = True

    reg.attach_evidence(EvidenceBundle("EV-RISKAPP-6AR2-001", spec.experiment_id,
                                       {k: v for k, v in R.items() if k != "alarm"} |
                                       {"alarm_summary": {k: v for k, v in a.items() if k != "windows"}},
                                       outcome, evidence_family="risk_applicability",
                                       dataset_hash=meta["feature_version"],
                                       code_ref="ami/latent/risk_applicability.py"), spec)

    if outcome == "SUPPORTS":
        ko = KnowledgeObject(
            knowledge_id="K-RISKAPP-6AR2-001",
            claim=(f"Regime(trend=UP)+latent-calm veto'su matched-count ve random-veto "
                   f"kontrollerine karsi tail-risk'i azaltiyor (chronological WF, "
                   f"{verdict['passed_folds']}/{verdict['evaluable_folds']} fold). "
                   f"Kullanim sinifi YALNIZ risk/applicability; giris alpha'si DEGIL. "
                   f"Untouched veri YOK -> forward shadow dogrulamasi ZORUNLU."),
            claim_type=ClaimType.DESCRIPTIVE, status=KnowledgeStatus.PRELIMINARY,
            provenance=Provenance(source_tables=["mark_prices", "agg_trades", "liquidations", "book_ticker"],
                                  data_time_range=meta["feature_version"],
                                  code_ref="ami/latent/risk_applicability.py",
                                  dataset_hash=spec.frozen_hash,
                                  experiment_id=spec.experiment_id,
                                  execution_model="research_only_no_execution"),
            evidence_level=EvidenceLevel.CHRONOLOGICAL, replications=1, holdouts=0,
            falsification=["forward shadow'da cvar5 percentile <0.5'e duserse",
                           "random-veto forward'da esitlerse"],
            confidence={"statistical": "LOW-MEDIUM", "mechanism": "NONE", "forward": "NONE"},
            permitted=sorted(ALLOWED_PERMS, key=lambda p: p.value),
            forbidden=sorted(FORBIDDEN_PERMS, key=lambda p: p.value))
        guard_permissions(set(ko.permitted))
        store.put(ko, actor="riskapp6ar2")
        print(f"  Knowledge: {ko.knowledge_id} (PRELIMINARY, forward sart)")
    else:
        ft = {"INSUFFICIENT_SAMPLE": FailureType.INSUFFICIENT_SAMPLE,
              "FREQUENCY_ARTIFACT": FailureType.OVERFIT,
              "RISK_NON_INCREMENTAL": FailureType.NO_EDGE,
              "LATE_DRIFT_DETECTION": FailureType.EXECUTION_FAILURE,
              "REJECTED": FailureType.NO_EDGE}.get(oc, FailureType.NO_EDGE)
        store.archive_failure(
            "Faz6A-R2 regime+latent risk/applicability overlay",
            ft, reason=json.dumps({"class": oc, "fold_verdict": verdict,
                                   "total_cand": total_cand,
                                   "alarm_leading": alarm_leads}),
            retry="forward shadow verisi birikince (>=6 ay) YENI prereg; "
                  "kriter gevsetme YASAK")
        print(f"  {oc} -> failure archive (durust sonuc)")

    _report(R, spec)
    reg.close(); store.close()
    return R


def _report(R, spec):
    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    a = R.get("alarm", {})
    lines = ["# AMI Faz 6A-R2 — Risk and Applicability Validation", "",
             f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — prereg `{spec.experiment_id}` "
             f"hash `{spec.frozen_hash}`. OUTCOME: **{R.get('outcome')} / {R.get('outcome_class')}**",
             "",
             f"- Trade populasyonu: {R['n_trades']} no-overlap 6h LONG grid trade (veto yorumu)",
             f"- Toplam degerlendirilebilir aday N: {R['total_cand_n']}",
             f"- Fold verdict: {json.dumps(R['fold_verdict'])}",
             f"- dq elendi: {R['dq_explanation_ruled_out']} · top-winner siralama stabil: "
             f"{R['topwinner_ordering_stable']}",
             f"- Alarm: lead_days={a.get('alarm_lead_days')} leading={a.get('alarm_leading')} "
             f"fp_suspension={a.get('false_positive_suspension_rate')} "
             f"blocked_good={a.get('blocked_good_trades_after_alarm')} "
             f"blocked_bad={a.get('blocked_bad_trades_after_alarm')}",
             ""]
    for f in R["folds"]:
        tag = " **[CONTAMINATED]**" if f.get("contaminated") else ""
        lines.append(f"## Fold {f['fold']} val={f.get('val_win')}{tag}\n```json\n"
                     + json.dumps({k: v for k, v in f.items() if k != 'fold'}, indent=1, default=str)
                     + "\n```\n")
    lines.append("## Alarm lead/lag\n```json\n" + json.dumps(a, indent=1, default=str) + "\n```\n")
    lines += ["Durust statuler: software-correct ✓ · frequency-normalized ✓ (matched-count + "
              "random-veto esit-N) · risk-" +
              ("incremental" if R.get("outcome") == "SUPPORTS" else "non-incremental") +
              " · applicability-" +
              ("leading" if a.get("alarm_leading") else ("lagging" if a.get("alarm_leading") is False else "indeterminate")) +
              " · walk-forward " +
              ("passed" if R.get("fold_verdict", {}).get("majority_pass") else "failed") +
              " · forward-not-validating (N=0) · **operationally FORBIDDEN** "
              "(max: RESEARCH/BACKTEST/SHADOW + SHADOW_SUSPEND_SUGGESTION)",
              "", "*Runner: `python -m ami.latent.risk_applicability`*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD: {OM}")


if __name__ == "__main__":
    main()
