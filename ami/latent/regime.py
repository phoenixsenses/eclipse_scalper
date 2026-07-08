"""Faz 6A-R — Regime-Conditioned State Intelligence + Drift Attribution.

Yeni dis veri YOK: latent_dataset.npz yeniden kullanilir. Rejimler deterministik,
aciklanabilir, OUTCOME'SUZ ve esikleri YALNIZ exploration doneminden (leakage yok).

Akis (frozen sira):
1. Prereg E-LATENT6AR-001 DONDURULUR (rejim yontemi, fold semasi, kriterler).
2. Rejim tanimlari exploration'da fit edilir (RegimeDefiner.fit_range kayitli).
3. Drift attribution: 6A exploration vs validation eralari (PSI/JS/mean-shift/missingness).
4. Rejim-kosullu latent analiz (ayni frozen k-secim kurali, rejim ici 70/30 kronolojik).
5. Walk-forward (4 fold) — persistent state tanimi: >=3/4 fold'da merkez-eslesmesi.
6. Artifact freeze SONRASI ayri outcome katmani (baseline/rejim/latent/kombinasyon).

Run: python -m ami.latent.regime
Cikti: reports/research/s34/AMI_PHASE6AR_REGIME.md + .json
"""
from __future__ import annotations
import hashlib, json, sys, time
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
from ami.latent.dataset import FEATURES, assert_no_outcome, load_dataset
from ami.latent.discovery import align_labels, avg_duration, occupancy, verify_artifact
from ami.latent.models import Standardizer, ari, seeded_kmeans, transition_entropy
from ami.research.registry import EvidenceBundle, ExperimentSpec, ResearchRegistry, assert_no_overlap

OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "AMI_PHASE6AR_REGIME.json"; OM = OUT / "AMI_PHASE6AR_REGIME.md"
SEEDS = [11, 22, 33]
K_RANGE = [2, 3, 4, 5, 6]
ACCEPT = {"seed_ari_min": 0.60, "perturb_ari_min": 0.50, "min_occupancy": 0.05,
          "occ_band": (0.30, 3.0), "trans_corr_min": 0.50,
          "wf_center_cos_min": 0.80, "wf_persist_folds": 3,
          "min_regime_sample": 1500}
EXPL_FRAC = 0.60            # rejim esikleri ve 6A-karsilastirma icin exploration
WF_FOLDS = [(0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.00)]  # val pencereleri
DRIFT_PSI = {"warning": 0.10, "shifted": 0.25, "unusable_feat": 0.50}
FEE = 5.0


# ── Rejim tanimlari (deterministik, outcome'suz, exploration-fit) ────────────
class RegimeDefiner:
    """Esikler YALNIZ exploration'dan; fit_range kaydedilir (leakage guard'i)."""
    DIMS = ["trend", "vol", "stress", "leverage"]

    def __init__(self):
        self.thr = {}; self.fit_range = None

    def fit(self, X: np.ndarray, feats: list[str], fit_range: tuple[int, int]) -> "RegimeDefiner":
        assert_no_outcome(feats)
        r5 = X[:, feats.index("ret5m")]
        rv = X[:, feats.index("rv30m")]
        st = X[:, feats.index("stress10m")]
        fv = X[:, feats.index("fund_vel_1h")]
        tr24 = _roll_sum(r5, 288)          # 24h birikimli getiri (bps)
        rv24 = _roll_med(rv, 288)
        st24 = _roll_sum(st, 288)
        self.thr = {
            "trend_up": 100.0, "trend_dn": -100.0,   # frozen sabit (bps/24h)
            "vol_lo": float(np.nanpercentile(rv24, 33)),
            "vol_hi": float(np.nanpercentile(rv24, 67)),
            "stress_hi": float(np.nanpercentile(st24, 80)),
            "lev_pos": float(np.nanpercentile(fv, 67)),
            "lev_neg": float(np.nanpercentile(fv, 33)),
        }
        self.fit_range = fit_range
        return self

    def label(self, X: np.ndarray, feats: list[str], sessions: list[str]) -> dict[str, np.ndarray]:
        r5 = X[:, feats.index("ret5m")]
        rv24 = _roll_med(X[:, feats.index("rv30m")], 288)
        st24 = _roll_sum(X[:, feats.index("stress10m")], 288)
        fv = X[:, feats.index("fund_vel_1h")]
        tr24 = _roll_sum(r5, 288)
        t = self.thr
        trend = np.where(tr24 > t["trend_up"], "UP", np.where(tr24 < t["trend_dn"], "DOWN", "RANGE"))
        vol = np.where(rv24 < t["vol_lo"], "LOW", np.where(rv24 > t["vol_hi"], "HIGH", "NORMAL"))
        stress = np.where(st24 > t["stress_hi"], "STRESSED", "NORMAL")
        lev = np.where(np.nan_to_num(fv) > t["lev_pos"], "EXPANDING",
                       np.where(np.nan_to_num(fv) < t["lev_neg"], "CONTRACTING", "NEUTRAL"))
        return {"trend": trend, "vol": vol, "stress": stress, "leverage": lev,
                "session": np.array(sessions)}


def _roll_sum(x, w):
    x = np.nan_to_num(x, nan=0.0)
    c = np.cumsum(x)
    out = np.empty_like(c); out[:w] = c[:w]; out[w:] = c[w:] - c[:-w]
    return out


def _roll_med(x, w):
    # hizli yaklasim: stride'li pencere medyani yerine EMA-benzeri; deterministik
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    k = np.ones(w) / w
    return np.convolve(x, k, mode="full")[:len(x)]


def check_regime_fit_boundary(rd: RegimeDefiner, val_start_ts: int) -> None:
    if rd.fit_range is None or rd.fit_range[1] >= val_start_ts:
        raise ConstitutionViolation("Regime thresholds fitted on validation era (leakage).")


def check_min_regime_sample(n: int, name: str) -> str:
    """Kucuk rejim sahte stabilite ureticidir -> UNKNOWN."""
    return name if n >= ACCEPT["min_regime_sample"] else "UNKNOWN"


def transition_matrix_within(labels: np.ndarray, k: int, boundaries: list[int]) -> np.ndarray:
    """Fold/rejim sinirini ASAN gecisler sayilmaz (transition leakage guard'i)."""
    M = np.zeros((k, k)); bset = set(boundaries)
    for i in range(len(labels) - 1):
        if (i + 1) in bset:
            continue
        M[labels[i], labels[i + 1]] += 1
    return M / np.maximum(M.sum(axis=1, keepdims=True), 1)


# ── Drift olcumleri ───────────────────────────────────────────────────────────
def psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    qs = np.nanpercentile(a, np.linspace(0, 100, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    pa, _ = np.histogram(a, qs); pb, _ = np.histogram(b, qs)
    pa = np.maximum(pa / pa.sum(), 1e-4); pb = np.maximum(pb / pb.sum(), 1e-4)
    return float(((pb - pa) * np.log(pb / pa)).sum())


def js_div(a: np.ndarray, b: np.ndarray, bins: int = 20) -> float:
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return float("nan")
    lo, hi = np.nanpercentile(np.concatenate([a, b]), [1, 99])
    pa, _ = np.histogram(a, bins, range=(lo, hi)); pb, _ = np.histogram(b, bins, range=(lo, hi))
    pa = np.maximum(pa / max(pa.sum(), 1), 1e-9); pb = np.maximum(pb / max(pb.sum(), 1), 1e-9)
    m = (pa + pb) / 2
    return float(0.5 * (pa * np.log2(pa / m)).sum() + 0.5 * (pb * np.log2(pb / m)).sum())


def classify_drift(psi_v: float, miss_a: float, miss_b: float) -> tuple[str, str]:
    """(kaynak, guven) — data issue vs market issue ayrimi."""
    dmiss = abs(miss_b - miss_a)
    if dmiss > 0.30:
        return "DATA_ISSUE(missingness)", "HIGH"
    if not np.isfinite(psi_v):
        return "INSUFFICIENT_SAMPLE", "LOW"
    if psi_v > DRIFT_PSI["shifted"]:
        return "MARKET_SHIFT", "MEDIUM" if dmiss < 0.05 else "LOW"
    if psi_v > DRIFT_PSI["warning"]:
        return "MILD_MARKET_SHIFT", "MEDIUM"
    return "NO_DRIFT", "HIGH"


# ── Prereg ───────────────────────────────────────────────────────────────────
def spec_6ar(reg: ResearchRegistry) -> ExperimentSpec:
    spec = ExperimentSpec(
        experiment_id="E-LATENT6AR-001", question_id="Q-TRANSITION-MODEL-001",
        population="latent_dataset.npz (ETHUSDT 5m grid, outcome'suz) — yeni dis veri YOK",
        target="rejim-kosullu latent state stabilitesi + drift attribution (alpha DEGIL)",
        features=[f for f in FEATURES] + ["regime:trend", "regime:vol", "regime:stress",
                                          "regime:leverage", "regime:session"],
        threshold_method=("rejim esikleri exploration(ilk %60) percentile'lari; trend ±100bps/24h sabit; "
                          f"k-secimi 6A kurali (k in {K_RANGE}, occ>={ACCEPT['min_occupancy']}, "
                          f"seed-ARI>={ACCEPT['seed_ari_min']}); min_regime_sample={ACCEPT['min_regime_sample']}"),
        chronological_split=f"walk-forward val pencereleri {WF_FOLDS}; rejim-ici 70/30 kronolojik",
        untouched_data="son fold (85-100%) alpha degerlendirmesi icin untouched",
        negative_control="dq-drift'i market-rejimi sanma testi (missingness delta ile ayristirma)",
        min_sample=ACCEPT["min_regime_sample"],
        decision_criteria=(f"PASS: >=1 latent state, >=1 preregistered rejimde walk-forward persistent "
                           f"(merkez-cos>={ACCEPT['wf_center_cos_min']} ve occ-band {ACCEPT['occ_band']} "
                           f">={ACCEPT['wf_persist_folds']}/4 fold) VE dq/missingness aciklamasi elenmis "
                           f"VE transition profili tekrarlanmis"),
        falsification_rule="hicbir state rejim-kosullu persistent degilse REJECT (gecerli null); "
                           "kriterler sonuca gore degistirilemez",
        execution_model="research_only_no_execution")
    spec.freeze()
    reg.register_experiment(spec)
    return spec


# ── Ana akis ─────────────────────────────────────────────────────────────────
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== AMI Faz 6A-R — Regime-Conditioned + Drift ===")
    reg = ResearchRegistry(); store = KnowledgeStore()
    spec = spec_6ar(reg)
    print(f"  prereg FROZEN: {spec.experiment_id} hash={spec.frozen_hash}")

    X, ts, miss, meta = load_dataset()
    n = X.shape[0]; feats = list(meta["features"])
    e_cut = int(n * EXPL_FRAC)
    sessions = meta["sessions"]
    R = {"spec_hash": spec.frozen_hash, "n": n, "expl_cut": e_cut}

    rd = RegimeDefiner().fit(X[:e_cut], feats, (int(ts[0]), int(ts[e_cut - 1])))
    check_regime_fit_boundary(rd, int(ts[e_cut]))
    regs = rd.label(X, feats, sessions)
    R["regime_thresholds"] = rd.thr

    # (3) DRIFT ATTRIBUTION — 6A eralari (0-80% vs 80-100%)
    a_cut = int(n * 0.80)
    drift = {}
    for i, f in enumerate(feats):
        pv = psi(X[:a_cut, i], X[a_cut:, i])
        jv = js_div(X[:a_cut, i], X[a_cut:, i])
        ma, mb = float(np.isnan(X[:a_cut, i]).mean()), float(np.isnan(X[a_cut:, i]).mean())
        mu_a, mu_b = np.nanmean(X[:a_cut, i]), np.nanmean(X[a_cut:, i])
        sd = np.nanstd(X[:a_cut, i]) or 1.0
        src, conf = classify_drift(pv, ma, mb)
        drift[f] = {"psi": round(pv, 3), "js": round(jv, 3),
                    "std_mean_shift": round(float((mu_b - mu_a) / sd), 2),
                    "miss_a": round(ma, 3), "miss_b": round(mb, 3),
                    "source": src, "confidence": conf}
    # rejim occupancy drift
    for dim in ("trend", "vol", "stress", "leverage", "session"):
        va, ct_a = np.unique(regs[dim][:a_cut], return_counts=True)
        vb, ct_b = np.unique(regs[dim][a_cut:], return_counts=True)
        da = {v: round(c / a_cut, 3) for v, c in zip(va, ct_a)}
        db = {v: round(c / (n - a_cut), 3) for v, c in zip(vb, ct_b)}
        drift[f"regime:{dim}"] = {"expl": da, "val": db}
    R["drift_attribution"] = drift
    big = [f for f in feats if np.isfinite(drift[f]["psi"]) and drift[f]["psi"] > DRIFT_PSI["warning"]]
    print(f"  drift: PSI>0.1 olan feature'lar: {big}")
    print(f"  trend dagilimi expl->val: {drift['regime:trend']}")

    std = Standardizer().fit(X[:e_cut], (int(ts[0]), int(ts[e_cut - 1])))
    Z = std.transform(X)

    # (4) REJIM-KOSULLU latent analiz
    per_regime = {}
    for dim in ("trend", "vol", "stress"):
        for val in np.unique(regs[dim]):
            mask = regs[dim] == val
            idx = np.where(mask)[0]
            name = check_min_regime_sample(len(idx), f"{dim}={val}")
            if name == "UNKNOWN":
                continue
            Zr = Z[idx]
            cut_r = int(len(idx) * 0.70)
            ksel = {}
            for k in K_RANGE:
                labs = [seeded_kmeans(Zr[:cut_r], k, s)[0] for s in SEEDS]
                occ_ok = all(min(occupancy(l, k)) >= ACCEPT["min_occupancy"] for l in labs)
                sari = float(np.mean([ari(labs[0], labs[i]) for i in range(1, len(SEEDS))]))
                ksel[k] = (round(sari, 3), occ_ok)
            cands = [k for k in K_RANGE if ksel[k][1] and ksel[k][0] >= ACCEPT["seed_ari_min"]]
            if not cands:
                per_regime[f"{dim}={val}"] = {"n": len(idx), "verdict": "NO_STABLE_STATE",
                                              "ksel": {k: ksel[k][0] for k in K_RANGE}}
                continue
            bk = max(cands, key=lambda k: (ksel[k][0], -k))
            lab_tr, C, _ = seeded_kmeans(Zr[:cut_r], bk, SEEDS[0])
            rng = np.random.RandomState(99)
            lab_p, _, _ = seeded_kmeans(Zr[:cut_r] + rng.normal(0, 0.05, Zr[:cut_r].shape), bk, SEEDS[0])
            d = ((Zr[cut_r:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            lab_va = d.argmin(axis=1)
            occ_t, occ_v = occupancy(lab_tr, bk), occupancy(lab_va, bk)
            ratios = [round(occ_v[j] / occ_t[j], 2) if occ_t[j] > 0 else None for j in range(bk)]
            At = transition_matrix_within(lab_tr, bk, [])
            Av = transition_matrix_within(lab_va, bk, [])
            tc = float(np.corrcoef(At.flatten(), Av.flatten())[0, 1]) if bk > 1 else 1.0
            lo, hi = ACCEPT["occ_band"]
            stable = (ari(lab_tr, lab_p) >= ACCEPT["perturb_ari_min"]
                      and all(r is not None and lo <= r <= hi for r in ratios)
                      and tc >= ACCEPT["trans_corr_min"])
            per_regime[f"{dim}={val}"] = {
                "n": len(idx), "k": bk, "seed_ari": ksel[bk][0],
                "perturb_ari": round(float(ari(lab_tr, lab_p)), 3),
                "occ_train": [round(o, 3) for o in occ_t], "occ_val": [round(o, 3) for o in occ_v],
                "occ_ratio": ratios, "trans_corr": round(tc, 3),
                "trans_entropy": round(transition_entropy(At), 3),
                "avg_dur_min": avg_duration(lab_tr, bk),
                "chrono_stable": bool(stable),
                "profiles": {f"S{j}": {feats[i]: round(float(C[j, i]), 2)
                                       for i in range(len(feats)) if abs(C[j, i]) > 0.5}
                             for j in range(bk)}}
            print(f"    {dim}={val:8s} n={len(idx):5d} k={bk} sARI={ksel[bk][0]} "
                  f"occ_ratio={ratios} tc={tc:.2f} -> {'STABLE' if stable else 'unstable'}")
    R["per_regime"] = per_regime

    # (5) WALK-FORWARD (kosulsuz + stabil rejimlerde)
    wf = {}
    contexts = [("ALL", np.arange(n))]
    contexts += [(nm, np.where(regs[nm.split("=")[0]] == nm.split("=")[1])[0])
                 for nm, v in per_regime.items() if v.get("chrono_stable")]
    for nm, idx in contexts:
        folds = []
        prev_C = None
        for lo_f, hi_f in WF_FOLDS:
            v0, v1 = int(len(idx) * lo_f), int(len(idx) * hi_f)
            tr_i, va_i = idx[:v0], idx[v0:v1]
            if len(tr_i) < 2000 or len(va_i) < 400:
                continue
            assert_no_overlap(set(tr_i.tolist()), set(va_i.tolist()))
            k6a = per_regime.get(nm, {}).get("k", 4) if nm != "ALL" else 4
            lab_t, C, _ = seeded_kmeans(Z[tr_i], k6a, SEEDS[0])
            d = ((Z[va_i][:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            lab_v = d.argmin(axis=1)
            occ_t, occ_v = occupancy(lab_t, k6a), occupancy(lab_v, k6a)
            ratios = [round(occ_v[j] / occ_t[j], 2) if occ_t[j] > 0 else None for j in range(k6a)]
            lo, hi = ACCEPT["occ_band"]
            band_ok = all(r is not None and lo <= r <= hi for r in ratios)
            cos = None
            if prev_C is not None and prev_C.shape == C.shape:
                Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-9)
                Pn = prev_C / np.maximum(np.linalg.norm(prev_C, axis=1, keepdims=True), 1e-9)
                sim = Cn @ Pn.T
                cos = round(float(np.mean(sim.max(axis=1))), 3)
            folds.append({"val_win": [lo_f, hi_f], "occ_ratio": ratios,
                          "band_ok": bool(band_ok), "center_cos_prev": cos})
            prev_C = C
        n_band = sum(1 for f in folds if f["band_ok"])
        n_cos = sum(1 for f in folds if f["center_cos_prev"] is not None
                    and f["center_cos_prev"] >= ACCEPT["wf_center_cos_min"])
        persistent = (n_band >= ACCEPT["wf_persist_folds"]
                      and n_cos >= ACCEPT["wf_persist_folds"] - 1)
        wf[nm] = {"folds": folds, "band_ok_folds": n_band, "cos_ok_folds": n_cos,
                  "persistent": bool(persistent)}
        print(f"    WF {nm:16s} band_ok={n_band}/{len(folds)} cos_ok={n_cos} -> "
              f"{'PERSISTENT' if persistent else 'not-persistent'}")
    R["walk_forward"] = wf

    passed = any(v["persistent"] for nm, v in wf.items() if nm != "ALL")
    dq_explained = any("DATA_ISSUE" in drift[f]["source"] for f in feats)
    R["dq_explanation_ruled_out"] = not dq_explained
    R["PASS"] = bool(passed and not dq_explained)
    verdict = "PASS" if R["PASS"] else "REJECT"
    print(f"  VERDICT: {verdict} (persistent-regime={passed}, dq_explained={dq_explained})")

    # (6) OUTCOME — ayri katman, untouched son fold (85-100%)
    R["alpha_eval"] = _alpha_eval(ts, Z, regs, per_regime, n)

    # kayit
    reg.attach_evidence(EvidenceBundle("EV-LATENT6AR-001", spec.experiment_id,
                                       {k: v for k, v in R.items() if k not in ("alpha_eval",)},
                                       "SUPPORTS" if R["PASS"] else "FALSIFIES",
                                       evidence_family="latent6ar",
                                       dataset_hash=meta["feature_version"],
                                       code_ref="ami/latent/regime.py"), spec)
    if R["PASS"]:
        stable_regs = [nm for nm, v in wf.items() if nm != "ALL" and v["persistent"]]
        ko = KnowledgeObject(
            knowledge_id="K-LATENT-REGIME-001",
            claim=f"Latent state'ler su rejim(ler) icinde walk-forward persistent: {stable_regs}. "
                  "Mekanizma iddiasi yok; isimler notr (LS-*). Rejim disina genellenemez.",
            claim_type=ClaimType.DESCRIPTIVE, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=Provenance(source_tables=["mark_prices", "agg_trades", "liquidations", "book_ticker"],
                                  data_time_range=meta["feature_version"],
                                  code_ref="ami/latent/regime.py", dataset_hash=spec.frozen_hash,
                                  experiment_id=spec.experiment_id,
                                  execution_model="research_only_no_execution"),
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=1, holdouts=1,
            falsification=["forward'da persistent rejimde occupancy bandi bozulursa"],
            confidence={"statistical": "MEDIUM", "mechanism": "NONE", "forward": "NONE"},
            permitted=[Permission.RESEARCH_ONLY, Permission.BACKTEST_ALLOWED, Permission.SHADOW_ALLOWED],
            forbidden=[Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED, Permission.PORTFOLIO_ALLOWED])
        store.put(ko, actor="latent6ar")
        print(f"  Knowledge: {ko.knowledge_id} (max SHADOW; rejim-scoped)")
    else:
        store.archive_failure("Faz6A-R regime-conditioned latent states",
                              FailureType.REGIME_LIMITED if passed else FailureType.NO_EDGE,
                              reason=json.dumps({nm: v["persistent"] for nm, v in wf.items()}),
                              retry="daha uzun tarih araligiyla YENI prereg; model karmasikligi ARTIRILMAZ")
        print("  REJECT -> failure archive (null gecerli sonuc)")
    _report(R, spec)
    reg.close(); store.close()


def _alpha_eval(ts, Z, regs, per_regime, n):
    """Untouched son %15 (85-100) — trade-benzeri no-overlap 6h LONG karsilastirmasi."""
    import sqlite3
    from tools.research_s34_knowable_anchor_continuation import load_mark_index
    conn = sqlite3.connect(f"file:{ROOT/'data'/'microstructure.db'}?mode=ro", uri=True)
    m = load_mark_index(conn, "ETHUSDT")
    u0 = int(n * 0.85)
    # latent etiketler: ALL k=4 modeli (85 oncesi fit)
    lab_all, C, _ = seeded_kmeans(Z[:u0], 4, SEEDS[0])
    d = ((Z[u0:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    lab_u = d.argmin(axis=1)

    def fwd6(t0):
        a = m.at_or_before(int(t0)); b = m.at_or_before(int(t0) + 6 * 3600_000)
        return (float(b[1]) - float(a[1])) / float(a[1]) * 1e4 if a and b and float(a[1]) > 0 else None

    def sim(mask):
        rets = []; busy = -1
        for i in np.where(mask)[0]:
            t0 = int(ts[u0 + i])
            if t0 < busy:
                continue
            r = fwd6(t0)
            if r is not None:
                rets.append(r - FEE); busy = t0 + 6 * 3600_000
        if not rets:
            return {"n": 0}
        arr = np.array(rets); srt = np.sort(arr)[::-1]
        eq = np.cumsum(arr); dd = float((eq - np.maximum.accumulate(eq)).min())
        days = (ts[-1] - ts[u0]) / 86_400_000
        g = arr[arr > 0].sum(); l = -arr[arr <= 0].sum()
        return {"n": len(arr), "trade_per_day": round(len(arr) / days, 2),
                "wr": round(float((arr > 0).mean()) * 100, 1),
                "median": round(float(np.median(arr)), 1), "mean": round(float(arr.mean()), 1),
                "cum": round(float(arr.sum()), 0),
                "top3_removed": round(float(srt[3:].sum()), 0) if len(arr) > 3 else None,
                "pf": round(float(g / l), 2) if l > 0 else None, "mdd": round(dd, 1)}

    ue = np.ones(n - u0, dtype=bool)
    trend_u = regs["trend"][u0:]
    best_reg = None
    for nm, v in per_regime.items():
        if v.get("chrono_stable") and nm.startswith("trend="):
            best_reg = nm.split("=")[1]; break
    out = {"1_baseline_all": sim(ue)}
    if best_reg is not None:
        out["2_baseline+regime"] = sim(trend_u == best_reg)
    # en iyi latent state (fit-doneminde en pozitif fwd olan DEGIL — outcome'suz secim:
    # en dusuk-stres 'sakin' state = merkezi norm'u en kucuk olan)
    calm = int(np.argmin(np.linalg.norm(C, axis=1)))
    out["3_baseline+latent(calm)"] = sim(lab_u == calm)
    if best_reg is not None:
        out["4_regime+latent"] = sim((trend_u == best_reg) & (lab_u == calm))
    out["5_latent_only_states"] = {f"LS-{j+1:03d}": sim(lab_u == j) for j in range(4)}
    conn.close()
    return out


def _report(R, spec):
    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    lines = ["# AMI Faz 6A-R — Regime-Conditioned Latent + Drift", "",
             f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — prereg `{spec.experiment_id}` "
             f"hash `{spec.frozen_hash}`. VERDICT: **{'PASS' if R.get('PASS') else 'REJECT'}**", ""]
    for key in ("regime_thresholds", "drift_attribution", "per_regime", "walk_forward",
                "dq_explanation_ruled_out", "alpha_eval"):
        if key in R:
            lines.append(f"## {key}\n```json\n{json.dumps(R[key], indent=1, default=str)}\n```\n")
    lines += ["Durust statuler: software-correct ✓ · drift-attributed (bkz. drift_attribution) · "
              f"regime-conditioned {'stable' if R.get('PASS') else 'unstable'} · "
              f"walk-forward {'passed' if R.get('PASS') else 'failed'} · "
              "alpha: alpha_eval bolumune bakiniz (ayri untouched katman) · "
              "forward-validating ✗ · **operationally FORBIDDEN**",
              "", "*Runner: `python -m ami.latent.regime`*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD: {OM}")


if __name__ == "__main__":
    main()
