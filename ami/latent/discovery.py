"""Faz 6A orkestratoru — prereg -> kesif -> stabilite -> FREEZE -> validasyon -> outcome.

Siralama sozlesmesi (mutation'larla korunur):
1. ExperimentSpec DONDURULUR (population, exploration/validation araligi, k-secim
   kurali, kabul/falsifikasyon kriterleri) — model kosulmadan ONCE.
2. Exploration'da fit + k-secimi + stabilite. Validation'a BAKILMAZ.
3. State'ler FREEZE edilir (artifact + hash).
4. Validation'da relabel + stabilite/tekrar-gorunum metrikleri.
5. Outcome degerlendirmesi EN SON, ayri katman (mark index'ten; dataset'te outcome yok).

Run: python -m ami.latent.discovery
Cikti: reports/research/s34/AMI_PHASE6A_LATENT.md + .json + data/ami/latent_artifacts.json
"""
from __future__ import annotations
import hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ami.enums import ClaimType, EvidenceLevel, FailureType, KnowledgeStatus, Permission
from ami.knowledge.objects import KnowledgeObject, Provenance
from ami.knowledge.store import KnowledgeStore
from ami.latent.dataset import FEATURES, load_dataset, assert_no_outcome
from ami.latent.models import (Standardizer, ari, cusum_changepoints, hmm_fit,
                               seeded_kmeans, transition_entropy, transition_matrix)
from ami.research.registry import EvidenceBundle, ExperimentSpec, ResearchRegistry

OUT = ROOT / "reports" / "research" / "s34"
OJ = OUT / "AMI_PHASE6A_LATENT.json"; OM = OUT / "AMI_PHASE6A_LATENT.md"
ART = ROOT / "data" / "ami" / "latent_artifacts.json"
SEEDS = [11, 22, 33, 44, 55]
K_RANGE = [2, 3, 4, 5, 6]
ACCEPT = {"seed_ari_min": 0.60, "perturb_ari_min": 0.50, "min_occupancy": 0.05,
          "val_occ_ratio": (0.30, 3.0), "val_trans_corr_min": 0.50}
VAL_SPLIT = 0.80   # kronolojik: son %20 untouched validation
FEE = 5.0


def spec_6a(reg: ResearchRegistry) -> ExperimentSpec:
    spec = ExperimentSpec(
        experiment_id="E-LATENT6A-001", question_id="Q-TRANSITION-MODEL-001",
        population="ETHUSDT 5m grid, book-coverage window (latent_dataset.npz), outcome'suz",
        target="stabil + kronolojik tekrarlanabilir latent state'ler (alpha DEGIL)",
        features=[f for f in FEATURES],
        threshold_method=(f"k-secim kurali: k in {K_RANGE}, once min_occupancy>={ACCEPT['min_occupancy']} "
                          f"ve seed-ARI>={ACCEPT['seed_ari_min']} saglayanlar, iclerinden en yuksek seed-ARI; "
                          "esitlikte kucuk k"),
        chronological_split=f"exploration ilk {int(VAL_SPLIT*100)}% / untouched validation son {int(100-VAL_SPLIT*100)}%",
        untouched_data=f"kronolojik son {int(100-VAL_SPLIT*100)}% — k-secimi ve stabilite SONRASI yalniz relabel",
        negative_control="missingness-mask'tan uretilmis sahte-feature seti stabil state VERMEMELI",
        min_sample=1000,
        decision_criteria=(f"KABUL: seed-ARI>={ACCEPT['seed_ari_min']} VE perturb-ARI>={ACCEPT['perturb_ari_min']} "
                           f"VE her state val occupancy orani {ACCEPT['val_occ_ratio']} icinde "
                           f"VE val transition korelasyonu>={ACCEPT['val_trans_corr_min']}"),
        falsification_rule="hicbir k kabul kriterini saglamazsa NO_STABLE_STATE (gecerli null)",
        execution_model="research_only_no_execution")
    spec.freeze()
    reg.register_experiment(spec)
    return spec


def verify_artifact(art: dict) -> bool:
    """Model artifact hash dogrulamasi (version/artifact mismatch guard'i)."""
    a = dict(art); stored = a.pop("artifact_hash", "")
    return hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()[:16] == stored


def assert_versions(artifact: dict, meta: dict) -> None:
    """Feature/schema version mismatch guard'i: eslesmeyen artifact kullanilabilir DEGIL."""
    if artifact.get("feature_version") != meta.get("feature_version") or \
       artifact.get("schema_version") != meta.get("schema_version"):
        raise ValueError(f"VERSION MISMATCH: artifact({artifact.get('schema_version')}/"
                         f"{artifact.get('feature_version')}) vs dataset({meta.get('schema_version')}/"
                         f"{meta.get('feature_version')})")


def occupancy(labels, k):
    return [float((labels == j).mean()) for j in range(k)]


def avg_duration(labels, k, step_min=5):
    durs = {j: [] for j in range(k)}
    cur = labels[0]; ln = 1
    for x in labels[1:]:
        if x == cur:
            ln += 1
        else:
            durs[cur].append(ln); cur = x; ln = 1
    durs[cur].append(ln)
    return {j: round(step_min * (np.mean(durs[j]) if durs[j] else 0), 1) for j in range(k)}


def align_labels(ref, other, k):
    """Permutation-invariant hizalama (greedy contingency)."""
    C = np.zeros((k, k))
    for a, b in zip(ref, other):
        C[a, b] += 1
    mapping = {}
    used = set()
    for _ in range(k):
        i, j = np.unravel_index(np.argmax(C), C.shape)
        mapping[j] = i; C[i, :] = -1; C[:, j] = -1; used.add(i)
    return np.array([mapping[x] for x in other])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=== AMI Faz 6A — Latent State Discovery ===")
    reg = ResearchRegistry(); store = KnowledgeStore()
    spec = spec_6a(reg)
    print(f"  prereg FROZEN: {spec.experiment_id} hash={spec.frozen_hash}")

    X, ts, miss, meta = load_dataset()
    n = X.shape[0]
    cut = int(n * VAL_SPLIT)
    # feature secimi: exploration>%30 VEYA herhangi bir erada >%90 eksik -> model DISI
    from ami.latent.dataset import era_missing_drop
    keep = era_missing_drop(np.asarray(miss), cut)
    expl_missing = np.isnan(X[:cut]).mean(axis=0)
    dropped = [(FEATURES[i], round(float(expl_missing[i]), 3)) for i in range(len(FEATURES)) if i not in keep]
    feats = [FEATURES[i] for i in keep]
    assert_no_outcome(feats)
    Xk = X[:, keep]
    print(f"  n={n} cut={cut} features={feats} dropped={dropped}")
    std = Standardizer().fit(Xk[:cut], (int(ts[0]), int(ts[cut - 1])))   # leakage yok
    Ze = std.transform(Xk[:cut]); Zv = std.transform(Xk[cut:])

    R = {"spec_hash": spec.frozen_hash, "n": n, "cut": cut, "features": feats,
         "dropped_features": dropped}

    # (1) change-point (kesif oncesi betimleyici)
    cps = cusum_changepoints(Ze[:, feats.index("rv30m")] if "rv30m" in feats else Ze[:, 0])
    R["changepoints"] = {"n_cp": len(cps), "per_day": round(len(cps) / ((ts[cut-1]-ts[0])/86_400_000), 2)}
    print(f"  changepoints: {R['changepoints']}")

    # (2/3) k-secimi: kmeans seed-stabilite (frozen kural)
    ksel = {}
    for k in K_RANGE:
        labs = [seeded_kmeans(Ze, k, s)[0] for s in SEEDS]
        occ_ok = all(min(occupancy(l, k)) >= ACCEPT["min_occupancy"] for l in labs)
        aris = [ari(labs[0], labs[i]) for i in range(1, len(labs))]
        ksel[k] = {"seed_ari": round(float(np.mean(aris)), 3), "occ_ok": bool(occ_ok)}
        print(f"    k={k} seed_ARI={ksel[k]['seed_ari']} occ_ok={occ_ok}")
    R["k_selection"] = ksel
    cands = [k for k in K_RANGE if ksel[k]["occ_ok"] and ksel[k]["seed_ari"] >= ACCEPT["seed_ari_min"]]
    if not cands:
        verdict = "NO_STABLE_STATE"
        R["verdict"] = verdict
        _finalize_null(store, reg, spec, R)
        return
    best_k = max(cands, key=lambda k: (ksel[k]["seed_ari"], -k))
    R["chosen_k"] = best_k
    print(f"  secilen k={best_k}")

    # ana fit (seed=SEEDS[0]) + perturbation stabilite
    lab_e, C, _ = seeded_kmeans(Ze, best_k, SEEDS[0])
    rng = np.random.RandomState(99)
    Zp = Ze + rng.normal(0, 0.05, Ze.shape)
    lab_p, _, _ = seeded_kmeans(Zp, best_k, SEEDS[0])
    perturb_ari = ari(lab_e, lab_p)
    R["perturb_ari"] = round(float(perturb_ari), 3)
    # HMM capraz-dogrulama (yontem #2): kmeans state'leriyle ortusme
    lab_h, A_h, mu_h, ll = hmm_fit(Ze[:: max(1, len(Ze)//8000)], best_k, SEEDS[0])
    R["hmm_crosscheck_ari"] = round(float(ari(lab_e[::max(1, len(Ze)//8000)][:len(lab_h)], lab_h[:len(lab_e[::max(1, len(Ze)//8000)])])), 3)
    print(f"  perturb_ARI={R['perturb_ari']}  hmm_crosscheck_ARI={R['hmm_crosscheck_ari']}")

    # (4) FREEZE — artifacts + hash (isimler notr LS-00x)
    artifact = {"schema_version": meta["schema_version"], "feature_version": meta["feature_version"],
                "features": feats, "standardizer": {"mu": std.mu.tolist(), "sd": std.sd.tolist(),
                                                    "fit_range": std.fit_range},
                "k": best_k, "centers": C.tolist(), "seed": SEEDS[0],
                "state_names": [f"LS-{i+1:03d}" for i in range(best_k)],
                "frozen_utc": datetime.now(timezone.utc).isoformat()}
    ah = hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest()[:16]
    artifact["artifact_hash"] = ah
    ART.write_text(json.dumps(artifact), encoding="utf-8")
    R["artifact_hash"] = ah
    print(f"  FROZEN artifacts: {ART} hash={ah}")

    # (5) validation relabel (untouched)
    d = ((Zv[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    lab_v = d.argmin(axis=1)
    occ_e, occ_v = occupancy(lab_e, best_k), occupancy(lab_v, best_k)
    occ_ratio = [round(occ_v[j] / occ_e[j], 2) if occ_e[j] > 0 else None for j in range(best_k)]
    Ae, Av = transition_matrix(lab_e, best_k), transition_matrix(lab_v, best_k)
    tc = float(np.corrcoef(Ae.flatten(), Av.flatten())[0, 1])
    lo, hi = ACCEPT["val_occ_ratio"]
    val_ok = all(r is not None and lo <= r <= hi for r in occ_ratio) and tc >= ACCEPT["val_trans_corr_min"]
    accepted = (R["perturb_ari"] >= ACCEPT["perturb_ari_min"]) and val_ok
    R.update({"occupancy_expl": [round(o, 3) for o in occ_e],
              "occupancy_val": [round(o, 3) for o in occ_v],
              "occ_ratio_val": occ_ratio,
              "avg_duration_min_expl": avg_duration(lab_e, best_k),
              "transition_matrix_expl": np.round(Ae, 3).tolist(),
              "transition_matrix_val": np.round(Av, 3).tolist(),
              "transition_entropy": round(transition_entropy(Ae), 3),
              "val_transition_corr": round(tc, 3), "validation_ok": bool(val_ok),
              "ACCEPTED": bool(accepted)})
    print(f"  occ_ratio_val={occ_ratio} trans_corr={tc:.3f} -> {'ACCEPTED' if accepted else 'REJECTED'}")

    # feature attribution + taxonomy overlap + dq dagilimi
    R["state_profiles"] = {f"LS-{j+1:03d}": {feats[i]: round(float(C[j, i]), 2) for i in range(len(feats))}
                           for j in range(best_k)}
    sess = np.array(meta["sessions"])[:cut]
    R["session_dist"] = {f"LS-{j+1:03d}": {s: round(float((sess[lab_e == j] == s).mean()), 2)
                                           for s in ("US", "EUROPE", "OFF")} for j in range(best_k)}
    dq = np.array(meta["dq_ok"])[:cut]
    R["dq_dist"] = {f"LS-{j+1:03d}": round(float(dq[lab_e == j].mean()), 3) for j in range(best_k)}
    # mevcut taxonomy overlap: stress-bazli cascade-aktif + trend isareti
    stress_i = feats.index("stress10m") if "stress10m" in feats else None
    ret1h_i = feats.index("ret1h") if "ret1h" in feats else None
    overlap = {}
    for j in range(best_k):
        mask = lab_e == j
        overlap[f"LS-{j+1:03d}"] = {
            "cascade_active_pct": round(float((Xk[:cut][mask, stress_i] > 200_000).mean()), 3) if stress_i is not None else None,
            "downtrend_pct": round(float((Xk[:cut][mask, ret1h_i] < -20).mean()), 3) if ret1h_i is not None else None}
    R["taxonomy_overlap"] = overlap
    R["unknown_rate"] = 0.0   # kmeans tum ornekleri atar; HMM'de unknown yok (rapor geregi alan)

    # (6) OUTCOME degerlendirmesi — freeze SONRASI, untouched validation'da
    R["outcome_eval"] = _outcome_eval(ts[cut:], lab_v, best_k)

    # kayitlar
    outcome = "SUPPORTS" if accepted else "WEAKENS"
    reg.attach_evidence(EvidenceBundle("EV-LATENT6A-001", spec.experiment_id,
                                       {k: v for k, v in R.items() if k != "outcome_eval"},
                                       outcome, evidence_family="latent6a",
                                       dataset_hash=meta["feature_version"], code_ref="ami/latent/discovery.py"), spec)
    if accepted:
        ko = KnowledgeObject(
            knowledge_id="K-LATENT-LS-001",
            claim=f"{best_k} latent state (LS-001..LS-{best_k:03d}) seed/perturbation-stabil ve "
                  f"kronolojik validasyonda tekrar goruluyor (occ_ratio {occ_ratio}, trans_corr {tc:.2f}). "
                  "Mekanizma iddiasi YOK; isimler notr.",
            claim_type=ClaimType.DESCRIPTIVE, status=KnowledgeStatus.HOLDOUT_VALIDATED,
            provenance=Provenance(source_tables=["mark_prices", "agg_trades", "liquidations", "book_ticker"],
                                  data_time_range=f"{datetime.fromtimestamp(int(ts[0])/1000, tz=timezone.utc):%Y-%m-%d}..{datetime.fromtimestamp(int(ts[-1])/1000, tz=timezone.utc):%Y-%m-%d}",
                                  code_ref="ami/latent/discovery.py", dataset_hash=ah,
                                  experiment_id=spec.experiment_id,
                                  execution_model="research_only_no_execution"),
            evidence_level=EvidenceLevel.UNTOUCHED_HOLDOUT, replications=1, holdouts=1,
            falsification=["forward doneminde occupancy/transition profili kaybolursa",
                           "farkli seed'lerde ARI<0.6'ya duserse"],
            confidence={"statistical": "MEDIUM", "mechanism": "NONE", "forward": "NONE"},
            permitted=[Permission.RESEARCH_ONLY, Permission.BACKTEST_ALLOWED, Permission.SHADOW_ALLOWED],
            forbidden=[Permission.LIVE_ALLOWED, Permission.SIZING_ALLOWED, Permission.PORTFOLIO_ALLOWED])
        store.put(ko, actor="latent6a")
        print(f"  Knowledge: {ko.knowledge_id} (max SHADOW_ALLOWED; LIVE/SIZING/PORTFOLIO YASAK)")
    else:
        store.archive_failure("Faz6A latent states (kmeans/hmm, 5m grid)",
                              FailureType.NO_EDGE,
                              reason=f"kabul kriteri saglanamadi: perturb_ari={R['perturb_ari']} val_ok={val_ok}",
                              data_period=str(R.get("n")), retry="farkli ozellik seti/resolution ile YENI prereg")
        print("  NO_STABLE_STATE / kriter disi -> failure archive")
    _write_report(R, spec, accepted)
    reg.close(); store.close()


def _outcome_eval(ts_v, lab_v, k):
    """Ayri katman: yalniz degerlendirme; mark index'ten forward 1h/6h."""
    import sqlite3
    from tools.research_s34_knowable_anchor_continuation import load_mark_index
    conn = sqlite3.connect(f"file:{ROOT/'data'/'microstructure.db'}?mode=ro", uri=True)
    m = load_mark_index(conn, "ETHUSDT")
    def fwd(t0, hz):
        a = m.at_or_before(int(t0)); b = m.at_or_before(int(t0) + hz)
        return (float(b[1]) - float(a[1])) / float(a[1]) * 1e4 if a and b and float(a[1]) > 0 else None
    out = {}
    base6 = [fwd(t, 6 * 3600_000) for t in ts_v[::12]]
    base6 = [x for x in base6 if x is not None]
    out["baseline_all"] = {"n": len(base6), "mean_fwd6h": round(float(np.mean(base6)), 1)}
    for j in range(k):
        idx = np.where(lab_v == j)[0][::12]   # saatlik seyreltme (overlap azalt)
        v6 = [fwd(ts_v[i], 6 * 3600_000) for i in idx]
        v6 = [x for x in v6 if x is not None]
        v1 = [fwd(ts_v[i], 3600_000) for i in idx]
        v1 = [x for x in v1 if x is not None]
        if v6:
            out[f"LS-{j+1:03d}"] = {"n": len(v6),
                                    "mean_fwd1h": round(float(np.mean(v1)), 1) if v1 else None,
                                    "mean_fwd6h": round(float(np.mean(v6)), 1),
                                    "wr6h": round(float(np.mean([x > 0 for x in v6])), 3)}
    conn.close()
    return out


def _finalize_null(store, reg, spec, R):
    reg.attach_evidence(EvidenceBundle("EV-LATENT6A-001", spec.experiment_id, R,
                                       "FALSIFIES", evidence_family="latent6a",
                                       dataset_hash="latent_ds_v1", code_ref="ami/latent/discovery.py"), spec)
    store.archive_failure("Faz6A latent states — NO_STABLE_STATE",
                          FailureType.NO_EDGE, reason=json.dumps(R.get("k_selection")),
                          retry="farkli resolution/ozellik seti ile YENI prereg")
    R["verdict"] = "NO_STABLE_STATE"
    _write_report(R, spec, False)
    print("  NO_STABLE_STATE — durust null, failure archive'a yazildi")


def _write_report(R, spec, accepted):
    OJ.write_text(json.dumps(R, indent=2, default=str), encoding="utf-8")
    lines = ["# AMI Faz 6A — Latent State Discovery", "",
             f"> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC — prereg `{spec.experiment_id}` "
             f"hash `{spec.frozen_hash}` (model oncesi frozen). SONUC: "
             f"**{'ACCEPTED — stabil latent state seti' if accepted else R.get('verdict', 'REJECTED')}**", ""]
    for key in ("features", "dropped_features", "changepoints", "k_selection", "chosen_k",
                "perturb_ari", "hmm_crosscheck_ari", "artifact_hash",
                "occupancy_expl", "occupancy_val", "occ_ratio_val", "avg_duration_min_expl",
                "transition_matrix_expl", "transition_matrix_val", "transition_entropy",
                "val_transition_corr", "state_profiles", "session_dist", "dq_dist",
                "taxonomy_overlap", "unknown_rate", "outcome_eval"):
        if key in R:
            lines.append(f"## {key}\n```json\n{json.dumps(R[key], indent=1, default=str)}\n```\n")
    lines += ["Durust statuler: software-correct ✓ · replay-validated (seed'li) ✓ · "
              f"latent-state {'stable' if accepted else 'unstable/null'} · "
              f"chronological-validation {'passed' if accepted else 'failed/na'} · "
              "alpha-incremental: outcome_eval bolumune bakiniz (ayri katman) · "
              "forward-validating ✗ · **operationally FORBIDDEN** (LIVE/SIZING/PORTFOLIO yasak)",
              "", "*Runner: `python -m ami.latent.discovery`*"]
    OM.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD: {OM}")


if __name__ == "__main__":
    main()
