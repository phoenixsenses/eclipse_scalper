# AMI Changelog

## 0.1.0 — 2026-07-02 (Faz 0-5 temeli)

```yaml
change_id: AMI-CHG-0001
date: 2026-07-02
author: claude (operator onaylı build talebi)
section_changed: tüm temel katmanlar (ilk implementasyon)
reason: whitepaper v0.2 Appendix F build brief
new_evidence: AMI_PHASE_VALIDATION.md (gerçek veri 10/10 Definition-of-Done kontrolü)
affected_knowledge: 9 S34 Knowledge Object tohumlandı, 12 mezarlık kaydı
status_change: yok (hiçbir bilgi terfi ettirilmedi)
implementation_change: ami/ paketi (12 modül), 2 test dosyası (17 test), data/ami/ store'ları
validation_required: forward — E-MECHCOMP-FWD-001 kayıtlı/donduruldu
```

Teslim edilenler:
- `ami/` paketi: constitution, enums, knowledge (objects+store), governance (governor),
  states (objects+engine+structure), lifecycle, research (registry+marketplace), decision, seed, runner
- Testler: `tests/test_ami_knowledge_governance.py` (10), `tests/test_ami_states_research.py` (7) — hepsi ✓
- Store'lar: `data/ami/knowledge.sqlite`, `data/ami/research.sqlite`, `data/ami/decisions.jsonl`
- Docs: gap analysis, roadmap, knowledge schema, state taxonomy, research protocols, DR-0001
- Doğrulama: `python -m ami.run_phase_checks` → `reports/research/s34/AMI_PHASE_VALIDATION.md`

Bilinçli sınırlar: Faz 6-9 (ML latent, world model, autonomous scientist, cross-market) iskelet DEĞİL,
hiç başlanmadı — roadmap'te. Governor kalibrasyon motoru Faz 6 ile gelecek.

## 0.2.0 — 2026-07-02 (Paket 1-3: forward pipeline + mutation suite + ilk deney)

```yaml
change_id: AMI-CHG-0002
date: 2026-07-02
author: claude (operatör paketi)
section_changed: research (forward_pipeline), mutation_suite, ilk alpha deneyi
reason: operatör Paket 1-3 talimatı + faz kapısı
new_evidence: AMI_FORWARD_EVIDENCE.md, AMI_MUTATION_REPORT.md (20/20), AMI_MFE50_EXPERIMENT.md (FALSIFIES)
affected_knowledge: >
  K-S34-HOUR17-001 ve K-S34-MECH-COMPOSITE-001 forward binding'e bağlandı
  (E-HOUR17-FWD-001, E-CONVCOMP-FWD-001, frozen). MFE50 ayrıştırma hipotezi
  failure archive'a NO_EDGE olarak eklendi (retry koşuluyla).
status_change: hiçbir bilgi terfi etmedi; hiçbir operasyonel izin verilmedi
implementation_change: >
  ami/research/forward_pipeline.py (R1-R6 kuralları), registry'ye forward_bindings
  + processed_trades + assert_no_overlap; ami/mutation_suite.py (20 senaryo) +
  tests/test_ami_mutation_suite.py + ami/run_mutation_report.py;
  KnowledgeStore WAL+busy_timeout; ami/run_forward_pipeline.py;
  tools/research_ami_mfe50_experiment.py (prereg E-MFE50-001).
validation_required: forward birikim (iki binding n=0'dan sayıyor, min_sample=20)
```

**Migration notu:** research.sqlite'a 2 yeni tablo (forward_bindings, processed_trades) —
CREATE IF NOT EXISTS, geriye dönük uyumlu. knowledge.sqlite WAL moduna geçti (otomatik).
**Rollback:** yeni modülleri sil + research.sqlite'taki 2 tabloyu DROP et; başka bağımlılık yok.

## 0.3.0 — 2026-07-02 (Faz 6A: Latent State Discovery — dürüst REJECTED)

```yaml
change_id: AMI-CHG-0003
date: 2026-07-02
author: claude (operatör Faz 6A paketi)
section_changed: ami/latent/* (yeni), shadow runner mech logging, Faz 6A mutation suite
reason: Faz 6A — research-only latent state discovery
new_evidence: >
  AMI_PHASE6A_LATENT.md (E-LATENT6A-001, prereg hash a059e89d80175704, artifact 4df1bf45d6dd2bbc):
  k=4 latent state exploration'da seed-ARI 0.851 / perturb-ARI 0.991 STABIL; kronolojik
  validasyonda occupancy [0.14x..4.99x] frozen bandı aştı -> REJECTED (rejim kayması).
  15/15 Faz 6A mutation testi geçti. Forward pipeline durum: 2 binding VALID, n=0 birikimde.
affected_knowledge: failure archive += Faz6A NO_STABLE_STATE (retry: daha uzun veri, YENİ prereg)
status_change: hiçbir bilgi terfi etmedi; latent çıktılara LIVE/SIZING/PORTFOLIO YASAK (yapısal)
implementation_change: >
  ami/latent/{dataset,models,discovery}.py (23,635 örnek 5m grid, outcome'suz;
  CUSUM+kmeans+HMM saf numpy); tests/test_ami_latent_mutations.py (15);
  shadow runner mech_v1 forward-only loglama (schema/feature version, provenance,
  missing-policy; geriye dönük evidence YASAK).
validation_required: mech_score forward birikimi; latent retry ancak yeni prereg ile
```

**Migration:** yeni dosyalar + data/ami/latent_* artifact'leri; mevcut şemalara dokunulmadı.
**Rollback:** ami/latent/ + latent_* dosyalarını sil; mech logging shadow-runner commit'ini geri al.

## 0.4.0 — 2026-07-03 (Faz 6A-R: PASS — rejim-koşullu, dar kapsam)

```yaml
change_id: AMI-CHG-0004
date: 2026-07-03
author: claude (operatör Faz 6A-R paketi)
section_changed: ami/latent/regime.py + drift_monitor.py (yeni), 14 mutation
reason: 6A kırılmasının nedeni + rejim-koşullu stabilite testi
new_evidence: >
  AMI_PHASE6AR_REGIME.md (E-LATENT6AR-001 hash 1b6d0b2b): drift=MARKET_SHIFT
  (dq elendi); trend=UP walk-forward PERSISTENT (3/4 band); alpha NON-incremental
  (regime+latent PF 1.41 mdd -416 ama top3-removed -458 = top-winner bağımlı;
  değer = market-description + risk). AMI_DRIFT_MONITOR.md: güncel durum UNUSABLE
  (öneriler governor'a). 14/14 mutation (toplam 66/66).
affected_knowledge: K-LATENT-REGIME-001 (HOLDOUT_VALIDATED, max SHADOW, trend=UP scoped)
status_change: hiçbir operasyonel izin verilmedi; Faz 6B'ye GEÇİLMEDİ
implementation_change: RegimeDefiner (exploration-fit, leakage-guard'lı),
  transition_matrix_within (boundary guard), DriftMonitor (öneri-only, alarm susturulamaz)
validation_required: forward'da trend=UP occupancy bandı izlenmeli (drift monitor)
```
**Rollback:** ami/latent/regime.py + drift_monitor.py + test dosyasını sil; K-LATENT-REGIME-001 REJECT'e çek.

## 0.5.0 — 2026-07-03 (Faz 6A-R2: FALSIFIES / INSUFFICIENT_SAMPLE — dürüst null)

```yaml
change_id: AMI-CHG-0005
date: 2026-07-03
author: claude (operatör Faz 6A-R2 paketi)
section_changed: ami/latent/risk_applicability.py (yeni), 13 mutation
reason: regime+latent katmanının risk/applicability değeri mi, frekans artefaktı mı?
new_evidence: >
  AMI_PHASE6AR2_RISK.md (E-RISKAPP-6AR2-001 hash db07a737): 328 no-overlap 6h LONG
  grid trade; aday veto fold 1-3'te SEÇİM ÜRETEMEDİ (n_cand=1/2/1 — calm-state kimliği
  refit'te kayıyor); tek değerlendirilebilir fold'da aday=regime-only (latent katkı 0),
  matched/random-veto pct ~0.65<0.75, retention 0.47<0.90. 6A-R'nin N=14 mdd -416
  hücresi per-fold dürüst refit'le YENİDEN ÜRETİLEMEDİ (ALL-era-fit artefaktıydı).
  Alarm SATÜRE (13/13 UNUSABLE, fp 0.69) -> applicability-degenerate.
affected_knowledge: Failure Archive += INSUFFICIENT_SAMPLE (riskapp overlay);
  K-LATENT-REGIME-001 kapsamı DEĞİŞMEDİ (market-description; risk iddiası artık
  desteklenmiyor olarak not edildi)
status_change: hiçbir operasyonel izin verilmedi; Faz 6B/World Model KAPALI
implementation_change: 13 yapısal guard (frequency-norm, kontrol-kolları, exposure-norm,
  top-winner disclosure, retroaktif-alarm, fold-aggregation, artifact-usable/version,
  permission tavanı, small-N bootstrap)
validation_required: retry yalnız forward shadow >=6 ay birikince YENİ prereg ile
```
**Rollback:** ami/latent/risk_applicability.py + tests/test_ami_risk_mutations.py +
AMI_PHASE6AR2_RISK.* sil; failure-archive satırı tarihsel kayıt olarak kalır.

## 0.6.0 — 2026-07-03 (C-BUY-FADE Yapısal + 8A Re-Entry: FALSIFIES ×2, silence-info istisnası)

```yaml
change_id: AMI-CHG-0006
date: 2026-07-03
author: claude (operatör C-BUY-FADE + 8A paketi)
section_changed: tools/research_s34_buyfade_{structural,reentry}.py (yeni), 24 mutation
reason: BUY-fade route'unun yapısal yolu, timing, horizon, multi-TF bağlam, silence
  decompose, management ve çift-yönlü re-entry sorularının preregistered testi
new_evidence: >
  BUYFADE_STRUCTURAL.md (E-BUYFADE-STRUCT-001, 391 event): route ALL tarihsel
  NEGATİF (-9.5/-1.1/-10.7); SILENCE tek gerçek bilgi (+20/+30/+20, matched-control
  diff +54 p<5e-5) ama T+30m-bilinir -> giriş alfası DEĞİL; timing/genesis/management
  NON_INCREMENTAL; horizon tüm ufuklarda negatif (reclaim %62-93).
  BUYFADE_REENTRY.md (E-BUYFADE-REENTRY-001): S->S churn (tüm cooldown'lar negatif,
  random-timing geçilemedi); S->L/L->L/L->S INSUFFICIENT; H-RE-NULL doğrulandı.
affected_knowledge: K-BUYFADE-SILENCE-INFO-001 (HOLDOUT_VALIDATED, max SHADOW);
  failure archive += 5 kayıt (timing, genesis, management, S->S churn, flip/L-kolları)
status_change: hiçbir operasyonel izin verilmedi; shadow/live route OTOMATİK DEĞİŞTİRİLMEDİ;
  "ALL tarihsel negatif" bulgusu operatör kararına sunuldu
implementation_change: 15 yapısal guard (backward-only, completed-bar, T0-silence yasağı,
  causal event-high, train-only seçim, split-purge, executable-fill, tiny-cell,
  UNKNOWN!=neutral, causal-prefix, fee-per-entry, entry-merge, attempt-drop,
  flip-claim ayrımı, family-p) — gelecek route-yapısal çalışmalar için yeniden kullanılabilir
validation_required: silence-info forward izlemi (shadow buy_state zaten logluyor);
  BAD_TIMING-stop-sonrası-re-entry yeni prereg adayı
```
**Rollback:** iki script + test + BUYFADE_* raporları sil; registry/knowledge kayıtları kalır.

## 0.7.0 — 2026-07-03 (Silence-Conditional Exit Timing: REJECTED[econ] + T45_ROBUST)

```yaml
change_id: AMI-CHG-0007
date: 2026-07-03
author: claude (operatör silence-exit paketi)
section_changed: tools/research_s34_buyfade_silence_exit.py (yeni), 16 mutation
reason: silence T+30m'de bilinir hale gelince T+45 sabit çıkış erken mi/geç mi/optimal mi?
new_evidence: >
  BUYFADE_SILENCE_EXIT.md (E-BUYFADE-SILEXIT-001 hash bd7d1f63): kazanç T0→T30'da
  (+37/+32/+23 brüt; T30→45 +0.6/+3.0/+5.4); T45 ROBUST (hiçbir uzun fixed tutarlı
  geçmedi); en iyi aday bd_first_buy50 8/9 kontrol geçti ama val econ +1.37<3bps →
  REJECTED (kriter gevşetilmedi); aynı çıkış noisy'de de +31 → silence'a özgü DEĞİL;
  Senaryo B (T+30 entry) −16/−13 → silence giriş sinyali değil (3. bağımsız teyit);
  4h-DOWN+silence dar hücre INSUFFICIENT kaldı. Survivor-audit: pre-T30 SL 7/0/1 evrende.
affected_knowledge: failure archive += silence-exit overlay (NO_EDGE, econ);
  K-BUYFADE-SILENCE-INFO-001 kapsam notu güçlendi (yönetim bilgisi, giriş değil)
status_change: hiçbir izin verilmedi; shadow/live route OTOMATİK DEĞİŞTİRİLMEDİ
implementation_change: 7 survivor/lookahead guard'ı (universe, pre-T30-use, breakdown-causal,
  manage-closed, realized-only, fee-extension, route-mutation) — exit-timing deneyleri şablonu
validation_required: forward öneri = shadow'da observation-only bd_first_buy50 çıkış
  gözlemcisi (operatör onayı ile)
```
**Rollback:** script + test + BUYFADE_SILENCE_EXIT.* sil; kayıtlar tarihsel kalır.
