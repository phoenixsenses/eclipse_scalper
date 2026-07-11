# CURRENT_STATE_VS_TARGET_GAP_ANALYSIS

**Tarih:** 2026-07-03 · Fable 5 · Statü sözlüğü: master protokol §4

| # | Hedef bileşen (kaynak) | Mevcut karşılık | Statü |
|---|---|---|---|
| 1 | Epistemic governor + KO + failure archive (WP Vol V) | `ami/governance`, `ami/knowledge` — 119 test | **EXISTS_VALIDATED** |
| 2 | Research OS: prereg-freeze, marketplace (WP Part X) | `ami/research/registry.py`, `marketplace.py` | **EXISTS_VALIDATED** |
| 3 | Forward evidence pipeline (dar kapsam) | `ami/research/forward_pipeline.py`, 2 binding VALID n=0/20 | **EXISTS_VALIDATED** |
| 4 | Multi-TF state engine (WP Part IV) | `ami/states/` | **EXISTS_VALIDATED** |
| 5 | Trade lifecycle + MFE classifier verisi (WP Part VIII) | `ami/lifecycle/engine.py` | **EXISTS_VALIDATED** |
| 6 | Latent/regime/drift research (WP Part XI) | `ami/latent/` — verdict'ler REJECTED/PASS-dar/FALSIFIES | **EXISTS_VALIDATED** (research-only; operationally forbidden) |
| 7 | Mutation/adversarial suite | `ami/mutation_suite.py` + 7 test dosyası (119 test) | **EXISTS_VALIDATED** |
| 8 | Canonical SQL warehouse (Protocol §7) | yok | **MISSING** — Phase 1 |
| 9 | Artifact registry + lineage + supersession (Protocol §4-5) | yok (dosya sistemi + SYSTEM_STATE anlatısı) | **MISSING** — Phase 1 |
| 10 | Numeric question registry Q001–Q1058 (Protocol §16, WP §69, CN §23) | 14 slug-ID soru (`research.sqlite`); matrix CSV bu audit'te üretildi | **PARTIALLY_IMPLEMENTED** — Phase 1 (CONFLICT-001) |
| 11 | Contamination ledger + researcher-exposure ledger (WP §70) | yok (disiplin manuel) | **MISSING** — Phase 2 |
| 12 | Multiple-testing family registry (WP §70.3) | deney bazında family_p var; registry yok | **PARTIALLY_IMPLEMENTED** — Phase 2 |
| 13 | Timestamp/known-at kontratı (Obs §6) | araştırma scriptlerinde ad-hoc known-at disiplini; merkezi kontrat yok | **PARTIALLY_IMPLEMENTED** — Phase 2 |
| 14 | Event/cycle identity + structural cycle resolver (Protocol §8, Obs §5) | 6h-gap proxy scriptlerde geçici; canonical yok | **MISSING** — Phase 3 |
| 15 | Immutable forward event master + snapshot scheduler (Obs §7-8) | yok | **MISSING** — Phase 8 (şema hazırlığı Phase 3) |
| 16 | Position master + trade path ledger (Obs §12-13) | shadow jsonl ledger'ları var (kaynak); canonical path ledger yok | **PARTIALLY_IMPLEMENTED** — Phase 3/8 |
| 17 | Timing labels/metrics (Obs §14-15) | araştırma raporlarında ad-hoc | **PARTIALLY_IMPLEMENTED** — Phase 7 |
| 18 | Chart-native objeler: Candle/Swing/Level/Pattern/Setup (CN §4-14) | yok | **MISSING** — Phase 4 |
| 19 | Shared feature engines (tek hesap, çift önleme) | feature'lar script-bazında dağınık | **CONFLICTING_IMPLEMENTATION riski** — Phase 5 konsolidasyon |
| 20 | Orderless observer engine (Obs §11/18/19, CN §20) | forward_pipeline dar örneği var | **PARTIALLY_IMPLEMENTED** — Phase 8 |
| 21 | Canonical dashboard 16 sayfa + API (Obs §26-27) | s34_live_chart (:5050) + navigation dashboard (farklı kapsam, read-only) | **PARTIALLY_IMPLEMENTED** — Phase 9 |
| 22 | Excel/Word rejenerasyonu (Protocol §13-14) | yok | **MISSING** — Phase 9 (CONFLICT-007 deferred) |
| 23 | Collector'lar: liq/bookticker/OI-spot/event-diary | canlı, healthy | **EXISTS_VALIDATED** |
| 24 | funding/OI history sürekliliği | DB'ler Mayıs'tan beri bayat | **BROKEN veya SUPERSEDED** — Phase 2 denetimi (OD-006) |
| 25 | OI/basis path collector (Protocol §17.4) | snapshot-only poller var | **PARTIALLY_IMPLEMENTED** — Phase 2/8, OD-006 |
| 26 | All-timestamp candidate universe (Protocol §17.8) | yok | **MISSING** — Phase 6 önkoşulu |
| 27 | Prediction readiness / World Model (Obs §24, WP Part XI) | yok (Faz 6B bilinçli kapalı) | **BLOCKED** — Phase 10, Opus REVIEW E |
| 28 | bd_first_buy50 exit observer | öneri operatör onayı bekliyor (SYSTEM_STATE §39) | **BLOCKED** — OD-004 |

**Özet:** Doğrulanmış epistemik çekirdek (1-7) güçlü ve KORUNACAK. En erken eksik dependency: **canonical warehouse + artifact/question registry (8-10)** → FIRST_SAFE_IMPLEMENTATION_BATCH.md.
