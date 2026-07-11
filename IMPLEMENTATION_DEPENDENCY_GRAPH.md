# IMPLEMENTATION_DEPENDENCY_GRAPH

**Tarih:** 2026-07-03 · Ok yönü: bağımlılık (A → B: B, A'yı gerektirir)

```
[P0 audit ✅]
   → [P1.B1 warehouse iskeleti]
        → [P1.B2 artifact ingest]
        → [P1.B3 question registry seed]  (CSV hazır ✅)
        → [P1.B4 registry görünümleri]
   → [P2 contamination/exposure/multiple-testing ledger]   (P1.B1 şart)
   → [P2 known-at kontrat modülü]                          (bağımsız modül; P1'e paralel OLABİLİR*)
        → [P3 event identity + cycle resolver]             (P2 kontrat + P1 warehouse şart; OD-003 onayı)
             → [P3 paths + split/purge yardımcıları]
                  → [P4 candle/swing/level/push]           (closed-candle + known-at şart)
                       → [P5 feature dictionary + konsolidasyon]
                            → [P6 historical waves]        (P2 ledger + P3 identity + P5 engine şart)
                                 → [P7 lifecycle/timing research]
                  → [P8 forward observatory]               (P3 identity + P2 kontrat şart; observer aktivasyonu OD)
                       → [P9 dashboard/API/Excel]          (yalnız canonical SQL'den)
                            → [P10 readiness diagnostics]  (OPUS E kapısı)
```
\* Paralellik sınırı: aynı warehouse/identity/vocabulary; prosesler sırayla (RAM).

## Kritik yol
P1.B1 → P2 ledger → P3 identity → (P6 research ∥ P8 observatory-şema). En uzun bekleme forward örneklem birikimi (P8 sonrası takvim süresi) — bu yüzden P8 şema/aktivasyon-öncesi hazırlığı P4-7 ile örtüşecek şekilde erkene çekilebilir ama AKTİVASYON operatör onayı + Opus D sonrası.

## Bloklayıcılar
- OD-001 (Q-metinleri) → P1.B3'ün text alanları; family-level seed bloklanmaz.
- OD-003 (cycle definition versiyonu) → P3 başlangıcı.
- OD-006 (funding/OI collector) → bazı P6 aileleri BLOCKED_BY_DATA kalır.
- Opus REVIEW A geçmeden P4+ başlamaz.
