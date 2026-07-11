# WORKSTREAM_AND_EPIC_MAP

**Tarih:** 2026-07-03. Paralellik şartı: aynı warehouse/identity/timestamp-kontrat/vocabulary/safety; prosesler SIRAYLA (RAM).

| Workstream | Epic'ler | Phase | Paralel? |
|---|---|---|---|
| WS-A Canonical truth | E-A1 warehouse iskeleti · E-A2 artifact ingest · E-A3 question registry · E-A4 registry görünümleri | 1 | E-A2 ∥ E-A3 (A1 sonrası) |
| WS-B Evidence integrity | E-B1 contamination+exposure ledger · E-B2 known-at kontratı · E-B3 veri-kapsama denetimi | 2 | E-B2, WS-A ile paralel OLABİLİR |
| WS-C Identity & paths | E-C1 event identity+cycle resolver · E-C2 path ingest · E-C3 split/purge yardımcıları | 3 | Hayır (kritik yol) |
| WS-D Chart objects | E-D1 candle+swing · E-D2 level+push · E-D3 feature dictionary | 4-5 | Opus A sonrası; kendi içinde sıralı |
| WS-E Historical research | W1–W12 dalgaları (HISTORICAL_RESEARCH_WAVES.md) | 6-7 | Dalga içi tek koşum |
| WS-F Forward observatory | E-F1 event master+scheduler şemaları · E-F2 observer engine · E-F3 aktivasyon paketleri (OD'li) | 8 | Şema hazırlığı WS-E ile örtüşebilir; aktivasyon Opus D + OD sonrası |
| WS-G Dashboard/API | E-G1 API kontratı · E-G2 sayfalar · E-G3 Excel/Word rejenerasyonu | 9 | Spec işi erken; implementasyon downstream |
| WS-H Readiness | E-H1 readiness diagnostics · E-H2 OOD/kalibrasyon önkoşulları | 10 | Opus E kapısı |
