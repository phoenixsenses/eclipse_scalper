# TEST_AND_MUTATION_STRATEGY

**Tarih:** 2026-07-03. Taban: mevcut 119/119 AMI testi (7 dosya) — bunlar KORUNUR ve genişletilir. Koşum kuralı: pytest ≤2 dosya/çağrı + `--basetemp` scratchpad + `-p no:cacheprovider`; sıralı.

## Katmanlar (kaynak: WP App I + Obs §34 + CN §28 + Protocol §23)
| Katman | Kapsam | Ne zaman |
|---|---|---|
| Unit/integration | Her yeni modül | Her batch |
| Schema/migration | init idempotency, round-trip, RO-open, version kaydı | Şema değişen batch |
| Timestamp/lookahead | known_at<=trigger; partial-candle reddi; swing-pivot known-at | P2+ her engine |
| Idempotency/restart/duplicate | ingest ve observer yazımları | P1.B2+, P8 |
| Rollback | new-files-only doğrulaması; protected diff boş | Her batch |
| Mutation (adversarial) | mevcut mutation_suite deseni: her yeni katmana ihlal-enjeksiyon testleri (ör. observer→order import'u, replay→forward etiketi, proxy→real karışımı, contamination atlaması) | Faz kapıları öncesi |
| Dashboard-to-SQL consistency | sayfa değeri == canonical sorgu | P9 |
| Scientific-protocol/negative-control | §9 kontrol setinin varlığı runtime'da assert | P6-7 her dalga |
| Reproducibility | hash'li artifact yeniden üretimi | Faz kapıları |
| Performance/capacity | ingest süresi, DB büyüme, RAM tavanı | P3, P8 öncesi |

Çift verdict zorunlu: `software_verdict` ayrı, `scientific_verdict` ayrı — hiçbir rapor tek verdict'le yayınlanmaz.
