# DR-0001 — Faz 0-5 Temel Mimarisi Kararları (2026-07-02)

**Durum:** KABUL — operatör talebi ("tamamını inşa et"), whitepaper Appendix F brief.

## Kararlar ve gerekçeler

1. **Tek repo, modüler paket (`ami/`), mikroservis YOK.** Whitepaper §76 açıkça izin veriyor;
   mevcut RAM/proses kısıtlarıyla uyumlu.
2. **Store'lar SQLite (`data/ami/*.sqlite`) + JSONL trace.** Mevcut altyapı deseniyle tutarlı;
   yeni bağımlılık sıfır.
3. **Promotion gate'leri kod sabiti** (`PROMOTION_GATES`), config değil — "no agent may override"
   ilkesinin mimari karşılığı. Değişiklik = bu DR dizinine yeni kayıt + kod diff'i.
4. **KnowledgeObject.validate() Store.put() içinde zorunlu** — provenance'sız iddia depoya giremez.
5. **CONTRADICTS edge'i çift taraflı objeye yansır** ve LIVE/SIZING iznini otomatik keser —
   "no contradiction is ignored" için pasif kayıt yeterli değil.
6. **ExperimentSpec.freeze() hash'i kanıt eklemede doğrulanır** — post-hoc kriter değişikliği
   yapısal olarak imkânsız (yalnızca yasak değil).
7. **State taksonomisi v0 kural-bazlı ve bilinçli basit** — Faz 6 latent keşfin karşılaştırma
   baseline'ı. Kurallar `AMI_STATE_TAXONOMY.md`'de belgeli.
8. **Governor kalibrasyon motoru ertelendi (Faz 6)** — DecisionTrace olasılıkları şimdiden
   loglanıyor ki kalibrasyon geriye dönük hesaplanabilsin.
9. **Eski S34 raporları toplu migre EDİLMEDİ** — 9 çekirdek iddia elle, tam bağlamla taşındı.
   Toplu otomatik migrasyon kalitesiz Knowledge Object üretirdi (trust ≠ lookahead-kontrolü).
10. **AMI hiçbir mevcut çalışan sisteme bağlanmadı** (dashboard/executor/shadow) —
    önce forward kanıt akışı (roadmap #1), sonra entegrasyon.

## Reddedilen alternatifler
- Pydantic/attrs bağımlılığı (stdlib dataclass yeterli)
- Knowledge'ı YAML dosyaları olarak tutmak (audit + eşzamanlılık zayıf)
- Governor'ı dashboard'a gömmek (kontrol düzlemi bağımsız kalmalı)
