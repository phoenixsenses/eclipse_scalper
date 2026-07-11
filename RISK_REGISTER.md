# RISK_REGISTER

**Tarih:** 2026-07-03 · Kaynak: Reconciliation Report (15+1) + Phase 0 audit eklemeleri. Her risk → ilk kontrol.

| ID | Risk | İlk kontrol |
|---|---|---|
| R-01 | Scope explosion (3 spec birden eşzamanlı okunursa) | Precedence + faz kilidi (MASTER_ROADMAP); chart-native/ML sonraya |
| R-02 | Dependency inversion (dashboard/observer önce) | P1-3 bitmeden P4+ yok; Opus A kapısı |
| R-03 | Bucket fishing / multiple testing | P2 family registry + contamination ledger önce |
| R-04 | Holdout contamination / researcher exposure | P2 exposure ledger; hipotez-doğum kaydı |
| R-05 | Event N ↔ cycle N karışması | P3 identity zorunlu; aggregate'lerde çift sayaç |
| R-06 | Lookahead / known-at hataları | P2 merkezi kontrat; known_at<=trigger testi her engine'de |
| R-07 | Historical/replay/shadow/forward/live karışımı | Protocol §9 vocabulary tek kaynak; şemalarda record_type zorunlu |
| R-08 | Observer→order sızıntısı | Orderless sınıfı; order-API import'u mutation testiyle yasak |
| R-09 | Real-liq vs proxy karışımı | event_family + source_quality alanları (P3 DoD) |
| R-10 | Chart-object tanım kayması | Definition versioning (P4 DoD); freeze yalnız descriptive+control sonrası |
| R-11 | Dashboard'un truth kaynağına dönüşmesi | Downstream-only kuralı (CONFLICT-006); dashboard-to-SQL testi |
| R-12 | Software yeşili = alpha sanılması | Çift verdict alanı her raporda |
| R-13 | Storage/compute patlaması (snapshot/path) | STORAGE_COMPUTE_CAPACITY_PLAN limitleri; retention sınıfları P8 öncesi |
| R-14 | Premature microservice | Repo-first modüler; servis ayrımı yalnız arayüz kanıtlanınca |
| R-15 | Kompleks route'un basit baseline'ı geçememesi | WAIT/NO_TRADE/T45/simple-trend benchmark her ailede zorunlu |
| R-16 | **Versiyon kontrol boşluğu**: tüm canonical belgeler + ami/ untracked; checkpoint git'te değil | OD-010: research-only commit önerisi |
| R-17 | RAM kısıtı: paralel proses/pytest çökmesi | Sıralı çalıştırma; ≤2 test dosyası; batch'lerde tek koşum |
| R-18 | Kaza ile live/shadow mutasyonu ("sadece araştırma yaparken") | UNTOUCHED manifest + batch-sonu diff komutu (Phase 0'da kuruldu) |
