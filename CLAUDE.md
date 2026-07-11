# Eclipse Scalper / AMI — Oturum Başlangıç Talimatı

> Bu dosya her yeni Claude oturumunda OTOMATİK yüklenir. Detay isteme sırası:
> önce `SYSTEM_STATE.md` (son bölümler en güncel), sonra göreve göre aşağıdaki haritadan.

## İlk yapılacak
1. `SYSTEM_STATE.md` oku — sistemin TEK master durum dosyası (her önemli değişiklikten sonra güncellenir; en yüksek bölüm numarası = en yeni durum).
2. AMI işi ise: `docs/ami/AMI_ROADMAP.md` + `docs/ami/AMI_CHANGELOG.md` oku; canonical spec `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md`.
3. Research işi ise: ilgili rapor `reports/research/s34/` altında; geçmiş sonuçların meta-DB'si `reports/research/s34/S34_ALL.db`.

## Kesin guardrail'lar (ihlal edilemez)
- `tools/s34_state_machine_live_executor.py`, `.env`, `execution/`, `risk/`, `brain/` — DOKUNMA (operatör sign-off şart).
- Leverage / ORDER_NOTIONAL / sizing — değişmez.
- Paralel Python/PowerShell prosesi ÇALIŞTIRMA (RAM çöker); araştırma scriptleri sırayla.
- pytest: en fazla 2 test dosyası/çağrı + `--basetemp` scratchpad'e + `-p no:cacheprovider` (repo tmp izin sorunu).
- Lookahead yasak; eşikler TRAIN'de seçilir, TEST raporlanır; FEE=5bps net; MC permütasyon standart.
- Kurulum/proje işleri D:\ sürücüsünde.
- Ana DB `data/microstructure.db` (650GB+) SALT-OKUNUR açılır (`file:...?mode=ro`).
- Mezarlığı tekrar test etme (buy-side fade, reversal, cross-asset transfer, gentleness, micro-timing, tight stop, partial exit, limit-entry) — tam liste: AMI failure archive + `docs/ami/AMI_RESEARCH_PROTOCOLS.md` §5.

## Kademeli bağımsız inceleme zinciri (ZORUNLU — sıkıştırılamaz)
Doğrulama gerektiren her önemli implementasyon/düzeltme için standart yaşam
döngüsü — **her zaman OTOMATİK uygulanır, sıkıştırılmaz**:

`implementation → bağımsız review → correction → bağımsız re-review → acceptance`

Kurallar:
- **Fazlar TEK TEK ve AYRI** yürütülür; aralarında **operatör sign-off** vardır.
  İki fazı asla aynı geçişte birleştirme (özellikle implementation + review'ı).
- **Neden:** review'ın epistemik değeri BAĞIMSIZLIĞINDADIR. Aynı aktör bir
  artefaktı tek geçişte hem üretip hem onaylarsa = kendi kodunu kendin
  onaylamak = değersiz. Bu yüzden her review **salt-okunur**, hiçbir dosyayı
  değiştirmez; düzeltmeler ayrı bir `correction` fazında yapılır.
- **Otomatik ama kapılı:** bu disiplini varsayılan olarak kendiliğinden uygula
  (her fazdan sonra bir sonraki fazın gerektiğini hatırlaman gerekmez); ama
  insan kapısını (operatör sign-off) kaldırma. "Otomatik" = disiplin standart,
  kapı değil.
- Zaman baskısı / "hızlı olsun" talebi bu zinciri sıkıştırmanın gerekçesi
  DEĞİLDİR. Sıkıştırılamaz, sıkıştırma.
- Her faz kendi verdict token'ıyla kapanır (ör.
  `..._CORRECTED_AWAITING_REREVIEW` → `..._ACCEPTED`) ve `SYSTEM_STATE.md`'ye
  kaydedilir. Emsal: liquidation-silence detector §107→§108→§109 zinciri.

## Dosya haritası
| Ne | Nerede |
|---|---|
| Master durum | `SYSTEM_STATE.md` |
| AMI spec (canonical) | `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md` |
| AMI docs (roadmap/changelog/şema/taksonomi/protokoller/kararlar) | `docs/ami/` |
| AMI kod | `ami/` — testler: `tests/test_ami_knowledge_governance.py`, `tests/test_ami_states_research.py` |
| AMI store'lar | `data/ami/knowledge.sqlite`, `data/ami/research.sqlite`, `data/ami/decisions.jsonl` |
| Research raporları (her test .md+.json) | `reports/research/s34/` |
| Research scriptleri | `tools/research_s34_*.py`, `tools/s34_mechanism_*.py` |
| Mekanizma feature store | `reports/research/s34/mechanism_store.sqlite` |
| Shadow runner + dashboard | `tools/s34_realtime_shadow_runner.py`, `tools/s34_live_chart.py` (:5050) |
| Proses yönetimi | `start_eclipse.ps1` / `stop_eclipse.ps1` / `status_eclipse.ps1` (live executor `-EnableLive` bayrağıyla — default KAPALI) |
| OI+spot poller | `data/oi_spot_poller.py` (60s, public endpoint) |
| Oturum sonuç raporları | repo kökü `S34_SESSION_SONUC_RAPORU_*.md` |

## Çalışma sözleşmesi
- Her önemli değişiklikten sonra `SYSTEM_STATE.md`'ye yeni bölüm ekle (numara artarak).
- AMI mimari kararı → `docs/ami/AMI_DECISION_RECORDS/DR-XXXX.md` + `AMI_CHANGELOG.md`
  **+ whitepaper'ın Appendix H'sine PATCH-XXXX kaydı** (Appendix G yaml formatı) —
  whitepaper yaşayan doküman, değişiklikler ORAYA DA işlenir (operatör tercihi, 2026-07-02).
- Her research testi → `reports/research/s34/` altına .md+.json rapor; bulgular Knowledge Object
  disiplinine uygun (kanıt seviyesi/kapsam/çürütme koşulu belirtilir); reddedilenler mezarlığa.
- Kod değişikliği sonrası `py_compile` + varsa `--once` dry-run; shadow/dashboard restart edilirse
  PID'ler doğrulanır. Sandbox'tan başlatılan prosesler kalıcı olmayabilir — kalıcılık için
  operatör sandbox dışında `start_eclipse.ps1` çalıştırır.
