# BOOKTICKER_COLLECTOR_CANONICALIZATION_IMPLEMENTATION_PLAN_V1

**Tarih:** 2026-07-15 · **Yazan:** Claude Sonnet 5 (bounded implementation-planning actor; bu revizyon: bounded corrective-planning actor) · **Durum:** `BOOKTICKER_CANONICALIZATION_IMPLEMENTATION_PLAN_OPERATOR_ACCEPTED`

> **Bu yalnızca bir plandır. Hiçbir kod/test/migration/runtime değişikliği
> yapılmadı; aday import/execute EDİLMEDİ; canlı DB/proses/task'a
> DOKUNULMADI.** Bu paket, operatör-kabul-edilmiş
> `BOOKTICKER_COLLECTOR_CANONICALIZATION_PREREGISTRATION_V1.md` (Contract B,
> `BOOKTICKER_CANONICALIZATION_PREREGISTRATION_OPERATOR_ACCEPTED`) davranışını
> gelecekteki bounded implementasyona çevirir. **B-LOW-1 ve B-LOW-2 açık
> implementation-planning advisory'leri, bu corrective'te tasarım-seviyesinde
> daha da sıkılaştırılmıştır** (§7/§8) — implementasyon ve testleri bağımsız
> kabul edilene kadar açık advisory olarak KALIRLAR. Contract B'nin substantive
> maddeleri DEĞİŞTİRİLMEZ.
>
> **Bu belge, ilk taslağın genuinely bağımsız fresh-context Opus review'unda
> `BOOKTICKER_IMPLEMENTATION_PLAN_REREVIEW_REQUIRED` verdiktiyle döndüğü
> `BOOK-B1` (BLOCKER), `BOOK-H1` (HIGH), `BOOK-M1`/`BOOK-M2` (MEDIUM),
> `BOOK-L1`/`BOOK-L2` (LOW) bulgularına ve paylaşılan `SEQ-H1`/`SEQ-M1`
> bulgularına yanıt veren bir corrective'tir.** §0 düzeltme geçmişini
> kaydeder. Bu corrective KENDİ KENDİNİ KABUL ETMEZ.
>
> **Operatör kabulü (2026-07-15):** Bu belge, LAUNCH-H2 corrective'ini
> kapsayan ikinci-tur bağımsız rereview'da **byte-identical/dokunulmamış**
> doğrulandı (bu belgenin kendi BOOK-*/SEQ-* bulguları zaten önceki turda
> kapanmıştı) ve genel `BOTH_IMPLEMENTATION_PLANS_INDEPENDENT_REREVIEW_
> ACCEPTED` verdiktinin kapsamına dahil edildi. Operatör bu belgeyi diğer iki
> plan belgesiyle birlikte **2026-07-15'te kabul etmiştir** — sequencing
> belgesi §3'teki `PLAN_ACCEPTED` kapısına karşılık gelir; zincirin ilerideki
> `OPERATOR_ACCEPTED` (IMPLEMENTATION_ACCEPTED sonrası) durumu DEĞİLDİR. Kabul
> implementasyon, `INTEGRATION_BASE_ESTABLISHED`, migration oluşturma/
> rezervasyon, aday canonicalization/import/execute, runtime probe,
> deployment veya activation YETKİLENDİRMEZ; B-LOW-1/B-LOW-2 açık advisory
> olarak KALIR.

Kaynak temeli (salt-okunur inceleme): `data/bookticker_collector.py` **yalnız
değişmez kanıt** (SHA-256 `136438fd…4c79`, 12,159 B, 330 satır — import/execute
EDİLMEDİ, bu corrective turunda da DEĞİŞMEDİ), `scripts/collector_supervisor.py`
**@ `main` (`cdeb9009`)** (`acquire_supervisor_lock`, `msvcrt.locking`,
`logs/pids/collector_supervisor.lock` — bkz. §0.1, mevcut branch tip'inde
YOK), `tools/health_state.py` (`write_component_health(component, status,
root="logs/health")` — mevcut branch tip'inde MEVCUT, `root=` parametresi
doğrulandı), `MIGRATION_LOG.md` + `SCHEMA_AND_DATA_MIGRATION_MAP.md` (M-XXXX
ledger'ları, en yüksek kayıt M-0036, book_ticker için hiçbir numara rezerve
EDİLMEDİ), `ami/*/*_canonical_migration.py` (migration emsalleri).

**Kaynak-türetilmiş kritik bulgu (değişmedi):** `book_ticker` şemasını **hiçbir
tracked production modülü OLUŞTURMUYOR** — yalnız test fixture'ları ve
(untracked) aday. `data/microstructure_collector.py` `book_ticker`'a referans
vermiyor. Dolayısıyla Contract B §B4.2'nin 3. seçeneği ("mevcut canonical şema
sahibi") kaynakta yok; `NUMBERED_MIGRATION_OWNED` gelecekteki migration'ı
**ilk canonical sahip** yapar.

---

## 0. Düzeltme geçmişi (bu corrective)

Kontrol eden bağımsız review: `BOOKTICKER_IMPLEMENTATION_PLAN_REREVIEW_
REQUIRED` (genuinely bağımsız, taze-context Opus reviewer; sıfır mutasyon;
ağaç review boyunca donmuş; aday hash `136438fd…4c79` bağımsız doğrulandı,
değişmedi).

| Bulgu | Şiddet | Prior metin | Bu corrective'te ne değişti | Etkilenen bölüm |
|---|---|---|---|---|
| BOOK-B1 | BLOCKER | Plan, `scripts/collector_supervisor.py`'nin `msvcrt.locking`/`acquire_supervisor_lock` modelini "canonical otorite" olarak dondururken, bu kod yalnız `main @ cdeb9009`'da var — mevcut branch tip'inde (HEAD `c6cf1451`) YOK (352 satırlık working-tree sürümünde hiç `msvcrt` yok) | Yeni §0.1: LAUNCH-B1 ile paylaşılan implementasyon-tabanı düzeltmesi; tüm `acquire_supervisor_lock`/`release_supervisor_lock` referansları `main @ cdeb9009`'a çıpalandı + gelecekteki integration SHA'sında yeniden doğrulama gereksinimi | §0.1, Kaynak temeli, §5.1 |
| BOOK-H1 | HIGH | §7 adım 10/§11 "mapped-drive/UNC ayrı kimlik kabul edilir, fail-closed basitlik" diyordu — bu aslında fail-OPEN'dı (iki farklı kimlik = iki eşzamanlı yazıcı, tek-yazıcı garantisini ihlal edebilir) | §7 düzeltildi: UNC ve mapped-network-drive path'leri **typed fail-closed REDDEDİLİR**; yalnız local absolute path desteklenir; production konfigürasyonu canonical local path'e zorlanır | §7, §6.1, §11/O-B1 |
| BOOK-M1 | MEDIUM | §8 karar tablosu "eşdeğer tanım, farklı isim → required-isim yoksa incompatible" diyordu — Contract B §B4.8'in "required indeksler eşdeğer var VEYA non-destructive oluşturulabilir" (tanım-tabanlı, isimden bağımsız) ifadesiyle ÇELİŞİYORDU | §8 karar tablosu düzeltildi: eşdeğer tanım, HERHANGİ bir isimle → KABUL (duplicate oluşturulmaz); yalnız aynı-isim+farklı-tanım → typed incompatible | §8, §6.4, §11/O-B2 |
| BOOK-M2 | MEDIUM | Health-output isolation seam'i yoktu; `write_component_health` default `root="logs/health"` ile test-hermeticity'yi tehlikeye atıyordu | Yeni §4a: `health_root` dependency-injection seam'i canonical collector'a eklenir; production default = mevcut `logs/health`; test = her zaman disposable tmp root | §4a, §6.5 |
| BOOK-L1 | LOW | O-B3 aday→canonical geçişi tanımlıyordu ama main-tree candidate'ın asla staged edilmeyeceğini ve byte-kopya-değil-yeniden-yazım ilkesini açıkça söylemiyordu | §2.2/§11 sıkılaştırıldı: main-tree candidate asla `git add` edilmez; canonical modül sözleşmeden yazılır, byte-kopya değil | §2.2, §6.6, §11/O-B3 |
| BOOK-L2 | LOW | §7 adım 8 `GetLongPathName`'i belirsiz "öneri" olarak bırakıyordu | §7 adım 8 düzeltildi: deterministik strateji dondurulur (realpath(parent)+normcase(leaf); DB oluşturulduktan sonra kimlik-değişmezliği doğrulanır; `GetLongPathNameW` yalnız post-existence VERIFICATION adımı, zorunlu normalizasyon DEĞİL) | §7 |
| SEQ-H1 (paylaşılan) | HIGH | Implementasyon tabanı belirsizdi | §0.1 ile kapatıldı — Launcher planıyla tutarlı | §0.1 |
| SEQ-M1 (paylaşılan) | MEDIUM | §12/Gate durumu "collector canonicalization migration'a bağımlı (şema-uyumluluk migration tarafından sağlanır)" diyerek kaynak-yazımını da migration'a bloklu gösteriyordu — sequencing belgesinin adım-3/adım-5-6 çelişkisiyle aynı kusur | §9/§12 düzeltildi: kaynak-yazımı (collector authoring + disposable-fixture unit test) migration'a BLOKLU DEĞİL, yalnız frozen şema sözleşmesine bağlı; migration yalnız integration/runtime-readiness'i bloklar | §9, §12 |

Bu corrective kendi kendini kabul ETMEZ. Bir sonraki adım genuinely bağımsız,
taze-context bir rereview'dır.

---

## 0.1 İmplementasyon tabanı — kritik düzeltme (BOOK-B1 / SEQ-H1)

**Bulgu (bağımsız review):** Bu planın ilk taslağı, `scripts/collector_
supervisor.py`'nin `acquire_supervisor_lock`/`release_supervisor_lock`
(`msvcrt.locking`) modelini **tek otoriter davranışsal kaynak** olarak
dondururken (Contract B §B5.1), bu fonksiyonlar yalnız `main` (`cdeb9009`)
üzerindeki `collector_supervisor.py`'de var. Mevcut branch tip'indeki
(`c6cf1451`, working-tree) `scripts/collector_supervisor.py` **352 satır ve
hiç `msvcrt`/lock kodu İÇERMİYOR.** Aynı `main`/HEAD ayrışması (merge-base
`74468c87`) burada da geçerli — bkz. Launcher planı §0.1, aynı kök-neden.

**Dondurulmuş karar:** Ne `c6cf1451` (yalnız governance, lock-kodu yok) ne de
`cdeb9009` (lock-kodu var, governance yok) tek başına implementasyon tabanı
olarak KULLANILAMAZ. Gelecekteki BookTicker implementasyonu, Launcher planı
§0.1'de dondurulan **aynı gelecekteki doğrulanmış integration SHA'sını**
gerektirir — bu SHA hem `main @ cdeb90096bfe7b448384b098094586cab750d5e6`'daki
`acquire_supervisor_lock`/`release_supervisor_lock` `msvcrt.locking` modelini
hem de `c6cf1451accf681c910e1a4e173560f55979fa38`'deki operatör-kabul-edilmiş
Contract A/B durumunu içermelidir. Bu corrective turu integration-base'i
seçmez, oluşturmaz veya yetkilendirmez — bkz. Launcher planı §0.1 (tam
gereksinim, integration-review kriterleri, yasak merge/cherry-pick/rebase
operasyonları oradaki tekil kaynak).

**Yeniden-çıpalanan kaynak referansları:** Bu belgedeki tüm
`acquire_supervisor_lock`/`release_supervisor_lock`/`msvcrt.locking`
referansları **yalnızca `main @ cdeb9009`'a** çıpalanmıştır ve implementasyon
başlamadan önce gelecekteki doğrulanmış integration SHA'sında **yeniden
doğrulanmalıdır** (dosya varlığı, fonksiyon imzaları, satır numaraları,
launcher testi `test_30_collector_supervisor_source_untouched`'ın hâlâ mevcut
olduğu).

---

## 1. Kapsam ve amaç
Contract B'nin dondurduğu davranışı zorlayan asgari canonical implementasyonu
tanımlar: tek-yazıcı OS lock (msvcrt), iki-saat invalid-message algoritması,
gerçekçi SQLite failure sözleşmesi, non-destructive migration (ayrı gate),
sanitize test izolasyonu, intake-otorite kuralı, canlı-proses sınırı.
Implementasyon, §0.1'de dondurulan gelecekteki doğrulanmış integration SHA'sı
ÜZERİNDE gerçekleşir — mevcut HEAD veya `main` üzerinde DEĞİL.

## 2. Hedef tracked modül ve aday→canonical geçiş

### 2.1 Hedef yol
Öneri: canonical tracked modül **`data/bookticker_collector.py`'nin canonical
sürümü aynı modül yolunda** olur (git add ile tracked hale gelir), AMA
implementasyon canlı dosyayı YERİNDE DEĞİŞTİRMEZ — Contract B §B7.1 ingestion
prosedürüne göre izole worktree'de üretilir, sonra canonical yol'a yerleşir.
**Deployment (canlı sürecin altını değiştirme) bu planın DIŞINDA.**

### 2.2 Aday→canonical geçiş (Contract B §B7.1, 9 adım — özet + BOOK-L1 sıkılaştırması)
hash-doğrula → tek immutable intake kopyası (production path DIŞINDA) →
intake-hash doğrula → provenance kanıtı olarak sakla → izole worktree'de tracked
modül oluştur → canlı path'e asla dokunma → tüm hash'leri raporla → hash
değiştiyse DUR. **Intake dosyası runtime kaynağı DEĞİLDİR; tracked modül tek
kaynak olur** (§B7.1 otorite kuralı, NB-6).

**Sıkılaştırılmış geçiş garantileri (BOOK-L1 düzeltmesi):**
- Mevcut main-tree'deki untracked aday (`data/bookticker_collector.py`, bu
  belgenin yazıldığı anda hash `136438fd…4c79`) **asla `git add`/staged/
  commit EDİLMEZ.** Live-tree candidate, yalnız kanıt olarak SHA-doğrulanır;
  hiçbir git işlemine tabi TUTULMAZ.
- Canonical modül, doğrulanmış intake kopyasından **byte-kopyalanıp sonra
  "canonical" ilan EDİLMEZ.** Contract B'nin klozları (§B3-B7, özellikle
  singleton/§B5, iki-saat pencere/§B4.6, gerçekçi SQLite/§B4.7) adaydan
  YAPISAL OLARAK EKSİK; canonical modül **sözleşmeden yazılır**, aday
  yalnızca hangi mevcut davranışın (reconnect/backoff/health/shutdown-flush,
  bkz. §2.3 tablosu) yeniden-kullanılabilir olduğunu gösteren kanıt olarak
  KULLANILIR.
- Canonical yazım işi yalnızca §0.1'de dondurulan gelecekteki doğrulanmış
  integration worktree'de gerçekleşir; bu worktree'de canonical hedef yol
  (`data/bookticker_collector.py`) implementasyon başlamadan önce ya YOK ya
  da açıkça karantinaya alınmıştır (integration-base main'den türediği için
  candidate orada zaten untracked/absent olacaktır — integration-review bunu
  doğrular).
- Test/import-resolution, testlerin **yeni tracked canonical modülü**
  yüklediğini kanıtlamalıdır — main-tree'deki untracked intake'i DEĞİL
  (§6.5/§6.6 testleri).
- Intake hash'i ve provenance kaydı korunur; hiçbir sonraki patch intake
  hash kaydını sessizce DEĞİŞTİRMEZ.

### 2.3 Uyumlu vs uyumsuz aday bölümleri (aday ≠ nihai doğruluk)
Aday, sözleşmeyle karşılaştırılır; yeniden-kullanılabilir ve uyumsuz bölümler:

| Aday bölümü (satır) | Sözleşme durumu | Aksiyon |
|---|---|---|
| `parse_bookticker` numerik red (`_float`, mid<=0) | KORUNUR ama davranış değişir (tick-local, reconnect değil) | Yeniden yaz: tick-local reddi (§B4.5) |
| ters-kitap (`ask<=bid`) reddi | **YOK** (aday kabul ediyor) | EKLE (tick-local, §B4.3) |
| stale/future timestamp reddi | **YOK** | EKLE (§B4.6 iki-saat) |
| `flush_ticks` 6-deneme lock retry | KORUNUR | Yeniden-kullan + gerçekçi termination (§B4.7) |
| singleton/lock | **YOK** | EKLE (msvcrt, §B5) |
| health_state entegrasyonu | KORUNUR | Yeniden-kullan + genişletilmiş sayaçlar + `health_root` seam (§4a) |
| SIGINT/SIGTERM graceful shutdown | KORUNUR | Yeniden-kullan + lock-release |
| reconnect/backoff/idle | KORUNUR | Yeniden-kullan |
| `--max-seconds` | KORUNUR (test-only işaretle) | Doküman notu |
| kimlik-bilgisi/order API | YOK (zaten) | Negatif testle kilitle |

## 3. Contract B madde → implementasyon izlenebilirliği (özet)

| Madde | Davranış | Yer |
|---|---|---|
| §B5.2 | lock kimliği = DB-path + tablo (sembol metadata) | yeni `bookticker_lock` helper |
| §B5.3 | msvcrt byte-range (`main @ cdeb9009`'daki `acquire_supervisor_lock` modeliyle hizalı, integration SHA'sında yeniden doğrulanacak), PID-liveness YOK | yeni `bookticker_lock` helper |
| §B4.6 | iki-saat + 10-adım invalid-window | canonical modül `collect()` |
| §B4.7 | gerçekçi SQLite failure | canonical modül `flush`/termination |
| §B4.8 | non-destructive migration + tanım-tabanlı indeks eşdeğerliği (§8) | ayrı migration (gate) + `schema_compat` helper |
| §B4.5 | tick-local vs reconnect | canonical modül parse/consume |
| §B7.1 | intake otorite + byte-kopya-değil-yeniden-yazım (§2.2) | süreç/kanıt (kod değil) |
| §B6.2 | sanitize test path'leri + `health_root` seam (§4a) | test paketi |

## 4. Helper ayrıştırması (en küçük tasarım)
Öneri — canonical modülden ayrı, test-edilebilir saf helper'lar:
- **`data/bookticker_lock.py`** (yeni) — msvcrt byte-range lock; kimlik türetimi
  (§7 B-LOW-1 normalizasyonu, UNC/mapped-drive fail-closed reddi dahil);
  acquire/release; typed `DuplicateInstanceError`; typed
  `UnsupportedPathError` (UNC/mapped-drive reddi için). Emsal: `main @
  cdeb9009:scripts/collector_supervisor.py` `acquire_supervisor_lock` (§0.1).
- **`data/bookticker_schema_compat.py`** (yeni) — salt-okunur şema/indeks
  uyumluluk doğrulaması (§8 B-LOW-2, tanım-tabanlı isim-bağımsız eşdeğerlik);
  `CREATE`/`ALTER` YOK.
- **`data/bookticker_collector.py`** (canonical) — WS lifecycle, batching,
  validation (§B4.5/§B4.6), flush (§B4.7), health (§4a `health_root` seam ile),
  lock-wiring, shutdown.
- Migration modülü **ayrı** (§9, migration gate).

### 4a. Health-output injection seam (BOOK-M2 düzeltmesi)

Önceki taslak `write_component_health("bookticker", …)`'ı adaydan aynen
yeniden-kullanıyordu, ama helper'ın default `root="logs/health"` parametresini
canonical collector'a hiç geçirmiyordu — bu, disposable test root'unun frozen
`<tmp>/runtime/health/bookticker.json` yoluna (Contract B §B6.2) ulaşmasını
engelliyordu (fragile cwd manipülasyonu olmadan).

**Dondurulmuş tasarım:**
- Canonical collector, bir `health_root` konfigürasyon parametresi kabul eder
  (CLI argümanı veya fonksiyon parametresi olarak) ve bunu doğrudan
  `write_component_health(..., root=health_root)`'a geçirir —
  `tools/health_state.py` bu parametreyi ZATEN destekliyor (satır 31),
  **değiştirilmesi GEREKMEZ**.
- **Production default:** mevcut repository konvansiyonu `logs/health`
  (helper'ın kendi default'u) — launcher tarafından açıkça override
  EDİLMEZ.
- **Test konfigürasyonu:** her zaman disposable `<tmp>/runtime/health/`
  kökü açıkça geçirilir; hiçbir test current-working-directory'e veya
  helper'ın default'una GÜVENMEZ; hiçbir test repository `logs/health`'e
  YAZMAZ.
- **Bileşen adı:** sabit `"bookticker"` (değişmez).
- **Payload beklentileri:** `status ∈ {"ok","degraded"}` (aday zaten bunu
  üretiyor — "failed" durumu process-fatal terminasyon anlamına gelir ve
  ayrıca hiçbir health payload'ı YAZILMAZ, çünkü proses zaten sonlanıyor).
- **Ardışık-hata sayacı:** health-yazma hatalarını sayan bir **wrapper**
  (canonical collector içinde, `write_component_health` çağrısını saran) —
  helper'ın kendisi bu sayacı TUTMAZ (saf I/O helper'dır). Wrapper:
  - başarılı yazım → ardışık-hata sayacı sıfırlanır;
  - 10. ardışık hata → typed process-fatal terminasyon (Contract B §B4.6
    dondurulmuş eşiği);
  - health-yazma hatası WebSocket'i reconnect ETMEZ, geçerli bir market
    event'ini ATMAZ;
  - health-hata handling'i kendisi rekürsif olarak yeni health-yazma
    denemesi TETİKLEMEZ (hata izole/yetkilendirilmiş logging path'ine
    gider, ör. `LOG.warning`, başka bir health-write DEĞİL).
- **Sıralama:** aktif-collector health durumu (status=ok, connected=true)
  yalnız OS lock ownership + şema-uyumluluk gate'leri geçtikten SONRA
  yayınlanır (§B5.3 sıralamasıyla tutarlı — health yayını da "DB
  mutasyonu/WS bağlantısından önce değil" invariant'ının bir parçasıdır).

**Testler (§6.5):** disposable health root kullanımı; production `logs/
health`'in hiç dokunulmadığının kanıtı (fs-izleme/mtime-karşılaştırma);
9 ardışık hata eşiği TETİKLEMEZ; 10. tetikler; başarı sayaç sıfırlar;
rekürsif hata-storm yok (hata handling'in kendisi health-write üretmediği
statik+davranışsal kanıt); ownership-önce-health sıralaması.

## 5. File-level change map (Workstream B)

| Yol | Mevcut rol | Önerilen değişiklik | Exec/Docs | Yeni/Değişen | Bağımlılık/gate | Risk | Doğrulama |
|---|---|---|---|---|---|---|---|
| `data/bookticker_collector.py` | Untracked aday (canlı PID 15332) | Canonical tracked sürüm (izole gelecekteki integration worktree'de sözleşmeden yazılır; candidate yalnız kanıt; canlı yerinde bırakılır) | Executable | Değişen (canonicalize) | intake (§B7.1) + lock/schema helper + frozen şema sözleşmesi (migration'a bloklu DEĞİL, bkz. §9) | **Yüksek** | §6 test paketi |
| `data/bookticker_lock.py` | — | msvcrt tek-yazıcı lock + kimlik (UNC/mapped-drive fail-closed reddi dahil) | Executable | **Yeni** | `main @ cdeb9009` `acquire_supervisor_lock` modeli (integration SHA'sında yeniden doğrulanacak) | **Yüksek** (concurrency) | lock testleri (§6.1) |
| `data/bookticker_schema_compat.py` | — | salt-okunur şema/indeks uyumluluk (tanım-tabanlı, isim-bağımsız) | Executable | **Yeni** | yok | Orta | schema testleri (§6.4) |
| `tests/test_bookticker_collector_canonical.py` | — | Contract B §B6 davranışsal testler + `health_root` seam testleri | Executable (test) | **Yeni** | canonical modül + helpers | Orta | pytest (izole tmp) |
| `tests/test_bookticker_singleton_lock.py` | — | cross-process exclusivity + B-LOW-1 kimlik testleri + UNC/mapped-drive reddi testleri | Executable (test) | **Yeni** | lock helper | Orta | pytest subprocess |
| (migration) | — | **AYRI gate — bu turda oluşturulmaz** | Executable | Yeni (gelecek) | migration preregistration | **Yüksek** | §9 |

**Açıkça DOKUNULMAYAN:** `scripts/collector_supervisor.py`, `data/microstructure_
collector.py`, canlı `data/microstructure.db`, PID 15332/20648, Scheduled Task'lar,
`tools/health_state.py` (yalnız çağrılır, mevcut `root=` parametresi üzerinden
kullanılır — değiştirilmez).

**Not (BOOK-B1 düzeltmesi):** "Bağımlılık/gate" sütunundaki `main`-tabanlı
referanslar yalnız `main @ cdeb9009`'a atıfta bulunur; implementasyon mevcut
HEAD üzerinde DOĞRUDAN çalışmaz — gelecekteki doğrulanmış integration SHA'sını
gerektirir (§0.1).

## 6. Test planı (Contract B §B6) — tip/izolasyon/failure-signal/fixture/mutasyon

### 6.1 Singleton/lock (unit + subprocess; izole tmp DB/lock)
- same DB/tablo + farklı sembol → conflict (kanıt: 2. instance typed non-zero).
- same DB/tablo + herhangi argüman varyasyonu → conflict.
- farklı disposable DB path → bağımsız.
- tablo adı kimliğe katılır (farklı tablo → farklı lock).
- sahtelenmiş/silinmiş diagnostic metadata OS lock'u bypass edemez.
- gerçek alt-proses crash → lock serbest (OS auto-release).
- PID-liveness API kullanılmaz (statik grep: `OpenProcess`/`psutil`/PID-poll yok
  + davranışsal).
- lock, DB mutasyonu VE WS bağlantısından VE aktif-health yayınından ÖNCE
  alınır (sıra testi, §4a ile genişletildi).
- gerçek alt-proses cross-process exclusivity.
- **B-LOW-1 kimlik testleri (BOOK-H1/BOOK-L2 düzeltmesiyle genişletildi):**
  1. desteklenen eşdeğer local yazımlar (relative↔absolute, case,
     trailing-sep, `.`/`..`) → AYNI lock kimliği;
  2. junction/symlink local alias (desteklenen durumlarda) → AYNI kimlik;
  3. farklı local DB'ler → FARKLI kimlik;
  4. **UNC path (`\\server\share\...`) → typed fail-closed reddi**
     (`UnsupportedPathError`), lock/DB/health/WS mutasyonu YOK;
  5. **mapped-network-drive path → typed fail-closed reddi**, aynı şekilde
     sıfır mutasyon;
  6. sembol-seti varyasyonları kimliği asla ETKİLEMEZ;
  7. DB henüz yoksa: parent-realpath+leaf-normcase kimliği; DB oluşturulduktan
     sonra resolve edilen kimlik pre-creation kimliğiyle AYNI kalmalı (aksi
     halde fail-closed, §7).

### 6.2 Validation/pencere (unit; mock local WS)
- tam 60000 ms stale KABUL; 60001 RET.
- tam 5000 ms future KABUL; 5001 RET.
- 5. geçersiz mesaj tetikler; sayılır ve atılır (insert yok).
- geçerli mesaj rolling pencereyi sıfırlamaz.
- 30.0s'i dolan ret timestamp'i pencereden düşer (aged-out).
- reconnect (başarılı/başarısız) pencereyi temizler.
- threshold-reconnect'te flush hatası → süreç sonlanır (reconnect değil).
- sıfır spread KABUL; negatif spread RET; ters-kitap RET (tick-local).
- eksik/pozitif-olmayan timestamp → tick-local ret (sayaca girer).
- monotonic saat kullanımı (wall-clock ile timestamp validasyonu ayrı).

### 6.3 SQLite failure (unit; gerçek sqlite3 lock/busy injection, disposable DB)
- başarılı commit → yalnız commit sonrası batch temizlenir.
- busy/locked → rollback + tam batch korunur + bounded retry.
- retry tükenmesi → batch korunur + typed non-zero termination (duplicate/partial
  satır YOK).
- non-lock `OperationalError`/`DatabaseError`/integrity → derhal typed termination
  (reconnect loop yok, kör replay yok).

### 6.4 Migration/schema-compat (unit; disposable DB + temsili fixture)
- temiz DB → tablo+indeks oluşturulur (migration testi, §9).
- uyumlu tablo byte/veri korunur.
- eksik required indeks non-destructive eklenir.
- uyumsuz tablo → hiçbir mutasyon olmadan fail-closed + mismatch raporu.
- **B-LOW-2 (BOOK-M1 düzeltmesiyle isim-bağımsız hale getirildi):**
  1. required indeksle **aynı isim, aynı tanım** → KABUL (dokunma);
  2. required indeksle **eşdeğer tanım, FARKLI isim** → KABUL (dokunma,
     duplicate OLUŞTURULMAZ) — önceki taslağın "required-isim yoksa
     incompatible" kararı KALDIRILDI, Contract B §B4.8 ile ÇELİŞİYORDU;
  3. **aynı isim, farklı tanım** → typed incompatible-schema failure
     (mutasyondan ÖNCE), hiçbir `CREATE INDEX IF NOT EXISTS` ile
     maskeleme YOK;
  4. hiçbir eşdeğer indeks yoksa → "eksik requirement" (migration
     non-destructive olarak canonical-isimle oluşturabilir; runtime
     collector OLUŞTURMAZ, yalnız migration);
  5. karşılaştırma-dışı indeks formları (expression-index/partial-predicate
     kombinasyonları parse edilemiyorsa) → typed fail-closed, "belirsiz"
     olarak KABUL EDİLMEZ;
  6. SQLite auto-index'ler (`sqlite_autoindex_*`, UNIQUE/PK'den otomatik
     üretilen) açıkça ayrı ele alınır, required-indeks karşılaştırmasına
     karıştırılmaz.
- şema eksik/uyumsuz → collector WS bağlantısından ÖNCE fail.
- migration yasak destructive SQL içermez (statik tarama).

### 6.5 Sağlık/negatif/izolasyon
- health payload doğru; sayaçlar reconnect'ler arası kalıcı.
- **`health_root` injection (BOOK-M2 düzeltmesi):** disposable root her
  zaman açıkça geçirilir; production `logs/health` hiç dokunulmaz (mtime/
  varlık kontrolü); 9 ardışık hata eşiği TETİKLEMEZ, 10. tetikler; başarı
  sayacı sıfırlar; hata-handling rekürsif health-write üretmez; ownership+
  şema-uyumluluk-önce-health sıralaması.
- kimlik-bilgisi/order API yok (statik + davranışsal).
- Scheduled Task mutasyonu yok.
- **sanitize ortam:** alt-proses production env değişkenlerini almaz; testler
  `data/microstructure.db`/production runtime dizinini çözemez; PID 15332 asla
  sorgulanmaz; intake asla runtime'da import edilmez.

### 6.6 Aday-otorite ve izolasyon testleri (BOOK-L1 düzeltmesi)
- main-tree'deki untracked candidate hiçbir git-add/staging işlemine tabi
  tutulmaz (statik: `git status --porcelain` candidate'ı `??` olarak
  gösterir, test-suite çalışmadan önce ve sonra).
- intake evidence, tracked canonical modül oluşturulduktan sonra hiçbir
  runtime import/exec kaynağı DEĞİLDİR.
- modül-resolution testleri, testlerin **canonical integration-worktree
  modülünü** yüklediğini kanıtlar — main-tree untracked candidate'ı DEĞİL
  (ör. `sys.modules['data.bookticker_collector'].__file__` beklenen
  integration-worktree yoluna eşit; main-tree candidate yoluna DEĞİL).
- candidate hash test-suite boyunca değişmeden kalır (`136438fd…4c79`).
- hiçbir test candidate'ı canonical modülün "sessiz byte-kopyası" olarak
  KABUL ETMEZ (statik diff: canonical modülün singleton/iki-saat/gerçekçi-
  SQLite implementasyonu candidate'ta YOK, dolayısıyla byte-eşitlik
  İMKANSIZ ve test bunu doğrular).

### 6.7 Zorunlu mutasyon-probeleri
- Sembol setini lock kimliğine geri ekle → conflict testleri (#1-2) FAIL.
- OS-lock otoritesi yerine PID-liveness koy → #6 FAIL.
- Lock'u ağ bağlantısından SONRA al → #7 FAIL.
- Invalid-window off-by-one → #10/#14 FAIL.
- Geçerli-mesaj sayaç reset → #12 FAIL.
- Destructive migration SQL → #24 FAIL.
- Retry tükenmesinde satır düşür → #27 FAIL.
- Public/production path escape → #28 FAIL.
- Intake evidence'ı runtime'da import et → §6.6 testi FAIL.
- Ters-kitap kabulünü aç → #18 FAIL; retry mantığını kaldır → #26 FAIL;
  shutdown-flush kaldır → #33 FAIL; idle-timeout kaldır → #34 FAIL;
  health-update kaldır → #35 FAIL; duplicate-instance fail-open → #8 FAIL.
- **UNC/mapped-drive reddini kaldır (aynı-kimlik kabul et) → §6.1 UNC/
  mapped-drive testleri FAIL (BOOK-H1 regresyon-kanıtı).**
- **aynı-isim/yanlış-tanım indeksi kabul et VEYA eşdeğer-farklı-isim indeksi
  incompatible say → §6.4 B-LOW-2 testleri FAIL (BOOK-M1 regresyon-kanıtı).**
- **`health_root`'u yok say, hardcoded `logs/health` kullan → §6.5 testi
  FAIL (BOOK-M2 regresyon-kanıtı, production path'e yazma tespit edilir).**
- main-tree candidate'ı staged/committed hale getir → §6.6 testi FAIL
  (BOOK-L1 regresyon-kanıtı).

Her materyal mutasyon en az bir anlamlı testi kırmalı; her probe sonrası ağaç
tam geri yüklenmeli.

**pytest guardrail:** çağrı başına ≤2 dosya, `--basetemp` scratchpad,
`-p no:cacheprovider`; paralel proses YOK; public Binance'e erişim testi FAIL.

## 7. B-LOW-1 tasarım kapanışı — Windows lock-path normalizasyonu (BOOK-H1/BOOK-L2 düzeltmesi)

**Dondurulmuş kimlik (değişmez):** `NORMALIZED_ABSOLUTE_LOCAL_DATABASE_PATH +
TABLE_NAME("book_ticker")`. Sembol seti kimliğe ASLA katılmaz.

**Dondurulmuş desteklenen-path politikası (BOOK-H1 düzeltmesi — önceki
"mapped-drive/UNC ayrı kimlik = fail-closed basitlik" iddiası YANLIŞTI,
KALDIRILDI):**

- **Yalnız local filesystem path'leri desteklenir** ilk canonical
  implementasyon için.
- **UNC path'ler (`\\server\share\...`) typed fail-closed KONFİGÜRASYON
  HATASIYLA REDDEDİLİR** — lock/DB/health/WS mutasyonu başlamadan.
- **Mapped network sürücüleri (ör. `Z:\` bir ağ paylaşımına bağlıysa) typed
  fail-closed KONFİGÜRASYON HATASIYLA REDDEDİLİR.**
- **Gerekçe:** mapped-drive ve UNC formlarının aynı fiziksel veritabanına
  farklı kimlikler türetmesi, iki eşzamanlı yazıcının aynı `book_ticker`
  tablosuna erişmesine izin VEREBİLİR — bu, Contract B §B5.2'nin "hiçbir iki
  desteklenen production instance'ı aynı tabloya eşzamanlı yazamaz"
  garantisini İHLAL EDER. "Ayrı kimlik" fail-closed DEĞİL, fail-OPEN'dır.
  Eşdeğerliği kanıtlanamadığı sürece, ret tek güvenli seçenektir.
- Production konfigürasyonu, canonical bir **local absolute path**'e
  ÇÖZÜMLENMEK ZORUNDADIR (mevcut `data/microstructure.db`, `D:\eclipse_
  scalper\data\microstructure.db` — zaten local, bu kısıtlama mevcut
  production yapılandırmasını ETKİLEMEZ).

**Önerilen kanonik normalizasyon algoritması (yalnız desteklenen local
path'ler için):**
1. `os.fspath` → str; boş/None → **fail-closed reject**.
2. Kullanıcı/ortam genişletme: `os.path.expanduser` + `os.path.expandvars`
   **YALNIZ açıkça izin verilirse**; production default = genişletme YOK
   (deterministik, kapalı).
3. Mutlak: `os.path.abspath` (cwd-relative çözümü).
4. **UNC/mapped-drive tespiti (yeni, BOOK-H1):** path `\\` ile başlıyorsa
   (UNC) → typed reject. Path bir sürücü harfiyle başlıyorsa, `GetDriveType`
   (ctypes, `kernel32.dll`) ile sürücü tipini sorgula; `DRIVE_REMOTE` (mapped
   network drive) dönerse → typed reject. `DRIVE_FIXED`/`DRIVE_REMOVABLE` →
   devam.
5. Fiziksel çözüm: `os.path.realpath` — junction/symlink resolution (Windows'ta
   `realpath` reparse-point'leri çözer); `.`/`..` normalize; ayraç normalize.
6. Windows case normalizasyonu: `os.path.normcase` (case-fold + `/`→`\`).
7. Drive-letter case: `normcase` ile normalize (C:↔c:).
8. Trailing separator: kaldır (kök hariç).
9. **8.3 kısa ad ve var-olmayan-leaf stratejisi (BOOK-L2 düzeltmesi —
   deterministik, artık belirsiz "öneri" DEĞİL):**
   - DB dosyası **mevcutsa**: `realpath` 8.3-kısa-adları ve junction/symlink'i
     uzun/fiziksel forma zaten çözer (adım 5); ek doğrulama olarak
     `GetLongPathNameW` (ctypes) **post-existence verification adımı** olarak
     çağrılır — realpath çıktısıyla eşleşmezse fail-closed reject (alias
     deterministik çözülemedi).
   - DB dosyası **henüz yoksa**: kimlik `realpath(parent_dir) + normcase(leaf_
     name)` olarak türetilir (parent zaten var olmalı; yoksa fail-closed
     reject — "erişilemeyen/çözülemeyen parent").
   - **Kimlik-değişmezlik doğrulaması:** DB dosyası ilk kez oluşturulduktan
     SONRA, koleksiyon süreci kimliği YENİDEN hesaplar (artık dosya mevcut,
     tam `realpath` uygulanabilir) ve bunu oluşturma-öncesi türetilen
     kimlikle KARŞILAŞTIRIR. Farklıysa → fail-closed (lock kimliği runtime
     sırasında SESSİZCE DEĞİŞTİRİLMEZ).
   - **`GetLongPathNameW` kullanım kararı (artık belirsiz DEĞİL):** yalnız
     mevcut-dosya post-existence doğrulama adımı olarak kullanılır; zorunlu
     ön-normalizasyon adımı DEĞİLDİR (realpath zaten birincil normalizasyon
     mekanizmasıdır).
10. **UNC/mapped-drive desteklenmez** (adım 4'te reddedildi — eski adım
    9-10'un "ayrı kimlik" önerisi KALDIRILDI).
11. Tablo adı kimliğe eklenir: `normcase(realpath(db)) + "\x00" + "book_ticker"`.
12. Lock-dosyası yeri: kimliğin **SHA-256 hash**'i → `runtime/locks/bookticker_
    <hash16>.lock` (collision-resistant, deterministik, path-length-safe,
    production konfigürasyonu bu yolu override EDEMEZ).

**Desteklenen path formları:** local absolute, local cwd-relative, `.`/`..`
içeren, farklı-case, local junction/symlink, trailing-sep, local 8.3 alias.
**Fail-closed reddedilen:** boş/None; çözülemeyen path; (default) env/user
genişletme gerektiren; **UNC path'ler**; **mapped network drive'lar**;
erişilemeyen/çözülemeyen parent dizini; oluşturma-sonrası kimlik-değişimi
(post-creation identity drift).

**Kanıt testleri (§6.1 B-LOW-1):** aynı fiziksel local DB'nin desteklenen tüm
yazımları AYNI lock kimliği türetir; farklı local DB'ler FARKLI kimlik; UNC
ve mapped-drive girişimleri REDDEDİLİR (sıfır mutasyon); sembol setleri
kimliği asla etkilemez.

## 8. B-LOW-2 tasarım kapanışı — indeks-tanımı eşdeğerliği (BOOK-M1 düzeltmesi)

**`CREATE INDEX IF NOT EXISTS` uyumluluk-doğrulaması yerine KULLANILMAZ.**
Uyumluluk, mutasyondan ÖNCE `data/bookticker_schema_compat.py` ile belirlenir.

**İndeks eşdeğerlik kriterleri (salt-okunur SQLite metadata):**
- hedef tablo (`sqlite_master.tbl_name`);
- sıralı indeksli kolonlar/expression'lar (`PRAGMA index_info` / `PRAGMA
  index_xinfo`);
- uniqueness (`PRAGMA index_list.unique` / `sqlite_master.sql`);
- collation (varsa, `index_xinfo`);
- sort order (ASC/DESC, `index_xinfo`);
- partial-index predicate (varsa, `sqlite_master.sql` `WHERE` cümlesi);
- expression-index semantiği (varsa, `sqlite_master.sql`).

**İndeks adı eşdeğerlik kriterlerine DAHİL DEĞİLDİR** — yalnız hangi
davranışın uygulanacağını (dokunma vs. yeni-isimle-oluştur) belirler, uyumluluk
kararını DEĞİL.

**Karar tablosu (düzeltilmiş — BOOK-M1, isim-bağımsız):**
| Durum | Sonuç |
|---|---|
| required indeksle **aynı isim, aynı/eşdeğer tanım** | KABUL (dokunma) |
| required indeksle **eşdeğer tanım, FARKLI isim** | **KABUL (dokunma; duplicate canonical-isimli indeks OLUŞTURULMAZ)** |
| **aynı isim, farklı (eşdeğer-olmayan) tanım** | **typed incompatible-schema failure, mutasyondan ÖNCE** |
| hiçbir eşdeğer indeks yok | eksik requirement — migration non-destructive olarak canonical-isimle oluşturabilir (yalnız migration, runtime collector DEĞİL) |
| karşılaştırma-dışı form (parse edilemeyen expression/partial) | typed fail-closed ("belirsiz" = uyumsuz kabul edilir, "uyumlu" DEĞİL) |
| SQLite auto-index (`sqlite_autoindex_*`) | required-indeks karşılaştırmasından AYRI ele alınır; PK/UNIQUE kaynaklı otomatik indeksler kendi başına "eşdeğer kullanıcı-indeksi" olarak SAYILMAZ |

**Önceki taslağın hatası:** "eşdeğer tanım, farklı isim → required-isim
yoksa incompatible" KALDIRILDI. Bu, Contract B §B4.8'in "required indeksler
eşdeğer var VEYA non-destructive oluşturulabilir" ifadesiyle doğrudan
çelişiyordu ve fonksiyonel olarak eksiksiz bir şemayı yanlışlıkla uyumsuz
ilan edebilir veya `CREATE INDEX IF NOT EXISTS` ile sessizce duplicate
indeks oluşturabilirdi.

**Kanıt testi (§6.4):**
1. `book_ticker`'da `idx_bt_symbol_ts` adıyla ama farklı kolon-tanımıyla
   (örn. yalnız `(ts_ms)`) bir indeks önceden varsa → migration typed
   incompatible ile fail-closed, hiçbir mutasyon YOK.
2. Aynı `book_ticker`'da `(symbol, ts_ms)` üzerinde ama `ix_custom_1` adıyla
   bir indeks önceden varsa → KABUL, `idx_bt_symbol_ts` adıyla duplicate
   OLUŞTURULMAZ.

## 9. Migration ayrımı (Contract B §B4.8) — AYRI GATE (SEQ-M1 düzeltmesi)

**Bu turda migration OLUŞTURULMAZ, numara ATANMAZ/rezerve EDİLMEZ, ledger
DÜZENLENMEZ, şema DEĞİŞTİRİLMEZ.** Aşağıda yalnız planlama:

**Sıralama netliği (SEQ-M1 düzeltmesi):** Önceki taslak, canonical collector
kaynak-yazımının migration'a "bağımlı/bloklu" olduğunu ima ederken aynı
zamanda sequencing belgesinin implementasyon sırasında kaynak-yazımını
migration'dan ÖNCE koymasıyla çelişiyordu. Bu artık ayrıştırılmıştır:

- **Migration'a BLOKLU DEĞİL (implementasyon-yetkilendirmesinden sonra
  hemen başlayabilir):** saf lock/path/schema-compat helper yazımı;
  canonical collector kaynak-yazımı (§B4.1'de dondurulan frozen şema
  sözleşmesine karşı, gerçek/canlı DB'ye dokunmadan); disposable-fixture
  unit testleri (§6.1-§6.7) — bunlar migration'ın VAROLMASINI gerektirmez,
  yalnız frozen DDL'yi (`CREATE TABLE`/`CREATE INDEX` metnini, §B4.1) test
  fixture'ı olarak kullanırlar.
- **Migration'a BLOKLU (ayrı, sonraki gate):** migration ile üretilen GERÇEK
  şemaya karşı **entegrasyon** testleri; `data-collection` launcher modunun
  **runtime-açılışı** (bkz. Launcher planı §4.3); deployment/runtime
  readiness; canlı veritabanı kullanımı.

Yani: **kaynak yazımı ve onun disposable-fixture unit testleri migration'ı
BEKLEMEZ; yalnız migration'ın gerçekten üretilmiş şemasına karşı entegrasyon
ve runtime-açılış migration kabul zincirini bekler.**

- **Önerilen şema sahipliği:** `NUMBERED_MIGRATION_OWNED` (Contract B §B4.2).
  Kaynak-doğrulandı: `book_ticker`'ın tracked production sahibi YOK → migration
  ilk canonical sahip olur.
- **Required tablo/indeks:** Contract B §B4.1 dondurulmuş şema (11 kolon + 2
  indeks).
- **Temiz DB:** tablo+indeks oluştur; ilgisiz tablolara dokunma.
- **Uyumlu tablo:** yalnız eksik required indeksi non-destructive ekle (§8
  tanım-tabanlı eşdeğerlik kuralına göre).
- **Uyumsuz tablo:** typed fail-closed + mismatch raporu (kolon/affinity/PK/indeks).
- **İzin verilen SQL:** `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT
  EXISTS`, eksik required indeks oluşturma — **ancak uyumluluk §8 ile ÖNCE
  belirlenir.**
- **Yasak SQL:** `DROP TABLE`, rename, recreate, copy-swap, satır silme/yazma,
  destructive kolon, implicit backfill, `VACUUM`, ilgisiz şema.
- **Migration-numarası preflight:** izole implementasyonda `MIGRATION_LOG.md` +
  `SCHEMA_AND_DATA_MIGRATION_MAP.md` (microstructure.db namespace, en yüksek
  mevcut kayıt M-0036, book_ticker için hiçbir numara rezerve EDİLMEDİ)
  incelenir, sonraki kullanılmamış M-XXXX seçilir, yazımdan önce kaydedilir;
  collision/gap/duplicate/eşzamanlı-main-ilerlemesi → abort. **Bu turda
  YAPILMAZ.**
- **Rollback beklentisi:** non-destructive olduğundan rollback = "no-op üzerinde
  idempotent"; oluşturulan indeks tek-yönlü ama yeniden-üretilebilir (emsal
  M-0007 idempotent ALTER deseni).
- **Production read-only:** implementasyon/review production şemasını yalnız
  salt-okunur inceler; migration'ı canlı `data/microstructure.db`'ye ASLA
  çalıştırmaz. Migration testleri disposable DB + temsili fixture.
- **Gelecek migration preregistration gereksinimi:** migration oluşturma, bu
  planning paketi bağımsız-kabul edildikten SONRA, AYRI açık yetkilendirme +
  kendi numaralı-migration preregistration'ı + bağımsız review gerektirir.

## 10. Runtime-safety ve deployment yasakları
- Aday (`data/bookticker_collector.py`, PID 15332) NON-CANONICAL, değişmez,
  import/execute EDİLMEZ. Bu corrective turunda da hash değişmedi
  (`136438fd…4c79`, açılış=kapanış).
- PID metadata'nın hiçbir lock otoritesi YOK (§B5.3); OS byte-range lock tek
  otorite.
- Hiçbir proses (15332/20648) sinyallenmez/durdurulmaz/başlatılmaz.
- Hiçbir Scheduled Task değiştirilmez.
- Canlı `data/microstructure.db` mutasyona uğratılmaz (testler disposable DB).
- Deployment/handoff/ikinci-instance/runtime-alignment kapsam DIŞI.

## 11. Açık sorular — düzeltilmiş kararlar (bağımsız rereview doğrulamalı)

- **O-B1 (düzeltildi, BOOK-H1):** **Karar:** mapped network drive ve UNC
  database path'leri ilk canonical implementasyonda **desteklenmez ve typed
  fail-closed reddedilir**. Desteklenen local path'ler deterministik fiziksel
  normalizasyon (realpath+normcase+UNC/mapped-drive-tespiti) kullanır (§7).
  Önceki taslağın "ayrı kimlik = fail-closed basitlik" iddiası YANLIŞTI (fail-
  open'dı) ve KALDIRILDI. B-LOW-1, implementasyon ve testleri bağımsız kabul
  edilene kadar **açık implementation advisory olarak KALIR** — bu corrective
  yalnız tasarım gereksinimini reconcile eder, implementasyonu YAPMAZ.
- **O-B2 (düzeltildi, BOOK-M1):** **Karar:** indeks uyumluluğu **tanım-
  tabanlı ve isimden bağımsızdır**. Eşdeğer tanım, herhangi bir isimle,
  requirement'ı KARŞILAR (duplicate oluşturulmaz); yalnız aynı-isim+farklı-
  tanım typed incompatible üretir. Önceki taslağın "required-isim yoksa
  incompatible" kararı Contract B §B4.8 ile ÇELİŞİYORDU ve KALDIRILDI.
  B-LOW-2 aynı şekilde açık advisory olarak KALIR.
- **O-B3 (sıkılaştırıldı, BOOK-L1):** **Karar:** canonical modül yolu aynı
  (`data/bookticker_collector.py`, gelecekteki integration worktree'de git
  add), **AMA** mevcut main-tree untracked candidate asla staged/committed
  edilmez; canonical modül candidate'ın byte-kopyası DEĞİL, Contract B'den
  yazılan bağımsız bir implementasyondur; candidate yalnız hangi mevcut
  davranışların (reconnect/backoff/health-iskelet/shutdown-flush) yeniden
  kullanılabilir olduğunu gösteren kanıt olarak kullanılır (§2.2/§2.3).

## 12. Gate durumu (SEQ-M1 düzeltmesi)

Bağımsız-rereview'dan ÖNCE implementasyon YASAK. **Kaynak-yazımı (collector +
helper'lar) ve disposable-fixture unit testleri migration'a BLOKLU DEĞİLDİR**
— yalnız Contract B §B4.1'in frozen şema sözleşmesine bağlıdır (§9). Migration
bileşeni AYRI gate (§9) ve şunları BLOKLAR: gerçek-şema entegrasyon testleri,
`data-collection` launcher modunun runtime-açılışı, deployment/runtime
readiness. Ayrıca implementasyonun kendisi §0.1'de dondurulan gelecekteki
doğrulanmış integration SHA'sını gerektirir — mevcut HEAD veya `main` üzerinde
DOĞRUDAN çalışmaz.

Bkz. sequencing belgesi (`LAUNCHER_BOOKTICKER_IMPLEMENTATION_SEQUENCING_AND_
GATES_V1.md`) §0/§1/§11 için birleşik integration-base ve sıralama modeli.

Bu belge implementasyon, migration, deployment veya runtime aksiyonu
yetkilendirmez. Bir sonraki adım: genuinely bağımsız, taze-context bir
rereview.
