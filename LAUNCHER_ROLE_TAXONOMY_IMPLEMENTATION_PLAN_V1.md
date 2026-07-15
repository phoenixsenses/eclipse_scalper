# LAUNCHER_ROLE_TAXONOMY_IMPLEMENTATION_PLAN_V1

**Tarih:** 2026-07-15 · **Yazan:** Claude Sonnet 5 (bounded implementation-planning actor; bu revizyon: LAUNCH-H2 bounded corrective-planning actor) · **Durum:** `LAUNCHER_ROLE_TAXONOMY_IMPLEMENTATION_PLAN_OPERATOR_ACCEPTED`

> **Bu yalnızca bir plandır. Hiçbir kod/test/launcher/migration/runtime
> değişikliği yapılmadı.** Bu paket, operatör-kabul-edilmiş
> `LAUNCHER_ROLE_TAXONOMY_PREREGISTRATION_V1.md` (Contract A,
> `LAUNCHER_ROLE_TAXONOMY_PREREGISTRATION_OPERATOR_ACCEPTED`) davranışını
> gelecekteki bounded implementasyona çevirir. Plan **genuinely bağımsız,
> taze-context Opus review**'undan geçmeden hiçbir implementasyon
> yetkilendirmez. Contract A'nın substantive maddeleri DEĞİŞTİRİLMEZ; bu belge
> yalnız onları tercüme eder.
>
> **Bu belge, ilk taslağın genuinely bağımsız fresh-context Opus review'unda
> `LAUNCHER_IMPLEMENTATION_PLAN_REREVIEW_REQUIRED` verdiktiyle döndüğü
> `LAUNCH-B1` (BLOCKER), `LAUNCH-H1` (HIGH), `LAUNCH-M1`/`LAUNCH-M2` (MEDIUM),
> `LAUNCH-L1` (LOW) bulgularına yanıt veren birinci corrective idi.** Bu
> **birinci corrective, genuinely bağımsız fresh-context Opus rereview'unda
> `BOTH_IMPLEMENTATION_PLANS_CORRECTIVE_REQUIRED` verdiktiyle döndü**: 13
> bulgudan 12'si kapandı, ancak `LAUNCH-H1` düzeltmesi FAZLA İLERİ GİTTİ ve
> yeni bir bloklayıcı bulgu doğurdu — `LAUNCH-H2` (HIGH): `data-collection`
> modunu "implementasyon-yetkili" ilan edip Contract A §3.2.2 ve §8'in
> "implementasyon-yetkilendirilmemiş / kapsam dışı" dondurmasıyla
> ÇELİŞİYORDU. **Bu revizyon (`LAUNCH-H2` corrective'i) yalnız o bulguya
> yanıt verir.** §0 düzeltme geçmişini kaydeder. Bu corrective KENDİ KENDİNİ
> KABUL ETMEZ — bir sonraki adım genuinely bağımsız, taze-context bir
> rereview'dır.

---

## 0. Düzeltme geçmişi

### 0.a Birinci corrective (LAUNCH-B1/H1/M1/M2/L1)

Kontrol eden bağımsız review: `LAUNCHER_IMPLEMENTATION_PLAN_REREVIEW_REQUIRED`
(genuinely bağımsız, taze-context Opus reviewer; sıfır mutasyon; ağaç review
boyunca donmuş).

| Bulgu | Şiddet | Prior metin | Birinci corrective'te ne değişti | Etkilenen bölüm |
|---|---|---|---|---|
| LAUNCH-B1 | BLOCKER | Plan, mevcut branch tip'i (`c6cf1451`) ile `main` (`cdeb9009`) kod temelini tek bir uygulanabilir taban gibi ele alıyordu; `$roleDeps`/preflight/exit-ailesi/69 test yalnız `main`'de var, HEAD'de YOK | Yeni §0.1: hiçbir mevcut commit implementasyon tabanı DEĞİL; ayrı, açıkça yetkilendirilmiş bir integration-preparation fazı gerekli; tüm kaynak referansları `main @ cdeb9009`'a yeniden-çıpalandı + "future integration SHA'da yeniden doğrulanmalı" notu eklendi | §0.1, §1, Kaynak temeli, §4.1-4.8, §5 |
| LAUNCH-H1 | HIGH | §4.3/§9 `data-collection` → `INVALID_REQUEST`+exit 21 öneriyordu; Contract A §3.2.2 ile ÇELİŞİYORDU (§3.2.2 + Contract A exit-ailesi eksik-DATA-bağımlılığı için exit 10 preflight-fail dili taşır, exit 21 DEĞİL) | Birinci corrective: `INVALID_REQUEST`+exit-21 icadını KALDIRDI. **ANCAK FAZLA İLERİ GİTTİ:** modu "implementasyon-yetkili" ve exit-10'u "mevcut yetkilendirilmiş davranış" ilan etti — bu, bağımsız rereview'da `LAUNCH-H2` olarak tespit edildi (aşağıda 0.b). Bu satırın "recognized+exit-10 mevcut davranış" kararı §0.b'de GEÇERSİZ KILINDI | §1, §4.3, §6.2/§6.3, §9 |
| LAUNCH-M1 | MEDIUM | Tekrarlı/çoklu `-Mode` testleri PowerShell scalar-param binding hatasıyla infeasible olabilirdi (launcher-owned exit 21 + JSON summary'ye asla ulaşmayan bir host-level binding hatası) | Yeni §4.3a: raw-invocation-token tabanlı mod-parse tasarımı dondurulur (scalar param binding'e GÜVENİLMEZ); 6 feasibility testi tanımlandı | §4.3a, §6.2/§6.3 |
| LAUNCH-M2 | MEDIUM | Modül import-failure semantiği dondurulmamıştı (fail-open riski) | Yeni §2.1: `$PSScriptRoot`-relative pure import, missing/malformed/wrong-export-surface → exit 20/`INTERNAL_FAILURE`, sıfır mutasyon, inline-fallback YASAK; testler eklendi | §2.1, §6.2/§6.3 |
| LAUNCH-L1 | LOW (advisory) | `$roleDeps` ve taksonomi arasında bidirectional invariant açıkça dondurulmamıştı | §2 güncellendi: `set($roleDeps.Keys) == set(taxonomy role names)` açık invariant + fail-closed davranış; testler bağımsız sabit-liste kullanır | §2, §6.2 |
| SEQ-H1 (paylaşılan) | HIGH | Implementasyon tabanı belirsizdi | §0.1 ile kapatıldı (bkz. yukarı) — sequencing belgesiyle tutarlı | §0.1 |
| SEQ-M1 (paylaşılan) | MEDIUM | Bu workstream'i doğrudan etkilemiyor (yalnız Contract B/migration sıralaması) | Değişiklik yok; bkz. sequencing belgesi §11 | — |

### 0.b İkinci corrective — bu revizyon (LAUNCH-H2)

Kontrol eden bağımsız rereview: `BOTH_IMPLEMENTATION_PLANS_CORRECTIVE_REQUIRED`
(genuinely bağımsız, taze-context Opus reviewer; sıfır mutasyon; ağaç review
boyunca donmuş; 12/13 önceki bulgu kapalı doğrulandı; base-divergence + BookTicker
plan + sequencing/migration sound doğrulandı).

| Bulgu | Şiddet | Prior metin (birinci corrective) | Bu (ikinci) corrective'te ne değişti | Etkilenen bölüm |
|---|---|---|---|---|
| LAUNCH-H2 | HIGH (BLOCKING) | Birinci corrective §1/§4.3/§9-O-A1 şunları iddia ediyordu: (1) `data-collection` **implementasyon-yetkilidir**; (2) "Contract A §3.2 tablosu hiçbir modu implementasyon-yetkisiz olarak dondurmaz"; (3) mod diğer açık modlar gibi Workstream A'da implemente edilir; (4) eksik-dep→exit-10 mevcut yetkilendirilmiş davranıştır; (5) hard-reserve YENİ bir Contract A amendment gerektirir. Bunlar Contract A **§3.2.2** (mod scope cümlesi) ve **§8** (kapsam dışı / implementasyon-yetkilendirilmemiş dondurma) ile ÇELİŞİYORDU — bu iki freeze §4.3'te HİÇ atıf almıyordu (§8 sıfır kez zikredildi) | **`data-collection` mevcut Launcher paketinden TAMAMEN DIŞLANDI** (§1, §4.3, §5). `dataCollectionImplementationAuthorized = FALSE` donduruldu. §3.2 tablosunda token'ın metadata olarak var olması ≠ executable-mod desteği; implementasyonun onu dispatch EDEMEYECEĞİ kanıtlanır (§4.3, §6.2). Contract A §3.2.2 exit-10 dili ve §3.2.2/§8 kapsam-dışı dondurması arasındaki gerilim çözülmeden, dürüstçe operatör-clarification'a taşındı (yeni O-A1, §9). test-21 (recognized→exit-10-mevcut-davranış) KALDIRILDI; yerine scope-fidelity + non-dispatch testleri (§6.3/§6.4). Traceability §3.2.2 + §8'i zikreder; sonuç = `CURRENT_DATA_COLLECTION_IMPLEMENTATION_EXCLUDED_PENDING_OPERATOR_CONTRACT_CLARIFICATION` (§3) | §1, §3, §4.3, §5, §6.3, §6.4, §9, §10 |

**Kapsam sınırı (bu ikinci corrective):** yalnız `LAUNCH-H2` ele alınır.
Birinci corrective'in kabul edilen 12 kapanışı (LAUNCH-B1/M1/M2/L1, BOOK-*,
SEQ-*) yeniden AÇILMAZ; base-divergence gate modeli (§0.1), raw-token mod-parse
(§4.3a), `$PSScriptRoot`-relative import (§2.1), exit-20 import-failure,
bidirectional taxonomy invariant (§2) korunur.

### 0.c Operatör kabulü (2026-07-15)

İkinci corrective'in genuinely bağımsız, taze-context Opus rereview'u
`BOTH_IMPLEMENTATION_PLANS_INDEPENDENT_REREVIEW_ACCEPTED` verdiktiyle döndü:
`LAUNCH-H2` tam düzeltilmiş bulundu; yeni BLOCKER/HIGH/MEDIUM bulgu YOK; önceki
12 kapanış (LAUNCH-B1/M1/M2/L1, BOOK-*, SEQ-*) regresyonsuz doğrulandı; iki
INFO-seviye gözlem (citation precision, O-A1 doc-presence probe) non-blocking
kabul edildi. Operatör bu rereview'u **2026-07-15 tarihinde kabul etmiştir.**

**Bu kabul, sequencing belgesi §3'teki `PLAN_ACCEPTED` kapısına karşılık gelir
— zincirin ilerideki (IMPLEMENTATION_ACCEPTED SONRASI) `OPERATOR_ACCEPTED`
durumuyla KARIŞTIRILMAMALIDIR (bkz. sequencing §3 gate-modeli, "ASLA tek
statüye birleştirilmez").** Bu belge başlığındaki `_OPERATOR_ACCEPTED` sonek
konvansiyonu, Contract A/B preregistration kabulünde kullanılan aynı
doküman-durumu deseninin (`LAUNCHER_ROLE_TAXONOMY_PREREGISTRATION_OPERATOR_
ACCEPTED`) devamıdır — bu belgenin operatör tarafından incelenip kabul
edildiğini belirtir, implementasyonun tamamlandığını veya kabul edildiğini
DEĞİL.

**Bu kabul açıkça şunları YETKİLENDİRMEZ:** kod/test/launcher implementasyonu;
`INTEGRATION_BASE_ESTABLISHED` (henüz `false`/oluşturulmamış — ayrı
integration-preparation yetkilendirmesi gerekir, §0.1); migration oluşturma/
rezervasyon; aday (`data/bookticker_collector.py`) canonicalization/import/
execute; runtime probe; deployment; activation. `data-collection` modu
implementasyon-dışı KALIR, `O-A1` (`DATA_COLLECTION_PREAUTHORIZATION_
BEHAVIOR_REQUIRES_OPERATOR_CONTRACT_CLARIFICATION`) ÇÖZÜLMEMİŞ ve
non-bypassable olarak kalır — bu kabul O-A1'i zımnen YANITLAMAZ.

Bu corrective kendi kendini kabul ETMEZ. Bir sonraki adım genuinely bağımsız,
taze-context bir rereview'dır.

---

## 0.1 İmplementasyon tabanı — kritik düzeltme (LAUNCH-B1 / SEQ-H1)

**Bulgu (bağımsız review):** Bu planın ilk taslağı, kaynak temelini "salt-okunur
inceleme, `main` @ `cdeb9009` + Contract A" olarak beyan ediyordu ama Contract
A'nın kabul edildiği ve bu plan belgelerinin yaşadığı dal (`codex/data-layer-
fallback-cleanup`, HEAD `c6cf1451`) ile `$roleDeps`/`Invoke-EclipsePreflight`/
`Start-OptionalRole`/`Get-EclipseStartFinalResult`/`Exit-EclipseStart`/exit-
ailesi/69-test launcher-corrective'inin yaşadığı `main` (`cdeb9009`) **birbirinin
atası DEĞİL** (merge-base `74468c87` = HEAD'in ebeveyni). HEAD'deki
`start_eclipse.ps1` 415 satır, `$roleDeps`/preflight/mod/exit-ailesi YOK;
`tests/test_start_eclipse_launcher_safety.py` HEAD'de YOK (yalnız `main`'de,
789 satırlık launcher + 69 test). **Hiçbir mevcut commit hem kabul-edilmiş
governance'ı hem de gerekli launcher-corrective kod temelini birlikte
içermiyor.**

**Dondurulmuş karar:** Ne `c6cf1451` (yalnız governance) ne de `cdeb9009`
(yalnız kod) tek başına implementasyon tabanı olarak KULLANILAMAZ.

**Gerekli gelecekteki integration-base kapısı (bu corrective turunun DIŞINDA,
ayrı yetkilendirme gerektirir):**

İmplementasyon yetkilendirilmeden ÖNCE, ayrı ve açıkça yetkilendirilmiş bir
**integration-preparation fazı**, izole bir integration branch/worktree
oluşturmalıdır; bu branch şunları birleştirir:

1. `main @ cdeb90096bfe7b448384b098094586cab750d5e6`'daki canonical kod
   temeli (launcher-corrective + `Invoke-EclipsePreflight` + 69 test +
   `scripts/collector_supervisor.py`'nin `acquire_supervisor_lock`/
   `release_supervisor_lock` `msvcrt.locking` modeli — bkz. BookTicker planı
   §0.1);
2. `c6cf1451accf681c910e1a4e173560f55979fa38`'deki operatör-kabul-edilmiş
   governance durumu (Contract A + Contract B + preregistration'lar +
   `OPERATOR_DECISION_QUEUE.md` OD-024/OD-025 kayıtları);
3. bu implementation-planning paketi — **yalnız** bağımsız rereview'dan geçip,
   operatör kabulü alıp, kendi bounded governance kapısından commit edildikten
   SONRA.

Gelecekteki integration actor, ayrı bir prompt tarafından yetkilendirilen
history-preserving bir işlem kullanmalıdır (örn. kabul-edilmiş governance/
planning dokümantasyon commit'lerini canonical `main` temelli izole bir dal
üzerine uygulamak). **Bu corrective turu hiçbir merge/cherry-pick/rebase/reset
işlemi seçmez, yürütmez veya yetkilendirmez.**

**Gerekli integration-base doğrulaması:** Gelecekteki integration tabanı,
bağımsız bir integration-review'da şunları içerdiğini KANITLAMALIDIR:
kabul-edilmiş Contract A; kabul-edilmiş Contract B; kabul-edilmiş
implementation plan'lar (bu üç belge, kendi rereview zincirinden geçtikten
sonra); `main`'in mevcut launcher-corrective'i; canonical launcher-safety test
paketi (69 test); `main`'in kabul-edilmiş `CollectorSupervisor` `msvcrt.locking`
modeli; hiçbir ilgisiz implementasyon değişikliği. Bu doğrulama implementasyon
başlamadan ÖNCE, kendi bağımsız integration review'undan geçmelidir.

**Yanıltıcı çıpaların düzeltilmesi:** Bu belgedeki hiçbir ifade artık
implementasyonun doğrudan mevcut HEAD (`c6cf1451`) üzerinde, "kabul edilmiş
commit" adı altında entegrasyon-doğrulaması olmadan, veya `main @ cdeb9009`
üzerinde governance-reconciliation olmadan çalışacağını İDDİA ETMEZ. Aşağıdaki
tüm `main @ cdeb9009` kaynak referansları (`$roleDeps`, `Invoke-
EclipsePreflight`, `Start-OptionalRole`, `Get-EclipseStartFinalResult`,
`Exit-EclipseStart`, exit ailesi, 69 test) **yalnızca `main @ cdeb9009`'a**
çıpalanmıştır — mevcut branch tip'ine DEĞİL — ve implementasyon başlamadan
ÖNCE gelecekteki doğrulanmış integration SHA'sında **yeniden doğrulanmalıdır**
(dosya varlığı, satır numaraları, fonksiyon imzaları, 69-test koleksiyon
sayısı).

---

## 1. Kapsam ve amaç

Contract A'nın dondurduğu davranışı zorlayan **asgari** implementasyonu tanımlar:
12/12 rol taksonomisi, `SCHEDULER_GATED`, `legacy-full-contract` varsayılanı,
mevcut pakette **implementasyon-yetkili açık modlar** (`core-infrastructure`,
`shadow-observation`, `diagnostics`), optional-rol dört-sonuç matrisi, exit
ailesi 10/11/12/20/21/0, `degraded_reason`/`optional_unavailable`/
`optional_failures` çıktı şeması, `live-execution` =
`RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT`, tek-mod kuralı, scheduler
enable-switch etkileşimi, stop-side uyumluluk, geriye-uyumluluk.
`data-collection` modu Contract A §3.2.2/§8 gereği bu paketin **DIŞINDADIR**
(aşağı). Implementasyon, §0.1'de dondurulan gelecekteki doğrulanmış integration
SHA'sı ÜZERİNDE gerçekleşir — mevcut HEAD veya `main` üzerinde DEĞİL.

**`data-collection` modunun disposition'ı (LAUNCH-H2 düzeltmesi — birinci
corrective'in FAZLA İLERİ giden iddiası GERİ ALINDI):** Bu workstream Contract
B'ye (`data/bookticker_collector.py` canonicalization) **bağımlı DEĞİLDİR**
(core/shadow/diagnostics/legacy tarafları için). Ancak `data-collection`
modu Contract A **§3.2.2**'nin son cümlesi ("Bu mod implementasyon-
yetkilendirilmiş DEĞİLDİR (§8 kapsam dışı)") ve **§8**'in ("`data-collection`
modu — bileşenleri kendi bounded-implementation zincirlerini tamamlamadıkça
implementasyon-yetkilendirilmemiştir") açık dondurmalarına tabidir:

- **`dataCollectionImplementationAuthorized = FALSE`** (bu Launcher paketi için).
- Contract A **modu implementasyon-yetkilendirmez**; birinci corrective'in
  "Contract A §3.2 tablosu hiçbir modu implementasyon-yetkisiz olarak
  dondurmaz" iddiası YANLIŞTI — §3.2 *tablosu* modu metadata olarak listeler,
  ama §3.2.2 *prose'u* ve §8 *kapsam-dışı bölümü* onu açıkça
  implementasyon-yetkilendirilmemiş dondurur.
- **Vocabulary recognition ≠ implementation authorization.** Token'ın taksonomi
  metadata'sında var olması (§3.2 tablosu) onu executable/dispatch-edilebilir
  YAPMAZ.
- Mevcut Launcher paketi `data-collection` için **hiçbir launch branch,
  dependency-map, preflight, dispatch veya proses mutasyonu EKLEMEZ** (§4.3, §5,
  §6.3).
- Contract A §3.2.2, eksik DATA bağımlılıklarını exit-10 preflight-ailesiyle
  *tanımlar*; ama bu davranışsal betimleme modun bu pakette implemente
  edilmesini **yetkilendirmez** (§9/O-A1) — betimleme ve kapsam-freeze
  arasındaki gerilim operatör-clarification'a taşınır, tek yönde çözülmez.

Contract B chain'inin tamamlanması + operatör yetkilendirmesi + uygulanabilir
contract-clarification KOŞULU altında, **gelecekteki** bir Launcher paketi
§3.2.2 dependency-preflight davranışını (eksik-DATA-bağımlılığı exit-ailesi
dahil) implemente edebilir (§4.3 "Gelecek koşulu"). Bu gelecek ifadesi mevcut
paket için bir yetkilendirme DEĞİLDİR.

## 2. Tasarım kararı — implementasyon nereye ait?

Değerlendirilen seçenekler ve karar:

| Seçenek | Değerlendirme |
|---|---|
| Doğrudan `start_eclipse.ps1` içinde | Taksonomi + mod sözleşmesi + outcome matrisi tek büyük dosyaya gömülür; testler production-path'i çağırmak zorunda (subprocess) ve taksonomiyi yapısal olarak assert edemez → §6.3 mutasyon-probeleri zorlaşır |
| **Saf PowerShell helper modülü (ÖNERİLEN, review'da RETAINED)** | Yeni `scripts/eclipse_launcher_taxonomy.psm1`: 12/12 rol→kategori haritası + mod required/optional/excluded setleri + outcome-matrisi saf fonksiyonları + status/exit türetimi. `start_eclipse.ps1` ve testler AYNI modülü import eder → tek doğruluk kaynağı, yapısal test mümkün, davranışsal wiring launcher'da kalır |
| Üretilen/statik manifest (JSON/PSD1) | Taksonomi verisi için iyi ama mod-semantiği ve outcome-matrisi mantık gerektirir; salt-veri manifest mantığı taşıyamaz → helper modülüyle birleştirilebilir (manifest = modül içi `[ordered]@{}` sabiti) |
| Mevcut launcher helper | `start_eclipse.ps1` içinde ayrı fonksiyon var ama ayrı dosya/modül yok; test-import için modül gerekir |

**Karar (bağımsız review'da doğrulandı, O-A2 kapandı):** Tek yeni saf-PowerShell
modülü `scripts/eclipse_launcher_taxonomy.psm1`. İçindekiler:
- `$EclipseRoleTaxonomy` — 12 rol → 6 kategori `[ordered]@{}` (Contract A §2.1
  tablosunun birebir kopyası, tek doğruluk kaynağı);
- `$EclipseModeContracts` — mevcut pakette **implementasyon-yetkili açık
  modlar** (`core-infrastructure`, `shadow-observation`, `diagnostics`) +
  `legacy-full-contract` için required/optional/excluded setleri (Contract A
  §3.1/§3.2 tablosu). `live-execution` **reserved** olarak ayrı ele alınır
  (§4.4). **`data-collection` metadata-only'dir (LAUNCH-H2):** token Contract A
  §3.2 tablosunda var olduğu için taksonomi metadata'sında görünebilir, ama
  **`Resolve-EclipseModeContract` onu dispatch-edilebilir bir mod olarak
  ÇÖZMEZ** — `Assert-EclipseModeInvocation` `-Mode data-collection` girişini
  mevcut pakette **implementasyon-yetkilendirilmemiş / non-dispatchable** olarak
  ele alır (kesin pre-authorization davranışı §9/O-A1'e bağlı, bu planda
  DONDURULMAZ). Metadata-varlığı ile executable-mod-desteği açıkça ayrıdır ve
  §6.2/§6.3 testleri implementasyonun bu modu dispatch EDEMEYECEĞİNİ kanıtlar;
- saf fonksiyonlar: `Get-EclipseRoleCategory`, `Resolve-EclipseModeContract`,
  `Get-EclipseOptionalOutcome` (dört-sonuç matrisi §3.3),
  `Get-EclipseOverallStatusAndExit` (§4.2/§5 tek karar kaynağı),
  `Assert-EclipseModeInvocation` (§4.3a — mod-parse/duplicate/malformed tespiti,
  LAUNCH-M1 düzeltmesi).

`start_eclipse.ps1`'in mevcut `$roleDeps`'i **dependency-path** kaynağı olarak
kalır (dosya varlığı için); rol→kategori ve mod setleri modülden gelir.

**Bidirectional invariant (LAUNCH-L1 düzeltmesi):** Taksonomiyi iki yerde
çoğaltmayı önlemek için implementasyon şu invariant'ı **mutasyondan önce**
zorlar:

```
set($roleDeps.Keys) == set($EclipseRoleTaxonomy role adları)
```

Fail-closed koşulları (herhangi biri → modül self-check fail → exit 20,
`INTERNAL_FAILURE`, sıfır mutasyon):
- bir `$roleDeps` rolü taksonomide YOK;
- bir taksonomi rolü `$roleDeps`'te YOK;
- çift veya farklı-case'li bir rol adı var;
- toplam tam olarak 12 DEĞİL.

Testler (§6.2) beklenen 12 canonical rol adını ve kategori sayılarını
**bağımsız, sabit bir listeden** doğrular — yalnız üretim modülünden
türetilen değerlerle karşılaştırma YAPILMAZ (aksi halde test, modülün
kendi hatasını da "doğru" olarak onaylar).

**En küçük tasarım ilkesi:** taksonomi TEK yerde (`$EclipseRoleTaxonomy`);
`start_eclipse.ps1` ve testler onu import eder; hiçbir test-only kopya taksonomi
yoktur. LOW-advisory kapsamını gereksiz yere büyütmemek için `$roleDeps`'i
ortadan kaldırmak (tek bir birleşik yapıya indirgemek) bu turda ZORUNLU
DEĞİLDİR — invariant + testler yeterli kapanış sağlar.

### 2.1 Modül yükleme semantiği — fail-closed (LAUNCH-M2 düzeltmesi)

Önceki taslak modül import-failure davranışını dondurmamıştı (fail-open riski:
import sessizce başarısız olursa self-check hiç çalışmaz). Dondurulmuş
davranış:

- Modül `$PSScriptRoot`-relative olarak resolve edilir (`Join-Path
  $PSScriptRoot "..\scripts\eclipse_launcher_taxonomy.psm1"` benzeri) — asla
  current-working-directory'e bağımlı DEĞİL (Scheduled Task farklı bir cwd'den
  çalıştırabilir).
- Import, herhangi bir rol/proses/task mutasyonundan **önce** gerçekleşir.
- Modül **saf ve yan-etkisiz** olmalıdır (dosya/proses/task dokunmaz; yalnız
  veri + fonksiyon tanımlar).
- Modül eksikse, parse hatası varsa, import başarısız olursa, veya beklenen
  export yüzeyi (fonksiyon/değişken adları) eşleşmiyorsa: **fail-closed**
  — `overall_status=INTERNAL_FAILURE`, exit `20`, sıfır mutasyon. Yeni bir
  exit kodu İCAT EDİLMEZ (§4.2 mevcut ailede `20` = "iç launcher/runtime
  hatası" zaten kapsıyor).
- Import başarısız olursa launcher **inline/duplike bir taksonomiye
  ASLA düşmez** (fallback yasak — bu, §2'nin tek-doğruluk-kaynağı ilkesini
  bypass eder).
- Import başarılı olduktan hemen sonra 12-rol yapısal self-check (§2)
  çalışır; self-check fail → aynı exit-20/`INTERNAL_FAILURE`/sıfır-mutasyon
  yolu.

Testler (§6.2/§6.3): Scheduled-Task-benzeri non-repository cwd'den çağrı;
modül eksik; modül malformed (syntax error); yanlış export yüzeyi (beklenen
fonksiyon/değişken eksik); yapısal self-check fail; inline-fallback YOK
kanıtı (statik: launcher kaynağında ikinci bir taksonomi tanımı yok); import
başarısızlığında sıfır mutasyon.

## 3. Contract A madde → implementasyon izlenebilirliği

| Contract A maddesi | Davranış | Uygulama yeri (öneri) |
|---|---|---|
| §2.1 (12/12 tablo) | rol→kategori tek doğruluk | `$EclipseRoleTaxonomy` (yeni modül) |
| §2 SCHEDULER_GATED | `liquidation_silence_scheduler` kategorisi + legacy/mode davranışı | `$EclipseRoleTaxonomy` + `Resolve-EclipseModeContract` |
| §3.1 legacy-full-contract | mod yoksa 12-rol preflight, eksik→exit 10, `effective_mode` | `start_eclipse.ps1` (mevcut preflight + yeni `effective_mode` alanı) |
| §3.2 açık mod tablosu | mevcut pakette implementasyon-yetkili açık modlar (`core-infrastructure`, `shadow-observation`, `diagnostics`) için required/optional/excluded set; her biri recognized + preflight-doğrulanır. `data-collection` **metadata-only, dispatch-edilmez** (§3.2.2/§8); `live-execution` reserved | `$EclipseModeContracts` |
| §3.2.2 data-collection (davranış dili) | Contract A eksik-DATA-bağımlılığını exit-10 preflight-ailesiyle *tanımlar* — ama bu betimleme mevcut pakette **implementasyonu YETKİLENDİRMEZ**; disposition operatör-clarification'a taşınır (LAUNCH-H2, §9/O-A1) | Mevcut pakette **implemente edilmez** (non-dispatch, §4.3) |
| **§3.2.2 (scope cümlesi) data-collection** | "Bu mod implementasyon-yetkilendirilmiş DEĞİLDİR (§8 kapsam dışı)" — **freeze** | Mevcut Launcher paketi DIŞI (§4.3, §9/O-A1) |
| **§8 kapsam dışı freeze** | "`data-collection` modu — bileşenleri kendi bounded-implementation zincirlerini tamamlamadıkça implementasyon-yetkilendirilmemiştir" — **freeze** | Mevcut Launcher paketi DIŞI; gelecek koşullu (§4.3) |
| §3.2.5 live-execution reserved | `-Mode live-execution` → exit 21, INVALID_REQUEST | `start_eclipse.ps1` mod-parse (mutasyondan önce) |
| §3.3 outcome matrisi | 4 sonuç, attempted-fail asla exit 0 | `Get-EclipseOptionalOutcome` + `Get-EclipseOverallStatusAndExit` |
| §3.4 tek-mod | çoklu/tekrarlı mod → exit 21 | `Assert-EclipseModeInvocation` (raw-token tabanlı, §4.3a) |
| §4.2 exit ailesi | 10/11/12/20/21/0, yeni kod yok | `Get-EclipseOverallStatusAndExit` (tek kaynak) |
| §5 çıktı şeması | 15 alan + degraded_reason enum | `start_eclipse.ps1` özet üretimi + modül türetimi |
| §4.1 non-regresyon (10 kural) | mevcut fail-closed korunur | mevcut `start_eclipse.ps1` davranışı + regresyon testleri |

**Traceability sonucu — `data-collection` (LAUNCH-H2):**
`CURRENT_DATA_COLLECTION_IMPLEMENTATION_EXCLUDED_PENDING_OPERATOR_CONTRACT_CLARIFICATION`.
Bu satır implemented / implementable / açık / yetkilendirilmiş olarak
ETİKETLENMEZ. Contract A'nın üç ilgili klozu birlikte zikredilir: §3.2.2
dependency/preflight dili (exit-10 betimlemesi), §3.2.2 scope cümlesi
(implementasyon-yetkilendirilmemiş), §8 kapsam-dışı freeze. Betimleyici
exit-10 dili tek başına implementasyon yetkisi türetmez.

## 4. Davranışsal implementasyon spesifikasyonu

### 4.1 Rol manifesti ve kategoriler (§2.1)
12 canonical rol adı `main @ cdeb9009:start_eclipse.ps1` satır ~315-328'den
birebir alınır (§0.1 gereği implementasyon başlamadan önce gelecekteki
integration SHA'sında yeniden doğrulanır): collector_supervisor,
heartbeat_watchdog, bookticker_collector, oi_spot_poller,
s34_shadow_paper_runner, s34_live_chart, s34_v_engine_v02_shadow_mirror,
s34_state_machine_shadow_runner, s34_state_machine_live_executor,
liquidation_silence_scheduler, orderflow_chart, s34_replay. Kategori sayıları:
CORE=2, DATA=2, SHADOW=3, DIAG=3, LIVE=1, SCHEDULER=1 = 12. Modül yükleme
sırasında bir **self-check** taksonomi tam 12 rol/tek-eşleme değilse
fail-closed olur (§2 bidirectional invariant).

### 4.2 legacy-full-contract (§3.1)
`-Mode` verilmezse: mevcut `Invoke-EclipsePreflight` davranışı (`main @
cdeb9009`, integration SHA'sında yeniden doğrulanacak) AYNEN korunur (12-rol
`Test-Path`, eksik→exit 10 mutasyondan önce); yeni tek fark özet çıktıya
`effective_mode=legacy-full-contract` + `legacy_compatibility_mode=true`
eklenir. Scheduler bağımlılığı manifestte kalır; proses başlatma mevcut
`-EnableLiquidationSilenceScheduler` ile yönetilir (switch yoksa başlatılmaz,
owned durum korunur).

### 4.3 Açık modlar (§3.2) + `data-collection` disposition (LAUNCH-H2 düzeltmesi)

`Resolve-EclipseModeContract` seçilen mod için required/optional/excluded
döner. **Mevcut pakette implementasyon-yetkili açık modlar** yalnız
`core-infrastructure`, `shadow-observation`, `diagnostics`'tir. `live-execution`
`RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT` olarak ayrı bir semantik alır
(§4.4, değişmedi). **`data-collection` bu paketin DIŞINDADIR** (aşağı).

**LAUNCH-H2 düzeltmesi — birinci corrective'in FAZLA İLERİ giden iddiası GERİ
ALINDI:** Birinci corrective, `data-collection`'ı "recognized ve
implementasyon-yetkili" ilan edip "Contract A §3.2 tablosu hiçbir modu
implementasyon-yetkisiz olarak dondurmaz" dedi. Bu YANLIŞTI. Contract A iki
ayrı **freeze** taşır:
- **§3.2.2 (scope cümlesi):** "Bu mod implementasyon-yetkilendirilmiş DEĞİLDİR
  (§8 kapsam dışı)."
- **§8 (kapsam dışı):** "`data-collection` modu — bileşenleri kendi
  bounded-implementation zincirlerini tamamlamadıkça
  implementasyon-yetkilendirilmemiştir."
§3.2 *tablosu* modu yalnız metadata olarak listeler; yetkilendirme kararını
§3.2.2 prose'u + §8 verir ve ikisi de modu implementasyon-DIŞI dondurur.
Not: Contract A §8, `data-collection`'ı `live-execution` ile **aynı**
kapsam-dışı listeye koyar.

**Dondurulmuş disposition (bu paket):**
1. **`dataCollectionImplementationAuthorized = FALSE`.** Mevcut Launcher paketi
   `data-collection` için **hiçbir launch branch, dependency-map, preflight,
   dispatch veya proses mutasyonu EKLEMEZ.**
2. **Vocabulary recognition ≠ implementation authorization.** Token Contract A
   §3.2 tablosunda geçtiği için taksonomi metadata'sında (`$EclipseModeContracts`)
   görünebilir, ama `Resolve-EclipseModeContract` onu dispatch-edilebilir bir
   moda ÇÖZMEZ; implementasyon onu dispatch EDEMEZ (§6.2/§6.3 kanıtı).
3. **Betimleyici exit-10 dili ≠ implementasyon yetkisi.** Contract A §3.2.2,
   eksik DATA bağımlılıklarını exit-10 preflight-ailesiyle *tanımlar*. Bu
   davranışsal betimleme, modun bu pakette implemente edilmesini
   **YETKİLENDİRMEZ**. Betimleme (exit-10) ile scope-freeze (§3.2.2 son cümle +
   §8) arasındaki gerilim bu planla tek yönde ÇÖZÜLMEZ.
4. **Pre-authorization davranışı DONDURULMAZ.** Bir kullanıcı `-Mode
   data-collection`'ı yetkilendirmeden önce verirse launcher'ın döneceği kesin
   exit kodu bu planda seçilmez (ne exit-10 ne exit-21 icat edilir); yalnız
   şart: **sıfır mutasyon + machine-readable fail-closed davranış** (§9/O-A1).
   Kesin davranış operatör-clarification'a taşınır.
5. **Burden inversion düzeltmesi:** Birinci corrective "hard-reserve YENİ bir
   Contract A amendment gerektirir" diyordu — bu TERSTİ. §8 modu ZATEN
   kapsam-dışı dondurur; dolayısıyla varsayılan sözleşme-pozisyonu
   *withhold*'dur. Modu implementasyon-yetkili SAYMAK, ek bir operatör
   clarification / contract amendment gerektirir — tersi DEĞİL.

**Gelecek koşulu (mevcut paket için yetkilendirme DEĞİLDİR):** Tüm gerekli
bounded component-chain'ler bağımsızca kabul edildikten VE operatör modu açmayı
ayrıca yetkilendirdikten VE uygulanabilir contract-clarification/amendment
mevcut olduktan SONRA, **gelecekteki** bir Launcher paketi Contract A §3.2.2
dependency-preflight davranışını (belirtilen eksik-DATA-bağımlılığı exit-ailesi
dahil) implemente edebilir. Formal olarak:

```
ComponentChainsAccepted ∧ OperatorAuthorization ∧ ApplicableContractClarification
  ⇒ FutureDataCollectionImplementationMayBePlanned
```

Bu, şunu İMA ETMEZ:

```
ContractVocabularyPresence ⇒ CurrentImplementationAuthorization
```

### 4.3a Mod-parse mekanizması — tekrarlı/çoklu/malformed tespiti (LAUNCH-M1 düzeltmesi)

**Sorun:** `-Mode` tipik bir skaler `[string]$Mode` PowerShell parametresi
olarak tanımlanırsa, `-Mode a -Mode b` PowerShell'in KENDİ parameter-binding
hatasıyla reddedilir — script gövdesi hiç ÇALIŞMAZ, launcher-owned exit `21`
ve Contract A §5 JSON özeti asla ÜRETİLMEZ (host-level binding hatası, script
exit code'u değil). Bu, §6.2 test-19'u (tekrarlı/çoklu mod → exit 21 +
JSON) infeasible/vacuous bırakır.

**Dondurulmuş tasarım:** Launcher, mod seçimini **skaler parametre-binding'e
GÜVENMEDEN**, ham invocation token'larını kendisi inceleyerek tespit eder:

1. Launcher `param()` bloğunda `-Mode` **skaler DEĞİL**, tolerant bir biçimde
   tanımlanır (ör. `[string[]]$Mode = @()` veya `$MyInvocation.UnboundArguments`
   + `$PSBoundParameters` birlikte incelenir) — böylece PowerShell'in kendisi
   birden fazla `-Mode` girişini reddetmeden önce script gövdesi çalışabilir
   ve launcher-owned karar mekanizmasına ulaşabilir.
2. `Assert-EclipseModeInvocation` (yeni modül fonksiyonu, saf) ham mod
   girdisini alır ve şunları **mutasyondan önce** tespit eder:
   - `-Mode` hiç verilmemiş → `legacy-full-contract` (§4.2);
   - tam olarak bir `-Mode <isim>` verilmiş, `<isim>` mevcut pakette
     **implementasyon-yetkili bir açık mod** (`core-infrastructure`,
     `shadow-observation`, `diagnostics`) → o mod kabul edilir;
   - **`-Mode data-collection` (LAUNCH-H2): §3.2 tablosunda metadata olarak
     bulunsa da bu pakette dispatch-edilebilir bir mod DEĞİLDİR — `kabul edilir`
     yoluna GİRMEZ, `Resolve-EclipseModeContract`'a GEÇİRİLMEZ.** Kesin
     pre-authorization dönüşü (exit kodu) bu planda dondurulmaz (§9/O-A1); tek
     şart: **sıfır mutasyon + machine-readable fail-closed**. İmplementer bu
     davranışı operatör-clarification'dan ÖNCE seçemez; test yalnız
     "dispatch/branch/mutasyon YOK" invariant'ını doğrular (§6.3), kesin exit
     kodunu hard-code ETMEZ;
   - `-Mode live-execution` → reserved (§4.4, exit 21);
   - `-Mode` birden fazla kez verilmiş (tekrarlı veya farklı değerlerle) →
     `MALFORMED_MODE_REQUEST`;
   - `-Mode` virgülle-ayrılmış veya dizi-benzeri çoklu değer aldıysa (ör.
     `-Mode core-infrastructure,diagnostics`) → `MALFORMED_MODE_REQUEST`;
   - `-Mode` değeri boş/eksik (ör. `-Mode` son argüman, değer yok) →
     `MALFORMED_MODE_REQUEST`;
   - `-Mode <isim>` verilmiş ama `<isim>` ne implementasyon-yetkili açık mod ne
     `live-execution` ne `data-collection`, yani tanımlı DEĞİL →
     `UNKNOWN_MODE_REQUEST`.
3. Herhangi bir `MALFORMED_MODE_REQUEST`/`UNKNOWN_MODE_REQUEST` durumu:
   **sıfır mutasyon**; `overall_status=INVALID_REQUEST`; exit `21`; Contract
   A §5 tam özeti (tüm 15 alan) yayınlanır — bir PowerShell host-level binding
   hatası DEĞİL, launcher-owned bir karar.
4. Tam olarak bir geçerli mod tespit edildikten SONRA, normal işleme için
   normalize edilip bağlanır (`Resolve-EclipseModeContract`'a geçirilir).

**Feasibility testleri (§6.2/§6.3, subprocess, production launcher path'i):**
1. `-Mode` yok → `legacy-full-contract`.
2. Tek geçerli `-Mode core-infrastructure` → kabul.
3. Tekrarlı `-Mode core-infrastructure -Mode diagnostics` → exit 21 +
   `MALFORMED_MODE_REQUEST` + tam JSON özeti (host-binding hatası DEĞİL).
4. Virgül/dizi-benzeri `-Mode core-infrastructure,diagnostics` → exit 21 +
   `MALFORMED_MODE_REQUEST` + tam JSON özeti.
5. Eksik mod değeri (`-Mode` son argüman) → exit 21 + `MALFORMED_MODE_REQUEST`.
6. Bilinmeyen mod adı (`-Mode nonexistent-mode`) → exit 21 +
   `UNKNOWN_MODE_REQUEST`.

Her geçersiz durum launcher-owned exit `21` + tam özete ulaşmalı; hiçbiri bir
PowerShell host-level parameter-binding hatasına düşmemeli.

### 4.4 live-execution reserved (§3.2.5)
`-Mode live-execution` → mutasyondan önce: exit 21, `overall_status=INVALID_REQUEST`,
`live_execution_requested=true`, `live_execution_authorized=false`, mesaj
`LIVE_EXECUTION_MODE_REQUIRES_SEPARATE_DATA_READINESS_PREREGISTRATION`. Mevcut
`-EnableLive` + 3-env-var gate'i DEĞİŞMEZ (legacy default-off korunur).

### 4.5 Optional-rol outcome matrisi (§3.3)
`Get-EclipseOptionalOutcome` her optional rol için: `SKIPPED_BY_MODE` /
`SKIPPED_MISSING_OPTIONAL_DEPENDENCY` / `ALREADY_OWNED` / operasyonel-fail. Sadece
absent (#2) → DEGRADED+exit 0; attempted-fail (#4) → SECONDARY_ROLE_FAILED+exit 12
(mevcut `secondary_failures` semantiğiyle hizalı, `main @ cdeb9009:
start_eclipse.ps1` satır ~500 civarı, integration SHA'sında yeniden
doğrulanacak). **Değişmez:** attempted-fail asla exit 0'a normalize edilmez.

### 4.6 Exit + status tek kaynak (§4.2/§5/§4.1-kural 8)
`Get-EclipseOverallStatusAndExit` tek karar kaynağı; mevcut `Exit-EclipseStart`
deseni (`main @ cdeb9009`) GENİŞLETİLİR (yeni alanlar), DEĞİŞTİRİLMEZ.
`overall_status` 7-değer sözlüğü + `degraded_reason` 3-değer sözlüğü (§5)
modülde sabit enum olarak dondurulur.

### 4.7 Scheduler enable-switch etkileşimi (§2 SCHEDULER_GATED)
Açık mod + `-EnableLiquidationSilenceScheduler` → mutasyondan önce exit 21
(kombinasyon geçersiz; `Assert-EclipseModeInvocation` tarafından tespit
edilir, §4.3a mekanizmasıyla aynı yol). legacy'de switch mevcut davranışı korur.

### 4.8 Stop-side uyumluluk
`stop_eclipse.ps1` mod-farkında OLMAK ZORUNDA DEĞİL (legacy stop tüm rolleri
durdurur). Plan: stop-side davranışı DEĞİŞMEZ; yalnız `start` mod eklerse
`stop`'un hâlâ tüm bilinen rolleri güvenle ele aldığı regresyon testiyle
doğrulanır (test_30c task-mutation-yok korunur). Bu bölüm `main @ cdeb9009`
üzerindeki mevcut `stop_eclipse.ps1`'e atıfta bulunur; gelecekteki integration
SHA'sında yeniden doğrulanmalıdır (§0.1).

## 5. File-level change map (Workstream A)

| Yol | Mevcut rol | Önerilen değişiklik | Executable/Docs | Yeni/Değişen | Bağımlılık/gate | Risk | Doğrulama |
|---|---|---|---|---|---|---|---|
| `scripts/eclipse_launcher_taxonomy.psm1` | — | Taksonomi + mod sözleşmesi + outcome/exit saf fonksiyonlar + `Assert-EclipseModeInvocation` | Executable (PS module) | **Yeni** | yok (saf) | Orta | Yapısal + davranışsal test (§6) |
| `start_eclipse.ps1` (integration SHA'sındaki `main` sürümü) | Launcher | `$PSScriptRoot`-relative modül import (fail-closed, §2.1); ham-token mod-parse (§4.3a, mutasyondan önce); implementasyon-yetkili açık modlar (`core-infrastructure`/`shadow-observation`/`diagnostics`) preflight-doğrulanır; **`data-collection` implemente EDİLMEZ — hiçbir branch/dependency-map/preflight/dispatch (LAUNCH-H2, §4.3)**; `effective_mode`/`degraded_reason`/`optional_unavailable`/`optional_failures` özet alanları; outcome matrisi wiring | Executable | Değişen | Modül | **Yüksek** (fail-closed regresyon riski) | 68 test tanımı / 69 collected case (integration SHA'sında yeniden doğrulanmış/remeasure) + yeni testler + subprocess |
| `tests/test_start_eclipse_launcher_safety.py` (integration SHA'sındaki `main` sürümü) | Launcher testleri | `test_38`-`test_42`/`test_27b` supersession (Contract A §6.1 tam assertion) + §6.2 yeni testler + §6.3 mutasyon-probeleri + §4.3a mod-parse feasibility testleri + §2.1 import-failure testleri | Executable (test) | Değişen | Modül + launcher | Orta | pytest (≤2 dosya/çağrı, --basetemp scratchpad) |
| `stop_eclipse.ps1` (integration SHA'sındaki `main` sürümü) | Stop | DEĞİŞMEZ (yalnız regresyon-doğrulanır) | Executable | Untouched | — | Düşük | Mevcut stop testleri |

**Açıkça DOKUNULMAYAN:** `scripts/collector_supervisor.py`, `.env`, `execution/`,
`risk/`, `brain/`, `tools/s34_state_machine_live_executor.py` (CLAUDE.md
guardrail); leverage/ORDER_NOTIONAL/sizing.

**Not (LAUNCH-B1 düzeltmesi):** Bu tablodaki "Mevcut rol" ve "Doğrulama"
sütunlarındaki `main`-tabanlı referanslar (69 test, mevcut launcher davranışı)
yalnız `main @ cdeb9009`'a atıfta bulunur; implementasyon bunları doğrudan
mevcut HEAD üzerinde ARAMAZ — gelecekteki doğrulanmış integration SHA'sında
çalışır (§0.1).

## 6. Test planı (Contract A §6)

### 6.1 Supersession'lar (tam assertion, §6.1)
- `test_38`-`test_42`: mevcut legacy assertion'lar KORUNUR + `effective_mode=
  legacy-full-contract`, aynı eksik dep → exit 10, sıfır mutasyon.
- `test_27b`: live default-off KORUNUR + no-mode/açık-modlar live başlatmaz +
  `-Mode live-execution`→exit 21/INVALID_REQUEST + presence asla
  `live_execution_authorized=true` yapmaz.

### 6.2 Yeni yapısal testler (modül-import, subprocess değil)
1. 12 canonical rol tam mevcut (manifest = `start_eclipse.ps1` `$roleDeps`
   keys **VE** bağımsız sabit-liste ile çapraz-doğrulanır — LAUNCH-L1).
2. Her rol tam bir kez eşlenir (çift/eksik → FAIL).
3. İcat edilmiş rol yok (taksonomi ⊆ canonical set).
4. Kategori sayıları 2/2/3/3/1/1 (bağımsız sabit değerlerle, üretim
   modülünden türetilmeden).
5. `liquidation_silence_scheduler` → yalnız `SCHEDULER_GATED`.
6. `$roleDeps` ↔ taksonomi bidirectional invariant (LAUNCH-L1): her iki
   yönde eksik/çift/case-farklı rol → modül self-check FAIL.
7. Modül import-failure fail-closed (LAUNCH-M2): eksik modül, malformed
   modül, yanlış export yüzeyi → hepsi exit 20/`INTERNAL_FAILURE`/sıfır
   mutasyon; inline-fallback yok (statik: launcher kaynağında ikinci
   taksonomi tanımı yok).
7-H2. **`data-collection` metadata ≠ executable-destek (LAUNCH-H2):**
   `$EclipseModeContracts` token'ı metadata olarak taşısa bile,
   `Resolve-EclipseModeContract('data-collection')` onu **dispatch-edilebilir
   bir mod sözleşmesine ÇÖZMEZ** (yapısal: implementasyon-yetkili mod-seti =
   {`core-infrastructure`, `shadow-observation`, `diagnostics`}, bağımsız
   sabit-listeyle doğrulanır; `data-collection` bu sette YOK). Bu test,
   token'ı "implementasyon-yetkili" olarak SAYMAZ; yalnız metadata-varlığı ile
   executable-desteğin ayrıldığını kanıtlar.

### 6.3 Yeni davranışsal testler (production-path, subprocess)
8. legacy scheduler davranışı (dosya eksik→10; switch yok→başlatmaz; owned korunur).
9. açık mod + `-EnableLiquidationSilenceScheduler` → exit 21.
10. `SKIPPED_BY_MODE` degrade/fail etmez, exit'i etkilemez.
11. absent optional → DEGRADED + exit 0 (yalnız required ok; `degraded_reason`
    doğru; `optional_unavailable` dolu; `optional_failures` boş).
12. partial optional availability → DEGRADED + `PARTIAL_OPTIONAL_AVAILABILITY`.
13. `ALREADY_OWNED` → available sayılır.
14. attempted optional `START_FAILED` → exit 12 (asla 0).
15. attempted optional `OWNERSHIP_CONFLICT` → exit 12.
16. attempted-fail asla exit 0'a normalize edilmez (mutasyon-kanıtı).
17. `optional_unavailable` vs `optional_failures` ayrı listeler.
18. `degraded_reason` yalnız {NONE, NO_OPTIONAL_ROLE_AVAILABLE,
    PARTIAL_OPTIONAL_AVAILABILITY}.
19. `-Mode live-execution` → exit 21 + reserved mesaj.
20. hiçbir production path live'ı yetkilendiremez.
21. **`data-collection` scope-fidelity + non-dispatch (LAUNCH-H2 — eski
    "recognized→exit-10" test-21 KALDIRILDI):** mevcut implementasyonun
    (a) `data-collection` için bir execution branch EKLEMEDİĞİ (statik:
    launcher kaynağında data-collection'a özgü dispatch/başlatma yolu yok);
    (b) `-Mode data-collection` verildiğinde HİÇBİR rol başlatmadığı / hiçbir
    proses-task mutasyonu yapmadığı (davranışsal: sıfır mutasyon);
    (c) `$roleDeps`'te data-collection'ı SESSİZCE açan bir giriş olmadığı;
    (d) taksonomi/import fallback'inin modu AÇMADIĞI;
    (e) `Resolve-EclipseModeContract`'ın data-collection'ı dispatch-edilebilir
    bir moda ÇÖZMEDİĞİ (metadata-varlığı ≠ executable-destek);
    (f) davranışın fail-closed kaldığı.
    **Test, pre-authorization için kesin exit kodunu (10 veya 21) HARD-CODE
    ETMEZ** — o davranış §9/O-A1 gereği operatör-clarification'a kadar
    bloklu; test yalnız "dispatch/branch/mutasyon YOK + fail-closed"
    invariant'ını doğrular. Bir determinizm uğruna yetkilendirilmemiş yeni
    bir davranış hard-code EDİLMEZ.
22. Mod-parse feasibility (LAUNCH-M1, §4.3a testleri 1-6): no-mode, tek
    geçerli mod, tekrarlı `-Mode`, virgül/dizi-benzeri çoklu değer, eksik
    mod değeri, bilinmeyen mod adı — hepsi launcher-owned exit 21 + tam JSON
    özetine ulaşır (host-level binding hatası DEĞİL).
23. default no-mode → `effective_mode=legacy-full-contract` + tam legacy davranış.
24. çıktı şeması: 15 alan mevcut, tek karar kaynağı (exit↔status uyumu),
    implementasyon-yetkili açık modlarda (`core-infrastructure`/
    `shadow-observation`/`diagnostics`). `data-collection` bu pakette
    implemente edilmediği için bir "başarılı-recognition" şema örneği olarak
    test EDİLMEZ.
25. stop-side regresyon (tüm roller güvenle ele alınır; test_30c korunur).

### 6.4 Yapısal mutasyon-probeleri (§6.3)
- rolü kategoriden kaldır → 12/12 testi FAIL.
- `$roleDeps`'ten bir rol kaldır ama taksonomide bırak (veya tersi) →
  bidirectional-invariant testi FAIL (LAUNCH-L1).
- scheduler'ı bir açık moda ekle → EXCLUDED_BY_MODE testi FAIL.
- attempted-fail 12→0 → outcome #4 testi FAIL.
- absence→failure 12 → outcome #2 testi FAIL.
- diagnostic-fail exit'e sızdır → summary/exit testi FAIL.
- shadow-fail core-required gibi davran → gate-sıralama testi FAIL.
- reserved live'ın devamına izin ver → reserved testi FAIL.
- LIVE_EXECUTION_GATED dosya-varlığını "başladı" ile eşitle → ayrım testi FAIL.
- mod-parse'ı scalar-param-binding'e geri çevir → feasibility testi 3/4/5
  FAIL veya vacuous hale gelir (LAUNCH-M1 regresyon-kanıtı).
- modül import-failure'ı sessizce yok say / inline fallback ekle → §6.2/7
  testi FAIL (LAUNCH-M2 regresyon-kanıtı).

**LAUNCH-H2 mutasyon-probeleri (data-collection scope-fidelity):**
- plan/kod metnini "data-collection implementasyon-yetkilidir"e geri çevir
  → §6.3 test-21 (c/d/e) + §6.2 test-7-H2 FAIL.
- `data-collection`'ı executable bir dispatcher'a ekle (`Resolve-Eclipse
  ModeContract` onu çözer / launch branch açar) → test-21 (a/b/e) FAIL.
- `$roleDeps`'e data-collection'ı mevcut branch'ı SESSİZCE açacak şekilde
  ekle → test-21 (c) FAIL.
- eski "recognized→exit-10" test-21 beklentisini geri getir → scope-fidelity
  test-21 ile ÇELİŞİR (yeni test hard-code exit-code'u reddeder), FAIL.
- §3.2.2 exit-10 dilini "yeterli implementasyon yetkisi" gibi ele al (modu
  implemente et) → test-21 (a/b) + traceability sonucu FAIL.
- operatör-clarification gate'ini (§9/O-A1) kaldır → statik: O-A1 unresolved
  item mevcut olmalı; yoksa gate-varlık testi FAIL.
- pre-authorization'da SESSİZCE exit 21 seç (contract desteği olmadan) →
  test-21 "kesin exit kodu hard-code EDİLMEZ" invariant'ı FAIL.
- pre-authorization'da (mod yetkilendirilmeden) SESSİZCE exit 10 seç →
  aynı test-21 invariant'ı FAIL.
- taksonomi vocabulary üyeliğini executable-destekle eşitle → §6.2 test-7-H2
  FAIL (metadata ≠ dispatch).

**pytest guardrail:** çağrı başına ≤2 test dosyası, `--basetemp` scratchpad,
`-p no:cacheprovider` (CLAUDE.md). Paralel PS/Python prosesi YOK.

## 7. Backward compatibility beklentileri
- Mevcut launcher-safety paketi **68 test tanımı / 69 collected case** (1
  `parametrize` bir def'i 2 collected case'e genişletir; `main @ cdeb9009`).
  Bu sayı gelecekteki integration SHA'sında **yeniden ölçülmelidir** (§0.1).
  Bu paketten hiçbiri sessizce bozulmaz (yalnız §6.1'de adlandırılan 2
  supersession genişletilir).
- `-DryRun` sıfır mutasyon korunur; task-mutation-yok korunur (test_30c).
- RR-01 unknown-task fail-closed korunur.
- Foreign-owned proses/task korunur.

## 8. A-INFO gözlemleri (opsiyonel kalite iyileştirmeleri)
- **A-INFO-1:** `overall_status`↔`exit_code` eşlemesini modülde tek tablo/harita
  olarak sunmak (davranış değişmez, netlik artar). Opsiyonel.
- **A-INFO-2:** `ALREADY_OWNED`'ın mevcut `ALREADY_RUNNING`/`TASK_OWNED` durumlarına
  eşlemesini açık dokümante etmek. Opsiyonel.
Bunlar bloklamaz; implementer isterse ekler.

## 9. Açık sorular — düzeltilmiş kararlar (bağımsız rereview doğrulamalı)

- **O-A1 — `DATA_COLLECTION_PREAUTHORIZATION_BEHAVIOR_REQUIRES_OPERATOR_
  CONTRACT_CLARIFICATION` (LAUNCH-H2, UNRESOLVED — bağımsız rereview
  doğrulamalı):** Birinci corrective'in O-A1 kararı ("data-collection
  recognized + eksik-dep→exit-10 mevcut yetkilendirilmiş davranış; hard-reserve
  amendment gerektirir") **GEÇERSİZ KILINDI** — bağımsız rereview'da LAUNCH-H2
  (HIGH) olarak tespit edildi. Dondurulan olgular:
  - Contract A **§3.2.2** eksik DATA bağımlılıklarını exit-10 preflight-ailesi
    diliyle *tanımlar*.
  - Contract A **§3.2.2 scope cümlesi** ("Bu mod implementasyon-yetkilendirilmiş
    DEĞİLDİR (§8 kapsam dışı)") **ve §8** ("`data-collection` modu — bileşenleri
    kendi bounded-implementation zincirlerini tamamlamadıkça
    implementasyon-yetkilendirilmemiştir") modu implementasyon-DIŞI dondurur.
  - Bu klozlar corrective-planning aktörüne modu implemente etme veya yeni bir
    "mevcut davranış" (exit-10 VEYA exit-21) seçme yetkisi **VERMEZ**.
  - **Operatör-clarification veya ayrıca kabul edilmiş bir contract-amendment
    var olana kadar, Launcher implementasyon planı `data-collection` branch'ini
    tamamen DIŞLAR.**
  - Bir kullanıcının yetkilendirmeden önce `-Mode data-collection` vermesinin
    kesin davranışı **bu planla dondurulmaz** (exit kodu seçilmez).
  - Gelecekteki çözüm **sıfır mutasyon + machine-readable fail-closed**
    davranışını korumalıdır.
  - **Burden:** §8 modu ZATEN kapsam-dışı dondurduğundan, varsayılan pozisyon
    withhold'dur; modu yetkilendirmek (reserve etmemek) ek clarification/amendment
    gerektirir — tersi DEĞİL. Planlama fazı kesin bir exit kodu İCAT ETMEZ.
- **O-A2 (RETAINED, review'da doğrulandı):** Yeni modülün tam yolu/adı —
  `scripts/eclipse_launcher_taxonomy.psm1`, `$PSScriptRoot`-relative pure
  import + exit-20 fail-closed import semantiği ile (§2.1, LAUNCH-M2
  düzeltmesi). Kaynak-inceleme bu konumu geçersiz kılmadı; RETAINED.

## 10. Bu workstream'in gate durumu

Bağımsız-rereview'dan ÖNCE implementasyon YASAK. Migration/canonicalization
kapılarına bağımlı DEĞİL (core/shadow/diagnostics/legacy tarafları bağımsız
ilerleyebilir), ancak:
- **`data-collection` modu bu Launcher paketinden TAMAMEN DIŞLANMIŞTIR**
  (Contract A §3.2.2/§8 freeze, LAUNCH-H2). Mevcut paket bu mod için hiçbir
  branch/dependency-map/preflight/dispatch eklemez. Modun gelecekte planlanması
  `ComponentChainsAccepted ∧ OperatorAuthorization ∧
  ApplicableContractClarification` (§4.3 gelecek koşulu + §9/O-A1) gerektirir;
- implementasyonun kendisi §0.1'de dondurulan gelecekteki doğrulanmış
  integration SHA'sını gerektirir — mevcut HEAD veya `main` üzerinde
  DOĞRUDAN çalışmaz.

Bkz. sequencing belgesi (`LAUNCHER_BOOKTICKER_IMPLEMENTATION_SEQUENCING_AND_
GATES_V1.md`) §0/§1/§11 için birleşik integration-base ve sıralama modeli.

Bu belge implementasyon, migration, deployment veya runtime aksiyonu
yetkilendirmez. Bir sonraki adım: genuinely bağımsız, taze-context bir
rereview.
