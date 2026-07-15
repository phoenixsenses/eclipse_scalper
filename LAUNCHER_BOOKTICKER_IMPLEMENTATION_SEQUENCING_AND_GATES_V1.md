# LAUNCHER_BOOKTICKER_IMPLEMENTATION_SEQUENCING_AND_GATES_V1

**Tarih:** 2026-07-15 · **Yazan:** Claude Sonnet 5 (bounded implementation-planning actor; bu revizyon: LAUNCH-H2 bounded corrective-planning actor) · **Durum:** `IMPLEMENTATION_SEQUENCING_AND_GATES_OPERATOR_ACCEPTED`

> Bu belge, iki plan belgesini (`LAUNCHER_ROLE_TAXONOMY_IMPLEMENTATION_PLAN_V1.md`
> = Workstream A; `BOOKTICKER_COLLECTOR_CANONICALIZATION_IMPLEMENTATION_PLAN_V1.md`
> = Workstream B) sıralama, kapı-modeli, risk-register ve acceptance-modeliyle
> birleştirir. **Hiçbir implementasyon yapılmadı.** Kapı: fresh-context bağımsız
> Opus review.
>
> **Bu belge, ilk taslağın genuinely bağımsız fresh-context Opus review'unda
> `BOTH_IMPLEMENTATION_PLANS_REREVIEW_REQUIRED` verdiktiyle döndüğü paylaşılan
> `SEQ-H1` (HIGH) ve `SEQ-M1` (MEDIUM) bulgularına yanıt veren bir
> corrective'tir.** §0 düzeltme geçmişini kaydeder.
>
> **LAUNCH-H2 cross-reference düzeltmesi (bu revizyon):** Bağımsız rereview
> Launcher planında `LAUNCH-H2` (HIGH) tespit etti — birinci corrective
> `data-collection`'ı yanlışlıkla "implementasyon-yetkili" ilan etmişti. Bu
> belgenin `data-collection`'a atıfta bulunan cross-reference/gate/acceptance
> satırları, Launcher düzeltmesiyle tutarlı olacak şekilde güncellendi (modun
> bu paketten tamamen DIŞLANMASI): §1 kritik-bağlantı + tablo, §2 adım 11 +
> sıra-gerekçesi, §4 acceptance kriterleri, §5 R29, §6 O-A1. `SEQ-H1`/`SEQ-M1`
> kararları ESASA dokunulmadan korundu. Bu corrective KENDİ KENDİNİ KABUL
> ETMEZ.
>
> **Operatör kabulü (2026-07-15):** Genuinely bağımsız, taze-context Opus
> rereview `BOTH_IMPLEMENTATION_PLANS_INDEPENDENT_REREVIEW_ACCEPTED`
> verdiktiyle döndü (LAUNCH-H2 tam düzeltilmiş; yeni BLOCKER/HIGH/MEDIUM yok;
> 12 önceki kapanış regresyonsuz doğrulandı). Operatör bu üç plan belgesini
> **2026-07-15'te kabul etmiştir** — bu, aşağıdaki §3 gate-modelinin
> `PLAN_ACCEPTED` durumuna karşılık gelir; zincirin ilerideki (IMPLEMENTATION_
> ACCEPTED SONRASI) `OPERATOR_ACCEPTED` durumu İLE KARIŞTIRILMAMALIDIR (§3,
> "ASLA tek statüye birleştirilmez"). Kabul implementasyon,
> `INTEGRATION_BASE_ESTABLISHED`, migration, aday-canonicalization, runtime
> probe, deployment veya activation YETKİLENDİRMEZ; `data-collection` modu
> implementasyon-dışı KALIR; O-A1 ÇÖZÜLMEMİŞ ve non-bypassable kalır.

Anchor: branch `codex/data-layer-fallback-cleanup`, HEAD `c6cf1451`, main
`cdeb9009`. Contract A/B `OPERATOR_ACCEPTED`. **HEAD ve main birbirinin atası
DEĞİLDİR** (merge-base `74468c87` = HEAD'in ebeveyni) — bkz. §0.1.

---

## 0. Düzeltme geçmişi (bu corrective)

Kontrol eden bağımsız review: `BOTH_IMPLEMENTATION_PLANS_REREVIEW_REQUIRED`
(genuinely bağımsız, taze-context Opus reviewer; sıfır mutasyon; ağaç review
boyunca donmuş).

| Bulgu | Şiddet | Prior metin | Bu corrective'te ne değişti | Etkilenen bölüm |
|---|---|---|---|---|
| SEQ-H1 | HIGH | Belge, `main`'in kod temeli ile HEAD'in governance durumunu tek bir uygulanabilir taban gibi ele alıyordu; hiçbiri diğerini içermiyor (merge-base `74468c87`) | Yeni §0.1: birleşik integration-base gate modeli — hem Launcher hem BookTicker planlarındaki §0.1 ile TUTARLI, tek kaynak burada dondurulur ve her iki plan oraya atıfta bulunur | §0.1 |
| SEQ-M1 | MEDIUM | §2 numaralı sıra, adım 3'te (BookTicker canonical modül) migration'ın (adım 5-6) onu BLOKLADIĞINI ima ederken, aynı adımda migration'dan ÖNCE yerleştiriyordu — içsel çelişki | §2 düzeltildi: kaynak-yazımı (collector authoring + disposable-fixture unit test) migration'ı BEKLEMEZ; yalnız migration'ın ürettiği gerçek şemaya karşı entegrasyon + `data-collection` runtime-açılışı migration'a BLOKLUDUR. Numaralı sıra ve gate haritası bu ayrımı yansıtacak şekilde yeniden yazıldı | §2, §3 |

Ayrıca risk register'a (§5) `SEQ-H1`/`SEQ-M1` ile ilişkili ve Launcher/
BookTicker corrective'lerinden (LAUNCH-B1/H1/M1/M2/L1, BOOK-B1/H1/M1/M2/L1/L2)
türeyen yeni riskler eklendi (§5, R21-R31).

Bu corrective kendi kendini kabul ETMEZ. Bir sonraki adım genuinely bağımsız,
taze-context bir rereview'dır.

---

## 0.1 Birleşik implementasyon-tabanı gate modeli (SEQ-H1 düzeltmesi)

**Bulgu (bağımsız review, hem Launcher hem BookTicker planında ayrı ayrı
tespit edildi — LAUNCH-B1/BOOK-B1):** Bu üç belge, `main @ cdeb9009` (kod
temeli: launcher-corrective, 69 test, `CollectorSupervisor` `msvcrt.locking`
modeli) ile `c6cf1451` (governance: Contract A/B kabulü + bu üç plan belgesi)
arasındaki ayrışmayı reconcile etmeden, implementasyonun her ikisine de aynı
anda erişebileceğini VARSAYIYORDU. Doğrulama: `git merge-base --is-ancestor`
her iki yönde de `false` döner; merge-base `74468c87` (HEAD'in ebeveyni).
Somut kanıt: HEAD'deki `start_eclipse.ps1` 415 satır (`$roleDeps`/preflight/
mod/exit-ailesi YOK); `tests/test_start_eclipse_launcher_safety.py` HEAD'de
YOK; HEAD'deki `scripts/collector_supervisor.py` 352 satır, `msvcrt` YOK —
hepsi yalnız `main`'de mevcut.

**Dondurulmuş karar (bu belge tek kaynak; Launcher §0.1 ve BookTicker §0.1
buraya atıfta bulunur, kendi kopyalarını bağımsızca İCAT ETMEZ):**

### Ne `c6cf1451` ne de `cdeb9009` tek başına implementasyon tabanıdır

- `c6cf1451` (mevcut HEAD): operatör-kabul-edilmiş governance + bu üç plan
  belgesini içerir, ama launcher-corrective kodunu VEYA
  `CollectorSupervisor` `msvcrt.locking` modelini İÇERMEZ.
- `cdeb9009` (`main`): launcher-corrective + 69 test + `msvcrt.locking`
  modelini içerir, ama Contract A/B kabulünü VEYA bu üç plan belgesini
  İÇERMEZ.
- **İkisi de tek başına implementasyon için yetersizdir.**

### Gerekli gelecekteki integration-base fazı (bu corrective turunun DIŞINDA)

İmplementasyon yetkilendirilmeden ÖNCE, **ayrı, açıkça yetkilendirilmiş bir
integration-preparation fazı**, izole bir integration branch/worktree
oluşturmalıdır. Bu faz:

1. **Kod temeli:** `main @ cdeb90096bfe7b448384b098094586cab750d5e6`'daki
   launcher-corrective (`$roleDeps`, `Invoke-EclipsePreflight`, `Start-
   OptionalRole`, `Get-EclipseStartFinalResult`, `Exit-EclipseStart`, exit
   ailesi, 69 test) VE `CollectorSupervisor`'ın `acquire_supervisor_lock`/
   `release_supervisor_lock` `msvcrt.locking` modelini içerir.
2. **Governance temeli:** `c6cf1451accf681c910e1a4e173560f55979fa38`'deki
   operatör-kabul-edilmiş Contract A + Contract B + preregistration'lar +
   `OPERATOR_DECISION_QUEUE.md` OD-024/OD-025 kayıtlarını içerir.
3. **Planning paketi:** bu üç implementation-planning belgesini, **yalnız**
   şu üç koşul karşılandıktan SONRA içerir:
   - genuinely bağımsız fresh-context rereview'dan geçmiş (bu corrective'in
     hedeflediği rereview);
   - operatör kabulü almış;
   - kendi bounded governance kapısından (ayrı bir commit/prompt) geçmiş.

**İzin verilen mekanizma:** Gelecekteki integration actor, **ayrı bir prompt
tarafından açıkça yetkilendirilen**, history-preserving bir git işlemi
kullanmalıdır — örn. kabul-edilmiş governance/planning dokümantasyon
commit'lerini canonical `main` temelli izole bir dal üzerine uygulamak (cherry-
pick/rebase/merge gibi bir operasyon, ama seçimi ve yürütülmesi o ayrı
prompt'a aittir). **Bu corrective turu hiçbir birleştirme/cherry-pick/rebase/
reset işlemi seçmez, yürütmez veya yetkilendirmez.**

### Gerekli integration-base doğrulaması

Gelecekteki integration tabanı, kendi **bağımsız integration review**'unda,
implementasyon başlamadan ÖNCE, şunları içerdiğini KANITLAMALIDIR:
- kabul-edilmiş Contract A;
- kabul-edilmiş Contract B;
- kabul-edilmiş implementation plan'lar (bu üç belge, kendi rereview
  zincirinden geçtikten sonra);
- `main`'in mevcut launcher-corrective'i (dosya varlığı + satır numaraları +
  fonksiyon imzaları + 69-test koleksiyon sayısı yeniden doğrulanmış);
- canonical launcher-safety test paketi;
- `main`'in kabul-edilmiş `CollectorSupervisor` `msvcrt.locking` modeli
  (fonksiyon imzaları yeniden doğrulanmış);
- hiçbir ilgisiz implementasyon değişikliği (porcelain diff, `main`'e karşı,
  yalnız beklenen dosyaları göstermeli).

### Yanıltıcı çıpaların kaldırılması

Bu üç belgedeki hiçbir ifade artık implementasyonun doğrudan mevcut HEAD
(`c6cf1451`) üzerinde, entegrasyon-doğrulaması olmadan "kabul edilmiş commit"
adı altında çalışacağını, veya `main @ cdeb9009` üzerinde governance-
reconciliation olmadan çalışacağını İDDİA ETMEZ. `main @ cdeb9009` kaynak
referansları (rol/fonksiyon adları, satır numaraları, test sayıları) yalnız
**kaynak-doğrulama çıpası** olarak kullanılır ve gelecekteki doğrulanmış
integration SHA'sında yeniden doğrulanmalıdır.

---

## 1. Paylaşılan bağımlılıklar ve ayrık ilerleyebilirlik

| Bileşen | A'ya bağlı | B'ye bağlı | Bağımsız ilerler mi | İmplementasyon tabanı |
|---|---|---|---|---|
| Launcher taksonomi helper/modül (A) | — | Hayır | **Evet** (B'den bağımsız) | Gelecekteki doğrulanmış integration SHA'sı (§0.1) |
| Launcher davranış + testler (A) | helper | Hayır | Evet (`data-collection` bu paketten DIŞLANMIŞ, LAUNCH-H2) | Gelecekteki doğrulanmış integration SHA'sı (§0.1) |
| BookTicker lock/schema helper (B) | Hayır | — | **Evet** (A'dan bağımsız) | Gelecekteki doğrulanmış integration SHA'sı (§0.1) |
| BookTicker canonical modül kaynak-yazımı (B) | Hayır | lock+schema helper + **frozen şema sözleşmesi** (migration'ın kendisi DEĞİL) | Evet — migration'ı BEKLEMEZ (SEQ-M1 düzeltmesi, bkz. §2) | Gelecekteki doğrulanmış integration SHA'sı (§0.1) |
| BookTicker canonical modül entegrasyon/runtime-readiness (B) | Hayır | migration KABULÜ | Migration'a bloklu | Migration sub-chain sonrası |
| BookTicker migration (B, ayrı gate) | Hayır | — | AYRI preregistration'a bloklu | Kendi sub-chain'i |
| `data-collection` modu (A) | — | **Contract A §3.2.2/§8 freeze** | **Bu paketten DIŞLANMIŞ (LAUNCH-H2)** | Gelecek paket: ComponentChains∧OperatorAuth∧ContractClarification |

**Kritik bağlantı (LAUNCH-H2 düzeltmesi):** Contract A'nın `data-collection`
modu, Launcher planı §4.3/§9-O-A1'de düzeltildiği gibi Contract A §3.2.2 ve §8
tarafından **implementasyon-DIŞI dondurulmuştur** ve mevcut Launcher paketinden
**tamamen DIŞLANMIŞTIR** (birinci corrective'in "recognized ve
implementasyon-yetkili" iddiası GERİ ALINDI). Mod ne bu pakette implemente
edilir ne de bir "runtime-açılışı adımı" olarak sıralanır; gelecekte planlanması
`ComponentChainsAccepted ∧ OperatorAuthorization ∧ ApplicableContractClarification`
gerektirir (Launcher §4.3 gelecek koşulu + §9/O-A1). Launcher A'nın geri kalanı
(core-infrastructure/shadow-observation/diagnostics/legacy/live-reserved) B'den
BAĞIMSIZ ilerler.

## 2. Önerilen implementasyon sırası (kaynak-türetilmiş gerekçe, SEQ-M1 düzeltmesi)

**Düzeltme notu:** önceki taslak adım 3'te ("BookTicker canonical modül")
migration'a "bağlı" diyordu (§1 eski satırı) ama aynı zamanda onu migration
adımlarından (5-6) ÖNCE sıralıyordu — içsel çelişki. Bu artık ayrıştırılmıştır:
**kaynak-yazımı ile disposable-fixture unit test migration'ı BEKLEMEZ**;
yalnız gerçek-şema **entegrasyonu** ve runtime-açılış migration kabulünü
bekler. Aşağıdaki sıra bu ayrımı yansıtır:

0. **(Bu corrective turunun dışında, ayrı yetkilendirme) Integration-
   preparation fazı** — §0.1'de dondurulan doğrulanmış integration SHA'sının
   oluşturulması + bağımsız integration review. Bu, aşağıdaki adım 1'den
   ÖNCE tamamlanmış OLMALIDIR.
1. **Paylaşılan saf helper'lar + yapısal sözleşmeler** — A taksonomi modülü +
   B lock/schema helper'ları (saf, bağımlılıksız, en düşük risk, test-
   edilebilir; migration'ı BEKLEMEZ).
2. **BookTicker lock/path/schema-inspection helper'ları** (B) — concurrency ve
   B-LOW-1/B-LOW-2 çekirdeği; canonical modülden önce izole doğrulanır;
   migration'ı BEKLEMEZ (schema-compat helper salt-okunur PRAGMA inceleme
   kodudur, gerçek migrated şemaya ihtiyaç duymaz — disposable fixture
   yeterli).
3. **BookTicker canonical modül — kaynak-yazımı** (B) — helper'lara ve
   Contract B §B4.1'in **frozen şema sözleşmesine** (DDL metni, migration'ın
   kendisi DEĞİL) karşı yazılır; migration'ı BEKLEMEZ.
4. **BookTicker testleri — disposable-fixture unit testleri** (B) — §6.1-6.7
   tam paket; disposable DB + temsili fixture kullanır; migration'ın var
   olmasını GEREKTİRMEZ.
5. **Ayrı BookTicker migration preregistration** (B, §9) — AYRI
   yetkilendirme; adım 1-4'ten BAĞIMSIZ paralel başlayabilir.
6. **Migration implementasyonu + review** (B) — ayrı gate; kendi sub-chain'i.
7. **BookTicker entegrasyon testleri — gerçek migrated şemaya karşı** (B) —
   **migration kabulüne BLOKLUDUR** (adım 5-6 tamamlanmadan başlayamaz).
8. **Launcher taksonomi helper/manifest** (A) — B'den bağımsız, adım 1 ile
   paralel başlayabilir ama ayrı review.
9. **Launcher davranış implementasyonu** (A).
10. **Launcher testleri** (A).
11. **`data-collection` modu — bu paketin DIŞINDA (LAUNCH-H2).** Bu paketin
    sırasında bir adım DEĞİLDİR. Mod Contract A §3.2.2/§8 tarafından
    implementasyon-DIŞI dondurulur; gelecekte ayrı bir Launcher paketi olarak,
    yalnız `ComponentChainsAccepted ∧ OperatorAuthorization ∧
    ApplicableContractClarification` sonrası planlanabilir (Launcher §4.3/§9).
12. **İzole entegrasyon provası** — yalnız disposable/izole ortam, tüm
    workstream'leri birlikte doğrular.
13. **Bağımsız implementasyon review** (fresh-context).
14. **Operatör kabulü.**
15. **Ayrı deployment/run yetkilendirmesi.**

**Sıra sapması gerekçesi (düzeltilmiş):** A ve B çoğunlukla PARALEL
ilerleyebilir (farklı dosyalar, çakışmayan). **Migration, BookTicker
kaynak-yazımını (adım 3-4) BLOKLAMAZ — yalnız gerçek-şema entegrasyonunu
(adım 7) BLOKLAR.** (`data-collection` bu paketten tümüyle DIŞLANDIĞI için —
LAUNCH-H2 — artık migration'a bloklu bir launcher runtime-açılış adımı
YOKTUR; adım 11 bir sıralama adımı değil, bir dışlama notudur.) Bu, önceki taslağın "canonical modül migration'a bağlı" (§1) ile
"canonical modül migration'dan önce sıralanır" (§2, eski adım 3) arasındaki
çelişkiyi çözer: her iki ifade de doğruydu ama farklı alt-adımlara (kaynak
vs. entegrasyon) atıfta bulunuyordu ve bu ayrım eskiden açık DEĞİLDİ.

## 3. Kapı modeli (gated state transitions — birleştirilemez)

Her workstream için AYRI durum; "implemented / review-accepted / operator-accepted
/ deployed" ASLA tek statüye birleştirilmez:

```
PLAN_DRAFTED
  → [fresh-context bağımsız plan review]
  → PLAN_REREVIEW_REQUIRED (bu turun girişi) veya PLAN_ACCEPTED
  → [corrective planning, gerekirse; bu belge bu adımdadır]
  → [fresh-context bağımsız plan REREVIEW]
  → PLAN_ACCEPTED                                  ← MEVCUT DURUM (operatör kabulü 2026-07-15)
  → [ayrı integration-preparation yetkilendirmesi — §0.1]
  → INTEGRATION_BASE_ESTABLISHED (izole worktree, bağımsız integration-review ile doğrulanmış)
  → [ayrı implementation yetkilendirmesi]
  → IMPLEMENTED (doğrulanmış integration worktree üzerinde)
  → [fresh-context bağımsız implementation review]
  → IMPLEMENTATION_ACCEPTED
  → [operatör sign-off]
  → OPERATOR_ACCEPTED
  → [ayrı deployment/run yetkilendirmesi]
  → DEPLOYED / RUNNING
```

Migration alt-zinciri (B) kendi PLAN→PREREGISTRATION→REVIEW→IMPLEMENT→REVIEW→
ACCEPT kapılarına sahip, AYRI, ve BookTicker'ın adım 7/11 (yukarı §2) geçişini
kendi `OPERATOR_ACCEPTED` durumuyla açar.

**Yeni kapı (SEQ-H1 düzeltmesi):** `INTEGRATION_BASE_ESTABLISHED`,
`PLAN_ACCEPTED` ile `IMPLEMENTED` arasına eklenmiştir — hiçbir implementasyon,
§0.1'de dondurulan doğrulanmış integration SHA'sı olmadan `IMPLEMENTED`
durumuna geçemez.

## 4. Acceptance kriterleri (ayrı — birleştirilmez)

| Faz | Tamamlanma kriteri |
|---|---|
| Planning paketi (bu corrective) | 3 plan belgesi (corrected) + risk-register + file-map + test-plan + B-LOW-1/B-LOW-2 tasarım-kapanışı (sıkılaştırılmış) + migration-ayrımı (SEQ-M1 düzeltilmiş) + integration-base gate modeli (§0.1); bağımsız REREVIEW'a hazır (bu belge) |
| Integration-base fazı (§0.1, ayrı yetkilendirme) | doğrulanmış integration SHA'sı; bağımsız integration-review ile onaylı; hiçbir ilgisiz değişiklik yok |
| Contract A implementasyonu | §6 tüm testler geçer (mod-parse feasibility + import-fail-closed + **`data-collection` scope-fidelity/non-dispatch** [LAUNCH-H2] dahil; eski data-collection-exit-10 testi KALDIRILDI); 68 test tanımı / 69 collected case korunur (integration SHA'sında yeniden ölçülmüş); 12/12 yapısal + bidirectional invariant; outcome matrisi; exit ailesi; fresh-context review ACCEPTED |
| Contract B implementasyonu | §6 tüm testler geçer (UNC/mapped-drive reddi + isim-bağımsız indeks eşdeğerliği + health_root injection dahil); tek-yazıcı + B-LOW-1 kimlik (local-only); msvcrt (PID-liveness yok, integration SHA'sında yeniden doğrulanmış); iki-saat window; gerçekçi SQLite; sanitize izolasyon; fresh review ACCEPTED |
| BookTicker migration preregistration | ayrı belge; şema/indeks/uyumluluk/yasak-SQL/numara-preflight dondurulmuş; bağımsız review |
| BookTicker migration implementasyonu | non-destructive kanıtı; disposable-DB testleri; B-LOW-2 isim-bağımsız tanım-eşdeğerliği fail-closed; review ACCEPTED |
| BookTicker entegrasyon (adım 7) | migration KABULÜ sonrası; gerçek migrated şemaya karşı testler |
| `data-collection` modu (LAUNCH-H2) | Bu paketin DIŞINDA; Contract A §3.2.2/§8 freeze; gelecek paket `ComponentChainsAccepted ∧ OperatorAuthorization ∧ ApplicableContractClarification` gerektirir (Launcher §4.3/§9-O-A1) — mevcut acceptance kapsamında bir faz DEĞİL |
| İzole entegrasyon | yalnız disposable ortam; canlı DB/proses/task DOKUNULMAZ; observability doğrulanır |
| Deployment readiness | AYRI, açık operatör yetkilendirmesi; bu paketin kapsamı DIŞINDA |

## 5. Risk register (SEQ-H1/SEQ-M1 + Launcher/BookTicker corrective'lerinden türeyen yeni riskler eklendi)

| # | Risk | Önleyici kontrol | Tespit kontrolü | Fail-closed davranış | Doğrulama |
|---|---|---|---|---|---|
| R1 | Duplicate collector writer | Tek-yazıcı OS lock (DB-path+tablo kimliği, §B5.2) | 2. instance typed non-zero | Lock alınamazsa çalışmaz | cross-process test (§6.1) |
| R2 | Lock-path aliasing (junction/case, yalnız LOCAL path'ler) | B-LOW-1 realpath+normcase normalizasyon (§7) | kimlik-eşdeğerlik testi | zayıf-norm → test FAIL | B-LOW-1 testleri |
| R3 | Stale-PID varsayımı | PID-liveness YOK; OS auto-release | statik grep (PID-poll yok) | metadata otorite değil | §6.1 PID-liveness testi |
| R4 | Lock geç alınır | Lock DB/WS/health-yayınından ÖNCE (§B5.3, §4a) | sıra testi | geç → test FAIL | §6.1 sıra testi |
| R5 | WS bağlantısı ownership'ten önce | acquire-before-WS invariant | sıra testi | ownership yoksa bağlanmaz | §6.1 |
| R6 | Şema uyumsuzluğu | salt-okunur compat + fail-before-WS (§B4.2) | compat kontrolü | uyumsuz → satır yazmadan sonlan | §6.4 |
| R7 | Aynı-isim/yanlış-tanım indeks | B-LOW-2 tanım-karşılaştırma, isim-bağımsız (§8) | index_xinfo | typed incompatible, mutasyon yok | B-LOW-2 testi |
| R8 | Batch kaybı | commit-sonrası-clear + retry-tükenmede batch-korunur (§B4.7) | retry testi | typed termination, batch tutulur | §6.3 |
| R9 | Retry sonrası duplicate yazım | append-only + atomik commit (§B3/§B4.7) | duplicate/partial testi | belirsiz durum → typed sonlan | §6.3 |
| R10 | Health-state hatası | 10-ardışık eşik; tek hata reconnect ETMEZ; `health_root` seam ile izole (§4a) | ardışık sayaç | eşikte typed fatal | §6.5 |
| R11 | Invalid-message flood | 5/30s rolling window → reconnect (§B4.6) | window sayacı | flush-fail → sonlan | §6.2 |
| R12 | Launcher optional-fail normalizasyonu | outcome matrisi #4 → exit 12 (§3.3) | exit testi | attempted-fail asla exit 0 | A §6.3 |
| R13 | Rol taksonomi drift | tek doğruluk kaynağı + 12/12 self-check + bidirectional invariant (LAUNCH-L1) | yapısal test | eksik/çift → modül fail | A §6.2 |
| R14 | Scheduler duplikasyonu | SCHEDULER_GATED + açık-mod exclusion | exit-21 testi | mod+switch → exit 21 | A §6.3 |
| R15 | Live-execution kazara yetkilendirme | reserved → exit 21 (§3.2.5); 2-CLI+3-env gate korunur | reserved testi | daima INVALID_REQUEST | A §6.3 |
| R16 | Dirty-tree kontaminasyonu | yalnız yetkili dosyalar; unrelated dokunulmaz | porcelain diff | staged-scope kontrolü | bu turda doğrulandı |
| R17 | Aday kazara runtime otoritesi | intake=immutable-evidence; tracked-modül tek-kaynak (§B7.1); main-tree candidate asla staged edilmez (BOOK-L1) | statik: intake import yok + candidate `??` durumu | intake import / candidate staging → test FAIL | §6.6 |
| R18 | Test'te production DB erişimi | sanitize ortam; production path çözülemez | fs-izleme | production path → test FAIL | §6.5 |
| R19 | Migration canlı DB'yi mutasyona uğratır | production read-only; disposable-DB testleri | commit-scope | canlı DB'ye çalıştırma YOK | §9 migration gate |
| R20 | Migration numarası stale/collision | preflight'ta seçim + abort-on-collision (§B4.8) | ledger inceleme | collision → abort | migration gate |
| **R21** | **Yanlış/divergent implementasyon tabanı** (SEQ-H1) — implementer HEAD veya main'i doğrudan taban alır, ne governance ne launcher-corrective'in tamamına sahip olur | §0.1 birleşik integration-base gate modeli; `INTEGRATION_BASE_ESTABLISHED` ayrı kapı | bağımsız integration-review; porcelain diff `main`'e karşı | doğrulanmış integration SHA yoksa implementasyon BAŞLAMAZ | §0.1 integration-review kriterleri |
| **R22** | **Mapped-drive/UNC alias → dual writer** (BOOK-H1) | UNC/mapped-drive typed fail-closed reddi (§7); yalnız local path desteklenir | `GetDriveType` kontrolü + statik test | reddedilir, sıfır mutasyon | §6.1 UNC/mapped-drive testleri |
| **R23** | **Health output production `logs/health`'e kaçar** (BOOK-M2) | `health_root` injection seam; test her zaman disposable root geçirir | production-path mtime/varlık kontrolü | production path'e hiç yazılmaz | §6.5 health_root testi |
| **R24** | **Main-tree untracked candidate, canonical Python import'unu gölgeler** | modül-resolution testi canonical integration-worktree yolunu doğrular; candidate hiç staged edilmez | `sys.modules[...].__file__` kontrolü | yanlış modül yüklenirse test FAIL | §6.6 modül-resolution testi |
| **R25** | **PowerShell modül önbelleği / stale modül durumu** | her pytest-subprocess çağrısı taze bir PowerShell host başlatır (`powershell -NoProfile`); modül `$PSScriptRoot`-relative import edilir, global session state'e GÜVENMEZ | subprocess-izolasyon testi | stale modül tespit edilirse test FAIL | Launcher §6.2/7 |
| **R26** | **Scheduled Task farklı çalışma dizininden çalıştırır, cwd-relative path'ler kırılır** | `$PSScriptRoot`-relative modül import (LAUNCH-M2); launcher zaten `Set-Location -LiteralPath $RepoRoot` yapıyor (mevcut davranış) | non-repository-cwd test | cwd farklıysa modül yine de bulunur; bulunamazsa fail-closed exit 20 | Launcher §6.2/§6.3 |
| **R27** | **Lock alındıktan sonra DB dosyası değiştirilir/taşınır** (kimlik-değişmezlik ihlali) | §7 post-creation kimlik-değişmezlik doğrulaması: DB oluşturulduktan sonra kimlik yeniden hesaplanır ve pre-creation kimlikle karşılaştırılır | kimlik-karşılaştırma testi | fark tespit edilirse fail-closed | §6.1 kimlik-değişmezlik testi |
| **R28** | **Şema, compat-inceleme sonrası ama collector başlamadan önce değişir** (TOCTOU) | schema-compat inceleme + lock alımı + WS bağlantısı arasındaki pencere minimize edilir: lock ÖNCE alınır, compat inceleme lock ALTINDA yapılır (§B5.3 sıralaması) | sıra testi | compat-sonrası şema değişimi bir sonraki collector restart'ında yeniden tespit edilir (her başlangıçta re-inspect, cache edilmez) | §6.4 |
| **R29** | **Collector/migration versiyon çarpıklığı** (collector migration'ın henüz oluşturmadığı bir şemayı bekler) | collector yalnız frozen §B4.1 DDL'ine karşı YAZILIR (kaynak-yazımı migration'ı beklemez, §2); ama collector'ın **entegrasyon/runtime-açılışı** migration KABULÜNE bloklu (§2 adım 7; launcher `data-collection` modu LAUNCH-H2 ile bu paketten tümüyle dışlandığından ayrı bir launcher runtime-açılış adımı yoktur) — collector hiçbir zaman migration'sız gerçek DB'ye karşı ÇALIŞTIRILMAZ | gate-sıralama testi (entegrasyon adımı migration-kabul-öncesi tetiklenemez) | runtime-açılış migration kabulü olmadan engellenir | §9, sequencing §2/§3 |
| **R30** | **WAL/checkpoint etkileşimi retry/lock-timeout ile çakışır** | mevcut `journal_mode=WAL`/`synchronous=NORMAL`/`busy_timeout=30000` (adaydan KORUNUR, §B2.1) + §B4.7 gerçekçi retry sözleşmesi bu ayarlarla birlikte test edilir (mock değil, gerçek sqlite3 WAL) | gerçek-WAL lock/busy injection testi | retry tükenmesi → typed sonlan (checkpoint'e güvenmeden) | §6.3 |
| **R31** | **Entegrasyon testleri yanlışlıkla production konfigürasyonuna çözümlenir** | sanitize subprocess ortamı (§6.5/§6.6); tüm test path'leri açıkça disposable tmp root'a geçirilir; hiçbir test default/env-fallback'e GÜVENMEZ | fs-izleme + env-diff testi | production path'e erişim → test FAIL | §6.5/§6.6 |

## 6. Bu turda çözülmeyen açık sorular (bağımsız review'a taşınır — düzeltilmiş kararlar)

- **O-A1 (LAUNCH-H2 ile YENİDEN AÇILDI — UNRESOLVED):** Birinci corrective'in
  "`data-collection` recognized + eksik-dep→exit-10; amendment yalnız
  hard-reserve için" kararı GEÇERSİZ KILINDI. Contract A §3.2.2 (scope cümlesi)
  ve §8 modu implementasyon-DIŞI dondurur; mod bu paketten DIŞLANMIŞTIR. Kesin
  pre-authorization davranışı (exit kodu) dondurulmaz; sıfır-mutasyon +
  fail-closed korunur; disposition operatör-clarification'a taşınır. Bkz.
  Launcher planı §9/O-A1 (`DATA_COLLECTION_PREAUTHORIZATION_BEHAVIOR_REQUIRES_
  OPERATOR_CONTRACT_CLARIFICATION`) + §4.3.
- **O-A2 (RETAINED):** `scripts/eclipse_launcher_taxonomy.psm1`,
  `$PSScriptRoot`-relative pure import + exit-20 fail-closed semantiği ile.
  Bkz. Launcher planı §9.
- **O-B1 (düzeltildi):** mapped-drive/UNC desteklenmez, typed fail-closed
  reddedilir; yalnız local path'ler desteklenir. B-LOW-1 açık advisory olarak
  kalır (implementasyon+test bağımsız kabul edilene kadar). Bkz. BookTicker
  planı §11.
- **O-B2 (düzeltildi):** indeks uyumluluğu tanım-tabanlı, isim-bağımsız.
  B-LOW-2 açık advisory olarak kalır. Bkz. BookTicker planı §11.
- **O-B3 (sıkılaştırıldı):** canonical modül sözleşmeden yazılır; main-tree
  candidate asla staged edilmez; byte-kopya-değil-yeniden-yazım ilkesi
  açık. Bkz. BookTicker planı §11.

Bunlar plan-tercihleridir; bağımsız rereview kabul/ret/düzeltme verir.

## 7. Governance ve scope notları
- Accepted Contract A/B **DEĞİŞTİRİLMEDİ** (bu corrective turunda da
  değiştirilmedi; bağlayıcı konvansiyon gerektirmedi).
- `OPERATOR_DECISION_QUEUE.md` **DEĞİŞTİRİLMEDİ** (bu corrective turunda da
  değiştirilmedi) — OD-024/OD-025 zaten "sonraki kapı = bounded
  implementation-planning" diyor; bağlayıcı bir planning-start-row kuralı
  yok, implementation başlamadı. Yeni OD ID oluşturulmadı.
- Yeni migration numarası ATANMADI/rezerve EDİLMEDİ; migration ledger
  DOKUNULMADI (bu corrective turunda da).
- Bu 3 plan belgesi **commit EDİLMEDİ** (bu corrective turunda da uncommitted
  bırakıldı; bağlayıcı "planning drafts must be committed before review"
  kuralı yok; emsal: preregistration'lar da review'a kadar untracked kaldı).
  Bağımsız rereview untracked planning artefaktları üzerinde çalışır.
- **Bu corrective turu hiçbir merge/cherry-pick/rebase/reset/checkout/
  integration-worktree-oluşturma işlemi yapmadı** — §0.1'de dondurulan
  integration-base fazı tamamen ayrı, gelecekteki bir yetkilendirmeye aittir.

## 8. Sonraki yetkilendirilmiş aksiyon

**Fresh-context bağımsız Opus REREVIEW** bu düzeltilmiş planning paketini (3
belge) inceler. Yalnız plan-kabulünden SONRA, önce §0.1'deki integration-base
fazı AYRI yetkilendirmeyle gerçekleşir; ardından, yalnız doğrulanmış
integration SHA'sı üzerinde, AYRI implementation yetkilendirmesiyle
Workstream başlar. Bu belge hiçbir implementasyon/migration/integration/
deployment/runtime aksiyonu yetkilendirmez.
