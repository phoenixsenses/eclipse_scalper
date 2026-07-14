# LAUNCHER_ROLE_TAXONOMY_PREREGISTRATION_V1

**Tarih:** 2026-07-14 · **Yazan:** Claude Sonnet 5 (bounded preregistration operator) · **Durum:** `LAUNCHER_ROLE_TAXONOMY_PREREGISTRATION_OPERATOR_ACCEPTED` (operatör kabulü 2026-07-15 — bkz. §12)

> Bu belge yalnızca gelecekteki bounded implementation + bağımsız review zinciri
> için bir **sözleşme donduruşudur** (contract freeze). Preregistration ≠
> acceptance. Hiçbir kod, test, launcher, Scheduled Task, proses, PID dosyası,
> veritabanı veya runtime yapılandırması bu belgeyle değiştirilmedi.
> Implementation, ayrı, açıkça yetkilendirilmiş bir batch içinde, izole bir
> temiz worktree'de gerçekleşmelidir; bu belge kabul edilmiş sayılmaz — kabul,
> genuinely bağımsız bir rereview'dan sonra ayrı bir verdict ile gelir.
>
> Bu sözleşme yalnız `main` (`cdeb90096bfe7b448384b098094586cab750d5e6`)
> üzerindeki mevcut launcher-safety corrective davranışının bir *revizyon
> teklifidir*; corrective'in kendisini geçersiz kılmaz veya reddetmez.
>
> **Sürüm zinciri:** ilk preregistration → bağımsız review
> (`CORRECTIVE_REQUIRED`, A-F1..A-F6) → birinci corrective → bağımsız rereview
> (`REREVIEW_REQUIRED`, NA-1..NA-5) → **bu ikinci corrective**. §9 (birinci
> corrective) ve §10 (ikinci corrective) düzeltme geçmişleri her bulgunun tam
> olarak nasıl kapatıldığını kaydeder. Bu düzeltme kendi kendini kabul etmez —
> genuinely bağımsız, taze-context bir rereview gerekir; bu belgedeki tüm
> kapanış ifadeleri **"corrective operator claims closed; pending independent
> rereview"** statüsündedir.

---

## 0. Kaynak temeli

Bu belge, `SIX_MODULE_CANONICALIZATION_AUDIT` özetine değil, doğrudan canonical
kaynağa dayanır:

- `start_eclipse.ps1` @ `main` (`cdeb9009`)
- `stop_eclipse.ps1` @ `main` (`cdeb9009`)
- `tests/test_start_eclipse_launcher_safety.py` @ `main` (68 fonksiyon, 69
  collected test — `test_20_21_22_real_pythonw_enumeration` parametrized, 2 ID
  üretiyor)

## 1. Mevcut kabul edilmiş sözleşme (A1 — re-derived)

> Bu bölüm yalnızca **mevcut** (`main` @ `cdeb9009`) davranışı tanımlar.
> Gelecekteki zorunlu davranış §2-§6'da ayrıca ve açıkça donduruludu; bu iki
> küme birbirine karıştırılmaz.

### 1.1 Deklare edilen roller (`$roleDeps`, satır ~315-327)

| Rol | Path | Mevcut `Required` | Start komutu (argümanlar) | `-EnableLive` veya başka mod kapısı |
|---|---|---|---|---|
| `collector_supervisor` | `scripts\collector_supervisor.py` | `$true` | `-u scripts\collector_supervisor.py --cwd $RepoRoot --symbols $Symbols` | yok |
| `heartbeat_watchdog` | `tools\heartbeat_watchdog.py` | `$true` | `-u -m tools.heartbeat_watchdog --interval-sec 5 --max-age-sec 420 --expect-bookticker` | yok |
| `bookticker_collector` | `data\bookticker_collector.py` | `$true` | `-W ignore -u -m data.bookticker_collector --symbols $Symbols --db-path data/microstructure.db --heartbeat-interval 5` | yok |
| `oi_spot_poller` | `data\oi_spot_poller.py` | `$true` | `-W ignore -u -m data.oi_spot_poller` | yok |
| `s34_shadow_paper_runner` | `tools\s34_shadow_paper_runner.py` | `$true` | `-u tools\s34_shadow_paper_runner.py --loop --interval-sec 60 --regime-filter-enabled ... --quality-gate-enabled ...` | yok |
| `s34_live_chart` | `tools\s34_live_chart.py` | `$true` | `-u tools\s34_live_chart.py --host 127.0.0.1 --port 5050 --no-browser` | yok |
| `s34_v_engine_v02_shadow_mirror` | `tools\s34_v_engine_v02_shadow_mirror.py` | `$true` | `-W ignore -u -m tools.s34_v_engine_v02_shadow_mirror --loop --interval-sec 180` | yok |
| `s34_state_machine_shadow_runner` | `tools\s34_realtime_shadow_runner.py` | `$true` | `-W ignore -u -m tools.s34_realtime_shadow_runner` | yok |
| `s34_state_machine_live_executor` | `tools\s34_state_machine_live_executor.py` | `$true` | `-W ignore -u -m tools.s34_state_machine_live_executor --live --confirm-live-orders --db data/microstructure.db` | **`-EnableLive`** (yoksa `Disable-EclipseRoleIfOwned` çağrılır, süreç asla başlamaz) |
| `liquidation_silence_scheduler` | `tools\liquidation_silence_scheduler.py` | `$true` | `-W ignore -u -m tools.liquidation_silence_scheduler` | `-EnableLiquidationSilenceScheduler` (Task-owned ise `TASK_OWNED`; yoksa disable) |
| `orderflow_chart` | `tools\orderflow_chart.py` | `$true` | `-u tools\orderflow_chart.py --host 127.0.0.1 --port 5051 --no-browser` | yok |
| `s34_replay` | `tools\s34_replay.py` | `$true` | `-u tools\s34_replay.py --host 127.0.0.1 --port 5052 --no-browser` | yok |

### 1.2 Preflight davranışı (yol yoksa)

`Invoke-EclipsePreflight` (satır ~267-333): her rol için `Test-Path` — salt
dosya varlığı, git-tracking'den bağımsız, import/syntax/config doğrulaması
YOK. Herhangi bir `Required=$true` rol eksikse → `requiredRoleFailures`'a
eklenir → `$result.ok=$false` → **hiçbir mutasyondan önce** `exit 10`
(`PREFLIGHT_FAILED`). Bu, mevcut testlerle davranışsal olarak kilitlenmiştir
(`test_02_canonical_checkout_fails_preflight_closed`,
`test_02b_every_declared_role_is_required`,
`test_04_skipped_missing_dependency_cannot_yield_success`).

### 1.3 Preflight sonrası runtime davranışı

Yalnız `collector_supervisor` gerçek bir required-role runtime gate'e
sahiptir (satır ~605-680): Task-owned/Ready/Running → `TASK_OWNED`; Disabled →
`TASK_PRESENT_DISABLED` (fail-closed); foreign/unknown → `OWNERSHIP_CONFLICT`;
yoksa `Start-RegisteredPythonProcess`. Bu rol için
`OWNERSHIP_CONFLICT`/`START_FAILED`/`SKIPPED_MISSING_DEPENDENCY`/
`TASK_PRESENT_DISABLED` → **required-role gate tetiklenir**: tüm diğer roller
`SKIPPED_REQUIRED_ROLE_GATE`, hiçbir optional süreç başlamaz, exit `11`.

Diğer 11 rol `Start-OptionalRole` üzerinden başlar (satır ~530-613): bu
fonksiyon yalnız `$preflight.dependencies[$Role].present` kontrol eder (zaten
preflight'ta garanti edilmiş olmalı) ve `Start-RegisteredPythonProcess`'i
çağırır — kendi ownership/conflict modeli `collector_supervisor` ile
aynıdır (`ALREADY_RUNNING`/`OWNERSHIP_CONFLICT`/`STARTED`/`START_FAILED`), ama
bu roller için conflict/failure yalnız `Get-EclipseStartFinalResult`'ta
**secondary_failures**'a girer → exit `12` (`SECONDARY_ROLE_FAILED`), asla
`11` değil, asla üründe collector_supervisor'ı engellemez.

### 1.4 Exit-kodu ailesi özeti (mevcut, `main` @ `cdeb9009`)

| Aile | Kod | Tetikleyici |
|---|---|---|
| Override reddi | `20` | Override, `-DryRun` olmadan verildi |
| Malformed override | `21` | Override dosyası okunamaz/eksik alan |
| Preflight | `10` | Herhangi bir required rol dosyası eksik (mutasyondan ÖNCE) |
| Required-role | `11` | `collector_supervisor` fail/conflict/disabled-task |
| Secondary-role | `12` | Herhangi bir diğer rol fail/conflict/missing-dep |
| Success | `0` | Tüm roller güvenli (STARTED/ALREADY_RUNNING/TASK_OWNED/DISABLED_BY_POLICY/vb.) |

> Gelecekteki zorunlu exit-kodu ailesi (mod-semantiği dahil) §4 (A-C4)'te
> ayrıca donduruludu — mevcut aile ile çakışmaz, `13` gibi yeni bir kod
> EKLEMEZ.

### 1.5 Mutasyon yüzeyi

Yalnız non-DryRun + preflight-ok sonrası: `logs/`+`logs/pids/` oluşturma,
delegated `stop_eclipse.ps1 -Quiet` çağrısı, `Start-Process` (her rol için),
PID/JSON kayıt dosyaları. `-DryRun`: sıfır mutasyon (test kilidiyle
kanıtlanmış — `test_13`, `test_14`, `test_15`). Hiçbir rol Scheduled Task
oluşturmaz/değiştirmez (`test_30c_no_task_mutation_cmdlets_in_either_script`).

---

## 2. Zorunlu rol taksonomisi (A2 — DONDURULMUŞ, ön-onaylı değil ama BAĞLAYICI)

> **Düzeltme notu (A-F2 birinci corrective + NA-1 ikinci corrective kapanışı):**
> "ön-onaylı değil" yalnızca "genuinely bağımsız bir rereview henüz bunu kabul
> etmedi" anlamına gelir, "implementer bunu değiştirebilir" anlamına GELMEZ.
> **NA-1 düzeltmesi:** birinci corrective yalnızca 11 rolü sınıflandırmıştı;
> `liquidation_silence_scheduler` (12. canonical rol) atlanmıştı ve "tüm 12 rol
> atandı" iddiası GERÇEK DIŞIYDI. Bu ikinci corrective yeni bir `SCHEDULER_GATED`
> kategorisi ekleyerek 12. rolü sınıflandırır ve makine-kontrol-edilebilir bir
> tam eşleme tablosu (§2.1) dondurur.

Canonical launcher'ın (`main` @ `cdeb9009`) tam rol adları kullanılır. **Her
canonical launcher rolü tam olarak bir kategoriye atanır; hiçbir rol
sınıflandırılmamış kalmaz.**

### 2.1 Makine-kontrol-edilebilir rol → kategori eşleme tablosu (TAM, 12/12)

| # | Canonical rol adı | Kategori |
|---|---|---|
| 1 | `collector_supervisor` | `CORE_REQUIRED` |
| 2 | `heartbeat_watchdog` | `CORE_REQUIRED` |
| 3 | `bookticker_collector` | `DATA_REQUIRED_FOR_MODE` |
| 4 | `oi_spot_poller` | `DATA_REQUIRED_FOR_MODE` |
| 5 | `s34_v_engine_v02_shadow_mirror` | `SHADOW_OPTIONAL` |
| 6 | `s34_shadow_paper_runner` | `SHADOW_OPTIONAL` |
| 7 | `s34_state_machine_shadow_runner` | `SHADOW_OPTIONAL` |
| 8 | `s34_live_chart` | `DIAGNOSTIC_OPTIONAL` |
| 9 | `orderflow_chart` | `DIAGNOSTIC_OPTIONAL` |
| 10 | `s34_replay` | `DIAGNOSTIC_OPTIONAL` |
| 11 | `s34_state_machine_live_executor` | `LIVE_EXECUTION_GATED` |
| 12 | `liquidation_silence_scheduler` | `SCHEDULER_GATED` |

**Dondurulmuş kategori sayıları:** `CORE_REQUIRED`=2, `DATA_REQUIRED_FOR_MODE`=2,
`SHADOW_OPTIONAL`=3, `DIAGNOSTIC_OPTIONAL`=3, `LIVE_EXECUTION_GATED`=1,
`SCHEDULER_GATED`=1. **Toplam = 12.** Her rol tam olarak bir kez görünür.

### `CORE_REQUIRED` (2)

- **`collector_supervisor`** — zaten tek gerçek required-role runtime gate;
  yokluğunda talep edilen mod dürüstçe çalışamaz.
- **`heartbeat_watchdog`** — core safety/health supervision'ın bir parçasıdır;
  bookticker heartbeat'ini bekleyip (`--expect-bookticker`) gözlemlenebilirlik
  omurgasını oluşturur; core-infrastructure modunda talep edilmesi gerekir.

### `DATA_REQUIRED_FOR_MODE` (2)

- **`bookticker_collector`**
- **`oi_spot_poller`**

Yalnız açıkça talep edilen `data-collection` modunda required'dır.
`data-collection` modunun kendisi, bileşenleri kabul edilene kadar
implementasyon-yetkilendirilmiş DEĞİLDİR (bkz. §8 kapsam sınırı).

### `SHADOW_OPTIONAL` (3)

- **`s34_v_engine_v02_shadow_mirror`**
- **`s34_shadow_paper_runner`**
- **`s34_state_machine_shadow_runner`**

`shadow-observation` modunda optional'dır; yokluğu core'u asla durdurmaz.
(`s34_v_engine_v02_shadow_mirror` untracked ve 5 ek untracked bağımlılığa
sahiptir — Contract A kapsamı DIŞINDA; sınıflandırma bunu canonical/kabul
edilmiş ilan ETMEZ.)

### `DIAGNOSTIC_OPTIONAL` (3)

- **`s34_live_chart`**
- **`orderflow_chart`**
- **`s34_replay`**

Salt-okunur HTTP dashboard'lar; `diagnostics` modunda optional'dır.

### `LIVE_EXECUTION_GATED` (1)

- **`s34_state_machine_live_executor`**

Yalnız açıkça talep edilen `live-execution` modunda dikkate alınır, ancak bu
mod §3.2.5'te `RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT` olarak dondurulur.
Dosya varlığı TEK BAŞINA onu asla başlatmaz veya yetkilendirmeyi ima etmez.

### `SCHEDULER_GATED` (1) — YENİ (NA-1)

- **`liquidation_silence_scheduler`**

Mevcut launcher'da kendi `-EnableLiquidationSilenceScheduler` switch'i ve
`TASK_OWNED` semantiği olan, canlı çalışan (PID 19084, Scheduled Task
`LiquidationSilenceScheduler`) bir roldür. Davranışı:

- **`legacy-full-contract`'te:** scheduler bağımlılığı, mevcut 12-rol required
  filesystem-presence manifestinin bir parçası KALIR (dosya varlığı doğrulanır,
  eksikse `legacy-full-contract` mutasyondan önce exit `10`); proses başlatma
  mevcut `-EnableLiquidationSilenceScheduler` switch'iyle YÖNETİLİR; switch
  yoksa **bağımlılık varlığı doğrulanır ama scheduler yeniden BAŞLATILMAZ**;
  var olan güvenle sahiplenilmiş task/proses durumu KORUNUR. (Mevcut kabul
  edilmiş uyumluluk davranışı — değişmez.)
- **Her açık modda (`core-infrastructure`, `data-collection`,
  `shadow-observation`, `diagnostics`, `live-execution`):**
  `liquidation_silence_scheduler` = **`EXCLUDED_BY_MODE`** (§3.2 tablosu).
  Açık bir `-Mode` ile `-EnableLiquidationSilenceScheduler`'ın BİRLİKTE
  verilmesi bu sözleşme revizyonunda **geçersizdir** → mutasyondan ÖNCE exit
  `21`. Scheduler'ın açık modlarla kompozisyonu, gelecekteki ayrı, bağımsız-
  review'lı bir amendment ile tanımlanabilir; bu sözleşme böyle bir kompozisyon
  İCAT ETMEZ.

### Kategori değişikliği kuralı

Hiçbir rol, yeni bir preregistration amendment + bağımsız review olmadan
kategoriler arasında taşınamaz. İmplementasyon sırasında **yeni bir launcher
modu EKLENEMEZ** (§3'te dondurulan beş açık mod + `legacy-full-contract`
dışında). **Yeni bir kategori EKLENEMEZ** (§2.1'deki altı kategori dışında).

---

## 3. Mod-spesifik semantik (A3 — DONDURULMUŞ)

### 3.1 Varsayılan (mod bayrağı verilmemiş) invocation — `legacy-full-contract`

Bir invocation açık bir mod argümanı OLMADAN çağrılırsa, bu **mevcut kabul
edilmiş tam-sözleşme fail-closed davranışını AYNEN korur**:

- Tüm 12 mevcut launcher rolü (§1.1 tablosu) tam olarak doğrulanır.
- Herhangi bir mevcut deklare edilmiş bağımlılık eksikse → mutasyondan ÖNCE
  fail-closed (exit `10`).
- `effective_mode=legacy-full-contract` olarak JSON özetinde raporlanır.
- Kısmi/degraded başarı olarak ETİKETLENEMEZ.
- İsim: **`legacy-full-contract`** — sabit, değiştirilemez.
- `legacy-full-contract` YALNIZCA `-Mode` yokluğuyla seçilir; hiçbir açık modla
  BİRLEŞTİRİLEMEZ.

### 3.2 Açık mod seçimi

Gelecekteki implementasyon bir mod seçici sunar, geçici olarak `-Mode
<mode-name>` şeklinde temsil edilir (PowerShell parametre mekaniği
implementasyonun tercihine bırakılır, ama aşağıdaki mod adları ve semantiği
DEĞİŞTİRİLEMEZ). **Tam olarak bir açık mod seçilebilir; modlar birleştirilemez;
tekrarlanan/çoklu mod talebi exit `21` ile fail eder (§3.4).**

| Mod adı | Required roller | Optional roller | Excluded roller |
|---|---|---|---|
| `core-infrastructure` | `collector_supervisor`, `heartbeat_watchdog` | *(yok)* | data, shadow, diagnostic, live-execution, **scheduler** rollerinin tümü |
| `data-collection` | `collector_supervisor`, `heartbeat_watchdog`, `bookticker_collector`, `oi_spot_poller` | *(yok)* | shadow, diagnostic, live-execution, **scheduler** rolleri |
| `shadow-observation` | `collector_supervisor`, `heartbeat_watchdog` | `s34_v_engine_v02_shadow_mirror`, `s34_shadow_paper_runner`, `s34_state_machine_shadow_runner` | data, diagnostic, live-execution, **scheduler** rolleri |
| `diagnostics` | `collector_supervisor`, `heartbeat_watchdog` | `s34_live_chart`, `orderflow_chart`, `s34_replay` | data, shadow, live-execution, **scheduler** rolleri |
| `live-execution` | *(RESERVED — bkz. §3.2.5)* | *(yok)* | — |

Geçersiz/boş/tekrarlanan/uyumsuz mod seçimleri **mutasyondan ÖNCE** `exit 21`
ile fail eder. Bilinmeyen bir mod adı için başka bir moda **implicit fallback
YASAKTIR**. Her açık modda `liquidation_silence_scheduler` `EXCLUDED_BY_MODE`'dur
ve `-EnableLiquidationSilenceScheduler` + açık `-Mode` kombinasyonu exit `21`
verir (§2 SCHEDULER_GATED).

#### 3.2.1 `core-infrastructure`

Başarı yalnızca adlandırılmış core-infrastructure sözleşmesinin karşılandığı
anlamına gelir. **"Full-stack success" olarak TANIMLANAMAZ.**

#### 3.2.2 `data-collection`

Kısmi başarı YOK: `bookticker_collector` veya `oi_spot_poller` canonical/
importable/configured değilse preflight'ta fail eder — sessizce atlanmaz. Bu mod
implementasyon-yetkilendirilmiş DEĞİLDİR (§8 kapsam dışı).

#### 3.2.3 `shadow-observation`

Optional-rol sonuçları §3.3 outcome matrisiyle değerlendirilir. Hiçbir shadow
rolü mevcut/available değilse → `overall_status=DEGRADED`,
`degraded_reason=NO_OPTIONAL_ROLE_AVAILABLE`, exit `0` (yalnız tüm required
roller başarılı/güvenle-sahiplenilmişse). Bir alt küme available ise →
`DEGRADED` + `degraded_reason=PARTIAL_OPTIONAL_AVAILABILITY` mümkündür.

#### 3.2.4 `diagnostics`

Aynı §3.3 outcome matrisi. Hiçbir diagnostic rolü available değilse →
`DEGRADED` + `degraded_reason=NO_OPTIONAL_ROLE_AVAILABLE`, exit `0` (required
koşulları sağlanmışsa).

#### 3.2.5 `live-execution` — `RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT` (NA-4)

> **Düzeltme notu (NA-4):** Rol sınıflandırması `LIVE_EXECUTION_GATED` KORUNUR,
> ama açık `live-execution` modu bu sözleşmede
> `RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT` olarak dondurulur.

Bu sözleşmenin yönettiği implementasyonda `-Mode live-execution` herhangi bir
invocation'ı:

- mutasyondan ÖNCE fail eder;
- exit `21` döndürür;
- `overall_status=INVALID_REQUEST` ayarlar;
- `live_execution_requested=true` ayarlar;
- `live_execution_authorized=false` ayarlar;
- şunu raporlar:
  `LIVE_EXECUTION_MODE_REQUIRES_SEPARATE_DATA_READINESS_PREREGISTRATION`.

**Gerekçe:** Bu sözleşme şunları DONDURMAZ: gerekli data collector'ları,
veritabanı tazeliği (freshness), observer/readiness durumu, sinyal tazeliği,
risk-engine hazırlığı, exchange/session hazırlığı. **CollectorSupervisor +
heartbeat watchdog + executor-dosya-varlığı, live execution için YETERLİ
olarak SUNULAMAZ.** Gelecekte ayrı olarak preregister edilip bağımsız review
edilen bir sözleşme, bu mod implementasyona geçmeden önce live data-readiness
ve execution-readiness'i tanımlamalıdır.

Legacy davranış değişmez: live executor default-off kalır ve mevcut tüm
gate'ler (2-CLI-flag + 3-env-var çok katmanlı) KORUNUR.

### 3.3 Optional-rol outcome matrisi (NA-2 — DONDURULMUŞ)

> **Düzeltme notu (NA-2):** Birinci corrective, "optional roller mevcut değildi
> VEYA fail etti → exit 0 DEGRADED" ile "optional/secondary rol hatası → exit 12"
> arasında bir ÇELİŞKİ bırakmıştı; attempted-and-failed bir optional rol her iki
> exit'e de eşlenebiliyordu. Bu ikinci corrective dört ayrı optional-rol
> sonucunu dondurur — attempted-and-failed ASLA exit 0'a normalize edilemez.

Her optional (SHADOW_OPTIONAL / DIAGNOSTIC_OPTIONAL) rol, tam olarak dört
sonuçtan birine düşer:

| # | Durum | `status` değeri | Mutasyon | Listelenir | overall/exit etkisi |
|---|---|---|---|---|---|
| 1 | Seçilen mod tarafından hariç tutuldu | `SKIPPED_BY_MODE` | Yok | — | Denenmez; failure DEĞİL; degraded üretmez; exit'i ETKİLEMEZ |
| 2 | Start denemesinden önce optional bağımlılık yok | `SKIPPED_MISSING_OPTIONAL_DEPENDENCY` | Yok | `optional_unavailable` | Mod `DEGRADED` + `degraded_reason=NO_OPTIONAL_ROLE_AVAILABLE` \| `PARTIAL_OPTIONAL_AVAILABILITY`, exit `0` — YALNIZ her required rol başarılı/güvenle-sahiplenilmişse |
| 3 | Optional rol zaten güvenle sahiplenilmiş | `ALREADY_OWNED` | Yok (yeni) | — | Available/satisfied sayılır; failure DEĞİL |
| 4 | Optional rol denendi ama operasyonel olarak başarısız | `SECONDARY_ROLE_FAILED` | (denenmiş) | `optional_failures` | `overall_status=SECONDARY_ROLE_FAILED`, exit `12`; ASLA `DEGRADED`/exit `0`'a normalize edilmez |

Sonuç #4 şunları KAPSAR: `START_FAILED`, `OWNERSHIP_CONFLICT`, malformed
owned-state sonucu, proses başlayıp verification sırasında ölmesi, required
optional-rol verification'ının güvenli sahipliği kuramaması.

**Değişmez kural:** Bir optional rol, denenmiş bir operasyonel start
başarısızlığından SONRA asla exit `0` üretemez. "Absent" (#2) ile
"attempted-and-failed" (#4) ayrı ayrı ele alınır ve karıştırılamaz.

### 3.4 Tek-mod kuralı (NA-5 — dead composition semantiği kaldırıldı)

> **Düzeltme notu (NA-5):** Birinci corrective'in §3.3'ü, açık modlar tek-seçim
> olmasına rağmen "birden fazla mod aynı anda" ve `degraded_modes` calculus'undan
> bahsediyordu — tetikleyicisi olmayan ölü bir cümleydi. Kaldırıldı.

- Tam olarak **bir** açık mod seçilebilir.
- Modlar **birleştirilemez**.
- Tekrarlanan veya çoklu mod talebi → mutasyondan önce **exit `21`**.
- `legacy-full-contract` YALNIZCA `-Mode` yokluğuyla seçilir; açık modlarla
  kompose edilemez.
- Kullanılmayan `degraded_modes` calculus'u **KALDIRILDI** — böyle bir alan
  yoktur.

---

## 4. Fail-closed non-regresyon kuralları + exit-kodu ailesi (A4 — DONDURULMUŞ)

### 4.1 Non-regresyon kuralları

1. Bilinmeyen/malformed task/process ownership → fail-closed (RR-01).
2. `collector_supervisor` required runtime gate olarak kalır.
3. Mode-required eksik bağımlılıklar mutasyondan ÖNCE fail eder.
4. Optional roller genel sonucu sessizce `SUCCESS`'e yükseltemez.
5. Live execution varsayılan kapalı kalır.
6. `-EnableLive` TEK BAŞINA gerçek order submission için yetersiz kalır.
7. Hiçbir dosya-varlık kontrolü, import-edilebilir/yapılandırılmış/operasyonel
   kanıtı olarak sunulamaz.
8. Exit kodu ve JSON özeti her zaman aynı fikirde (tek production karar kaynağı).
9. `-DryRun` sıfır proses/task/veritabanı mutasyonu yapar.
10. Var olan foreign-owned proses ve task'lar korunur.

### 4.2 Zorunlu exit-kodu ailesi (yeni kod EKLENMEZ)

| Kod | Anlam |
|---|---|
| `10` | `legacy-full-contract` veya seçilen açık mod için required bağımlılık eksik |
| `11` | Required runtime ownership/task/process gate hatası |
| `12` | Required gate başarılı olduktan sonra, talep edilen optional/secondary rol operasyonel hatası (§3.3 sonuç #4) |
| `20` | İç launcher/runtime hatası |
| `21` | Malformed argümanlar, geçersiz/tekrarlanan/çoklu mod, `live-execution` (reserved), açık mod + `-EnableLiquidationSilenceScheduler`, uyumsuz switch'ler, malformed synthetic override, geçersiz konfigürasyon |
| `0` | Seçilen sözleşme başarıyla VEYA açıkça izin verilen degraded optional-rol sonucuyla (§3.3 sonuç #2) tamamlandı |

`0` yalnızca şu koşullar TÜMÜ sağlandığında degraded sonuç için kullanılabilir:
- seçilen mod için her required rol başarılı/güvenle-sahiplenilmiş;
- hiçbir required bağımlılık veya ownership gate'i fail etmedi;
- yalnız optional roller **absent** idi (§3.3 #2), attempted-and-failed (#4)
  DEĞİL;
- `overall_status=DEGRADED`, `degraded_reason ∈ {NO_OPTIONAL_ROLE_AVAILABLE,
  PARTIAL_OPTIONAL_AVAILABILITY}`;
- `optional_unavailable` tam enumerate edilir; `optional_failures` boş.

`live-execution` için: bu sözleşmede daima reserved → exit `21`,
`overall_status=INVALID_REQUEST` (§3.2.5).

**Hiçbir implementasyon başka bir exit ailesi oluşturamaz.**

---

## 5. Zorunlu çıktı semantiği (A5 — DONDURULMUŞ)

Her özet şu alanları İÇERMELİDİR:

- `requested_mode`
- `effective_mode`
- `legacy_compatibility_mode`
- `required_roles`
- `optional_roles`
- `excluded_roles`
- `required_failures`
- `optional_unavailable`  *(NA-3: absent optional roller — failure'dan ayrı)*
- `optional_failures`  *(yalnız §3.3 sonuç #4)*
- `overall_status`
- `degraded_reason`  *(NA-3: yeni zorunlu alan)*
- `exit_code`
- `mutation_performed`
- `live_execution_requested`
- `live_execution_authorized`

İzin verilen `overall_status` değerleri (TAM makine sözlüğü):

- `SUCCESS`
- `DEGRADED`
- `REQUIRED_DEPENDENCY_FAILED`
- `REQUIRED_ROLE_FAILED`
- `SECONDARY_ROLE_FAILED`
- `INVALID_REQUEST`
- `INTERNAL_FAILURE`

İzin verilen `degraded_reason` değerleri (NA-3):

- `NONE`  *(degraded-olmayan tüm sonuçlar)*
- `NO_OPTIONAL_ROLE_AVAILABLE`
- `PARTIAL_OPTIONAL_AVAILABILITY`

> **NA-3 kapanışı:** `DEGRADED_NO_SHADOW_ROLE_AVAILABLE` /
> `DEGRADED_NO_DIAGNOSTIC_ROLE_AVAILABLE` gibi rakip `overall_status` değerleri
> KULLANILMAZ. Degraded durumu daima `overall_status=DEGRADED` + ayrı
> `degraded_reason` alanıyla temsil edilir. Bu makine sözlüğü belgede ve
> gelecekteki testlerde HER YERDE aynıdır.

`exit_code` ve `overall_status`, **tek bir production karar kaynağından**
türetilmelidir.

---

## 6. Gerekli test sözleşmesi (A6 — DONDURULMUŞ)

### 6.1 Mevcut kabul edilmiş testler (supersession'lar)

`tests/test_start_eclipse_launcher_safety.py` @ `main`: **68 fonksiyon, 69
collected test**. **Hepsi geçerliliğini korumalıdır**, aşağıdaki İKİ AÇIKÇA
ADLANDIRILMIŞ supersession hariç:

#### `test_38`–`test_42` ailesi

Mevcut legacy-invocation assertion'ları KORUNUR. Her ilgili test EK OLARAK:
- mod bayrağı verilmeyen invocation `effective_mode=legacy-full-contract` üretir;
- aynı eksik bağımlılık hâlâ exit `10` üretir;
- hiçbir mutasyon oluşmaz;
- mevcut required-role ve summary assertion'ları DEĞİŞMEDEN kalır.

#### `test_27b_healthy_full_run_is_zero`

Mevcut "live executor varsayılan kapalı" assertion'ı KORUNUR. EK OLARAK:
- mod argümanı verilmemesi live executor'ı başlatmaz;
- `-Mode core-infrastructure`/`data-collection`/`shadow-observation`/
  `diagnostics` live executor'ı asla başlatamaz;
- `-Mode live-execution` → reserved, mutasyondan önce exit `21`,
  `overall_status=INVALID_REQUEST`;
- yalnız bağımlılık varlığı asla `live_execution_authorized=true` ayarlamaz.

### 6.2 Yeni zorunlu davranışsal testler (NA-1..NA-4 dahil)

1. **Tüm 12 rol taksonomi eşlemesinde tam bir kez görünür** (§2.1 tablosu
   makine-kontrol edilir; eksik/çift rol → FAIL).
2. **`liquidation_silence_scheduler` yalnız `SCHEDULER_GATED`'e eşlenir.**
3. **No-mode legacy invocation scheduler bağımlılık doğrulamasını korur**
   (dosya eksikse exit `10`; switch yoksa yeniden başlatmaz; owned durum korunur).
4. **Açık mod + `-EnableLiquidationSilenceScheduler` → mutasyondan önce exit
   `21`.**
5. **`SKIPPED_BY_MODE` degrade veya fail ETMEZ** (exit'i etkilemez).
6. **Absent optional bağımlılık → `DEGRADED` + exit `0`** (yalnız required
   koşulları sağlanmışsa; `degraded_reason` doğru; `optional_unavailable`
   dolu).
7. **Attempted optional `START_FAILED` → exit `12`** (asla `0` değil).
8. **Optional `OWNERSHIP_CONFLICT` → exit `12`.**
9. **Optional `ALREADY_OWNED` → available sayılır** (failure değil).
10. **`degraded_reason` sözlüğü tam** (`NONE`/`NO_OPTIONAL_ROLE_AVAILABLE`/
    `PARTIAL_OPTIONAL_AVAILABILITY` dışında değer yok).
11. **`-Mode live-execution` → reserved/not-implementable exit `21`** +
    `LIVE_EXECUTION_MODE_REQUIRES_SEPARATE_DATA_READINESS_PREREGISTRATION`.
12. **Hiçbir production path bu sözleşmede live execution'ı yetkilendiremez.**
13. Temiz canonical checkout — `legacy-full-contract` + her açık mod ayrı ayrı.
14. `CORE_REQUIRED` (`collector_supervisor`/`heartbeat_watchdog`) eksik → her
    modda mutasyon öncesi durdurma.
15. `DATA_REQUIRED_FOR_MODE` eksik (`data-collection`) → exit `10`.
16. Tekrarlanan/çoklu `-Mode` → exit `21` (§3.4).
17. Summary/exit uyumu — her mod için, tüm §5 alanları dahil.
18. Mutasyon-öncesi preflight-fail → sıfır mutasyon (her mod).
19. Bilinmeyen task state → fail-closed (RR-01 regresyon YOK).
20. Foreign ownership → preserved; malformed override → exit `21`.
21. Required-role gate sıralaması — `collector_supervisor` optional/mode
    rollerinden ÖNCE.

### 6.3 Zorunlu mutasyon-probeleri

- `CORE_REQUIRED` sınıflandırmasını kaldır → test FAIL.
- `SCHEDULER_GATED`'den scheduler rolünü kaldır (sınıflandırılmamış bırak) →
  12/12 eşleme testi FAIL.
- Scheduler'ı sessizce bir açık moda dahil et → EXCLUDED_BY_MODE testi FAIL.
- `DATA_REQUIRED_FOR_MODE` rolünü sessizce optional yap → mode-required test FAIL.
- Attempted optional failure'ı `12`'den `0`'a değiştir → §3.3 #4 testi FAIL.
- Optional absence'ı failure `12`'ye çevir → §3.3 #2 testi FAIL.
- `DIAGNOSTIC_OPTIONAL` başarısızlığını exit'e sızdır → summary/exit testi FAIL.
- `SHADOW_OPTIONAL` başarısızlığını core-required gibi davran → gate-sıralama
  testi FAIL.
- Reserved `live-execution`'ın devam etmesine izin ver → reserved testi FAIL.
- `LIVE_EXECUTION_GATED` dosya-varlığını "process başladı" ile eşitle → ayrım
  testi FAIL.

Tüm testler production launcher path'lerini çalıştırmalıdır, test-only bir
taksonomi helper'ı DEĞİL.

---

## 7. Contract-A durumu

**`LAUNCHER_ROLE_TAXONOMY_SECOND_CORRECTIVE_PREREGISTERED_PENDING_INDEPENDENT_
REREVIEW`**

Bu taksonomi kabul edilmiş SAYILMAZ. Bağımsız rereview, en az şunları
doğrulamalıdır: (a) §2.1'de tüm 12 rolün tam bir kez sınıflandırıldığı ve
"12/12" iddiasının gerçek olduğu, (b) optional-rol outcome matrisinin (§3.3)
attempted-and-failed'i asla exit `0`'a normalize etmediği, (c) degraded
sözlüğünün (§5) tutarlı olduğu, (d) `live-execution`'ın reserved/not-
implementable olarak dondurulduğu, (e) exit ailesinin yeni kod eklemeden
tutarlı kaldığı.

---

## 8. Kapsam dışı (bu sözleşmede DEĞİŞMEZ, implementasyon-yetkilendirilmemiş)

- `data/oi_spot_poller.py`, `tools/s34_v_engine_v02_shadow_mirror.py`,
  `tools/s34_state_machine_live_executor.py`, `tools/orderflow_chart.py`,
  `tools/s34_replay.py`, `tools/liquidation_silence_scheduler.py`
  canonicalization'ı — ayrı workstream'ler.
- `data-collection` modu — bileşenleri kendi bounded-implementation
  zincirlerini tamamlamadıkça implementasyon-yetkilendirilmemiştir.
- `live-execution` modu — `RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT`; ayrı
  data-readiness preregistration'ı gerekir.
- Scheduler'ın açık modlarla kompozisyonu — ayrı amendment gerekir.
- CollectorSupervisor deployment, PID `20648` handoff, runtime alignment,
  full-stack launch — yetkilendirilmemiştir.
- Alpha/sinyal geçerliliği.

---

## 9. Düzeltme geçmişi — birinci corrective (A-F1..A-F6)

Kontrol eden bağımsız review: `LAUNCHER_ROLE_TAXONOMY_PREREGISTRATION_
CORRECTIVE_REQUIRED`.

| Bulgu | Şiddet | Bu düzeltmede ne yapıldı (birinci corrective) |
|---|---|---|
| A-F1 | BLOCKER | §3.1 `legacy-full-contract` donduruldu |
| A-F2 | BLOCKER | §2 rol atamaları (ancak 12. rol atlandı — NA-1'de düzeltildi) |
| A-F3 | HIGH | §4.2 exit ailesi `10/11/12/20/21/0` ile sınırlandı, `13` kaldırıldı |
| A-F4 | MEDIUM | §6.1 supersession replacement assertion'ları donduruldu |
| A-F5 | INFO | §2 çerçeveleme netleştirildi |
| A-F6 | INFO | §1.1 satır referansı düzeltildi |

## 10. Düzeltme geçmişi — ikinci corrective (NA-1..NA-5)

Kontrol eden bağımsız rereview: `LAUNCHER_ROLE_TAXONOMY_CORRECTIVE_REREVIEW_
REQUIRED`. **Durum: corrective operator claims closed; pending independent
rereview.**

| Bulgu | Şiddet | Prior metin | Düzeltilmiş deterministik kural | Etkilenen testler | Kalan yetki sınırı |
|---|---|---|---|---|---|
| NA-1 | BLOCKER | §2 yalnız 11 rol sınıflandırdı; `liquidation_silence_scheduler` atlandı; "tüm 12 atandı" GERÇEK DIŞIYDI | §2.1'de yeni `SCHEDULER_GATED` kategorisi + 12/12 makine-kontrol-edilebilir eşleme tablosu; legacy'de dependency-presence korunur + `-EnableLiquidationSilenceScheduler`; açık modlarda `EXCLUDED_BY_MODE`; açık mod + switch → exit `21` | §6.2/1-4, §6.3 scheduler probe'ları | Scheduler'ın açık-mod kompozisyonu ayrı amendment gerektirir |
| NA-2 | HIGH | Attempted-and-failed optional rol hem exit `12` hem exit `0`'a eşlenebiliyordu | §3.3 dört-sonuç matrisi: `SKIPPED_BY_MODE` / `SKIPPED_MISSING_OPTIONAL_DEPENDENCY`(→DEGRADED+0) / `ALREADY_OWNED` / operasyonel-fail(→`SECONDARY_ROLE_FAILED`+`12`); #4 asla `0`'a normalize edilmez | §6.2/5-9, §6.3 optional probe'ları | — |
| NA-3 | MEDIUM | `DEGRADED_NO_*_AVAILABLE` §5 `overall_status` sözlüğünde yoktu | §5'e `degraded_reason` alanı (`NONE`/`NO_OPTIONAL_ROLE_AVAILABLE`/`PARTIAL_OPTIONAL_AVAILABILITY`) + `optional_unavailable` listesi eklendi; rakip status değerleri kaldırıldı | §6.2/6,10 | — |
| NA-4 | MEDIUM | live-execution required-set data-readiness gate'i olmadan definitive görünüyordu | §3.2.5 `RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT`: `-Mode live-execution` → exit `21`, `INVALID_REQUEST`, data-readiness preregistration mesajı | §6.2/11-12 | Ayrı data-readiness + execution-readiness preregistration'ı gerekir |
| NA-5 | INFO | §3.3 (eski) tetikleyicisi olmayan multi-mode/`degraded_modes` calculus'u | §3.4 tek-mod kuralı; `degraded_modes` kaldırıldı; çoklu/tekrarlı mod → exit `21` | §6.2/16 | — |

Bu düzeltme kendi kendini kabul ETMEZ. Bir sonraki gerekli adım genuinely
bağımsız, taze-context bir rereview'dır.

---

## 11. Bağımsız review gereksinimi

Bu belge, genuinely bağımsız (bu belgeyi yazan aktörden farklı, taze context)
bir rereview'dan geçmeden implementasyona TAŞINAMAZ. Kabul, ayrı bir verdict
ile kayıt altına alınmalıdır — bu belgenin kendisi kabul beyanı DEĞİLDİR.

---

## 12. Operatör kabulü (2026-07-15)

**Durum: `LAUNCHER_ROLE_TAXONOMY_PREREGISTRATION_OPERATOR_ACCEPTED`.**

- **Bağımsız rereview verdikti:** `BOTH_SECOND_CORRECTED_PREREGISTRATION_
  CONTRACTS_ACCEPTED_PENDING_OPERATOR_SIGNOFF` / Contract A =
  `LAUNCHER_ROLE_TAXONOMY_ACCEPTED` (genuinely bağımsız, taze-context, salt-
  okunur rereviewer; ağaç review boyunca donmuş; sıfır mutasyon).
- **Operatör kabul tarihi:** 2026-07-15.
- **Bulgu kapanışı:** NA-1, NA-2, NA-3, NA-4, NA-5 — tümü CLOSED (bağımsız
  rereview'da doğrulandı; §10 tablosu).
- **Kalan gözlemler (non-blocking, kabulü engellemez):** A-INFO-1
  (`overall_status`↔`exit_code` eşlemesi tek tabloda toplanabilir — türetilebilir,
  davranış değişmez); A-INFO-2 (`ALREADY_OWNED` mevcut safe-owned durumlar
  üzerine yeni bir sözleşme soyutlaması — benign). Bunlar açık gözlem olarak
  KORUNUR; "düzeltildi" olarak sunulmaz.

**Operatör kabulünün preserve ettiği güvenlik sınırları (değişmez):**
- Sözleşme kabulü implementasyon yetkisi DEĞİLDİR.
- Sözleşme kabulü deployment/launcher/collector çalıştırma yetkisi DEĞİLDİR.
- `live-execution` modu `RESERVED_NOT_IMPLEMENTABLE_IN_THIS_CONTRACT` olarak
  rezerve ve erişilemez kalır (§3.2.5); live execution açılmamıştır.
- Hiçbir proses sinyallenmez/yeniden başlatılmaz; hiçbir Scheduled Task
  değiştirilmez; hiçbir canlı veritabanı mutasyona uğratılmaz.
- Bir sonraki izin verilen faz YALNIZCA ayrı, açıkça yetkilendirilmiş bounded
  **implementation-planning** kapısıdır — bu belge onu yetkilendirmez.

Bu operatör kabulü, §2-§6'daki substantive sözleşme davranışını DEĞİŞTİRMEZ;
yalnız preregistration'ı operatör-kabul-edilmiş terminal duruma reconcile eder.
