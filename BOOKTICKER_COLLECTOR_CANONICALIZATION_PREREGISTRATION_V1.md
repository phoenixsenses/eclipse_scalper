# BOOKTICKER_COLLECTOR_CANONICALIZATION_PREREGISTRATION_V1

**Tarih:** 2026-07-14 · **Yazan:** Claude Sonnet 5 (bounded preregistration operator) · **Durum:** `BOOKTICKER_CANONICALIZATION_PREREGISTRATION_OPERATOR_ACCEPTED` (operatör kabulü 2026-07-15 — bkz. §B11)

> Bu belge `data/bookticker_collector.py`'nin gelecekteki bounded
> implementation + bağımsız review zinciri için bir sözleşme donduruşudur.
> Preregistration ≠ acceptance. Şu ana kadar hiçbir kod satırı, test, dosya,
> Scheduled Task, proses veya veritabanı bu belgeyle değiştirilmedi.
> Implementation, izole bir temiz worktree'de, ayrı yetkilendirmeyle
> gerçekleşmelidir. Deployment bu belgenin kapsamı DIŞINDADIR.
>
> Bu belge, `data/bookticker_collector.py`'nin şu an canonical, kabul edilmiş
> veya deploy edilebilir olduğunu **İDDİA ETMEZ**.
>
> **Sürüm zinciri:** ilk preregistration → bağımsız review
> (`CORRECTIVE_REQUIRED`, B-F1..B-F6) → birinci corrective → bağımsız rereview
> (`REREVIEW_REQUIRED`, NB-1..NB-6) → **bu ikinci corrective**. §B9 (birinci)
> ve §B10 (ikinci) düzeltme geçmişleri her bulgunun nasıl kapatıldığını
> kaydeder. Bu düzeltme kendi kendini kabul etmez; tüm kapanış ifadeleri
> **"corrective operator claims closed; pending independent rereview"**
> statüsündedir.

---

## B1. Dondurulmuş kaynak kimliği

| Alan | Değer |
|---|---|
| Path | `data/bookticker_collector.py` |
| Tam SHA-256 | `136438fdcf91cdae1a645b3fc229e26b6047d5a4bd53748f9a395a6ae70d4c79` |
| Boyut | 12,159 bayt |
| Satır sayısı | 330 |
| Git-tracked mi | Hayır |
| `main` tree'sinde mevcut mu | Hayır |
| Canlı proses | **Evet** — PID 15332, cmdline: `python.exe -W ignore -u -m data.bookticker_collector --symbols BTCUSDT,ETHUSDT,SOLUSDT --db-path data/microstructure.db --heartbeat-interval 5` |
| Proses durduruldu/sinyallendi mi | **Hayır** |

Aday hash **değişmedi** — devam ediliyor.

---

## B2. Canonicalization kapsamı (dondurulmuş)

### B2.1 Zorunlu unsurlar

1. `data/bookticker_collector.py`'nin canonical eklenmesi (git add + commit,
   ayrı bir implementasyon batch'inde).
2. `book_ticker` şeması için `NUMBERED_MIGRATION_OWNED` (bkz. B4.2).
3. Migration ve collector değişiklikleri diff/testlerde ayrı ayrı
   tanımlanabilir olmalıdır.
4. Singleton/cross-process exclusivity (bkz. B5) — mevcut dosyada YOK.
5. Stale lock kurtarma — OS-lock otomatik serbest bırakmayla (bkz. B5).
6. Graceful shutdown — mevcut SIGINT/SIGTERM KORUNUR; lock-release `finally`'ye.
7. Duplicate-instance prevention — B5 (proses-seviyesi, OS byte-range lock).
8. Public WebSocket reconnect — mevcut exponential-backoff (1.0→×1.7, max 30s)
   KORUNUR.
9. Idle-timeout — `IDLE_RECONNECT_SEC = 30.0` KORUNUR.
10. Database lock retry — `flush_ticks` 6 deneme + 0.25s×attempt KORUNUR.
11. WAL/busy-timeout — `journal_mode=WAL`, `synchronous=NORMAL`,
    `busy_timeout=30000` KORUNUR.
12. Health-state raporlama — `write_component_health("bookticker", ...)`
    KORUNUR + genişletilmiş sayaçlar (bkz. B4.6).
13. Bounded batching — `FLUSH_INTERVAL_SEC=5.0`, `FLUSH_BATCH_SIZE=5000`,
    `MAX_PENDING_TICKS=50000` + `trim_pending` KORUNUR.
14. Validation/reconnect — **DEĞİŞİYOR** (bkz. B4.4/B4.5); tick-local reddi +
    dondurulmuş eşik.
15. Kimlik bilgisi gerektirmez — negatif testle kilitlenir.
16. Trading/order kapasitesi yok — negatif testle kilitlenir.
17. Otomatik Scheduled Task oluşturma yok.
18. Runtime deployment yok — PID 15332'ye dokunulmaz (bkz. B7).

### B2.2 Kapatılmış açık sorular

- Singleton koruması `collect()`'i nasıl etkiler? → lock alımı `collect()`
  başında, `init_db()`'den sonra, herhangi bir WebSocket bağlantısından ve DB
  mutasyonundan ÖNCE; signal-handler DEĞİŞMEZ; `finally`'ye lock-release
  eklenir (bkz. B5).
- `UNIQUE` gerekli mi? → Hayır, ilk canonicalization için (bkz. B3, append-only).
- `--max-seconds` kalmalı mı? → Evet, `test-only` olarak dokümante edilir.

---

## B3. Duplicate-veri politikası (dondurulmuş — B-F2)

**Politika: append-only event capture.**

Bir duplicate, aynı fiyat/miktar değerlerine sahip iki satır olarak
TANIMLANMAZ — tekrarlanan özdeş exchange-state event'leri geçerli piyasa
olaylarıdır. Canonical collector, başarıyla alınan ve geçerli her event'i
**ayrı bir satır olarak** saklar.

Proses-seviyesi duplikasyon önlemesi **singleton exclusivity** ile sağlanır
(bkz. B5) — şema-seviyesi kısıtlama ile DEĞİL.

Transaction politikasının tamamı, gerçekçi SQLite failure sözleşmesi olarak
**B4.7'de** dondurulur (NB-5: aspirasyonel "ambiguous commit" kaldırıldı).

Reconnect ve normal proses restart'ı, önceden commit edilmiş event'leri
DEDUPLICATE ETMEZ (kasıtlı, append-only sonucu). **Content-based `INSERT OR
IGNORE` veya sessiz deduplication YASAKTIR.**

---

## B4. Veritabanı sözleşmesi (dondurulmuş)

### B4.1 Şema (mevcut kolonlar ve indeksler dondurulmuş)

```sql
CREATE TABLE IF NOT EXISTS book_ticker (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_ms INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  bid_price REAL NOT NULL,
  bid_qty REAL NOT NULL,
  bid_depth_usd REAL,
  ask_price REAL NOT NULL,
  ask_qty REAL NOT NULL,
  mid_price REAL NOT NULL,
  spread_pct REAL NOT NULL,
  book_imbalance REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bt_symbol_ts ON book_ticker(symbol, ts_ms);
CREATE INDEX IF NOT EXISTS idx_bt_ts ON book_ticker(ts_ms);
```

`UNIQUE` kısıtlaması ilk canonicalization için gerekli DEĞİLDİR (bkz. B3).

### B4.2 Şema sahipliği — `NUMBERED_MIGRATION_OWNED` (B-F1) + non-destructive (NB-4)

**Seçilen model: `NUMBERED_MIGRATION_OWNED`.** Gelecekteki collector, normal
runtime sırasında `book_ticker` şemasını OLUŞTURMAZ/DEĞİŞTİRMEZ; yalnız
salt-okunur uyumluluk doğrular (`PRAGMA table_info`), `CREATE`/`ALTER`
ÇALIŞTIRMAZ. Şema eksik/uyumsuzsa:
- public WebSocket'e bağlanmadan ÖNCE fail eder;
- hiçbir veritabanı satırı yazmaz;
- hiçbir retry loop'u başlatmaz;
- typed, non-zero startup sonucu döndürür;
- yalnız izole/yetkilendirilmiş health-state path'i mevcutsa günceller.

Migration'ın tam davranışı **B4.8'de non-destructive olarak dondurulur** (NB-4).

### B4.3 Diğer zorunlu unsurlar

- İndeksler: mevcut ikisi korunur.
- Zaman damgası: tamsayı ms epoch (`ts_ms`).
- Symbol normalizasyonu: `str(data["s"]).upper()` KORUNUR.
- Bid/ask: sonlu ve sıfırdan büyük.
- `bid_price <= ask_price` (ters kitap reddi — yeni davranış, tick-local).
- Sıfır spread (`bid_price == ask_price`) GEÇERLİDİR.
- Negatif spread reddedilir (tick-local).
- Miktarlar: sonlu ve negatif-olmayan.
- Desteklenmeyen/talep edilmemiş semboller reddedilir (tick-local).
- Event timestamp gerekli ve pozitif; receive timestamp yerel ve pozitif.
- Duplicate-row: B3 (append-only).
- Transaction: B4.7 (gerçekçi SQLite sözleşmesi).
- Batch/flush/retry/lock-timeout/shutdown-flush: mevcut değerler KORUNUR.

### B4.4 Mevcut kaynağın gerçek davranışı (kayıt için)

- **Malformed numeric/content** (`_float()` NaN/negatif, `json.loads` hatası,
  eksik alan, `mid_price<=0`) → `parse_bookticker()` raise → `collect()` dış
  `except Exception` → **tüm soket reconnect edilir** (tick-local DEĞİL).
- **Sembol uyuşmazlığı** (`row[1] not in symbols`) → satır 262 `continue` ile
  **tick-local atlanır**.

### B4.5 Gelecekteki validation/reconnect (dondurulmuş, yeni)

#### Tick-local reddi (reconnect YOK)

Yalnız mevcut mesajı reddet, aynı soket üzerinde devam et:
malformed JSON (tek mesaja izole); eksik alanlar; geçersiz numerik dönüşüm;
sonlu-olmayan değerler; pozitif-olmayan bid/ask; `bid_price > ask_price`;
negatif miktar; eksik/pozitif-olmayan event timestamp; stale event time;
future event time; sembol uyuşmazlığı; desteklenmeyen/talep-edilmemiş sembol.

Bounded health/error sayacı kaydedilir, ama tek bir geçersiz mesaj YÜZÜNDEN
sağlıklı soket YIKILMAZ.

#### Tam reconnect (yalnız)

WebSocket kapanması; transport exception'ı; devam eden tüketimi engelleyen
protokol/frame hatası; idle timeout; yerel stream subscription hatası;
dondurulmuş eşiği aşan tekrarlanan parsing hataları (bkz. B4.6).

#### Sıfır spread

`bid_price == ask_price` GEÇERLİDİR.

### B4.6 Tam invalid-message pencere semantiği (NB-3 — DONDURULMUŞ)

> **Düzeltme notu (NB-3):** Birinci corrective yalnız "5 ret / 30s rolling"
> başlığını donduruyordu; counter reset, pencere reset, saat kaynağı, 5.
> mesajın akıbeti, reconnect sırasında flush-fail belirsizdi. Bu ikinci
> corrective iki ayrı saat ve tam algoritma dondurur.

**İki saat, iki amaç:**

**(a) Exchange timestamp doğrulama — UTC wall-clock epoch ms:**
- stale: `receive_epoch_ms - event_epoch_ms > 60000` (tam `60000` ms KABUL);
- future: `event_epoch_ms - receive_epoch_ms > 5000` (tam `5000` ms KABUL).

**(b) Rolling invalid-message penceresi — monotonic saat:**
1. Her tick-local reddedilen mesaj için `now_monotonic` al.
2. `now_monotonic - timestamp >= 30.0` olan saklı ret timestamp'lerini SİL.
3. Mevcut ret timestamp'ini EKLE.
4. Reddedilen mesajı at; ASLA insert edilmez.
5. Sonuç sayı `5` veya daha fazlaysa:
   - process-lifetime reconnect-threshold sayacını artır;
   - pending geçerli satırları flush etmeyi dene;
   - flush başarılıysa: soketi kapat, normal reconnect backoff'a gir;
   - flush dondurulmuş DB-retry politikasından sonra başarısızsa: reconnect
     ETME, süreci typed non-zero failure ile SONLANDIR;
   - rolling ret penceresini yalnız soket reconnect için kapatıldıktan SONRA
     sıfırla.
6. Geçerli bir mesaj rolling ret penceresini TEMİZLEMEZ.
7. Transport reconnect (başarılı/başarısız) yeni bir soketten mesaj tüketmeden
   ÖNCE rolling ret penceresini temizler.
8. 5. reddedilen mesaj sayılır, atılır ve tetikleyicidir.
9. Rolling pencere proses-lokaldir, restart'lar arası kalıcı DEĞİLDİR.
10. Health-state ile raporlanan process-lifetime sayaçları reconnect'ler arası
    kalıcıdır, yalnız proses restart'ında sıfırlanır.

**Dondurulmuş health sayaçları:** total tick-local ret; malformed JSON; eksik
alanlar; geçersiz numerik; geçersiz kitap; geçersiz miktar; geçersiz timestamp;
desteklenmeyen sembol; stale timestamp; future timestamp; threshold-triggered
reconnect'ler.

**Health-state yazma hatası:**
- WebSocket'i reconnect ETMEZ;
- izole/yetkilendirilmiş logging path'i üzerinden loglanır;
- geçerli bir market event'i ATMAZ;
- yalnız dondurulmuş ardışık-hata eşiğinden sonra process-fatal olur.

**Dondurulmuş eşik: `10` ardışık health-state yazma hatası.** Başarılı bir
health-state yazımı ardışık-hata sayacını sıfırlar.

### B4.7 Gerçekçi SQLite failure sözleşmesi (NB-5 — DONDURULMUŞ)

> **Düzeltme notu (NB-5):** "Ambiguous commit outcome" client-server DB
> semantiğidir; lokal sqlite3 için uygulanamaz/aspirasyoneldi. Kaldırıldı;
> gerçek SQLite semantiği donduruldu.

1. Her batch için tek explicit transaction başlat.
2. Tüm insert'leri çalıştır.
3. Bir kez commit et.
4. Satırları yalnız `commit()` başarıyla döndükten SONRA memory'den kaldır.
5. `locked`/`busy` olarak sınıflanan `sqlite3.OperationalError`'da:
   - best-effort rollback; tam batch'i koru; dondurulmuş lock-retry
     tarifesine göre retry et.
6. Retry tükendikten sonra:
   - batch'i ATMA; typed non-zero DB-lock failure ile SONLAN; otomatik olarak
     ikinci bir proses BAŞLATMA veya production'a replay ETME.
7. Lock-dışı `OperationalError`, `DatabaseError`, integrity hatası veya şema
   uyuşmazlığında:
   - best-effort rollback; typed non-zero failure ile derhal SONLAN; reconnect
     loop YOK; kör batch replay YOK.
8. Başarılı commit'te: yalnız commit edilen batch'i temizle; devam et.

"Ambiguous commit" kaldırıldığından, ilgili test (eski #19) B6.1'de gerçek
lock/busy retry testleriyle DEĞİŞTİRİLDİ.

### B4.8 Non-destructive migration sözleşmesi (NB-4 — DONDURULMUŞ)

**Temiz veritabanı** (`book_ticker` yoksa): dondurulmuş tabloyu oluştur;
dondurulmuş indeksleri oluştur; ilgisiz tablolara DOKUNMA.

**Mevcut uyumlu tablo** — uyumluluk şunu gerektirir: her required kolon var;
required SQLite affinity uyumlu; primary-key sözleşmesi uyumlu; required
indeksler eşdeğer var VEYA non-destructive oluşturulabilir; ilgisiz ek kolonlar
otomatik olarak uyumsuz YAPMAZ. İzin verilen değişiklikler: `CREATE TABLE IF
NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`, eksik required indeks oluşturma.

**YASAK değişiklikler:** `DROP TABLE`; tablo rename; tablo recreation;
copy-and-swap replacement; mevcut satırların silinmesi/yeniden yazılması;
destructive kolon değişiklikleri; implicit data backfill; canlı veritabanını
vacuum; ilgisiz şema değişikliği.

**Mevcut uyumsuz tablo** — migration: typed incompatibility sonucuyla
fail-closed; hiçbir şema/veri değişikliği yapmaz; tam olarak uyuşmayan
kolonları/affinity'leri/PK-kuralını/indeks-tanımını raporlar; remediation
öncesi ayrı bir migration preregistration'ı gerektirir.

**Migration identifier:** Bu belge potansiyel olarak stale bir migration
numarasını DONDURMAZ. İzole implementasyon preflight'ında: (1) canonical
main'in numaralı migration'larını incele; (2) tam olarak sonraki kullanılmamış
sıralı identifier'ı seç; (3) herhangi bir implementasyon yazımından önce kaydet;
(4) collision/gap/duplicate/eşzamanlı main ilerlemesinde abort; (5) seçilen
identifier implementasyonla birlikte bağımsız review edilir.

**Production uyumluluk kanıtı:** Implementasyon ve review production şemasını
SALT-OKUNUR inceleyebilir; migration'ı canlı veritabanına ÇALIŞTIRAMAZ.
Migration testleri yalnız disposable veritabanları ve yapısal olarak temsili
bir şema fixture'ı kullanır. Deployment ayrıca yetkilendirilir.

---

## B5. Singleton gereksinimi (dondurulmuş — B-F4 + NB-1 + NB-2)

### B5.1 Otoriter tasarım kaynağı — DONDURULDU

**Tek otoriter davranışsal model, kabul edilmiş canonical CollectorSupervisor
Windows byte-range lock'udur** (`scripts/collector_supervisor.py` @ `main` —
`acquire_supervisor_lock`/`release_supervisor_lock`, `msvcrt.locking`
kullanan; launcher testi `test_30_collector_supervisor_source_untouched` ile
referans alınan). **Untracked `tools/s34_v_engine_v02_shadow_mirror.py`
KULLANILMAZ.**

### B5.2 Tek-yazıcı lock kimliği (NB-1 — DONDURULMUŞ)

> **Düzeltme notu (NB-1):** Birinci corrective lock kimliğine sembol setini
> dahil ediyordu → aynı DB'ye farklı sembol setleriyle iki eşzamanlı yazıcı
> mümkün oluyordu (single-writer güvenlik hedefi ihlali). Bu ikinci corrective
> sembol setini kimlikten ÇIKARIR.

**Exclusivity kimliği:**
`NORMALIZED_ABSOLUTE_DATABASE_PATH + TABLE_NAME("book_ticker")`.

**Sembol seti yalnız metadata'dır; lock kimliğine ASLA katılmaz.**

Sonuçlar:
- aynı DB path + aynı tablo + aynı semboller → conflict;
- aynı DB path + aynı tablo + **farklı** semboller → conflict;
- aynı DB path + aynı tablo + herhangi bir argüman varyasyonu → conflict;
- farklı izole test DB path'leri bağımsız çalışabilir;
- **hiçbir iki desteklenen production instance'ı aynı `book_ticker` tablosuna
  eşzamanlı yazamaz.**

Lock path'i normalize edilmiş DB/tablo kimliğinden deterministik olarak
türetilir (VEYA o kimlik için özdeş olması garanti edilir). Production
konfigürasyonu lock'u farklı bir path'e override etmeye izin VERMEZ. Test-only
dependency injection runtime root'u yönlendirebilir, ama o root içindeki aynı
mantıksal DB/tablo kimliği hâlâ TEK lock'a eşlenmelidir.

### B5.3 CollectorSupervisor lock semantiği ile hizalama (NB-2 — DONDURULMUŞ)

> **Düzeltme notu (NB-2):** Birinci corrective, B5.1'de CollectorSupervisor'ı
> otorite ilan ederken B5.2'de PID metadata + proses-oluşturma-zamanı +
> PID-reuse tespiti (reddedilen shadow_mirror/OpenProcess modeli) donduruyordu
> — otoriteyle çelişki. Canonical CollectorSupervisor SAF `msvcrt.locking`
> byte-range lock kullanır, PID-value kontrolünü "racy ve PID-reuse'a açık"
> diye açıkça reddeder. Bu corrective B5.2'yi (artık B5.3) canonical modele
> hizalar.

Dondurulmuş davranış:
- Canonical `msvcrt.locking` desenini kullanan OS-arbitrated exclusive
  byte-range lock;
- açık lock-dosyası handle'ı collector yaşam süresince canlı kalır;
- OS, proses çıktığında/crash'te lock'u OTOMATİK serbest bırakır;
- lock şunlardan ÖNCE alınır: veritabanı mutasyonu; WebSocket bağlantısı;
  aktif-collector'ı temsil eden health-state geçişi;
- duplicate acquisition typed non-zero exit ile fail-closed;
- başka bir PID'in **polling'i YOK**;
- PID-liveness ownership çıkarımı YOK;
- proses-oluşturma-zamanı karşılaştırması YOK;
- stale-PID takeover algoritması YOK;
- ownership otoritesi olarak PID metadata YOK;
- unlocked fallback YOK.

**Diagnostic metadata dosyası** yalnızca şu koşullarda var olabilir: açıkça
non-authoritative; yazılamaması unlocked execution'a NEDEN OLMAZ; lock
ownership'i belirlemek için ASLA kullanılmaz; testler silinmesinin/
sahtelenmesinin OS lock'u bypass EDEMEYECEĞİNİ kanıtlar. **İlk implementasyonda
PID metadata'yı tamamen ELEMEK tercih edilir.** Tüm shadow-mirror/OpenProcess/
PID-reuse semantiği normatif sözleşmeden KALDIRILMIŞTIR.

---

## B6. Gerekli test sözleşmesi (dondurulmuş)

### B6.1 Zorunlu davranışsal testler (NB-1..NB-6 dahil)

**Singleton / lock (NB-1, NB-2):**
1. Aynı DB/tablo + farklı sembol setleri → conflict.
2. Aynı DB/tablo + herhangi bir argüman varyasyonu → conflict.
3. Farklı disposable DB path'leri bağımsız çalışabilir.
4. Sahtelenmiş/silinmiş diagnostic metadata OS lock'u bypass EDEMEZ.
5. Gerçek alt-proses crash'i sonrası lock serbest bırakılır.
6. Lock otoritesi için hiçbir PID-liveness API kullanılmaz (statik + davranışsal).
7. Lock, WebSocket bağlantısından ve DB mutasyonundan ÖNCE alınır.
8. Aynı-proses duplicate → typed non-zero.
9. Gerçek alt-proses cross-process exclusivity (izole, disposable DB).

**Validation / pencere (NB-3):**
10. 5. geçersiz mesaj reconnect tetikler.
11. Tam 30 saniye eski ret timestamp'i aged-out.
12. Geçerli mesajlar ret penceresini sıfırlamaz.
13. Reconnect rolling pencereyi sıfırlar.
14. 5. reddedilen mesaj insert edilmez.
15. Threshold reconnect sırasında flush hatası → süreç sonlanır (reconnect değil).
16. Tam 60-saniye stale sınırı KABUL; 60s'den fazla RET.
17. Tam 5-saniye future sınırı KABUL; 5s'den fazla RET.
18. Sıfır spread KABUL; negatif spread RET; ters kitap RET (tick-local).
19. Health sayaçları reconnect'ler arası kalıcı.
20. 10 ardışık health-yazma hatası sonlandırır; başarı sayacı sıfırlar.

**Migration (NB-4):**
21. Mevcut uyumlu tablo byte/veri olarak korunur.
22. Eksik indeksler non-destructive eklenir.
23. Uyumsuz şema → hiçbir mutasyon olmadan fail.
24. Migration hiçbir YASAK destructive SQL içermez (statik tarama).
25. Şema eksik/uyumsuz → collector WebSocket bağlantısından ÖNCE fail.

**SQLite failure (NB-5):**
26. Gerçek lock/busy retry → duplicate/partial satır YOK.
27. Retry tükenmesi batch'i korur ve typed sonlanır.

**İzolasyon / kimlik (NB-6):**
28. Sanitize edilmiş alt-proses ortamı production path'lere ulaşamaz.
29. Testler PID `15332`'ye asla referans vermez/kullanmaz.
30. Intake evidence, tracked modül oluşturulduktan sonra import/runtime kaynağı
    DEĞİLDİR.

**Mevcut korunan davranışlar:**
31. Temiz checkout'ta gerçek import; `websockets` yoksa `SystemExit`.
32. Geçerli tick insersiyonu (izole DB).
33. Shutdown flush (SIGTERM/SIGINT sonrası pending flush).
34. Socket kapanması / idle-timeout sonrası reconnect; backoff sınırları.
35. Health-state payload doğru (mock/spy).
36. Kimlik bilgisi erişimi yok; order/execution API yok (statik + davranışsal).
37. Scheduled Task mutasyonu yok.

### B6.2 Zorunlu izole test path'leri (B-F5)

Tek test-owned temporary root altında AÇIKÇA disposable:
- Veritabanı: `<tmp>/data/microstructure.db`
- Lock: `<tmp>/runtime/bookticker_collector.lock`
- PID/diagnostic metadata (yalnız non-authoritative): `<tmp>/runtime/bookticker_collector.pid.json`
- Health state: `<tmp>/runtime/health/bookticker.json`
- Loglar: `<tmp>/logs/bookticker/`

Hiçbir test varsayılanlar üzerinden production path'lerine çözümlenemez.
Mock/local WebSocket açıkça inject edilir; public Binance'e ulaşma girişimi
testi FAIL eder. **Alt-proses testleri sanitize edilmiş ortam alır; production
path environment değişkenleri kaldırılır/override edilir. Testler
`data/microstructure.db`'yi veya production runtime dizinini çözümleyemez;
PID `15332`'yi ownership için asla sorgulamaz.** (NB-6)

### B6.3 Zorunlu mutasyon-probeleri

- Sembol setini lock kimliğine geri ekle → conflict testleri (#1-2) FAIL.
- OS-lock otoritesi yerine PID-liveness koy → #6 FAIL.
- Lock'u ağ bağlantısından SONRA al → #7 FAIL.
- Invalid-window off-by-one → #10/#14 FAIL.
- Geçerli-mesaj sayaç reset → #12 FAIL.
- Destructive migration SQL → #24 FAIL.
- Retry tükenmesinde satır düşür → #27 FAIL.
- Public/production path escape → #28 FAIL.
- Intake evidence'ı runtime'da import et → #30 FAIL.
- Ters-kitap kabulünü aç → #18 FAIL; retry mantığını kaldır → #26 FAIL;
  shutdown-flush kaldır → #33 FAIL; idle-timeout kaldır → #34 FAIL;
  health-update kaldır → #35 FAIL; duplicate-instance fail-open → #8 FAIL.

Her materyal mutasyon en az bir anlamlı testi kırmalı; her probe sonrası ağaç
tam geri yüklenmeli.

---

## B7. Canlı proses sınırı (dondurulmuş, mutlak)

Canlı `data/bookticker_collector.py` süreci (**PID 15332**) foreign-owned
runtime state'tir. Gelecekteki zincir **KESİNLİKLE**: onu durdurmaz;
yeniden başlatmaz; sinyallemez; altındaki dosyayı değiştirmez; canonicalized
sürümü deploy ETMEZ; Scheduled Task oluşturmaz; handoff yapmaz; ikinci instance
başlatmaz; test için canlı `data/microstructure.db`'yi mutasyona uğratmaz.

Tüm davranışsal testler izole geçici veritabanları, mock/local WebSocket ve
disposable alt-süreçler kullanır. Deployment/handoff kapsam dışıdır.

### B7.1 Kaynak-aday ingestion + otorite kuralı (B-F6 + NB-6)

Gelecekteki implementasyon, kabul edilmiş canonical commit'e dayanan izole
temiz bir worktree'de gerçekleşir. Aday transferi:

1. Canlı adayın SHA-256'sını yeniden hesapla.
2. Şununla tam eşitliği zorunlu kıl: `136438fdcf91cdae1a645b3fc229e26
   b6047d5a4bd53748f9a395a6ae70d4c79`.
3. Aday byte'larını, production path DIŞINDA izole, canlı-olmayan bir intake
   dosyasına BİR KEZ kopyala.
4. Intake hash'ini yeniden hesapla ve tam eşitliği zorunlu kıl.
5. Intake dosyasını byte-özdeş provenance kanıtı olarak koru.
6. Gelecekteki tracked modülü, doğrulanmış intake kopyasından, izole worktree
   içinde oluştur.
7. Canlı production path'i asla düzenleme/değiştirme.
8. Orijinal, intake ve ilk worktree hash'lerini raporla.
9. Canlı aday hash'i değiştiyse DUR.

**Aday otorite kuralı (NB-6):**
- hash-doğrulanmış intake dosyası yalnız **immutable evidence**'tır;
- Git deposu DIŞINDA ve canlı production path DIŞINDA kalır;
- bir runtime kaynağı DEĞİLDİR; asla import edilmez/çalıştırılmaz;
- ilk tracked modül oluşturulup baseline hash'i kaydedildikten sonra, **tracked
  worktree modülü tek implementasyon kaynağı olur**;
- sonraki tüm implementasyon düzenlemeleri yalnız tracked modülde yapılır;
- intake dosyası yalnız provenance karşılaştırması için byte-özdeş kalır;
- üretilen hiçbir patch, intake hash kaydını sessizce değiştiremez.

---

## B8. Contract-B durumu

**`BOOKTICKER_CANONICALIZATION_SECOND_CORRECTIVE_PREREGISTERED_PENDING_
INDEPENDENT_REREVIEW`**

Bağımsız rereview en az şunları doğrulamalıdır: (a) lock kimliğinin
DB-path+tablo olduğu, sembol setinin dahil OLMADIĞI (tek-yazıcı garantisi);
(b) singleton semantiğinin canonical `msvcrt` byte-range lock ile hizalandığı,
PID-liveness semantiği İÇERMEDİĞİ; (c) invalid-message pencere semantiğinin
tam/deterministik olduğu (iki saat, reset kuralları, sınırlar); (d)
migration'ın non-destructive donduruluduğu; (e) SQLite failure sözleşmesinin
gerçekçi (ambiguous-commit'siz) olduğu; (f) testlerin vacuous-olmadığı ve beş
izole path + sanitize ortam + PID-15332-referanssızlık kapsadığı; (g) B7
canlı-proses sınırının hiç ihlal edilmediği.

---

## B9. Düzeltme geçmişi — birinci corrective (B-F1..B-F6)

Kontrol eden bağımsız review: `BOOKTICKER_CANONICALIZATION_PREREGISTRATION_
CORRECTIVE_REQUIRED`.

| Bulgu | Şiddet | Birinci corrective |
|---|---|---|
| B-F1 | BLOCKER | `NUMBERED_MIGRATION_OWNED` seçildi |
| B-F2 | HIGH | Append-only politikası donduruldu |
| B-F3 | HIGH | Malformed-tick gerçek davranışı kaydedildi; tick-local/reconnect ayrımı (eksik exactness NB-3'te düzeltildi) |
| B-F4 | MEDIUM | shadow_mirror kaldırıldı; ancak lock kimliği+semantiği hatalıydı (NB-1/NB-2'de düzeltildi) |
| B-F5 | MEDIUM | 5 izole path donduruldu |
| B-F6 | LOW | 9-adım ingestion donduruldu (otorite kuralı NB-6'da tamamlandı) |

## B10. Düzeltme geçmişi — ikinci corrective (NB-1..NB-6)

Kontrol eden bağımsız rereview: `BOOKTICKER_CANONICALIZATION_CORRECTIVE_
REREVIEW_REQUIRED`. **Durum: corrective operator claims closed; pending
independent rereview.**

| Bulgu | Şiddet | Prior metin | Düzeltilmiş deterministik kural | Etkilenen testler | Kalan yetki sınırı |
|---|---|---|---|---|---|
| NB-1 | BLOCKER | Lock kimliği `db_path + symbol_set` → aynı DB'ye farklı-sembol eşzamanlı yazıcılar | §B5.2 kimlik = `NORMALIZED_ABS_DB_PATH + TABLE_NAME("book_ticker")`; sembol seti yalnız metadata; farklı semboller yine conflict | §B6.1/1-3,9; probe: sembol-set-geri-ekleme | — |
| NB-2 | HIGH | §B5.2 PID metadata + proc-creation-time + PID-reuse (shadow_mirror) canonical otoriteyle çelişiyordu | §B5.3 saf `msvcrt` byte-range lock ile hizalandı; PID-liveness/proc-time/stale-PID-takeover/PID-metadata-otorite KALDIRILDI; metadata yalnız non-authoritative | §B6.1/4-8; probe: PID-liveness ikamesi | İlk implementasyonda PID metadata elenmesi tercih edilir |
| NB-3 | HIGH | Threshold reset/pencere-reset/saat/5.-mesaj/flush-fail belirsizdi | §B4.6 iki saat (wall-clock validasyon, monotonic pencere) + 10-adım algoritma + health sayaçları + 10-ardışık-health-fail eşiği | §B6.1/10-20; probe: off-by-one, sayaç-reset | — |
| NB-4 | MEDIUM | Non-destructive migration invariant'ı donmamıştı | §B4.8 non-destructive sözleşme: temiz/uyumlu/uyumsuz tablo davranışı; YASAK destructive SQL listesi; migration-id preflight'ta seçilir; production read-only | §B6.1/21-25; probe: destructive SQL | Migration ayrı bağımsız review + preregistration gerektirir |
| NB-5 | MEDIUM | "Ambiguous commit" lokal sqlite için uygulanamaz/vacuous test | §B4.7 gerçek SQLite sözleşmesi; ambiguous-commit kaldırıldı; lock/busy retry + tükenmede typed sonlanma | §B6.1/26-27; eski #19 değiştirildi | — |
| NB-6 | LOW | Env-var non-inheritance + never-probe-15332 + intake/tracked otorite kuralı eksikti | §B6.2 sanitize ortam + PID-15332-referanssızlık; §B7.1 intake=immutable-evidence, tracked-modül=tek-kaynak | §B6.1/28-30; probe: intake runtime import | — |

Bu düzeltme kendi kendini kabul ETMEZ. Bir sonraki adım genuinely bağımsız,
taze-context bir rereview'dır.

---

## Kapsam dışı (bu sözleşmede DEĞİŞMEZ, implementasyon-yetkilendirilmemiş)

- `data/oi_spot_poller.py`, `tools/s34_v_engine_v02_shadow_mirror.py`,
  `tools/s34_state_machine_live_executor.py`, `tools/orderflow_chart.py`,
  `tools/s34_replay.py` — ayrı workstream'ler.
- Ek untracked shadow/live bağımlılıkları.
- CollectorSupervisor deployment, PID `20648` handoff, runtime alignment,
  full-stack launch, alpha/sinyal geçerliliği.
- `book_ticker` migration'ının kendi implementasyonu/deployment'ı — ayrı
  numaralı migration preregistration + bağımsız review gerektirir.

---

## B11. Operatör kabulü (2026-07-15)

**Durum: `BOOKTICKER_CANONICALIZATION_PREREGISTRATION_OPERATOR_ACCEPTED`.**

- **Bağımsız rereview verdikti:** `BOTH_SECOND_CORRECTED_PREREGISTRATION_
  CONTRACTS_ACCEPTED_PENDING_OPERATOR_SIGNOFF` / Contract B =
  `BOOKTICKER_CANONICALIZATION_ACCEPTED` (genuinely bağımsız, taze-context,
  salt-okunur rereviewer; ağaç review boyunca donmuş; sıfır mutasyon; aday
  hash `136438fd…4c79` bağımsız doğrulandı, değişmedi).
- **Operatör kabul tarihi:** 2026-07-15.
- **Bulgu kapanışı:** NB-1, NB-2, NB-3, NB-4, NB-5, NB-6 — tümü CLOSED
  (bağımsız rereview'da canonical kaynağa karşı doğrulandı; §B10 tablosu).
- **Açık implementation-planning advisory'leri (non-blocking, "düzeltildi"
  DEĞİL — planlama fazına AKTARILIR):**
  - **B-LOW-1 (lock-path normalizasyon sertleştirme):** sözleşme doğru
    tek-yazıcı invariant'ını dondurur ama Windows path-eşdeğerlik
    normalizasyonunu (realpath + case + junction/symlink/relative) tam
    enumerate etmez ve buna karşılık gelen testleri içermez. Planlama fazında:
    normalizasyonu canonical fiziksel-path stratejisine (realpath + Windows
    case normalizasyonu) pinle; relative/case/junction/symlink/eşdeğer-path
    yazımlarını test et.
  - **B-LOW-2 (indeks-tanımı eşdeğerlik sertleştirme):** migration
    non-destructive'dir ama uyumlu-indeks eşdeğerliğini (isim + kolonlar +
    uniqueness) açıkça tanımlamalı ve aynı-isim/yanlış-tanım indeksi
    fail-closed test etmelidir. Planlama fazında: indeks tanımını karşılaştır,
    uyumsuzsa fail-closed, tam davranışsal test ekle.
  - Bu advisory'ler NB-1…NB-6'yı YENİDEN AÇMAZ; kabulü engellemez;
    implementasyonu yetkilendirmez.

**Operatör kabulünün preserve ettiği güvenlik sınırları (değişmez):**
- Sözleşme kabulü implementasyon yetkisi DEĞİLDİR.
- Sözleşme kabulü migration oluşturma/çalıştırma yetkisi DEĞİLDİR; herhangi bir
  numaralı migration kendi ayrı preregistration + kapısını gerektirir.
- BookTicker aday'ı (`data/bookticker_collector.py`, PID 15332) NON-CANONICAL
  ve değişmemiş kalır; import/execute EDİLMEZ.
- PID metadata'nın hiçbir lock otoritesi YOKTUR (§B5.3); OS byte-range lock tek
  otoritedir.
- Hiçbir proses (PID 15332/20648 dahil) sinyallenmez/yeniden başlatılmaz;
  hiçbir Scheduled Task değiştirilmez; hiçbir canlı veritabanı
  (`data/microstructure.db`) mutasyona uğratılmaz.
- Bir sonraki izin verilen faz YALNIZCA ayrı, açıkça yetkilendirilmiş bounded
  **implementation-planning** kapısıdır — bu belge onu yetkilendirmez.

Bu operatör kabulü, §B2-§B7'deki substantive sözleşme davranışını DEĞİŞTİRMEZ;
yalnız preregistration'ı operatör-kabul-edilmiş terminal duruma reconcile eder.
