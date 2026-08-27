# Eclipse Scalper / AMI — Oturum Başlangıç Talimatı

> Bu dosya her yeni Claude oturumunda OTOMATİK yüklenir. Detay isteme sırası:
> önce `SYSTEM_STATE.md` (son bölümler en güncel), sonra göreve göre aşağıdaki haritadan.

## 🧭 ÇOK-HATLI ÇALIŞMA — BU OTURUM YALNIZ DEĞİL (2026-08-27, D-E6)
> **Başka Claude oturumları AYNI ANDA, AYNI dosyalarda çalışıyor.** Onları göremezsin.
> **Bir hat olarak çalışıyorsan (A / B / C / D) ilk üç dosyayı ve üç komutu ATLAMA:**
> `reports/atlas/LANE_CHARTERS_V1.md` (kapsamın, sınırın, DURDURMA KURALIN) ·
> `reports/atlas/LANE_MIND_PROTOCOL_V1.md` (hatlar birbirini nasıl görür, tek sayfa) ·
> `reports/atlas/CORPUS_AUDIT_PROMPT_V1.md` (satır şeması + kapalı verdict sözlüğü).
> ```
> python tools/lane_mind_v1.py --brief <HAT>    kendi son blogundan beri ne kacirdin
> python tools/lane_mind_v1.py --ct             acik celiskiler
> python tools/lane_mind_v1.py --who <terimler> BUNU DAHA ONCE KIM OLCTU?   <- acmadan once
> ```
> **`reports/atlas/_SHARED_LOG.md` KAYITTIR: append-only, ASLA düzenlenmez.** Düzeltme, neyi geri
> çektiğini adlandıran **YENİ bir bloktur**. Her tur sonunda bir blok EKLE (`to A/B/C/D` satırları
> `-` olsa bile dolu) **+** `SYSTEM_STATE`'e fenced ```verdict bloğuyla kapanan bölüm.
> **Başka hattın dosyasını DEĞİŞTİRME** — bulduğun kusur, o hatta yazılmış bir **bulgudur**;
> çelişkiler `CONTRADICTION_REGISTER.md`'ye. Kimlik **kararlı ID** (`D-E5`), `§` numaraları
> çakışır ve **asla yeniden numaralandırılmaz**.
> Tam onboarding ve her-tur prompt'ları: `reports/atlas/LANE_ONBOARDING_PROMPTS_V1.md`.

## İlk yapılacak
1. `SYSTEM_STATE.md` oku — sistemin TEK master durum dosyası (her önemli değişiklikten sonra güncellenir; en yüksek bölüm numarası = en yeni durum).
2. AMI işi ise: `docs/ami/AMI_ROADMAP.md` + `docs/ami/AMI_CHANGELOG.md` oku; canonical spec `AMI_ARTIFICIAL_MARKET_INTELLIGENCE_WHITEPAPER_v0.2.md`.
3. Research işi ise: ilgili rapor `reports/research/s34/` altında; geçmiş sonuçların meta-DB'si `reports/research/s34/S34_ALL.db`.

## Kesin guardrail'lar (ihlal edilemez)
- **🚨 `git push` YASAK — `origin` PUBLIC (2026-08-27, D-E19'da ölçüldü).**
  `origin = github.com/phoenixsenses/eclipse_scalper` ve `gh repo view` onu **PUBLIC** gösteriyor
  (orada zaten 17 707 dosya var). Bu depo **iç araştırma** taşıyor: `SYSTEM_STATE.md` tek başına
  **4.4 MB** eşik/sonuç/ön-kayıt. Bir push **geri alınamaz** — silsen de cache'lenir ve indekslenir.
  **Commit SERBEST** (dayanıklılık için gerekli), **push OPERATÖR ONAYI ŞART**.
  İç iş için doğru hedef ayrı bir remote: `phoenixsenses/eclipse_scalper_internal` (**PRIVATE**).
- `tools/s34_state_machine_live_executor.py`, `.env`, `execution/`, `risk/`, `brain/` — DOKUNMA (operatör sign-off şart).
- Leverage / ORDER_NOTIONAL / sizing — değişmez.
- Paralel Python/PowerShell prosesi ÇALIŞTIRMA (RAM çöker); araştırma scriptleri sırayla.
- pytest: en fazla 2 test dosyası/çağrı + `--basetemp` scratchpad'e + `-p no:cacheprovider` (repo tmp izin sorunu).
- Lookahead yasak; eşikler TRAIN'de seçilir, TEST raporlanır; FEE=5bps net; MC permütasyon standart.
- Kurulum/proje işleri D:\ sürücüsünde.
- Ana DB `data/microstructure.db` (650GB+) SALT-OKUNUR açılır (`file:...?mode=ro`).
- Mezarlığı tekrar test etme (buy-side fade, reversal, cross-asset transfer, gentleness, micro-timing, tight stop, partial exit, limit-entry) — tam liste: AMI failure archive + `docs/ami/AMI_RESEARCH_PROTOCOLS.md` §5.
- **🔓 MÜHÜR KISMEN AÇILDI — OPERATÖR YETKİSİ, `UNSEAL_TS 2026-08-23T11:42:24Z` (SYSTEM_STATE §334).**
  `echo_forward_ledger.jsonl`, `hold_horizon_forward_ledger.jsonl` ve
  `runtime/s34_bucket_live_harness_state.json` **artık okunabilir ve agregat hesaplanabilir**.
  Mühür öncesi tüm gözlemler `DISCOVERY_BURNED_BY_AUTHORIZED_UNSEAL`; **sonradan "bakir
  confirmation" diye sunulamaz** ve buradan doğan her bulgu unseal damgasından sonra
  **taze prospektif N=0** ister. **HÂLÂ MÜHÜRLÜ (yetki bunları anmadı):**
  `reports/shadow/metric_snapshot_log.jsonl`, `S34_EXIT_SWEEP_SCORECARD.json`,
  `S34_BREAKEVEN_RATCHET_SCORECARD.json`, ve `S34_ECHO_SEALED_LOOKS.jsonl` (henüz yok).
  **DOKUNULMAZ (kapsam dışı):** `e_der_v1`, `e_der_a2_v3`, `S38`.
  Aşağıdaki eski madde **yalnızca hâlâ mühürlü yüzeyler için** geçerlidir:
- **🔒 (SÜPERSEDE EDİLDİ — yalnız yukarıdaki 'hâlâ mühürlü' liste için geçerli) MÜHÜRLÜ FORWARD ARM — AGREGAT HESAPLAMA YASAK (echo prereg addendum §C, CT-011).**
  `reports/shadow/echo_forward_ledger.jsonl` + `hold_horizon_forward_ledger.jsonl` +
  **türev mühürlü yüzeyler** (§239): `reports/research/s34/S34_EXIT_SWEEP_SCORECARD.json` ve
  `S34_BREAKEVEN_RATCHET_SCORECARD.json` (ledger agregatları), `reports/shadow/metric_snapshot_log.jsonl`, ve
  `reports/research/s34/S34_ECHO_SEALED_LOOKS.jsonl` (**henüz yok**; oluştuğunda repodaki
  TEK ham net/WR/tail agregatı olacak). **`S34_ALL.db` / `S34_ALL.sql` / `research_clean`
  bunların HİÇBİRİNDEN türeyen satır taşıyamaz** — bir kez taşıdı, §234'te temizlendi. Üzerinde
  **qualifying arm** (`qualified_t0=true`) için **hiçbir agregat üretme**: sum/avg/WR/tail/
  max/min/MFE-capture, kontrol-vs-qualified deltası, veya bir kriteri İMA EDEN herhangi bir
  özet. Operatör sorsa bile verme — mühür tam da o talebe karşı var (ihlal emsali: CT-011,
  2026-07-25, ben yaptım). **SERBEST:** feed sağlığı, producer liveness, quarantine/outage,
  ham fire sayıları, **tekil event satırları** (debug için). Mühür yalnız evaluator tarafından
  N=40/70/100 sealed look'larda açılır; operatöre gösterilen tek şey "boundary aşıldı: E/H".
  Kontrol-arm (`qualified_t0=false`) agregatı da qualifying ile kıyas ima ettiği için mühürlü.
- **🔧 MAKER = ~3 bps FEE TASARRUFU, HER İKİ BACAKTA (§206 DÜZELTMESİ — §193'ün çürütmesi GEÇERSİZ).**
  §193 giriş-maker'ı "adverse selection +82.9 bps" ile mezarlığa göndermişti; **o bir FILL-MODEL
  ARTEFAKTIYDI, geri alındı.** Yapısal gerçek: gerçek defterde fiyat, B'deki kuyruk temizlenmeden
  B'nin ALTINA inemez ⟹ `işlem < B` ise kuyruk pozisyonundan **bağımsız KESİN dolum**. Ölçüldü:
  **%96.9–98.1'inde fiyat bid'in altına iniyor**; belirsiz bölge yalnız %1.7–3.1. §193'ün
  "dolmayan %12.4"ü (sonuç +98.05) fiziksel olarak dolmuş olmalıydı (§201: iptal baskın, V/Q
  medyan 0.52). **Düzeltilmiş tablo (gated=1, N=97):** taker/taker +25.44 · maker ÇIKIŞ **+28.50**
  · maker GİRİŞ **+28.44** ⟹ her iki bacakta **+3.0 bps**, tam olarak fee farkı kadar.
  **Spread yakalama YOK** (top-of-book 1 tick: ETH 0.0536, BTC 0.0156 bps) · **adverse selection
  cezası YOK** · zamanlama avantajı YOK. Ayakta kalan yerleşim kuralları (§205, hepsi reachability
  tabanlı): **touch'a post et** — derine post (X>0) monoton kötü, gecikmeli giriş kötü (%99.5
  dolum, kaçış yok), çıkışta merdiven monoton kötü. **Maker bir alfa değil, ~3 bps'lik bir
  maliyet kalemidir**; uygulaması `execution/` (DOKUNMA) + sign-off ister ve holdout-kanıtlı
  route yokken erkendir.
- **🧮 ARAMA KAPASİTESİ TÜKENDİ — YANMIŞ ÖRNEKLEMDE YENİ HİPOTEZ TESTİ KAPALI (§194/§199/§200/§202).**
  Üç bağımsız yol aynı sonuca varıyor: · **MinBTL:** 39 §F-geçerli gün → örneklem **~1 bağımsız
  deneme** taşıyor (N=10 için ~3 yıl gerek); yüzlerce yapıldı. · **Monoton çöküş:** kayıtlı 4 369
  denemede ort `avg_bps`, n-eşiği yükseldikçe +33.5→+23.1→+12.8→**+5.5** (gerçek edge'in örneklem
  büyüklüğüyle sistematik azalması için sebep yok). · **Çokluk düzeltmesi (Gumbel p95):** repo'nun
  EN İYİ tarihsel sonucu (`hour17...feature_ranking[3]`, n=59, WR 74.6, **mc_p=0.0, WF 5/5**)
  **anlamlı DEĞİL**; anlamlı olması için tüm program ≤**135 bağımsız fikir** olmalıydı (ölçülen: 472).
  *(§236 düzeltmesi: eskiden **73** yazıyordu — o rakam Gumbel ASİMPTOTİĞİNDEN geliyordu; N maksimumunun
  TAM dağılımı kapalı formda, `crit = σ·Φ⁻¹((1−α)^(1/N))`, ve doğru cevap **135**. **HÜKÜM DEĞİŞMİYOR**
  — 135 < 472, sonuç hâlâ anlamsız — ama sayı %85 hatalıydı.)*
  **Yöntem dış-doğrulandı:** aynı analiz hour17'nin forward başarısızlığını (CT-009) forward'ı hiç
  görmeden öngörüyor. · **Echo'ya uygulandığında:** 232 echo varyantı arasından seçilen causal
  **+41.2, hiçbir katman/kümelemede anlamlı değil** (null-max ortalamasının bile altında).
  **SONUÇ:** *"bir sonraki indikatörü/eşiği deneyelim"* matematiksel olarak kapalı — her yeni feature
  N'i artırır, gereken veriyi artırır, veri sabittir. **mc_p=0 + WF 5/5 tek başına HİÇBİR ŞEY ifade
  etmez.** Forward'ın sınırı da net: 12 ay ≈ **2** bağımsız hipotez. Echo forward'ı bu çürütmez ama
  **beklentiyi sıfıra yakın kılar** → §G futility/güç kalibrasyonu in-sample +41.2'ye göre YAPILAMAZ.
- **✅ N TÜKETMEYEN İŞ SINIFI (tek meşru ilerleme yolu).** Yeni hipotez yerine: **(a) replikasyon** —
  aynı donmuş kural, keşfinde kullanılmamış sembolde (SOL denendi §203: `NOT_REPLICATED` p=0.13 ama
  yön ETH-tutarlı, 6h +37.8 vs kontrol −1.4; güç düşük N=35). **(b) muhasebe/veri-bütünlüğü denetimi**
  (funding §195, feed outage §191). **(c) çokluk düzeltmesi** (§200). **(d) örneklem UZUNLUĞU ekleyen
  her şey** — payda büyür, pay büyümez.
  **🎯 KULLANILMAYAN VARLIK: `liquidations` tablosunda 761 SEMBOL var** (2026-06-06'dan; tokenize
  hisse/emtia dahil: XAUUSDT, XAGUSDT, NVDAUSDT, AAPLUSDT, QQQUSDT). Fiyat feed'i yalnız 3 sembolde
  (BTC/ETH/SOL) → diğerleri için Binance klines ücretsiz (`data.binance.vision/data/futures/um/monthly/klines/`,
  758 sembol × 2 ay ≈ **1 GB**). **Çapraz-kesit replikasyon HAVUZLANMIŞ tek test + semboller-arası
  işaret testi olarak kurulmalı** — sembol sembol bakıp en iyisini seçmek §200'ün suçunu tekrarlar.
- **⚠️ TOP-OF-BOOK KUYRUK MODELİ GEÇERSİZ (§201, ölçüldü).** *"Seviye ancak agresif akış displayed
  qty'yi aşınca tükenir"* varsayımı yanlış: seviye kırıldığında görünen miktarın medyan **%52'si**
  işlem görmüş; **%47.8'inde iptal baskın**, %18.3'ünde refill; dağılım p10=0.02 → p90=4.42 (iki
  mertebe). Kaskad anında best_bid'in medyan ömrü **1.03 sn**. ⟹ V1/V2 fill oranları olay-bazında
  güvenilmez (yön: gerçek dolum modellenenden ÇOK). **Reachability (§198) ETKİLENMEZ** — kuyruk
  tüketimi kullanmaz, yalnız "fiyata bir işlem oldu mu" sorar; %98.8 ayakta.
- **📉 VERİ BÜTÜNLÜĞÜ — TARİHSEL OUTAGE PENCERELERİ (§191, ölçüldü).** Bu tarihlerde tarihsel
  araştırma yapmadan önce feed'i KONTROL ET:
  · `liquidations` **2026-04-28 → 06-05 TAM ÖLÜ** (39 gün, tüm semboller; "May 40-day gap"in
    tarih-kesin karşılığı) → o pencerede anchor doğamaz.
  · `book_ticker` **2026-06-06 → 06-11** (~5.3 gün, ayrı outage; measured-cost fiyat kaynağı).
  · `agg_trades` aynı genel pencerede kısmi bozulma.
  · §F eşiği: gün içinde **≥3 sıfır-saat ⟹ INVALID**; ayırıcı **yoğunluk değil EKSİK SAAT**
    (yoğunluk dağılımları örtüşüyor). Burned döneme uygulanınca `liquidations` **44/83 gün INVALID**.
  · **`mechanism_store` KONTROL KOLU KİRLİ:** `is_event=0`'ın **248/418'i** ölü pencerede
    (feed ölü olduğu için "olay yok" işaretlenmiş) → **temiz kontrol 418 değil 170**. Sinyal kolu
    (`is_event=1`, `gated=1`) TEMİZ. Event-vs-control kıyası yapan sonuçlar etkilenmiş.
  · Kurtarma doğrulandı (§192): Tardis `binance-futures/liquidations` bizim veriyle %1 içinde
    örtüşüyor, sign-flip yok. **Frozen DB'ye YAZMA** — ayrı store + opt-in union okuma.
- **✅ KANONİK FEE VARSAYIMI (karar 2026-07-25, §197): `BINANCE_BASE`.** Operatör VIP kademesini
  yazılı teyit edene kadar **taker 5.0 bps/taraf (=10.0 round-trip), maker 2.0 bps/taraf (=4.0)**
  varsayılır. Gerekçe: muhafazakâr yön **güvenli** — gerçek kademe daha iyiyse sonuçlar
  olduğundan kötü görünür, ki bu asla yanlış-pozitif üretmez. Yeni her ölçüm bu tabanı
  **açıkça** raporlar (gizli default YOK). Eski 5.0/8.0 tabanlı sonuçlar bu kararla
  düzeltilmez — yalnız **çapraz kıyaslanamaz** olarak kalır (CT-012).
- **💱 FEE SABİTİ TUTARSIZ (CT-012) — sonuçları ÇAPRAZ KIYASLAMA.** Aktif kodda round-trip taker:
  `5.0` (echo/hold_horizon/attribution/buyfade) vs `8.0` (btc_lead/daytrend/cascade) vs `2.0`
  (`DEFAULT_MAKER_FEE_BPS`×2). Binance **baz**: taker 5.0/taraf = **10.0 round-trip**, maker
  2.0/taraf = 4.0. `.env`'de fee anahtarı TANIMLI DEĞİL. Operatörün gerçek VIP kademesi yazılı
  teyit edilene kadar 5.0-tabanlı ve 8.0-tabanlı sonuçlar kıyaslanamaz. (echo `gated1` fee'ye
  dayanıklı: gross +34.2 → +24.2@10bps.)
- **Maker/quote-quality ailesi PARK (`MAKER_LINE_RESEARCH_ONLY_PARKED`, SYSTEM_STATE §146):** book-state depletion/refill, trade-flow shock, liq quote-suppression, SELL-burst maker → 4/4 sub-fee / not-harvestable; deploy edilmiş best-of-book maker motoru YOK + deep-bid fill top-of-book'tan doğrulanamaz. **Yeniden açma yalnız gerçek bir best-of-book maker motoru kurulursa.** Book-state Stage-1 geometrisi reproducible epistemik varlık olarak arşivde.

- **🛑 PROSES SONLANDIRMA — "SADECE BU İŞİN SAHİP OLDUĞU PID" (2026-08-21 olayı).**
  Temizlik **asla** *"korunan PID hariç her şey"* şeklinde yazılmaz; **daima** *"yalnız bu işin
  başlattığı tam PID"* olarak yazılır. `Get-Process python | Where-Object {...} | Stop-Process`
  kalıbı — filtre olsa bile — YASAK: 2026-08-21'de tam bu kalıp (13224 hariç) **24 prosesi**
  öldürdü, tüm Eclipse runtime yığınını düşürdü ve geri getirilemez forward-only gözlem kaybetti.
  Öldürmeden önce PID + komut satırı + oluşturma zamanı gösterilip işin çıktısına eşlenir.
  Zorla öldürme **bayat lock** bırakır (`runtime/*.lock`) ve rolün kendi restart'ını engeller.
  Olay müdahalesinde `start_eclipse.ps1` ÖNCESİ `logs/pids/*.json` snapshot'lanır — script
  onları üzerine yazıp olay-öncesi PID↔rol haritasını yok eder.
  Tam kural: `docs/OPERATOR_PROCESS_SAFETY.md` · olay kaydı:
  `reports/governance/incidents/2026-08-21_python_mass_kill_incident/`.

- **📏 ÖLÇÜM MERDİVENİ — HER SORU BU SIRAYLA SORULUR (§380, 2026-08-24).**
  Bir "piyasa sonucu" sandığın şey önce bir **ölçüm sonucu** olabilir. Basamakları
  karıştırmak yanlış hikâye seçtirir:
  ```
  MARKET QUESTION -> OBSERVABILITY -> MEASUREMENT FIDELITY -> TARGET SEMANTICS
                  -> STATISTICAL INFORMATION -> MECHANISM -> ECONOMICS
  ```
  Emsal: L1 hattında **fidelity mükemmel** çıktı (`%99.975` doğru olay), ama **target
  semantics kaba**ydı (bir `aggTrade` medyan **3 ham işlem** sıkıştırıyor) — ikisi ayrı
  basamak, ve zayıf sonucun sebebi ikincisiydi.
- **🚫 `UNOBSERVED ≠ ZERO` — YOKLUK ANCAK GÖZLEMLENEBİLİRLİK KANITLANDIKTAN SONRA VERİDİR.**
  Bir pencerede kayıt görmemek "olay olmadı" demek değildir. Emsal (§376/§380): `2026-06-05`'te
  `31 508` mid değişimi *"öncesinde işlem yok"* diye sayıldı; oysa işlem feed'i o saatlerde
  hiç örtüşmüyordu. Boş-veri koruması bir günü/pencereyi düşürüyorsa bu bir **OLAYDIR**,
  loglanır ve `N`'in neden eksik olduğu denetlenir — sessizce geçilmez.
- **🔗 `MULTISTREAM_JOIN_HEALTH_GATE` — BİREYSEL FEED SAĞLIĞI ≠ ORTAK GÖZLEMLENEBİLİRLİK.**
  Çok-feed'li her çalışma, **herhangi bir sonuç okunmadan önce** şunu yayınlar: her feed'in
  span + saniye kapsamı · **ORTAK** saniye sayısı/oranı · iç boşluklar · ZERO/PARTIAL/THIN
  günler · kullanılabilir olay sayısı. `FULL` = ortak saniye ≥ `%95` (hedeflenen takvim
  gününe göre; "küçük feed büyüğün içinde mi" DEĞİL).
- **🧪 KAPI TANIMININ NULL DEĞERİNİ BİLMEDEN PASS/FAIL DONDURMA.** Bir eşiği/kapıyı
  dondurmadan önce **beklenen değeri** (gürültüde veya sağlıklı veride) ölç. Emsal (§380-C):
  `f_t == l_{t-1}+1` sürekliliğini kapı yaptım, 16/16 FAIL verdi — oysa null değeri sıfır
  değildi (borsa ID tahsisi `%0.156` boşluk üretiyor). Kapı yanlıştı, veri değil.

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

## 📚 KİTAP KÜLLİYATI — DİSKTE, HER ZAMAN KULLANIMA HAZIR (CKR-01, 2026-08-26)
> **13 KAYNAK · 4 299 SAYFA · 3 985 metin sayfası — hash'lenmiş, çıkarılmış, aranabilir.**
> Üç ayrı geçiş (§416, §417, PVE-01) külliyatın yokluğunu **YANLIŞ** raporladı. Bir daha
> *"diskte kitap yok"* DEME — önce buraya bak. **Çıkarılmış metinler:
> `data/literature_v2/text/*.txt`** (tek seferlik `pypdf`; doğrudan grep'le).
> Defter: `reports/literature/ECLIPSE_BOOK_ACCESS_LEDGER_V2.md` ·
> 13 kitap notu: `reports/literature/full_corpus_v2/*_FULL_RESEARCH_NOTES_V2.md` ·
> Sentez: `ECLIPSE_FULL_LITERATURE_SYNTHESIS_V2.md` · Crosswalk:
> `ECLIPSE_LITERATURE_RESULT_CROSSWALK_V1.md`

| # | kaynak | yazar | sayfa | ne için |
|---|---|---|--:|---|
| 1 | STK4080 Survival & Event History (72 PDF) | UiO | 793 | counting process · competing risks · left truncation |
| 2 | **Survival and Event History Analysis (ABG)** | Aalen/Borgan/Gjessing | 550 | **STK4080'in atıf yaptığı ders kitabı · frailty (603 hit)** |
| 3 | **The Science of Algorithmic Trading** | Kissell | 492 | **implementation shortfall ARİTMETİĞİ · TCA** |
| 4 | Trades, Quotes and Prices | Bouchaud ve dig. | 463 | kuyruk yarışı · impact · icra |
| 5 | **Causal Inference: What If** | Hernán & Robins | 365 | **confounding vs effect modification vs interaction** |
| 6 | **Algorithmic and High-Frequency Trading** | Cartea/Jaimungal/Penalva | 360 | **fill probability `exp(-κδ)` · optimal execution** |
| 7 | Econophysics of Order-driven Markets | Abergel (ed.) | 326 | tick/spread |
| 8 | Limit Order Books | Abergel ve dig. | 241 | queue-reactive · subordination |
| 9 | Algorithmic Trading | Chan | 225 | backtest overfit · latency |
| 10 | Empirical Market Microstructure | Hasbrouck | 209 | kalıcı/geçici · shortfall teorisi |
| 11 | Machine Learning for Asset Managers | López de Prado | 152 | False Strategy teoremi |
| 12 | Trading and Exchanges (**TASLAK, 113 s.**) | Harris | 113 | piyasa yapısı |
| 13 | Honoré (1993) | Honoré | 10 | çok-spell tanımlama |

**DİSKTE YOK (okundu İDDİA ETME):** O'Hara *Market Microstructure Theory*, ISLP.
**Sayfa sayısı PDF page object'inden** alınır — form-feed bölmesi Abergel'i 241→390 şişirmişti.

**KARAR VEREN YEDİ PASAJ:**
1. **Bouchaud §11.2** — `gözlenen = reaksiyon + öngörü`; karşı-olgu kurulamaz, özel veri ön koşul
   ⟹ **reaksiyon etkisi HİÇBİR kamusal Eclipse örnekleminde tanımlanamaz.**
2. **Bouchaud §7.5** — yarış sonrası ortalamaya dönüş MEKANİK öngörüdür; §417 ayırıcısının
   **işaretini** önceden bu verdi.
3. **Bouchaud §21.3.2-3** — martingal altında limit emir shortfall'ı d'den bağımsız **TAM SIFIR**;
   `E[Δ]=φ_exec·d·ρ`. §206 ile birebir uyuşuyor.
4. **Cartea böl.8** — `P(δ derinliğine konan emir alınır) = exp(−κδ)` ⟹ **Bouchaud'nun ölçülmemiş
   bıraktığı `φ_exec` terimini verir**; PVE-01'in örtük tuttuğu çarpan artık hesaplanabilir.
5. **Hasbrouck böl.10 + trading costs** — information share'ler **SINIRDIR** (toplulaştırma
   genişletir); shortfall = icra + **fırsat maliyeti**, ikincisi sıfır-toplamlı DEĞİL.
6. **Kissell böl.3** — `IS = (S·Pn − S·Pd) − (Σsⱼ·Pn − Σsⱼpⱼ − fees)` ⟹ Hasbrouck'ın çerçevelediği
   şeyin **ARİTMETİĞİ**. `Pd` = karar-anı fiyatı ⟹ **niyet kaydı olmadan IS tanımsızdır.**
7. **Hernán & Robins Teknik Nokta 4.2** — kovaryatlar arası tüm çarpım terimlerini içeren ama
   **tedavi × kovaryat çarpımı İÇERMEYEN** model, L'de doygundur ama **kurgu gereği
   effect-modification YOKTUR varsayar** ⟹ PVE-02'nin tam olarak düştüğü tuzak.
   **`λ₀ₕ=α₀ₕ·Y₀`, frailty selection, Aalen additive: ABG.**

**BU KİTAPLARIN GEÇERSİZ KILDIĞI İKİ YAYIMLANMIŞ İDDİA:**
· **PVE-02'nin "ders kitabı ebeveyni yok" iddiası YANLIŞ** — Hernán & Robins Teknik Nokta 4.2 tam
  olarak bunu söylüyor. Türetme doğruydu, **yenilik iddiası değildi** ⟹
  `TEXTBOOK_CAUSAL_INFERENCE_INDEPENDENTLY_REDISCOVERED`.
· **§408-B4'ün `TIME_VARYING_GEOMETRY_HAZARD_SUPPORTED` etiketi ÇEKİNCE İSTER** — ABG'ye göre
  **frailty selection** hiçbir bireysel zaman değişimi olmadan tam olarak o görüntüyü üretir; ve
  Eclipse'in kendi MPH kapısı K=10 günün frailty şeklini tanımlayamayacağını zaten kaydediyor
  ⟹ iki açıklama **mevcut veriyle ayrıştırılamaz.**

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
| **Canonical operator dashboard** | `tools/s34_cascade_navigation_dashboard.py` — Eclipse'in canonical, sürekli kullanılan operator güvenlik/karar-destek yüzeyi (read-only, DB `mode=ro`, trade/order/cancel/executor/scheduler/process kontrolü YOK, fail-closed). Yeni operasyonel/güvenlik panelleri varsayılan olarak BURAYA entegre edilir. Supersession ancak bağımsız-kabul edilmiş governance migration ile. Dashboard işi read-only + test/review guardrail'larına tabi (SYSTEM_STATE §133). |
| Secondary/diagnostic dashboard'lar | `tools/s34_live_chart.py` (:5050), orderflow chart (:5051), S34 replay (:5052), **iki-lead monitör `tools/s34_leads_monitor_dashboard.py` (:8771)** — canonical yüzeyin YERİNE GEÇMEZ. Leads monitör: echo(causal)+hour17 forward trade izleme, read-only GET/HEAD, DB `mode=ro`, outage/gap artifact karantinası (§141). **§262'den beri OPT-IN: `-EnableLeadsMonitor`** — default OFF, çünkü ölçüldü: 1.64 GB/dk (22 dk'da 36 GB), ve yoklayan tek şey bir VS Code preview paneliydi. |
| Shadow runner | `tools/s34_realtime_shadow_runner.py` |
| **FORWARD LEDGER'LAR (paper, read-only, un-burned birikim — BURADA ARA)** | **echo:** `tools/research_s34_echo_forward_ledger.py` → `reports/shadow/echo_forward_ledger.jsonl` (+`_state.json`) — echo_30_90+regime, 4h, `qualified_t0`(causal) vs `qualified_full`(lookahead) + indikatör snapshot. **hold-horizon:** `tools/research_s34_hold_horizon_forward_ledger.py` → `reports/shadow/hold_horizon_forward_ledger.jsonl` (+`_state.json`) — hour17 & echo(causal) × **2/4/6/12/24/48h** × **nostop/−150/−300 stop** (her RESOLVE'da `net_bps`,`net_bps_s150`,`net_bps_s300`). Forward-only (post-2026-07-20), FEE=5bps. start_eclipse rolleri `echo_forward_ledger` + `hold_horizon_forward_ledger` (default ON). **İzleme:** leads monitör dashboard :8771 (**opt-in `-EnableLeadsMonitor`, default OFF** §262). **Tarihsel causal karşılığı** (BURNED, kanıt değil): `reports/research/s34/S34_HOLD_HORIZON_SWEEP.md` + `S34_ECHO_CAUSAL_VS_LOOKAHEAD.md`. Detay: SYSTEM_STATE §166/§167. |
| Proses yönetimi | `start_eclipse.ps1` / `stop_eclipse.ps1` / `status_eclipse.ps1` (live executor `-EnableLive` bayrağıyla — default KAPALI). Forward ledger'lar default ON gelir; **leads dashboard §262'den beri default OFF (`-EnableLeadsMonitor`)**; canonical dashboard :8770 default ON kalır; reset sonrası kalıcılık için start_eclipse.ps1 tekrar çalıştırılır (boot auto-start YOK). |
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

## Canlı Lead & Route Durum Defteri (YAŞAYAN — kaybolmasın, buraya devam et)
> Operatör talebi (2026-07-20): "elimizde olanların bunlar ve bunlardan sonrakiler
> tamamen kaydedilip kaybolmasın." Yeni her route/lead değerlendirmesi bu tabloya
> **satır olarak eklenir** (silme yok; çürüyen → durum güncellenir). Kanonik sayı
> kaynağı ilgili `reports/research/s34/*` raporu; bu tablo özet + işaretçi.

**Ayrım (kritik):** "Canlı motor" = `s34_state_machine_live_executor.py` (default-OFF,
üçlü-gate). "Canlı lead" = alpha anlamında hâlâ yanmamış tek hipotez = echo_30_90+regime,
ama TRADE EDİLMİYOR — sadece `research_s34_echo_forward_ledger.py` ile forward izleniyor.
İkisi AYRI. Bugün ne motor ne echo holdout-proven.

### A) Canlı motor — GERÇEK trade tablosu
- `runtime/s34_v_engine_live_state.json`: `active=None`, orders=0, pending=0, son
  güncelleme 2026-07-01 (bayat). LIVE modda koştuğu dönemde her tick
  `no_fresh_eligible_anchor` → **SIFIR gerçek trade, kazanan yok.**

### B) Shadow (PAPER, forward) route tablosu — 2026-07-19→20, küçük N
| Route | N | WR% | avgNet | totNet |
|---|--:|--:|--:|--:|
| LONG_HOUR17_HOLD6H | 3 | 100 | +74.2 | +222.7 |
| LONG_HOUR17_COMPOSITE | 1 | 100 | +95.1 | +95.1 |
| BUY_FADE_SHORT_H45_SL75 (mezarlık) | 3 | 66.7 | +24.8 | +74.5 |
| LONG_ECHO_45_120_SILENCE | 1 | 100 | +48.5 | +48.5 |
| LONG_OFI_SILENCE_BUYERS | 1 | 100 | +48.5 | +48.5 |
| LONG_T15_BOUNCE | 2 | 50 | +13.6 | +27.3 |
| SHORT_NOISY_BTC1M_D5_H180 | 2 | 0 | -86.5 | -173.0 |
| SHORT_NOISY | 5 | 20 | -35.1 | -175.7 |
> Sadece 1 gün + kâğıt; hour17 LONG kolu net+, short kolları net−. İstatistik iddia DEĞİL.

### C) echo_30_90+regime — GEÇMİŞTE, LOOKAHEAD'SİZ (causal) — asıl bulgu
Kaynak: `reports/research/s34/S34_ECHO_CAUSAL_VS_LOOKAHEAD.md` (2026-07-20). `not noisy`
gate = T+30m LOOKAHEAD (hindsight'la tam olarak kuyruklu event'leri atıyor). Tek o gate
toggle'landı, evren=695 anchor 5.15 ay, FEE=5bps:

| arm | N | /ay | WR | Avg | Worst | Tail(<-100) | mc_p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| FULL (lookahead'lı) 4h | 38 | 7.4 | 81.6% | **+87.8** | -85.9 | **0** | 0.0 | 5/5 |
| CAUSAL (lookahead'sız) 4h | 118 | 22.9 | 69.5% | **+41.2** | **-338.9** | **14** | 0.001 | 5/5 |
| CAUSAL 6h | 118 | 22.9 | 69.5% | +49.0 | -412.4 | 12 | 0.0 | 5/5 |
| noov CAUSAL 4h | 63 | 12.2 | 61.9% | — | — | — | (sum +560.5) | — |

**Yorum:** Pristine "+92.5/tail0" büyük ölçüde LOOKAHEAD'di — causal'da avg ~yarıya
düşüyor (+87.8→+41.2) ve 14 felaket-kuyruk (worst -338.9) ORTAYA ÇIKIYOR. Yine de causal
arm in-sample POZİTİF ve mc0.001 → **KILL değil ama BLESS de değil** (diğer gate'ler de
burned sample'da seçildi → necessary-not-sufficient). Gerçek kanıt = FORWARD (post-2026-07-20).

### D) Tüm route ailesi — GEÇMİŞTE causal gauntlet verdict'leri
Kaynak: `reports/research/s34/S34_ECHO_LIVE_GAUNTLET.md` (no-overlap + verdict):
| Aday (causal, T0) | noov N | noov /ay | noov WR | verdict |
|---|--:|--:|--:|---|
| F_e3090_T0 (echo_30_90+regime) | 26 | 5.7 | 76.9% | **PAPER_CANDIDATE** (tek) |
| F_e3090_t15 | 24 | 5.3 | 62.5% | RESEARCH_ONLY |
| F_e30120_T0 | 35 | 7.7 | 68.6% | SHADOW_ONLY |
| F_e30120_t15 | 33 | 7.3 | 54.5% | RESEARCH_ONLY |
| F_e30120_prebuild | 27 | 6.0 | 77.8% | SHADOW_ONLY |
> NOT: gauntlet'in F_e3090 satırı hâlâ `not noisy` (lookahead) içeren cand_9090 — (C)
> deki causal düzeltme bunun ÜstÜne okunur. hour17=§142 forward-unsupported; silence=FALSIFIED;
> v_engine=BLOCKED n=0; buy_fade=tarihsel negatif (mezarlık). **Hiçbiri deploy/holdout-proven değil.**

### Sıradaki (kaybolmasın)
- FORWARD echo ledger biriktir (qualified_t0 causal vs qualified_full lookahead ayrı) —
  `reports/shadow/echo_forward_ledger.jsonl`, post-2026-07-20, re-mine YOK.
- İlk cevaplanacak: causal echo, noisy gate olmadan forward'da yaşıyor mu? (tail yönetimi şart.)
