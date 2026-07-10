# S34 HOUR17 — Cycle-Adjusted Recompute + May 2026 Liquidation-Feed Gap Forensic

**Tarih:** 2026-07-11 (iki oturum: ilk salt-okunur recompute + adli inceleme; ardından operatör yetkisiyle tek-satır `K-S34-HOUR17-001` KO düzeltmesi — bkz. §7). Hiçbir kod/eşik/route/runtime/checkpoint/PID değişmedi, hiçbir proses restart edilmedi, `canonical.sqlite` DEĞİŞTİRİLMEDİ.

## 0. K-S34-HOUR17-001 KO Düzeltmesi — UYGULANDI (§7'ye bakınız)

Bu raporun ilk sürümü (§1-6) salt-okunuydu; `knowledge.sqlite` DEĞİŞTİRİLMEMİŞTİ. Operatörün açık yetkisiyle o KO artık düzeltildi (kanonik `KnowledgeStore.put()`/`touch_version()` API'si, el yazması SQL yok) — tam before/after, checksum'lar ve doğrulama §7'de.

## 1. Önceki "68 event → 53 cycle" rakamı — GEÇERSİZ (`INVALID_WRONG_POPULATION`)

SYSTEM_STATE §101/4, OPERATOR_DECISION_QUEUE OD-011 ve IMPLEMENTATION_PROGRESS_LEDGER'daki batch satırında kayıtlı "HOUR17 popülasyonu: 68 event → 53 bağımsız cycle (deflasyon 0.78)" rakamı **yanlış popülasyondan üretildi**. Kanıt: `data/ami/canonical.sqlite:ami_events` tablosunda **hiçbir satır `LONG_HOUR17_HOLD6H`/`LONG_HOUR17_COMPOSITE`/`LONG_HOUR17_100K_COMPOSITE` route'una etiketli değil** — tablonun `route_version` değerleri yalnız HOUR17-öncesi route adları (`LONG_SILENCE`, `SHORT_NEITHER`, `BUY_FADE_SHORT_H45_SL75` vb.). O rakam, `ami_events`'teki TÜM anchor'ları route'tan bağımsız yalnızca "anchor saati≥17" ile kaba filtrelemekten üretilmişti — gerçek K-S34-HOUR17-001 kanıt popülasyonuyla hiçbir ilgisi yok. **Bu rakam bir daha HOUR17 kararlarında KULLANILMAMALI.** Eski kayıt tarihsel olarak korunuyor (silinmedi), yalnız bu rapora ve aşağıdaki doğru rakama işaret edilecek şekilde üstüne bir supersession notu ekleniyor.

## 2. Doğru donmuş iddia kimliği

- **KO:** `K-S34-HOUR17-001` (`data/ami/knowledge.sqlite`, status `FORWARD_VALIDATING`, `frozen: false`)
- **Deployed route:** `LONG_HOUR17_HOLD6H` (`tools/s34_realtime_shadow_runner.py:612-645`) — ETHUSDT SELL cascade ≥200K anchor, gate = not-bull ∧ not-EUROPE ∧ (btc4h<0 ∨ btc7d<0) ∧ hour≥17 (hardcoded), entry T0, hold sabit 6h, TP/SL/BE YOK, fee=5bps mark-fill, **shadow-only (sipariş yok)**.
- **KO payload düzeltmesi gerekli:** payload'daki `effect_size` (avg=40.8bps, wr=0.615) aslında araştırma script'inin **150K eşik** konfigürasyonuna (`S5_150K_pred_FULL`, N≈156) ait — hem KO metninin hem deploy edilen kodun kullandığı **200K + hour≥17** kombinasyonuna tam eşleşen yayınlanmış konfigürasyon **`S6_200K_full`**'dur (N=126, WR=65.1%, avg=+39.8bps, mc_p=0.0, wf=5/5). `knowledge.sqlite` bu batch'te DEĞİŞTİRİLMEDİ; KO payload düzeltmesi ayrı, açıkça yetkilendirilmiş bir yazma işlemi gerektirir.

## 3. Bağımsız-cycle recompute (doğru popülasyon)

Kaynak: `tools/research_s34_silence_predictor.py`'nin kendi metodolojisi (`build()`, `reconstruct_anchors`, TRAIN/TEST 70/30) `data/microstructure.db` (mode=ro) üzerinden birebir reprodükte edildi (doğrulama: 456≈454, 375≈373, N=127≈126, WR=65.4%≈65.1%, avg=+40.2≈+39.8 — küçük fark "now" kesim anındaki ~9 saatlik gecikmeden). Ardından `ami/identity/cycle_resolver.py`'nin canonical-v1 tanımı (4h continuity gap + point-in-time 1h StructureState, `ami.states.engine.StateEngine`) BİREBİR import edilerek (yeniden türetilmeden) uygulandı.

**Sonuç: 127 raw event → 93 bağımsız cycle (deflasyon 0.732).** Cycle boyutu: 73 singleton, 20 multi-event, max 5, ortalama 1.37. 0 direction_conflict.

**Cycle-düzeyi birincil istatistikler (N=93):** WR=62.4% (58W/35L), mean net=+32.47bps, median net=+24.04bps, cum net=+3019.4bps, stdev=126.4, MFE ort/med=+135.2/+101.6bps, MAE ort/med=-99.5/-77.4bps, top1/top3/top5 payı=%15.0/%37.3/%54.5.

**Kronolojik fold'lar:** TRAIN(ilk %70, n=65) WR=60.0% mean=+24.31; TEST/holdout(son %30, n=28) WR=67.9% mean=+51.40 — **holdout train'i geçiyor**, decay yok. (Not: fold sınırı Mayıs boşluğunu kapsıyor — bkz. §5.)

**Concentration/leave-out:** en iyi gün %19.8, en iyi hafta %26.2, top-3 gün payı %48.2 (cycle'ların yalnız %7.5'i). En-iyi-cycle/en-kötü-cycle/top3-cycle/en-iyi-gün/en-iyi-hafta çıkarıldığında sonuç HİÇBİR senaryoda negatife dönmüyor (kalan mean +21..+38bps aralığında).

**Bağımlılık-farkında bootstrap (hafta-blok, seed=42, 2000 iter):** mean 95% CI [3.1,56.3] P(mean>0)=0.986; median CI [1.0,43.9] P(median>0)=0.984; cum CI [290.4,5240.2] P(cum>0)=0.986. Naif IID'ye göre CI genişlemesi yalnız 1.05× — bağımlılık düzeltmesi sonucu ciddi değiştirmiyor.

**Monte Carlo (hafta-blok path, 2000 path):** terminal cum p5/p50/p95 = +714.4/+3016.7/+5190.6bps; **paths_below_zero=%1.5**. Max drawdown p5(en kötü)=-1334.4bps.

**Maliyet duyarlılığı:** fee 5→15bps (3×) sonrası hâlâ pozitif: mean +22.5bps, cum +2089.4bps, pozitif-oran %55.9.

**Event-vs-cycle:** ham event-düzeyi görünüm N'i %37, WR'yi 3pp, toplam edge'i %41 ŞİŞİRİYORDU — cycle-düzeltmesi olmadan iddia gerçekte olduğundan güçlü görünüyordu.

**İleri (forward) shadow kanıtı (bu recompute anında, salt-okunur):** `reports/shadow/s34_state_machine_shadow_state.json`'daki gerçek pnl objesi — `LONG_HOUR17_HOLD6H`: **N=1, net=-57.0bps, WR=0%**. `LONG_HOUR17_COMPOSITE`/`LONG_HOUR17_100K_COMPOSITE`: N=0. Tek gözlem istatistiksel olarak anlamsız; ne tarihsel bulguyu doğrular ne çürütür.

**Sınıflandırma: `HOUR17_HISTORICAL_CYCLE_EVIDENCE_POSITIVE_BUT_FRAGILE`.** Nokta tahminleri gerçekten pozitif ve bağımlılık/maliyet/kronoloji testlerine dayanıklı, ama cycle-düzeyi yoğunlaşma (top5=%54.5) ve aşağıdaki veri sürekliliği boşluğu nedeniyle "fragile" — "ready for frozen forward shadow" değil.

## 4. Mayıs 2026 ETH likidasyon boşluğu — ADLİ BULGULAR

**Gerçek sınır (gün-bazlı değil, saniye hassasiyetinde):**

| Sembol | Son geçerli satır (boşluk öncesi) | İlk geçerli satır (boşluk sonrası) |
|---|---|---|
| ETHUSDT | 2026-04-27 14:27:26.345 UTC | 2026-06-06 17:47:03.630 UTC |
| BTCUSDT | 2026-04-27 14:27:26.345 UTC | 2026-06-06 17:47:04.094 UTC |
| SOLUSDT | 2026-04-27 14:24:51.798 UTC | 2026-06-06 17:47:05.191 UTC |

Üç sembol de boşluğa saniyeler içinde birlikte giriyor ve saniyeler içinde birlikte çıkıyor → **tek, paylaşılan bir alt-nedeni işaret ediyor**, sembole özgü değil. **Gerçek süre ≈ 40 gün 3 saat** ("sadece Mayıs" ifadesi yanlış — boşluk 27 Nisan öğleden sonra başlıyor, 6 Haziran akşamına kadar sürüyor).

**Sınıflandırma: `ALL_LIQUIDATION_SYMBOLS_GAP` + `TRANSPORT_SPECIFIC_OUTAGE`** (kategori kanıtlı; kesin nihai kök-neden kanıtı eksik — bkz. aşağı).

**Doğrudan kanıt — 2026-05-03 tarihli, dönemin içinde yapılmış first-party diagnostik:**
`logs/diagnostics/endpoint_matrix_result.json` (üretildi 2026-05-03T09:13:55Z) — 6 farklı endpoint konfigürasyonu (`!forceOrder@arr` global, `btcusdt@forceOrder`/`ethusdt@forceOrder` tekil, combined-stream, + **control** `!markPrice@arr` ve `btcusdt@aggTrade`) 300 saniye test edildi. **HEPSİ** `connection_successful=true, subscription_confirmed=true` ama **total_frames=0, verdict="DEAD"** — **kontrol (liquidation-dışı) stream'ler dahil**. Araştırmacının kendi sonucu (`path_rationale`): *"control endpoints are not reliably working; this points to a network/VPN path blocker."* `matrix_vpnfree/` alt-testi (VPN kapalıyken) aynı sonucu verdi (dosyalar birebir aynı) — **VPN'in kapatılması sorunu tek başına çözmedi**, ya da bu test tamamlanmamış/tekrar niteliğinde.

**Arşiv/rotasyon kontrolü:** `data/archives/raw_v1/catalog_index.json` yalnız `agg_trades`, `book_ticker`, `mark_prices` tablolarını içeriyor — **`liquidations` tablosu arşivde hiç yok**. Bu, verinin "aktif DB'den arşive taşındığı" ihtimalini (ACTIVE_DB_ROTATION_OR_ARCHIVE_ONLY) ELİYOR — veri hiç toplanmamış, silinmemiş/taşınmamış.

**Silme kanıtı yok, şema/ingestion-filtre değişikliği kanıtı yok** (kod yolu bugünkü oturum boyunca defalarca okundu, likidasyon parse mantığı bu dönemde değişmiş görünmüyor).

**`collector_restarts.log`:** 2026-04-21→2026-05-28 arası onlarca `"Collector canonical start FAILED"` / seyrek `"OK"` girişi (yoğun instabilite deseni); 2026-06-05 19:25-19:43 ve 2026-06-06 20:43-20:59'da çoklu restart — **kurtarma (17:47:03) bu restart penceresinin İÇİNDE**, tam nedensellik kanıtlanamıyor ama zamansal olarak tutarlı.

**Temmuz olayıyla ilişki — 2026-07-11 derinleştirmesiyle KESİN OLARAK AYRIŞTIRILDI:** `bd7feb32`'nin kendi commit mesajı: `BINANCE_WS` **2026-07-03**'te (`5cda3122`) `/market/stream`'den (doğru) `/stream`'e (yanlış — bağlanıyor ama sıfır frame) REGRESE OLDU; bu yanlış değer 2026-07-06'daki restart'ta canlıya geçti, 4 gün fark edilmeden çalıştı, 07-10'da düzeltildi. **Bu regresyon commit'i Mayıs-Haziran penceresinden SONRA (Temmuz 3) geldi** — `git log --since=2026-04-20 --until=2026-06-10 -- data/microstructure_collector.py` bu pencerede `BINANCE_WS`'e dokunan SIFIR commit gösteriyor; boşluk boyunca `BINANCE_WS` zaten DOĞRU (`/market/stream`) değerdeydi. **Sonuç: Mayıs boşluğu ile Temmuz olayı KANITLANMIŞ ŞEKİLDE FARKLI mekanizmalar** — kod-seviyesi endpoint regresyonu Mayıs'ta yoktu, bu yüzden Temmuz'un nedeni Mayıs'ın nedeni OLAMAZ. `reports/RECONNECTION_AUDIT.md` (2026-03-04, ayrı/daha erken bir olay) zaten "ISP intermittently blocks Binance WS → VPN/SOCKS5 path" senaryosunu öngörüyor — bu ortamda tekrarlayan bir transport-katmanı risk sınıfı olduğunu bağımsız olarak doğruluyor.

**Kök-neden güven sınıflandırması: `ROOT_CAUSE_PROBABLE`** (IDENTIFIED değil — görev kabul kriteri: gözlenen mekanizma ✓, mekanizmanın boşluk sırasında varlığı ✓ [2026-05-03 first-party diagnostik], kod-seviyesi alternatiflerin kanıtla elenmesi ✓ [yukarıda], AMA kurtarma anında neyin özel olarak değiştiğine dair confirmed kanıt YOK — yalnız zamansal çakışan restart'lar var, nedensellik kanıtlanmadı). **Kapatılamayan eksik kanıt:** (a) 2026-06-05/06 kurtarma anında spesifik bir düzeltici eylemi (commit, operatör notu, ağ-yapılandırma değişikliği) kaydeden HİÇBİR kayıt yok; (b) dönemin OS/VPN/router-seviyesi logları hiç tutulmamış, geriye dönük incelenemez; (c) `matrix_vpnfree/` alt-testi sonuçsuz (girdi dosyaları birebir aynı — VPN kapalıyken temiz bir A/B karşılaştırması elde edilememiş).

## 5. Boşluğun HOUR17 kanıtına etkisi

- Gözlenen takvim aralığı: 2026-02-17→2026-07-03. **Gerçek sürekli kapsama ≈3.5 ay, 40 günlük tam kör nokta ile** (iddia edilen "4.5 ay sürekli" YANLIŞ — bu ifade artık kullanılmamalı).
- Aylık cycle dağılımı: Şub=12, Mar=28, Nis=17, **May=0 (boşluk, "kanıtlanmış sıfır cascade" DEĞİL)**, Haz=35, Tem=1.
- Kronolojik TRAIN/TEST (%70/%30) sınırı boşluğu KAPSIYOR: TRAIN esas olarak Şub-Nis (+birkaç Haziran) cycle'ları, TEST neredeyse tamamen Haziran (boşluk-sonrası, potansiyel farklı rejim) cycle'ları — "holdout train'i geçiyor" bulgusu kısmen boşluk-sonrası rejim farkını yansıtıyor olabilir, temiz rastgele holdout değil.
- Hafta-blok bootstrap/MC sadece GÖZLENEN 16 haftadan örnekliyor — eksik haftalar için YAPAY boş aralık YOK, ama eksik haftaların gerçek davranışı hiç temsil edilmiyor.
- Boşluk iddiayı **geçersiz kılmıyor, ama zayıflatıyor**: gerçek kanıt süresi/sürekliliği iddia edilenden daha kısa ve daha kesintili.

## 6. İleri (forward) karar sınırı

Terfi YOK. Route donmuş/değişmedi. İleri shadow gözlemi devam ediyor (şu an kapanmış N=1). Gelecekteki herhangi bir değerlendirme için minimum: ≥20 bağımsız ileri cycle, ≥8 takvim haftası, ≥15 farklı işlem günü, hiçbir gün/haftanın ileri kümülatifin >%30'unu oluşturmaması, runtime semantik uyuşmazlığı yok, health GREEN ve her iki live executor OFF. Bu gereksinim tarihsel recompute'un pozitif olması nedeniyle DÜŞÜRÜLMEDİ.

## 7. K-S34-HOUR17-001 KO Düzeltmesi — Uygulama Kaydı (2026-07-11, operatör yetkisiyle)

**Yöntem:** kanonik `ami/knowledge/store.py:KnowledgeStore.put()` + `ami/knowledge/objects.py:KnowledgeObject.touch_version()` (el yazması SQL kullanılmadı). Önce dry-run (yazma yapılmadan tam before/after diff'i üretildi ve incelendi), sonra tek transaction'lık gerçek yazma.

**DB checksum:** önce sha256=`710b3f689db2238f11efa04230600b9ddd06e500807b5fb69c7e797e6053dc65` (110592B) → sonra sha256=`24db632924f97959248577b96123d7189bdd96fca41943e0ab03843d91227760` (114688B). Beklenen ikili değişiklik: yalnız eklenen payload içeriği + WAL büyümesi.

**Diğer satırların değişmezliği kanıtı:** DB'deki 11 knowledge satırının tamamının payload sha256'sı yazma öncesi/sonrası karşılaştırıldı — **10 satır birebir DEĞİŞMEDİ**, yalnız `K-S34-HOUR17-001` değişti. `PRAGMA integrity_check`=`ok`. Şema (`CREATE TABLE knowledge...`) ve `user_version` DEĞİŞMEDİ. `audit_log`'a tam olarak 1 yeni satır eklendi: `PUT actor=claude-sonnet-5-hour17-ko-correction knowledge_id=K-S34-HOUR17-001 detail="status=FORWARD_VALIDATING v2"`.

**Değişen alanlar (before→after):**
- `claim`: "~+40bps net, WR~62%" (event-belirsiz) → 200K/S6_200K_full + cycle-düzeyi tam rakamlarla (N=93, WR=62.4%, mean+32.47bps, median+24.04bps, cum+3019.4bps) + event-düzeyi rakamlar açıkça "secondary/dependence-inflated" etiketli
- `provenance.experiment_id`: `""` → `"S6_200K_full;canonical-v1_cycle_recompute_2026-07-11"`
- `provenance.data_time_range`: `"2026-02-15..2026-07-02"` → aynı + `"(DISCONTINUOUS: ... ~40d3h gap ... ~3.5 months, not continuous 4.5 months)"` eklendi
- `effect_size`: `{avg_net_bps:40.8, wr:0.615}` (150K'ya ait, yanlış atıf) → `avg_net_bps`/`wr` artık **cycle-düzeyi** birincil değerler (32.47/0.624) + `cycle_n/cycle_wr/cycle_mean_net_bps/cycle_median_net_bps/cycle_cumulative_net_bps/event_to_cycle_deflation_ratio/event_n/event_wr/event_mean_net_bps/event_level_note/top5_cycle_share_of_cumulative_pct/superseded_config` eklendi
- `scope`: `coverage_note` eklendi (tam boşluk detayı + TRAIN/TEST fold uyarısı)
- `assumptions`: +1 ("Historical coverage is NOT continuous...")
- `confidence`: `cycle_adjustment_classification: "HOUR17_HISTORICAL_CYCLE_EVIDENCE_POSITIVE_BUT_FRAGILE"` eklendi
- `evidence_families`: `[]` → 2 kayıt (S6_200K_full reproduction, canonical-v1 cycle resolver recompute)
- `version`: 1→2 (`touch_version()`: eski durum `history`'e eklendi, `forward_events` anayasal olarak 0'a resetlendi — materyal değişiklik sonrası ileri-kanıt sıfırlanması doğru davranış)

**DEĞİŞMEYEN alanlar (23/30 üst-seviye alan):** `knowledge_id, claim_type, status(FORWARD_VALIDATING), mechanism, direction, evidence_level, required_evidence_level, replications, holdouts, forward_events(0), contradictions, falsification, permitted, forbidden(SIZING_ALLOWED hâlâ yasak), parents, children, decay_half_life_days, created_ms, owner, frozen(false), freeze_hash, history(+1 satır)`. **Terfi/izin genişletme YOK.**

**Runtime etkisi:** `effect_size` alanı kod tabanında yalnız `ami/knowledge/objects.py`'nin kendisi tarafından okunuyor (grep ile doğrulandı) — hiçbir strateji/route/eşik/risk kodu bu KO'yu runtime'da okumuyor. Düzeltme **sıfır davranışsal etki** yarattı.

**Kök-neden derinleştirmesi (Temmuz olayından kesin ayrım) ve nihai sınıflandırma `ROOT_CAUSE_PROBABLE` için bkz. §4 sonu (güncellendi).**

**Nihai verdict: `HOUR17_KO_CORRECTED_ROOT_CAUSE_PROBABLE`.**
