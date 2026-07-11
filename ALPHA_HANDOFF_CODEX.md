# Alpha Handoff → Codex

> Bu dosyayı Codex'e direkt paste et. Hangi alpha üzerinde çalışıyoruz, kanıtı,
> kod haritası, kesin kurallar ve geliştirme fikirleri burada. Tam sistem
> bağlamı için `SYSTEM_STATE.md` (özellikle §24-28).
> Tarih: 2026-07-01

---

## 1. TL;DR — Üzerinde çalıştığımız alpha

**Aile:** ETH SELL likidasyon **cascade → LONG mean-reversion** (bounce). Büyük bir
long-likidasyon cascade'i sonrası fiyat sıklıkla toparlıyor; biz reversion'ı LONG'luyoruz.
Tüm route'lar (silence, echo, double-cascade, hour17) bu tek çekirdeğin varyantları.

**Şu an CANLI olan tek deploy adayı:** `LONG_HOUR17_HOLD6H`
- ETH SELL cascade ≥200K, not bull, not EUROPE, regime(btc4h<0 OR btc7d<0), **hour ≥ 17 UTC**
- → T0'da LONG aç, **6h tut, ERKEN ÇIKIŞ YOK**, 300bps geniş güvenlik stopu.
- OOS: ~16/ay, WR %60, mc_p 0.003, WF 5/5, mdd -391 bps (no-overlap).

---

## 2. Nasıl buraya geldik (kanıt zinciri — kritik)

Bu yolculuk çok fikri çürüttü. Codex bunları TEKRAR yapmasın:

1. **Attribution (§25):** "silence" (cascade sonrası ilk 30dk follow-on YOK) tek başına
   tüm parayı taşıyor gibi göründü (TOTAL +7225, mc 0.0). regime/echo/score sadece booster.
2. **Falsification (§26) — silence LOOKAHEAD'di:** silence ancak T+30'da bilinir.
   Gerçek mekanizmalar test edildi:
   - ideal-silence (lookahead): TOT +5960 mc 0.001
   - **provisional early-exit (noisy'de çık) = MEVCUT eski live mekanizması: TOT -137 mc 0.514 → ÖLÜ**
   - **hold-all (erken çıkış YOK, tut): TOT +7755 mc 0.005 → GERÇEK**
   - Random kontrol: random-regime-hold = -1654, cascade-hold = +9546 → cascade timing gerçek edge.
   - **ASIL KATİL = ERKEN ÇIKIŞ.** Follow-on'da satmak = kapitülasyon dibinde satmak.
3. **hold-all kırılgan:** raw significant ama no-overlap'ta mc 0.267 (anlamsız), stop edge'i kesiyor.
4. **T0 predictor (§27) — ÇÖZÜM:** silence T0'da tahmin EDİLEMİYOR, ama **hour** hold-6h getirisini
   doğrudan öngörüyor (OOS lift +35.7, mc 0.005). `hour≥17` filtresi hold-all'ı mc 0.267→0.003 yaptı.

**Sonuç:** Deploy-hazır edge = hour17. Yüksek-frekans "silence core" bir seraptı (lookahead).

---

## 3. KESİN KURALLAR (yapısal dersler — ihlal etme)

1. **ERKEN ÇIKIŞ YOK.** Noisy follow-on gelince pozisyonu kapatma; hold et. Erken çıkış edge'i öldürüyor.
2. **STOP dikkatli.** Dar stop (≤150bps) bu edge'i öldürüyor (dip-recover mekanizması). Sadece geniş
   güvenlik stopu (~300bps, nadiren tetiklenir). Live'da operatör 300bps onayladı.
3. **LOOKAHEAD YASAK.** Sadece T0'da bilinen feature ile filtrele. Silence/noisy/exit-outcome T0'da bilinmez.
4. **Metodoloji zorunlu:** kronolojik 70/30 holdout (eşik TRAIN'de, rapor TEST'te), no-overlap
   (tek-pozisyon gerçekçi frekans), gerçek maliyet (fee 8-10 bps), MC permütasyon, walk-forward fold.
   TOTAL net PnL öncelikli metrik (WR değil — WR yüksek ama az-trade route'lar para getirmiyor).

---

## 4. Kapalı hipotezler (TEKRAR TEST ETME)

- ❌ Yüksek-freq silence core canlıya alınabilir (lookahead + no-overlap'ta anlamsız)
- ❌ Provisional early-exit edge (mc 0.5) — erken çıkış öldürüyor
- ❌ Dar stop (≤150) hour17'yi kurtarır (mc 0.247, öldürüyor)
- ❌ Eşik düşürmek (100K) net frekans+edge getirir (tail-ağır gürültü)
- ❌ BTC-led / SOL-led ETH sinyali (mc 0.77 / 0.57)
- ❌ ETH BUY → SHORT (mc 0.988); failed_cascade → SHORT (kapalı)
- ❌ silence T0'da tahmin edilebilir (hiçbir feature lift vermedi)
- ❌ be_ratio_pre / btc_conc_pre yüksek = İYİ (tam tersi — bunlar VETO)

---

## 5. Kod haritası

**Araştırma scriptleri** (tek process, read-only DB, `python tools/<script>.py`):
| Script | Ne yapar | Rapor |
|---|---|---|
| `research_s34_alpha_attribution.py` | Hangi filtre para taşıyor (TOTAL PnL) | `S34_ALPHA_ATTRIBUTION.md` |
| `research_s34_silence_core_final.py` | Realizm + tail + falsification | `S34_SILENCE_CORE_FINAL.md` |
| `research_s34_silence_predictor.py` | T0 feature screening + holdout + FINAL | `S34_SILENCE_PREDICTOR.md` |
| `research_s34_echo_expansion.py` | Echo genişletme + regime tail | `S34_ECHO_EXPANSION.md` |
| `research_s34_echo_live_gauntlet.py` | Echo giriş/hold/tail/cost | `S34_ECHO_LIVE_GAUNTLET.md` |

Ortak altyapı: `research_s34_knowable_anchor_continuation.py` (load_liquidations,
load_mark_index, reconstruct_anchors, MarkIndex.slice_range), `research_s34_wave_absorption.py`
(book_features_at). Anchor eşiği 200K, bucket 300s, min_gap 900s.

**Canlı/observation:**
| Dosya | Rol |
|---|---|
| `tools/s34_state_machine_live_executor.py` | LIVE executor — `LONG_HOUR17_HOLD6H` route eklendi (§28) |
| `tools/s34_realtime_shadow_runner.py` | Shadow observation — `LONG_HOUR17_HOLD6H` route mirror |
| `tools/s34_live_chart.py` | Dashboard :5050 — `active_alpha` kartı + `C_hour17_hold6h` bucket |

**hour17 route parametreleri (executor sabitleri):** `HOUR17_MIN_HOUR=17`,
`HOUR17_STOP_BPS=300`, `LONG_HORIZON_H6_MS=6h`. `open_market_position(..., stop_bps_override=)`
geri-uyumlu param. Sizing/leverage/.env DOKUNULMADI (guardrail).

---

## 6. Yeni fikirler (Codex — geliştirme yönleri, öncelik sırasıyla)

### A. hour17'yi sağlamlaştır / anla
1. **Mekanizma:** hour≥17 neden çalışıyor? (US kapanış akış, funding saatleri 00/08/16 UTC,
   likidasyon clustering?) Saati daha ince ayır: 17-19 vs 20-21 vs 22-23 ayrı test et.
2. **Regime-decay alarmı:** rolling 30-günlük WR izle; US-afternoon-reversion rejimi dönerse
   (WR < %50 son N trade) otomatik uyarı. Forward-validation harness.
3. **hour × regime-şiddeti:** derin btc7d (<-300) ile hour17 daha mı güçlü?

### B. Frekans + kalite (yeni deploy adayları)
4. **`hour17 & sync_k=mid`** yüksek-WR varyantı: zaten bulundu (6/ay, WR %79, mc 0.008). OOS holdout
   tekrarla, forward izle, ikinci live route adayı.
5. **2-feature combo'lar** daha büyük TEST seti ile: hour + {btc7d=mid, rn=lo(küçük cascade), n2h=hi}.
   Küçük TEST-N overfit riski var — daha fazla veri biriktikçe tekrar.
6. **hour17 @ 100K/150K + tail mgmt:** daha çok frekans mümkün mü, mdd kontrollü kalır mı?

### C. Portföy / çeşitlendirme
7. **SHORT tarafı hour predictor:** SHORT_NOISY / BTC-confirm route'larında saat etkisi var mı?
8. **hour17 + SHORT_NOISY non-overlapping portföy:** birleşik /ay ve TOTAL (echo core ile örtüşüyordu,
   SHORT_NOISY gerçek çeşitlendirme +1000 TOTAL veriyordu).

### D. Exit / execution
9. **Exit taraması:** hour17 için 6h vs 8h vs trailing-after-100bps. (6h > 4h bulunmuştu.)
10. **Oversize sizing fix (URGENT, engineering):** per-trade margin tail-budget'a (~$0.50); şu an 60x üstü.
    hour17 canlıyken bu risk aktif.

### E. Metodoloji
11. **Otomatik haftalık OOS re-gauntlet:** biriken live+shadow trade üzerinde holdout+MC tekrar,
    edge teyidi/decay tespiti.

---

## 7. Codex'e ilk görev önerisi

`research_s34_silence_predictor.py`'yi baz al. Ya (A1) hour bin'ini ince ayır ve mekanizmayı araştır,
ya (B4) `hour17 & sync_k=mid` varyantını genişletilmiş holdout ile doğrula. Her ikisi de research-only;
live executor / .env / sizing'e DOKUNMA (operator sign-off gerekir). Sonuçları `SYSTEM_STATE.md`'ye
yeni bölüm olarak ekle.
