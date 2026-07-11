# S34 Mekanizma Araştırma Planı — Event Filtering → Mechanism Discovery + Execution Optimization

> Tarih: 2026-07-02. Yön değişimi: likidasyon filtre-istifleme azalan verimde.
> Yeni eksen: (A) kârlı cascade'i YARATAN piyasa durumu, (B) cascade SONRASI optimal execution.
> Canlı strateji kurulmayacak — önce mekanizma anlaşılacak. Tümü research-only.

---

## 0. Veri Gerçekliği (denetlendi 2026-07-02)

| İstenen | Elimizde | Kapsam | Not |
|---|---|---|---|
| Order book davranışı | `book_ticker` **L1** (bid/ask px+qty, spread, imb, bid_depth_usd) | 11 Nis → canlı (~2.7 ay) | L2 yok — absorption L1+impact'ten çıkarsanır |
| Maker absorption | `agg_trades` (is_buyer_maker) + L1 refill | 15 Şub → canlı | impact-per-$1M + depth-recovery proxy'leri |
| Funding dinamiği | `mark_prices.funding_rate` | tam geçmiş, canlı | **velocity = Δrate/Δt** hesaplanabilir ✓ |
| OI dinamiği | `open_interest` | 🔴 18 Nis'te ölü (501 satır) | tarihsel çalışılamaz → **yeni collector** (Faz 5) |
| Cross-exchange flow | — | ❌ yok | **yeni collector** (Faz 5, opsiyonel) |
| Spot-perp basis | `spot_prices` | 7 Mar – 5 Haz (producer öldü) | tarihsel test edilebilir; forward için producer canlandırma |
| Cascade + fiyat | `liquidations`, `mark_prices` | tam, canlı | mevcut altyapı |

**Sonuç:** Ana çalışma evreni = **11 Nis → bugün** (book kapsamı), ETH SELL anchor ≥100K
(dar gate YOK — mekanizma için geniş evren + kontrol örneklemi şart).

---

## 1. Mimari: Feature Store (tek sefer çıkar, sonsuz kez analiz et)

650GB DB'yi her analizde taramak yerine, her event etrafındaki zengin pencere özellikleri
**tek kompakt SQLite'a** çıkarılır:

```
tools/s34_mechanism_feature_store.py  →  reports/research/s34/mechanism_store.sqlite
```

Her satır = 1 event (anchor VEYA kontrol). Pencereler: `pre10m, pre1m, at, post1m, post5m, post10m`.

**Özellik grupları:**
- **BOOK (L1):** pencere başına bid/ask qty ort+min, spread ort+max, imbalance ort+eğim,
  quote güncelleme sayısı; `pull` = pre1m min bid_qty / pre10m ort (likidite çekilmesi);
  `refill` = post5m bid_qty / pre10m (maker geri dönüşü)
- **FLOW:** buy/sell notional, OFI oranı, trade sayısı, ort/maks trade boyutu,
  `impact` = |Δfiyat| / $1M notional (Kyle-λ proxy — absorption ölçüsü)
- **PRICE:** pre getiriler (10m/1h/4h), rv-proxy, post: dip-zamanı, maxDD-30m, MFE/MAE (30m/1h/6h)
- **LIQ:** rn, cascade süresi, liq sayısı, prebuild 1h/24h, two_sided, BTC/SOL sync
- **FUNDING:** seviye, **velocity 1h/8h**, funding'e dakika
- **BASIS:** spot-perp basis + eğimi (kapsam olan dönemde), mark-trade basis
- **EXECUTION GRID:** entry offset {+2s, +5s, +10s, +30s, +1m, +5m, +15m} → her birinden y_6h;
  ayrıca y_30m/1h/2h/6h (T0'dan)
- **ETİKETLER:** `is_event` (anchor=1 / kontrol=0), continuation (30m'de düşüş sürdü) vs
  reversal (döndü), profitable_long_6h

**Kontrol örneklemi (pre-cascade prediction için kritik):** event sayısı kadar,
saat-dağılımı eşlenmiş rastgele zaman; ±30m içinde cascade OLMAYAN anlar.
Soru: pre10m özellikleri cascade'i kontrollerden ayırıyor mu?

---

## 2. Fazlar

### Faz 1 — Feature store kurulumu ✅ (bu oturum)
Script yazıldı, store dolduruldu, sanity check yapıldı.

### Faz 2 — Mekanizma taksonomisi: continuation vs reversal NEDEN?
Store üzerinde (hızlı, tekrar koşulabilir):
- Reversal'ı öngören pre/at özellikleri: absorption (impact düşük = emiliyor) mu,
  pull (likidite kaçtı = devam) mü? refill hızı reversal'ı mı işaret ediyor?
- Funding velocity vs seviye — hangisi ayırıyor?
- Hipotez seti: H1 "düşük impact + hızlı refill → reversal (LONG iyi)";
  H2 "yüksek pull + OFI satıcı → continuation"; H3 "funding velocity ekstrem → squeeze dönüşü".
- Metod: TRAIN/TEST 70/30, tercile lift, MC permütasyon, no-overlap EV.
- Script: `tools/s34_mechanism_taxonomy.py` (Faz 2'de yazılacak)

### Faz 3 — Pre-cascade prediction (erken giriş −5/−10dk)
- Event-vs-kontrol ayrımı: pre10m özellikleriyle basit skor (eşikler TRAIN'de).
- Ölçüt: precision@yüksek-eşik (yanlış alarm maliyetli), sonra **tradeable EV**:
  sinyal anında gir (cascade'den önce!) → cascade gelmezse time-stop, gelirse bounce'ı yakala.
- Dürüstlük şartı: "cascade oldu" bilgisi girişte YOK; her sinyal anı işlem sayılır.
- Script: `tools/s34_precascade_predictor.py` (Faz 3)

### Faz 4 — Execution optimizasyonu (cascade SONRASI)
Store'daki execution grid + L1 quote'larla:
- **Entry timing:** 2s/5s/10s/30s/1m/5m/15m gecikme eğrisi — edge decay haritası.
  (Bilinen: T0/T+1m iyi, T+15m echo hariç öldürüyor — şimdi saniye çözünürlüğünde.)
- **Order tipi:** market (ask'ten al) vs limit −Xbps (L1 bid path'inden gerçekçi fill:
  Q2 dersi — fill survivorship'e dikkat, dolmayanlar EV'ye dahil) vs 5m VWAP.
- **Dinamik TP:** sabit +200 vs rv-ölçekli (k×rv_proxy) vs MFE-yüzdesi trailing.
- **Vol-ölçekli stop:** sabit −150/−300 vs k×rv stop — tail'i kesip edge'i koruyor mu?
  (Bilinen: sabit dar stop edge'i öldürüyor; vol-ölçekli farklı olabilir mi test.)
- Script: `tools/s34_execution_optimizer.py` (Faz 4)

### Faz 5 — Yeni veri toplayıcılar (forward için; operatör onayıyla)
- **OI poller:** Binance `/fapi/v1/openInterest` 1dk polling → `open_interest` tablosunu canlandır.
- **Spot producer canlandırma:** `spot_prices` 5 Haz'da vol_state ile aynı anda ölmüş
  (aynı silinen producer) — basit spot bookticker poller yeterli.
- **Cross-exchange (opsiyonel):** Bybit/OKX liq + mid poller. Ancak collector RAM/disk
  bütçesi operatör kararı.
> Bunlar collector altyapısı — mevcut collector'lara dokunmadan AYRI hafif poller olarak eklenir.

---

## 3. Metod Guardrail'ları (değişmez)

1. Lookahead yok: özellikler yalnızca t≤T0 verisinden; etiketler yalnızca t>giriş.
2. Eşikler TRAIN'de seçilir, TEST raporlanır; MC permütasyon; no-overlap EV.
3. Fill gerçekçiliği: girişler ask/bid'den (mark değil); limit fill L1 path'ten; dolmayanlar EV'de.
4. Tek Python prosesi; 650GB ana DB'ye salt-okunur; analiz store üzerinde.
5. Canlıya hiçbir şey otomatik geçmez — her faz research-only rapor üretir.

## 4. Başarı Kriterleri

- Faz 2: continuation/reversal'ı TEST'te ≥15 WR-puanı ayıran ≥1 mekanizma değişkeni.
- Faz 3: kontrollere karşı precision ≥2× base-rate + pozitif tradeable EV; yoksa DÜRÜSTÇE KAPAT.
- Faz 4: mevcut T0-market-sabitTP'ye karşı net-bps iyileştirme (holdout + no-overlap).
