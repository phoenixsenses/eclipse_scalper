# DR-0006 — C-BUY-FADE Yapısal + 8A Re-Entry Paketi Kararları (2026-07-03)

**Durum:** KABUL — iki deney de dürüst sonuçla kapandı:
`E-BUYFADE-STRUCT-001` (hash 70cf5acb…) **FALSIFIES** (tek istisna: silence-info),
`E-BUYFADE-REENTRY-001` (hash 82a4e56b…) **FALSIFIES** (H-RE-NULL doğrulandı).

## Kararlar
1. **Route AYNEN korundu:** event universe = ETH BUY cascade ≥200K (bucket 300s, gap 900s),
   EUROPE + bear-squeeze veto, T0 mark-fill, 45m, SL75, FEE 5bps. Sonuca göre
   yeniden yorumlanmadı. Mezarlıktaki eski "ETH BUY→SHORT mc 0.988" kaydına karşı retry
   gerekçesi prereg'e yazıldı (yeni route tanımı + shadow forward).
2. **Silence v1 tanımı ve versiyonu donduruldu:** [T0+60s, T0+30m] içinde ETH BUY ≥50K yok —
   **T+30m'de bilinir**; T0 feature seti için yapısal yasak (`g_no_t0_silence`).
3. **p=0.010 denetimi:** dashboard `mc_p` değerleri HARDCODED metadata; 100-permütasyon
   tabanı (1/101≈0.0099) ile tutarlı. Bu pakette N_PERM=20.000 (çözünürlük 5e-5).
4. **NOT_AVAILABLE dürüstçe raporlandı, proxy uydurulmadı:** OI slope/accel,
   price-OI divergence, spot/perp participation, basis (kapsama <%60), depth>L1,
   impact-per-dollar.
5. **Splitler:** event-bazlı kronolojik 60/25/15 + 24h purge (13 event purged);
   seçimler yalnız TRAIN'de; delayed girişler executable-fill (staleness≤120s yoksa MISSED);
   delayed varyantlar aynı-pencere konvansiyonuyla T0+45m'de kapanır (beyan edildi).
6. **Verdict-kodu falsy-zero bug'ı** (`(p or 1) < 0.05`, p=0.0 → 1) bulundu, düzeltildi,
   deterministik rerun yapıldı — kriter DEĞİŞMEDİ, yazılım düzeltmesi kayıtlı.
7. **8A frozen tasarım notu:** trigger'lar sürekli causal tarama; pencere listesi dağılım
   raporu (ayrı optimize edilmez); cooldown seçimi TRAIN'de; FEE her girişte;
   NO_POSITION/WAIT geçerli state; cap99 = kontrol kolu (operasyonel aday olamaz).

## Sonuç yorumu (391 event; train 234 / val 90 / untouched 54)
- **Route'un ALL varyantı tarihsel replay'de NEGATİF** (−9.5 / −1.1 / −10.7 bps/trade) —
  shadow N=26 (+2.8) genellemiyor. Dashboard'daki "tail repair" statüsü iyimserdi.
- **Tek gerçek bilgi: SILENCE (T+30m)** — silence-subset +20/+30/+20 bps üç split'te
  tutarlı; matched-control (7 boyut) diff +54bps, p<5e-5 → hour/vol/size proxy'si DEĞİL.
  Ama kademeli erken versiyonları (s30s..s10m) zayıflıyor ve delayed-entry ile
  YAKALANAMIYOR (silence_3m_entry val +0.7) → **giriş alfası değil, geç-aşama bilgi**.
- **Genesis/maturity:** mature/exhausted LONG + silence hücresi güçlü (train +31.7 n=73,
  val +32.8 n=23) ama silence-koşullu → T0 filtresi olarak NON_PREDICTIVE.
- **Timing:** TRAIN seçimi ofi_flip family_p=0.133 → geçmez. Val'de parlak delay_600s
  (+10.5) untouched'ta −12.3 → split-instability; frozen protokol overfit'i yakaladı.
- **Horizon:** TÜM ufuklar (5m..7D) unconditional negatif; event-high reclaim %62-93.
  SHORT continuation yapısal olarak YOK; H2 (4h-DOWN+silence multi-hour) val n=8-9,
  h240 +48 yönünde ama INSUFFICIENT; H1/H3/H4 hücreleri boş-küçük.
- **Management:** hiçbir varyant fixed_45m'i geçmedi; SHORT→LONG transition
  (reclaim sonrası 4h LONG) train +32/val +24 AMA top3_removed +2/−60 →
  top-winner bağımlı, yüksek kanıt standardını geçmez.
- **8A:** S→S re-entry TÜM cooldown'larda train incremental negatif (en iyi cd=30m,
  −709); val'de de negatif, random-timing'i geçemedi (0.746<0.75) → **REENTRY_CHURN
  karakterli NON_INCREMENTAL; H-RE-NULL doğrulandı.** S→L flip val n=2, L kolları
  eligible=0 → INSUFFICIENT_SAMPLE. Stop-taxonomy: BAD_TIMING sonrası re-entry
  train +19bps (n=16) tek ilginç alt-sinyal — yeni prereg adayı, iddia değil.

## Reddedilenler
- Silence'ı T0 entry feature'ı yapmak (lookahead; SELL-side saga'nın tekrarı önlendi)
- Val-parlak delay_600s'i post-hoc seçmek (untouched negatif)
- SHORT→LONG transition'ı top-winner bağımlılığına rağmen geçirmek
- Küçük hücre (H2/H5, context-matrix 4/5) alpha ilanı
- Kriter gevşetme / event universe değişikliği

## Mutation suite
`tests/test_buyfade_mutations.py` — **24/24**: future-genesis, unfinished-candle,
T0-silence yasağı, causal event-high, train-only seçim, split-purge, no-overlap,
outcome-leak, session-UTC sınırları, UNKNOWN≠neutral, executable-fill, top-winner
disclosure, tiny-cell, post-hoc silence-threshold (freeze), family-p zorunluluğu,
candidate_pass pozitif/negatif; re-entry: causal-prefix, fee-per-entry, entry-merge,
attempt-drop, flip-claim ayrımı, cooldown-train-only, entry3-small-N, cycle-purge.
Toplam AMI+research testi: 79 + 24 = **103**.

## Rollback
`tools/research_s34_buyfade_structural.py`, `tools/research_s34_buyfade_reentry.py`,
`tests/test_buyfade_mutations.py`, `reports/research/s34/BUYFADE_*.{json,md}` sil;
research.sqlite/knowledge.sqlite kayıtları tarihsel olarak kalır.

## Dokunulmayan canlı bileşenler
Live executor, order logic, risk, leverage, sizing, `.env`, mevcut live rules,
shadow runner route'ları — SIFIR değişiklik. **Shadow/live route otomatik
DEĞİŞTİRİLMEDİ** (paket şartı); "ALL varyantı tarihsel negatif" bulgusu operatör
kararı için rapor edildi.
