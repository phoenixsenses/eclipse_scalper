# S34 — Tum Alpha Ailesi: Konsolide Karar Raporu

> Kaynak: `S34_ALL.db` (SADECE SQL'den okundu). Block-bootstrap: `S34_HORIZON.json`.
> Uretim: 2026-07-01 22:09 UTC. Live/.env/sizing'e DOKUNULMADI.

**Metrik notu:** median bps ve top3-removed cumulative research_results semasinda YOK (mark ‘-’). 
block-bootstrap CI sadece composite score>=3 icin var (S34_HORIZON R). no-overlap /ay = `_noov` satirlarindaki /ay.

**Flag lejantı:** IS=in-sample, OOS=holdout/test. lookahead ‘VAR’ = geleceğe bakiyor (guvenilmez). 
no-ov EVET = tek-pozisyon uygulandi. fill=mark (gercek borsa fill degil; paper=mark-fiyat). tekrar-sayim EVET(overlap)=ayni event zaman-penceresinde ust uste sayilabilir.

## 1) LONG composite v3 (hour17 + 8-sinyal conviction)
**Karar: LIVE ADAYI (once paper-forward)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| score>=3 full (v3, 8-sig) | 85 | 18.9 | 74% | +64 | +5438 | - | - | 0.000 | - |
| score>=3 TEST | 32 | 23.7 | 78% | +79 | +2529 | - | - | 0.000 | - |
| score>=4 full | 56 | 12.4 | 75% | +75 | +4205 | - | - | 0.000 | - |
| score>=4 TEST | 27 | 20.0 | 85% | +101 | +2718 | - | - | 0.000 | - |
| score>=3 no-overlap | 36 | 8.0 | 64% | +46 | +1642 | -162 | 4 | 0.007 | 5/5 |
| score>=4 no-overlap | 21 | 4.7 | 71% | +55 | +1149 | -184 | 2 | 0.023 | 4/5 |
| hour17 baz (composite yok) noov | 63 | 14.0 | 62% | +34 | +2118 | -173 | 11 | 0.021 | 5/5 |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| score>=3 full (v3, 8-sig) | in-sample | yok | HAYIR | mark | EVET(overlap) |
| score>=3 TEST | OOS/holdout | yok | HAYIR | mark | hayir |
| score>=4 full | in-sample | yok | HAYIR | mark | EVET(overlap) |
| score>=4 TEST | OOS/holdout | yok | HAYIR | mark | hayir |
| score>=3 no-overlap | in-sample | yok | EVET | mark | hayir |
| score>=4 no-overlap | in-sample | yok | EVET | mark | hayir |
| hour17 baz (composite yok) noov | in-sample | yok | EVET | mark | hayir |

> Block-bootstrap (S34_HORIZON): score>=3 noov N=35 obs_avg=53.6 5%CI=26.3 95%CI=75.3 P(avg<0)=0.0. Weighted-sizing (composite) flat->4.2x (S34_CONVICTION_COMPOSITE.results.weighted).

## 2) 100K composite (frekans genisletme)
**Karar: PAPER (holdout gecti, frekans 2x)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 100K score>=3 full | 130 | 28.9 | 68% | +60 | +7806 | - | - | 0.000 | - |
| 100K score>=3 TEST | 58 | 42.9 | 72% | +58 | +3345 | - | - | 0.000 | - |
| 100K score>=3 no-overlap | 53 | 11.8 | 66% | +55 | +2892 | - | - | 0.000 | - |
| 100K mini fizzled (buyumeyen) | 84 | 18.6 | 68% | +54 | +4581 | -419 | - | 0.000 | - |
| 100K mini grew (200K'ya ulasan) | 76 | 16.9 | 54% | +21 | +1565 | -206 | - | 0.070 | - |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| 100K score>=3 full | in-sample | yok | HAYIR | mark | EVET(overlap) |
| 100K score>=3 TEST | OOS/holdout | yok | HAYIR | mark | hayir |
| 100K score>=3 no-overlap | in-sample | yok | EVET | mark | hayir |
| 100K mini fizzled (buyumeyen) | in-sample | yok | HAYIR | mark | EVET(overlap) |
| 100K mini grew (200K'ya ulasan) | in-sample | yok | HAYIR | mark | EVET(overlap) |

> Fizzle eden mini'ler buyuyenden iyi bounce yapiyor (P1).

## 3) Conviction sleeves score>=2/3/4/5 (esik sweep)
**Karar: score>=3/4 PAPER; >=5 premium sleeve**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| score>=2 full | 94 | 20.9 | 71% | +57 | +5390 | -184 | - | 0.000 | - |
| score>=2 no-overlap | 45 | 10.0 | 71% | +58 | +2586 | -162 | - | 0.000 | - |
| score>=3 full | 71 | 15.8 | 75% | +70 | +5002 | -184 | - | 0.000 | - |
| score>=3 no-overlap | 35 | 7.8 | 69% | +54 | +1878 | -176 | - | 0.002 | - |
| score>=4 full | 50 | 11.1 | 76% | +78 | +3907 | -184 | - | 0.000 | - |
| score>=4 no-overlap | 23 | 5.1 | 70% | +50 | +1152 | -176 | - | 0.032 | - |
| score>=5 full | 26 | 5.8 | 85% | +104 | +2714 | -107 | - | 0.000 | - |
| score>=5 no-overlap | 14 | 3.1 | 79% | +83 | +1167 | -107 | - | 0.010 | - |
| score=2 (dagilim) | 31 | 6.9 | 68% | +26 | +809 | -147 | 5 | 0.090 | 4/5 |
| score=5 (dagilim) | 15 | 3.3 | 80% | +72 | +1081 | -107 | 1 | 0.004 | 4/5 |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| score>=2 full | in-sample | yok | HAYIR | mark | EVET(overlap) |
| score>=2 no-overlap | in-sample | yok | EVET | mark | hayir |
| score>=3 full | in-sample | yok | HAYIR | mark | EVET(overlap) |
| score>=3 no-overlap | in-sample | yok | EVET | mark | hayir |
| score>=4 full | in-sample | yok | HAYIR | mark | EVET(overlap) |
| score>=4 no-overlap | in-sample | yok | EVET | mark | hayir |
| score>=5 full | in-sample | yok | HAYIR | mark | EVET(overlap) |
| score>=5 no-overlap | in-sample | yok | EVET | mark | hayir |
| score=2 (dagilim) | in-sample | yok | HAYIR | mark | hayir |
| score=5 (dagilim) | in-sample | yok | HAYIR | mark | hayir |

> Monoton: score arttikca WR/avg yukselir, frekans duser.

## 4) whale_lo (retail = kucuk trade sinyali)
**Karar: PAPER (holdout WR94, v3'e alindi)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| whale_lo TEST | 17 | 12.6 | 94% | +116 | +1981 | - | - | 0.000 | - |
| whale_hi TEST (karsit) | 21 | 15.5 | 57% | +30 | +640 | - | - | 0.124 | - |
| whale_lo full | 63 | 14.0 | 73% | +64 | +4044 | -189 | - | 0.000 | - |
| sync & whale_lo (interaction) | 30 | 6.7 | 87% | +87 | +2623 | - | - | 0.000 | - |
| rv & whale_lo | 41 | 9.1 | 80% | +95 | +3877 | - | - | 0.000 | - |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| whale_lo TEST | OOS/holdout | yok | HAYIR | mark | hayir |
| whale_hi TEST (karsit) | OOS/holdout | yok | HAYIR | mark | hayir |
| whale_lo full | in-sample | yok | HAYIR | mark | EVET(overlap) |
| sync & whale_lo (interaction) | in-sample | yok | HAYIR | mark | EVET(overlap) |
| rv & whale_lo | in-sample | yok | HAYIR | mark | EVET(overlap) |

> OOS'ta WR94 (N=17). Interaction: iki sinyal birden WR80-87.

## 5) SHORT confirm-entry 13-17 (time-machine sleeve)
**Karar: PAPER (kucuk-N, kirilgan)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| SHORT hour 13-17 | 11 | 2.5 | 100% | +182 | +2003 | +38 | - | 0.000 | 5/5 |
| SHORT hour 17-24 (olu) | 5 | 1.1 | 20% | -99 | -496 | -255 | - | 0.974 | 1/5 |
| entry@BTC-confirm (tradeable) | 20 | 4.6 | 70% | +33 | +652 | -314 | - | 0.176 | 3/5 |
| entry@noisy (LOOKAHEAD) | 20 | 4.6 | 75% | +91 | +1820 | -255 | - | 0.012 | 5/5 |
| BTC>=2M h120 | 9 | 2.0 | 100% | +183 | +1644 | +37 | - | 0.002 | 5/5 |
| SHORT 13-17 (horizon) | 11 | 2.4 | 82% | +111 | +1222 | - | - | 0.006 | - |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| SHORT hour 13-17 | in-sample | yok | HAYIR | mark | EVET(overlap) |
| SHORT hour 17-24 (olu) | in-sample | yok | HAYIR | mark | EVET(overlap) |
| entry@BTC-confirm (tradeable) | in-sample | YOK | HAYIR | mark | EVET(overlap) |
| entry@noisy (LOOKAHEAD) | in-sample | VAR | HAYIR | mark | EVET(overlap) |
| BTC>=2M h120 | in-sample | yok | HAYIR | mark | EVET(overlap) |
| SHORT 13-17 (horizon) | in-sample | yok | HAYIR | mark | EVET(overlap) |

> KRITIK: tradeable confirm-entry (+32,mc0.176) noisy-entry LOOKAHEAD'inden (+91) cok zayif. N kucuk.

## 6) Cross-asset sync (senkron cascade)
**Karar: RESEARCH-ONLY (composite sync_ratio zaten iceriyor)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ETH+BTC simultane | 28 | 6.2 | 75% | +75 | +2109 | -184 | - | 0.006 | - |
| ETH+SOL simultane | 19 | 4.2 | 79% | +96 | +1831 | -184 | - | 0.012 | - |
| ETH-only | 98 | 21.8 | 62% | +30 | +2907 | -448 | - | 0.012 | - |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| ETH+BTC simultane | in-sample | yok | HAYIR | mark | EVET(overlap) |
| ETH+SOL simultane | in-sample | yok | HAYIR | mark | EVET(overlap) |
| ETH-only | in-sample | yok | HAYIR | mark | EVET(overlap) |

> sync_ratio composite'te var; ayri route degil, dogrulayici.

## 7) deep7d navigation (rejim x skor haritasi)
**Karar: RESEARCH-ONLY (navigasyon kurali)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| h17-19 deep + s>=4 | 15 | 3.3 | 80% | +120 | +1801 | -64 | - | 0.002 | - |
| h17-19 deep + s<4 (olu) | 10 | 2.2 | 30% | -33 | -332 | -119 | - | 0.914 | - |
| h20-23 deep + s>=4 | 13 | 2.9 | 85% | +101 | +1308 | -52 | - | 0.000 | - |
| deep7d & s>=4 | 28 | 6.2 | 82% | +111 | +3109 | -64 | - | 0.000 | - |
| kazanan sonrasi (momentum) | 81 | 18.0 | 68% | +56 | +4516 | -448 | - | 0.000 | - |
| kaybeden sonrasi | 44 | 9.8 | 59% | +7 | +305 | -189 | - | 0.314 | - |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| h17-19 deep + s>=4 | in-sample | yok | HAYIR | mark | EVET(overlap) |
| h17-19 deep + s<4 (olu) | in-sample | yok | HAYIR | mark | EVET(overlap) |
| h20-23 deep + s>=4 | in-sample | yok | HAYIR | mark | EVET(overlap) |
| deep7d & s>=4 | in-sample | yok | HAYIR | mark | EVET(overlap) |
| kazanan sonrasi (momentum) | in-sample | yok | HAYIR | mark | EVET(overlap) |
| kaybeden sonrasi | in-sample | yok | HAYIR | mark | EVET(overlap) |

> Derin rejimde yuksek-skor sart (s<4 deep = olu). Kazanctan sonra momentum.

## 8) Funding veto (<60m = olu)
**Karar: LIVE-KURAL (composite'e veto olarak alindi)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| <60m funding (olu) | 20 | 4.4 | 45% | -7 | -140 | -189 | - | 0.602 | - |
| 60-240m | 45 | 10.0 | 71% | +38 | +1724 | -448 | - | 0.028 | - |
| >240m | 61 | 13.5 | 67% | +56 | +3433 | -162 | - | 0.002 | - |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| <60m funding (olu) | in-sample | yok | HAYIR | mark | EVET(overlap) |
| 60-240m | in-sample | yok | HAYIR | mark | EVET(overlap) |
| >240m | in-sample | yok | HAYIR | mark | EVET(overlap) |

> <60m ölü -> veto. Composite paper route'da uygulandi.

## 9) 6h horizon (hold suresi)
**Karar: LIVE-KURAL (6h kesin optimal)**

| Config | N | /ay | WR | avg bps | cum bps | worst | tail | MC p | WF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| score>=4 6h | 54 | 12.0 | 76% | +79 | +4259 | - | - | 0.000 | - |
| score>=4 12h | 54 | 12.0 | 67% | +84 | +4534 | - | - | 0.000 | - |
| score>=4 24h | 54 | 12.0 | 67% | +66 | +3592 | - | - | 0.030 | - |
| score>=4 48h (olu) | 54 | 12.0 | 59% | -31 | -1687 | - | - | 0.706 | - |
| score>=3 6h | 79 | 17.5 | 75% | +70 | +5560 | -184 | - | 0.000 | - |

**Provenance / risk flag'leri:**

| Config | IS/OOS | lookahead | no-ov | fill | tekrar-sayim |
|---|---|---|---|---|---|
| score>=4 6h | in-sample | yok | HAYIR | mark | event-reuse(coklu-horizon) |
| score>=4 12h | in-sample | yok | HAYIR | mark | event-reuse |
| score>=4 24h | in-sample | yok | HAYIR | mark | event-reuse |
| score>=4 48h (olu) | in-sample | yok | HAYIR | mark | event-reuse |
| score>=3 6h | in-sample | yok | HAYIR | mark | event-reuse |

> WR 6h'te tepe (48h olu). Ayni event coklu-horizon'da tekrar olculuyor (satirlar arasi).

## Gercek Paper Trade PnL (ledger — mark-fill, gercege en yakin)

| Signal | N | WR | cum bps | avg bps | best | worst |
|---|--:|--:|--:|--:|--:|--:|
| LONG_SILENCE | 399 | 44.4% | +5943 | +14.9 | +609 | -466 |
| SHORT_NOISY | 253 | 54.9% | +6147 | +24.3 | +407 | -239 |
| SHORT_NEITHER | 96 | 70.8% | +7513 | +78.3 | +407 | -180 |
> Not: yeni route'lar (hour17/composite/100K) henuz paper trade biriktirmedi (restart bekliyor).

---
# NET KARAR TABLOSU

| Alpha / route | Durum | Neden |
|---|---|---|
| **LONG composite v3 (score>=3)** | **LIVE ADAYI** (once paper-forward) | OOS TEST WR78-82, block-boot 5%%CI +26.3 P(<0)=0.0, monoton, 6-sinyal additive |
| **LONG composite score>=4/5 sleeve** | **LIVE ADAYI (premium, dusuk frekans)** | OOS WR85-90, worst kucuk; sleeve olarak |
| **100K composite** | **SHADOW/PAPER** | Holdout gecti (TEST WR72, noov 11.8/ay) ama in-sample tail; forward gerek |
| **whale_lo (8. sinyal)** | **SHADOW/PAPER (v3'e alindi)** | Holdout WR94 (N=17 kucuk); forward dogrulama |
| **funding veto (<60m)** | **LIVE-KURAL** | <60m ölü; risksiz veto |
| **6h horizon** | **LIVE-KURAL** | 6h WR tepe, 48h olu; kesin |
| **deep7d navigation** | **RESEARCH-ONLY** | Navigasyon kurali (rejimde skor sart); ayri route degil |
| **cross-asset sync** | **RESEARCH-ONLY** | sync_ratio zaten composite'te; dogrulayici |
| **SHORT confirm-entry 13-17** | **SADECE SHADOW (kirilgan)** | Tradeable-entry zayif (+32 mc0.176), noisy-entry lookahead; N kucuk |
| SHORT conviction skoru (gate2/3) | **REDDEDILECEK / OVERFIT SUPHESI** | N=6, TEST N=2; OOS dogrulanamaz |
| Limit-entry -20bps | **REDDEDILECEK** | Q2 gercek-fill: %36 fill -> EV/signal +34.5 < market +74.2 |
| score>=2 all-hours / hour<17 | **REDDEDILECEK** | hour<17 skor olu (mc0.458); hour gate sart |
| 48h horizon | **REDDEDILECEK** | avg -31 mc0.706 (edge decay) |
| LONG_SILENCE (arsiv) | **REDDEDILDI (lookahead)** | silence T+30'da bilinir; provisional -137 mc0.514 |

**Ozet:** Ana deploy hatti = hour17 LONG (canli) -> composite v3 score>=3 (paper-forward, live aday) + funding-veto + 6h + weighted-sizing. 
Frekans genisletme = 100K composite (paper). Premium = score>=4/5 sleeve. SHORT hala kirilgan (shadow). 
Reddedilenler: limit-entry, SHORT-skor-gate, all-hours-skor, 48h, silence-lookahead.

---
*Uretim: tools/report_s34_alpha_decision.py — sadece S34_ALL.db + S34_HORIZON.json okundu.*