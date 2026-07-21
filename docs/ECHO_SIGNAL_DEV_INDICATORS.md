# ECHO SİNYAL GELİŞTİRME — İNDİKATÖR LİSTESİ

**Amaç:** `echo_30_90+regime` lead'ini geliştirmek/rafine etmek için aday indikatörler.
**Kritik kural (OD-029):** Bu geliştirme **yalnız forward-temiz veride** yapılır. Yanmış Feb–Jul 2026
örneklemi üzerinde yeni eşik/feature/gate taraması YASAK (garantili aşırı-uyum). Bu liste
`tools/research_s34_echo_forward_ledger.py`'nin her echo olayında kaydettiği snapshot ile hizalıdır —
böylece lead olgunlaştığında geliştirme **yeni madencilik değil, önceden-kaydedilmiş forward veri** üstünde olur.
**Statü:** `● captured` = ledger şu an kaydediyor · `○ candidate` = eklenebilir (ledger'a alan ekle).

> **ÖNCE ŞUNU ÇÖZ (her şeyden önemli):** frozen kuraldaki `not noisy` gate'i **T+30m ileriye bakıyor = lookahead**.
> "+92.5bps / tail 0" muhtemelen bu yüzden. Ledger `qualified_t0` (causal) ile `qualified_full` (lookahead'li) ayrı
> kaydeder. **İlk soru "hangi yeni indikatör?" değil — "causal echo (noisy'siz) forward'da hâlâ kazanıyor mu?"**
> Bu doğrulanmadan indikatör eklemek anlamsız.

---

## 1. REJİM / TREND (rejim gate'in kalbi — mevcut sadece btc4h<0 OR btc7d<0)
| # | İndikatör | Statü | Hipotez / neden |
|---|---|---|---|
| 1 | btc4h_bps, btc7d_bps, btc3d_bps | ● | Mevcut gate. Çoklu-ufuk → rejim gate'i eşik yerine **süreklilik** olarak test et (forward). |
| 2 | eth1h_bps, eth4h_bps, eth15m_bps | ● | ETH kendi trend'i (bull gate). Downtrend derinliği tail'i besliyor (§162 prior-1h tilt). |
| 3 | btc_dominance_trend | ○ | ETH-idio vs BTC-led ayrımı; idio flush'lar farklı davranabilir. |
| 4 | trend-persistence (ardışık negatif bar sayısı) | ○ | "rejim ne kadar yerleşik" — tek-bar gürültüsünü ayıklar. |

## 2. ECHO YAPISI (sinyali TANIMLAYAN önceki-cluster geometrisi)
| # | İndikatör | Statü | Hipotez |
|---|---|---|---|
| 5 | echo_30_90, echo_30_120 (prior anchor var mı) | ● | Mevcut gate. Boolean → **echo sayısı/yoğunluğu**na genişlet. |
| 6 | prebuildup_30m_cnt, sell_liq_2h_cnt | ● | Önceki likidasyon yoğunluğu; "kaçıncı dalga" olduğunu kodlar. |
| 7 | echo_gap_min (önceki anchor'a tam mesafe) | ○ | 30–90 penceresi kaba; gerçek gecikme sürekli feature olarak daha bilgili. |
| 8 | prior_anchor_size_ratio (önceki/şimdiki notional) | ○ | Büyüyen mi sönen mi kaskad — exhaustion proxy'si. |

## 3. ÇAPRAZ-VARLIK SENKRON (BTC/SOL forced flow)
| # | İndikatör | Statü | Hipotez |
|---|---|---|---|
| 9 | btc_sell_10m_$M, sol_sell_10m_$M | ● | Sync-liq; sistemik mi ETH-özel mi. |
| 10 | cross_asset_dispersion (eth5m−btc5m spread) | ○ | Diverging flush'lar (§159 cross-asset replike olmadı) — ayrı davranış testi. |

## 4. POZİSYONLANMA / KALDIRAÇ (funding + OI + basis)
| # | İndikatör | Statü | Hipotez |
|---|---|---|---|
| 11 | funding_rate, funding_pctile_14d | ● | Kalabalık pozisyon → daha şiddetli squeeze/bounce. §162'de tek başına AUC~0.5 ama echo-koşullu farklı olabilir. |
| 12 | open_interest_$B, oi_chg_1h | ● / ○ | OI unwind = forced-flow teyidi; oi_chg eklenebilir. |
| 13 | basis_bps | ● (fresh-gated §156) | Spot-perp dislokasyonu; **stale ise geçersiz** (spot_age). |

## 5. AKIŞ / MİKROYAPI (giriş-anı baskı)
| # | İndikatör | Statü | Hipotez |
|---|---|---|---|
| 14 | flow_sell_imb_60s, cvd_60s_$M | ● | T0 agresör dengesi. **DİKKAT:** reaktif-CVD 4h tail'i öngörmüyor (§162 AUC 0.505), whipsaw riski (§163) — gate olarak değil, context olarak. |
| 15 | OFI / Kyle-λ / microprice_dev / Amihud | ○ | `microstructure_indicators.py`'de var; echo-koşullu likidite/impact — eklenmeye değer. |
| 16 | trade_count_60s / large-print footprint | ○ | Violent-capitulation izi (§159 max-forced-print tilt'i echo'da test et). |

## 6. VOLATİLİTE
| # | İndikatör | Statü | Hipotez |
|---|---|---|---|
| 17 | rv_5m (kendi hesap; vol_state güvenilmez) | ● | Rejim boyutlandırma; yüksek-vol'de bounce büyüklüğü farklı. |
| 18 | vol_decile | ● | Kaba rejim kovası. |
| 19 | vol-of-vol / jump-flag | ○ | Tek-tick sıçraması (§159 price_jump) — tail öncüsü olabilir. |

## 7. ZAMAN / SEANS
| # | İndikatör | Statü | Hipotez |
|---|---|---|---|
| 20 | session, hour_utc, dow | ● | Mevcut gate (sess≠EUROPE, dow∉{Mon,Wed}). **DİKKAT:** §162 session-tail hipotezini ÇÜRÜTTÜ — bu gate'ler in-sample fit riski taşıyor, forward'da doğrulanmalı. |
| 21 | blocked-hours (US 13–14) | ○ | Tail'ler bu saatlerde (echo expansion bulgusu); forward-teyit. |

---

## Geliştirme sırası (disiplinli)
1. **Faz A — causal doğrulama:** forward ledger yeterli olgunlaştığında `qualified_t0` (noisy'siz) net'i pozitif mi? Değilse echo = silence-lookahead tekrarı, DUR.
2. **Faz B — indikatör zenginleştirme:** yalnız Faz A geçerse, yukarıdaki `○ candidate`'leri ledger'a ekle (yeni alan; geçmişi yeniden madenleme). Her yeni gate = ayrı prereg (OD-028).
3. **Faz C — forward-koşullu test:** her aday indikatörün echo alt-popülasyonundaki ayrımı SADECE forward veride ölç; TRAIN/TEST forward-fold.

**Hiçbir indikatör yanmış veride "iyileştirme" için taranmaz. Liste forward veri şeması + hipotez kaydıdır, backtest hedefi değil.**
